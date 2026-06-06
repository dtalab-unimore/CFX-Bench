import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from llm_clients.utils import get_model_token
from llm_costs.prompts import (
    # SYSTEM_PROMPT as SYST, USER_PROMPT_TEMPLATE as USER,
    # SYSTEM_PROMPT_INC as SYST,
    SYSTEM_PROMPT_INC_V2 as SYST,
    USER_PROMPT_TEMPLATE_INC as USER
)
from llm_costs.utils import setup_logging
from llm_clients import get_llm

# Module-level counters so we can summarize at the end
_call_stats = {
    "n_calls": 0,
    "n_parse_failures": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_wall_time": 0.0,
}


# ---------------------------------------------------------------------------
# Pairwise elicitation
# ---------------------------------------------------------------------------
def build_messages(feat_a_key, feat_b_key):
    return [
        {"role": "system", "content": SYST},
        {
            "role": "user",
            "content": USER.format(
                feature_a_name=feat_a_key,
                feature_a_description=FEATURES[feat_a_key],
                feature_b_name=feat_b_key,
                feature_b_description=FEATURES[feat_b_key],
            ),
        },
    ]


'''def parse_verdict(llm_response_text):
    """
    Return 'A', 'B', or None if the response is malformed.

    Handles three common failure modes:
      1. Output wrapped in code fences or stray prose
      2. Output truncated mid-generation (no closing brace)
      3. Valid JSON but unexpected verdict value
    """
    text = llm_response_text.strip()
    start = text.find("{")
    if start == -1:
        return None

    end = text.rfind("}")
    if end == -1 or end <= start:
        # Truncated output — try to salvage the verdict field with a regex
        # before giving up, since verdict is now the first field in the schema
        import re
        match = re.search(
            r'"verdict"\s*:\s*"(A_HARDER|B_HARDER)"',
            text,
        )
        if match:
            logger.debug("Salvaged verdict from truncated output (len=%d)", len(text))
            return {"A_HARDER": "A", "B_HARDER": "B"}[match.group(1)]
        return None

    try:
        obj = json.loads(text[start: end + 1])
        v = obj.get("verdict", "").strip().upper()
        return {"A_HARDER": "A", "B_HARDER": "B"}.get(v)
    except json.JSONDecodeError:
        return None'''


def parse_verdict(llm_response_text):
    """
    Return the name of the harder feature, or None if the response is malformed.

    Handles common failure modes:
      1. Output wrapped in markdown code fences or stray prose
      2. Valid JSON but missing the expected key
    """
    text = llm_response_text.strip()

    # Locate the first and last JSON braces to strip out markdown formatting or prose
    start = text.find("{")
    if start == -1:
        return None

    end = text.rfind("}")
    if end == -1 or end <= start:
        # If the output is truncated (no closing brace), the feature name
        # (which is now at the end of the JSON) is likely lost. Give up.
        logger.debug("Output truncated; cannot salvage harder_feature_name (len=%d)", len(text))
        return None

    try:
        obj = json.loads(text[start: end + 1])

        # Extract the exact feature name instead of an enum
        feature_name = obj.get("harder_feature_name")

        if feature_name and isinstance(feature_name, str):
            return feature_name.strip()

        return None

    except json.JSONDecodeError:
        logger.debug("JSONDecodeError on parsing output")
        return None


OrderingStrategy = Literal["shuffle", "i_first", "j_first"]


def elicit_pairwise(feat_i, feat_j, n_repeats=5, rng=None, ordering: OrderingStrategy = "shuffle"):
    """
    Run n_repeats comparisons with randomized A/B ordering.
    Returns (wins_i, wins_j) as integers.
    """
    rng = rng or random.Random()
    wins_i = wins_j = 0
    n_failures = 0

    logger.debug("Eliciting pairwise: %r vs %r (n=%d)", feat_i, feat_j, n_repeats)

    for trial in range(n_repeats):
        if ordering == "shuffle":
            i_first = rng.random() < 0.5
        elif ordering == "i_first":
            i_first = True
        elif ordering == "j_first":
            i_first = False
        else:
            raise ValueError(
                f"Unknown ordering strategy: {ordering!r}. "
                f"Expected one of 'shuffle', 'i_first', 'i_last'."
            )

        if i_first:
            messages = build_messages(feat_i, feat_j)
        else:
            messages = build_messages(feat_j, feat_i)

        t0 = time.time()
        content, tokens = llm_client.ask(
            messages, max_response_length=1024, temperature=0.2
        )
        elapsed = time.time() - t0

        _call_stats["n_calls"] += 1
        _call_stats["total_prompt_tokens"] += tokens["input"]
        _call_stats["total_completion_tokens"] += tokens["output"]
        _call_stats["total_wall_time"] += elapsed

        logger.debug(
            "LLM call: %d in + %d out tokens in %.2fs (%.1f tok/s)",
            tokens["input"],
            tokens["output"],
            elapsed,
            tokens["output"] / elapsed if elapsed > 0 else 0.0,
        )

        # log raw answer before parsing
        logger.debug(
            "LLM raw output (len=%d) for pair (%s, %s) trial %d:\n%s",
            len(content),
            *((feat_i, feat_j) if i_first else (feat_j, feat_i)),
            trial + 1,
            content,
        )

        verdict = parse_verdict(content)

        if verdict is None:
            n_failures += 1
            _call_stats["n_parse_failures"] += 1
            logger.warning(
                "Parse failure on trial %d for (%s, %s). Raw output: %r",
                trial + 1,
                feat_i,
                feat_j,
                content[:200],
            )
            continue

        # Map A/B back to feat_i / feat_j depending on which was presented as A
        if i_first:
            # if verdict == "A":
            if verdict == feat_i:
                wins_i += 1
            else:
                wins_j += 1
        else:
            # if verdict == "A":
            if verdict == feat_j:
                wins_j += 1
            else:
                wins_i += 1

    n_valid = n_repeats - n_failures
    logger.info(
        "  %s vs %s -> %d-%d (over %d valid trials, %d parse failures)",
        feat_i,
        feat_j,
        wins_i,
        wins_j,
        n_valid,
        n_failures,
    )
    return wins_i, wins_j


def save_wins_matrix_csv(wins, feature_keys, path):
    """
    Save the wins matrix as a CSV with feature names as both row and column
    headers. The cell at row i, column j is the number of times feature i was
    judged harder to change than feature j.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(wins, index=feature_keys, columns=feature_keys)
    df.to_csv(path)
    logger.info("Saved wins matrix CSV to %s", path)


# ---------------------------------------------------------------------------
# Top-level: build the wins matrix for Bradley-Terry
# ---------------------------------------------------------------------------
def collect_all_pairwise_wins(
        feature_keys, n_repeats=5, seed=0, save_path=None, ordering: OrderingStrategy = "shuffle"
):
    """
    Returns an (m, m) integer numpy array `wins` where wins[i, j] is the
    number of times feature i was judged HARDER to change than feature j.
    """
    rng = random.Random(seed)
    m = len(feature_keys)
    n_pairs = m * (m - 1) // 2
    n_calls_expected = n_pairs * n_repeats

    logger.info("=" * 70)
    logger.info("Starting pairwise elicitation")
    logger.info("  Features:    %d", m)
    logger.info("  Unique pairs: %d", n_pairs)
    logger.info("  Repeats/pair: %d", n_repeats)
    logger.info("  Ordering strategy:  %s", ordering)
    logger.info("  Expected LLM calls: %d", n_calls_expected)
    logger.info("=" * 70)

    logger.debug(f"System prompt: {SYST}\n\n")
    logger.debug(f"User prompt template: {USER}")

    wins = np.zeros((m, m), dtype=int)
    pair_iter = [(i, j) for i in range(m) for j in range(i + 1, m)]

    overall_start = time.time()
    pbar = tqdm(pair_iter, desc="Pairs", unit="pair")
    for i, j in pbar:
        pbar.set_postfix_str(f"{feature_keys[i][:12]} vs {feature_keys[j][:12]}")
        w_i, w_j = elicit_pairwise(
            feature_keys[i], feature_keys[j], n_repeats=n_repeats, rng=rng, ordering=ordering
        )
        wins[i, j] = w_i
        wins[j, i] = w_j

    total_elapsed = time.time() - overall_start

    # Final summary
    logger.info("=" * 70)
    logger.info("Elicitation complete")
    logger.info("  Wall time:           %.1fs (%.2fs/call avg)",
                total_elapsed,
                total_elapsed / max(_call_stats["n_calls"], 1))
    logger.info("  Total LLM calls:     %d", _call_stats["n_calls"])
    logger.info("  Parse failures:      %d (%.1f%%)",
                _call_stats["n_parse_failures"],
                100 * _call_stats["n_parse_failures"] / max(_call_stats["n_calls"], 1))
    logger.info("  Prompt tokens:       %d", _call_stats["total_prompt_tokens"])
    logger.info("  Completion tokens:   %d", _call_stats["total_completion_tokens"])
    logger.info("  Avg gen time/call:   %.2fs",
                _call_stats["total_wall_time"] / max(_call_stats["n_calls"], 1))
    logger.info("=" * 70)

    # Sanity check: the comparison graph must be connected for a unique BT MLE
    row_sums = wins.sum(axis=1)
    col_sums = wins.sum(axis=0)
    isolated = [
        feature_keys[i]
        for i in range(m)
        if row_sums[i] == 0 or col_sums[i] == 0
    ]
    if isolated:
        logger.warning(
            "Features with all-wins or all-losses (BT MLE may diverge): %s",
            isolated,
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_wins_matrix_csv(wins, feature_keys, save_path.with_suffix(".csv"))

    return wins


def log_wins_matrix(feature_keys, wins):
    """Pretty-print the wins matrix to the log."""
    m = len(feature_keys)
    col_width = max(12, max(len(k) for k in feature_keys) + 1)
    short_names = [k[:col_width - 1] for k in feature_keys]

    header = " " * col_width + "".join(f"{n:>{col_width}}" for n in short_names)
    logger.info("Wins matrix (rows judged harder than cols):")
    logger.info(header)
    for i, name in enumerate(feature_keys):
        row = "".join(f"{wins[i, j]:>{col_width}d}" for j in range(m))
        logger.info(f"{name[:col_width - 1]:<{col_width}}{row}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=[
        'german-credit', 'german-credit-crif', 'german-credit-crif-full',
        'lending-club', 'lending-club-2', 'lending-club-3'
    ])
    parser.add_argument('--model', type=str, required=True, choices=[
        'llama-3.1-8b', 'qwen', 'gpt-4o-mini', 'mistral', 'gpt-5-mini', 'gpt-5.4-mini'
    ])
    parser.add_argument('--n-repeats', type=int, default=5)
    parser.add_argument('--ordering', type=str, choices=["shuffle", "i_first", "j_first"], default="shuffle")
    parser.add_argument('--seed', type=int, default=0)
    args = vars(parser.parse_args())

    dataset = args['dataset']

    output_dir = os.path.join(
        "output", "llm_costs", dataset.replace('-', '_'),
        f"{args['model']}__n{args['n_repeats']}__{args['ordering']}__s{args['seed']}"  # experiment tag
    )
    output_dir += "__prompt=inc_cat_v2"
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir + "/llm.log")

    config_path = {
        'german-credit': 'data/german_credit_config.json',
        'lending-club': 'data/lending_club_config.json',
    }[dataset]
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_dir = os.path.dirname(config['data_path'])
    features_path = data_dir + '/features_descr.json'
    with open(features_path, 'r') as f:
        FEATURES = json.load(f)
    FEATURES = {f: d for f, d in FEATURES.items() if f in config['act_features']}

    model_name = {
        'llama-3.1-8b': 'meta-llama/Llama-3.1-8B-Instruct',
        'qwen': 'Qwen2.5-14B-Instruct',
        'mistral': 'Mistral-Small-3.2-24B-Instruct-2506'
    }.get(args['model'], args['model'])
    token = get_model_token(model_name)

    llm_load_time = time.time()
    llm_client = get_llm(model_name, token)
    llm_load_time = time.time() - llm_load_time

    feature_keys = list(FEATURES.keys())
    wins = collect_all_pairwise_wins(
        feature_keys,
        n_repeats=args['n_repeats'],
        seed=args['seed'],
        save_path=f"{output_dir}/wins_matrix.csv",
        ordering=args['ordering'],
    )
    log_wins_matrix(feature_keys, wins)

    with open(f"{output_dir}/call_stats.json", 'w') as f:
        _call_stats['load_llm_time'] = llm_load_time
        json.dump(_call_stats, f)
