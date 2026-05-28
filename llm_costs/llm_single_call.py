import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from llm_clients import get_llm
from llm_clients.utils import get_model_token
from llm_costs.prompts import (
    # SINGLE_CALL_SYSTEM_PROMPT as SYST, SINGLE_CALL_USER_PROMPT_TEMPLATE as USER,
    # SINGLE_CALL_SYSTEM_PROMPT_INC as SYST,
    SINGLE_CALL_SYSTEM_PROMPT_INC_V2 as SYST,
    SINGLE_CALL_USER_PROMPT_TEMPLATE_INC as USER
)
from llm_costs.utils import setup_logging

call_stats = {
    "n_calls": 0,
    "n_parse_failures": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_wall_time": 0.0,
}


def build_direct_messages(feature_keys, features_dict, rng=None):
    """
    Build the chat messages for a single direct-elicitation call.

    feature_keys are presented to the LLM in a randomized order to guard
    against position-based ordering effects in the output. The returned
    presentation_order tells you which order was used, so you can map the
    LLM's JSON output back to your canonical feature indexing.
    """
    rng = rng or random.Random()
    presentation_order = list(feature_keys)
    rng.shuffle(presentation_order)

    feature_block = "\n".join(
        f"  - {name}: {features_dict[name]}" for name in presentation_order
    )

    messages = [
        {"role": "system", "content": SYST},
        {
            "role": "user",
            "content": USER.format(
                feature_block=feature_block
            ),
        },
    ]
    return messages, presentation_order


def parse_direct_scores(llm_response_text, feature_keys):
    """
    Parse the JSON response from a direct-elicitation call.

    Returns
    -------
    scores : dict[str, int] or None
        Maps each feature key to its integer score in [1, 10].
        Returns None if the response is malformed, has missing features,
        or contains out-of-range values.
    """
    text = llm_response_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.warning("No JSON object found in response")
        return None

    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error: %s", e)
        return None

    """raw_scores = obj.get("scores")
    if not isinstance(raw_scores, dict):
        logger.warning("Response missing 'scores' object or it is not a dict")
        return None"""
    evaluations = obj.get("evaluations")
    raw_scores = {e['feature_name']: e['difficulty_score'] for e in evaluations}

    # Validate that every feature appears exactly once and the score is in range
    parsed = {}
    for name in feature_keys:
        if name not in raw_scores:
            logger.warning("Response missing feature %r", name)
            return None
        try:
            score = int(raw_scores[name])
        except (ValueError, TypeError):
            logger.warning("Non-integer score for %r: %r", name, raw_scores[name])
            return None
        if not 1 <= score <= 10:
            logger.warning("Out-of-range score for %r: %d", name, score)
            return None
        parsed[name] = score

    # Check for any extras (the LLM hallucinated a feature)
    extras = set(raw_scores.keys()) - set(feature_keys)
    if extras:
        logger.warning("Response contains unexpected features: %s", extras)
        # Not fatal — we just ignore them

    return parsed


def elicit_direct_scores(
    feature_keys,
    features_dict,
    n_repeats=5,
    seed=0,
    output_path="direct_scores.csv",
):
    """
    Run direct cost elicitation: each trial asks the LLM to score every
    feature in one call. Repeated n_repeats times to average over sampling
    noise. The feature presentation order is randomized each trial.

    Saves:
      - <output_path>: a CSV where each row is one trial's scores (columns
        are features, last column is the mean score per trial for sanity)
      - <output_path with .json suffix>: run configuration

    Returns
    -------
    scores_df : pandas.DataFrame
        Shape (n_repeats, n_features). Rows are trials, columns are features.
        NaN for trials that failed to parse.
    """
    rng = random.Random(seed)

    logger.info("=" * 70)
    logger.info("Starting direct score elicitation")
    logger.info("  Features:     %d", len(feature_keys))
    logger.info("  Trials:       %d", n_repeats)
    logger.info("=" * 70)

    logger.debug(f"System prompt: {SYST}\n\n")
    logger.debug(f"User prompt template: {USER}")

    all_scores = []
    for trial in range(n_repeats):
        messages, presentation_order = build_direct_messages(
            feature_keys, features_dict=features_dict, rng=rng
        )

        t0 = time.time()
        content, tokens = llm_client.ask(
            messages,
            max_response_length=1024*len(feature_keys),
            temperature=0.2,
        )
        elapsed = time.time() - t0

        call_stats["n_calls"] += 1
        call_stats["total_prompt_tokens"] += tokens["input"]
        call_stats["total_completion_tokens"] += tokens["output"]
        call_stats["total_wall_time"] += elapsed

        logger.debug(
            "LLM raw output:\n%s", content
        )

        parsed = parse_direct_scores(content, feature_keys)

        if parsed is None:
            call_stats["n_parse_failures"] += 1
            logger.warning(
                "Parse failure on trial %d. Raw output preview: %r",
                trial + 1,
                content[:300],
            )
            # Record NaNs so we still get a row in the output DataFrame
            all_scores.append({k: float("nan") for k in feature_keys})
        else:
            all_scores.append(parsed)
            logger.info(
                "Trial %d scores: %s",
                trial + 1,
                ", ".join(f"{k}={v}" for k, v in parsed.items()),
            )

    # Assemble into a DataFrame, one row per trial
    scores_df = pd.DataFrame(all_scores, columns=feature_keys)

    # Add summary statistics as extra rows
    mean_scores = scores_df.mean(axis=0, skipna=True)
    std_scores = scores_df.std(axis=0, skipna=True)

    logger.info("=" * 70)
    logger.info("Direct elicitation complete")
    logger.info("  Wall time:           %.1fs", call_stats["total_wall_time"])
    logger.info("  Parse failures:      %d / %d", call_stats["n_parse_failures"], n_repeats)
    logger.info(
        "  Prompt tokens:       %d (avg %d/call)",
        call_stats["total_prompt_tokens"],
        call_stats["total_prompt_tokens"] // max(call_stats["n_calls"], 1),
    )
    logger.info("  Completion tokens:   %d", call_stats["total_completion_tokens"])
    logger.info("=" * 70)
    logger.info("Mean scores across trials:")
    for name in feature_keys:
        logger.info("  %-24s %.2f (std %.2f)", name, mean_scores[name], std_scores[name])

    # Save outputs
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Per-trial scores CSV with summary rows appended
        output_df = scores_df.copy()
        output_df.index = [f"trial_{i+1}" for i in range(n_repeats)]
        output_df.loc["mean"] = mean_scores
        output_df.loc["std"] = std_scores
        output_df.to_csv(output_path)
        logger.info("Saved direct scores to %s", output_path)

    return scores_df


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
    parser.add_argument('--seed', type=int, default=0)
    args = vars(parser.parse_args())

    dataset = args['dataset']

    output_dir = os.path.join(
        "output", "llm_costs", dataset.replace('-', '_'),
        f"{args['model']}__n{args['n_repeats']}__direct__s{args['seed']}"  # experiment tag
    )
    output_dir += "__prompt=inc_cat_v2"

    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir + "/llm.log")

    config_path = {
        'german-credit': 'data/german_credit_config.json',
        'german-credit-crif': 'data/german_credit_crif_config.json',
        'german-credit-crif-full': 'data/german_credit_crif_full_config.json',
        'lending-club': 'data/lending_club_config.json',
        'lending-club-2': 'data/lending_club_2_config.json',
        'lending-club-3': 'data/lending_club_3_config.json',
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
    elicit_direct_scores(
        feature_keys,
        FEATURES,
        n_repeats=args['n_repeats'],
        seed=args['seed'],
        output_path=output_dir+'/direct_scores.csv',
    )

    with open(f"{output_dir}/call_stats.json", 'w') as f:
        call_stats['load_llm_time'] = llm_load_time
        json.dump(call_stats, f)
