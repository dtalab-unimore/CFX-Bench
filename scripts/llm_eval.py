import argparse
import pandas as pd
import random
import numpy as np
import dotenv
import os
import json
from tqdm import tqdm

from llm_clients import get_llm, get_model_source
from llm_prompts.cf_eval_prompt import CFEvalPromptManager
from llm_prompts.cf_verbalize_prompt import CFVerbalizePromptManager


def get_model_token(model_name: str):
    model_source = get_model_source(model_name)
    dotenv.load_dotenv()  # Read from .env file
    if model_source == 'gpt':
        token = os.getenv('OPENAI_API_KEY')
    else:
        token = os.getenv('HF_TOKEN')  # Get the HuggingFace token
    return token


def verbalize_explanations(cf_df, llm, output_dir, use_cache):
    prompt_manager = CFVerbalizePromptManager()
    # cache_dir = os.path.join('..', 'verbalization_cache')
    # os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, 'verbalization_cache.csv')

    if use_cache and os.path.exists(cache_file):
        print("Loaded counterfactual verbalization from the cache.")
        return pd.read_csv(cache_file)

    verbalized = []
    for ix, cf_item in tqdm(cf_df.iterrows(), total=len(cf_df)):
        if len(cf_item['list_cf']) == 0:
            verbalized.append("")
            continue
        cf = cf_item['list_cf'][0]
        del cf_item['list_cf']
        cf_item['cf'] = cf

        # Create user prompt
        cf_json = cf_item.copy()
        ignore_attrs = ['id', 'category', 'label', 'target']
        for attr in ignore_attrs:
            if attr in cf_json:
                del cf_json[attr]
        del cf_json['cf']['cost']
        cf_json = cf_json.to_json(indent=4)
        prompt = prompt_manager.get_user_prompt_content(cf_json)

        # Add the system prompt
        prompt = prompt_manager.generate(prompt)

        # Ask the LLM
        response, _ = llm.ask(prompt)

        # Parse the response
        verbalized_cf = prompt_manager.parse_response(response)
        if cf_item['pred'] == 0:
            assert "Current outcome: your loan application was rejected" in verbalized_cf
        else:
            assert "Current outcome: your loan application was approved" in verbalized_cf
        verbalized.append(verbalized_cf)

    cf_df["verbalized"] = verbalized
    cf_df.to_csv(cache_file, index=False)

    return cf_df


def compute_overall_score(scores):
    def convert_to_score(x):
        if x == 'low':
            return 0
        elif x == 'medium':
            return 0.5
        else:
            return 1
    num_scores = {}
    for prop, prop_scores in scores.items():
        num_scores[prop] = [convert_to_score(x) for x in prop_scores]
    num_scores = np.array(list(num_scores.values()), dtype=np.float32)
    overall_scores = np.sum(num_scores, axis=0)
    scores['overall'] = [float(x) for x in overall_scores]

    return [dict(zip(scores.keys(), values)) for values in zip(*scores.values())]


def evaluate_explanations(cf_df, llm, prompt_style):
    examples = None
    if prompt_style == 'full-examples':
        examples = get_examples()
        prompt_style = 'full'

    properties = CFEvalPromptManager.properties
    all_scores = {}
    for prop in properties:
        print(f"\nEvaluating the counterfactual explanations based on the {prop} property...")
        eval_prompt_manager = CFEvalPromptManager(
            property_name=prop, prompt_style=prompt_style, examples=examples
        )

        scores = []
        for _, cf_item in tqdm(cf_df.iterrows(), total=len(cf_df)):
            verbalized_cf = cf_item['verbalized']

            # Create user prompt
            prompt = eval_prompt_manager.get_user_prompt_content(verbalized_cf)

            # Add the system prompt
            prompt = eval_prompt_manager.generate(prompt)

            # Ask the model
            response, _ = llm.ask(prompt)

            # Parse the response
            score = eval_prompt_manager.parse_response(response)
            scores.append(score)

        all_scores[prop] = scores

    all_scores = compute_overall_score(all_scores)
    cf_df['metrics'] = all_scores

    return cf_df


def get_examples():
    examples = json.load(open(os.path.join('..', 'llm_examples.json'), "r", encoding="utf-8"))
    return examples


def main():
    parser = argparse.ArgumentParser(
        description='Script for evaluating the quality of counter-factual explanations with an LLM-based approach.',
        usage='llm_eval.py [<args>] [-h | --help]'
    )
    parser.add_argument('--cf_file', type=str)
    parser.add_argument('--llm', type=str)
    parser.add_argument('--prompt_style', type=str, choices=['base', 'full', 'full-examples'])
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = os.path.dirname(args.cf_file)

    llm_name = {
        'llama-3.1-8b': 'meta-llama/Llama-3.1-8B-Instruct',
        'qwen': 'Qwen2.5-14B-Instruct',
        'mistral': 'Mistral-Small-3.2-24B-Instruct-2506'
    }.get(args.llm, args.llm)

    # Load the LLM
    token = get_model_token(llm_name)
    llm = get_llm(
        model_name=llm_name,
        token=token,
        seed=args.seed
    )

    # Load the explanations
    cf_df = pd.read_json(args.cf_file)
    if 'id' not in cf_df.columns:
        cf_df['id'] = range(1, len(cf_df) + 1)
    # cf_df = cf_df.iloc[:2]  # debug

    # Verbalize the explanations
    verbalized_cf = verbalize_explanations(cf_df, llm, output_dir, args.use_cache)

    # Evaluate the quality of the explanations
    out_cf = evaluate_explanations(verbalized_cf, llm, args.prompt_style)

    # Save the results
    out_file = f'LLM_metrics__{args.llm}__{args.prompt_style}.json'
    out_cf.to_json(os.path.join(output_dir, out_file), indent=4, orient='records')


if __name__ == '__main__':
    main()
