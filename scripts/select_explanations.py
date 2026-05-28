import argparse
import json
import os

import pandas as pd
import itertools

from main import conf_to_str, compute_stats


def filter_explanations(cf: pd.DataFrame, metrics: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    out_cf = cf.copy()
    out_metrics = metrics.copy()

    # Select valid explanations only
    out_cf = out_cf[out_metrics['num_valid_cf'] == 1]
    out_metrics = out_metrics[out_metrics['num_valid_cf'] == 1]

    # Select explanations that mention actionable feature only
    out_cf = out_cf[out_metrics['avg_num_violations_per_cf'] == 0]
    out_metrics = out_metrics[out_metrics['avg_num_violations_per_cf'] == 0]

    return out_cf, out_metrics


def add_metric_categories(cf: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    # Convert the discriminative scores into discrete classes
    metrics["discriminative_power_classes"] = pd.cut(
        metrics["accuracy_knn_sklearn"],
        bins=[0, .5, .8, 1],
        labels=["basso", "medio", "alto"],
        include_lowest=True
    )

    # Convert the delta scores into discrete classes
    metrics["delta_classes"] = pd.cut(
        metrics["delta"],
        bins=3,
        labels=["basso", "medio", "alto"]
    )

    # Add a class that categorize the explanation as referring to a correct or wrong prediction
    metrics['correct'] = cf['label'] == cf['pred']

    # Convert the average number of changes into discrete classes
    metrics["change_classes"] = pd.cut(
        metrics["avg_num_changes_per_cf"],
        bins=[0, 1, 2, 3],
        labels=["basso", "medio", "alto"],
        include_lowest=True
    )

    return metrics


def select_explanations(cf: pd.DataFrame, metrics: pd.DataFrame, explore_by: list) -> pd.DataFrame:
    for col in explore_by:
        assert col in metrics, f"Missing metrics column {col}"

    out_cf = []
    unique_vals = [metrics[col].unique() for col in explore_by]
    combs = list(itertools.product(*unique_vals))
    for comb in combs:
        mask = (metrics[explore_by] == comb).all(axis=1)
        subset = metrics[mask]
        if len(subset) > 0:
            example_ix = subset.index[0]
            example = cf.loc[example_ix].to_dict()
            example['category'] = {col: val for col, val in zip(explore_by, comb)}
            out_cf.append(example)

    return pd.DataFrame(out_cf)


def main():
    parser = argparse.ArgumentParser(
        description='Script for selecting relevant explanations to analyze.',
        usage='select_explanations.py [<args>] [-h | --help]'
    )

    # Main params
    parser.add_argument('--dataset', type=str, choices=[
        'german-credit', 'german-credit-crif', 'german-credit-crif-v2', 'give-me-credit', 'adult'
    ])
    parser.add_argument('--model_name', type=str, default='lr', choices=['lr'])
    parser.add_argument('--explainer_name', type=str, default='nice', choices=[
        'ar', 'nice', 'nice-actionables', 'optbin', 'dice-random', 'dice-genetic', 'dice-kdtree', 'face', 'proce',
        'bfcf-opt', 'bfcf-exp'
    ])
    parser.add_argument('--test_case_sel_method', type=str, default='border',
                        choices=['border', 'neg_border', 'pos_border', 'auto-refuse', 'fp', 'fn'])

    # Extra params
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--proce-solver', type=str, default=None,
                        choices=['ae', 'mh_dist', 'opt', 'exp', 'opt_exp'])

    parser.add_argument('--disable_sampling', action='store_true')
    args = parser.parse_args()

    conf_key = conf_to_str({k: v for k, v in vars(args).items() if k not in ['disable_sampling']})
    print("Configuration:", conf_key)
    setattr(args, 'cf_file', os.path.join('output', conf_key, 'CF.json'))
    setattr(args, 'metric_file', os.path.join('output', conf_key, 'metrics.csv'))

    cf = pd.read_json(args.cf_file)
    cf['id'] = range(1, len(cf) + 1)
    metrics = pd.read_csv(args.metric_file, index_col=0)

    stats = compute_stats(metrics)
    with open(args.cf_file.replace('CF', 'STATS'), 'w') as f:
        json.dump(stats.to_dict(), f, indent=4)
    sel_cf, sel_metrics = filter_explanations(cf, metrics)

    metrics_with_cat = add_metric_categories(sel_cf, sel_metrics)

    if args.disable_sampling:
        out_cf = sel_cf.copy()
        out_file = args.cf_file.replace('CF_', 'SELECTED_CF_NO_SAMPL_')
    else:
        out_cf = select_explanations(
            sel_cf, metrics_with_cat,
            explore_by=['correct', 'discriminative_power_classes', 'delta_classes', 'change_classes']
        )
        out_file = args.cf_file.replace('CF', 'SELECTED_CF')

    out_cf.to_json(out_file, indent=4, orient='records')


if __name__ == '__main__':
    main()
