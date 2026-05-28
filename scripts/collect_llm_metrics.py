import argparse
import os
import pandas as pd


def extract_params(file_name: str):
    file_name = file_name.split(os.sep)[-1].replace('LLM_metrics__', '')
    llm, prompt_style, dataset, _, explainer, _, _, _ = file_name.split("__")
    params = {
        'llm': llm, 'prompt_style': prompt_style, 'dataset': dataset, 'explainer': explainer
    }
    return params


def class_to_score(x: str):
    if x == 'low':
        return 0
    elif x == 'medium':
        return 0.5
    else:
        return 1


def convert_to_continuous(metrics):
    cont_metrics = metrics.copy()
    disc_metrics = [x for x in metrics.columns if x != 'overall']
    for metric in disc_metrics:
        cont_metrics[metric] = cont_metrics[metric].apply(lambda x: class_to_score(x))
    return cont_metrics


def compute_agg_metrics(metric_file: str):
    df = pd.read_json(metric_file, orient='records')
    disc_metrics = pd.DataFrame([x for x in df['metrics']])
    cont_metrics = convert_to_continuous(disc_metrics)
    avg_metrics = cont_metrics.mean().to_dict()
    avg_metrics.update({'count': len(cont_metrics)})
    params = extract_params(metric_file)
    avg_metrics.update(params)
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Script for collecting quality scores provided by LLMs.',
        usage='collect_llm_metrics.py [<args>] [-h | --help]'
    )
    parser.add_argument('--res_dir', type=str)
    args = parser.parse_args()

    res_files = [os.path.join(args.res_dir, x) for x in os.listdir(args.res_dir) if x.startswith('LLM_metrics_')]
    results = []
    for res_file in res_files:
        res = compute_agg_metrics(res_file)
        results.append(res)
    results = pd.DataFrame(results)

    overall_results = pd.pivot_table(
        results[['overall', 'explainer', 'prompt_style']],
        index=['prompt_style'], columns=['explainer'], values='overall'
    )
    properties = ['satisfaction', 'feasibility', 'consistency', 'completeness',
                  'trust', 'understandability', 'fairness', 'complexity']
    explainers = results['explainer'].unique()
    for explainer in explainers:
        expl_results = results[results['explainer'] == explainer][properties]
        a = []


    a = []


if __name__ == '__main__':
    main()
