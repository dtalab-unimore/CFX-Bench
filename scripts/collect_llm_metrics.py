import argparse
import os
import pandas as pd


def extract_params(file_name: str, base_output_dir='output/'):
    # file_name = output/german-credit/llm-global/lr__auto-refuse__s42/LLM_metrics__gpt-4o-mini__base.json
    file_name = file_name.replace(base_output_dir, '').split(os.sep)
    if len(file_name) == 4:
        dataset, explainer, _, file_name = file_name
    elif len(file_name) == 5:
        dataset, explainer, solver, _, file_name = file_name
        explainer = f'{explainer}-{solver}'
    llm, prompt_style = file_name.replace('LLM_metrics__', '').replace('.json', '').split("__")
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
    # parser = argparse.ArgumentParser(
    #     description='Script for collecting quality scores provided by LLMs.',
    #     usage='collect_llm_metrics.py [<args>] [-h | --help]'
    # )
    # parser.add_argument('--res_dir', type=str)
    # args = parser.parse_args()

    # res_files = [os.path.join(args.res_dir, x) for x in os.listdir(args.res_dir) if x.startswith('LLM_metrics_')]
    res_files = [
        f"output/german-credit/{explainer}/lr__auto-refuse__s42/LLM_metrics__gpt-4o-mini__base.json"
        for explainer in ['dice/random', 'nice', 'ares', 'llm-global']
    ]
    results = []
    for res_file in res_files:
        if os.path.exists(res_file):
            res = compute_agg_metrics(res_file)
            results.append(res)
    results = pd.DataFrame(results)

    results.to_csv('output/LLM_metrics.csv', index=False)

    # overall_results = pd.pivot_table(
    #     results[['overall', 'explainer', 'prompt_style']],
    #     index=['prompt_style'], columns=['explainer'], values='overall'
    # )
    # properties = ['satisfaction', 'feasibility', 'consistency', 'completeness',
    #               'trust', 'understandability', 'fairness', 'complexity']
    # explainers = results['explainer'].unique()
    # for explainer in explainers:
    #     expl_results = results[results['explainer'] == explainer][properties]
    #     a = []
    #
    #
    # a = []


if __name__ == '__main__':
    main()
