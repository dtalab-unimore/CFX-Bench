import argparse
import pandas as pd
import os
import json


def get_config(task: str):
    configs = json.load(open(os.path.join('..', 'expl_configs.json')))
    configs = {conf['name']: conf for conf in configs}
    config = configs[task]

    return config


def write_json_file(data: list, save_path: str):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description='Script for comparing explanations from different explainers.',
        usage='compare_explanations.py [<args>] [-h | --help]'
    )
    parser.add_argument('--conf_name', type=str, default='german-credit-crif__lr__auto-refuse__0.2__42')
    args = parser.parse_args()

    config = get_config(args.conf_name)
    expl_list = [pd.read_json(os.path.join('..', file_path)) for file_path in config['explanations']]
    explainer_names = [path.split('__')[2] for path in config['explanations']]

    # Find common explanations
    common_ids = set.intersection(*[set(df['id']) for df in expl_list])
    expl_list = [df[df['id'].isin(common_ids)] for df in expl_list]
    # Sort only for coherence
    for df in expl_list:
        df.sort_values('id', inplace=True)

    # Combine the explanations into a single structure
    combined = []
    for i, _id in enumerate(sorted(common_ids)):
        base = expl_list[0].loc[expl_list[0]['id'] == _id].iloc[0]

        entry = {
            "id": int(_id),
            "record": base["record"],
            "label": int(base["label"]),
            "pred": int(base["pred"]),
            "proba": base["proba"],
            "target": int(base["target"]),
            "explanations": {},
            "category": {'correct': base['category']['correct']} if 'category' in base and 'correct' in base[
                'category'] else {}
        }

        for expl_name, df in zip(explainer_names, expl_list):
            row = df.loc[df['id'] == _id].iloc[0]
            entry["explanations"][expl_name] = row["list_cf"]
            entry["category"][expl_name] = {k: v for k, v in row["category"].items() if k != 'correct'}

        combined.append(entry)

    # Save to JSON
    write_json_file(
        data=combined,
        save_path=os.path.join('..', 'output', f'COMPARISON_CF_{args.conf_name}.json')
    )


if __name__ == '__main__':
    main()
