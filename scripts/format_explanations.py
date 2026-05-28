import json
import pandas as pd
import argparse


def summarize_profile(record_str):
    parts = record_str.split(", ")
    keep = ["age.in.years", "present.employment.since", "savings.account.and.bonds",
            "other.installment.plans", "housing", "purpose"]
    summary = [p for p in parts if any(k in p for k in keep)]
    return "; ".join(summary)


def main():
    parser = argparse.ArgumentParser(
        description='Script for formatting counterfactual explanations into a clean table.',
        usage='format_explanations.py [<args>] [-h | --help]'
    )
    parser.add_argument('--expl_file', type=str)
    args = parser.parse_args()

    with open(args.expl_file) as f:
        data = json.load(f)

    rows = []
    for rec in data:
        # profile = summarize_profile(rec["record"])
        proba_orig = rec["proba"]

        for cf in rec["list_cf"]:
            # feature = cf["changes"].split("=")[0]
            # value = cf["changes"].split("=")[1]
            proba_cf = float(cf["proba"])
            proba_str = f"{proba_orig:.3f} -> {proba_cf:.3f}"

            metrics = rec["category"]
            metric_str = f"Power: {metrics['discriminative_power_classes']}, " \
                         f"Delta pd: {metrics['delta_classes']}, " \
                         f"# Feat changed: {metrics['change_classes']}"

            rows.append({
                "ID": rec["id"],
                "Record": rec["record"],
                "CF": cf["changes"],
                "Proba": proba_str,
                "Correct": rec["category"]["correct"],
                "Metriche": metric_str,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="ID")
    df.to_csv(args.expl_file.replace('.json', '.csv'), index=False)


if __name__ == '__main__':
    main()
