import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from explanation import ExplanationSet
from tqdm import tqdm
import pandas as pd
import time
import json

LOWER_LIMIT_RANGE_FOR_D = 1
UPPER_LIMIT_RANGE_FOR_D = 40
UPPER_LIMIT_FOR_K = 30

allowed_subkeys = ["Coverage", "Avg. distance", "Avg. path cost"]


def compute_metrics(expl_sets: list[ExplanationSet], features, cat_features, num_features,
                    model, binning_process):
    """
    Compute metrics for global counterfactual explanations.
    :returns:
        correct_recourse, percentage of instances offered a recourse for which the recourse flips the prediction.
        incorrect_recourse, percentage of instances offered a recourse for which the recourse doesn't flip the prediction.
        coverage, percentage of instances offered a recourse.
        success, percentage of instances for which the recourse flips the prediction.
        avg_cost_l1, average of l1 costs between factual and counterfactual when the counterfactual flips the prediction.
        median_cost_l1, median of l1 costs between factual and counterfactual when the counterfactual
        flips the prediction.
        avg_cost_l2, average of l2 costs between factual and counterfactual when the counterfactual flips the prediction.
        median_cost_l2, median of l2 costs between factual and counterfactual when the counterfactual flips the prediction.
        unique_cfs, percentage of unique counterfactuals for instances for which the counterfactual flips the prediction
        (lower is better, more interpretable).
        unique_deltas, percentage of unique transformation vectors for instances for which the counterfactual flips the
        prediction (lower is better, more interpretable).
        attribute_change_frequency, frequency of attribute changes for instances for which the counterfactual flips
        the prediction.
        inference_efficiency, average time required to generate the counterfactuals.
    """

    binned = True

    total = len(expl_sets)
    flipped, covered = 0, 0
    costs_l1, costs_l2 = [], []
    unique_cfs, unique_deltas = [], []
    # inference_efficiency = []
    attribute_change_frequency = pd.Series(0, index=features)

    if binned:
        cat_features = []
        num_features = features
        scaler = None
        encoder = None
        _scale_and_encode = lambda X, a, b, c, d: model.transform(X.to_frame().T, 'id').reset_index(drop=True)
    else:
        # scaler = MinMaxScaler()
        # scaler.fit(data[num_features])
        # encoder = OneHotEncoder(sparse_output=False)
        # encoder.fit(data[cat_features])
        _scale_and_encode = scale_and_encode
        ...  # todo

    # for _, x in tqdm(factuals.iterrows(), total=total, desc="Computing metrics"):
    for expl in tqdm(expl_sets, desc="Computing metrics"):
        # record = x[features]
        # label = x[target]
        # pred = model.predict(x[features].to_frame().T)[0]
        # proba = model.predict_proba(x[features].to_frame().T)[0][pred]

        # # start timer to measure inference efficiency
        # start = time.perf_counter()
        # cfs = explain(test_item=[record, label, pred, proba, target])
        # # end timer to measure inference efficiency
        # end = time.perf_counter()
        # inference_efficiency.append(end - start)

        # cfs = ExplanationSet(**cfs)
        # cfs_pred = model.predict(cfs.get_list_expl_full())[0]
        record = expl.get_record()
        if len(expl) > 0 and expl.get_new_preds().iloc[0] != expl.get_pred():
            flipped += 1
            record_processed = _scale_and_encode(record, scaler, encoder, num_features, cat_features)
            cfs_processed = _scale_and_encode(expl.get_list_expl_full().loc[0], scaler, encoder, num_features, cat_features)
            costs_l1.append((record_processed - cfs_processed).abs().sum(axis=1)[0])
            costs_l2.append(np.linalg.norm(record_processed - cfs_processed))
            if binned:
                record_bin = record
                cfs_bin = expl.get_list_expl_full().loc[0]
            else:
                record_bin = binning_process.transform(record.to_frame().T, metric="bins").squeeze()
                cfs_bin = binning_process.transform(expl.get_list_expl_full(), metric="bins").loc[0]
            unique_cfs.append(cfs_bin)
            unique_deltas.append((record_processed - cfs_processed).loc[0])
            attribute_change_frequency += (record_bin != cfs_bin)
        # if (expl.get_list_expl_full().iloc[0] != expl.get_record()).any():
            covered += 1

    correct_recourse = flipped / covered
    incorrect_recourse = 1 - correct_recourse
    coverage = covered / total
    success = flipped / total
    avg_cost_l1, median_cost_l1 = np.mean(costs_l1), np.median(costs_l1)
    avg_cost_l2, median_cost_l2 = np.mean(costs_l2), np.median(costs_l2)
    unique_cfs, unique_deltas = pd.DataFrame(unique_cfs), pd.DataFrame(unique_deltas)
    unique_cfs = len(unique_cfs.drop_duplicates()) / flipped
    unique_deltas = len(unique_deltas.drop_duplicates()) / flipped
    attribute_change_frequency /= flipped
    # inference_efficiency = np.mean(inference_efficiency)

    metrics = {
            'correct_recourse': correct_recourse,
            'incorrect_recourse': incorrect_recourse,
            'coverage': coverage,
            'success': success,
            'avg_cost_l1': avg_cost_l1,
            'median_cost_l1': median_cost_l1,
            'avg_cost_l2': avg_cost_l2,
            'median_cost_l2': median_cost_l2,
            'unique_cfs': unique_cfs,
            'unique_deltas': unique_deltas,
            # 'inference_efficiency': inference_efficiency,
            # 'training_efficiency': training_efficiency
        }
    return metrics, attribute_change_frequency


def write_metrics(name, metrics, attribute_change_frequency):
    metrics = {k: round(v, 2) for k, v in metrics.items()}
    attribute_change_frequency = attribute_change_frequency.round(2).to_dict()
    with open(name + '/metrics_global.json', 'w') as f: json.dump(metrics, f, indent=4)
    with open(name + '/attribute_change_frequency.json', 'w') as f: json.dump(attribute_change_frequency, f, indent=4)


def compute_metrics_global(expl_sets, features, cat_features, num_features, model,
                           binning_process, name, write=False):
    metrics, attribute_change_frequency = compute_metrics(
        expl_sets, features, cat_features, num_features, model, binning_process
    )
    if write:
        write_metrics(name, metrics, attribute_change_frequency)
    return metrics, attribute_change_frequency


def scale_and_encode(x, scaler, encoder, num_features, cat_features):

    x_scaled = scaler.transform(x[num_features].to_frame().T)
    x_encoded = encoder.transform(x[cat_features].to_frame().T)
    scaled_cols = num_features
    encoded_cols = encoder.get_feature_names_out(cat_features)
    combined_array = np.hstack([x_scaled, x_encoded])

    return pd.DataFrame(combined_array, columns=list(scaled_cols) + list(encoded_cols))


def kAUC(adapter, distances,
         upper_limit_for_k=10, lower_limit_range_for_d=None, upper_limit_range_for_d=None, stepsd=12, stepsk=4):
    auc_matrix = {}
    saturation_points = {}
    cov_for_saturation_points = {}

    if lower_limit_range_for_d is None:
        lower_limit_range_for_d = 0.1
    if upper_limit_range_for_d is None:
        upper_limit_range_for_d = np.max(distances)

    k_values = nice_numbers(1, upper_limit_for_k, stepsk, score='k')
    d_values = nice_numbers(lower_limit_range_for_d, upper_limit_range_for_d, stepsd, score='d')
    for cfes in k_values:
        auc_matrix[cfes] = {}
        results = {}

        for max_d in d_values:
            results_input = adapter.run_kd(max_d, cfes)
            r = filter_subdict(results_input, allowed_subkeys)
            if "Total coverage" in r.keys():
                r.pop("Total coverage")
            if "Graph Stats" in r.keys():
                r.pop("Graph Stats")
            results[max_d] = (r)

        group_keys = list(r.keys())
        saturation_points_g = {key: 0 for key in group_keys}
        coverage_till_now_g = {key: 0 for key in group_keys}

        group_coverages = {key: [] for key in group_keys}

        for d in results:
            for key in group_keys:
                cov = results[d][key]['Coverage']
                if cov > coverage_till_now_g[key]:
                    coverage_till_now_g[key] = cov
                    saturation_points_g[key] = d
                group_coverages[key].append(cov)

        saturation_points[cfes] = saturation_points_g
        cov_for_saturation_points[cfes] = coverage_till_now_g

        max_auc = auc(d_values, [100] * len(d_values))

        auc_matrix[cfes] = {}

        for key in group_keys:
            group_coverages_array = np.array(group_coverages[key])
            normalized_auc = np.round(auc(d_values, group_coverages_array) / max_auc, 2)
            auc_matrix[cfes][key] = normalized_auc

    return saturation_points, cov_for_saturation_points, auc_matrix


def dAUC(adapter, distances,
         upper_limit_for_k=10, lower_limit_range_for_d=None, upper_limit_range_for_d=None, stepsd=12, stepsk=4):
    auc_matrix = {}
    saturation_points = {}
    cov_for_saturation_points = {}

    if lower_limit_range_for_d is None:
        lower_limit_range_for_d = 0.1
    if upper_limit_range_for_d is None:
        upper_limit_range_for_d = np.max(distances)

    d_values = nice_numbers(lower_limit_range_for_d, upper_limit_range_for_d, stepsd, score='d')
    k_values = nice_numbers(1, upper_limit_for_k, stepsk, score='k')
    for d in d_values:
        d = np.round(d, 2)
        auc_matrix[d] = {}
        results = {}
        for cfes in k_values:
            results_input = adapter.run_kd(d, cfes)
            r = filter_subdict(results_input, allowed_subkeys)
            if "Total coverage" in r.keys():
                r.pop("Total coverage")
            if "Graph Stats" in r.keys():
                r.pop("Graph Stats")
            results[cfes] = (r)

        group_keys = list(r.keys())
        saturation_points_g = {key: 0 for key in group_keys}
        coverage_till_now_g = {key: 0 for key in group_keys}

        group_coverages = {key: [] for key in group_keys}

        for cfes in results:
            for key in group_keys:
                cov = results[cfes][key]['Coverage']
                if cov > coverage_till_now_g[key]:
                    coverage_till_now_g[key] = cov
                    saturation_points_g[key] = cfes
                group_coverages[key].append(cov)

        saturation_points[d] = saturation_points_g
        cov_for_saturation_points[d] = coverage_till_now_g

        max_auc = auc(k_values, [100] * len(k_values))

        auc_matrix[d] = {}
        for key in group_keys:
            group_coverages_array = np.array(group_coverages[key])
            normalized_auc = np.round(auc(k_values, group_coverages_array) / max_auc, 2)
            auc_matrix[d][key] = normalized_auc

    return saturation_points, cov_for_saturation_points, auc_matrix


def cAUC(adapter,
         upper_limit_for_k=10, coverage_values=[0.25, 0.5, 0.75, 1], plot_=False):
    auc_matrix = {}
    saturation_points = {}
    cost_for_saturation_points = {}

    k_values = list(range(1, upper_limit_for_k + 1))

    for coverage in coverage_values:
        auc_matrix[coverage] = {}
        saturation_points[coverage] = {}
        cost_for_saturation_points[coverage] = {}
        total_cost_group_0 = []

        for idx, k in enumerate(k_values):
            results_per_group, max_cost = adapter.run_c(k, coverage)

            group_keys = list(results_per_group.keys())

            total_cost_group_0.append(results_per_group[group_keys[0]]["Total Cost"])

        AUC_max = auc(k_values, [max_cost] * len(k_values))
        AUC_group_0 = auc(k_values, total_cost_group_0) / AUC_max

        auc_matrix[coverage]["group_0"] = AUC_group_0
        saturation_points[coverage]["group_0"], cost_for_saturation_points[coverage]["group_0"] = find_saturation_point(
            total_cost_group_0, k_values)

        if plot_:
            plt.plot(k_values, total_cost_group_0, marker='o', color='red')
            plt.xlabel('k', fontsize=12, fontfamily='serif')
            plt.ylabel('Max cost', fontsize=12, fontfamily='serif')
            plt.text(0.8, 0.2, f'AUC: {AUC_group_0:.2f}', fontsize=10, ha='center',
                     transform=plt.gca().transAxes)

            plt.xticks(k_values)
            plt.legend()
            plt.tight_layout()
            fig_size = (8, 6)
            plt.gcf().set_size_inches(fig_size)
            plt.show()

    return saturation_points, cost_for_saturation_points, auc_matrix


def plot_k_or_dAUC(saturation_points, cov_for_saturation_points, auc_matrix, score='k',
                   expand_left_x_axis=0, expand_right_x_axis=1, expand_bottom_y_axis=0.5, expand_top_y_axis=0.5):
    x_values = list(auc_matrix.keys())
    group_keys = list(auc_matrix[x_values[0]].keys())
    x_values_g0 = [auc_matrix[x][group_keys[0]] for x in x_values]

    sp_G0 = [saturation_points[x][group_keys[0]] for x in x_values]

    max_cov_G0 = [np.round(cov_for_saturation_points[x][group_keys[0]], 2) for x in x_values]

    sns.set(style="white")
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=x_values, y=x_values_g0, marker='o', color='mediumseagreen', linewidth=2)
    plt.xlabel(score, fontsize=20)
    plt.ylabel(f'{score.upper()}AUC Score', fontsize=20)
    plt.xticks(x_values)
    plt.legend(fontsize=16, framealpha=0.2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 6))
    sns.lineplot(x=x_values, y=sp_G0, color='mediumseagreen', marker='o', linewidth=2, alpha=0.7)

    ## get the axis limit from plt
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plt.xlim(x_min - expand_left_x_axis, max(x_values) + expand_right_x_axis)
    plt.ylim(y_min - expand_bottom_y_axis, y_max + expand_top_y_axis)

    x_min, x_max = 0, 100
    extend_x = 0
    extend_y = 0

    for i, (x, sp0, max_cov0) in enumerate(zip(x_values, sp_G0, max_cov_G0)):
        offset0 = (10, 10)

        annotation_x_max_0 = i * offset0[0]
        annotation_y_max_0 = i * offset0[1]
        if score == 'k':
            if annotation_x_max_0 > x_max:
                extend_x = extend_x + 1
                extend_x = extend_x + 1
        elif score == 'd':
            if annotation_x_max_0 > x_max:
                extend_x = extend_x + 1
                extend_x = extend_x + 1
        if annotation_y_max_0 > max(sp_G0):
            extend_y = extend_y + 1

        plt.annotate(f'{max_cov0}', (x, sp0), textcoords="offset points", xytext=offset0, \
                     ha='center', color='mediumseagreen', weight='bold')

    plt.legend(fontsize=16, framealpha=0.2)
    plt.xticks(x_values, fontsize=20)
    plt.yticks(fontsize=20)

    if score == 'k':
        plt.ylabel('Saturation Point: sp(k)', fontsize=20)
        plt.xlabel('k', fontsize=20)
    else:
        plt.ylabel('Saturation Point: sp(d)', fontsize=20)
        plt.xlabel('d', fontsize=20)

    plt.show()


def filter_subdict(experiment, allowed_subkeys):
    result = {}
    for key, value in experiment.items():
        if isinstance(value, dict):
            filtered_subdict = {}
            for subkey, subvalue in value.items():
                if subkey in allowed_subkeys:
                    filtered_subdict[subkey] = subvalue
            result[key] = filtered_subdict
        else:
            result[key] = value
    return result


def find_saturation_point(cost_values, k_values):
    max_cost_value = float('inf')
    saturation_k = k_values[-1]

    for i in range(1, len(cost_values)):
        if round(cost_values[i], 3) < round(max_cost_value, 3):
            max_cost_value = cost_values[i]
            saturation_k = k_values[i]
    return saturation_k, cost_values[i]


def nice_numbers(range_min, range_max, num_ticks, score='k', round_precision=2):

    if range_min >= range_max:
        raise ValueError("range_min must be less than range_max.")
    # if score == 'k' and num_ticks > range_max - range_min:
    #     raise ValueError("num_ticks must be greater than the range size.")

    range_size = range_max - range_min
    raw_spacing = range_size / (num_ticks - 1)

    exponent = np.floor(np.log10(raw_spacing))
    fraction = raw_spacing / (10 ** exponent)

    if score == 'k':
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 2.5:
            nice_fraction = 2
        elif fraction < 3.5:
            nice_fraction = 3
        elif fraction < 4.5:
            nice_fraction = 4
        elif fraction < 7:
            nice_fraction = 5
        else:
            nice_fraction = 10

        nice_tick_spacing = nice_fraction * 10 ** exponent
        ticks = range_min + np.arange(num_ticks) * nice_tick_spacing
        ticks = ticks.astype(int)

    elif score == 'd':
        nice_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for option in nice_options:
            if fraction <= option:
                nice_fraction = option
                break
        nice_tick_spacing = nice_fraction * 10 ** exponent
        if range_min + (num_ticks - 1) * nice_tick_spacing < range_max:
            nice_tick_spacing = (range_max - range_min) / (num_ticks - 1)
        ticks = range_min + np.arange(num_ticks) * nice_tick_spacing
        ticks = np.round(ticks, round_precision)
    return ticks


def compute_metrics_auc(adapter, lower_limit_range_for_d, upper_limit_for_k, upper_limit_range_for_d, distances=None, plot=False):

    saturation_points_k, cov_for_saturation_points_k, auc_matrix_k = kAUC(adapter, distances,
        lower_limit_range_for_d=lower_limit_range_for_d, upper_limit_range_for_d=upper_limit_range_for_d,
        upper_limit_for_k=upper_limit_for_k)
    if plot:
        plot_k_or_dAUC(saturation_points_k, cov_for_saturation_points_k, auc_matrix_k, 'k')

    saturation_points_d, cov_for_saturation_points_d, auc_matrix_d = dAUC(adapter, distances,
        lower_limit_range_for_d=lower_limit_range_for_d, upper_limit_range_for_d=upper_limit_range_for_d,
        upper_limit_for_k=upper_limit_for_k)
    if plot:
        plot_k_or_dAUC(saturation_points_d, cov_for_saturation_points_d, auc_matrix_d, 'd')

    saturation_points_c, cost_for_saturation_points, auc_matrix_c = cAUC(adapter,
        upper_limit_for_k=upper_limit_for_k, plot_=plot)

    return (saturation_points_k, cov_for_saturation_points_k, auc_matrix_k,
            saturation_points_d, cov_for_saturation_points_d, auc_matrix_d,
            saturation_points_c, cost_for_saturation_points, auc_matrix_c)


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {str(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj


def write_metrics_auc(name, sat_k, cov_k, auc_k, sat_d, cov_d, auc_d, sat_c, cost, auc_c):
    metrics = {"sat_k": sat_k, "cov_k": cov_k, "auc_k": auc_k, "sat_d": sat_d, "cov_d": cov_d,
               "auc_d": auc_d, "sat_c": sat_c, "cost": cost, "auc_c": auc_c}
    metrics = convert_numpy(metrics)
    with open(name + "/metrics_auc.json", "w") as f:
        json.dump(metrics, f, indent=4)


def compute_metrics_auc_global(adapter, lower_limit_range_for_d, upper_limit_for_k, upper_limit_range_for_d,
                               name, distances=None, plot=False, write=False):
    (saturation_points_k, cov_for_saturation_points_k, auc_matrix_k,
    saturation_points_d, cov_for_saturation_points_d, auc_matrix_d,
    saturation_points_c, cost_for_saturation_points, auc_matrix_c) = compute_metrics_auc(adapter, lower_limit_range_for_d,
                                                                     upper_limit_for_k, upper_limit_range_for_d, distances, plot)
    if write:
        write_metrics_auc(name, saturation_points_k, cov_for_saturation_points_k, auc_matrix_k, saturation_points_d,
            cov_for_saturation_points_d, auc_matrix_d, saturation_points_c, cost_for_saturation_points, auc_matrix_c)
    return (saturation_points_k, cov_for_saturation_points_k, auc_matrix_k,
            saturation_points_d, cov_for_saturation_points_d, auc_matrix_d,
            saturation_points_c, cost_for_saturation_points, auc_matrix_c)