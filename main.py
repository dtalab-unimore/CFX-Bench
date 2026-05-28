import argparse
import json
import os
import pickle
import random
import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from optbinning import BinningProcess
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, r2_score
from statsmodels.base.covtype import descriptions
from tqdm import tqdm

from aux_models import ClassifierForBinnedData, ClassifierForMixedDataV2, UniversalProbabilityRescaler
from dataset import get_dataset
from explainers import get_cf_explainer
from test_case_generator import TestCaseGenerator
from utils import get_binning_maps, OrdinalBinsEncoder


def parse_args():
    """ Parse arguments. """
    parser = argparse.ArgumentParser(
        description='Main script.',
        usage='main.py [<args>] [-h | --help]'
    )

    # Main params
    parser.add_argument('--dataset', type=str, choices=[
        'german-credit', 'german-credit-crif', 'german-credit-crif-mt', 'german-credit-crif-full',
        'lending-club', 'lending-club-2', 'lending-club-2-mt', 'lending-club-3',
        'adult', 'heloc', 'compas',
    ])
    parser.add_argument('--model_name', type=str, default='lr', choices=['lr', 'catboost'])
    parser.add_argument('--test_case_sel_method', type=str, default='border',
                        choices=['border', 'neg_border', 'pos_border', 'auto-refuse', 'fp', 'fn'])
    parser.add_argument('--explainer_name', type=str, choices=[
        'ar', 'dice', 'face', 'nice', 'nice-actionables', 'optbin', 'proce', 'crif',
        'ares'
    ])
    parser.add_argument('--tag', type=str, default='', help='Optional tag to append to the output directory name.')
    parser.add_argument('--timestamp', action='store_true', help='Whether to append a timestamp to the output directory name. Ignored if --tag is provided.')
    parser.add_argument('--seed', type=int, default=42)

    # Explainer specific params
    parser.add_argument('--crif-solver', type=str, default='opt', choices=['opt', 'exp'])
    parser.add_argument('--dice-solver', type=str, default='random', choices=['random', 'genetic', 'kdtree'])
    parser.add_argument('--mixed', action="store_true")
    args = parser.parse_args()

    return args


def conf_to_str(conf):
    dataset = conf['dataset']
    model = conf['model_name']
    explainer = conf['explainer_name']
    test_case = conf['test_case_sel_method']
    seed = conf['seed']

    if explainer == 'crif':
        crif_solver = conf['crif_solver']
        explainer = f'{explainer}/{crif_solver}'
    elif explainer == 'dice':
        dice_solver = conf['dice_solver']
        explainer = f'{explainer}/{dice_solver}'

    conf_key = f'{dataset}/{explainer}/{model}__{test_case}__s{seed}'
    return conf_key


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _warp_classifier(clf, threshold_pos):
    clf = deepcopy(clf)

    _supported_classifiers = (ClassifierForBinnedData,)
    if isinstance(clf, ClassifierForBinnedData):
        _supported_estimators = (LogisticRegression,)
        if isinstance(clf.estimator_, LogisticRegression):
            clf.estimator_.intercept_ -= logit(threshold_pos)
        else:
            raise NotImplementedError(f"Unsupported estimator type for warping: {type(clf.estimator_)}")
    else:
        raise NotImplementedError(f"Unsupported classifier type for warping: {type(clf)}")
    return clf


def warp_classifier(clf, threshold_pos):
    clf = deepcopy(clf)

    _supported_classifiers = (LogisticRegression,)
    if isinstance(clf, LogisticRegression):
        clf.intercept_ -= logit(threshold_pos)
    else:
        # raise NotImplementedError(f"Unsupported classifier type for warping: {type(clf)}")
        warnings.warn(f"Unsupported classifier type for warping: {type(clf)}. Returning original classifier.")
    return clf


def compute_stats(metrics: pd.DataFrame) -> pd.Series:
    is_counterfactual = metrics['num_cf_'] > 0
    perc_generated = round(sum(is_counterfactual) / len(is_counterfactual) * 100, 4)
    perc_valid = round(sum(metrics['num_valid_cf'] > 0) / len(metrics) * 100, 4)
    perc_act = round(sum(metrics['num_act_cf'] > 0) / len(metrics) * 100, 4)
    stats = {'perc_generated': perc_generated, 'perc_valid': perc_valid, 'perc_act': perc_act,
             'cf_time (ms)': round(float(metrics['cf_time'].mean()) * 1000, 4)}

    # Metrics below are computed over generated counterfactuals only
    metrics = metrics[is_counterfactual]

    perc_minimal = round(sum(metrics['avg_num_changes_per_cf'] == 1) / len(metrics) * 100, 4)
    stats['perc_minimal_gen'] = perc_minimal

    violations = metrics['avg_num_violations_per_cf'].describe()
    changes = metrics['avg_num_changes_per_cf'].describe()
    target_agg = ['mean', 'min', "50%", "max"]
    agg_map = {'mean': 'avg', 'min': 'min', '50%': 'median', 'max': 'max'}
    stats.update({f'violation_{agg_map[agg]}_gen': round(float(violations[agg]), 4) for agg in target_agg})
    stats.update({f'change_{agg_map[agg]}_gen': round(float(changes[agg]), 4) for agg in target_agg})

    stats['distance_mh_mean_gen'] = round(float(metrics['distance_mh'].mean()), 4)
    stats['plausibility_mean_gen'] = round(float(metrics['plaus_num_cf'].mean()), 4)
    stats['discriminative_power_mean_gen'] = round(float(metrics['accuracy_knn_sklearn'].mean()), 4)
    stats['delta_mean_gen'] = round(float(metrics['delta'].mean()), 4)

    stats = pd.Series(stats)
    stats.name = 'score'

    return stats


def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    args = parse_args()
    conf = vars(args)

    # Set random seed
    set_seed(args.seed)

    # Data loading
    dataset = get_dataset(args.dataset, random_state=args.seed)
    (X_train, X_test), (y_train, y_test) = dataset.get_X(), dataset.get_y()
    features = dataset.get_features()
    num_features, cat_features = dataset.get_num_features(), dataset.get_cat_features()
    act_features = dataset.get_act_features()
    monotonic_features = dataset.get_monotonic_features()
    threshold_pos = dataset.get_threshold_pos()
    threshold_pos = 0.5

    output_dir = 'output'
    if args.timestamp or args.tag:
        output_dir = 'output2'
    output_dir = os.path.join(output_dir, conf_to_str(conf))
    if not np.isclose(threshold_pos, 0.5):
        thr_str = "%.2f" % threshold_pos
        output_dir += f"__thr{thr_str[1:]}"
    if args.tag:
        output_dir = os.path.join(output_dir, args.tag)
    elif args.timestamp:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory:", output_dir)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(dataset.get_config(), f, indent=4)

    if args.model_name == 'lr':
        estimator = LogisticRegression(solver="newton-cholesky", penalty=None)
    elif args.model_name == 'catboost':
        estimator = CatBoostClassifier(random_seed=args.seed)
    else:
        raise NotImplementedError("Supported models: ['lr', 'catboost']")

    X_train_orig, X_test_orig = X_train.copy(), X_test.copy()

    if conf['mixed']:
        # todo: is binning necessary at all?
        # === Mixed version ===
        # act_features = list(set(act_features).union(set(num_features)))
        # monotonic_features = {}
        # n = 0
        # _cat_features = deepcopy(cat_features)
        # cat_features = cat_features + num_features[n:]
        # num_features = num_features[:n]
        # cat_features = sorted(cat_features, key=lambda x: features.index(x))
        # num_features = sorted(num_features, key=lambda x: features.index(x))
        binning_fit_params = dataset.get_binning_fit_params()
        binning_process = BinningProcess(cat_features, binning_fit_params=binning_fit_params, categorical_variables=cat_features)
        # binning_process = BinningProcess(cat_features, binning_fit_params=binning_fit_params, categorical_variables=_cat_features)
        binning_process.fit(X_train[cat_features], y_train)

        X_train.loc[:, cat_features] = binning_process.transform(X_train.loc[:, cat_features], metric='bins')
        X_test.loc[:, cat_features] = binning_process.transform(X_test.loc[:, cat_features], metric='bins')

        if monotonic_features:
            resolve_monotonic_features_placeholders(monotonic_features, binning_process, cat_features, num_features)

        # classifier = ClassifierForMixedData(estimator, binning_process, cat_features, num_features).fit(X_train, y_train)
        classifier = ClassifierForMixedDataV2(
            estimator, binning_process, dataset.get_cat_features(False), num_features, dataset.get_ord_features()
        ).fit(X_train, y_train)
        # ===
    else:
        # === Fully binned version ===
        saved_models_dir = os.path.join('saved_models', conf['dataset'], conf['model_name'], f's{conf["seed"]}')
        os.makedirs(saved_models_dir, exist_ok=True)
        saved_model_path = saved_models_dir + '/model.pkl'
        if os.path.exists(saved_model_path):
            print("Loading saved model")
            with open(saved_model_path, 'rb') as f:
                classifier = pickle.load(f)
            binning_process = classifier.binning_process_
            X_train = binning_process.transform(X_train, metric='bins')
            X_test = binning_process.transform(X_test, metric='bins')
            if monotonic_features:
                resolve_monotonic_features_placeholders(monotonic_features, binning_process, cat_features, num_features)
        else:
            # temporary circuit-breaker to force the use of the already trained model
            # raise RuntimeError(f"No saved model found at {saved_model_path}. Please train the model first by running the script without this check.")

            # monotonic_features = {}
            binning_fit_params = dataset.get_binning_fit_params()
            # for f in num_features:
            #     del binning_fit_params[f]['max_n_bins']
            binning_process = BinningProcess(
                features,
                binning_fit_params=binning_fit_params,
                categorical_variables=cat_features
            )

            binning_process.fit(X_train, y_train)

            X_train = binning_process.transform(X_train, metric='bins')
            X_test = binning_process.transform(X_test, metric='bins')

            if monotonic_features:
                resolve_monotonic_features_placeholders(monotonic_features, binning_process, cat_features, num_features)

            transform_ = {
                'lr': 'woe',
                'catboost': None,
            }[args.model_name]
            estimator_fit_kwargs = {
                'lr': None,
                'catboost': {
                    'cat_features': list(range(len(features))),
                    'eval_set': (X_test, y_test),  # fixme
                }
            }[args.model_name]

            classifier = ClassifierForBinnedData(
                estimator, binning_process, transform_, estimator_fit_kwargs
            ).fit(X_train, y_train)
            with open(saved_model_path, 'wb') as f:
                pickle.dump(classifier, f)
            with open(saved_models_dir + '/binning_fit_params.json', 'w') as f:
                json.dump(binning_fit_params, f, indent=4)
            if monotonic_features:
                with open(saved_models_dir + '/monotonic_features.json', 'w') as f:
                    json.dump(monotonic_features, f, indent=4)

        num_features, cat_features = [], features
        # ===

    # Inference
    y_proba = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)  # threshold=0.5
    report : dict = classification_report(y_test, y_pred, output_dict=True)
    report['auc'] = roc_auc_score(y_test, y_proba[:, 1])
    report['r2'] = r2_score(y_test, y_proba[:, 1])
    print(json.dumps(report, indent=4))
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

    # WARPED CLASSIFIER
    is_classifier_warped = False
    if not np.isclose(threshold_pos, 0.5):
        is_classifier_warped = True
        classifier_orig = deepcopy(classifier)
        # classifier.estimator_ = warp_classifier(classifier.estimator_, threshold_pos)
        classifier.estimator_ = UniversalProbabilityRescaler(classifier.estimator_, threshold_pos).fit(None, None)
        y_proba = classifier.predict_proba(X_test)
        y_pred = classifier.predict(X_test)  # custom threshold

    # Explanation
    test_case = TestCaseGenerator(
        sel_method=args.test_case_sel_method,
        X_test=(X_test_orig if args.explainer_name in ['optbin'] else X_test),
        y_test=y_test,
        y_pred=y_pred,
        y_pos_proba=y_proba[:, 1],
        sample=dataset.test_sample
    )

    # Explainer
    explainer_params = {
        'model': classifier,
        'X_train': X_train,
        'y_train': y_train,
        'features': features,
        'cat_features': cat_features,
        'num_features': num_features,
        'act_features': act_features,
        'target': dataset.get_target(),
        'monotonic_features': monotonic_features,
    }

    # update explainer specific params
    if args.explainer_name == 'dice':
        explainer_params['method'] = args.dice_solver
    elif args.explainer_name == 'crif':
        crif_solver = {'opt': 'optimal', 'exp': 'expert'}[args.crif_solver]
        explainer_params['method'] = crif_solver
        if crif_solver == 'expert':
            explainer_params['efforts'] = dataset.get_feature_costs()
    elif args.explainer_name == 'proce':
        explainer_params['ae_dir'] = os.path.join('/'.join(output_dir.split('/')[:-1]) + '/AE')
        explainer_params['random_state'] = args.seed
        explainer_params['verbose'] = True
    elif args.explainer_name == 'optbin':
        explainer_params.update({
            'binning_process': classifier.binning_process_,
            'estimator': classifier.estimator_,
            'X_train': X_train_orig,  # original, non-binned data
        })
        explainer_params.pop('model')
    elif args.explainer_name == 'ares':
        del explainer_params['X_train']
        del explainer_params['y_train']
        explainer_params['df_train'] = pd.concat([X_train, y_train], axis=1)
        explainer_params['output_dir'] = output_dir
        explainer_params['dataset_name'] = args.dataset

    global_start_time = time.time()

    explainer = get_cf_explainer(
        expl_name=args.explainer_name,
        expl_params=explainer_params
    )

    # from evaluation._evaluator import ExplSetEvaluator
    # evaluator = ExplSetEvaluator(
    #     model=classifier,
    #     continuous_features=num_features,
    #     categorical_features=cat_features,
    #     X_train=X_train,
    #     X_test=X_test,
    #     y_pred=y_pred,
    #     act_features=act_features,
    # )
    from evaluation.evaluator import ExplSetEvaluator
    evaluator = ExplSetEvaluator(
        model=classifier,
        X_train=X_train,
        X_test=X_test,
        cat_features=[], ord_features=[], num_features=[],
        act_features=act_features,
        monotonic_features=monotonic_features,
        binning_process=binning_process,
    )

    list_metrics = []
    list_expl = []
    records_orig = []
    cf_times = []
    for i in tqdm(range(len(test_case))):
        test_item = test_case[i]  # (record, label, pred, proba, target)

        start_time = time.time()
        expl_set = explainer.explain(
            test_item=test_item, n_cf=1
        )
        elapsed_time = time.time() - start_time

        record_id = test_item[0].name
        expl_set.id = int(record_id)
        list_expl.append(expl_set)
        cf_times.append(elapsed_time)
        records_orig.append(X_test_orig.loc[record_id])

        # if i == 2: break  # TODO: debug

    global_elapsed_time = time.time() - global_start_time

    print("Evaluating explanations... ", end='')
    _, bin_id_map = get_binning_maps(classifier.binning_process_)
    feature_costs = np.array([dataset.get_feature_costs().get(f, 0) for f in features])
    enc = None
    for expl_set, elapsed_time in tqdm(zip(list_expl, cf_times), total=len(list_expl)):
        metrics = evaluator.evaluate(
            expl_set=expl_set,
            max_num_cf=1
        )
        metrics['cf_time'] = elapsed_time
        list_metrics.append(metrics)

        if hasattr(expl_set, 'costs') and expl_set.costs is not None:
            costs = expl_set.costs
            cost = np.mean(costs)
        else:
            # diffs = (expl_set.list_expl_full.replace(bin_id_map).values -
            #          expl_set.record.to_frame().T.replace(bin_id_map).values)
            if enc is None:
                enc = OrdinalBinsEncoder(features, monotonic_features, binning_process).fit(X_train, y_train)
            if (cfs := expl_set.list_expl_full).shape[0] > 0:
                diffs = enc.transform(cfs) - enc.transform(expl_set.record.to_frame().T)
                costs = np.abs(diffs) * feature_costs
                costs = costs.sum(axis=-1)
                cost = np.mean(costs)
            else:
                costs = None
                cost = np.nan
            expl_set.costs = costs
        metrics['Total cost'] = cost
    print("Done.")

    metrics_df = pd.DataFrame(list_metrics)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'))
    global_stats = compute_stats(metrics_df)
    global_stats['Total generation time (s)'] = round(global_elapsed_time, 4)
    global_stats['Average total cost'] = round(metrics_df['Total cost'].mean(), 4)
    with open(os.path.join(output_dir, 'STATS.json'), 'w') as f:
        json.dump(global_stats.to_dict(), f, indent=4)

    # REVERT TO ORIGINAL CLASSIFIER IF WARPED
    if is_classifier_warped:
        for expl in list_expl:
            expl.proba = classifier_orig.predict_proba(expl.get_record().to_frame().T)[0, 1]
            if len(expl) > 0:
                expl.list_new_probs = pd.Series(classifier_orig.predict_proba(expl.list_expl_full)[:, 1])

    # # --- Temporary patch: flip every label and prediction (1-x) for consistency with previous results.
    # for expl in list_expl:
    #     expl.label = 1 - expl.label
    #     expl.pred = 1 - expl.pred
    #     expl.proba = 1 - expl.proba
    #     expl.target = 1 - expl.target
    #     expl.list_new_preds = 1 - expl.list_new_preds
    #     expl.list_new_probs = 1 - expl.list_new_probs
    # # ---
    for record_orig, expl in zip(records_orig, list_expl):
        expl.record = record_orig
    with open(os.path.join(output_dir, 'CF.json'), 'w') as f:
        json.dump([expl.to_json() for expl in list_expl], f, indent=4)


def resolve_monotonic_features_placeholders(monotonic_features: dict, binning_process_: BinningProcess,
                                            cat_features: list=None, num_features: list=None):
    bp = binning_process_
    if cat_features is None:
        cat_features = []
    if num_features is None:
        num_features = []
    for feature, categories in monotonic_features.items():
        if feature in num_features and categories in ['num_asc', 'num_desc'] and feature in bp._binned_variables:
            binned_variable = bp.get_binned_variable(feature)
            binning_table = binned_variable.binning_table.build(add_totals=False)
            binning_table = binning_table.query("Bin not in ['Missing', 'Special']")
            categories_ = binning_table['Bin'].astype(str).tolist()
            if categories == 'num_desc':
                categories_ = categories_[::-1]
            monotonic_features[feature] = categories_
        elif feature in cat_features and categories in ['event_asc', 'event_desc']:
            binned_variable = bp.get_binned_variable(feature)
            binning_table = binned_variable.binning_table.build(add_totals=False)
            binning_table = binning_table.query("Bin not in ['Missing', 'Special']")
            binning_table = binning_table.sort_values(by='Event rate', ascending=(categories == 'event_asc'))
            categories_ = binning_table['Bin'].astype(str).tolist()
            monotonic_features[feature] = categories_


if __name__ == '__main__':
    main()
