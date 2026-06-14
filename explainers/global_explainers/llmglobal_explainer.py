import json
import os.path
import time

import pandas as pd
from optbinning import Scorecard
from sklearn.preprocessing import OneHotEncoder

from llm_clients import get_llm
from llm_clients.utils import get_model_token
from utils import clean_numpy2_strings
from .adapters import LlmAdapter
from .cf_explainer import BaseExplainer, _empty_explanation_dict
from .utils_explainers import prepare_output, apply_rules_llm

NAME = "llm-global"


class LlmGlobalExplainer(BaseExplainer):
    def __init__(self, model: Scorecard, df_train, features, cat_features, num_features, act_features, target,
                 dataset_name, output_dir, **kwargs):
        self.model = model
        self.df_train = df_train
        self.features = features
        self.cat_features = cat_features
        self.num_features = num_features
        self.act_features = act_features
        self.target = target
        self.dataset_name = dataset_name
        self.output_dir = output_dir

        # prepare data
        # self.model_bin, X_bins, self.binning_process, y = prepare_data_bin(self.df_train, self.features, self.target, self.model)
        self.model_bin, X_bins, self.binning_process, y = self.model, df_train[features].copy(), self.model.binning_process_, df_train[target].copy()

        # start timer to measure training efficiency
        start = time.perf_counter()

        # define the prompt to generate the rules
        # consider the correct prompt depending on the dataset
        prompt = ""
        max_tokens = 0
        if self.dataset_name == "german-credit":
            from .prompts import GERMAN_CREDIT_RULES_PROMPT
            prompt = GERMAN_CREDIT_RULES_PROMPT
            max_tokens = 2048
        elif self.dataset_name == "lending":
            from prompts import LENDING_RULES_PROMPT
            prompt = LENDING_RULES_PROMPT
            max_tokens = 100
        elif self.dataset_name == "compas":
            from prompts import COMPAS_RULES_PROMPT
            prompt = COMPAS_RULES_PROMPT
            max_tokens = 100
        elif self.dataset_name == "adult":
            from prompts import ADULT_RULES_PROMPT
            prompt = ADULT_RULES_PROMPT
            max_tokens = 400

        # model_name = 'Meta-Llama-3.1-8B-Instruct'
        model_name = 'gpt-4.1-mini'
        rules_file = f'{output_dir}/rules_{model_name}.json'
        if os.path.exists(rules_file):
            with open(rules_file, 'r') as f:
                rules = json.load(f)
                self.rules = [pd.Series(r) for r in rules]
        else:
            self.llm = get_llm(model_name, get_model_token(model_name))
            text, tokens = self.llm.ask(
                [{'role': 'user', 'content': prompt}],
                max_response_length=max_tokens, temperature=0.2
            )
            ans = json.loads(text)
            self.rules = [pd.Series(obj['changes']) for obj in ans]
            with open(rules_file, 'w') as f:
                json.dump([r.to_dict() for r in self.rules], f)
        rules_df = pd.DataFrame(self.rules, columns=self.features)
        rules_df = self.model.binning_process_.transform(rules_df, metric='bins')
        self.rules = [clean_numpy2_strings(r[r!='Missing']) for _, r in rules_df.iterrows()]
        # if self.dataset_name == "adult":
        #     for q, series in enumerate(self.rules):
        #         for k, value in series.items():
        #             if k in self.cat_features:
        #                 if not self.rules[q][k].startswith(" "):
        #                     self.rules[q][k] = " " + value

        # end timer to measure training efficiency
        end = time.perf_counter()

        self.training_efficiency = end - start
        OHE = OneHotEncoder(sparse_output=False, drop=None).fit(X_bins)
        self.adapter = LlmAdapter(None, self.rules, self.binning_process, OHE)

    def _explain(self, test_item, n_cf=1):
        # unpack test item
        record, label, pred, proba, target = test_item

        # apply rules to compute cfs
        cfs = apply_rules_llm(record, self.rules, self.model)

        # prepare the output in the required format
        if cfs is None:
            return _empty_explanation_dict(test_item)
        return prepare_output(self.model, cfs[self.features], test_item)