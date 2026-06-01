from sklearn.preprocessing import OneHotEncoder

from llm_clients import get_llm
from llm_clients.utils import get_model_token
from .cf_explainer import BaseExplainer
from optbinning import Scorecard
from .utils_explainers import prepare_output, apply_rules_llm
from .scorecard_one_hot import prepare_data_bin
# from llama_cpp import Llama
import time
from .metrics_gcfes import (compute_metrics_global, compute_metrics_auc_global,
                           LOWER_LIMIT_RANGE_FOR_D, UPPER_LIMIT_RANGE_FOR_D, UPPER_LIMIT_FOR_K)
import re
import json
import pandas as pd
from .adapters import LlmAdapter

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

        # define the LLM
        # self.llm = Llama(
        #     model_path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        #     n_ctx=2048,
        #     n_threads=8,
        #     n_gpu_layers=0
        # )
        model_name = 'meta-llama/Llama-3.1-8B-Instruct'
        self.llm = get_llm(model_name, get_model_token(model_name))

        # define the prompt to generate the rules
        # consider the correct prompt depending on the dataset
        prompt = ""
        max_tokens = 0
        if self.dataset_name == "german-credit":
            from prompts import GERMAN_CREDIT_RULES_PROMPT
            prompt = GERMAN_CREDIT_RULES_PROMPT
            max_tokens = 200
        elif dataset_name == "german-credit-crif-mt":
            from .prompts import GERMAN_CREDIT_CRIF_RULES_PROMPT
            prompt = GERMAN_CREDIT_CRIF_RULES_PROMPT
            max_tokens = 1024
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

        # generate the rules
        # output = self.llm(
        #     prompt,
        #     max_tokens=max_tokens,
        #     temperature=0.0,
        #     stop=["</s>"]
        # )
        # text = output["choices"][0]["text"]
        text, tokens = self.llm.ask(
            [{'role': 'user', 'content': prompt}],
            max_response_length=max_tokens, temperature=0.2
        )
        blocks = re.findall(r"\{.*?}", text, re.DOTALL)
        self.rules = [json.loads(block) for block in blocks]
        self.rules = [pd.Series(rule) for rule in self.rules]
        if self.dataset_name == "adult":
            for q, series in enumerate(self.rules):
                for k, value in series.items():
                    if k in self.cat_features:
                        if not self.rules[q][k].startswith(" "):
                            self.rules[q][k] = " " + value

        # end timer to measure training efficiency
        end = time.perf_counter()

        # compute metrics
        # name = self.dataset_name + "_" + NAME
        factuals = self.df_train[self.model.predict(self.df_train[self.features]) == 0]
        training_efficiency = end - start
        _, _ = compute_metrics_global(self.df_train, factuals, self.features, cat_features, num_features, self.target,
                                      self.model, self._explain, self.binning_process, training_efficiency, self.output_dir, write=True)

        # compute metrics auc
        OHE = OneHotEncoder(sparse_output=False, drop=None).fit(X_bins)
        adapter = LlmAdapter(factuals[self.features], self.rules, self.binning_process, OHE)
        # _, _, _, _, _, _, _, _, _ = compute_metrics_auc_global(adapter, LOWER_LIMIT_RANGE_FOR_D, UPPER_LIMIT_FOR_K,
        #                                 UPPER_LIMIT_RANGE_FOR_D, self.output_dir, None, plot=True, write=True)

    def _explain(self, test_item, n_cf=1):
        # unpack test item
        record, label, pred, proba, target = test_item

        # apply rules to compute cfs
        cfs = apply_rules_llm(record, self.rules, self.model)

        # prepare the output in the required format
        return prepare_output(self.model, cfs[self.features], test_item)