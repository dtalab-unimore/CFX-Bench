import json
import re

import pandas as pd
from optbinning import Scorecard

from llm_clients import get_llm
from llm_clients.utils import get_model_token
from utils import clean_numpy2_strings
from .base import BaseExplainer, prepare_output
# from .global_explainers.cf_explainer import BaseExplainer
from .global_explainers.prompts import build_local_prompt, get_spec
# from .global_explainers.utils_explainers import prepare_output


class LlmLocalExplainer(BaseExplainer):
    def __init__(self, model: Scorecard, X_train, y_train, features, cat_features, num_features, act_features, target,
                 dataset_name, model_name_or_path='meta-llama/Llama-3.1-8B-Instruct', **kwargs):
        self.dataset_name = dataset_name
        self.model_name_or_path = model_name_or_path
        # self.model = model
        # self.df_train = df_train
        # self.features = features
        # self.cat_features = cat_features
        # self.num_features = num_features
        # self.act_features = act_features
        # self.target = target
        # self.binning_process = model.binning_process_
        super().__init__(
            model, X_train, y_train, features, cat_features, num_features, act_features, target
        )

    def _init(self):
        self.llm = get_llm(self.model_name_or_path, get_model_token(self.model_name_or_path))

        # One spec per dataset drives the prompt; all common prompt logic lives
        # in prompts.build_local_prompt. Adding a dataset is a new spec, not a
        # new branch here.
        self.spec = get_spec(self.dataset_name)

        # The output is now a SPARSE object (only the changed mutable features),
        # so it is small. 512 tokens is generous headroom; the previous value of
        # 100 truncated full-record JSON and made json.loads fail.
        self.max_tokens = 512

    def _explain(self, test_item, n_cf=1):
        # unpack test item
        record, label, pred, proba, target = test_item

        record_text = "\n".join(f"{col}: {val}" for col, val in record.items())

        # the whole prompt (scaffold + dataset specifics + this record) is built
        # by the shared builder
        prompt = build_local_prompt(self.spec, record_text)

        cfs = []
        for _ in range(n_cf):
            text, tokens = self.llm.ask(
                [{'role': 'user', 'content': prompt}],
                max_response_length=self.max_tokens, temperature=0.2
            )
            match = re.search(r"\{.*}", text, re.DOTALL)
            # parse the model's proposed changes
            proposed = json.loads(match.group())

            # Defensive guard: only accept changes to mutable features. Even if the
            # model emits an immutable feature or a hallucinated key, it is dropped
            # here, so immutables can never be altered.
            changes = {k: v for k, v in proposed.items() if k in self.spec.mutable}

            # build the counterfactual: start from the original record and apply the
            # accepted changes (any feature not mentioned keeps its original value)
            cf = record.to_dict()
            cf.update(changes)
            if self.dataset_name == "adult":
                for f in self.cat_features:
                    if not cf[f].startswith(" "):
                        cf[f] = " " + cf[f]
            cfs.append(cf)
        cfs = pd.DataFrame(cfs)

        # prepare the output in the required format
        # print("Record:", record.to_dict())
        # print("Counterfactual:", cfs.iloc[0].to_dict())
        # print()
        record = self.model.binning_process_.transform(record.to_frame().T, metric='bins').iloc[0]
        cfs = self.model.binning_process_.transform(cfs, metric='bins')
        record, cfs = clean_numpy2_strings(record), clean_numpy2_strings(cfs)
        if 'unknown' in cfs.values:
            cfs = cfs.where(cfs != 'unknown', record.values.reshape(1, -1))
            # print("After replacing 'unknown' with original values:", cfs.iloc[0].to_dict())

        # return prepare_output(self.model, cfs[self.features], (record, label, pred, proba, target))
        list_new_probs = self.model.predict_proba(cfs)[:, 1]

        return prepare_output(record, label, pred, proba, target, cfs, list_new_probs)
