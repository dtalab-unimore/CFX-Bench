import json
import re

import pandas as pd
from optbinning import Scorecard

from llm_clients import get_llm
from llm_clients.utils import get_model_token
from .cf_explainer import BaseExplainer
from .utils_explainers import prepare_output


class LlmLocalExplainer(BaseExplainer):
    def __init__(self, model: Scorecard, df_train, features, cat_features, num_features, act_features, target, dataset_name, **kwargs):
        self.model = Scorecard(
            model.binning_process_, model.estimator_
        ).fit(df_train[features], df_train[target])
        self.df_train = df_train
        self.features = features
        self.cat_features = cat_features
        self.num_features = num_features
        self.act_features = act_features
        self.target = target
        self.dataset_name = dataset_name

        # define the LLM
        model_name = 'meta-llama/Llama-3.1-8B-Instruct'
        self.llm = get_llm(model_name, get_model_token(model_name))

        # consider the correct prompt depending on the dataset
        self.column_values_local_prompt, self.context = "", ""
        if self.dataset_name == "german-credit":
            from .prompts import COLUMN_VALUES_LOCAL_PROMPT_GERMAN_CREDIT, GERMAN_CREDIT_CONTEXT
            self.column_values_local_prompt = COLUMN_VALUES_LOCAL_PROMPT_GERMAN_CREDIT
            self.context = GERMAN_CREDIT_CONTEXT
            self.max_tokens = 1024
        elif self.dataset_name == "lending":
            from prompts import COLUMN_VALUES_LOCAL_PROMPT_LENDING, LENDING_CONTEXT
            self.column_values_local_prompt = COLUMN_VALUES_LOCAL_PROMPT_LENDING
            self.context = LENDING_CONTEXT
            self.max_tokens = 100
        elif self.dataset_name == "compas":
            from prompts import COLUMN_VALUES_LOCAL_PROMPT_COMPAS, COMPAS_CONTEXT
            self.column_values_local_prompt = COLUMN_VALUES_LOCAL_PROMPT_COMPAS
            self.context = COMPAS_CONTEXT
            self.max_tokens = 100
        elif self.dataset_name == "adult":
            from prompts import COLUMN_VALUES_LOCAL_PROMPT_ADULT, ADULT_CONTEXT
            self.column_values_local_prompt = COLUMN_VALUES_LOCAL_PROMPT_ADULT
            self.context = ADULT_CONTEXT
            self.max_tokens = 100

    def _explain(self, test_item, n_cf=1):
        # unpack test item
        record, label, pred, proba, target = test_item

        record_text = "\n".join(f"{col}: {val}" for col, val in record.items())

        # define the prompt to generate local counterfactuals
        prompt = (
                  "You are a system that generates counterfactual explanations for tabular records " + self.context + "\n" +
                  "Modify the following record to flip the prediction.\n\n" + "Rules:\n" + "- Return ONLY valid JSON.\n" +
                  "- Do NOT include ANY extra text.\n" + "- Keys must match and contain ALL the record's column names.\n" +
                  "- Values must respect column types.\n" + "- Use only allowed values for features.\n" +
                  "- Do NOT change IMMUTABLE features, use the existing value you find in the record\n\n" +
                  self.column_values_local_prompt + "Record:\n\n" + record_text +
                  "\n\nReturn ONLY a valid JSON object enclosed in {} with NO extra text."
        )

        # generate the output
        # output = self.llm(
        #     prompt,
        #     max_tokens=self.max_tokens,
        #     temperature=0.0,
        #     stop=["</s>"]
        # )
        # text = output["choices"][0]["text"]
        text, tokens = self.llm.ask(
            [{'role': 'user', 'content': prompt}],
            max_response_length=self.max_tokens, temperature=0.2
        )
        match = re.search(r"\{.*}", text, re.DOTALL)
        # build the counterfactual
        cfs = json.loads(match.group())
        cfs = pd.Series(cfs).to_frame().T
        if self.dataset_name == "adult":
            for f in self.cat_features:
                if not cfs[f][0].startswith(" "):
                    cfs[f] = " " + cfs[f]

        # prepare the output in the required format
        return prepare_output(self.model, cfs[self.features], test_item)