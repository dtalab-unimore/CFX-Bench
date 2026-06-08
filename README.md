# CFX-Bench

This is the official codebase for the paper **"CFX-Bench: A Benchmark for Counterfactual Explanations"**.

---

The full pipeline implemented in the repository is as follows:

![full_pipeline](CFX_bench.png)

---

To run the first three layers, namely Dataset, Generation and Evaluation, run:

```bash
python -u main.py \ 
  --dataset german-credit \ 
  --explainer_name dice \ 
  --model lr \ 
  --test_case_sel_method auto-refuse
```

For each argument, the options are:

- `dataset`
  - `german-credit`
  - `lending-club`
  - `adult`
  - `compas`
  - More datasets can be manually added by exteding the base class `Dataset` in `dataset.py`
- `explainer_name`: the counterfactual generation method
  - `ar`
  - `dice`
  - `face`
  - `nice`
  - `optbin`
  - `proce`
  - `ares`
  - `facegroup`
  - `glance`
  - `globe-ce`
  - `llm-global`
  - `llm-local`
  - More explainers can be added by estending the base class `BaseExplainer` in `explainers/base.py`
- `model_name`: the classifier to train
  - `lr` - **default**
  - More models can be added in `classifiers.py`
- `test_case_sel_method`: which factuals to select
  - `auto-refuse`: records whose positive class predicted probability is below `0.5` - **default**
  - `border`: records whose positive class predicted probability is between `0.45` and `0.55`
  - `neg_border`: records whose positive class predicted probability is between `0.45` and `0.50`
  - `pos_border`: records whose positive class predicted probability is between `0.50` and `0.55`
  - `fp`: false positives
  - `fn`: false negatives
  - Model factual selection methods can be added in `test_case_generator.py`

---

To verbalize the generated counterfactual explanations and evaluate LLM-based metrics, run:

```bash
python -u scripts/llm_eval.py \ 
  --dataset german-credit \ 
  --test_case_sel_method auto-refuse \ 
  --explainer_name dice \ 
  --model lr \ 
  --llm gpt-4o-mini
```

For each arguments, the options are:
- `llm`
  - `llama-3.1-8b`: `Llama-3.1-8B-Instruct`, can run locally with the HuggingFace interface
  - `mistral-small-3.2`: `Mistral-Small-3.2-24B-Instruct-2506`, can run locally but needs its custom interface on top of HuggingFace
  - `gpt-4o-mini`: requires API calls
  - More LLMs can be added `llm_clients`
- `dataset`, `explainer_name`, `model_name`, `test_case_sel_method`: see above.