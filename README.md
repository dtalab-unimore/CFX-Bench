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
  --expl dice \ 
  --test_case auto-refuse \ 
  --model lr
```

---

To verbalize the generated counterfactual explanations and evaluate LLM-based metrics, run:

```bash
python -u scripts/llm_eval.py \ 
  --dataset german-credit \ 
  --expl dice \ 
  --test_case auto-refuse \ 
  --model lr \ 
  --llm gpt-4o-mini
```
