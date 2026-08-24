# Reproducibilty

CFX-Bench is designed to support reproducible comparisons across counterfactual explanation methods. 
Each experiment is identified by its configuration and produces a dedicated output directory. A standard run generates:

- `metrics.csv`: instance-level counterfactual evaluation metrics;
- `STATS.json`: metrics aggregated over the evaluated instances;
- `CF.json`: generated counterfactual explanations.

For global counterfactual explainers, the global evaluation module generates additional population-level coverage/cost statistics and AUC-based results.

## Reproducing an experiment

For example, the following command evaluates DiCE on German Credit using Logistic Regression and automatically selects rejected instances as factual cases:

```
python -u main.py \
  --dataset german-credit \
  --explainer_name dice \
  --model_name lr \
  --test_case_sel_method auto-refuse \
  --seed 42
```

The same experimental protocol can be applied to another supported explainer by changing `--explainer_name`, while 
keeping the dataset, classifier, factual-selection strategy, and random seed fixed.

The seed specification and this common execution pipeline ensure that methods are evaluated using the same 
preprocessing, predictive model, factual instances, actionability constraints, feature costs, and metric 
implementations.
