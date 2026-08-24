# Evaluation Metrics

CFX-Bench evaluates counterfactual explanations along complementary quantitative and semantic dimensions. The 
quantitative metrics are mainly adapted from the [Scamander repository](https://github.com/riccotti/Scamander), while 
semantic dimensions are derived from the [CounterEval repository](https://github.com/anitera/CounterEval).

## Quantitative metrics

| **Metric**               | **Direction** | **Description** |
|--------------------------|---------------|-----------------|
| **Actionability**        | ↑             |Fraction of factual instances for which the method produces counterfactual recourse satisfying the configured actionability constraints.
| **Proximity**            | ↓             |Distance between the factual instance and its counterfactual; lower values indicate smaller interventions.
| **Plausibility**         | ↓             |Measures how compatible the generated counterfactual is with the observed data distribution.
| **Discriminative Power** | ↑             |Measures the informativeness of generated counterfactuals through the predictive accuracy of a 1-NN classifier trained on factual and counterfactual instances.
| **Delta Probability**    | ↓             |Change in model confidence associated with the counterfactual prediction flip.
| **Sparsity**             | ↓             |Number of features modified by the counterfactual; lower values correspond to simpler interventions.
| **Recourse Cost**        | ↓             |Effort associated with the proposed intervention, computed using dataset-specific feature costs.
| **CF Time**              | ↓             |Average time required to generate a counterfactual explanation.

For global counterfactual methods, CFX-Bench additionally evaluates population-level recourse through coverage-cost analyses. The framework reports:

- **kAUC**, summarizing coverage as the number of available recourse actions increases;
- **dAUC**, summarizing coverage under different cost constraints;
- **cAUC**, summarizing the cost required to reach different population coverage levels.

## Semantic and human-centered metrics

|Metric |Direction|Interpretation
|-|-|-
|**Satisfaction**|↑|Overall usefulness of the proposed recourse from a user-oriented perspective.
|**Feasibility**|↑|Practical realism of the proposed changes.
|**Consistency**|↑|Internal coherence of the explanation and proposed intervention.
|**Completeness**|↑|Whether the explanation provides sufficient information to understand the proposed recourse.
|**Trust**|↑|Degree to which the explanation appears reliable and credible.
|**Understandability**|↑|Ease with which the proposed recourse can be interpreted by a user.
|**Fairness**|↑|Whether the explanation avoids inappropriate or discriminatory recommendations.
|**Complexity**|↓|Cognitive or structural complexity of the explanation.
|**Overall**|↑|Aggregate semantic usefulness score.

Semantic metrics can be computed independently of the counterfactual generation method, allowing explanations produced 
by local, global, and LLM-based methods to be assessed through a shared human-centered protocol.

For exact metric definitions and implementation details, refer to the corresponding evaluation modules in `evaluation/`.

### Semantic evaluation

Semantic evaluation is performed after counterfactual generation through
`scripts/llm_eval.py`.

The evaluation assesses each explanation along the previous dimensions.

```bash
python -u scripts/llm_eval.py \ 
  --dataset german-credit \ 
  --test_case auto-refuse \ 
  --explainer_name dice \ 
  --model_name lr \ 
  --llm gpt-4o-mini
```

Evaluation prompts are available in `llm_prompts/`, allowing the judging
protocol to be inspected and adapted.

