# Extending CFX-Bench

CFX-Bench supports new datasets, predictive models, counterfactual explainers, and LLM backends without modifying the evaluation workflow.

## Adding a counterfactual explainer

New instance-level explainers should extend `BaseExplainer` in `explainers/base.py`.
A minimal implementation has the following structure:

```python
from explainers.base import BaseExplainer


class MyExplainer(BaseExplainer):

    def _init(self):
        """Initialize method-specific objects or preprocessing."""
        pass

    def _explain(self, test_item, n_cf=1):
        record, label, pred, proba, target = test_item

        # Run the counterfactual generation method.
        #
        # cfs must contain the generated counterfactual instances.
        # cfs_proba must contain their predicted probabilities.
        cfs = ...
        cfs_proba = ...

        return BaseExplainer.prepare_output(
            record=record,
            label=label,
            pred=pred,
            proba=proba,
            target=target,
            cfs=cfs,
            cfs_proba=cfs_proba,
        )
```

The new explainer must then be registered in the explainer factory so that it can be selected through:

```--explainer_name my-explainer```

Once registered, the method automatically benefits from the common CFX-Bench evaluation and output pipeline.

## Adding a dataset

New tabular classification datasets can be integrated by extending the dataset abstraction implemented in `dataset.py`. 
A dataset configuration must expose the information required by the common pipeline, including its feature schema, 
numerical and categorical attributes, actionable features, target variable, monotonicity constraints when applicable, 
and feature-specific recourse costs.

## Adding a predictive model
Additional predictive models can be registered in `classifiers.py`. Models integrated into CFX-Bench must expose the 
prediction interface required by the counterfactual generation and evaluation components.

## Adding an LLM backend
New language models can be integrated through `llm_clients/`. This allows LLM-based counterfactual generation and 
semantic evaluation to use different local or API-based models while preserving the remaining benchmark pipeline.
