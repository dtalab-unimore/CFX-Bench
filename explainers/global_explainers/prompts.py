"""
Counterfactual-rule prompt builder.

All datasets share ONE scaffold (role framing, no-code guard, objective,
reasoning, hard constraints, output format). Each dataset supplies only what
actually differs: its name, the record/flip narrative, the direction hint, and
its feature lists. The four public prompt strings
(GERMAN_CREDIT_RULES_PROMPT, LENDING_RULES_PROMPT, COMPAS_RULES_PROMPT,
ADULT_RULES_PROMPT) are produced by build_rules_prompt() so they can never
drift apart again.

Feature lists are stored as DATA (mutable dict + immutable list) and rendered
into the prompt, which lets validate_spec() catch a feature that is
simultaneously marked mutable and immutable.
"""

from dataclasses import dataclass, field
from typing import Union

# A mutable feature is either an integer range (lo, hi) or a list of allowed
# categorical values (verbatim strings).
IntRange = tuple[int, int]
Categorical = list[str]
FeatureValue = Union[IntRange, Categorical]


# ---------------------------------------------------------------------------
# Shared scaffold (written once)
# ---------------------------------------------------------------------------

_NO_CODE = """You do NOT write, describe, or output code of any kind (no Python, no
pseudocode, no scripts). You do NOT train or invoke any model. You directly
produce the JSON output described below, reasoning only from the feature
semantics given in this prompt. Any code in your answer is a failure."""

_OBJECTIVE = """OBJECTIVE
Produce rules that flip the prediction with the smallest, most plausible
changes: change as few features as possible (sparsity) and by the smallest
magnitude that still flips the outcome (proximity). Across the full set of
rules, prefer DIVERSE feature combinations so the rules cover different
subgroups of {subject_plural}."""

_REASONING = """REASONING
Reason internally about which mutable features to change and in which
direction {direction_hint}. Do NOT output this internal deliberation.
The only explanatory text in your answer is the per-rule "reasoning" field
defined in OUTPUT FORMAT — one concise sentence per rule, inside the JSON.
Emit no prose, commentary, or markdown outside the JSON array."""

HARD_CONSTRAINTS = """HARD CONSTRAINTS
- Each rule changes exactly 2-3 features.
- Only the MUTABLE features below may appear in a rule.
- NEVER include an immutable feature in any rule.
- Categorical features: use only the listed allowed values, verbatim.
- Numeric features: integers within the stated [min, max] range.
- The "reasoning" field is a single sentence; do not restate the feature
  values as a list within it.
- Do NOT default to a feature's maximum or best value: a rule that
  moves each of its features by one or two levels is preferred over a rule
  that pins any feature to its extremes."""

# Backward-compatible alias for any downstream code importing the old name.
RULES_PROMPT = HARD_CONSTRAINTS

_OUTPUT_FORMAT = """OUTPUT FORMAT
Return a single JSON array containing {n_rules} objects. Each object has exactly
two fields:
- "reasoning": one concise sentence explaining why these changes {flip_phrase}.
- "changes": an object of 2-3 "feature": value pairs drawn only from the
  mutable features above.

Structure to follow (placeholders, not real values — replace each <...> with
an actual mutable feature name and a valid value for it; categorical values
must be quoted strings, numeric values must be bare integers):
[
  {{
    "reasoning": "<one concise sentence>",
    "changes": {{"<feature>": "<categorical_value>", "<feature>": <integer_value>}}
  }},
  {{
    "reasoning": "<one concise sentence>",
    "changes": {{"<feature>": "<categorical_value>", "<feature>": <integer_value>, "<feature>": "<categorical_value>"}}
  }}
]

Output only the JSON array. No code, no prose."""


# ---------------------------------------------------------------------------
# Per-dataset specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    name: str            # display name, e.g. "German Credit"
    record_desc: str     # singular record description (the to-be-flipped case)
    subject_plural: str  # plural subject, used in role + diversity sentences
    source_label: str    # current predicted class, e.g. "REJECTED"
    target_label: str    # target class after the flip, e.g. "APPROVED"
    direction_hint: str  # how to move features (reasoning clause)
    flip_phrase: str     # gloss for the per-rule reasoning field
    mutable: dict[str, FeatureValue]
    immutable: list[str]


def _render_value(v: FeatureValue) -> str:
    if isinstance(v, list):  # categorical
        return "[" + ", ".join(f"'{item}'" for item in v) + "]"
    lo, hi = v  # integer range
    return f"integer, {lo} - {hi}"


def render_features_block(spec: DatasetSpec,
                          immutable_note: str = "never include in any rule") -> str:
    mut = "\n".join(f"- {name}: {_render_value(val)}"
                    for name, val in spec.mutable.items())
    imm = ", ".join(spec.immutable)
    return (
        "MUTABLE FEATURES (allowed values)\n"
        f"{mut}\n\n"
        f"IMMUTABLE FEATURES ({immutable_note})\n"
        f"{imm}"
    )


def validate_spec(spec: DatasetSpec) -> None:
    """Fail loudly on the inconsistency class found in the original file:
    a feature listed as both mutable and immutable, or duplicate names."""
    overlap = set(spec.mutable) & set(spec.immutable)
    if overlap:
        raise ValueError(
            f"[{spec.name}] feature(s) marked BOTH mutable and immutable: "
            f"{sorted(overlap)}"
        )


def build_rules_prompt(spec: DatasetSpec, n_rules: str = "5-10") -> str:
    validate_spec(spec)
    role = (
        "You are the generator itself. You produce global counterfactual "
        f"explanation\nrules for the {spec.name} dataset. Every record is "
        f"{spec.record_desc}. A rule\nis a small set of feature changes that, "
        f"applied to {spec.subject_plural}, is\nintended to flip the model's "
        f"prediction from {spec.source_label} to {spec.target_label}."
    )
    return "\n\n".join([
        role,
        _NO_CODE,
        _OBJECTIVE.format(subject_plural=spec.subject_plural),
        _REASONING.format(direction_hint=spec.direction_hint),
        HARD_CONSTRAINTS,
        render_features_block(spec),
        _OUTPUT_FORMAT.format(n_rules=n_rules, flip_phrase=spec.flip_phrase),
    ])


# ---------------------------------------------------------------------------
# Local (single-record) counterfactual builder
# ---------------------------------------------------------------------------
# Shares the dataset specs, the no-code guard, and the feature data with the
# global builder. Differs in task and output: given ONE record, return ONE
# JSON object containing only the mutable features that were changed. Immutable
# features are never emitted, so they cannot be corrupted, and the merge step
# in the explainer keeps every unchanged value from the original record.

_LOCAL_OBJECTIVE = """OBJECTIVE
Modify the single record shown below so the model's prediction flips from
{source_label} to {target_label}. Make the smallest, most plausible change:
alter as few mutable features as needed (sparsity) and move each one by the
smallest magnitude that still flips the outcome (proximity). Do NOT pin
features to their maximum or best value."""

_LOCAL_REASONING = """REASONING
Reason internally about which mutable features to change and in which
direction {direction_hint}. Do NOT output this internal deliberation, and do
NOT add any explanatory field to the JSON. Emit only the JSON object described
below — no prose, commentary, or markdown."""

_LOCAL_CONSTRAINTS = """HARD CONSTRAINTS
- Only MUTABLE features may be changed; never change or emit an immutable one.
- Categorical features: use a listed allowed value, copied verbatim (including
  any leading spaces).
- Numeric features: an integer within the stated [min, max] range.
- Prefer moving a feature by one or two levels over jumping to its extreme."""

_LOCAL_OUTPUT = """OUTPUT FORMAT
Return a single JSON object mapping each CHANGED mutable feature to its new
value. Include only the features you actually change (as few as needed,
typically 2-4); omit unchanged features and omit every immutable feature.

Structure to follow (placeholders — replace each <...> with a mutable feature
name and a valid new value; categorical values are quoted strings, numeric
values are bare integers):
{{
  "<feature>": "<categorical_value>",
  "<feature>": <integer_value>
}}

Output only the JSON object. No code, no prose."""


def build_local_prompt(spec: DatasetSpec, record_text: str) -> str:
    validate_spec(spec)
    role = (
        "You are the generator itself. You produce a local counterfactual "
        f"explanation\nfor a single record from the {spec.name} dataset. The "
        f"record is {spec.record_desc};\nyour job is to change a few of its "
        "mutable features so the model's prediction\nflips from "
        f"{spec.source_label} to {spec.target_label}."
    )
    features = render_features_block(
        spec, immutable_note="never change these; never put them in the output")
    return "\n\n".join([
        role,
        _NO_CODE,
        _LOCAL_OBJECTIVE.format(source_label=spec.source_label,
                                target_label=spec.target_label),
        _LOCAL_REASONING.format(direction_hint=spec.direction_hint),
        _LOCAL_CONSTRAINTS,
        features,
        f"RECORD TO MODIFY\n{record_text}",
        _LOCAL_OUTPUT,
    ])


def get_spec(dataset_name: str) -> DatasetSpec:
    """Look up a spec by the dataset_name used in the explainers
    (hyphen or underscore both accepted, e.g. 'german-credit')."""
    return SPECS[dataset_name.replace("-", "_")]


# ---------------------------------------------------------------------------
# Dataset specs
# ---------------------------------------------------------------------------
# NOTE: source_label / target_label are written to match each dataset's
# narrative. Make sure they match the actual class strings your classifier
# emits, or map them where you parse the output.

GERMAN_CREDIT_SPEC = DatasetSpec(
    name="German Credit",
    record_desc="a loan applicant whose application was REJECTED",
    subject_plural="rejected applicants",
    source_label="REJECTED",
    target_label="APPROVED",
    direction_hint=("increases the chance of approval (move features toward "
                    "values associated with higher creditworthiness)"),
    flip_phrase="move the applicant toward approval",
    mutable={
        "account_check_status": ["no checking account", "< 0 DM",
                                 "0 <= ... < 200 DM",
                                 ">= 200 DM / salary assignments for at least 1 year"],
        "duration_in_month": (4, 72),
        "credit_amount": (250, 18424),
        "savings": ["unknown/ no savings account", "... < 100 DM",
                    "100 <= ... < 500 DM", "500 <= ... < 1000 DM",
                    "... >= 1000 DM"],
        "present_emp_since": ["unemployed", "... < 1 year",
                              "1 <= ... < 4 years", "4 <= ... < 7 years",
                              "... >= 7 years"],
        "installment_as_income_perc": (1, 4),
        "other_debtors": ["co-applicant", "none", "guarantor"],
        "property": ["real estate",
                     "building society savings/life insurance",
                     "unknown/no property", "car or other"],
        # "age": (19, 75),
        "other_installment_plans": ["none", "bank", "stores"],
        "credits_this_bank": (1, 4),
        "job": ["skilled employee/official", "unskilled resident",
                "management/self-employed/highly qualified",
                "unemployed/unskilled non-resident"],
        "telephone": ["yes", "none"],
    },
    immutable=["credit_history", "purpose", "personal_status_sex",
               "present_res_since", "housing", "people_under_maintenance",
               "foreign_worker", "age"],
)

LENDING_SPEC = DatasetSpec(
    name="Lending Club",
    record_desc="a borrower who did NOT repay their loan",
    subject_plural="borrowers who defaulted",
    source_label="DEFAULTED",
    target_label="REPAID",
    direction_hint=("increases the chance of repayment (move features toward "
                    "values associated with higher repayment reliability)"),
    flip_phrase="move the borrower toward repayment",
    mutable={
        "emp_length": (1, 10),
        "annual_inc": (12252, 195000),
        "credit_years": (3, 43),
    },
    immutable=["open_acc", "grade", "home_ownership", "purpose", "addr_state"],
)

COMPAS_SPEC = DatasetSpec(
    name="COMPAS",
    record_desc="a defendant predicted NOT to reoffend",
    subject_plural="defendants predicted not to reoffend",
    source_label="WILL NOT REOFFEND",
    target_label="WILL REOFFEND",
    direction_hint="increases the predicted likelihood of recidivism",
    flip_phrase="raise the predicted recidivism risk",
    mutable={
        "priors_count": (0, 36),
        "days_b_screening_arrest": (0, 812),
        "length_of_stay": (0, 530),
        "c_charge_degree": ["F", "M"],
        "is_violent_recid": (0, 1),
        "score_text": ["Low", "Medium", "High"],
    },
    immutable=["sex", "age", "race"],
)

ADULT_SPEC = DatasetSpec(
    name="Adult Income",
    record_desc="a person whose annual income is at or below the threshold (<=50K)",
    subject_plural="people earning at or below the threshold",
    source_label="<=50K",
    target_label=">50K",
    direction_hint=("increases the predicted income above the threshold (move "
                    "features toward values associated with higher earnings)"),
    flip_phrase="move the person above the income threshold",
    mutable={
        "age": (17, 90),
        # Leading spaces in the categorical values are preserved on purpose:
        # the raw Adult CSV stores them that way, and the verbatim-match
        # constraint requires the rules to reproduce them exactly.
        "workclass": [" State-gov", " Self-emp-not-inc", " Private",
                      " Federal-gov", " Local-gov", " Self-emp-inc"],
        "education": [" Bachelors", " HS-grad", " 11th", " Masters", " 9th",
                      " Some-college", " Assoc-acdm", " 7th-8th", " Doctorate",
                      " Assoc-voc", " Prof-school", " 5th-6th", " 10th",
                      " Preschool", " 12th", " 1st-4th"],
        "occupation": [" Adm-clerical", " Exec-managerial",
                       " Handlers-cleaners", " Prof-specialty",
                       " Other-service", " Sales", " Transport-moving",
                       " Farming-fishing", " Machine-op-inspct",
                       " Tech-support", " Craft-repair", " Protective-serv",
                       " Armed-Forces", " Priv-house-serv"],
        "hours-per-week": (1, 99),
    },
    immutable=["marital-status", "race", "sex"],
)

SPECS: dict[str, DatasetSpec] = {
    "german_credit": GERMAN_CREDIT_SPEC,
    "lending": LENDING_SPEC,
    "compas": COMPAS_SPEC,
    "adult": ADULT_SPEC,
}


# ---------------------------------------------------------------------------
# Public prompt strings (same names as before; now generated, not hand-written)
# ---------------------------------------------------------------------------

GERMAN_CREDIT_RULES_PROMPT = build_rules_prompt(GERMAN_CREDIT_SPEC)
LENDING_RULES_PROMPT = build_rules_prompt(LENDING_SPEC)
COMPAS_RULES_PROMPT = build_rules_prompt(COMPAS_SPEC)
ADULT_RULES_PROMPT = build_rules_prompt(ADULT_SPEC)


if __name__ == "__main__":
    sample_record = "account_check_status: < 0 DM\nage: 28\ncredit_amount: 4200"
    for key, spec in SPECS.items():
        prompt = build_rules_prompt(spec)
        local = build_local_prompt(spec, sample_record)
        assert "IMMUTABLE FEATURES" in prompt
        assert "RECORD TO MODIFY" in local
        validate_spec(spec)
        print(f"{key:14s} ok  rules={len(prompt):>4} chars  "
              f"local={len(local):>4} chars  "
              f"({len(spec.mutable)} mutable / {len(spec.immutable)} immutable)")
    print("\n--- German Credit LOCAL prompt preview ---\n")
    print(build_local_prompt(GERMAN_CREDIT_SPEC, sample_record))