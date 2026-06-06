COLUMN_VALUES_LOCAL_PROMPT_LENDING = (
                  "Column names and allowed values:\n" +
                  "- emp_length: 1 - 10\n" +
                  "- annual_inc: 12252 - 195000\n" +
                  "- open_acc: 2 - 34\n" +
                  "- credit_years: 3 - 43\n" +
                  "- grade: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- home_ownership: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- purpose: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- addr_state: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n\n"
)

COLUMN_VALUES_LOCAL_PROMPT_COMPAS = (
                  "Column names and allowed values:\n" +
                  "- sex: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- age: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- race: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- priors_count: 0 - 36\n" +
                  "- days_b_screening_arrest: 0 - 812\n" +
                  "- length_of_stay: 0 - 530\n" +
                  "- c_charge_degree: ['F' 'M']\n" +
                  "- is_violent_recid: 0 - 1\n" +
                  "- score_text: ['Low' 'Medium' 'High']\n\n"
)

COLUMN_VALUES_LOCAL_PROMPT_ADULT = (
                  "Column names and allowed values:\n" +
                  "- age: 17 - 90\n" +
                  "- workclass: [' State-gov' ' Self-emp-not-inc' ' Private' ' Federal-gov' ' Local-gov' ' Self-emp-inc']\n" +
                  "- education: [' Bachelors' ' HS-grad' ' 11th' ' Masters' ' 9th' ' Some-college' ' Assoc-acdm' ' 7th-8th' ' Doctorate' ' Assoc-voc' ' Prof-school' ' 5th-6th' ' 10th' ' Preschool' ' 12th' ' 1st-4th']\n" +
                  "- marital-status: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- occupation: [' Adm-clerical' ' Exec-managerial' ' Handlers-cleaners' ' Prof-specialty' ' Other-service' ' Sales' ' Transport-moving' ' Farming-fishing' ' Machine-op-inspct' ' Tech-support' ' Craft-repair' ' Protective-serv' ' Armed-Forces' ' Priv-house-serv']\n" +
                  "- race: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- sex: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- hours-per-week: 1 - 99\n\n"
)

COLUMN_VALUES_RULES_PROMPT_GERMAN_CREDIT = """
MUTABLE FEATURES (allowed values)
- account_check_status: ['no checking account', '< 0 DM', '0 <= ... < 200 DM', '>= 200 DM / salary assignments for at least 1 year']
- duration_in_month: integer, 4 - 72
- credit_amount: integer, 250 - 18424
- savings: ['unknown/ no savings account', '... < 100 DM', '100 <= ... < 500 DM', '500 <= ... < 1000 DM', '... >= 1000 DM']
- present_emp_since: ['unemployed', '... < 1 year', '1 <= ... < 4 years', '4 <= ... < 7 years', '... >= 7 years']
- installment_as_income_perc: 1 - 4
- other_debtors: ['co-applicant', 'none', 'guarantor']
- property: ['real estate' 'building society savings/life insurance' 'unknown/no property' 'car or other']
- age: integer, 19 - 75
- other_installment_plans: ['none', 'bank', 'stores']
- credits_this_bank: 1 - 4
- job: ['skilled employee/official' 'unskilled resident' 'management/self-employed/highly qualified' 'unemployed/unskilled non-resident']
- telephone: ['yes' 'none']

IMMUTABLE FEATURES (never include in any rule)
credit_history, purpose, personal_status_sex, present_res_since, housing, people_under_maintenance, foreign_worker
"""

COLUMN_VALUES_LOCAL_PROMPT_GERMAN_CREDIT = COLUMN_VALUES_RULES_PROMPT_GERMAN_CREDIT

COLUMN_VALUES_RULES_PROMPT_LENDING = (
                  "Column names and allowed values:\n" +
                  "- emp_length: 1 - 10\n" +
                  "- annual_inc: 12252 - 195000\n" +
                  "- open_acc: 2 - 34\n" +
                  "- credit_years: 3 - 43\n" +
                  "- grade: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
                  "- home_ownership: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
                  "- purpose: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
                  "- addr_state: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n\n"
)

COLUMN_VALUES_RULES_PROMPT_COMPAS = (
                  "Column names and allowed values:\n" +
                  "- sex: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
                  "- age: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
                  "- race: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
                  "- priors_count: 0 - 36\n" +
                  "- days_b_screening_arrest: 0 - 812\n" +
                  "- length_of_stay: 0 - 530\n" +
                  "- c_charge_degree: ['F' 'M']\n" +
                  "- is_violent_recid: 0 - 1\n" +
                  "- score_text: ['Low' 'Medium' 'High']\n\n"
)

COLUMN_VALUES_RULES_PROMPT_ADULT = (
                  "Column names and allowed values:\n" +
                  "- age: 17 - 90\n" +
                  "- workclass: [' State-gov' ' Self-emp-not-inc' ' Private' ' Federal-gov' ' Local-gov' ' Self-emp-inc']\n" +
                  "- education: [' Bachelors' ' HS-grad' ' 11th' ' Masters' ' 9th' ' Some-college' ' Assoc-acdm' ' 7th-8th' ' Doctorate' ' Assoc-voc' ' Prof-school' ' 5th-6th' ' 10th' ' Preschool' ' 12th' ' 1st-4th']\n" +
                  "- marital-status: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
                  "- occupation: [' Adm-clerical' ' Exec-managerial' ' Handlers-cleaners' ' Prof-specialty' ' Other-service' ' Sales' ' Transport-moving' ' Farming-fishing' ' Machine-op-inspct' ' Tech-support' ' Craft-repair' ' Protective-serv' ' Armed-Forces' ' Priv-house-serv']\n" +
                  "- race: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
                  "- sex: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
                  "- hours-per-week: 1 - 99\n\n"
)

RULES_PROMPT = """
HARD CONSTRAINTS
- Each rule changes exactly 2-3 features.
- Only the MUTABLE features below may appear in a rule.
- NEVER include an immutable feature in any rule.
- Categorical features: use only the listed allowed values, verbatim.
- Numeric features: integers within the stated [min, max] range.
- The "reasoning" field is a single sentence; do not restate the feature
  values as a list within it.
- Do NOT default to a feature's maximum or best value: a rule that
  moves each of its features by one or two levels is preferred over a rule
  that pins any feature to its extremes.
"""

IMMUTABLES_PROMPT_GERMAN_CREDIT = (
    "Immutable features that must never be changed:\n" +
    "- credit_history\n" +
    "- personal_status_sex\n" +
    "- residence_duration\n" +
    "- age\n" +
    "- people_liable\n"
    "- foreign_worker\n\n"
)

GERMAN_CREDIT_CONTEXT = ("in the context of loan approval (German Credit dataset), you'll see a record containing information "
                         "about a person for which a loan has been rejected, proceed step by step: first think about which "
                         "features you can modify in order to get the loan accepted, then modify those features in the appropriate "
                         "direction in order to flip the prediction, your main goal is flipping the prediction so you have to "
                         "modify many features and by a big magnitude but remember that you can NEVER modify features marked as IMMUTABLE.")

LENDING_CONTEXT = ("in the context of loan repayment (Lending Club dataset), you'll see a record containing information about a "
                   "person that has not returned a loan, proceed step by step: first think about which features you can modify "
                   "in order to get the loan returned, then modify those features in the appropriate direction in order to flip "
                   "the prediction, your main goal is flipping the prediction so you have to modify many features and by a big "
                   "magnitude but remember that you can NEVER modify immutable features marked as IMMUTABLE.")

COMPAS_CONTEXT = ("in the context of criminal likelihood of recidivism / reoffending (Compas dataset), you'll see a record containing information "
                  "about a person who has not reoffended, proceed step by step: first think about which features you can modify in "
                  "order for the person to reoffend, then modify those features in the appropriate direction in order to flip the "
                  "prediction, your main goal is flipping the prediction so you have to modify many features and by a big magnitude"
                  " but remember that you can NEVER modify features marked as IMMUTABLE.")

ADULT_CONTEXT = ("in the context of annual income prediction (Adult Income dataset), you'll see a record containing information about a person whose annual "
                 "income does not exceed a certain threshold, proceed step by step: first think about which features you can modify "
                 "in order to increase the income, then modify those features in the appropriate "
                 "direction in order to flip the prediction, your main goal is flipping the prediction so you have to modify "
                 "many features and by a big magnitude but remember that you can NEVER modify features marked as IMMUTABLE.")

GERMAN_CREDIT_RULES_PROMPT = ("""
You are the generator itself. You produce global counterfactual explanation
rules for the German Credit dataset. Every record is a loan applicant whose
application was REJECTED. A rule is a small set of feature changes that,
applied to rejected applicants, is intended to flip the model's prediction
from REJECTED to APPROVED.

You do NOT write, describe, or output code of any kind (no Python, no
pseudocode, no scripts). You do NOT train or invoke any model. You directly
produce the JSON rules described below, reasoning only from the feature
semantics given in this prompt. Any code in your answer is a failure.

OBJECTIVE
Produce rules that flip the prediction with the smallest, most plausible
changes: change as few features as possible (sparsity) and by the smallest
magnitude that still flips the outcome (proximity). Across the full set of
rules, prefer DIVERSE feature combinations so the rules cover different
subgroups of rejected applicants.
""" + 
"""
REASONING
Reason internally about which mutable features to change and in which
direction increases the chance of approval (move features toward values
associated with higher creditworthiness). Do NOT output this internal
deliberation. The only explanatory text in your answer is the per-rule
"reasoning" field defined in OUTPUT FORMAT — one concise sentence per rule,
inside the JSON. Emit no prose, commentary, or markdown outside the JSON
array.
"""
+ RULES_PROMPT + COLUMN_VALUES_RULES_PROMPT_GERMAN_CREDIT + 
"""
OUTPUT FORMAT
Return a single JSON array containing 5-10 objects. Each object has exactly
two fields:
- "reasoning": one concise sentence explaining why these changes move the
  applicant toward approval.
- "changes": an object of 2-3 "feature": value pairs drawn only from the
  mutable features above.

Structure to follow (placeholders, not real values — replace each <...> with
an actual mutable feature name and a valid value for it; categorical values
must be quoted strings, numeric values must be bare integers):
[
  {
    "reasoning": "<one concise sentence>",
    "changes": {"<feature>": "<categorical_value>", "<feature>": <integer_value>}
  },
  {
    "reasoning": "<one concise sentence>",
    "changes": {"<feature>": "<categorical_value>", "<feature>": <integer_value>, "<feature>": "<categorical_value>"}
  }
]

Output only the JSON array. No code, no prose."""
)

LENDING_RULES_PROMPT = (
        "You are a system that generates global counterfactual explanations for tabular records in the context of loan repayment "
        "(Lending Club dataset), the dataset contains information about people that have not returned a loan, proceed step by step:"
        " first think about which features you can modify in order to get the loan returned, then you have to generate rules that"
        " modify those features in the appropriate direction in order to flip the prediction, your main goal is flipping the"
        " prediction so you have to modify many features and by a big magnitude but remember that you can NEVER modify features"
        " marked as IMMUTABLE.\n\n" +
        RULES_PROMPT + COLUMN_VALUES_RULES_PROMPT_LENDING +
        "You have to generate rules that make minimal changes so only include a FEW attributes, each rule enclosed in {} is made "
        "up of only 2-3 pairs column: value.\nReturn each rule ONLY in a valid JSON object enclosed in {} with NO extra text, "
        "return directly each minimal rule enclosed in {} in valid JSON format, do NOT include anything else but the rules in the answer."
)

COMPAS_RULES_PROMPT = (
        "You are a system that generates global counterfactual explanations for tabular records in the context of criminal "
        "likelihood of recidivism / reoffending (Compas dataset), the dataset contains information about people who have not "
        "reoffended, proceed step by step: first think about which features you can modify in order for the person to reoffend, "
        "then you have to generate rules that modify those features in the appropriate direction in order to flip the prediction, "
        "your main goal is flipping the prediction so you have to modify many features and by a big magnitude but remember that "
        "you can NEVER modify features marked as IMMUTABLE.\n\n" +
        RULES_PROMPT + COLUMN_VALUES_RULES_PROMPT_COMPAS +
        "You have to generate rules that make minimal changes so only include a FEW attributes, each rule enclosed in {} is made "
        "up of only 2-3 pairs column: value.\nReturn each rule ONLY in a valid JSON object enclosed in {} with NO extra text, "
        "return directly each minimal rule enclosed in {} in valid JSON format, do NOT include anything else but the rules in the answer."
)

ADULT_RULES_PROMPT = (
        "You are a system that generates global counterfactual explanations for tabular records in the context of annual income "
        "prediction (Adult Income dataset), the dataset contains information about people whose income does not exceed a certain "
        "threshold, proceed step by step: first think about which features you can modify in order to increase the income, then "
        "modify those features in the appropriate direction in order to flip the prediction, your main goal is flipping the "
        "prediction so you have to modify many features and a big magnitude but remember that you can NEVER modify features "
        "marked as IMMUTABLE.\n\n" +
        RULES_PROMPT + COLUMN_VALUES_RULES_PROMPT_ADULT +
        "You have to generate rules that make minimal changes so only include a FEW attributes, each rule enclosed in {} is made "
        "up of only 2-3 pairs column: value.\nReturn each rule ONLY in a valid JSON object enclosed in {} with NO extra text, "
        "return directly each minimal rule enclosed in {} in valid JSON format, do NOT include anything else but the rules in the "
        "answer, each rule consists only of a FEW attributes."
)