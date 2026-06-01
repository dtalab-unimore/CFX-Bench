
COLUMN_VALUES_LOCAL_PROMPT_GERMAN_CREDIT = (
                  "Column names and allowed values:\n" +
                  "- checking_account: ['< 0 DM' '0-200 DM' 'no checking account' '>= 200 DM']\n" +
                  "- duration: 4 - 72\n" +
                  "- credit_history: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- purpose: ['radio/television' 'education' 'furniture/equipment' 'car (new)' 'car (used)' 'business' 'domestic appliances' 'repairs' 'others' 'retraining']\n" +
                  "- credit_amount: 250 - 18424\n" +
                  "- savings_account: ['unknown/no savings account' '< 100 DM' '500-1000 DM' '>= 1000 DM' '100-500 DM']\n" +
                  "- employment: ['>= 7 years' '1-4 years' '4-7 years' 'unemployed' '< 1 year']\n" +
                  "- installment_rate: 1 - 4\n" +
                  "- personal_status_sex: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- guarantors: ['none' 'guarantor' 'co-applicant']\n" +
                  "- residence_duration: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- property: ['real estate' 'building society savings/life insurance' 'unknown/no property' 'car or other']\n" +
                  "- age: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- other_installment_plans: ['none' 'bank' 'stores']\n" + "- housing: ['own' 'for free' 'rent']\n" +
                  "- existing_credits: 1 - 4\n" +
                  "- job: ['skilled employee/official' 'unskilled resident' 'management/self-employed/highly qualified' 'unemployed/unskilled non-resident']\n" +
                  "- people_liable: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n" +
                  "- telephone: ['yes' 'none']\n"
                  "- foreign_worker: IMMUTABLE FEATURE, USE THE EXISTING VALUE\n\n"
)

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

COLUMN_VALUES_RULES_PROMPT_GERMAN_CREDIT = (
                  "Column names and allowed values:\n" +
                  "- checking_account: ['< 0 DM' '0-200 DM' 'no checking account' '>= 200 DM']\n" +
                  "- duration: 4 - 72\n" +
                  "- credit_history: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
                  "- purpose: ['radio/television' 'education' 'furniture/equipment' 'car (new)' 'car (used)' 'business' 'domestic appliances' 'repairs' 'others' 'retraining']\n" +
                  "- credit_amount: 250 - 18424\n" +
                  "- savings_account: ['unknown/no savings account' '< 100 DM' '500-1000 DM' '>= 1000 DM' '100-500 DM']\n" +
                  "- employment: ['>= 7 years' '1-4 years' '4-7 years' 'unemployed' '< 1 year']\n" +
                  "- installment_rate: 1 - 4\n" +
                  "- personal_status_sex: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
                  "- guarantors: ['none' 'guarantor' 'co-applicant']\n" +
                  "- residence_duration: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
                  "- property: ['real estate' 'building society savings/life insurance' 'unknown/no property' 'car or other']\n" +
                  "- other_installment_plans: ['none' 'bank' 'stores']\n" + "- housing: ['own' 'for free' 'rent']\n" +
                  "- existing_credits: 1 - 4\n" +
                  "- job: ['skilled employee/official' 'unskilled resident' 'management/self-employed/highly qualified' 'unemployed/unskilled non-resident']\n" +
                  "- people_liable: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
                  "- telephone: ['yes' 'none']\n"
                  "- foreign_worker: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n\n"
)

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

RULES_PROMPT = (
    "Rules:\n" +
    "- Return ONLY valid JSON.\n" + "- Do NOT include ANY extra text.\n" +
    "- A rule consists of FEW attributes / columns, do NOT use all of them.\n" +
    "- Keys must match column names, but DON'T include all of them.\n" +
    "- Values must respect column types.\n" + "- Use only allowed values for features.\n" +
    "- Do NOT change IMMUTABLE features, do NOT include them in the rules.\n\n"
)

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

GERMAN_CREDIT_RULES_PROMPT = ("You are a system that generates global counterfactual explanations for tabular records in the "
                              "context of loan approval (German Credit dataset), the dataset contains information about people that "
                              "have had loans rejected, proceed step by step: first think about which features you can modify in "
                              "order to get the loan accepted, then you have to generate rules that modify those features in the "
                              "appropriate direction in order to flip the prediction, your main goal is flipping the prediction "
                              "so you have to modify many features and by a big magnitude but remember that you can NEVER modify "
                              "features marked as IMMUTABLE.\n\n"
                              + RULES_PROMPT + COLUMN_VALUES_RULES_PROMPT_GERMAN_CREDIT +
                              "You have to generate rules that make minimal changes so only include a FEW attributes, each rule "
                              "enclosed in {} is made up of only 2-3 pairs column: value.\nReturn each rule ONLY in a valid "
                              "JSON object enclosed in {} with NO extra text, return directly each minimal rule enclosed in {} "
                              "in valid JSON format, do NOT include anything else but the rules in the answer."
)

GERMAN_CREDIT_CRIF_RULES_PROMPT = GERMAN_CREDIT_RULES_PROMPT.replace(
    COLUMN_VALUES_RULES_PROMPT_GERMAN_CREDIT,
    "Column names and allowed values:\n" +
    "- savings.account.and.bonds: ['unknown/no savings account' '... < 100 DM' '100 <= ... < 500 DM' '500 <= ... < 1000 DM' '... >= 1000 DM']\n" +
    # "- savings.account.and.bonds: ['... < 100 DM&&unknown/ no savings account', '100 <= ... < 500 DM&&500 <= ... < 1000 DM', '... >= 1000 DM']\n"
    "- age.in.years: 19 - 75\n"
    # "- age.in.years: [(-inf, 27.00), [27.00, 32.00), [32.00, 45.00), [45.00, inf)]\n"
    "- property: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
    "- present.employment.since: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
    "- housing: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n"
    "- other.installment.plans: ['none' 'bank' 'stores']\n" +
    # "- other.installment.plans: ['bank&&stores', 'none']\n" +
    "- foreign.worker: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
    "- purpose: IMMUTABLE FEATURE, DON'T INCLUDE IN THE RULES\n" +
    "\n"
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