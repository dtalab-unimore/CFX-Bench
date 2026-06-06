from llm_prompts.prompt_manager import PromptManager
import json


class CFVerbalizePromptManager(PromptManager):
    """
    Prompt for verbalizing a counterfactual explanation.
    """

    def get_system_prompt(self) -> dict[str, str]:
        cf_template = {
            "record": "<string describing the original instance>",
            # "label": "<original label>",
            "pred": "<model prediction on the original instance>",
            "proba": "<probability of the predicted class>",
            # "target": "<desired target label>",
            "cf": {
                "record": "<string or dictionary describing the counterfactual instance>",
                "changes": "<list or description of the features that need to change>",
                "proba": "<probability of the desired class for the counterfactual>",
                "pred": "<predicted class for the counterfactual>"
            }
        }

        # cf_example = {
        #     'input': {
        #         "record": "savings.account.and.bonds=unknown/ no savings account, age.in.years=35, "
        #                   "property=unknown / no property, present.employment.since=1 <= ... < 4 years, "
        #                   "housing=for free, other.installment.plans=none, foreign.worker=yes, purpose=education",
        #         "pred": 1,
        #         "proba": "0.500",
        #         "cf": {
        #             "record": "savings.account.and.bonds=['... >= 1000 DM'], age.in.years=[32.00, 45.00), "
        #                       "property=['unknown / no property'], present.employment.since=['1 <= ... < 4 years'], "
        #                       "housing=['for free'], other.installment.plans=['bank', 'stores'], "
        #                       "foreign.worker=['yes'], "
        #                       "purpose=['others', 'OTHER', 'education', 'repairs', 'retraining', 'business']",
        #             "changes": "savings.account.and.bonds=['... >= 1000 DM'], "
        #                        "other.installment.plans=['bank', 'stores']",
        #             "proba": "0.382",
        #             "pred": 0
        #         }
        #     },
        #     'output': "Imagine you are in this scenario: you are 35 years old, you currently have no savings, "
        #               "you do not own any property, you have been employed for 1 to 4 years, you live without paying "
        #               "rent, you have no other installment plans, you are a foreign worker, and your loan request is "
        #               "for education purposes. Current outcome: your loan application was rejected. To have your loan "
        #               "approved, you would need to make the following changes: you would need to increase your "
        #               "savings to over 1000 DM, and you would need to establish installment plans with a bank or a "
        #               "store. The rest of the values will remain constant."
        # }
        cf_example = {
            'input': {
                "record": {
                    "credit.amount": 9055,
                    "savings.account.and.bonds": "unknown\/ no savings account",
                    "age.in.years": 35,
                    "property": "unknown \/ no property",
                    "present.employment.since": "1 <= ... < 4 years",
                    "housing": "for free",
                    "other.installment.plans": "none",
                    "other.debtors.or.guarantors": "none",
                    "foreign.worker": "yes",
                    "installment.rate": 2,
                    "duration.in.month": 36,
                    "purpose": "education"
                },
                "pred": 0,
                "proba": "0.4187",
                "cf": {
                    "record": {
                        "credit.amount": "[6000.00, 8000.00)",
                        "savings.account.and.bonds": "['... < 100 DM', 'unknown\/ no savings account']",
                        "age.in.years": "[32.00, 45.00)",
                        "property": "['unknown \/ no property']",
                        "present.employment.since": "['1 <= ... < 4 years']",
                        "housing": "['for free']",
                        "other.installment.plans": "['none']",
                        "other.debtors.or.guarantors": "['co-applicant', 'none']",
                        "foreign.worker": "['yes']",
                        "installment.rate": "(-inf, 3.00)",
                        "duration.in.month": "[36.00, 48.00)",
                        "purpose": "['others', 'OTHER', 'education', 'repairs', 'retraining', 'business']"
                    },
                    "changes": {
                        "credit.amount": "[6000.00, 8000.00)",
                    },
                    "proba": "0.5005",
                    "pred": 1
                }
            },
            'output': "Imagine you are in this scenario: you are 35 years old, you are requesting a loan of 9055 for a duration of 36 months and for educational purposes. You have no savings account, you do not own any property, and you have been employed for 1 to 4 years. You live rent-free and you have no other installment plans. You also have no co-debtors or guarantors, and your installment rate is 2/4 (relative to your income). Additionally, you are a foreign worker.  Current outcome: your loan application was rejected. To have your loan approved, you would need to make the following changes: you would need to reduce your loan amount in the range between 6000 and 8000. The rest of your situation would remain the same."
        }

        prompt = (
            f"You are given a counterfactual explanation in the following JSON format:\n"
            f"{json.dumps(cf_template, indent=4)}\n\n"
            "Your task is to generate a natural, human-readable explanation that describes:\n"
            "1. The original situation (interpreting the \"record\" field).\n"
            "2. The original outcome (pred=0 -> loan rejected, pred=1 -> load approved).\n"
            "3. The minimal changes required to obtain the desired outcome (using the \"cf\" and especially the "
            "\"changes\" field).\n"
            "4. A clear, concise counterfactual narrative explaining what would need to be different to achieve the "
            "target outcome.\n"
            "Do not repeat the JSON structure.\n"
            "Do not invent additional information.\n"
            "Transform the structured information into a coherent and fluent paragraph or short explanation, "
            "written in natural language, as if addressing a non-technical person.\n\n"
            "Output format:\n"
            "The output must begin with: \"Answer: \"\n"
            "The explanation must include the phrases:\n"
            "-\"Imagine you are in this scenario: \"\n"
            "-\"Current outcome: \"\n"
            "-\"..., you would need to make the following changes: \"\n\n"
            "Below is an example of expected output style for a given counterfactual explanation:\n"
            f"Counterfactual explanation to verbalize: {json.dumps(cf_example['input'], indent=4)}\n"
            f"Answer: {cf_example['output']}"
        )

        return {
            "role": "system",
            "content": prompt
        }

    def get_user_prompt_content(self, cf: str) -> str:
        user_prompt = f"Counterfactual explanation to verbalize: {cf}"
        return user_prompt

    def parse_response(self, response: str) -> str:
        """
        Parse the LLM's response into a structured format.
        """
        response = response.split("Answer: ")[1].strip()
        assert "Imagine you are in this scenario:" in response
        assert "Current outcome:" in response
        assert "you would need to make the following changes:" in response
        return response
