class PromptManager:
    """
    Generate a prompt to be given as input to the LLM
    """

    def __init__(self) -> None:
        pass

    def get_system_prompt(self) -> dict[str, str]:
        """
        :return: system prompt for the LLM
        """
        return {
            "role": "system",
            "content": "",
        }

    def generate(self, record: str) -> list[dict[str, str]]:
        """
        Generate a prompt that includes both the system and user sub-prompts.
        :param record: the user-provided data to be used in the user sub-prompt
        :return: prompt for the LLM
        """
        prompts = [self.get_system_prompt()]
        prompts += [
            {
                "role": "user",
                "content": f"{record}"
            }
        ]
        return prompts

    def parse_response(self, response: str) -> dict:
        pass
