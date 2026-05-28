from openai import OpenAI

from llm_clients.llm_client import LLMClient


class GPTClient(LLMClient):
    """OpenAI GPT client."""

    def __init__(
        self,
        model_name: str,
        token: str | None = None,
        seed: int | None = None,
        device: str = "cuda",  # unused for API clients
    ):
        super().__init__(model_name, token=token, seed=seed, device=device)
        self.client = OpenAI(api_key=token)

    def ask(
        self,
        prompts: list[dict[str, str]],
        max_response_length: int = 512,
        temperature: float = 0.0,
    ) -> tuple[str, dict]:
        kwargs = {}
        if self.model_name in ['gpt-5-mini']:
            kwargs['max_completion_tokens'] = max_response_length
            ...
        elif self.model_name in ['gpt-4o-mini']:
            kwargs['max_tokens'] = max_response_length
            kwargs['temperature'] = temperature
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompts,
            **kwargs,
            seed=self.seed,
        )

        message = resp.choices[0].message.content
        usage = resp.usage
        tokens = {
            "input": usage.prompt_tokens if usage else None,
            "output": usage.completion_tokens if usage else None,
            "total": usage.total_tokens if usage else None,
        }
        return message, tokens
