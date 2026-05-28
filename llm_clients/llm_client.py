from abc import ABC, abstractmethod


class LLMClient(ABC):
    """
    Abstract client for interacting with a language model. Concrete subclasses
    wrap different backends (OpenAI API, local HuggingFace model, etc.) behind
    a common interface.

    Subclasses must call super().__init__(...) to populate the shared
    attributes (model_name, token, seed, device) before doing their own setup.
    """

    @abstractmethod
    def __init__(
        self,
        model_name: str,
        token: str | None = None,
        seed: int | None = None,
        device: str = "cuda",
    ):
        """
        Parameters
        ----------
        model_name : str
            Backend-specific model identifier (e.g. 'gpt-4o-mini',
            'meta-llama/Llama-3.1-8B-Instruct').
        token : str or None
            Authentication token. For OpenAI this is the API key; for
            HuggingFace this is the HF access token needed for gated models.
        seed : int or None
            Seed for reproducible generation. If None, sampling uses the
            backend's default (typically nondeterministic).
        device : str
            Device for local backends ('cuda', 'cpu'). Ignored by API backends.
        """
        self.model_name = model_name
        self.token = token  # FIXME: risky
        self.seed = seed
        self.device = device

    @abstractmethod
    def ask(
        self,
        prompts: list[dict[str, str]],
        max_response_length: int = 512,
        temperature: float = 0.0,
    ) -> tuple[str, dict]:
        """
        Send a conversation to the model and return the response.

        Parameters
        ----------
        prompts : list of {'role': str, 'content': str}
            The conversation as a list of chat messages.
        max_response_length : int
            Maximum number of tokens in the generated response.
        temperature : float
            Sampling temperature. 0.0 means deterministic (greedy) decoding;
            values > 0 enable stochastic sampling with the given temperature.

        Returns
        -------
        message : str
            The assistant's response text.
        tokens : dict
            Token usage with keys 'input', 'output', 'total'.
        """
        pass
