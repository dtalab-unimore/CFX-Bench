import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_clients.llm_client import LLMClient


class HuggingFaceClient(LLMClient):

    def __init__(
            self,
            model_name: str,
            token: str | None = None,
            seed: int | None = None,
            device: str = "cuda",
            local_files_only: bool = False,
            quantization_config = None,
            torch_dtype: torch.dtype = "auto",
            top_p: float = 0.95,
            local_model_dir: str | None = None,
    ):
        """
        Parameters
        ----------
        model_name : str
            HF Hub model identifier (e.g., 'meta-llama/Llama-3.1-8B-Instruct').
            Still used for metadata/logging even when loading from disk.
        token : str or None
            HF access token, needed for gated models.
        seed : int or None
            Seed for reproducible generation.
        device : str
            'cuda' uses device_map='auto'; 'cpu' forces CPU.
        local_files_only : bool
            If True, avoids downloading models and relies on local cache.
        torch_dtype : torch.dtype
            Compute dtype for weights / activations.
        top_p : float
            Nucleus sampling cutoff used when temperature > 0.
        local_model_dir : str or None
            If set, load the model and tokenizer from this local directory
            instead of from the Hub.
        """
        super().__init__(model_name, token=token, seed=seed, device=device)

        self.top_p = top_p

        # A local directory is passed straight to from_pretrained, which then
        # bypasses the Hub entirely.
        load_target = local_model_dir if local_model_dir is not None else model_name

        load_kwargs = {
            "device_map": "auto" if device == "cuda" else device,
            "torch_dtype": torch_dtype,
            "token": token,
            "local_files_only": local_files_only,
            # "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(load_target, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_target,
            token=token,
            local_files_only=local_files_only
        )

        # Fallback to EOS token if Pad token is missing to ensure generate() works
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Fail-fast validation for instruction-tuned models
        if self.tokenizer.chat_template is None:
            raise ValueError(
                f"Tokenizer for {model_name!r} has no chat_template. "
                "This client requires instruction-tuned models with a registered chat template."
            )

    def ask(
            self,
            prompts: list[dict[str, str]],
            max_response_length: int = 512,
            temperature: float = 0.0,
    ) -> tuple[str, dict]:

        prompt_text = self.tokenizer.apply_chat_template(
            prompts, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs.input_ids.shape[1]

        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_response_length,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0][prompt_tokens:]
        completion_tokens = len(generated_ids)
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        tokens = {
            "input": prompt_tokens,
            "output": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        }

        return response_text, tokens
