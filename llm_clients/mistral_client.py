import os
import torch
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration

from llm_clients.llm_client import LLMClient


class MistralClient(LLMClient):
    """
    Text-only client for Mistral models whose official checkpoints ship in
    Mistral format (e.g. Mistral-Small-3.2-24B-Instruct-2506). Tokenization
    goes through `mistral-common`; vision inputs are intentionally not wired.
    """

    def __init__(
            self,
            model_name: str,
            token: str | None = None,
            seed: int | None = None,
            device: str = "cuda",
            local_files_only: bool = False,
            torch_dtype: torch.dtype = "auto",
            top_p: float = 0.95,
            local_model_dir: str | None = None,
    ):
        """
        Parameters
        ----------
        model_name : str
            HF Hub model identifier (e.g. 'mistralai/Mistral-Small-3.2-24B-Instruct-2506').
            Still used for metadata/logging even when loading from disk.
        token : str or None
            HF access token, needed for gated models.
        seed : int or None
            Seed for reproducible generation.
        device : str
            'cuda' uses device_map='auto'; 'cpu' forces CPU.
        local_files_only : bool
            If True, avoids downloading and relies on the HF cache.
        torch_dtype : torch.dtype
            Compute dtype for weights / activations.
        top_p : float
            Nucleus sampling cutoff used when temperature > 0.
        local_model_dir : str or None
            If set, load the model weights and the Mistral tokenizer file
            from this local directory instead of from the Hub.
        """
        super().__init__(model_name, token=token, seed=seed, device=device)

        self.top_p = top_p

        load_target = local_model_dir if local_model_dir is not None else model_name
        print({
            'model_name': model_name,
            'local_model_dir': local_model_dir,
            'load_target': load_target,
        })

        load_kwargs = {
            "device_map": "auto" if device == "cuda" else device,
            "torch_dtype": torch_dtype,
            "token": token,
            "local_files_only": local_files_only,
        }

        # Mistral 3 is a multimodal architecture; for text-only we just skip
        # pixel_values at generate-time.
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            load_target, **load_kwargs
        )

        # mistral-common's tokenizer is the source of truth for chat formatting.
        # `from_hf_hub` honours `local_files_only` via the standard HF cache.
        if local_model_dir is not None:
            self.tokenizer = MistralTokenizer.from_file(os.path.join(local_model_dir, 'tekken.json'))
        else:
            self.tokenizer = MistralTokenizer.from_hf_hub(model_name, token=token)

        # Mistral tokenizers don't expose a dedicated pad token; reuse EOS so
        # generate() has something to feed into pad_token_id.
        inner = self.tokenizer.instruct_tokenizer.tokenizer
        self.eos_token_id = inner.eos_id
        self.pad_token_id = self.eos_token_id

    def ask(
            self,
            prompts: list[dict[str, str]],
            max_response_length: int = 512,
            temperature: float = 0.0,
    ) -> tuple[str, dict]:

        tokenized = self.tokenizer.encode_chat_completion(
            ChatCompletionRequest(messages=prompts)
        )

        input_ids = torch.tensor([tokenized.tokens], device=self.model.device)
        attention_mask = torch.ones_like(input_ids)
        prompt_tokens = input_ids.shape[1]

        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_response_length,
            "do_sample": do_sample,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        generated_ids = outputs[0][prompt_tokens:].tolist()
        completion_tokens = len(generated_ids)

        # Drop a trailing EOS so it doesn't leak into the returned string.
        if completion_tokens > 0 and generated_ids[-1] == self.eos_token_id:
            generated_ids = generated_ids[:-1]

        response_text = self.tokenizer.decode(generated_ids)

        tokens = {
            "input": prompt_tokens,
            "output": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        }

        return response_text, tokens
