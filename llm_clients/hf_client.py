# import logging
# import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from llm_clients.llm_client import LLMClient

# logger = logging.getLogger(__name__)


# class HuggingFaceClient(LLMClient):
#     """
#     Generic client for Hugging Face causal language models. Works with any
#     instruction-tuned model on the Hub that:
#       - Is a causal LM compatible with AutoModelForCausalLM
#       - Has a chat template registered in its tokenizer (i.e. supports
#         tokenizer.apply_chat_template)
#
#     By default the model is loaded with 4-bit NF4 quantization for memory
#     efficiency. Pass quantization_config=None to load in full precision,
#     or provide your own BitsAndBytesConfig for other schemes.
#     """
#
#     DEFAULT_QUANTIZATION = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     )
#
#     def __init__(
#         self,
#         model_name: str,
#         token: str | None = None,
#         seed: int | None = None,
#         device: str = "cuda",
#         quantization_config: BitsAndBytesConfig | None = "default",
#         torch_dtype: torch.dtype = torch.bfloat16,
#         top_p: float = 0.95,
#     ):
#         """
#         Parameters
#         ----------
#         model_name : str
#             HF Hub model identifier, e.g. 'meta-llama/Llama-3.1-8B-Instruct',
#             'mistralai/Mistral-7B-Instruct-v0.3', 'Qwen/Qwen2.5-7B-Instruct'.
#         token : str or None
#             HF access token, needed for gated models.
#         seed : int or None
#             Seed for reproducible generation (TODO).
#         device : str
#             'cuda' uses device_map='auto'; 'cpu' forces CPU (quantization
#             config is ignored in that case).
#         quantization_config : BitsAndBytesConfig, None, or 'default'
#             'default' uses 4-bit NF4 quantization. None disables quantization
#             and loads the model in torch_dtype. Pass a custom config for
#             other schemes (8-bit, fp4, etc.).
#         torch_dtype : torch.dtype
#             Compute dtype for non-quantized weights / activations.
#         top_p : float
#             Nucleus sampling cutoff used when temperature > 0.
#         """
#         super().__init__(model_name, token=token, seed=seed, device=device)
#
#         if quantization_config == "default":
#             quantization_config = self.DEFAULT_QUANTIZATION
#         self.top_p = top_p
#
#         logger.info("Loading %s...", model_name)
#         t0 = time.time()
#
#         load_kwargs = {
#             "device_map": "auto" if device == "cuda" else device,
#             "torch_dtype": torch_dtype,
#             "token": token,
#         }
#         if quantization_config is not None and device == "cuda":
#             load_kwargs["quantization_config"] = quantization_config
#         elif quantization_config is not None:
#             logger.warning(
#                 "quantization_config given but device=%s; ignoring "
#                 "(bitsandbytes requires CUDA).",
#                 device,
#             )
#
#         self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
#
#         # Some models ship without a pad token; fall back to EOS to keep
#         # generate() happy.
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#
#         # Fail loudly if the tokenizer has no chat template — this class
#         # only supports instruction-tuned models.
#         if self.tokenizer.chat_template is None:
#             raise ValueError(
#                 f"Tokenizer for {model_name!r} has no chat_template. "
#                 "HuggingFaceClient only supports instruction-tuned models "
#                 "with a registered chat template."
#             )
#
#         logger.info("Model loaded in %.1fs on %s", time.time() - t0, self.model.device)
#
#     def ask(
#         self,
#         prompts: list[dict[str, str]],
#         max_response_length: int = 512,
#         temperature: float = 0.0,
#     ) -> tuple[str, dict]:
#         prompt = self.tokenizer.apply_chat_template(
#             prompts, tokenize=False, add_generation_prompt=True
#         )
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
#         prompt_tokens = inputs.input_ids.shape[1]
#
#         # TODO: honor self.seed for reproducible generation.
#
#         do_sample = temperature > 0
#         gen_kwargs = {
#             "max_new_tokens": max_response_length,
#             "do_sample": do_sample,
#             "pad_token_id": self.tokenizer.pad_token_id,
#         }
#         if do_sample:
#             gen_kwargs["temperature"] = temperature
#             gen_kwargs["top_p"] = self.top_p
#
#         t0 = time.time()
#         with torch.no_grad():
#             outputs = self.model.generate(**inputs, **gen_kwargs)
#         elapsed = time.time() - t0
#
#         generated_ids = outputs[0][prompt_tokens:]
#         completion_tokens = len(generated_ids)
#         message = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
#
#         tokens = {
#             "input": prompt_tokens,
#             "output": completion_tokens,
#             "total": prompt_tokens + completion_tokens,
#         }
#
#         return message, tokens


class HuggingFaceClient(LLMClient):

    def __init__(
            self,
            model_name: str,
            token: str | None = None,
            seed: int | None = None,
            device: str = "cuda",
            local_files_only: bool = False,
            torch_dtype: torch.dtype = "auto",
            top_p: float = 0.95,
    ):
        """
        Parameters
        ----------
        model_name : str
            HF Hub model identifier (e.g., 'meta-llama/Llama-3.1-8B-Instruct').
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
        """
        super().__init__(model_name, token=token, seed=seed, device=device)

        self.top_p = top_p

        # Safely map devices across available GPUs
        load_kwargs = {
            "device_map": "auto" if device == "cuda" else device,
            "torch_dtype": torch_dtype,
            "token": token,
            "local_files_only": local_files_only
        }

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
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

        # Apply the model's native chat template
        prompt_text = self.tokenizer.apply_chat_template(
            prompts, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs.input_ids.shape[1]

        # Smartly infer sampling needs based on temperature
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_response_length,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = self.top_p

        # torch.no_grad() is essential here to prevent VRAM bloat during inference
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Slice off the prompt to extract only the generated tokens
        generated_ids = outputs[0][prompt_tokens:]
        completion_tokens = len(generated_ids)
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        tokens = {
            "input": prompt_tokens,
            "output": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        }

        return response_text, tokens
