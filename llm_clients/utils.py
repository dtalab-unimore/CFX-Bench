import os

import dotenv

from llm_clients.gpt_client import GPTClient
from llm_clients.hf_client import HuggingFaceClient
from llm_clients.mistral_client import MistralClient


def get_model_source(model_name):
    if 'gpt' in model_name.lower():
        return "gpt"
    if 'mistral' in model_name.lower():
        return "mistral"
    return "hf"


def get_model_token(model_name: str):
    model_source = get_model_source(model_name)
    dotenv.load_dotenv()  # Read from .env file
    if model_source == 'gpt':
        token = os.getenv('OPENAI_API_KEY')
    else:
        token = os.getenv('HF_TOKEN')  # Get the HuggingFace token
    return token


def get_llm(model_name, token=None, seed: int = 42, **kwargs):
    model_source = get_model_source(model_name)

    if model_source == 'gpt':
        llm = GPTClient(
            model_name=model_name,
            token=token,
            seed=seed,
            **kwargs
        )
    elif model_source == 'hf':
        # local_model_dir = "/leonardo_work/IscrB_ESG-NEXT/HF_models/" + model_name
        llm = HuggingFaceClient(
            model_name=model_name,
            token=token,
            seed=seed,
            # local_model_dir=local_model_dir, local_files_only=True,
            **kwargs
        )
    elif model_source == 'mistral':
        # local_model_dir = "/leonardo_work/IscrB_ESG-NEXT/HF_models/" + model_name
        llm = MistralClient(
            model_name=model_name,
            token=token,
            seed=seed,
            # local_model_dir=local_model_dir, local_files_only=True,
            **kwargs
        )
    else:
        raise ValueError(f'Model source {model_source} not supported')

    return llm
