import os

import dotenv


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
        from llm_clients.gpt_client import GPTClient
        llm = GPTClient(
            model_name=model_name,
            token=token,
            seed=seed,
            **kwargs
        )
    elif model_source == 'hf':
        from llm_clients.hf_client import HuggingFaceClient
        llm = HuggingFaceClient(
            model_name=model_name,
            token=token,
            seed=seed,
            **kwargs
        )
    elif model_source == 'mistral':
        from llm_clients.mistral_client import MistralClient
        llm = MistralClient(
            model_name=model_name,
            token=token,
            seed=seed,
            **kwargs
        )
    else:
        raise ValueError(f'Model source {model_source} not supported')

    return llm
