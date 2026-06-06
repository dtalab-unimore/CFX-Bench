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
