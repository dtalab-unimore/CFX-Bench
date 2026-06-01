from llm_clients.utils import get_model_source


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
    else:
        raise ValueError(f'Model source {model_source} not supported')

    return llm


if __name__ == '__main__':
    MODEL_NAME = 'gpt-4o-mini'
    # MODEL_NAME = 'meta-llama/Llama-3.1-8B-Instruct'
    from llm_clients.utils import get_model_token
    TOKEN = get_model_token(MODEL_NAME)

    llm_client = get_llm(MODEL_NAME, TOKEN)

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    ans, _ = llm_client.ask(messages, temperature=0.01, max_response_length=512)

    print(ans)
