from llm_clients.utils import get_llm, get_model_token


if __name__ == '__main__':
    # MODEL_NAME = 'gpt-4o-mini'
    MODEL_NAME = 'meta-llama/Llama-3.1-8B-Instruct'
    TOKEN = get_model_token(MODEL_NAME)

    llm_client = get_llm(MODEL_NAME, TOKEN)

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    ans, _ = llm_client.ask(messages, temperature=0.01, max_response_length=512)

    print(ans)
