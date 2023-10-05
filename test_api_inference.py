import os
import requests

with open('config.env') as f:
    os.environ.update(
        line.replace('export ', '', 1).strip().split('=', 1) for line in f
        if 'export' in line
    )

#api_end_point = "models/distilgpt2"
#api_end_point = "models/gpt2"
#api_end_point = "models/gpt2-large"
api_end_point="models/bigscience/bloom"



def query_models(api_end_point: str, string: str, max_new_tokens: int,
                 temperature: float, top_k: int, top_p: float, repetition_penalty: float):
    """
    Queries the specified Hugging Face model API endpoint with the 
    given string input, requesting the generation of a new text sequence 
    of up to max_new_tokens length.

    Parameters:
    -----------
        - api_end_point (str): the API endpoint of the Hugging Face model to be queried
        - string (str): the input text string to be used as the basis for the generated text
        - max_new_tokens (int): the maximum number of new tokens to be generated
        - temperature (float): the sampling temperature to be used in generating the new text (higher values lead to more random output)
        - top_k (int): the number of top-k candidates to be considered at each decoding step
        - repetition_penalty (float): the penalty to be applied to repeated tokens in the generated text (higher values lead to less repetition)

    Returns:
    ----------
        - A tuple containing:
            1. A string representing the generated text from the queried model
            2. A string representing the name of the queried model
    """

    payload = {"inputs": string,
            "parameters": {"top_k": top_k, "top_p": top_p, "temperature": temperature,
                            "max_new_tokens": max_new_tokens,
                            "num_return_sequences": 1,
                            "repetition_penalty": repetition_penalty,
                            'max_time': 30,
                            'do_sample': True,
                            'return_full_text': True},
            "options": {'use_cache': False,
                        'wait_for_model': True}}
    try:

        response = requests.post(
            f"https://api-inference.huggingface.co/{api_end_point}",
            headers={"Authorization": f"Bearer {os.environ.get('API_KEY')}"},
            json=payload).json()

        print(f"{api_end_point.split('/')[-1].title()} online")
        print(response)

        return response[0]['generated_text'], api_end_point.split('/')[-1].title()

    except:

        print(f"{api_end_point.split('/')[-1].title()} offline")

        return f"Server error. {api_end_point.split('/')[-1].title()} is offline.", api_end_point.split('/')[-1].title()

response, model = query_models(api_end_point, "What is Eco?", 250, 0.1, 10, .9, 1.1)

print(model)
print(response)