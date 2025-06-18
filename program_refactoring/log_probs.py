# import os
# # from openai import OpeAnAI
# import openai

# openai_api_key = os.getenv("OPENAI_API_KEY")
# openai_api_base = "http://badfellow:9646/v1"

# if openai_api_key is None:
#     raise ValueError("OPENAI_API_KEY environment variable not set!")

# # client = OpenAI(
# client = openai.Client(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# def input_log_probs(input_text):
#     """
#     Function to get log probabilities for each token in the input text
#     and return the total log probability of the input text.
#     """

#     models = client.models.list()
#     model = models.data[0].id

#     print("Getting log probabilities for input text...")
#     completion = client.completions.create(
#         model=model,
#         prompt=input_text,
#         echo=True,  # input tokens are included in the response
#         logprobs=1,
#         max_tokens=1  # We only need log probabilities for the input tokens
#     )

#     tokens = completion.choices[0].logprobs.tokens
#     token_logprobs = completion.choices[0].logprobs.token_logprobs

#     total_log_prob = sum(token_logprobs)

#     return {
#         "tokens": tokens,
#         "logprobs": token_logprobs,
#         "total_log_prob": total_log_prob
#     }

# # Example usage
# # input_text = "Hello, how are you? I am just testing if this actually works you know."
# # result = input_log_probs(input_text)

# # print("Tokens:", result["tokens"])
# # print("Log probabilities:", result["logprobs"])
# # print("Total log probability:", result["total_log_prob"])


import os

import openai

openai_api_key = os.getenv("OPENAI_API_KEY")  # vLLM doesn't require API key
openai_api_base = "http://badfellow:9646/v1"

# Set the API base for OpenAI
openai.api_base = openai_api_base
openai.api_key = openai_api_key

# Manually set the model (vLLM doesn't list models)
# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Change this to your vLLM-hosted model


def input_log_probs(input_text):
    """
    Function to get log probabilities for each token in the input text
    and return the total log probability of the input text.
    """
    print("Getting log probabilities for input text...")

    completion = openai.Completion.create(
        model="Qwen2.5-Instruct",
        prompt=input_text,
        echo=True,  # Include input tokens in response
        logprobs=1,
        max_tokens=1,  # Only need log probs for input tokens
    )

    tokens = completion["choices"][0]["logprobs"]["tokens"]
    token_logprobs = completion["choices"][0]["logprobs"]["token_logprobs"]

    total_log_prob = sum(token_logprobs[1:])

    return {
        "tokens": tokens,
        "logprobs": token_logprobs,
        "total_log_prob": total_log_prob,
    }


# Example usage
# input_text = "Hello, how are you? I am just testing if this actually works you know."
# result = input_log_probs(input_text)

# print("Tokens:", result["tokens"])
# print("Log probabilities:", result["logprobs"])
# print("Total log probability:", result["total_log_prob"])
