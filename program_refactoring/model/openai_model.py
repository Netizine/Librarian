import re
import time

import openai
import tiktoken
import timeout_decorator

from program_refactoring.model.model import Model

# OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]


client = openai.OpenAI()


class OpenAIModel(Model):
    def __init__(self, model_name):
        super().__init__(model_name)

    def clean(self, text):
        if "Program" in text:
            # find the last program
            results = re.findall("Program:(.*)", text, flags=re.DOTALL)
            if len(results) > 0:
                return results[-1]
        if "```" in text:
            text = re.sub("```python", "", text)
            text = re.sub("```", "", text)
        return text

    def build_model(self):
        if self.model_name not in [
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "o3-mini",
            "o4-mini",
        ]:
            raise ValueError(f"Invalid model: {self.model_name}")

        print("Model name: ", self.model_name)
        if self.model_name == "o3-mini" or self.model_name == "o4-mini":
            completion_lambda = lambda x: client.chat.completions.create(
                # engine=deployment_name,
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Python coder. You must return code in Markdown blocks:\n```python\n...\n```. Do not use carriage return.",
                    },
                    {"role": "user", "content": x},
                ],
                n=1,
            )
        else:
            completion_lambda = lambda x: client.chat.completions.create(
                # engine=deployment_name,
                model=self.model_name,
                # prompt = x,
                messages=[{"role": "user", "content": x}],
                temperature=0.2,
                n=1,
                max_tokens=2000,
            )

        # openai_api_key = os.getenv("OPENAI_API_KEY")  # vLLM doesn't require API key
        # openai_api_base = "http://badfellow:9646/v1"

        # # Set the API base for OpenAI
        # openai.api_base = openai_api_base
        # openai.api_key = openai_api_key

        # print("Getting log probabilities for input text...")

        # completion = openai.Completion.create(
        #     model="Qwen2.5-Instruct",
        #     prompt="hello world",
        #     echo=True,  # Include input tokens in response
        #     logprobs=1,
        #     max_tokens=1  # Only need log probs for input tokens
        # )

        # tokens = completion["choices"][0]["logprobs"]["tokens"]
        # token_logprobs = completion["choices"][0]["logprobs"]["token_logprobs"]

        # print(sum(token_logprobs[1:]))

        # openai.api_base = "https://api.openai.com/v1"

        return completion_lambda

    # def build_model(self):
    #     # Use OPENROUTER_API_KEY instead of OPENAI_API_KEY
    #     api_key = os.environ.get("OPENROUTER_API_KEY")

    #     if not api_key:
    #         raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    #     # Configure OpenAI library to use OpenRouter's base URL and API key
    #     openai.api_key = api_key
    #     openai.api_base = "https://openrouter.ai/api/v1"

    #     # Validate model names compatible with OpenRouter
    #     if self.model_name not in ["openai/o3-mini, openai/gpt-3.5-turbo", "openai/gpt-4-turbo", "openai/gpt-4o"]:
    #         raise ValueError(f"Invalid model: {self.model_name}")

    #     completion_lambda = lambda x: openai.ChatCompletion.create(
    #         model=self.model_name,
    #         messages=[{"role": "user", "content": x}],
    #         temperature=0.2,
    #         n=1,
    #         max_tokens=2000,
    #     )

    #     return completion_lambda

    # def build_model(self):

    #     completion_lambda = lambda x: openai.ChatCompletion.create(
    #         model=self.model_name,
    #         messages=[{"role": "user", "content": x}],
    #         temperature=0.2,
    #         n = 1,
    #         max_tokens=900,
    #         )

    #     return completion_lambda

    def build_wide_model(self, beam_size=10):
        completion_lambda = lambda x: openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": x}],
            temperature=0.7,
            n=beam_size,
            max_tokens=900,
        )

        return completion_lambda

    @timeout_decorator.timeout(
        240, timeout_exception=ValueError
    )  # set timeout for 4 minutes in case API gets stuck
    def call_helper(self, x, agent):
        output = self.build_model()(x)
        text = output.choices[0].message.content
        if agent:
            return self.clean(text)
        return text

    def __call__(
        self,
        x,
        attempts=0,
        infilling=False,
        agent=False,
        language=None,
        comment_tok=None,
    ):
        # for now, infilling is just here for Llama compatability
        try:
            output = self.call_helper(x, agent=agent)
            return output

        except openai.APIError:
            if attempts > 3:
                raise Exception("OpenAI API error, please try again later")
            time.sleep(5)
            return self(x, attempts + 1, agent=agent)
        except ValueError:
            if attempts > 3:
                raise Exception("API timed out")
            time.sleep(5)
            return self(x, attempts + 1, agent=agent)

    @timeout_decorator.timeout(
        240, timeout_exception=ValueError
    )  # set timeout for 4 minutes in case API gets stuck
    def wide_call_helper(self, x, agent):
        output = self.build_wide_model()(x)
        texts = []
        for choice in output["choices"]:
            text = choice["message"]["content"]
            texts.append(text)
        # no duplicates
        return list(set(texts))

    def wide_call(self, x, attempts=0, agent=False):
        try:
            return self.wide_call_helper(x, agent=agent)

        except openai.error.APIError:
            if attempts > 3:
                raise Exception("OpenAI API error, please try again later")
            time.sleep(5)
            return self.wide_call(x, attempts + 1, agent=agent)
        except ValueError:
            if attempts > 3:
                raise Exception("API timed out")
            time.sleep(5)
            return self.wide_call(x, attempts + 1, agent=agent)


class TokenCounter(Model):
    def __init__(self, model_name):
        super().__init__(model_name)

        self.tokenizer = tiktoken.get_encoding("c1100k_base")

    def __call__(self, x):
        return len(self.tokenizer.encode(x))
