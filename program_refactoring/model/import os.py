import os
import re
import time

import openai
import tiktoken
import timeout_decorator

from program_refactoring.model.model import Model

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class OpenAIModel(Model):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.api_key = OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def clean(self, text):
        if "Program" in text:
            results = re.findall("Program:(.*)", text, flags=re.DOTALL)
            if results:
                return results[-1]
        if "```" in text:
            text = text.replace("```python", "").replace("```", "")
        return text

    def build_model(self):
        if self.model_name not in ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"]:
            raise ValueError(f"Invalid model: {self.model_name}")

        def completion_lambda(x):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": x}],
                temperature=0.2,
                n=1,
                max_tokens=2000,
            )
            return response

        return completion_lambda

    def build_wide_model(self, beam_size=10):
        def completion_lambda(x):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": x}],
                temperature=0.7,
                n=beam_size,
                max_tokens=900,
            )
            return response

        return completion_lambda

    @timeout_decorator.timeout(240, timeout_exception=ValueError)
    def call_helper(self, x, agent):
        output = self.build_model()(x)
        text = output.choices[0].message.content
        return self.clean(text) if agent else text

    def __call__(self, x, attempts=0, agent=False):
        try:
            return self.call_helper(x, agent=agent)

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

    @timeout_decorator.timeout(240, timeout_exception=ValueError)
    def wide_call_helper(self, x, agent):
        output = self.build_wide_model()(x)
        texts = [choice.message.content for choice in output.choices]
        return list(set(texts))  # remove duplicates

    def wide_call(self, x, attempts=0, agent=False):
        try:
            return self.wide_call_helper(x, agent=agent)
        except openai.APIError:
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
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def __call__(self, x):
        return len(self.tokenizer.encode(x))
