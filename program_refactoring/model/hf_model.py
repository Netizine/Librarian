import pdb
import re

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from program_refactoring.model.model import Model


class HFModel(Model):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Tokenizer with model...", self.model_name)
        if model_name == "Qwen/Qwen2.5-14B-Instruct-1M":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, padding_side="left"
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )  # , padding_side="left")
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        self.gen_pipeline = pipeline(
            "text-generation",
            batch_size=1,
            model=self.model,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            device_map="auto",
        )
        self.gen_pipeline.tokenizer.pad_token_id = self.model.config.eos_token_id

    def run(
        self,
        prompt,
        infilling=False,
        agent=False,
        language="PYTHON",
        comment_tok="#",
        pipeline_kwargs={},
    ):
        """Run the model on a prompt"""

        output = self.gen_pipeline(
            prompt,
            max_new_tokens=500,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            num_return_sequences=1,
            **pipeline_kwargs,
        )

        # output = self.tokenizer.decode(outputs[0])
        return output[0]["generated_text"]

    def __call__(
        self,
        x,
        infilling=False,
        agent=False,
        language="PYTHON",
        comment_tok="#",
        pipeline_kwargs={},
    ):
        return self.run(x, infilling, agent, language, comment_tok, pipeline_kwargs)


class CodeLlamaModel(HFModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def clean_result(
        self,
        prompt,
        output,
        infilling=False,
        agent=False,
        language="PYTHON",
        comment_tok="#",
    ):
        output = output[0]["generated_text"]
        if infilling:
            # everything after prompt
            gen_text = re.split(f"\[\/{language}\]", output)[-1]

            # everything before it starts making up new prompts
            program = re.split(f"{comment_tok} Query:", gen_text)[0].strip()
            return program
        else:
            # model often generates a bunch of additional queries and programs
            # try to find the one that matches the query
            my_query = re.findall(".*Query:(.*)", prompt)[-1].strip()
            # pdb.set_trace()
            try:
                query_m = re.search(
                    f"({comment_tok} Query: {my_query}\n(.*?))(({comment_tok} Query)|($))",
                    output,
                    flags=re.DOTALL,
                )
            except:
                pdb.set_trace()
            if query_m is not None:
                prog = query_m.group(1)
                # take everything before the close
                prog = re.split(f"\[\/?{language}\]", prog)[0]
                prog = re.sub(f"\[\/?{language}\]", "", prog)
                return prog
            else:
                if my_query in output:
                    after_query_idx = output.index(my_query) + len(my_query)
                    prog = output[after_query_idx:]
                    prog = re.split(f"{comment_tok} Query", prog)[0]
                    prog = re.sub(f"\[\/?{language}\]", "", prog)
                    return prog
                else:
                    # extract final [PYTHON] text

                    matches = re.findall(
                        f"\[{language}\](.*?)\[\/{language}\]", output, flags=re.DOTALL
                    )
                    try:
                        final_match = matches[-1]
                    except IndexError:
                        pdb.set_trace()
                    prog = re.sub(f"\[\/?{language}\]", "", final_match)

                    # get query
                    test_query = re.findall(".*Query:.*", prompt)[-1].strip()

                    try:
                        # often the model generates multiple programs, so we take the first one
                        all_progs = re.split(f"({comment_tok} Query:.*)", prog)
                        progs_by_query = {}
                        for i, row in enumerate(all_progs):
                            if row.startswith(f"{comment_tok} Query:"):
                                try:
                                    body = all_progs[i + 1]
                                except IndexError:
                                    body = "NONE"
                                progs_by_query[row.strip()] = body
                        try:
                            prog_for_query = progs_by_query[test_query]
                        except KeyError:
                            prog_for_query = "".join(all_progs[0:3])

                    except IndexError:
                        prog_for_query = prog

                    # split off any remaining [INST] things
                    if "[INST" in prog_for_query:
                        prog_for_query = re.split("\[\/?INST.*?\]?", prog_for_query)[0]

                return prog_for_query

    def run(
        self,
        prompt,
        infilling=False,
        agent=False,
        language="PYTHON",
        comment_tok="#",
        pipeline_kwargs={},
    ):
        text = super().run(prompt, pipeline_kwargs=pipeline_kwargs)
        return self.clean_result(prompt, text, infilling, agent, language, comment_tok)

    def run_multiple(
        self,
        prompts,
        batch_size=5,
        infilling=False,
        agent=False,
        language="PYTHON",
        comment_tok="#",
    ):
        def dataset():
            print("Running prompts...")
            for prompt in tqdm(prompts, total=len(prompts)):
                yield prompt

        texts = self.gen_pipeline(
            dataset(),
            max_new_tokens=500,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            num_return_sequences=1,
            batch_size=batch_size,
        )

        texts = [x for x in texts]
        assert len(prompts) == len(texts)
        results = []
        print("Cleaning results...")
        for prompt, text in tqdm(zip(prompts, texts), total=len(prompts)):
            res = self.clean_result(
                prompt, text, infilling, agent, language, comment_tok
            )
            results.append(res)
        return results


class QwenModel(HFModel):
    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct-1M"):
        model_name = "Qwen/Qwen2.5-14B-Instruct-1M"
        super().__init__(model_name)

    # def clean_result(self, prompt, output, infilling=False, agent=False, language="PYTHON", comment_tok="#"):
    #     """Clean the model output to extract the generated code."""
    #     output = output[0]['generated_text']

    #     if infilling:
    #         # everything after prompt
    #         gen_text = re.split(f"\[\/{language}\]", output)[-1]

    #         # everything before it starts making up new prompts
    #         program = re.split(f"{comment_tok} Query:", gen_text)[0].strip()
    #         return program
    #     else:
    #         # Extract query from prompt
    #         my_query = re.findall(".*Query:(.*)", prompt)[-1].strip()

    #         try:
    #             # Try to find the section that matches the query
    #             query_m = re.search(f"({comment_tok} Query: {my_query}\n(.*?))(({comment_tok} Query)|($))", output, flags=re.DOTALL)

    #             if query_m is not None:
    #                 prog = query_m.group(1)
    #                 # take everything before the close
    #                 prog = re.split(f"\[\/?{language}\]", prog)[0]
    #                 prog = re.sub(f"\[\/?{language}\]", "", prog)
    #                 return prog
    #             elif my_query in output:
    #                 after_query_idx = output.index(my_query) + len(my_query)
    #                 prog = output[after_query_idx:]
    #                 prog = re.split(f"{comment_tok} Query", prog)[0]
    #                 prog = re.sub(f"\[\/?{language}\]", "", prog)
    #                 return prog
    #             else:
    #                 # extract final [PYTHON] text
    #                 matches = re.findall(f"\[{language}\](.*?)\[\/{language}\]", output, flags=re.DOTALL)

    #                 try:
    #                     final_match = matches[-1]
    #                 except IndexError:
    #                     # If no matches found, return the full output
    #                     return output.strip()

    #                 prog = re.sub(f"\[\/?{language}\]", "", final_match)

    #                 # get query
    #                 test_query = re.findall(".*Query:.*", prompt)[-1].strip()

    #                 try:
    #                     # often the model generates multiple programs, so we take the first one
    #                     all_progs = re.split(f"({comment_tok} Query:.*)", prog)
    #                     progs_by_query = {}

    #                     for i, row in enumerate(all_progs):
    #                         if row.startswith(f"{comment_tok} Query:"):
    #                             try:
    #                                 body = all_progs[i+1]
    #                             except IndexError:
    #                                 body = "NONE"
    #                             progs_by_query[row.strip()] = body

    #                     try:
    #                         prog_for_query = progs_by_query[test_query]
    #                     except KeyError:
    #                         prog_for_query = "".join(all_progs[0:3])
    #                 except IndexError:
    #                     prog_for_query = prog

    #                 # Handle Qwen specific formatting: split off any remaining assistant/user markers
    #                 # Adjust these patterns based on Qwen's actual output format
    #                 if "<|im_start|>assistant" in prog_for_query.lower():
    #                     prog_for_query = re.split("<\|im_start\|>assistant", prog_for_query, flags=re.IGNORECASE)[0]

    #                 if "<|im_start|>user" in prog_for_query.lower():
    #                     prog_for_query = re.split("<\|im_start\|>user", prog_for_query, flags=re.IGNORECASE)[0]

    #                 # Clean up any remaining Qwen chat markers
    #                 prog_for_query = re.sub("<\|im_(start|end)\|>(user|assistant)?:?", "", prog_for_query, flags=re.IGNORECASE)

    #                 return prog_for_query.strip()

    #         except Exception as e:
    #             # Return original output if parsing fails
    #             return output.strip()

    def clean_result(
        self,
        prompt,
        output,
        infilling=False,
        agent=False,
        language="PYTHON",
        comment_tok="#",
    ):
        """Clean the model output to extract the generated code for Qwen model."""
        output = output[0]["generated_text"]

        # Extract query from prompt
        my_query = re.findall(".*Query:(.*)", prompt)[-1].strip()

        try:
            # First, try to extract code from markdown code blocks
            code_blocks = re.findall(r"```python\n(.*?)```", output, re.DOTALL)

            if code_blocks:
                # Get the first code block that contains "# Thought:"
                for block in code_blocks:
                    if "# Thought:" in block:
                        return block.strip()

                # If no block with "# Thought:", return the first code block
                return code_blocks[0].strip()

            # If no code blocks found, try other extraction methods
            # Look for content after assistant marker
            if "<|im_start|>assistant" in output:
                parts = re.split("<\|im_start\|>assistant", output, flags=re.IGNORECASE)
                if len(parts) > 1:
                    assistant_part = parts[1]
                    # Extract content between the start of assistant's response and the end marker
                    if "<|im_end|>" in assistant_part:
                        assistant_part = assistant_part.split("<|im_end|>")[0]

                    # Remove any remaining markdown code block markers
                    assistant_part = re.sub(r"```python|```", "", assistant_part)
                    return assistant_part.strip()

            # If all else fails, try to extract anything that looks like Python code
            # This is a fallback approach
            lines = output.split("\n")
            code_lines = []
            in_code = False

            for line in lines:
                if (
                    line.strip().startswith("# Thought:")
                    or line.strip().startswith("def ")
                    or line.strip().startswith("for ")
                ):
                    in_code = True

                if in_code and "<|im_" not in line:
                    code_lines.append(line)

                if in_code and "<|im_end|>" in line:
                    in_code = False

            if code_lines:
                return "\n".join(code_lines).strip()

            # Last resort: return everything after "# Thought:" if it exists
            if "# Thought:" in output:
                return output.split("# Thought:")[1].strip()

            # If all extraction attempts fail, return the original output
            return output.strip()

        except Exception as e:
            # Return original output if parsing fails
            print(f"Error cleaning output: {e}")
            return output.strip()

    def run(
        self,
        prompt,
        infilling=False,
        agent=False,
        language="PYTHON",
        comment_tok="#",
        pipeline_kwargs={},
    ):
        """Run the model on a prompt and clean the result."""
        # Format prompt for Qwen2.5 chat format if needed
        formatted_prompt = self._format_prompt_for_qwen(prompt)

        # Run the model
        text = super().run(formatted_prompt, pipeline_kwargs=pipeline_kwargs)

        # Clean and return the result
        return self.clean_result(prompt, text, infilling, agent, language, comment_tok)

    def _format_prompt_for_qwen(self, prompt):
        """
        Format the prompt for Qwen2.5 chat format if it's not already formatted.
        This helps ensure consistent input format for the model.
        """
        # Check if the prompt is already in Qwen chat format
        if "<|im_start|>" in prompt:
            return prompt

        # Basic formatting for code generation
        formatted_prompt = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        return formatted_prompt

    def run_multiple(
        self,
        prompts,
        batch_size=5,
        infilling=False,
        agent=False,
        language="PYTHON",
        comment_tok="#",
    ):
        """Run the model on multiple prompts and clean the results."""
        # Format all prompts for Qwen
        formatted_prompts = [self._format_prompt_for_qwen(prompt) for prompt in prompts]

        def dataset():
            print("Running prompts...")
            for prompt in tqdm(formatted_prompts, total=len(formatted_prompts)):
                yield prompt

        texts = self.gen_pipeline(
            dataset(),
            max_new_tokens=500,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            num_return_sequences=1,
            batch_size=batch_size,
        )

        texts = [x for x in texts]
        assert len(prompts) == len(texts)

        results = []
        print("Cleaning results...")
        for prompt, text in tqdm(zip(prompts, texts), total=len(prompts)):
            res = self.clean_result(
                prompt, text, infilling, agent, language, comment_tok
            )
            results.append(res)

        return results


class LemurModel(HFModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def run(
        self,
        prompt,
        infilling=False,
        agent=False,
        language="PYTHON",
        comment_tok="#",
        pipeline_kwargs={},
    ):
        text = super().run(prompt)
        return self.clean_result(prompt, text, infilling, agent, language, comment_tok)

    def clean_result(
        self,
        prompt,
        output,
        infilling=False,
        agent=False,
        language="PYTHON",
        comment_tok="#",
    ):
        # model often generates a bunch of additional queries and programs
        # try to find the one that matches the query
        output = output[0]["generated_text"]
        my_query = re.findall(".*Query:(.*)", prompt)[-1].strip()

        # query_m = re.search(f"({comment_tok} Query: {my_query}(.*?))(({comment_tok} Query)|($))", output, flags=re.DOTALL)
        query = re.sub("\?", "\?", my_query)
        try:
            query_m = re.findall(
                f"(?<=({comment_tok} Query: {query}))(.*?)(?=({comment_tok} Query))",
                output,
                flags=re.DOTALL,
            )
        except re.error:
            query_m = None
        if len(query_m) == 0:
            # only 1 query so take the last one
            query_m = re.findall(
                f"(?<=({comment_tok} Query: {query}))(.*)", output, flags=re.DOTALL
            )

        for match_obj in query_m:
            body = match_obj[1]
            # replace all comment lines
            body = re.sub(f"{comment_tok}.*", "", body)
            body = body.strip()
            # if len(body) > 0:
            # prog = match_obj[1]
            prog = body
            prog = re.sub(f"\[\/?{language}\]", "", prog)
            prog = re.sub("<\|im_end\|>", "", prog)
            prog = re.sub("<\|im_start\|> .*:", "", prog)
            prog = re.sub("Please generate.*", "", prog)
            prog = prog.strip()
            if len(prog) > 0:
                return prog

        # extract first [PYTHON] text

        matches = re.findall(
            "<\|im_start\|> user:(.*?)<\|im_end\|>", output, flags=re.DOTALL
        )
        try:
            final_match = matches[-1]
        except IndexError:
            if my_query in output:
                after_query_idx = output.index(my_query) + len(my_query)
                prog = output[after_query_idx:]
                prog = re.split(f"{comment_tok} Query", prog)[0]
                prog = re.sub("Please generate.*", "", prog)
                return prog
            else:
                return output

        prog = re.sub(f"\[\/?{language}\]", "", final_match)
        prog = re.sub("<\|im_end\|>", "", prog)
        prog = re.sub("<\|im_start\|> user:", "", prog)

        # get query
        test_query = re.findall(".*Query:.*", prompt)[-1].strip()

        try:
            # often the model generates multiple programs, so we try to get from anywhere
            all_progs = re.split(f"({comment_tok} Query:.*)", output)
            progs_by_query = {}
            for i, row in enumerate(all_progs):
                if row.startswith(f"{comment_tok} Query:"):
                    try:
                        body = all_progs[i + 1]
                    except IndexError:
                        body = "NONE"
                    progs_by_query[row.strip()] = body
            try:
                prog_for_query = progs_by_query[test_query]
            except KeyError:
                # take the last one
                prog_for_query = "".join(all_progs[-2:])
                # prog_for_query = "".join(all_progs[0:3])

        except IndexError:
            prog_for_query = prog

        prog_for_query = re.sub("<\|im_end\|>", "", prog_for_query)
        prog_for_query = re.sub("<\|im_start\|> user:", "", prog_for_query)
        prog_for_query = re.sub("Please generate.*", "", prog_for_query)

        return prog_for_query

    def run_multiple(
        self,
        prompts,
        batch_size=3,
        infilling=False,
        agent=False,
        language="PYTHON",
        comment_tok="#",
    ):
        def dataset():
            print("Running prompts...")
            for prompt in tqdm(prompts, total=len(prompts)):
                yield prompt

        texts = self.gen_pipeline(
            dataset(),
            max_new_tokens=700,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            num_return_sequences=1,
            batch_size=batch_size,
        )

        texts = [x for x in texts]
        assert len(prompts) == len(texts)
        results = []
        print("Cleaning results...")
        for prompt, text in tqdm(zip(prompts, texts), total=len(prompts)):
            res = self.clean_result(
                prompt, text, infilling, agent, language, comment_tok
            )
            results.append(res)
        # pdb.set_trace()
        return results
