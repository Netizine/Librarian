import ast
import asyncio
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import openai
from radon.complexity import cc_visit
from radon.raw import analyze
from tenacity import retry, stop_after_attempt, wait_exponential
from together import AsyncTogether

from program_refactoring.codebank.codebank import CodeBank
from program_refactoring.codebank.test_case import (
    CodeContestTestCase,
)
from program_refactoring.domains.python.utils import get_func_names
from program_refactoring.model.prompts import (
    RETRIEVE_CODEBANK_TEMPLATE,
    python_tuple_refactor_prompt,
)
from program_refactoring.tree.node import CodeContestNode, Node
from program_refactoring.tree.tuple import Tuple
import json
import time
from pathlib import Path 

np.random.seed(12)

logger = logging.getLogger(__name__)
client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))
aclient = openai.AsyncOpenAI()


RESULTS_TEMPLATE = """Unit test {i}

Input:
{test_input}

Expected output:
{test_output}

Program output:
{predicted_output}

"""


def extract_query_and_code(text):
    pattern = r"# Program\s+(\d+)\s*\n```python\n([\s\S]+?)\n```"
    matches = re.finditer(pattern, text.replace("\r\n", "\n"))
    results = []

    for match in matches:
        query_num = match.group(1)
        code = match.group(2)
        results.append({"query_number": query_num, "code": code})

    return results


def construct_repair_prompt(
    idx, node_before, node_after, results_after, helper_functions
):
    prompt_internal_str = f"""Query {idx}: {node_before.query}
Program {idx}:
```python
{node_after.program}
```

Here are the failed unit tests:
"""

    failed_test_data = [
        (inp, out, pout, stderr)
        for inp, out, pout, stderr, passed in zip(
            results_after[idx].test_inputs,
            results_after[idx].test_outputs,
            results_after[idx].predicted_outputs,
            results_after[idx].stderrs,
            results_after[idx].passed,
        )
        if not passed
    ]
    for i, (test_input, test_output, predicted_output, stderr) in enumerate(
        failed_test_data
    ):
        prompt_internal_str += RESULTS_TEMPLATE.format(
            i=i,
            test_input=test_input,
            test_output=test_output + stderr,
            predicted_output=predicted_output,
        )

    prompt = f"""ERROR: the following program failed to pass the unit tests.
Helper functions:\n```python\n{helper_functions}\n```
{prompt_internal_str}

Please fix the codebank and main program. Do not add classes. Do not use nonlocal variables.
If the broken code contains nonlocal variables, rewrite the program without nonlocal variables.
Do not add the main function to the codebank.
Your answer must follow the Markdown format of the two examples below.

The first code block should contain the helper functions:

# Codebank
```python
def helper_function():
    ...
```
Do not add constant variables to the codebank. Instead, declare those variables in the main functions as shown below.

The remaining code blocks should contain the main functions for each respective query.
Be sure to use the helper functions.
For query i, this would be:

# Program i
```python
from codebank import *

CONSTANT = 1

def main():
    ...

if __name__ == "__main__":
    main()
```"""
    return prompt


class PythonTuple(Tuple):
    def __init__(
        self,
        nodes: List[Node],
        use_removed=False,
        task: str = "python",
        temp_dir: str = "temp",
    ):
        super().__init__(nodes, task)

        self.use_removed = use_removed
        self.temp_dir = temp_dir
        if self.task == "scan":
            self.tuple_refactor_prompt = scan_tuple_refactor_prompt
        else:
            self.tuple_refactor_prompt = python_tuple_refactor_prompt

    def import_codebank(self, program):
        program = f"""from codebank import *\n\n{program}"""
        return program

    def remove_import(self, x):
        return re.sub("from .*\.codebank import \*", "", x).strip()

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def oai_and_parse(self, prompt):
        response = await aclient.chat.completions.create(
            model="o4-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python coder. You must return code in Markdown blocks:\n```python\n...\n```. Do not use carriage return.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content
        local_new_helpers, new_programs = self.parse_result(text)
        logging.info("OAI PROMPT")
        logging.info(prompt)
        logging.info("OAI RESPONSE")
        logging.info(text)
        return local_new_helpers, new_programs

    async def retry_merge(
        self,
        codebank,
        model,
        helper_functions,
        nodes_before,
        nodes_after,
        nodes_succeed,
        functions_used,
        results_before,
        results_after,
    ):
        """Retry the merge with a new prompt, including some feedback"""

        # NOTE (elias): Moving to retrying individual examples
        new_helpers = ""
        # new_programs = {idx: None for idx in nodes_after.keys()}
        new_programs = {}
        repair_prompts = []  # list[(idx, prompt)]
        for idx, node_before in nodes_before.items():
            try:
                node_after = nodes_after[idx]
            except KeyError:
                logger.info(f"ERROR: node {idx} not in nod")
                continue
            funcs_used = functions_used[idx]

            if len(funcs_used) > 0:
                funcs_used = set(funcs_used)
                funcs_used = ", ".join(funcs_used)
                # TODO: maybe need to only mention the codebank functions?
                functions_used_str = f"Pay special attention to the following functions: {funcs_used} and refactor them if needed."
            else:
                functions_used_str = ""

            node_success = nodes_succeed[idx]
            if not node_success:
                print(f"Node {idx} failed to execute")
                prompt = construct_repair_prompt(
                    idx, node_before, node_after, results_after, helper_functions
                )
                # Get the actual test cases for repair.
                repair_prompts.append((idx, prompt))
            else:
                continue
                # return None, {idx: None for idx in nodes_before.keys()}
            logger.info(f"Running retry with prompt: {prompt}")

        # execute repair model calls in parallel
        results = await asyncio.gather(
            *[self.oai_and_parse(prompt) for i, prompt in repair_prompts]
        )
        assert len(results) == len(repair_prompts)

        for (idx, prompt), (local_new_helpers, new_program_dict) in zip(
            repair_prompts, results
        ):
            # sometimes model hallucinates additional programs
            # justin: wut?
            new_program = list(new_program_dict.values())[0]
            # new_programs = {k:v for k, v in new_programs.items() if k in results_before.keys()}
            # return new_helpers, new_programs
            # if local_new_helpers is not None and len(local_new_helpers) > 0:
            #     pdb.set_trace()
            new_helpers += "\n" + local_new_helpers + "\n"
            new_programs[idx] = new_program
        if len(new_helpers) == 0:
            new_helpers = None

        return new_helpers, new_programs

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_logprobs_together(self, client, model: str, text: str):
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            max_tokens=1,
            echo=True,
            logprobs=1,
        )
        logprobs = response.prompt[0].logprobs.token_logprobs[1:]
        tokens = response.prompt[0].logprobs.tokens[1:]
        return logprobs, tokens

    def compute_code_metrics(self, code: str) -> dict:
        try:
            raw = analyze(code)
            cc_blocks = cc_visit(code)

            return {
                "loc": raw.loc,
                "sloc": raw.sloc,
                "lloc": raw.lloc,
                "comments": raw.comments,
                "multi": raw.multi,
                "blank": raw.blank,
                "cyclomatic": sum(b.complexity for b in cc_blocks),
            }
        except Exception as e:
            print(f"[!] Error computing code metrics: {e}")
            return {
                "loc": 0,
                "sloc": 0,
                "lloc": 0,
                "comments": 0,
                "multi": 0,
                "blank": 0,
                "cyclomatic": 1000,  # penalty for failure
            }

    async def _process_program(
        self, client, model, codebank_str, program_code
    ):
        """Processes a single program within a refactoring (fetches logprobs, computes metrics)."""
        full_code = codebank_str + "\n" + program_code
        code_metrics = self.compute_code_metrics(program_code)
        try:
            prog_logprobs, program_tokens  = await self.get_logprobs_together(client, model, full_code)
        except json.JSONDecodeError as e:
            print("[!] JSON decoding failed in Together response:", e)
            print("[!] Full code:\n", full_code[:1000], "...")  # truncate if long
            return {"logprob": float('-inf')}  # or a sensible fallback
        # prog_logprobs, _ = await self.get_logprobs_together(client, model, full_code)
        #program_logprob_sum = sum(prog_logprobs[len(codebank_tokens) :])
        # sum over everything, since we want to include whether the new helper functions improve compression
        program_logprob_sum = sum(prog_logprobs)
        program_token_num = len(program_tokens)
        return program_logprob_sum, code_metrics, program_token_num

    async def _process_single_refactoring(
        self, refactoring, client, model, metrics_to_consider, codebank_retrieved
    ):
        """Processes all programs within a single refactoring concurrently."""
        new_programs = refactoring["new_programs"]
        codebank_str = ''
        if codebank_retrieved is not None:
            codebank_str = codebank_retrieved + refactoring["new_helpers"] 
        else:
            codebank_str = refactoring["new_helpers"]
        
        helper_code_metrics = self.compute_code_metrics(codebank_str)
        helper_code_logprobs = await self.get_logprobs_together(client, model, codebank_str)
        helper_token_num = len(helper_code_logprobs[1])
        
        
        # --- Parallelize over programs ---
        program_tasks = []
        for p_code in new_programs.values():
            program_tasks.append(self._process_program(client, model, codebank_str, p_code))

        program_results = await asyncio.gather(*program_tasks)

        # metrics_total = {
        #     "logprob": 0,
        #     "cyclomatic": 0,
        #     "lloc": 0,
        #     "sloc": 0,
        #     "pass_rate": 0,
        #     # Add others if needed by compute_code_metrics result
        # }
        
        metrics_total = {
            "logprob": sum(helper_code_logprobs[0]) if helper_code_logprobs else 0,
            "cyclomatic": helper_code_metrics.get("cyclomatic", 0),
            "lloc": helper_code_metrics.get("lloc", 0),
            "sloc": helper_code_metrics.get("sloc", 0),
            "pass_rate": 0.0,
        }

        for prog_logprob_sum, code_metrics, program_token in program_results:
            metrics_total["logprob"] += prog_logprob_sum
            for k in metrics_total:
                if k != "logprob" and k != "pass_rate":  # handled separately
                    metrics_total[k] += code_metrics.get(k, 0)

        metrics_total["pass_rate"] = sum(
            refactoring.get("ratio_passed_after", {}).values()
        )

        refactoring_metrics = {}
        for metric in metrics_to_consider:
            if metric == "logprob":
                # lower magnitude (less negative) is better, sum includes new codebank
                refactoring_metrics[metric] = -metrics_total["logprob"]
            elif metric == "pass_rate":
                # Higher is better, negate for minimization
                refactoring_metrics[metric] = -metrics_total["pass_rate"]
            else:
                # Assume lower is better for other metrics (like lloc, sloc, cyclomatic)
                refactoring_metrics[metric] = metrics_total.get(metric, 0)
        
        refactoring_metrics["total_tokens"] = sum([program_token for _, _, program_token in program_results]) + helper_token_num

        # return refactoring_metrics
        return metrics_total, refactoring_metrics


    async def select_best_refactoring(
        self,
        sampled_refactorings,
        codebank_str,
        client,
        model = "Qwen/Qwen2.5-7B-Instruct-Turbo",
        metrics=["logprob"],
    ):
        if metrics is None:
            metrics = ["logprob"]

        
        # --- Parallelize over refactorings ---
        refactoring_tasks = []
        for refactoring in sampled_refactorings:
            refactoring_tasks.append(
                self._process_single_refactoring(refactoring, client, model, metrics, codebank_str)
            )

        all_results_tuples = await asyncio.gather(*refactoring_tasks) 

        # all_results is a list of dictionaries, e.g., [{'logprob': -x, 'lloc': y}, {'logprob': -z, 'lloc': w}, ...]
        all_results = [refactoring_metrics for (_, refactoring_metrics) in all_results_tuples]

        if not all_results_tuples:
            print("Warning: No refactoring results to process.")
            return -1  

        # acccumulate metrics for entire refactoring
        metric_accumulators = {metric: [] for metric in metrics}
        for result_tuple in all_results_tuples:
            if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                loss_values_for_current_refactoring = result_tuple[1] # This is refactoring_metrics
                for metric in metrics:
                    metric_accumulators[metric].append(
                        loss_values_for_current_refactoring.get(metric, float("inf"))
                    )
            else:
                print(f"Unexpected result format in all_results_tuples: {result_tuple}")
                for metric in metrics:
                    metric_accumulators[metric].append(float("inf"))
                    
        # --- Normalize and compute total loss ---
        normalized = {}
        for metric, values in metric_accumulators.items():
            arr = np.array(values, dtype=np.float64)
            # Handle cases where all values are infinite (e.g., all failed) or constant
            valid_arr = arr[np.isfinite(arr)]
            if valid_arr.size > 0 and valid_arr.max() != valid_arr.min():
                min_val = valid_arr.min()
                max_val = valid_arr.max()
                normalized_arr = np.where(
                    np.isfinite(arr), (arr - min_val) / (max_val - min_val), 1.0
                )  # Map inf to 1 after normalization
                if max_val == min_val:
                    normalized_arr = np.where(np.isfinite(arr), 0.0, 1.0)
            else:  # All values are the same or all are infinite
                normalized_arr = np.zeros_like(arr)  # Assign 0 if constant and finite
                normalized_arr[np.isinf(arr)] = 1.0  # Assign 1 if infinite

            normalized[metric] = normalized_arr
            
            

        # Simpler loss accumulation
        first_metric_key = metrics[0]
        total_loss = np.zeros_like(normalized[first_metric_key], dtype=np.float64)
        for metric_key in metrics:
            if metric_key in normalized:  # Ensure metric was processed
                total_loss += normalized[metric_key]
            else:
                print(
                    f"Warning: Metric '{metric_key}' not found in normalized results."
                )
                

        print("Collected Metrics:", metric_accumulators)
        print("Normalized metrics:", normalized)
        print("Total loss:", total_loss)

        if len(total_loss) == 0:
            print("Warning: Total loss calculation resulted in an empty array.")
            return -1  

        final_best_idx = -1     
        sorted_indices = np.argsort(total_loss)

        # try to find the first refactoring with improved mean pass rate
        for idx in sorted_indices:
            refactoring = sampled_refactorings[idx]
            # ratio_before = np.mean(list(refactoring.get("ratio_passed_before", {}).values()) or [0.0])
            # ratio_after = np.mean(list(refactoring.get("ratio_passed_after", {}).values()) or [0.0])
            
            before_vals = list(refactoring.get("ratio_passed_before", {}).values())
            
            ratio_before = np.mean(before_vals) if before_vals else 0.0

            after_vals = list(refactoring.get("ratio_passed_after", {}).values())
           
            ratio_after = np.mean(after_vals) if after_vals else 0.0

            print(f"Checking idx {idx} | mean before: {ratio_before:.3f}, mean after: {ratio_after:.3f}")
            if np.isfinite(ratio_after) and ratio_after >= ratio_before:
                final_best_idx = idx
                # return idx
                break

        if final_best_idx == -1 and len(sorted_indices) > 0:
            final_best_idx = int(sorted_indices[0])
        # Fallback: return the one with lowest total loss
        
        # --- Printing all metrics 
        print("\n--- Refactoring Evaluation Summary ---")
        for i, refactoring_data in enumerate(sampled_refactorings):
            highlight = "    <--- SELECTED" if i == final_best_idx else ""
            print(f"Refactoring {i}:{highlight}")

            # Pass rates
            before_pass_values = list(refactoring_data.get("ratio_passed_before", {}).values())
            mean_pass_rate_before = np.mean(before_pass_values) if before_pass_values else 0.0
            
            after_pass_values = list(refactoring_data.get("ratio_passed_after", {}).values())
            mean_pass_rate_after = np.mean(after_pass_values) if after_pass_values else 0.0
            
            print(f"  Mean Pass Rate Before: {mean_pass_rate_before:.3f}")
            print(f"  Mean Pass Rate After:  {mean_pass_rate_after:.3f}")

            # Individual metric loss values (from all_results)
            # all_results[i] contains a dictionary like {'logprob': value, 'lloc': value}
            if i < len(all_results): # Ensure index is valid
                current_refactoring_metric_losses = all_results[i]
                for metric_name in metrics: # 'metrics' is the list of metric names like ["logprob", "lloc"]
                    loss_value = current_refactoring_metric_losses.get(metric_name, 'N/A')
                    print(f"  Metric '{metric_name}' (loss): {loss_value}")
            else:
                # This case should ideally not be reached if all_results has an entry for each refactoring
                print(f"  Metrics (loss): Data not available for refactoring {i}")

            # print logprobs
            if i < len(all_results_tuples):
                print(f"  Logprobs: {all_results_tuples[i][0].get('logprob', 'N/A')}")

            # Total normalized loss
            if i < len(total_loss): # Ensure index is valid
                print(f"  Total Normalized Loss: {total_loss[i]:.4f}")
            else:
                # This case should ideally not be reached if total_loss has an entry for each refactoring
                print(f"  Total Normalized Loss: N/A for refactoring {i}")
            print("-" * 20)
        print("--- End of Refactoring Evaluation Summary ---")
        
        # === logging for human eval ===

        human_eval = False 
        # Save refactoring data for human evaluation =====
        if human_eval:
            node_ids = [node.node_id for node in self.nodes.values()]
            tuple_folder_name = f"tuple_{'_'.join(map(str, node_ids))}"
            os.makedirs(tuple_folder_name, exist_ok=True)
            
            
            # 0. Save Problem Queries
            problem_queries_folder = os.path.join(tuple_folder_name, "problem_queries")
            os.makedirs(problem_queries_folder, exist_ok=True)
            if self.nodes:
                for idx, node in self.nodes.items():
                    if hasattr(node, 'query') and hasattr(node, 'node_id'):
                        query_file_path = os.path.join(problem_queries_folder, f"node_{node.node_id}_query.txt")
                        with open(query_file_path, "w", encoding='utf-8') as f:
                            f.write(node.query)
                    else:
                        print(f"Warning: Node (idx: {idx}) missing 'query' or 'node_id'. Skipping query save.")
                    
            # 1. Save original programs
            original_folder = os.path.join(tuple_folder_name, "original")
            os.makedirs(original_folder, exist_ok=True)
            for idx, node in self.nodes.items():
                original_program_path = os.path.join(original_folder, f"node_{node.node_id}.py")
                with open(original_program_path, "w") as f:
                    f.write(node.program)
            
            # 2. Save each refactoring in its own subfolder
            for i, refactoring_data in enumerate(sampled_refactorings):
                refactoring_folder = os.path.join(tuple_folder_name, f"refactoring_{i}")
                os.makedirs(refactoring_folder, exist_ok=True)
                
                # Combine and save helper functions (retrieved and new)
                # codebank_str is the retrieved codebank content passed to select_best_refactoring
                retrieved_helpers_code = codebank_str if codebank_str is not None else ""
                new_helpers_code = refactoring_data.get("new_helpers", "")

                combined_helpers_file_path = os.path.join(refactoring_folder, "combined_helpers.py")
                with open(combined_helpers_file_path, "w", encoding='utf-8') as f:
                    if retrieved_helpers_code:
                        f.write("# ==== RETRIEVED HELPER FUNCTIONS ====\n")
                        f.write(retrieved_helpers_code)
                        f.write("\n\n") # Add some separation
                    else:
                        f.write("# ==== RETRIEVED HELPER FUNCTIONS (NONE) ====\n\n")

                    if new_helpers_code:
                        f.write("# ==== NEW HELPER FUNCTIONS ====\n")
                        f.write(new_helpers_code)
                        f.write("\n")
                    else:
                        f.write("# ==== NEW HELPER FUNCTIONS (NONE) ====\n")
                
                # Save refactored programs
                new_programs = refactoring_data.get("new_programs", {})
                for prog_idx, prog_code in new_programs.items():
                    if prog_code:
                        node = self.nodes.get(prog_idx)
                        if node:
                            program_path = os.path.join(refactoring_folder, f"node_{node.node_id}_refactored.py")
                            with open(program_path, "w") as f:
                                f.write(prog_code)
                    # import pdb; pdb.set_trace()
                
                # Save metrics for this refactoring
                metrics_path = os.path.join(refactoring_folder, "metrics.txt")
                with open(metrics_path, "w") as f:
                    # Compute mean pass rates
                    before_pass_values = list(refactoring_data.get("ratio_passed_before", {}).values())
                    mean_pass_rate_before = np.mean(before_pass_values) if before_pass_values else 0.0
                    
                    after_pass_values = list(refactoring_data.get("ratio_passed_after", {}).values())
                    mean_pass_rate_after = np.mean(after_pass_values) if after_pass_values else 0.0
                    
                    f.write(f"Mean Pass Rate Before: {mean_pass_rate_before:.3f}\n")
                    f.write(f"Mean Pass Rate After: {mean_pass_rate_after:.3f}\n")
                    
                    # Write metric values if available
                    if i < len(all_results_tuples):
                        metrics_total, metrics_normalized = all_results_tuples[i]
                        f.write(f"Total LogProb: {metrics_total.get('logprob', 'N/A')}\n")
                        f.write(f"Total Tokens: {metrics_normalized.get('total_tokens', 'N/A')}\n")
                        f.write(f"Cyclomatic Complexity: {metrics_total.get('cyclomatic', 'N/A')}\n")
                        f.write(f"LLOC: {metrics_total.get('lloc', 'N/A')}\n")
                        f.write(f"SLOC: {metrics_total.get('sloc', 'N/A')}\n")
                        
                    
                    # Write normalized loss if available
                    if i < len(total_loss):
                        f.write(f"Total Normalized Loss: {total_loss[i]:.4f}\n")
                    
                    # import pdb; pdb.set_trace()
            
            # 3. Save which refactoring was chosen
            with open(os.path.join(tuple_folder_name, "chosen_refactoring.txt"), "w") as f:
                f.write(f"Selected Refactoring: {final_best_idx}\n")
            # ===== END: Save refactoring data for human evaluation =====
        
        
        return final_best_idx

    async def _run_one_sample(
        self,
        model: str,
        use_self_consistency,
        prompt: str,
        self_consistency_width,
        codebank: CodeBank,
        codebank_str: str,
        node_names,
        results_before,
        logging_path,
        do_retry,
    ):
        print("Not Using model: ", model.model_name, "just using oai_and_parse")
        if use_self_consistency:
            raise NotImplementedError
            print("## Running self-consistency")
            logger.info("Running self-consistency with n = 3...")
            new_helpers, new_programs = self.run_self_consistency(
                model, prompt, n_variants=self_consistency_width
            )
            for p in new_programs:
                if p == "" or p is None:
                    print("!!! New program is empty / None")
                else:
                    print("Program not empty")
        else:
            print("## Getting result from model")
            # prompt has all 3 programs
            new_helpers, new_programs = await self.oai_and_parse(prompt)
            for p in new_programs:
                if p == "" or p is None:
                    print("!!! New program is empty / None")
                else:
                    print("Program not empty")
        print("New programs", new_programs)
        (
            new_codebank,
            success_by_idx,
            new_nodes_by_idx,
            funcs_called_by_idx,
            results_after,
            better_list,
            ratio_passed_before,
            ratio_passed_after,
        ) = self.test_programs(
            codebank,
            new_helpers,
            new_programs,
            node_names,
            results_before,
            logging_path,
        )

        print("Results before retry")

        logger.info(f"success_by_idx coming out of test: {success_by_idx}")
        print("Success_by_idx coming out of test ", success_by_idx)

        if do_retry:
            print("## Retrying failed programs.")
            logger.info("doing retry")
            start_time = time.perf_counter()
            retry_helpers, retry_programs = await self.retry_merge(
                codebank,
                model,
                new_helpers,
                self.nodes,
                new_nodes_by_idx,
                success_by_idx,
                funcs_called_by_idx,
                results_before,
                results_after,
            )
            retry_time = time.perf_counter() - start_time
            logger.info(f"[TIMING] RETRY MERGE TOOK: {retry_time} secs")
            # if retry_helpers is not None:
            #     pdb.set_trace()
            start_time = time.perf_counter()
            (
                new_codebank,
                retry_success_by_idx,
                retry_nodes_by_idx,
                retry_funcs_called_by_idx,
                retry_results_after,
                better_list_after,
                ratio_passed_before_retry,
                ratio_passed_after_retry,
            ) = self.test_programs(
                codebank,
                retry_helpers,
                retry_programs,
                node_names,
                results_before,
                logging_path,
            )
            retry_test_time = time.perf_counter() - start_time
            logger.info(f"[TIMING] RETRY MERGE TEST TOOK: {retry_test_time} secs")
            logger.info(f"retry success_by_idx: {retry_success_by_idx}")
            for idx, success in retry_success_by_idx.items():
                if success and not success_by_idx[idx]:
                    logger.info(f"Retry succeeded for program {idx}, overwriting...")
                    print(f"Retry succeeded for program {idx}, overwriting...")
                    new_programs[idx] = retry_programs[idx]
                    success_by_idx[idx] = success
                    new_nodes_by_idx[idx] = retry_nodes_by_idx[idx]
                    new_nodes_by_idx[idx].is_success = True
                    funcs_called_by_idx[idx] = retry_funcs_called_by_idx[idx]
                    ratio_passed_after[idx] = ratio_passed_after_retry[idx]
        else:
            retry_helpers = None
            retry_programs = None

        print("after retry: success_by_idx", success_by_idx)
        return {
            "new_helpers": new_helpers,
            "new_programs": new_programs,
            "success_by_idx": success_by_idx,
            "new_nodes_by_idx": new_nodes_by_idx,
            "funcs_called_by_idx": funcs_called_by_idx,
            "results_after": results_after,
            "better_list": better_list,
            "ratio_passed_before": ratio_passed_before,
            "ratio_passed_after": ratio_passed_after,
            "retry_helpers": retry_helpers,
            "retry_programs": retry_programs,
            "retry_success_by_idx": retry_success_by_idx,
            "better_list_after": better_list_after if do_retry else {},
        }

    async def merge(
        self,
        codebank,
        model,
        retrieval_method="ask",
        done=[],
        do_retry=True,
        round_added: int = None,
        helpers_first: bool = False,
        use_self_consistency: bool = False,
        self_consistency_width: int = 3,
        sample_k = 1,
    ):
        """Merge nodes using a merge prompt"""
        logger.info("====================================\n\n")
        logging_filename = logger.manager.root.handlers[0].baseFilename
        logging_path = Path(logging_filename.split(".log")[0])
        logging_path.mkdir(parents=True, exist_ok=True)

        print("Running merge")

        # get original results
        node_names = {i: re.sub(" ", "_", node.name) for i, node in self.nodes.items()}

        codebank_code = codebank.str()

        results_before = {}
        for i, node in self.nodes.items():
            results_before[i] = node.execute(
                codebank_code,
                additional_path=logging_path / f"{node.name}_before.py",
            )
        print("results_before", results_before)

        print("Getting relevant codebank info")
        codebank_str, codebank_ids = await self.retrieve_from_codebank(
            retrieval_method, self.nodes, codebank
        )
        print("codebank_ids", codebank_ids)

        if len(codebank_str) > 0:
            codebank_instr = f"\nYou can also choose from the following helper functions:\n{codebank_str}"
        else:
            codebank_instr = ""

        print("Getting prompt")
        queries_and_code = []
        queries_only = []
        if self.task == "scan":
            answer_format_short = [
                "NEW HELPERS (NEVER REDEFINE jump, run, walk, look, turn_left, turn_right! NEVER DEFINE perform_actions):\n"
            ]
            answer_format_long = [
                "NEW HELPERS (NEVER REDEFINE jump, run, walk, look, turn_left, turn_right! NEVER DEFINE perform_actions):\n# Thoughts:\n# 1. The following functions are shared by multiple programs: <function names>\n<code for helper functions>\n"
            ]
        else:
            answer_format_short = []
            answer_format_long = []
        for i, node in self.nodes.items():
            # skip things already done
            if node.node_id in done:
                continue
            queries_and_code.append(
                f"Query {i}: {node.query}\nProgram {i}:\n{self.remove_import(node.program)}"
            )
            queries_only.append(f"Query {i}: {node.query}")
            answer_format_short.append(f"New Program {i}:")
            answer_format_long.append(
                f"NEW PROGRAM {i}:\n# Thoughts:\n# 1. The query asks for: <query intention>\n# 2. <query> can be solved by <components>.\n# 3. I will use/define helper function <function> to <goal>.\n<code for program {i}>\n"
            )

        queries_and_code = "\n".join(queries_and_code)
        answer_format_short = "\n".join(answer_format_short)
        answer_format_long = "\n".join(answer_format_long)

        print("Refactor prompt format")
        # create the merge prompt
        prompt = self.tuple_refactor_prompt.format(
            codebank_instr=codebank_instr,
            queries_and_code=queries_and_code,
            queries_only=queries_only,
            answer_format_short=answer_format_short,
            answer_format_long=answer_format_long,
        )

        logger.info(f"Running {prompt}")
        logger.info(f"Using MODEL: {model.model_name}")
        logger.info(f"program names: {[node.name for i, node in self.nodes.items()]}")

        # TODO: add sample + rerank
        print(
            "============================\n \n Starting sample + rerank \n \n ============================"
        )

        k = sample_k

        tasks = []
        for i in range(k):
            # Create a task for each sample run
            tasks.append(
                self._run_one_sample(
                    model=model,
                    use_self_consistency=use_self_consistency,
                    prompt=prompt,
                    self_consistency_width=self_consistency_width,
                    codebank=codebank,
                    codebank_str=codebank_str,
                    node_names=node_names,
                    results_before=results_before,
                    logging_path=logging_path,
                    do_retry=do_retry,
                )
            )
        start_gather_time = time.perf_counter()
        sampled_refactorings_results = await asyncio.gather(
            *tasks, return_exceptions=True
        )
        gather_time = time.perf_counter() - start_gather_time
        logger.info(f"Finished gathering {k} samples in {gather_time:.2f}s")

        sampled_refactorings = []
        for i, result in enumerate(sampled_refactorings_results):
            if isinstance(result, Exception):
                logger.error(f"Sample task {i} failed with exception: {result}")
            elif result.get("error"):
                 logger.error(f"Sample task {i} reported error: {result['error']}")
            else:
                sampled_refactorings.append(result)

        self.original_nodes_stored = {idx: node for idx, node in self.nodes.items()}
    
        best_idx = await self.select_best_refactoring(sampled_refactorings, codebank_str, client=client, metrics=["logprob"]) 
        
        print("Best idx", best_idx)
        
        if best_idx == -1:
            # TODO: really needs to be wrapped in pydantic
            return (
                {x: False for x in self.nodes.keys()},
                codebank,
                results_before,
                #better_combined, # i dont know what this is
                {x: False for x in self.nodes.keys()},
                sampled_refactorings[0]["ratio_passed_before"],
                sampled_refactorings[0]["ratio_passed_before"],
            )
        best = sampled_refactorings[best_idx]

        new_helpers = best["new_helpers"]
        new_programs = best["new_programs"]
        success_by_idx = best["success_by_idx"]
        new_nodes_by_idx = best["new_nodes_by_idx"]
        funcs_called_by_idx = best["funcs_called_by_idx"]
        results_after = best["results_after"]
        better_list = best["better_list"]
        ratio_passed_before = best["ratio_passed_before"]
        ratio_passed_after = best["ratio_passed_after"]
        retry_helpers = best["retry_helpers"]
        retry_programs = best["retry_programs"]
        retry_success_by_idx = best["retry_success_by_idx"]
        better_list_after = best["better_list_after"]

        better_combined = {}

        for key in better_list:
            if key in better_list_after:
                better_combined[key] = better_list[key] or better_list_after[key]
            else:
                better_combined[key] = better_list[key]

        # compute the number of codebank funcs added
        helpers_split, _ = self.split_helpers(new_helpers)
        if retry_helpers is not None:
            split_retry_helpers, __ = self.split_helpers(retry_helpers)
            helpers_split += split_retry_helpers
        num_codebank_funcs_added = len(helpers_split)

        extracted_helper_defs = defaultdict(set)
        codebank_defs = defaultdict(set)
        for helper in helpers_split:
            # get function name
            name = re.search("(?<=def )([\w_\d]+)\(", helper).group(1)
            for idx in funcs_called_by_idx.keys():
                if name in funcs_called_by_idx[idx]:
                    extracted_helper_defs[idx].add(name)
                if name in codebank_ids:
                    codebank_defs[idx].add(name)

        intersect = set()
        for defs in extracted_helper_defs.values():
            intersect |= defs
        for defs in extracted_helper_defs.values():
            intersect &= defs
        print("number of shared extracted helper functions", len(intersect))
        logger.info(f"number of shared extracted helper functions: {len(intersect)}")

        codebank_intersect = set()
        for defs in codebank_defs.values():
            codebank_intersect |= defs
        for defs in codebank_defs.values():
            codebank_intersect &= defs
        print("number of shared codebank functions", len(codebank_intersect))
        logger.info(f"number of shared codebank functions: {len(codebank_intersect)}")

        # compute the number of codebank funcs used
        num_codebank_funcs_used = 0
        codebank_funcs_used = []
        for idx, funcs in funcs_called_by_idx.items():
            for func in funcs:
                if func in codebank_ids:
                    num_codebank_funcs_used += 1
                    codebank_funcs_used.append(func)

        print("CODEBANK: num codebank funcs added", num_codebank_funcs_added)
        print("CODEBANK: num codebank funcs used", num_codebank_funcs_used)
        print(codebank_funcs_used)

        for idx, new_program in new_programs.items():
            if success_by_idx[idx]:
                print(f"Merge succeeded! for program {idx}, adding new codebank")
                logger.info(f"Merge succeeded! New programs:\n{new_program}")
                logger.info(f"success_by_idx: {success_by_idx}")
                logger.info("Adding new codebank to collection")

                helpers_split, __ = self.split_helpers(new_helpers)
                if (
                    retry_helpers is not None
                    and idx in retry_success_by_idx.keys()
                    and retry_success_by_idx[idx]
                ):
                    logger.info(
                        f"Retry succeeded for program {idx} -- adding helper functions"
                    )
                    split_retry_helpers, __ = self.split_helpers(retry_helpers)
                    helpers_split += split_retry_helpers

                final_helpers = []
                for helper in helpers_split:
                    # get function name
                    name = re.search("(?<=def )([\w_\d]+)\(", helper).group(1)
                    if name in funcs_called_by_idx[idx]:
                        final_helpers.append(helper)
                final_helpers = "\n".join(final_helpers)
                # if the merge succeeded we want to add the helper functions to the collection for retrieval
                # just re-add all the helpers to the collection, which will trigger adding to the collection
                __ = codebank.add_multiple(final_helpers)

                # get imports and add them to the codebank if they're not already there
                # imports are free
                new_imports = [
                    line.lstrip()
                    for line in new_helpers.splitlines()[:10]
                    if "import" in line
                ]
                for new_import in new_imports:
                    if new_import not in codebank._imports:
                        codebank._imports.append(new_import)
                codebank._imports = list(set(codebank._imports))
                # ideally this would be taken care of in add_multiple,
                # but fixing the parsing logic in this function is too bug risky
            else:
                print(f"Merge failed for program {idx}, not adding to codebank")

        for idx in new_nodes_by_idx.keys():
            # update node with the refactored version
            new_nodes_by_idx[idx].is_success = success_by_idx[idx]
            if success_by_idx[idx]:
                print("Updating node with refactored version, it's correct")
                self.nodes[idx] = new_nodes_by_idx[idx]

        # get function names from all programs
        # rules here:
        # 1. if function already exists, and it is used in a failed program, assign a failure
        # 2. if function doesn't already exist and is used in a failed program, don't add it or assign failure
        # 3. if function doesn't exist and program succeeded, then it will have been added above, assign success
        for idx, funcs in funcs_called_by_idx.items():
            # add success info

            for f in funcs:
                # if function already exists, assign a failure
                if f in codebank._codebank.keys():
                    codebank._codebank[f].was_success.append(success_by_idx[idx])
                    codebank._codebank[f].num_programs_used.append(
                        len(funcs_called_by_idx[idx])
                    )
                    # add test case
                    left_tc = CodeContestTestCase(
                        new_nodes_by_idx[idx],
                        self.original_nodes[idx],
                        model,
                        is_correct=success_by_idx[idx],
                    )
                    codebank._codebank[f].test_cases.append(left_tc)

        before_succeeded = results_before

        # TODO: really needs to be wrapped in pydantic
        return (
            success_by_idx,
            codebank,
            before_succeeded,
            better_combined,
            ratio_passed_before,
            ratio_passed_after,
        )

    def get_imports(self, program, no_codebank_imports=False):
        parsed = ast.parse(program)
        imports = [
            node
            for node in parsed.body
            if (isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom))
        ]
        if no_codebank_imports:
            imports = [x for x in imports if "codebank" not in ast.unparse(x)]
        return ast.unparse(imports)

    def test_programs(
        self,
        codebank: CodeBank,
        # new_helpers: List[str],
        new_helpers: str | None,
        new_programs: Dict[int, str | None],
        node_names: Dict[int, str],
        results_before: Dict[int, np.array],
        logging_path: Path,
        round_added: int = None,
    ):
        # add helper functiosn to the codebank and import the codebank
        print("## Testing programs...")
        new_codebank = CodeBank.clone(codebank)

        if new_helpers is not None:
            _ = new_codebank.add_multiple(new_helpers)

        #new_codebank.write_to_file()
        new_codebank_code = new_codebank.str()

        success_by_idx = {}
        new_nodes_by_idx = {}
        funcs_called_by_idx = {}
        results_after = {}
        better_dict = {}

        ratio_passed_before = {}
        ratio_passed_after = {}

        for idx, prog in new_programs.items():
            new_program = self.import_codebank(prog)

            # only occurs at k=1 ablation
            if idx not in self.nodes.keys():
                continue
            # make new node objects
            prog_dict = {
                k: v
                for k, v in self.nodes[idx].__dict__.items()
                if k in CodeContestNode.__init__.__code__.co_varnames
            }
            prog_dict["type"] = "pred"
            prog_dict["program"] = new_program
            node_copy = self.nodes[idx].__class__(**prog_dict)

            # check that the new programs execute to the same result
            result_after = node_copy.execute(
                new_codebank_code,
                additional_path=logging_path / f"{node_names[idx]}_after.py",
            )
            results_after[idx] = result_after
            print(logging_path / f"{node_names[idx]}_after.py")

            success = False
            better = False
            if result_after is not None:
                # passes more unit tests
                success = sum(result_after.passed) >= sum(results_before[idx].passed)
                if sum(result_after.passed) > sum(results_before[idx].passed):
                    better = True

                ratio_passed_before[idx] = sum(results_before[idx].passed) / 10
                ratio_passed_after[idx] = sum(result_after.passed) / 10

            if not success:
                pass

            success_by_idx[idx] = success
            new_nodes_by_idx[idx] = node_copy
            funcs_called = get_func_names(node_copy.program, new_codebank_code)
            funcs_called_by_idx[idx] = funcs_called

            better_dict[idx] = better

            if new_helpers and "nonlocal" in new_helpers:
                # import pdb; pdb.set_trace()
                continue
            if prog and "nonlocal" in prog:
                # import pdb; pdb.set_trace()
                continue
        return (
            new_codebank,
            success_by_idx,
            new_nodes_by_idx,
            funcs_called_by_idx,
            results_after,
            better_dict,
            ratio_passed_before,
            ratio_passed_after,
        )

    # def parse_result(self, text) -> tuple[str, dict[int, str]]:
    def parse_result(self, text) -> tuple[str, dict[int, str]]:
        ids_and_calls = extract_query_and_code(text)

        pattern = r"#\sCodebank\s*\n```(?:python|py)\s*\n(.*?)\n\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        codebank = matches[0]

        return codebank, {int(x["query_number"]): x["code"] for x in ids_and_calls}

    def retrieve_embedding(
        self, retrieval_method, nodes, codebank
    ) -> tuple[str, set[str]]:
        codebank_ids = []
        for idx, node in nodes.items():
            # no more than 20 total examples
            # import pdb; pdb.set_trace()
            if retrieval_method == "problem":
                codebank_ids += codebank.get_relevant(
                    node.query, k=20 // len(self.nodes)
                )
            elif retrieval_method == "solution":
                codebank_ids += codebank.get_relevant(
                    node.description, k=20 // len(self.nodes)
                ) # use short descriptions from LLM
            else:
                raise ValueError(f"Invalid retrieval_method: {retrieval_method}")
        codebank_ids = set(codebank_ids)
        codebank_funcs = [codebank.get(id) for id in codebank_ids]
        print("Codebank_ids", codebank_ids)

        # format codebank, adding in the body of the functions
        codebank_str = [func.summarize() for func in codebank_funcs if func is not None]
        codebank_str = "\n".join(codebank_str)
        return codebank_str, codebank_ids

    async def retrieve_ask(self, nodes, codebank) -> tuple[str, set[str]]:
        codebank_str = codebank.str()
        tasks = []
        prompts = []
        for idx, node in nodes.items():
            prompt = RETRIEVE_CODEBANK_TEMPLATE.format(
                problem_statement=node.query,
                solution_plan=node.description,
                codebank=codebank_str,
            )
            prompts.append(prompt)
            tasks.append(aclient.responses.create(model="o4-mini", input=prompt))
        responses = await asyncio.gather(*tasks)
        helpers = [
            re.findall(
                r"```python(.*)```",
                response.output_text,
                flags=re.DOTALL | re.MULTILINE,
            )[0]
            for response in responses
        ]
        flattened_helpers = "\n".join(helpers)
        function_defs = set(re.findall(r"def (.*)\(", flattened_helpers))

        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            logger.info(f"RETRIEVAL ASK PROMPT {i}")
            logger.info(prompt)
            logger.info(f"RETRIEVAL ASK RESPONSE {i}")
            logger.info(response.output_text)
        return flattened_helpers, function_defs

    async def retrieve_from_codebank(
        self, retrieval_method: str, nodes: list[Node], codebank: CodeBank
    ) -> tuple[str, set[str]]:
        assert retrieval_method in ["ask", "problem", "solution"]

        # empty codebank
        if len(codebank._codebank) == 0:
            return "", []

        if retrieval_method in ["problem", "solution"]:
            return self.retrieve_embedding(retrieval_method, nodes, codebank)

        return await self.retrieve_ask(nodes, codebank)
