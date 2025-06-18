from collections import Counter
import argparse
import asyncio
import json
import math # Used by radon, but not directly here. Good to keep if other parts need it.
import os
import re
from pathlib import Path

import numpy as np
from radon.complexity import cc_visit
from radon.raw import analyze
from tenacity import retry, stop_after_attempt, wait_exponential
from together import AsyncTogether
from tqdm.asyncio import tqdm

from program_refactoring.domains.python.utils import get_func_names

client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))

semaphore = asyncio.Semaphore(16)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def compute_logprob_together(text, model):
    async with semaphore:
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


def compute_function_usage_stats(codebank: str, filepaths: list[Path]) -> Counter:
    codebank_functions = re.findall("def (.*?)\(", codebank)
    counter = Counter()
    for filepath in filepaths:
        code = filepath.read_text()
        funcs_called = get_func_names(code, codebank)
        for func in funcs_called:
            if func in codebank_functions:
                counter[func] += 1
    return counter


def compute_code_metrics(code: str):
    raw_metrics = analyze(code)
    complexity_metrics = cc_visit(code)

    return {
        "loc": raw_metrics.loc,  # total lines of code
        "sloc": raw_metrics.sloc,  # source lines of code (non-blank, non-comment)
        "lloc": raw_metrics.lloc,  # logical lines (statements, e.g. `if`, `for`, `return`)
        "comments": raw_metrics.comments,
        "multi": raw_metrics.multi,
        "blank": raw_metrics.blank,
        "cyclomatic": sum(block.complexity for block in complexity_metrics),
    }


async def compute_total_logprob_before(directory, model="Qwen/Qwen2.5-7B-Instruct-Turbo"):
    directory = Path(directory)

    total_logprob = 0.0
    total_tokens = 0
    failed_files = []
    logprobs_dict = {}
    metrics_dict = {}
    tokens_per_program = {}  # <-- New dictionary to store tokens per program

    tasks = []
    program_names = []
    codes = {}

    for file in directory.glob("*_before.py"):
        program_name = file.name.replace("_before.py", "")
        program_names.append(program_name)
        try:
            with open(file, "r") as f:
                code = f.read()
                codes[program_name] = code
            tasks.append(compute_logprob_together(code, model))
        except Exception as e:
            print(f"[ERROR] Failed to process {file.name}: {e}")
            failed_files.append(file.name)

    results = await tqdm.gather(*tasks)

    for program_name, result in zip(program_names, results):
        logprobs, tokens = result
        if logprobs is None:
            failed_files.append(f"{program_name}_before.py")
            # Assign default/nan values if failed
            logprobs_dict[program_name] = float("nan")
            metrics_dict[program_name] = {}
            tokens_per_program[program_name] = 0
            continue

        sum_logprob = sum(logprobs)
        num_tokens = len(logprobs)

        logprobs_dict[program_name] = sum_logprob
        metrics_dict[program_name] = compute_code_metrics(codes[program_name])
        tokens_per_program[program_name] = num_tokens  # <-- Store token count

        total_logprob += sum_logprob
        total_tokens += num_tokens

    # Calculate total metrics safely, handling potential missing keys if logprob failed
    valid_metrics = metrics_dict.values()
    total_lloc = sum(m.get("lloc", 0) for m in valid_metrics)
    total_sloc = sum(m.get("sloc", 0) for m in valid_metrics)
    total_cyclomatic = sum(m.get("cyclomatic", 0) for m in valid_metrics)

    if failed_files:
        print(f"Failed files (before): {failed_files}")

    # Return the new dictionary along with others
    return logprobs_dict, total_logprob, metrics_dict, total_tokens, tokens_per_program


def print_before_summary(
    logprobs_dict: dict,
    metrics_dict: dict,
    total_tokens: int,
    label: str = "Before Summary",
):
    total_logprob = sum(lp for lp in logprobs_dict.values() if np.isfinite(lp)) # Sum only finite logprobs
    total_lloc = sum(m.get("lloc", 0) for m in metrics_dict.values())
    total_sloc = sum(m.get("sloc", 0) for m in metrics_dict.values())
    total_cyclomatic = sum(m.get("cyclomatic", 0) for m in metrics_dict.values())

    print(f"\n=== {label} ===")
    print(f"Total Log Probability: {total_logprob:.2f}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Logical LOC (LLOC): {total_lloc}")
    print(f"Total Source LOC (SLOC): {total_sloc}")
    print(f"Total Cyclomatic Complexity: {total_cyclomatic}")


async def compute_after_logprobs(directory, model = "Qwen/Qwen2.5-7B-Instruct-Turbo"):
    directory = Path(directory)

    codebank_path = directory / "codebank.py"
    with open(codebank_path, "r") as f:
        codebank = f.read()

    codebank_logprobs, codebank_tokens = await compute_logprob_together(codebank, model)
    if codebank_logprobs is None:
        print("[ERROR] Codebank logprob computation failed.")
        # Ensure consistent NaN return for logprobs if it fails
        return {}, float("nan"), 0, {}, 0, compute_code_metrics(codebank) if codebank else {}, {}


    num_codebank_tokens = len(codebank_tokens) if codebank_tokens else 0
    codebank_total_logprob = sum(codebank_logprobs) if codebank_logprobs else float("nan")


    after_logprobs = {}
    after_metrics = {}
    after_tokens_per_program = {}
    failed_files = []

    program_tokens = 0
    tasks = []
    program_names = []
    codes = {}

    for file in directory.glob("*_after.py"):
        program_name = file.name.replace("_after.py", "")
        program_names.append(program_name)
        try:
            with open(file, "r") as f:
                program_code = f.read()
                codes[program_name] = program_code

            full_text = codebank + program_code
            tasks.append(compute_logprob_together(full_text, model))
        except Exception as e:
            print(f"[ERROR] Failed to process {file.name}: {e}")
            failed_files.append(file.name)

    results = await tqdm.gather(*tasks)

    for program_name, result in zip(program_names, results):
        if result is None: # Handles failure in compute_logprob_together if it returns None directly
            failed_files.append(f"{program_name}_after.py")
            after_logprobs[program_name] = float("nan")
            after_metrics[program_name] = {} # Or some default metrics structure
            after_tokens_per_program[program_name] = 0
            continue

        full_logprobs, full_tokens = result
        if full_logprobs is None: # Redundant if above check catches it, but safe
            failed_files.append(f"{program_name}_after.py")
            after_logprobs[program_name] = float("nan")
            after_metrics[program_name] = {}
            after_tokens_per_program[program_name] = 0
            continue
        
        # Ensure num_codebank_tokens does not exceed len(full_logprobs)
        actual_codebank_tokens_in_full = min(num_codebank_tokens, len(full_logprobs))
        program_logprobs = full_logprobs[actual_codebank_tokens_in_full:]
        program_total_logprob = sum(program_logprobs)
        
        actual_codebank_tokens_in_full_tokens = min(num_codebank_tokens, len(full_tokens))
        num_program_tokens = len(full_tokens) - actual_codebank_tokens_in_full_tokens

        program_tokens += num_program_tokens
        after_tokens_per_program[program_name] = num_program_tokens

        after_logprobs[program_name] = program_total_logprob
        after_metrics[program_name] = compute_code_metrics(codes[program_name])

        print(
            f"Processed {program_name}: program logprob = {program_total_logprob:.2f} ({num_program_tokens})"
        )

    # Sum only finite logprobs
    program_logprobs_sum = sum(lp for lp in after_logprobs.values() if np.isfinite(lp))
    # Ensure metrics are accessed safely if a program failed logprob computation
    # and might not have metrics or has empty ones.
    total_lloc = sum(m.get("lloc", 0) for m in after_metrics.values())
    total_sloc = sum(m.get("sloc", 0) for m in after_metrics.values())
    total_cyclomatic = sum(m.get("cyclomatic", 0) for m in after_metrics.values())

    codebank_metrics = compute_code_metrics(codebank)
    if failed_files:
        print(f"Failed files: {failed_files}")

    return (
        after_logprobs,
        codebank_total_logprob,
        num_codebank_tokens,
        after_metrics,
        program_tokens,
        codebank_metrics,
        after_tokens_per_program,
    )


def compute_conditionally_filtered_metrics(
    before_logprobs,
    after_logprobs,
    before_metrics,
    after_metrics,
    before_tokens_per_program,
    after_tokens_per_program,
    ratio_map,
):
    selected_logprobs = {}
    selected_metrics = {}
    selected_tokens = {}
    selected_pass_rates = {}

    # Iterate through all programs that have 'before' logprobs as a baseline
    # or union of all known programs if structure allows.
    # For now, using before_logprobs keys as the set of programs to consider.
    for name in before_logprobs:
        pass_rate_before_entry = ratio_map.get(name, {})
        pass_rate_before = pass_rate_before_entry.get("ratio_before", float("nan"))

        pass_rate_after_entry = ratio_map.get(name, {})
        pass_rate_after = pass_rate_after_entry.get("ratio_after", float("nan"))

        # Ensure pass_rate_before and pass_rate_after are floats for comparison
        # (already handled by the improved load_ratio_metrics, but defense in depth)
        if not isinstance(pass_rate_before, float): pass_rate_before = float("nan")
        if not isinstance(pass_rate_after, float): pass_rate_after = float("nan")


        # Determine if the 'after' version is valid and better or equal
        # np.isfinite(pass_rate_after) checks if pass_rate_after is a "normal" number (not nan, not inf)
        # The comparison pass_rate_after >= pass_rate_before will be False if pass_rate_before is nan.
        use_after = False
        if np.isfinite(pass_rate_after):
            if np.isfinite(pass_rate_before):
                if pass_rate_after >= pass_rate_before:
                    use_after = True
            else: # pass_rate_before is nan, so if pass_rate_after is finite, consider it "better"
                use_after = True # Or, define behavior: if before is nan, do we always take after if finite?
                                 # Current logic `pass_rate_after >= pass_rate_before` makes this path False.
                                 # Let's stick to the original logic: use_after if after is finite AND after >= before
                                 # This means if before is NaN, after will not be chosen by this rule unless after is also NaN (then first clause is false)
                                 # or if after >= NaN (which is false). So, if before is NaN, use_after is effectively False.
                                 # This seems like a reasonable default: if 'before' is unusable, don't automatically prefer 'after'
                                 # without a stronger signal.
                                 # The original line was:
                                 # use_after = np.isfinite(pass_rate_after) and pass_rate_after >= pass_rate_before
                                 # This handles NaN in pass_rate_before correctly (comparison becomes False).

        use_after = np.isfinite(pass_rate_after) and pass_rate_after >= pass_rate_before


        if use_after:
            selected_logprobs[name] = after_logprobs.get(name, float("nan"))
            selected_metrics[name] = after_metrics.get(name, {})
            selected_tokens[name] = after_tokens_per_program.get(name, 0)
            selected_pass_rates[name] = pass_rate_after
        else:
            selected_logprobs[name] = before_logprobs.get(name, float("nan"))
            selected_metrics[name] = before_metrics.get(name, {})
            selected_tokens[name] = before_tokens_per_program.get(name, 0)
            selected_pass_rates[name] = pass_rate_before if np.isfinite(pass_rate_before) else pass_rate_after # Fallback logic for selected_pass_rates

    return selected_logprobs, selected_metrics, selected_tokens, selected_pass_rates


def print_after_summary(
    label: str,
    summary_type: str,  # Explicitly 'conditional' or 'unconditional'
    before_logprobs: dict,
    after_logprobs: dict,
    codebank_logprobs: float,
    codebank_tokens: int,
    after_metrics: dict,
    codebank_metrics: dict,
    token_count_after_programs: int,
    token_count_before_programs: int,
    ratio_map: dict,
    selected_pass_rates: dict = None,
):
    print(f"\n=== {label} ===")

    total_logprob_programs = sum(
        lp for lp in after_logprobs.values() if np.isfinite(lp)
    )
    total_logprob_after = total_logprob_programs + (
        codebank_logprobs if np.isfinite(codebank_logprobs) else 0
    )
    total_tokens_after = token_count_after_programs + codebank_tokens

    total_logprob_before_valid = sum(
        lp for lp in before_logprobs.values() if np.isfinite(lp)
    )

    token_ratio = float("nan")
    logprob_ratio = float("nan")
    if token_count_before_programs > 0 and total_tokens_after is not None : # Make sure total_tokens_after is not None
        token_ratio = total_tokens_after / token_count_before_programs
    
    if (
        total_logprob_before_valid != 0 # Avoid division by zero
        and np.isfinite(total_logprob_before_valid)
        and np.isfinite(total_logprob_after)
    ):
        logprob_ratio = total_logprob_after / total_logprob_before_valid

    print(f"Codebank logprob: {codebank_logprobs:.2f} ({codebank_tokens} tokens)")
    print(
        f"Programs logprob ({label}): {total_logprob_programs:.2f} ({token_count_after_programs} tokens)"
    )
    print(f"Total Log Probability ({label}): {total_logprob_after:.2f}")

    if not np.isnan(logprob_ratio):
        print(f"Logprob ratio ({label}/before): {logprob_ratio:.4f}")
    else:
        print(f"Logprob ratio ({label}/before): N/A")
    if not np.isnan(token_ratio):
        print(f"Token ratio ({label}/before):   {token_ratio:.4f}")
    else:
        print(f"Token ratio ({label}/before):   N/A")

    total_lloc = sum(m.get("lloc", 0) for m in after_metrics.values())
    total_sloc = sum(m.get("sloc", 0) for m in after_metrics.values())
    total_cyclomatic = sum(m.get("cyclomatic", 0) for m in after_metrics.values())

    print("-----------------------------")
    print("Number of programs:", len(after_logprobs))
    print(f"Programs Logical LOC (LLOC) ({label}):   {total_lloc}")
    print(f"Programs Source LOC (SLOC) ({label}):    {total_sloc}")
    print(f"Programs Cyclomatic Complexity ({label}): {total_cyclomatic}")
    print(f"Codebank Logical LOC (LLOC):       {codebank_metrics.get('lloc', 0)}")
    print(f"Codebank Source LOC (SLOC):        {codebank_metrics.get('sloc', 0)}")
    print(f"Codebank Cyclomatic Complexity:  {codebank_metrics.get('cyclomatic', 0)}")
    print(f"Total LLOC ({label}): {total_lloc + codebank_metrics.get('lloc', 0)}")
    print(f"Total SLOC ({label}): {total_sloc + codebank_metrics.get('sloc', 0)}")
    print(
        f"Total Cyclomatic ({label}): {total_cyclomatic + codebank_metrics.get('cyclomatic', 0)}"
    )

    print("=== Pass Rate Summary ===")

    valid_pass_before = []
    for v_entry in ratio_map.values(): # Iterate through ratio_map directly
        before_rate = v_entry.get("ratio_before", float("nan"))
        if np.isfinite(before_rate):
            valid_pass_before.append(before_rate)

    if valid_pass_before:
        print(f"Mean pass rate (original before): {np.mean(valid_pass_before):.4f}")
    else:
        print("Mean pass rate (original before): N/A")

    effective_pass_rates_for_summary = []

    if summary_type == "conditional":
        if selected_pass_rates is None:
            print(
                "[ERROR] summary_type='conditional' requires 'selected_pass_rates' argument."
            )
        else:
            print(
                f"Calculating conditional mean pass rate using {len(selected_pass_rates)} selected program rates."
            )
            for rate in selected_pass_rates.values():
                if np.isfinite(rate): # Ensure rate is finite before adding
                    effective_pass_rates_for_summary.append(rate)

    elif summary_type == "unconditional":
        print(
            "Calculating unconditional mean pass rate using ONLY ratio_after from ratio_map for programs in this summary."
        )
        programs_in_summary = after_logprobs.keys()
        for name in programs_in_summary:
            v_entry = ratio_map.get(name, {})
            after_rate = v_entry.get("ratio_after", float("nan"))
            if np.isfinite(after_rate):
                effective_pass_rates_for_summary.append(after_rate)

    else:
        print(
            f"[ERROR] Invalid summary_type provided: '{summary_type}'. Use 'conditional' or 'unconditional'."
        )

    if effective_pass_rates_for_summary:
        print(
            f"Mean pass rate ({label}): {np.mean(effective_pass_rates_for_summary):.4f}"
        )
    else:
        print(
            f"Mean pass rate ({label}): N/A (no valid rates found for this summary type)"
        )


def package_all_metrics(before_logprobs, after_logprobs, before_metrics, after_metrics):
    result = {}
    all_programs = (
        set(before_logprobs)
        | set(after_logprobs)
        | set(before_metrics)
        | set(after_metrics)
    )

    for program in all_programs:
        before_lp = before_logprobs.get(program, float("nan"))
        after_lp = after_logprobs.get(program, float("nan"))

        result[program] = {
            "logprobs": {
                "before": before_lp,
                "after": after_lp,
            },
            "metrics": {
                "before": before_metrics.get(program, {}),
                "after": after_metrics.get(program, {}),
            },
        }

    return result


def load_ratio_metrics(directory: Path) -> dict:
    ratio_map = {}
    jsonl_path = directory / "tuple_metrics.jsonl"
    if not jsonl_path.exists():
        print("[WARNING] tuple_metrics.jsonl not found.")
        return ratio_map

    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f): # Add line number for better error messages
            try:
                entry = json.loads(line)
                node_id_val = entry.get("node_id") 
                if not isinstance(node_id_val, str): # Check if node_id is a string
                    print(f"[WARNING] Skipping entry with invalid or missing 'node_id' in {jsonl_path} (line {i+1}): {entry}")
                    continue
                
                name = node_id_val
                if ":" in node_id_val:
                    _, name = node_id_val.split(":", 1)
                
                # Get raw values for ratios
                rb_raw = entry.get("ratio_before")
                ra_raw = entry.get("ratio_after")

                # Convert to float or float("nan")
                # This handles numbers, None (from JSON null), and other types gracefully.
                ratio_before_val = float(rb_raw) if isinstance(rb_raw, (int, float)) else float("nan")
                ratio_after_val = float(ra_raw) if isinstance(ra_raw, (int, float)) else float("nan")
                
                ratio_map[name] = {
                    "ratio_before": ratio_before_val,
                    "ratio_after": ratio_after_val,
                }
            except json.JSONDecodeError:
                print(f"[WARNING] Skipping malformed JSON line in {jsonl_path} (line {i+1}): {line.strip()}")
            except Exception as e: # Catch any other unexpected error during entry processing
                print(f"[WARNING] Error processing entry in {jsonl_path} (line {i+1}): {entry}. Error: {e}")
    return ratio_map


def parse_args():
    parser = argparse.ArgumentParser(description="Process metrics for experiments")
    experiments = {
        "ask": "experiment_2025-04-29_17_32_task_python_dataset_code_contests_refactor_5_filter_20_redo_done_False_comments_False_helpers_second_True_ret_ask",
        "problem": "experiment_2025-04-29_17_32_task_python_dataset_code_contests_refactor_5_filter_20_redo_done_False_comments_False_helpers_second_True_ret_problem",
        "solution": "experiment_2025-04-29_16_52_task_python_dataset_code_contests_refactor_5_filter_20_redo_done_False_comments_False_helpers_second_True_ret_solution",
    }
    parser.add_argument(
        "--experiment",
        choices=list(experiments.keys()),
        default="ask",
        help="Short name of the experiment to process",
    )
    parser.add_argument(
        "--base_dir", default="logs", help="Base directory for experiments"
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Override the experiment path with a direct log directory (takes precedence over experiment/base_dir)",
    )
    args = parser.parse_args()
    return args, experiments


async def main():
    args, experiments = parse_args()

    if args.log_dir:
        log_dir = args.log_dir
        print(f"Using provided log directory: {log_dir}")
    else:
        experiment_name = experiments[args.experiment]
        log_dir = f"{args.base_dir}/{experiment_name}"
        print(f"Experiment: {experiment_name}")
    log_dir_path = Path(log_dir)

    if not log_dir_path.exists() or not log_dir_path.is_dir():
        print(f"[ERROR] Log directory not found: {log_dir_path}")
        return

    codebank_file = log_dir_path / "codebank.py"
    if codebank_file.exists():
        codebank = codebank_file.read_text()
        program_paths = list(log_dir_path.glob("*_after.py"))
        usage_counter = compute_function_usage_stats(codebank, program_paths)
        print("Usage counter")
        print(usage_counter)
    else:
        print(f"[WARNING] codebank.py not found in {log_dir_path}. Usage stats will be empty.")
        codebank = "" # Define codebank as empty string if not found
        usage_counter = Counter()


    ratio_map = load_ratio_metrics(log_dir_path)

    (
        before_logprobs_dict,
        _, # total_logprob_before (not directly used later, sum is recalculated)
        before_metrics,
        before_tokens,
        before_tokens_per_program,
    ) = await compute_total_logprob_before(log_dir)
    (
        after_logprobs_dict,
        codebank_logprobs,
        codebank_tokens,
        after_metrics,
        after_tokens,
        codebank_metrics,
        after_tokens_per_program,
    ) = await compute_after_logprobs(log_dir)

    filtered_logprobs, filtered_metrics, filtered_tokens_dict, filtered_pass_rates = (
        compute_conditionally_filtered_metrics(
            before_logprobs_dict,
            after_logprobs_dict,
            before_metrics,
            after_metrics,
            before_tokens_per_program,
            after_tokens_per_program,
            ratio_map,
        )
    )
    conditional_tokens = sum(filtered_tokens_dict.values())

    print_before_summary(
        before_logprobs_dict, before_metrics, before_tokens, label="Before Summary"
    )

    print_after_summary(
        label="Unconditional After Summary",
        summary_type="unconditional",
        before_logprobs=before_logprobs_dict,
        after_logprobs=after_logprobs_dict,
        codebank_logprobs=codebank_logprobs,
        codebank_tokens=codebank_tokens,
        after_metrics=after_metrics,
        codebank_metrics=codebank_metrics,
        token_count_after_programs=after_tokens,
        token_count_before_programs=before_tokens,
        ratio_map=ratio_map,
    )

    print_after_summary(
        label="Conditional After Summary (Pass Rate ≥ Before)",
        summary_type="conditional",
        before_logprobs=before_logprobs_dict,
        after_logprobs=filtered_logprobs,
        codebank_logprobs=codebank_logprobs,
        codebank_tokens=codebank_tokens,
        after_metrics=filtered_metrics,
        codebank_metrics=codebank_metrics,
        token_count_after_programs=conditional_tokens,
        token_count_before_programs=before_tokens,
        ratio_map=ratio_map,
        selected_pass_rates=filtered_pass_rates,
    )

    full_metrics = package_all_metrics(
        before_logprobs_dict, after_logprobs_dict, before_metrics, after_metrics
    )
    output_json_path = log_dir_path / "full_logprobs_and_metrics.json"
    with open(output_json_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"Full metrics saved to {output_json_path}")


if __name__ == "__main__":
    asyncio.run(main())
