import asyncio
import datasets
import json
from collections import defaultdict
import numpy as np
import tiktoken
from openai import AsyncOpenAI
import tqdm
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

client = AsyncOpenAI()
tokenizer = tiktoken.get_encoding("cl100k_base")


LABEL_TEMPLATE = """Analyze the provided programming solution in the following structured format:
1. One-sentence summary: Provide a clear, concise description of the fundamental problem the solution addresses, without reference to specific domain elements.
2. Core algorithmic approach: Identify the fundamental algorithm(s) used (e.g., BFS, DFS, dynamic programming, two pointers, greedy, divide and conquer, binary search, segment trees, union-find, sliding window, Dijkstra's, Floyd-Warshall, topological sort, KMP, Z-algorithm, MST, LCA, Fenwick tree, sparse table, suffix array, trie, sweep line, meet-in-the-middle, convex hull).
3. Reusable components: List 2-3 key functions, data structures, or patterns that could be extracted and reused in similar problems.
Keep your analysis precise, technical, and concise - focus on algorithmic insights rather than implementation details.

# Solution
```python
{solution}
```"""


def count_tokens(text):
    return len(tokenizer.encode(text))


def get_problems(xs, max_solutions):
    problems = []
    for i, x in enumerate(xs):
        name = x["name"]
        stuff = x["solutions"]
        languages = stuff["language"]
        solutions = stuff["solution"]
        description = x["description"]
        selected_solutions = []
        for lang, sol in zip(languages, solutions):
            # if lang == 2 and count_tokens(description) + count_tokens(sol) <= 8192:
            # get python3
            #if lang == 3 and count_tokens(sol) <= 8192:
            if lang == 3:
                selected_solutions.append(sol)
            if len(selected_solutions) >= max_solutions:
                break
        problems.append(
            dict(
                problem_id=f"{i}-{name}",
                question=x["description"],
                tests=[
                    dict(stdin=stdin, stdout=stdout)
                    for stdin, stdout in zip(
                        x["public_tests"]["input"] + x["private_tests"]["input"] + x["generated_tests"]["input"],
                        x["public_tests"]["output"] + x["private_tests"]["output"] + x["generated_tests"]["output"],
                    )
                ],
                source="codeforces",
                difficulty=x["cf_rating"],
                human_solutions=selected_solutions,
                original_code=None,
                language="python",
            )
        )
    return problems


def generate_descriptions():
    dataset = datasets.load_dataset("deepmind/code_contests")
    # train = dataset["train"]
    train = dataset["train"]

    skills = sorted(set([skills for ex in train["cf_tags"] for skills in ex]))
    skill = "graphs"

    idxs_with_tags = [(i, x) for i, x in enumerate(train["cf_tags"]) if x]
    #problems = [train[i] for i, x in idxs_with_tags if skill in x]
    problems = train

    examples = get_problems(problems, max_solutions=1)
    # concatenate prompt and solution
    ids = []
    texts = []
    fulltexts = []
    tasks = []
    problems_with_descriptions = []
    for i, problem in enumerate(tqdm.tqdm(examples, desc="Getting solution descriptions")):
        for solution in problem["human_solutions"]:
            ids.append(problem["problem_id"])
            texts.append(solution)
            fulltext = LABEL_TEMPLATE.format(
                solution=solution
            )
            fulltexts.append(fulltext)

    async def gather_results():
        # Create a semaphore with a limit of 32 concurrent tasks
        semaphore = asyncio.Semaphore(32)
        
        async def bounded_request(fulltext):
            # Use the semaphore as a context manager to limit concurrency
            async with semaphore:
                return await client.chat.completions.create(
                    model="o4-mini",
                    messages=[{"role": "user", "content": fulltext}],
                    reasoning_effort="low",
                )
        
        # Create tasks with the bounded request function
        tasks = [bounded_request(fulltext) for fulltext in fulltexts]
        return await tqdm_asyncio.gather(*tasks, desc="Getting labels")

    responses = asyncio.run(gather_results())
    descriptions = [response.choices[0].message.content for response in responses]

    with open("data/codecontests_all_train_descriptions_detailed.json", "w") as f:
        json.dump(list(zip(ids, descriptions)), f)


def main():
    generate_descriptions()


if __name__ == "__main__":
    main()
