# Librarian





This repo includes the official implementation of the **Librarian** method introduced in the paper `Refactoring Code through Library Design`, specifically applied to the CodeContests dataset.

[Žiga Kovačič](https://zzigak.github.io/)\* $^{1}$, [Justin T Chiu](https://justinchiu.netlify.app/)\* $^{2}$, [Celine Lee](https://celine-lee.github.io/)\* $^{1}$, [Wenting Zhao](https://wenting-zhao.github.io/) $^1$, [Kevin Ellis](https://www.cs.cornell.edu/~ellisk/) $^1$.<br>

$^1$ Cornell University, $^2$ Cohere


![Librarian Method Overview](librarybench_fig1.svg)

**Abstract:**
Maintainable and general software allows developers to build robust applications efficiently, yet achieving these qualities often requires refactoring specialized solutions into reusable components. This challenge becomes particularly relevant as code agents become increasingly accurate at solving isolated programming problems. We investigate code agents' capacity to refactor code in ways supporting growth and reusability. We present both a method and a benchmark for refactoring: LIBRARIAN, a sample-and-rerank method for generating reusable libraries, and MINICODE, a benchmark where code agents must "minimize" and refactor multiple independent solutions into a joint library. Compared to state-of-the-art code agents, LIBRARIAN achieves strong results on both compression and correctness on MINICODE, obtaining compression rates 1.6-2x than coding agents while also improving correctness.


---


## Installation

To install the environment for running Librarian on CodeContests do the following:
```bash
# Install the package and dependencies using uv
uv sync

# Or install in development mode
uv pip install -e .

```
## Requirements
-  `OPENAI_API_KEY` environment variable for refactoring
- `TOGETHER_API_KEY` environment variable for computing metrics


## Preparting the Data
You can either use the clustered data in `data/LLM_desription_clusters` that we prepared, or you can use the `data/create_collections` notebook to generate new ones.

### Preprocess Collections
If using our data, preprocess the collections of CodeContests problems by running the following:

```bash
# Only run the following once
chmod +x scripts/codecontests/minicluster/preprocess_LLM_desc_clusters.sh

# Train
./scripts/codecontests/minicluster/preprocess_LLM_desc_clusters.sh
```


## Running Librarian on CodeContests
To run Librarian on our data (after preprocessing) run the following:

```bash
# Only run the following once:
chmod +x scripts/codecontests/minicluster/train_librarian.sh

# Train
./scripts/codecontests/minicluster/train_librarian.sh
```

The refactored programs and the codebank will be saved in `logs/experiments_.../` folder. The metrics of the refactoring will be saved in `train_metrics/librarian_metrics_minicluster${i}_sample_budget_${SAMPLE_K}.txt"`, depending on the parameters you choose in the script.


## Acknowledgement
This codebase is based off of REGAL [![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-md-dark.svg)](https://huggingface.co/papers/2401.16467)
