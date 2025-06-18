import argparse
import json
import os

# from num2words import num2words
from program_refactoring.utils import get_and_save_embeddings

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        help="path to the jsonl file containing the dataset",
        default="python_data/codecontests/train_100_python_single.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="path to the output directory",
        default="python_data/codecontests/my_vectordb",
    )
    parser.add_argument(
        "--name", type=str, help="name of the dataset", default="python"
    )
    args = parser.parse_args()

    with open(args.data_file) as f1:
        data = [json.loads(x) for x in f1.readlines()]

    queries = []
    codes = []
    ids = []

    for i, d in enumerate(data):
        query = d["description"]
        code = d["solution"]
        id = i

        queries.append(query)
        codes.append(code)
        ids.append(f"{args.name}_{id}")

    # embed solutions
    get_and_save_embeddings(
        queries,
        codes,
        ids,
        persist_directory=args.output_dir,
        name=args.name,
        use_query=False,
    )
