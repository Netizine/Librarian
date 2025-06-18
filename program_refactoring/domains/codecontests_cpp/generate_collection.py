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
        "--name", type=str, help="name of the dataset", default="cc_python"
    )
    args = parser.parse_args()

    with open(args.data_file) as f1:
        data = [json.loads(x) for x in f1.readlines()]

    names = []
    queries = []
    codes = []
    ids = []
    alltests = []

    for i, d in enumerate(data):
        name = d["name"]
        query = d["description"]
        code = d["solution"]
        id = i
        tests = {
            "input": d["public_tests"]["input"]
            + d["private_tests"]["input"]
            + d["generated_tests"]["input"],
            "output": d["public_tests"]["output"]
            + d["private_tests"]["output"]
            + d["generated_tests"]["output"],
        }

        names.append(name)
        queries.append(query)
        codes.append(code)
        ids.append(f"{args.name}_{id}")
        alltests.append(json.dumps(tests))

    # embed solutions
    get_and_save_embeddings(
        names,
        queries,
        codes,
        ids,
        alltests,
        persist_directory=args.output_dir,
        name=args.name,
        use_query=False,
    )
