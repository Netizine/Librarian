import argparse
import json
import logging
import pdb
import random
import subprocess
from pathlib import Path
import ast

import numpy as np
import torch
from tqdm import tqdm

from program_refactoring.agent.agent import MODEL_DICT, Agent, Example
from program_refactoring.codebank.codebank import CodeBank
from program_refactoring.domains.logos.visual_sim import load_img, vis_compare
from program_refactoring.domains.python.utils import (
    get_func_names as get_python_func_names,
)
from program_refactoring.tree.node import CodeContestNode

logger = logging.getLogger(__name__)
try:
    import sys

    sys.path.insert(0, "third_party/Faithful-COT/source")
    from evaluate.evaluate_answer_acc import (
        extract_gold_answer,
        extract_pred_answer,
        is_correct,
    )
except ModuleNotFoundError:
    logger.warn("Faithful-COT not installed, skipping import")

AGENT_DICT = {"logos": Agent, "python": None}


class FakeModel:
    def __init__(self, model_name):
        self.model_name = model_name


def read_data_from_logdir(logdir, existing=[], task_type="logos"):
    # with open(logdir / "node_dict.pkl", "rb") as f1:
    #     node_dict = pkl.load(f1)

    with open(logdir / "test_cases.jsonl") as f1:
        test_cases = [json.loads(x) for x in f1.readlines()]

    # NOTE (elias): for now: suspend filtering for python because of imports
    get_func_names = get_python_func_names
    do_filter = False
    # TODO: dont we want to filter? i guess dont filter, lose too many help funs

    train_data = []
    for line in test_cases:
        for test_case in line:
            if "pred_node" not in test_case:
                pdb.set_trace()
                continue

            pred_node_data = test_case["pred_node"]
            # TODO: remove unsuccessful nodes? Unclear if you actually want to do that
            if True or pred_node_data["is_success"]:
                functions_used = get_func_names(pred_node_data["program"])
                skip = False
                for func in functions_used:
                    if func not in existing and do_filter:
                        skip = True
                        print(
                            f"removing {pred_node_data['program']} because {func} not in codebank"
                        )
                        break
                if not skip:
                    ex = Example(
                        pred_node_data["node_id"],
                        pred_node_data["query"],
                        program=pred_node_data["program"],
                        provenance="log",
                    )
                    train_data.append(ex)
    return train_data


def read_data_from_project(project_path: Path, task_name: str, ids=None):
    """Load training or evaluation data from a local project directory.

    Every ``.py`` file is treated as a single example.  The function uses the
    module's top level docstring (if present) as the query description and the
    file contents as the program body.  Files located inside ``tests``
    subdirectories or named ``test_*.py`` are ignored.
    """

    examples = []
    for i, file_path in enumerate(project_path.rglob("*.py")):
        if "tests" in file_path.parts or file_path.name.startswith("test_"):
            continue

        code = file_path.read_text()
        try:
            doc = ast.get_docstring(ast.parse(code))
        except SyntaxError:
            # skip files that do not parse correctly
            continue

        description = doc if doc else file_path.stem
        idx = f"{task_name}_{i}"
        ex = Example(idx, description, program=code, provenance="project")
        examples.append(ex)

    print(f"Number of examples from project {project_path}: {len(examples)}")
    return examples


def read_data_from_json(path, task_name, ids=None):
    path = Path(path)

    if path.is_dir():
        return read_data_from_project(path, task_name, ids)

    with open(path) as f1:
        data = [json.loads(x) for x in f1.readlines()]
    examples = []
    for i, line in enumerate(data):
        if ids is not None and "id" in line.keys() and line["id"] not in ids:
            continue
        idx = f"{task_name}_{i}"
        # format the tests into input + output
        tests = {
            "input": line["public_tests"]["input"]
            + line["private_tests"]["input"]
            + line["generated_tests"]["input"],
            "output": line["public_tests"]["output"]
            + line["private_tests"]["output"]
            + line["generated_tests"]["output"],
        }
        ex = Example(
            idx,
            line["description"],
            program=line["solution"],
            provenance="json",
            expected_answer=tests,
        )
        examples.append(ex)

    print("Number of examples from json", len(examples))
    return examples


def rerun_acc_logos(example, path):
    pred_path = path / f"{example.id}_pred.py"
    gold_path = path / f"{example.id}_gold.py"
    # run gold and pred
    out, errs = subprocess.Popen(
        ["python", str(gold_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).communicate()
    if len(errs.decode("utf-8")) > 0:
        return False, None
    gold_img = load_img(str(path / "result_gold.jpg"))

    out, errs = subprocess.Popen(
        ["python", str(pred_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).communicate()
    if len(errs.decode("utf-8")) > 0:
        return False, None
    pred_img = load_img(str(path / "result_pred.jpg"))
    sim = vis_compare(gold_img, pred_img)
    if sim > 0.98:
        return True, None
    return False, None


def rerun_acc_python(example, path):
    pred_path = path / f"{example.id}_pred.py"
    codebank_path = path / "codebank.py"

    if not pred_path.exists():
        return False, None

    pred_program = pred_path.read_text()
    codebank = codebank_path.read_text() if codebank_path.exists() else ""

    node = CodeContestNode(
        example.query,
        pred_program,
        example.expected_answer,
        type="pred",
        name=example.id,
        temp_dir=path,
    )

    result = node.execute(codebank, pred_path)
    return all(result.passed), pred_program


def check_acc_all(agent, examples, path, batch_size=5, rerun=False):
    if agent.task == "logos":
        return check_acc_logo(agent, examples, path, batch_size=batch_size, rerun=rerun)
    elif agent.task == "textcraft":
        return check_acc_textcraft(
            agent, examples, path, batch_size=batch_size, rerun=rerun
        )
    else:
        return check_acc_python(
            agent, examples, path, batch_size=batch_size, rerun=rerun
        )


def check_acc(agent, example, path):
    if agent.task == "logos":
        return check_acc_logo_single(agent, example, path)
    elif agent.task == "textcraft":
        return check_acc_textcraft_single(agent, example, path)
    else:
        return check_acc_python_single(agent, example, path)


def check_acc_textcraft(agent, examples, path, batch_size, rerun):
    print("Checking accuracy...")
    tfs = []
    pred_progs = []
    # ignoring batchsize and rerun
    for example in tqdm(examples):
        pred_res, pred_prog = agent(
            example, retry=args.test_retry, craft_retrieve=args.craft_retrieve
        )
        tfs.append(pred_res == True)
        pred_progs.append(pred_prog)
    return tfs, pred_progs


def check_acc_logo(agent, examples, path, batch_size=5, rerun=False):
    pred_imgs, pred_progs = agent.do_multiple(examples, batch_size, rerun)
    tfs = []
    print("Checking accuracy...")
    for example, pred_img, pred_prog in tqdm(
        zip(examples, pred_imgs, pred_progs), total=len(examples)
    ):
        if type(pred_prog) == int:
            # just ounting tokens
            return pred_img, pred_prog

        if example.program is not None:
            gold_node = agent.node_cls(
                example.query,
                example.program,
                type="gold",
                temp_dir=path,
                name=example.id,
            )
            fname = path / f"{example.id}_gold.py"
            gold_img = gold_node.execute(fname)
        else:
            gold_img = example.expected_answer

        tfs.append(vis_compare(pred_img, gold_img) > 0.98)
    return tfs, pred_progs


def check_acc_logo_single(agent, example, path):
    # check accuracy (without batching)
    pred_img, pred_prog = agent(example)
    if type(pred_prog) == int:
        # just ounting tokens
        return pred_img, pred_prog

    if example.program is not None:
        gold_node = agent.node_cls(
            example.query, example.program, type="gold", temp_dir=path, name=example.id
        )
        fname = path / f"{example.id}_gold.py"
        gold_img = gold_node.execute(fname)
    else:
        gold_img = example.expected_answer

    if vis_compare(pred_img, gold_img) > 0.98:
        return True, pred_prog
    else:
        return False, pred_prog


def check_acc_textcraft_single(agent, example, path):
    pred_res, pred_prog = agent(example)
    answer = example.expected_answer
    return pred_res == answer, pred_prog


def check_acc_python_single(agent, example, path):
    # check accuracy (without batching)
    pred_res, pred_prog = agent(example)

    if example.expected_answer is None:
        # No ground truth available; simply return the generated program
        return True, pred_prog
    elif example.program is not None:
        gold_node = agent.node_cls(
            example.query, example.program, type="gold", temp_dir=path, name=example.id
        )
        fname = path / f"{example.id}_gold.py"
        gold_res = gold_node.execute(fname)

        gold_ans = extract_pred_answer(agent.dataset, gold_res)
    else:
        gold_ans = extract_gold_answer(agent.dataset, example.expected_answer)

    pred_ans = extract_pred_answer(agent.dataset, pred_res)
    return is_correct(agent.dataset, gold_ans, pred_ans), pred_prog


def check_acc_python(agent, examples, path, batch_size=5, rerun=False):
    # check accuracy with batching
    pred_ress, pred_progs = agent.do_multiple(examples, batch_size, rerun=rerun)
    tfs = []
    print("Checking accuracy...")

    for example, pred_res, pred_prog in tqdm(
        zip(examples, pred_ress, pred_progs), total=len(examples)
    ):
        # print("Example", example.query, example.program, example.expected_answer)
        if example.expected_answer is None:
            # No evaluation possible; assume success
            tfs.append(True)
            continue

        if example.program is not None:
            gold_node = agent.node_cls(
                example.query,
                example.program,
                example.expected_answer,
                type="gold",
                temp_dir=path,
                name=example.id,
            )
            fname = path / f"{example.id}_gold.py"
            gold_res = gold_node.execute(fname)
            gold_ans = gold_res.passed
        else:
            gold_ans = extract_gold_answer(agent.dataset, example.expected_answer)

        tfs.append(all(pred_res.passed))
    return tfs, pred_progs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_json_path",
        type=str,
        required=True,
        help="a json file like logo_programs/train_data.jsonl, the output of converting the lisp data",
    )
    parser.add_argument(
        "--train_log_path",
        type=Path,
        required=True,
        help="a path to a logdir from refactoring",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="a path to a json file like logo_programs/test_data.jsonl, the output of converting the lisp data",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="gpt-3.5-turbo",
        help="model name to use",
    )
    parser.add_argument(
        "--task", type=str, required=False, default="logos", help="task to use"
    )
    parser.add_argument(
        "--dataset", type=str, required=False, default="logos", help="task to use"
    )
    parser.add_argument(
        "--test_retry",
        type=bool,
        required=False,
        default=False,
        help="retrial at test time",
    )
    parser.add_argument(
        "--craft_retrieve",
        type=bool,
        default=False,
        help="Use crafting commands in input to retreive codebank functions",
    )
    # parser.add_argument("--codebank_path", type=str, required=False, help="path to generated codebank file", default=None)
    parser.add_argument(
        "--logdir", action="store_true", help="set if loading from a logdir"
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="set to take a 10 example subset (first 5 last 5)",
    )
    parser.add_argument(
        "--single", action="store_true", help="set to take a 1 example subset (first 1)"
    )
    parser.add_argument(
        "--oracle", action="store_true", help="set if using manually-edited codebank"
    )
    parser.add_argument(
        "--out_dir", type=str, default="test_run", help="where to save the results"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="set to overwrite the results"
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="set to re-run existing .py files without doing any prediction",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="set to filter codebank by success and usage",
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--use_docstring_exs",
        action="store_true",
        help="set to use a successful example in the docstring for the codebank",
    )
    parser.add_argument(
        "--use_success",
        action="store_true",
        help="set to provide success rate in the agent prompt for each codebank function",
    )
    parser.add_argument(
        "--use_thought", action="store_true", help="set to use thoughts in ICL examples"
    )
    parser.add_argument(
        "--max_budget", type=int, default=5, help="maximum budget of ICL examples"
    )
    parser.add_argument(
        "--budget_split",
        type=float,
        default=0.6,
        help="percentage of ICL examples that should use functions (if codebank provided)",
    )
    parser.add_argument(
        "--use_infilling",
        action="store_true",
        help="if set to true, use infilling prompt instead of completion prompt (requires Instruction-tuned Llama, not compatible with Python Llama)",
    )
    parser.add_argument(
        "--id_file",
        type=str,
        default=None,
        help="path to a file containing ids of the training subset",
    )
    parser.add_argument("--batch_size", type=int, default=5)
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.id_file is not None:
        ids = json.load(open(args.id_file))
    else:
        ids = None

    print("train json path", args.train_json_path)
    json_train_data = read_data_from_json(args.train_json_path, args.dataset, ids=ids)

    print("####### JSON train data loaded")

    save_path = Path(args.out_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "args.json", "w") as f1:
        to_write = {k: v for k, v in args.__dict__.items()}
        for k, v in to_write.items():
            if isinstance(v, Path):
                to_write[k] = str(v)
        json.dump(to_write, f1)

    codebank = None
    codebank_path = args.train_log_path
    removed = []
    if args.rerun:
        model = FakeModel(args.model_name)
    else:
        model = MODEL_DICT[args.model_name](args.model_name)

    if (
        args.task in ["logos", "python", "textcraft"]
        and (codebank_path / "codebank.py").exists()
    ):
        pypath = codebank_path / "codebank.py"
        jsonpath = codebank_path / "success_info.json"
        testcase_path = codebank_path / "test_cases.jsonl"

        codebank = CodeBank.load(
            pypath,
            jsonpath,
            testcase_path,
            "agent_codebank",
            model,
            run_dir=save_path,
            temp_dir=save_path,
            task=args.task,
        )
        print("Codebank, " + str(codebank_path), "loaded")
        print("PRINTING CODEBANK USAGE STATS")
        import json

        to_save = {}

        print("len of keys in codebank", len(codebank._codebank.keys()))
        for key in codebank._codebank.keys():
            to_save[key] = codebank._codebank[key].compute_success()
            # print(key, codebank._codebank[key].compute_success())
        # with open("after-first-run-stats.json", "w") as f:
        with open("after-second-run-stats.json", "w") as f:
            json.dump(to_save, f, indent=4)

        # write codebank to temp
        codebank.write_to_file()

        # filter codebank by success
        if False and not args.oracle and args.filter:
            removed = codebank.filter(
                success_thresh=0.0, min_usage=2, keep_low_usage=False, round_idx=-1
            )
        print("########### CODEBANK LOADED")

    existing = []
    if codebank is not None:
        existing = [x._name for x in codebank._codebank.values()]

    if args.logdir:
        print("#### LOG train data loaded")
        log_train_data = read_data_from_logdir(
            Path(args.train_log_path), existing, task_type=args.task
        )
        # first do json then overwrite with log
        # for ex in json_train_data:
        #     train_data_by_query[ex.query] = ex
        # for ex in log_train_data:
        #     train_data_by_query[ex.query] = ex

        train_datas = {"test_cases": log_train_data, "train": json_train_data}
    else:
        # no log, just use json data
        print("####### NO LOG DIR, USING JSON DATA")
        train_datas = {"train": json_train_data}

    print("####### LOADING TEST data from", args.test_path)

    test_data = read_data_from_json(args.test_path, args.dataset)

    if args.subset:
        new_data = test_data[0:5] + test_data[-5:]
        test_data = new_data
    if args.single:
        new_data = [test_data[102]]
        test_data = new_data

    print(
        "### TEST DATA IS: ",
        [test_data[i].program for i in range(len(test_data))],
        [test_data[i].expected_answer for i in range(len(test_data))],
    )

    agent_cls = Agent

    print("####### AGENT LOADED")
    agent = agent_cls(
        train_datas,
        model,
        save_path=save_path,
        codebank=codebank,
        task=args.task,
        dataset=args.dataset,
        use_docstring_exs=args.use_docstring_exs,
        use_success=args.use_success,
        use_thought=args.use_thought,
        max_budget=args.max_budget,
        budget_split=args.budget_split,
        infilling=args.use_infilling,
    )

    total_toks = 0
    # load from existing if they exist and are not to be overwritten
    if args.overwrite or args.rerun:
        test_correct = []
        test_incorrect = []
    else:
        if (save_path / "test_correct.jsonl").exists():
            with open(save_path / "test_correct.jsonl") as f1:
                test_correct = [Example(**json.loads(x)) for x in f1.readlines()]
            with open(save_path / "test_incorrect.jsonl") as f1:
                test_incorrect = [Example(**json.loads(x)) for x in f1.readlines()]
        else:
            test_correct = []
            test_incorrect = []

    already_done = [x.id for x in test_correct] + [x.id for x in test_incorrect]
    if args.rerun and args.task == "logos":
        print("Already tested. Rerunning test data...")
        for ex in tqdm(test_data, total=len(test_data)):
            correct, pred = rerun_acc_logos(ex, save_path)
            if correct:
                test_correct.append(ex)
            else:
                test_incorrect.append(ex)

            # checkpoint after each run
            with open(save_path / "rerun_test_correct.jsonl", "w") as f1:
                for ex in test_correct:
                    f1.write(json.dumps(ex.__dict__) + "\n")
            with open(save_path / "rerun_test_incorrect.jsonl", "w") as f1:
                for ex in test_incorrect:
                    f1.write(json.dumps(ex.__dict__) + "\n")

    else:
        print("######### Checking test data...")
        corrects, preds = check_acc_all(
            agent, test_data, save_path, batch_size=args.batch_size, rerun=args.rerun
        )

        for ex, correct, pred in tqdm(
            zip(test_data, corrects, preds), total=len(test_data)
        ):
            print("Checking accuracy...")
            if type(pred) == int:
                total_toks += pred
            if correct:
                ex.program = pred
                test_correct.append(ex)
            else:
                ex.program = pred
                test_incorrect.append(ex)

            # checkpoint after each run
            with open(save_path / "test_correct.jsonl", "w") as f1:
                for ex in test_correct:
                    f1.write(json.dumps(ex.__dict__) + "\n")
            with open(save_path / "test_incorrect.jsonl", "w") as f1:
                for ex in test_incorrect:
                    f1.write(json.dumps(ex.__dict__) + "\n")

        if False:
            # single version
            for ex in tqdm(test_data):
                if ex.id in already_done:
                    continue
                correct, pred = check_acc(agent, ex, save_path)
                if type(pred) == int:
                    total_toks += pred
                if correct:
                    ex.program = pred
                    test_correct.append(ex)
                else:
                    ex.program = pred
                    test_incorrect.append(ex)

                # checkpoint after each run
                with open(save_path / "test_correct.jsonl", "w") as f1:
                    for ex in test_correct:
                        f1.write(json.dumps(ex.__dict__) + "\n")
                with open(save_path / "test_incorrect.jsonl", "w") as f1:
                    for ex in test_incorrect:
                        f1.write(json.dumps(ex.__dict__) + "\n")

    print("correct tests: ", len(test_correct))
    print("num tests: ", len(test_data))
    print(f"Test accuracy: {len(test_correct) / len(test_data) * 100:.2f}")

    with open(save_path / "stats.json", "w") as f1:
        config_dict = args.__dict__
        for k, v in config_dict.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)

        stats_dict = {
            "correct": len(test_correct),
            "incorrect": len(test_incorrect),
            "acc": len(test_correct) / len(test_data),
            "model": args.model_name,
            "config_dict": config_dict,
        }
        json.dump(stats_dict, f1)

    if total_toks > 0:
        print(f"Total tokens in all prompts: {total_toks}")
