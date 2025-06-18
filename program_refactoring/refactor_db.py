import argparse
import datetime
import json
import logging
import os
import subprocess
from pathlib import Path

from program_refactoring.codebank import CodeBank
from program_refactoring.headers import (
    LOGO_HEADER,
    PYTHON_HEADER,
    SIMPLE_LOGO_HEADER,
    TEXTCRAFT_HEADER,
)
from program_refactoring.model.openai_model import OpenAIModel

# from program_refactoring.tree.tree import Tree
from program_refactoring.tree.big_tree import BiggerTree
from program_refactoring.utils import load_from_dir

# from program_refactoring.logger import prog_logger

# logger = prog_logger

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

logger = logging.getLogger(__name__)


def gpt_resolve(
    logo_collection_path,
    reset_codebank: bool,
    previous_codebank_path: str | None = None,
    existing_log_dir=None,
    filter_every=5,
    refactor_every=200,
    header=LOGO_HEADER,
    simple_header=SIMPLE_LOGO_HEADER,
    pair_cls_key="python",
    dataset="codecontests",
    tree_cls_key="tree",
    use_modular=False,
    temp_dir="temp",
    args=None,
):
    logger.info(f"Resolving from {logo_collection_path}")
    log_dir = Path(logger.manager.root.handlers[0].baseFilename.split(".log")[0])
    log_dir.mkdir(exist_ok=True, parents=True)
    print(f"LOG_DIR_PATH:{log_dir}")
    with open(log_dir / "args.json", "w") as f1:
        git_info = subprocess.check_output(["git", "log", "-1"]).decode("utf-8")
        args.__dict__["git_info"] = git_info
        json.dump(args.__dict__, f1, indent=4)

    print("Pair_cls", pair_cls_key)
    collection = load_from_dir(logo_collection_path, "cc_python")
    model = OpenAIModel(args.model_name)
    exp_name = f"test_{pair_cls_key}"
    print("BigTree from collections.")
    tree = BiggerTree.from_collection(
        collection=collection,
        model=model,
        exp_name=exp_name,
        previous_codebank_path=previous_codebank_path,
        pair_cls_key=pair_cls_key,
        use_modular=use_modular,
        temp_dir=temp_dir,
        max_tuple_size=args.max_tuple_size,
        wide_refactor=args.wide_refactor,
        add_comments=args.add_comments,
        use_ascii=args.use_ascii,
        curriculum=not args.no_curriculum,
        use_self_consistency=args.use_self_consistency,
        self_consistency_width=args.self_consistency_width,
    )
    # add header to codebank
    # TODO (elias): make this part of the tree constructor
    tree.codebank._header = header

    # so the functions dont carry over from previous experiments
    if reset_codebank:
        tree.codebank.reset()

    # save initial codebank
    (log_dir / "codebank_history").mkdir(exist_ok=True, parents=True)
    tree.codebank.write_to_file(log_dir / "codebank_history" / "initial_codebank.py")

    resolved = False
    attempts = 0

    while not resolved:
        print("Starting resolve")
        # auto-restart if something breaks
        if attempts == 0 and existing_log_dir is None:
            # start from scratch
            log_dir_to_run = None
        elif attempts == 0 and existing_log_dir is not None:
            # resume from existing log dir but in a new log dir
            log_dir_to_run = existing_log_dir
        else:
            # resume within while loop from current log_dir
            log_dir_to_run = log_dir

        print("Recursive resolve ", attempts)
        resolved, error = tree.recursive_resolve(
            log_dir_to_run,
            filter_every=filter_every,
            refactor_every=refactor_every,
            task=args.task,
            retrieval_method=args.retrieval_method,
            header=simple_header,
            do_retry=args.do_retry,
            redo_done=args.redo_done,
            helpers_first=not args.helpers_second,
            log_dir=log_dir,
            sample_k=args.sample_k,
        )
        if not resolved:
            print(error)
        attempts += 1

        if attempts > 3:
            break

    # save filtered codebank
    filtered_codebank = CodeBank.clone(tree.codebank)
    filtered_codebank.collection = tree.codebank.collection # not going to use it either way
    filtered_codebank.filter(round_idx=1000, min_usage=3) # filter everything that is used < 3x
    filtered_codebank.write_to_file(log_dir / "filtered_codebank.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection_path",
        type=str,
        required=True,
        help="path to collection to resolve",
    )
    parser.add_argument(
        "--reset_codebank",
        action="store_true",
        help="reset the chroma db for the helper functions",
    )
    parser.add_argument(
        "--previous_codebank_path",
        default=None,
        help="Previous codebank path (default: None)",
    )
    parser.add_argument(
        "--filter_every",
        type=int,
        default=20,
        help="filter the codebank every n examples",
    )
    parser.add_argument(
        "--refactor_every",
        type=int,
        default=200,
        help="refactor the codebank every n examples",
    )
    parser.add_argument(
        "--existing_log_dir",
        type=str,
        required=False,
        default=None,
        help="resume from an existing dir ",
    )
    parser.add_argument(
        "--task", type=str, required=False, default="logos", help="task to resolve"
    )
    parser.add_argument(
        "--dataset", type=str, default="logos", help="dataset to do (e.g. GSM8K, etc)"
    )
    parser.add_argument(
        "--tree_type",
        type=str,
        default="tree",
        help="tree type to use (e.g. tree, big_tree)",
        choices=["tree", "big_tree"],
    )
    parser.add_argument(
        "--retrieval_method",
        choices=["problem", "solution", "ask"],
        default="ask",
        help="",
    )
    parser.add_argument(
        "--use_modular", action="store_true", help="use modular refactoring prompt"
    )
    parser.add_argument(
        "--temp_dir", type=str, default="temp", help="temp dir to store codebank"
    )
    parser.add_argument(
        "--max_tuple_size",
        type=int,
        default=5,
        help="max tuple size to use for modular refactoring",
    )
    parser.add_argument(
        "--do_retry", action="store_true", help="retry failed refactorings immediately"
    )
    parser.add_argument(
        "--wide_refactor",
        action="store_true",
        help="use wide beam (5) for refactoring single functions, take best of 5",
    )
    parser.add_argument(
        "--redo_done", action="store_true", help="do another pass of refactoring"
    )
    # parser.add_argument("--model_name", type=str, default="openai/gpt-4o", help="model to use")
    # parser.add_argument("--model_name", type=str, default="openai/o3-mini", help="model to use")
    parser.add_argument(
        "--model_name", type=str, default="gpt-3.5-turbo", help="model to use"
    )
    # parser.add_argument("--model_name", type=str, default="gpt-4o", help="model to use")
    parser.add_argument(
        "--add_comments",
        action="store_true",
        help="set to true to comment original code before refactoring",
    )
    parser.add_argument(
        "--use_ascii",
        action="store_true",
        help="set to true if using ascii feedback for retry in logos ",
    )
    parser.add_argument(
        "--helpers_second",
        action="store_true",
        help="set to true if using helper_second for logos",
    )
    parser.add_argument(
        "--no_curriculum",
        action="store_true",
        help="set to true to ablate curriculum sorting",
    )
    parser.add_argument(
        "--use_self_consistency",
        action="store_true",
        help="set to true to use self-consistency in refactoring",
    )
    parser.add_argument("--self_consistency_width", type=int, default=3)
    
    parser.add_argument("--sample_k", type=int, default=1, help="sample number (k) for sample + rerank in merge")
    
    args = parser.parse_args()

    if args.task == "logos":
        header = LOGO_HEADER
        simple_header = SIMPLE_LOGO_HEADER
        raise NotImplementedError

    elif args.task == "textcraft":
        header = TEXTCRAFT_HEADER
        simple_header = TEXTCRAFT_HEADER
        raise NotImplementedError

    elif args.task == "python":
        header = PYTHON_HEADER
        simple_header = PYTHON_HEADER
    else:
        raise ValueError(f"Unknown task {args.task}")
    # todo: codecontests header?

    option_str = (
        f"task_{args.task}_dataset_{args.dataset}_refactor_{args.refactor_every}_"
        + f"filter_{args.filter_every}_redo_done_{args.redo_done}_comments_"
        + f"{args.add_comments}_helpers_second_{args.helpers_second}_"
        + f"ret_{args.retrieval_method}"
    )
    logname = f"logs/experiment_{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')}_{option_str}.log"

    os.makedirs(os.path.dirname(logname), exist_ok=True)

    logging.basicConfig(
        filename=logname,
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    gpt_resolve(
        args.collection_path,
        reset_codebank=args.reset_codebank,
        previous_codebank_path=args.previous_codebank_path,
        existing_log_dir=args.existing_log_dir,
        filter_every=args.filter_every,
        refactor_every=args.refactor_every,
        header=header,
        simple_header=simple_header,
        pair_cls_key=args.task,
        tree_cls_key=args.tree_type,
        dataset=args.dataset,
        use_modular=args.use_modular,
        temp_dir=args.temp_dir,
        args=args,
    )
