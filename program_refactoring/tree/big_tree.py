import asyncio
import json
import logging
import pickle
import re
import time
from pathlib import Path

import chromadb
import networkx as nx
import numpy as np
from tqdm import tqdm

np.random.seed(12)

from program_refactoring.codebank import CodeBank, LogoTestCase  # , TextCraftTestCase
from program_refactoring.codebank.codebank import FUNC_NAME_BY_KEY, CodeBank
from program_refactoring.codebank.test_case import (
    CodeContestTestCase,
    LogoTestCase,
)
from program_refactoring.domains.logos.visual_sim import vis_compare
from program_refactoring.headers import SIMPLE_LOGO_HEADER
from program_refactoring.model.model import Model
from program_refactoring.model.prompts import (
    logo_comment_prompt,
    logo_decompose_prompt,
    python_comment_prompt,
    python_decompose_prompt,
    textcraft_comment_prompt,
    textcraft_decompose_prompt,
)
from program_refactoring.tree.logo_tuple import LogoTuple
from program_refactoring.tree.node import (
    CodeContestNode,
    LogoNode,
    TextCraftNode,
)
from program_refactoring.tree.python_tuple import PythonTuple
from program_refactoring.tree.textcraft_tuple import TextCraftTuple
from program_refactoring.tree.tuple import Tuple
from program_refactoring.utils import cluster_embeddings, create_graph

logger = logging.getLogger(__name__)

NODES_BY_KEY = {
    "python": CodeContestNode,
    "logos": LogoNode,
    "textcraft": TextCraftNode,
}
# TESTCASES_BY_KEY = {"python": PythonTestCase, "logos": LogoTestCase, "textcraft": TextCraftTestCase}
TESTCASES_BY_KEY = {"python": CodeContestTestCase, "logos": LogoTestCase}
CODEBANK_BY_KEY = {"python": CodeBank, "logos": CodeBank, "textcraft": CodeBank}
TUPLES_BY_KEY = {"logos": LogoTuple, "python": PythonTuple, "textcraft": TextCraftTuple}


COMMENT_PROMPTS_BY_KEY = {
    "logos": logo_comment_prompt,
    "python": python_comment_prompt,
    "textcraft": textcraft_comment_prompt,
}

DECOMPOSE_PROMPTS_BY_KEY = {
    "logos": logo_decompose_prompt,
    "python": python_decompose_prompt,
    "textcraft": textcraft_decompose_prompt,
}


class BiggerTree:
    def __init__(
        self,
        graph: nx.DiGraph,
        node_dict: dict,
        model: Model,
        exp_name: str,
        pair_cls=Tuple,
        tc_cls=CodeContestTestCase,
        codebank_cls=CodeBank,
        previous_codebank_path: str | None = None,
        task: str = "logos",
        use_modular: bool = False,
        temp_dir: str = "temp",
        run_dir: str = "temp",
        max_tuple_size: int = 5,
        wide_refactor: bool = False,
        add_comments: bool = False,
        use_explanation: bool = False,
        use_ascii: bool = False,
        curriculum: bool = True,
        use_self_consistency: bool = False,
        self_consistency_width: int = 3,
    ):
        # graph is a networkx graph storing the clustered graph structure
        self.graph = graph
        # node_dict stores all metadata, including the quer, program, description
        self.node_dict = node_dict
        # codebank stores the current bank of helper functions
        if previous_codebank_path is None:
            self.codebank = codebank_cls(
                exp_name,
                model,
                task=task,
                run_dir=run_dir,
                temp_dir=temp_dir,
                use_modular=use_modular,
                use_explanation=use_explanation,
                wide_refactor=wide_refactor,
            )
        else:
            # load from seed codebank
            self.codebank = codebank_cls.load(
                pypath=previous_codebank_path,
                succpath=None,
                testcase_path=None,
                model=model,
                name=exp_name,
                task=task,
                run_dir=run_dir,
                temp_dir=temp_dir,
            )
        self.model = model
        self.pair_cls = pair_cls

        self.exp_name = exp_name

        self.tc_cls = tc_cls
        self.task = task
        self.temp_dir = temp_dir
        self.add_comments = add_comments
        self.use_ascii = use_ascii
        self.curriculum = curriculum
        self.use_self_consistency = use_self_consistency
        self.self_consistency_width = self_consistency_width

        self.get_func_names = FUNC_NAME_BY_KEY[task]
        self.max_tuple_size = max_tuple_size

    def get_tuples(self, max_tuple_size=5):
        if self.task == "scan":
            return self.get_tuples_scan(max_tuple_size)
        else:
            return self.get_tuples_clustered(max_tuple_size)

    def get_tuples_scan(self, max_tuple_size):
        # just return sequentially
        all_node_items = self.node_dict.items()
        # sort by original scan index
        all_node_items = sorted(
            list(all_node_items), key=lambda x: int(x[0].split(":")[1].split("_")[-1])
        )
        tuples = []
        done_tuples = []

        for i in range(0, len(all_node_items), max_tuple_size):
            tuples.append([x[0] for x in all_node_items[i : i + max_tuple_size]])
        final_tuples = []

        for node_id_tuple in tuples:
            # create tuple
            tuple = self.pair_cls(
                {
                    i + 1: self.node_dict[node_id]
                    for i, node_id in enumerate(node_id_tuple)
                },
                task=self.task,
                temp_dir=self.temp_dir,
            )

            tuple_id = "_".join(
                sorted([str(x.node_id) for i, x in tuple.nodes.items()])
            )
            if tuple_id in done_tuples:
                continue
            final_tuples.append(tuple)
            done_tuples.append(tuple_id)
        return final_tuples, done_tuples

    def get_tuples_clustered(self, max_tuple_size):
        print(f"\n--- Starting get_tuples_clustered ---")
        print(f"Max tuple size: {max_tuple_size}")
        initial_node_count = len([n for n in self.graph.nodes() if type(n) not in [int, np.int64]]) # Approximate count
        print(f"Approx. initial relevant nodes in graph: {initial_node_count}") # May differ slightly from your 24 if graph has extras

        # group binary graph into clusters of maximum size max_tuple_size
        tuples = []
        done_nodes = []
        done_queries = []
        # traverse the graph f
        curr_tuple = []
        # sort graph bottom-up topo sort

        # sorted_ids = list(
        #     reversed(list(nx.algorithms.dag.topological_sort(self.graph)))
        # )
        try:
            sorted_ids = list(
                reversed(list(nx.algorithms.dag.topological_sort(self.graph)))
            )
            print(f"Total nodes in sorted order: {len(sorted_ids)}")
        except nx.NetworkXUnfeasible:
            print("ERROR: Graph is not a DAG, topological sort failed!")
            return [], [] # Or handle error appropriately
        except Exception as e:
            print(f"ERROR during topological sort: {e}")  
            return [], []

        for node_id in sorted_ids:
            if type(node_id) in [int, np.int64]:
                # not a content node, skip it
                continue

            node_query = self.node_dict[node_id].query

            if len(curr_tuple) == max_tuple_size:
                # add to tuples
                tuples.append(curr_tuple)
                curr_tuple = []


            # if we hit a root-level node, add and break
            if len(list(self.graph.in_edges(node_id))) == 0:
                curr_tuple.append(node_id)
                tuples.append(curr_tuple)
                curr_tuple = []
                continue


            # otherwise, add to the current cluster
            if node_id not in done_nodes:
                # add to tuple
                curr_tuple.append(node_id)
                done_nodes.append(node_id)
                done_queries.append(node_query)

        if curr_tuple:
            tuples.append(curr_tuple)

        done_tuples = []
        final_tuples = []

        for node_id_tuple in tuples:
            # create tuple
            tuple = self.pair_cls(
                {
                    i + 1: self.node_dict[node_id]
                    for i, node_id in enumerate(node_id_tuple)
                },
                self.task,
                temp_dir=self.temp_dir,
            )
            tuple_id = "_".join(
                sorted([str(x.node_id) for i, x in tuple.nodes.items()])
            )
            if tuple_id in done_tuples:
                continue
            final_tuples.append(tuple)
            done_tuples.append(tuple_id)
        return final_tuples, done_tuples

    def add_tuple_comments(self, tuples_and_idxs):
        """
        Add comments to each chunk of the original code
        """

        def strip_comments(program):
            no_comments = re.sub("#.*$", "", program, flags=re.MULTILINE).strip()
            # delete newline only lines
            no_newline_only = re.sub(
                "(\s*\n)+", "\n", no_comments, flags=re.MULTILINE
            ).strip()
            no_newline_only = re.sub(
                "^[\s\n]+$", "", no_newline_only, flags=re.MULTILINE
            ).strip()
            return no_newline_only

        def clean_comment_response(response):
            response = re.sub(
                "Commented code:", "", response, flags=re.MULTILINE
            ).strip()
            response = re.sub("Comments:", "", response, flags=re.MULTILINE).strip()
            response = response.split("Explanation:")[0]
            return response

        decompose_prompt = DECOMPOSE_PROMPTS_BY_KEY[self.task]
        comment_add_prompt = COMMENT_PROMPTS_BY_KEY[self.task]
        successes = 0
        failures = 0
        logger.info("Adding comments to programs...")
        for pair_idx, tup in enumerate(tqdm(tuples_and_idxs)):
            for k, node in tup.nodes.items():
                if self.task != "textcraft":
                    filled_decompose_prompt = decompose_prompt.format(query=node.query)
                    response = self.model(filled_decompose_prompt)
                if self.task == "textcraft":
                    filled_comment_add_prompt = comment_add_prompt.format(
                        query=node.query, program=node.program
                    )

                else:
                    filled_decompose_prompt = decompose_prompt.format(query=node.query)
                response = self.model(filled_decompose_prompt)

                filled_comment_add_prompt = comment_add_prompt.format(
                    query=node.query, program=node.program, decomposed_query=response
                )
                response = self.model(filled_comment_add_prompt)
                response = clean_comment_response(response)
                # check if code is identical before executing
                before_nocom = strip_comments(node.program)
                after_nocom = strip_comments(response)
                if before_nocom == after_nocom:
                    is_correct = True

                else:
                    # need to execute if not already equal
                    prog_dict = {
                        k: v
                        for k, v in node.__dict__.items()
                        if k in node.__class__.__init__.__code__.co_varnames
                    }
                    prog_dict["type"] = "pred"
                    prog_dict["program"] = response
                    try:
                        node_copy = node.__class__(**prog_dict)
                    except (SyntaxError, IndexError, AttributeError, ValueError):
                        failures += 1
                        continue

                    result0 = node.execute()
                    result1 = node_copy.execute()

                    if self.task == "logos":
                        is_correct = vis_compare(result0, result1) == 1.0
                    elif self.task == "textcraft":
                        is_correct = result0 == result1
                    elif self.task == "python":
                        is_correct = sum(result1.passed) >= sum(result0.passed)

                if is_correct:
                    successes += 1
                    node.program = response
                    tup.nodes[k] = node
                else:
                    failures += 1
            tuples_and_idxs[pair_idx] = tup

        logger.info(
            f"Finished adding comments: success rate: {successes / (successes + failures) * 100:.2f}%"
        )
        return tuples_and_idxs

    def checkpoint(self, i=None):
        log_dir = Path(logger.manager.root.handlers[0].baseFilename.split(".log")[0])
        # networkx save graph
        pickle.dump(self.graph, open(log_dir / "graph.pkl", "wb"))
        # save node dict
        pickle.dump(self.node_dict, open(log_dir / "node_dict.pkl", "wb"))
        # save codebank
        ext = "py"
        self.codebank.save(log_dir / f"codebank.{ext}")
        if i is not None:
            (log_dir / "codebank_history").mkdir(exist_ok=True, parents=True)
            self.codebank.write_to_file(log_dir / "codebank_history" / f"codebank_{i}.{ext}")

    def recursive_resolve(
        self,
        existing_log_dir,
        refactor_every=20,
        filter_every=20,
        task="logos",
        retrieval_method="ask",
        header=SIMPLE_LOGO_HEADER,
        round_added=None,
        do_retry=False,
        redo_done=False,
        helpers_first=True,
        craft_retrieve=False,
        log_dir=None,
        sample_k=None,
    ) -> tuple[bool, str]:
        """Recursively resolve and merge all leaf nodes in the tree but with bigger clusters

        Parameters:
        - existing_log_dir: Path
            If provided, resolving will resume from previous log dir (e.g. if interrupted)
        - refactor_every: int
            Trigger refactor process every n examples
        - filter_ever:
            Trigger filter process every n examples.
            Note that filtering happens before refactoring, so if refactor_every and
            filter_every are equal, filtering only happens once.
        """

        if refactor_every == filter_every:
            filter_every = None

        # read programs from log dir
        # load checkpoint
        if existing_log_dir is not None:
            existing_log_dir = Path(existing_log_dir)
            # copy chromadb dir
            existing_chroma_dir = existing_log_dir / "chromadb"
            if existing_chroma_dir.exists():
                chromadb_dir = (
                    Path(logger.manager.root.handlers[0].baseFilename.split(".log")[0])
                    / "chromadb"
                )
                chromadb_dir.mkdir(exist_ok=True)
                for f in existing_chroma_dir.iterdir():
                    f.rename(chromadb_dir / f.name)

        if existing_log_dir is not None:
            # load node_dict and graph
            with open(existing_log_dir / "node_dict.pkl", "rb") as f1:
                self.node_dict = pickle.load(f1)
            with open(existing_log_dir / "graph.pkl", "rb") as f2:
                self.graph = pickle.load(f2)

            # load codebank
            ext = "py"

            self.codebank = self.codebank.__class__.load(
                existing_log_dir / f"codebank.{ext}",
                existing_log_dir / "success_info.json",
                existing_log_dir / "test_cases.jsonl",
                self.exp_name,
                self.model,
                tc_class=self.tc_cls,
                task=task,
            )

        # add in case first time running
        self.codebank.write_to_file()
        # codebank starts empty

        if redo_done:
            done = []
        else:
            done = [x.node_id for x in self.node_dict.values() if x.is_done]

        tuples, done_tuple_ids = self.get_tuples(max_tuple_size=self.max_tuple_size)

        if self.curriculum:
            sorted_tuples = sorted(
                tuples,
                key=lambda x: len(" ".join([y.query for i, y in x.nodes.items()])),
            )
        else:
            np.random.shuffle(tuples)
            sorted_tuples = tuples

        if self.add_comments and existing_log_dir is None:
            sorted_tuples = self.add_tuple_comments(sorted_tuples)

        # TODO: track improvements in all programs over all tuples
        success_by_tuple = []
        success_by_tuple_before = []
        succcess_running_score = 0
        successs_running_score_before = 0
        better = []
        better_running_score = 0

        tuple_metrics = {}  # maps node_id to list of  "succeeded", "succeeded_before", and "better"

        number_of_programs_in_tuples = sum(
            [len(tup.nodes) for tup in sorted_tuples]
        )
        node_ids_in_tuple_sorted = [
            [x.node_id for x in tup.nodes.values()] for tup in sorted_tuples]

        for pair_idx, tup in enumerate(tqdm(sorted_tuples)):
            print("BIGTREE TUPLE IDX", pair_idx)
            tuple_done = [x.node_id in done for x in tup.nodes.values()]
            if all(tuple_done):
                logger.info("Already done, skipping...")
                continue

            logger.info(f"Iteration {pair_idx} of {len(tuples) - 1}")

            # URGENT: REMOVE True
            if False and filter_every is not None and (pair_idx + 1) % filter_every == 0:
                print("FILTERING CODEBANK")
                logger.info("Filtering codebank...")
                start_time = time.perf_counter()
                removed = self.codebank.filter(pair_idx)
                filter_time = time.perf_counter() - start_time
                logger.info(f"[TIMING] FILTER TOOK: {filter_time} secs")
                logger.info(f"Removed {len(removed)} unsuccessful functions")
            if refactor_every is not None and (pair_idx + 1) % refactor_every == 0:
                print("REFACTORING CODEBANK")
                logger.info("Refactoring codebank...")
                start_time = time.perf_counter()
                self.codebank.refactor(
                    do_filter=False, header=header, task=task, round_added=pair_idx
                )
                refactor_time = time.perf_counter() - start_time
                logger.info(f"[TIMING] REFACTOR TOOK: {refactor_time} secs")

            # merge the pair
            tup_logstr = "\n\n".join(
                [
                    f"# Problem\n\n{x.query}\n\n# Solution\n\n```python\n{x.program}\n```"
                    for i, x in tup.nodes.items()
                ]
            )
            logger.info(f"Attempting to merge pair:\n{tup_logstr}")

            orig_codebank_functions = [x for x in self.codebank._codebank.keys()]

            print("Merging pair ", pair_idx)
            start_time = time.perf_counter()
            (
                merge_succeeded,
                merged_codebank,
                before_succeeded,
                better_combined,
                ratio_before,
                ratio_after,
            ) = asyncio.run(
                tup.merge(
                    self.codebank,
                    self.model,
                    retrieval_method=retrieval_method,
                    done=done,
                    do_retry=do_retry,
                    round_added=pair_idx,
                    helpers_first=helpers_first,
                    # craft_retrieve = craft_retrieve,
                    use_self_consistency=self.use_self_consistency,
                    self_consistency_width=self.self_consistency_width,
                    sample_k = sample_k,
                )
            )

            # add to successful tuple
            tuple_sum = sum(1 for v in merge_succeeded.values() if v is True)
            succcess_running_score += tuple_sum

            tuple_sum_before = sum(
                1 for v in before_succeeded.values() if all(v.passed)
            )
            successs_running_score_before += tuple_sum_before

            better_sum = sum(1 for v in better_combined.values() if v is True)
            better_running_score += better_sum

            success_by_tuple.append(merge_succeeded)
            success_by_tuple.append(before_succeeded)
            better.append(better_combined)

            for i, node in tup.nodes.items():
                node_id = node.node_id
                tuple_metrics[node_id] = {
                    "succeeded": bool(merge_succeeded.get(i, False)),
                    "succeeded_before": all(before_succeeded.get(i).passed)
                    if before_succeeded.get(i)
                    else False,
                    "better": bool(better_combined.get(i, False)),
                    "ratio_before": ratio_before.get(i),
                    "ratio_after": ratio_after.get(i),
                }

            print(type(better_combined), type(merge_succeeded), type(before_succeeded))

            # log_dir = Path(logger.manager.root.handlers[0].baseFilename.split(".log")[0])
            # out_path = log_dir / "tuple_metrics.jsonl"

            # with open(out_path, "w") as f:
            #     for node_id, metrics in tuple_metrics.items():
            #         row = {"node_id": node_id, **metrics}
            #         f.write(json.dumps(row) + "\n")

            merge_time = time.perf_counter() - start_time
            logger.info(f"[TIMING] MERGE TOOK: {merge_time} secs")
            # set the left and right done vars
            for i, n in tup.nodes.items():
                n.is_done = True
                try:
                    n.is_success = merge_succeeded[i]
                except KeyError:
                    n.is_success = False

            # filename in done, skip
            if merge_succeeded == "skipped":
                logger.info("Skipping pair already found in existing log dir")
                continue

            logger.info(f"Merge succeeded is: {merge_succeeded}")

            # if the merge was successful, update the codebank with the helper functions
            merged_codebank_str = merged_codebank.str()
            for i, success in merge_succeeded.items():
                node = tup.nodes[i]
                if success:
                    # update node dict with new program
                    self.node_dict[node.node_id] = node

                functions_used = self.get_func_names(node.program, merged_codebank_str)
                merged_codebank_functions = [
                    k for k in merged_codebank._codebank.keys() if k in functions_used
                ]
                new_functions = set(merged_codebank_functions) - set(
                    orig_codebank_functions
                )
                logger.info(f"Updating codebank. New functions: {new_functions}")
                self.codebank = merged_codebank

            # checkpoint the graph
            self.checkpoint(pair_idx)

        # TODO: sum over the successes over all programs in the tuples
        # success_by_tuple = [item for sublist in success_by_tuple for item in sublist]
        # success_by_tuple = [1 for x in success_by_tuple if x == True]

        # Now, AFTER the for loop finishes:
        log_dir = Path(logger.manager.root.handlers[0].baseFilename.split(".log")[0])
        out_path = log_dir / "tuple_metrics.jsonl"

        with open(out_path, "w") as f:
            for node_id, metrics in tuple_metrics.items():
                row = {"node_id": node_id, **metrics}
                f.write(json.dumps(row) + "\n")

        # succ_rate = sum(1 for sublist in success_by_tuple for item in sublist if item is True)
        print("Success over before: ", succcess_running_score)
        print("Success before: ", successs_running_score_before)
        print("Better running score", better_running_score)
        print("Better running score normalized", better_running_score / 100)
        print("tuple_metrics", tuple_metrics)

        # just pretend successful and no error lol
        # yeah right XD

        return True, ""

    @classmethod
    def from_collection(
        cls,
        collection: chromadb.types.Collection,
        model: Model,
        exp_name: str,
        previous_codebank_path: str | None = None,
        pair_cls_key: str = "python",
        use_modular: bool = False,
        temp_dir: str = "temp",
        max_tuple_size: int = 5,
        wide_refactor: bool = False,
        add_comments: bool = False,
        use_ascii: bool = False,
        curriculum: bool = True,
        use_self_consistency: bool = False,
        self_consistency_width: int = 3,
    ):
        assert pair_cls_key == "python"
        pair_cls = TUPLES_BY_KEY[pair_cls_key]
        tc_cls = TESTCASES_BY_KEY[pair_cls_key]
        node_cls = NODES_BY_KEY[pair_cls_key]
        codebank_cls = CODEBANK_BY_KEY[pair_cls_key]
        # get embeddings and ids from collection
        all_data = collection.get(include=["embeddings", "metadatas"])
        all_embeddings = all_data["embeddings"]
        all_ids = all_data["ids"]
        all_metadata = all_data["metadatas"]

        # cluster embeddings
        clustering, ids = cluster_embeddings(all_embeddings, all_ids)

        # create the tree structure
        graph = create_graph(clustering, ids)

        # check that metadats has short_description
        for metadata in all_metadata:
            if "short_description" not in metadata:
                raise ValueError(
                    f"Metadata does not contain short_description: {metadata}"
                )

        # create node dict
        node_dict = {}
        for node_id in graph.nodes:
            print("Node id: ", node_id)
            if type(node_id) in [int, np.int64]:
                # not a content node, skip it
                continue

            # is a leaf node
            id, name = node_id.split(":")

            idx = ids.index(name)
            metadata = all_metadata[idx]
            query = metadata["query"]
            program = metadata["program"]
            description = metadata["short_description"]
            # ideally the only place we do this...
            tests = json.loads(metadata["tests"])
            # try:
            # meta = metadata["metadata"]
            # if "description" in meta:
            #    description = meta["description"]
            # else:
            #    description = None

            # except KeyError:
            # description = None

            if "textcraft" in name:
                env_idx = name.split("_")[-1]
                node = node_cls(
                    query=query,
                    program=program,
                    tests=tests,
                    type="gold",
                    description=description,
                    metadata=metadata["metadata"],
                    name=name,
                    node_id=node_id,
                    temp_dir=temp_dir,
                )

            else:
                node = node_cls(
                    query=query,
                    program=program,
                    tests=tests,
                    type="gold",
                    description=description,
                    name=name,
                    node_id=node_id,
                    temp_dir=temp_dir,
                )
            node_dict[node_id] = node

        return cls(
            graph,
            node_dict,
            model,
            exp_name,
            pair_cls,
            tc_cls,
            codebank_cls,
            previous_codebank_path=previous_codebank_path,
            task=pair_cls_key,
            use_modular=use_modular,
            temp_dir=temp_dir,
            run_dir=temp_dir,
            max_tuple_size=max_tuple_size,
            wide_refactor=wide_refactor,
            add_comments=add_comments,
            use_ascii=use_ascii,
            curriculum=curriculum,
            use_self_consistency=use_self_consistency,
            self_consistency_width=self_consistency_width,
        )
