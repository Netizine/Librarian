import ast
import json
import os
import pdb
import re
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

from program_refactoring.domains.logos.visual_sim import load_img
from program_refactoring.headers import LOGO_HEADER, TEXTCRAFT_HEADER


class ExecutionResult(BaseModel):
    passed: list[bool]
    test_inputs: list[str]
    test_outputs: list[str]
    predicted_outputs: list[str]
    stderrs: list[str]


def execute_main_codebank_inputs(
    main: str,
    codebank: str,
    inputs: list[str],
) -> tuple[list[str], list[str]]:
    """
    Execute a program consisting of multiple files where one imports from another.

    Args:
        files_dict (dict): Dictionary where keys are filenames and values are file contents
        main_file (str): The main file to execute
        interpreter (str): The interpreter to use (default: 'python')

    Returns:
        subprocess.CompletedProcess: The result of the execution
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Write all files to the temporary directory
        (temp_path / "main.py").write_text(main)
        (temp_path / "codebank.py").write_text(codebank)

        # Execute the main file on each input serially
        outputs = []
        errors = []
        for inp in inputs:
            try:
                p = subprocess.run(
                    ["python", "main.py"],
                    cwd=temp_dir,  # Set working directory so imports work
                    capture_output=True,
                    text=True,
                    check=False,
                    input=inp,
                    timeout=10, # maybe 30 later?
                    env={**os.environ, "PYTHONPATH": str(temp_path)},
                )
                outputs.append(p.stdout)
                errors.append(p.stderr)
            except subprocess.TimeoutExpired:
                outputs.append("")
                errors.append("timeout")

        return outputs, errors


class Node:
    def __init__(
        self,
        query,
        program,
        type="gold",
        name=None,
        description=None,
        metadata=None,
        node_id=None,
        temp_dir=None,
        is_success=False,
        is_done=False,
    ):
        self.query = query
        self.query_tok = re.split("\s+", query)
        self.program = program
        self.description = description
        self.metadata = metadata
        self.name = name
        self.node_id = node_id
        self.is_done = is_done
        self.is_success = is_success
        self.type = type
        self.temp_dir = temp_dir

        self.is_leaf = (
            self.query is not None
            and self.program is not None
            and len(self.query) > 0
            and len(self.program) > 0
        )

    def to_json(self):
        toret = self.__dict__
        try:
            toret["temp_dir"] = str(toret["temp_dir"])
        except KeyError:
            pass
        return toret

    @classmethod
    def from_json(cls, json):
        # filter json
        jsond = {
            k: v for k, v in json.items() if k in cls.__init__.__code__.co_varnames
        }
        return cls(**jsond)

    def is_returning(self, program):
        raise NotImplementedError

    def wrap_program(self, program):
        raise NotImplementedError

    def execute(self, additional_path):
        raise NotImplementedError


class PythonNode(Node):
    def __init__(
        self,
        query,
        program,
        type="gold",
        name=None,
        description=None,
        metadata=None,
        node_id=None,
        temp_dir=None,
        is_success=False,
        is_done=False,
    ):
        super().__init__(
            query,
            program,
            type=type,
            name=name,
            description=description,
            metadata=metadata,
            node_id=node_id,
            temp_dir=temp_dir,
            is_success=is_success,
            is_done=is_done,
        )

        # check to see if it executes
        if not self.is_returning(program.strip()):
            self.exec_program = self.wrap_program(program.strip())
        else:
            self.exec_program = program

        if temp_dir is None:
            self.temp_dir = "temp"
        else:
            self.temp_dir = Path(temp_dir)

    def is_returning(self, program):
        # check if final part of program is an invocation
        try:
            last_expr = ast.parse(program).body[-1]
        except (SyntaxError, IndexError):
            return False
        if isinstance(last_expr, ast.Expr):
            return True
        return False

    def wrap_program(self, program):
        # remove all punc and replace space with underscore
        func_name = re.sub(r"\s+", "_", self.name)
        func_name = re.sub(r"[^\w\s]", "", func_name)

        # add a def to the start
        header = f"def {func_name}():\n    _=0\n"
        # identify all code that isn't part of a function
        # and add it to the function
        header_parsed = ast.parse(header)

        parsed = ast.parse(program)
        helpers = []
        # imports = [ast.parse("import json"), ast.parse("from program_refactoring.domains.scan.scan_helpers import *")]

        # TODO; check if this oik
        imports = [ast.parse("import json")]

        for node in parsed.body:
            if isinstance(node, ast.FunctionDef):
                helpers.append(node)
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                imports.append(node)
            else:
                header_parsed.body[0].body.append(node)

        temp_dir_safe = re.sub("\/+", ".", str(self.temp_dir))
        if Path(f"{self.temp_dir}/codebank.py").exists():
            # create an __init__.py file and write codebank
            with open(f"{self.temp_dir}/__init__.py", "w") as f1:
                f1.write("\n")
            import_stmt = f"from {temp_dir_safe}.codebank import *"
            imports.append(ast.parse(import_stmt))
        try:
            final_line = parsed.body[-1]
        except IndexError:
            return ""
        # if it's a print statement, take the thing before (what will be printed )
        if (
            isinstance(final_line, ast.Expr)
            and hasattr(final_line.value, "func")
            and final_line.value.func.id == "print"
        ):
            assignee = ast.unparse(final_line.value.args[0])
        else:
            try:
                assignee = final_line.targets[0].id
            except AttributeError:
                return ""

        # instead of return we will write to file
        return_stmt = ast.parse(
            f"with open('{self.temp_dir}/result_{self.type}.json', 'w') as f1:\n    json.dump({assignee}, f1)"
        )
        header_parsed.body[0].body.append(return_stmt.body[0])
        # remove placeholder _=0
        header_parsed.body[0].body = header_parsed.body[0].body[1:]
        exec_stmt = ast.parse(f"{func_name}()")
        header_parsed.body.append(exec_stmt)

        # add the helper functions into the LOCAL scope
        header_parsed.body[0].body = helpers + header_parsed.body[0].body
        # add imports before everything
        header_parsed.body = imports + header_parsed.body

        new_program = ast.unparse(header_parsed)

        if Path(f"{self.temp_dir}/result_{self.type}.json").exists():
            with open(f"{self.temp_dir}/result_{self.type}.json") as f1:
                result = json.load(f1)
            print(
                " +++ WRAP, JSON File found", f"{self.temp_dir}/result_{self.type}.json"
            )
        else:
            print(
                " !!! WRAP, JSON File not found",
                f"{self.temp_dir}/result_{self.type}.json",
            )

        print("Wrap new program", new_program)

        return new_program

    def execute(self, additional_path=None):
        """Execute the program to obtain a result"""
        self.exec_program = self.wrap_program(self.program)
        if additional_path is not None:
            with open(additional_path, "w") as f1:
                f1.write(self.exec_program)
        try:
            with open(f"{self.temp_dir}/prog_{self.type}.py", "w") as f1:
                f1.write(self.exec_program)
            p = subprocess.Popen(
                ["python", f"{self.temp_dir}/prog_{self.type}.py"],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            out, errs = p.communicate()
            (
                out,
                errs,
            ) = out.decode(), errs.decode()

        except Exception as e:
            print(f"Error executing program: {e}")
            raise e
        # read from file
        try:
            with open(f"{self.temp_dir}/result_{self.type}.json") as f1:
                result = json.load(f1)
            # delete file
            Path(f"{self.temp_dir}/result_{self.type}.json").unlink()
        except FileNotFoundError:
            result = "error"

        return result


class LogoNode(Node):
    def __init__(
        self,
        query,
        program,
        type="gold",
        name=None,
        description=None,
        metadata=None,
        node_id=None,
        temp_dir=None,
        is_success=False,
        is_done=False,
    ):
        super().__init__(
            query=query,
            program=program,
            type=type,
            name=name,
            description=description,
            metadata=metadata,
            node_id=node_id,
            is_success=is_success,
            is_done=is_done,
        )
        # check to see if it executes
        # if not self.is_returning(program.strip()):
        #     self.exec_program = program
        # else:
        if temp_dir is None:
            self.temp_dir = "temp"
        else:
            self.temp_dir = Path(temp_dir)
        self.exec_program = self.wrap_program(program)
        self.name = re.sub(" ", "_", self.name)

    def wrap_program(self, program):
        header = LOGO_HEADER
        temp_dir_safe = re.sub("\/+", ".", str(self.temp_dir))

        if Path(f"{self.temp_dir}/codebank.py").exists():
            # create an __init__.py file and write codebank
            with open(f"{self.temp_dir}/__init__.py", "w") as f1:
                f1.write("\n")
            program = f"{header}\nfrom {temp_dir_safe}.codebank import *\n\n{program}"
        else:
            program = f"{header}\n\n\n{program}"

        program = f"{program}\nturtle.save('{self.temp_dir}/result_{self.type}.jpg')"
        return program

    def execute(self, additional_path=None):
        """Execute the program to obtain a result"""
        if additional_path is not None:
            with open(additional_path, "w") as f1:
                f1.write(self.exec_program)
        try:
            with open(f"{self.temp_dir}/prog_{self.type}.py", "w") as f1:
                f1.write(self.exec_program)
            p = subprocess.Popen(
                ["python", f"{self.temp_dir}/prog_{self.type}.py"],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            out, errs = p.communicate()
            (
                out,
                errs,
            ) = out.decode(), errs.decode()
            # print(out)
            # print(errs)
            result = load_img(f"{self.temp_dir}/result_{self.type}.jpg")
        except Exception as e:
            # pdb.set_trace()
            print(f"Error executing program: {e}")
            result = None
        # read from file
        # fname = get_fname(self.exec_program)

        return result


class TextCraftNode(Node):
    def __init__(
        self,
        query,
        program,
        type="gold",
        name=None,
        description=None,
        metadata=None,
        node_id=None,
        temp_dir=None,
        is_success=False,
        is_done=False,
    ):
        super().__init__(
            query=query,
            program=program,
            type=type,
            name=name,
            description=description,
            metadata=metadata,
            node_id=node_id,
            is_success=is_success,
            is_done=is_done,
        )
        self.metadata = metadata
        self.idx = int(name.split("_")[-1])
        self.target_obj = self.query.split("craft ")[-1].rstrip(".")
        if temp_dir is None:
            self.temp_dir = "temp"
        else:
            self.temp_dir = Path(temp_dir)
        self.exec_program = self.wrap_program(program)
        self.name = re.sub(" ", "_", self.name)

    def wrap_program(self, program):
        header = TEXTCRAFT_HEADER
        temp_dir_safe = re.sub("\/+", ".", str(self.temp_dir))
        env_idx = self.idx
        if Path(f"{self.temp_dir}/codebank.py").exists():
            # create an __init__.py file and write codebank
            with open(f"{self.temp_dir}/__init__.py", "w") as f1:
                f1.write("\n")
            program = f"{header}\nfrom {temp_dir_safe}.codebank import *\n\ninit_obs, init_info = env.reset(seed={env_idx})\n\n{program}"
        else:
            program = f"{header}\n\ninit_obs, init_info = env.reset(seed={env_idx})\n\n{program}"

        program = f"{program}\nresult = check_inventory()\nprint('RESULT: ', result)"
        return program

    def execute(self, additional_path=None, verbose=False):
        """Execute the program to obtain a result"""
        if additional_path is not None:
            with open(additional_path, "w") as f1:
                f1.write(self.exec_program)
        try:
            with open(f"{self.temp_dir}/prog_{self.type}.py", "w") as f1:
                f1.write(self.exec_program)
            p = subprocess.Popen(
                ["python", f"{self.temp_dir}/prog_{self.type}.py"],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            out, errs = p.communicate()
            (
                out,
                errs,
            ) = out.decode(), errs.decode()
            out = out.rstrip("\n")
            result = out.split("\n")[-1].split("RESULT: ")[-1]
            result = f"[{self.target_obj}]" in result
        except Exception as e:
            print(f"Error executing program: {e}")
            result = None
        if verbose:
            return result, out
        return result


class CodeContestNode(Node):
    def __init__(
        self,
        query,
        program,
        tests,
        type="gold",
        name=None,
        description=None,
        metadata=None,
        node_id=None,
        temp_dir=None,
        is_success=False,
        is_done=False,
    ):
        super().__init__(
            query,
            program,
            type=type,
            name=name,
            description=description,
            metadata=metadata,
            node_id=node_id,
            temp_dir=temp_dir,
            is_success=is_success,
            is_done=is_done,
        )

        self.tests = tests
        self.k = 10
        self.exec_program = program

    def execute(self, codebank, additional_path=None) -> ExecutionResult:
        """Execute the program to obtain a result"""
        # self.exec_program = self.wrap_program(self.program)
        self.exec_program = self.program
        print(self.program)
        if additional_path is not None:
            with open(additional_path, "w") as f1:
                f1.write(self.exec_program)

        test_inputs = self.tests["input"][: self.k]
        assert len(test_inputs) > 0
        test_outputs = self.tests["output"][: self.k]

        outputs, errors = execute_main_codebank_inputs(self.exec_program, codebank, test_inputs)

        for x, y, z in zip(outputs, test_outputs, errors):
            print("GOLD OUTPUT")
            print(y)
            print("PREDICTED OUTPUT")
            print(x)
            print("ERROR")
            print(z)

        matches = [
            error == "" and x.strip() == y.strip()
            for x, y, error in zip(outputs, test_outputs, errors)
        ]

        return ExecutionResult(
            passed=matches,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            predicted_outputs=outputs,
            stderrs=errors,
        )

    def wrap_program(self, program):
        return program
