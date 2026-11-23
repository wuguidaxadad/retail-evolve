"""Bounded code evolution: pure prompt function plus validated execution switches."""

from __future__ import annotations

import ast
import hashlib

BASE_CODE = """INSTRUCTION = "You are a helpful assistant that completes tasks according to the provided policy."
REQUIRE_CONFIRMATION = False
RETRY_ERROR_HINT = False
MAX_STEPS = 60

def build_prompt(base_instruction, skill_context, error_context):
    return base_instruction + "\\n\\n" + skill_context
"""

ALLOWED_NODES = (
    ast.Module,
    ast.Assign,
    ast.Name,
    ast.Store,
    ast.Load,
    ast.Constant,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.BinOp,
    ast.Add,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.Expr,
)
CONSTANTS = {"INSTRUCTION", "REQUIRE_CONFIRMATION", "RETRY_ERROR_HINT", "MAX_STEPS"}
PARAMS = {"base_instruction", "skill_context", "error_context"}


def validate_code(code: str) -> dict:
    """No imports, calls, attributes, loops, IO, or arbitrary executable code."""
    if len(code) > 20000:
        raise ValueError("候选代码过长")
    tree = ast.parse(code)
    names = set()
    functions = []
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"候选代码包含未允许的语法：{type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in CONSTANTS | PARAMS:
            raise ValueError("候选代码包含未允许的变量")
        if isinstance(node, ast.FormattedValue) and node.conversion not in {-1, 115}:
            raise ValueError("格式转换不允许")
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ValueError("常量定义格式不正确")
            name = node.targets[0].id
            if name not in CONSTANTS or name in names:
                raise ValueError("常量定义不正确")
            ast.literal_eval(node.value)
            names.add(name)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node)
            if (
                node.name != "build_prompt"
                or node.decorator_list
                or node.returns
                or [arg.arg for arg in node.args.args]
                != ["base_instruction", "skill_context", "error_context"]
                or node.args.defaults
                or node.args.vararg
                or node.args.kwarg
                or node.args.kwonlyargs
                or node.args.posonlyargs
                or len(node.body) != 1
                or not isinstance(node.body[0], ast.Return)
            ):
                raise ValueError("只能修改 build_prompt 的纯表达式")
        else:
            raise ValueError("不允许额外顶层代码")
    if names != CONSTANTS or len(functions) != 1:
        raise ValueError("候选代码缺少必要定义")
    namespace = {"__builtins__": {}}
    exec(compile(tree, "<bounded-harness>", "exec"), namespace)
    if not isinstance(namespace["INSTRUCTION"], str):
        raise ValueError("INSTRUCTION 必须为文本")
    if any(type(namespace[k]) is not bool for k in ["REQUIRE_CONFIRMATION", "RETRY_ERROR_HINT"]):
        raise ValueError("执行开关必须是布尔值")
    if type(namespace["MAX_STEPS"]) is not int or not 10 <= namespace["MAX_STEPS"] <= 80:
        raise ValueError("MAX_STEPS 必须在 10 到 80 之间")
    output = namespace["build_prompt"]("BASE", "SKILLS", "ERRORS")
    if (
        not isinstance(output, str)
        or len(output) > 25000
        or "BASE" not in output
        or "SKILLS" not in output
    ):
        raise ValueError("候选提示必须保留基础指令与经验上下文")
    return namespace


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]
