"""The tools the agent can call.

A LangChain tool is a plain Python function wrapped by @tool. The model never
sees your code. It sees only the function's NAME, its TYPE-ANNOTATED PARAMETERS
and its DOCSTRING, which LangChain turns into a JSON schema. That makes the
docstring part of the prompt: write it for the model, not for other developers.

To add a tool: write a function, decorate it, append it to TOOLS at the bottom.
Nothing else in the project needs to change.
"""

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time in an IANA timezone, e.g. 'Africa/Lagos' or 'America/New_York'."""
    try:
        now = datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError:
        # Returning the problem as text, instead of raising, lets the model read
        # it and retry with a valid name. Exceptions are for bugs; errors the
        # model can fix are better returned as strings.
        return f"Unknown timezone '{timezone}'. Use an IANA name such as 'Europe/London'."
    return now.strftime("%A, %Y-%m-%d %H:%M:%S %Z (UTC%z)")


# --- calculate ---------------------------------------------------------------
# Never eval() text that came from a model: it can contain anything. Instead we
# parse the expression into a syntax tree and walk it, accepting only numbers
# and a whitelist of arithmetic operators. Anything else is rejected.

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")  # stops 9**9**9 from hogging the CPU
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("only numbers and + - * / % ** ( ) are allowed")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(12.5 * 4) / 3 + 2 ** 8'. Supports + - * / % ** and parentheses."""
    try:
        result = _evaluate(ast.parse(expression, mode="eval").body)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        return f"Could not evaluate '{expression}': {exc}"
    return str(result)


# The graph imports this list twice: once to tell the model what exists
# (bind_tools) and once to actually run the calls (ToolNode).
TOOLS = [get_current_time, calculate]
