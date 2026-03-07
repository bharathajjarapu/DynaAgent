"""Built-in tools and sandboxed runtime-generated tools."""

import ast
import json
import keyword
import math
import operator
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from llm import KEY


@dataclass
class Tool:
    """A model-callable tool: Responses API spec plus a local runner (None for hosted tools)."""

    name: str
    spec: dict
    run: object = None


def schema(name, description, params):
    """Strict function-tool spec."""
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": params, "required": list(params), "additionalProperties": False},
        "strict": True,
    }


OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}
FUNCS = {
    name: getattr(math, name)
    for name in "sqrt cbrt exp log log10 log2 sin cos tan asin acos atan atan2 sinh cosh tanh "
    "floor ceil factorial comb perm gcd lcm degrees radians hypot".split()
} | {"abs": abs, "round": round, "min": min, "max": max}


def calculate(expression: str) -> str:
    """Evaluate arithmetic safely by walking the AST; never calls eval."""

    def walk(node):
        """Evaluate one whitelisted AST node."""
        match node:
            case ast.Constant(value=int() | float() as value):
                return value
            case ast.Name(id=name) if name in NAMES:
                return NAMES[name]
            case ast.UnaryOp(op=op) if type(op) in OPS:
                return OPS[type(op)](walk(node.operand))
            case ast.BinOp(op=op) if type(op) in OPS:
                left, right = walk(node.left), walk(node.right)
                if isinstance(op, ast.Pow) and (abs(right) > 10_000 or abs(left) > 1 and math.log2(abs(left)) * right > 100_000):
                    raise ValueError("result too large")
                return OPS[type(op)](left, right)
            case ast.Call(func=ast.Name(id=func), keywords=[]) if func in FUNCS:
                args = [walk(arg) for arg in node.args]
                if func in ("factorial", "comb", "perm") and max(args, default=0) > 10_000:
                    raise ValueError("argument too large")
                return FUNCS[func](*args)
        raise ValueError(f"unsupported expression: {ast.unparse(node)}")

    try:
        return str(walk(ast.parse(expression.replace("^", "**"), mode="eval").body))
    except Exception as error:
        return f"Error: {error}"


calculator = Tool(
    "calculator",
    schema(
        "calculator",
        "Evaluate a math expression exactly, e.g. '(3.5*2)**3 / sqrt(7)'. Supports + - * / // % ** and math functions.",
        {"expression": {"type": "string"}},
    ),
    calculate,
)
search = Tool("search", {"type": "web_search"})
BUILTIN = [calculator, search]

# Runs inside the sandbox process: limit CPU/memory, load the tool, call it with JSON args from stdin.
RUNNER = """
import json, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (15, 16))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
except Exception:
    pass
scope = {}
exec(open(sys.argv[1]).read(), scope)
print("\\n" + json.dumps(str(scope[sys.argv[2]](**json.load(sys.stdin)))))
"""
TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
SECRETS = {"LLM_KEY", "OPENAI_API_KEY"}


def sandbox(code, name, args, secret="", timeout=25):
    """Run generated code in an isolated interpreter: clean env, CPU and memory caps, timeout."""
    # ponytail: process isolation only; the child can still read files the user can. Use a container/nsjail for untrusted multi-user deployments.
    env = {"PATH": os.environ.get("PATH", "")}
    if secret and secret not in SECRETS and os.environ.get(secret) and os.environ[secret] != KEY:
        env[secret] = os.environ[secret]
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "tool.py")
        with open(path, "w") as file:
            file.write(code)
        try:
            process = subprocess.run(
                [sys.executable, "-I", "-c", RUNNER, path, name],
                input=json.dumps(args), capture_output=True, text=True, timeout=timeout, cwd=folder, env=env,
            )
        except subprocess.TimeoutExpired:
            return f"Error: tool timed out after {timeout}s"
    if process.returncode == -getattr(signal, "SIGXCPU", 0):
        return "Error: tool exceeded its CPU time limit"
    if process.returncode:
        return f"Error: {(process.stderr.strip().splitlines() or ['tool crashed'])[-1]}"
    try:
        return json.loads(process.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return "Error: tool returned no result"


def forge(name, description, code, secret=""):
    """Validate generated code and wrap it as a sandboxed Tool; raises ValueError if invalid."""
    if not name.isidentifier() or keyword.iskeyword(name) or name in {tool.name for tool in BUILTIN}:
        raise ValueError(f"bad tool name {name!r}")
    if secret in SECRETS:
        raise ValueError(f"tools may not read {secret}")
    func = next((node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef) and node.name == name), None)
    if not func:
        raise ValueError(f"code does not define {name}()")
    params = {arg.arg: {"type": TYPES.get(getattr(arg.annotation, "id", ""), "string")} for arg in func.args.args}
    return Tool(name, schema(name, description, params), lambda **kwargs: sandbox(code, name, kwargs, secret))
