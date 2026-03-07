import os
import requests

from agents import WebSearchTool, function_tool

search = WebSearchTool()

@function_tool
def calculator(operation: str, a: float, b: float) -> float:
    print(f"🔧 calculator({operation}, {a}, {b})")
    op = operation.lower()
    if op in ["+", "add", "addition", "plus", "sum"]:
        result = a + b
    elif op in ["-", "subtract", "minus", "difference"]:
        result = a - b
    elif op in ["*", "multiply", "times", "product"]:
        result = a * b
    elif op in ["/", "divide", "division", "divided by"]:
        result = a / b if b != 0 else float("inf")
    else:
        result = 0.0
    print(f"   Result: {result}")
    return result


async def build(coordinator, tools_list, spec):
    
    from agents import Runner
    from prompts import codegen

    result = await Runner.run(coordinator, codegen(spec))
    code = result.final_output.strip()

    if "```" in code:
        code = code.split("```")[1]
        if code.startswith("python"):
            code = code[6:]
        code = code.strip()

    if "@function_tool" in code:
        code = code.replace("@function_tool\n", "").replace("@function_tool", "")

    try:
        namespace = {"requests": requests, "os": os}
        exec(code, namespace)
        tool_func = namespace.get(spec["name"])

        if tool_func and callable(tool_func):
            decorated_tool = function_tool(tool_func)
            tools_list.append(decorated_tool)
            print(f"🔧 Tool {spec['name']} created and ready")
            return decorated_tool
    except Exception as e:
        print(f"✗ Tool creation failed: {e}")

    return None
