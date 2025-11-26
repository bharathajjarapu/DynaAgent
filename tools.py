from agents import WebSearchTool, function_tool

search = WebSearchTool()

@function_tool
def calculator(operation: str, a: float, b: float) -> float:
    print(f"🔧 calculator({operation}, {a}, {b})")
    op = operation.lower()
    if op in ["+", "add", "plus", "sum"]:
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