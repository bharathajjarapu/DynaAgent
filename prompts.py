def codegen(spec):
    return f"""
    Generate a complete working Python function for this tool:
    Name: {spec["name"]}
    Description: {spec["description"]}
    Requirements:
    - Function MUST have proper type hints like: def {spec["name"]}(city: str) -> str:
    - All parameters need type hints (use str, int, float as needed)
    - Return type must be -> str
    - Add print(f"🔧 {spec["name"]}({{locals()['city']}})") at start to show tool is being used
    - Include all imports inside the function (import requests, import os, import json)
    - Handle errors with try/except and return error messages as strings
    - Use os.getenv() for API keys if needed
    Return ONLY the Python function code, no decorator, no explanations, no markdown.
    """

def analyze(question):
    return f"""
    Analyze if we need to create new tools for: {question}
    Current tools: calculator (for math), search (for web search)
    Create specialized API tools for: weather data, stock prices, news, currency rates, etc.
    Web search is INEFFICIENT for real-time data. Use APIs instead.
    Examples:
    - Weather query -> create weather_api tool with OpenWeatherMap
    - Stock price -> create stock_api tool with Alpha Vantage
    - News -> create news_api tool with NewsAPI
    Return ONLY valid JSON:
    {{"needs_new_tools": true/false, "tools": [{{"name": "tool_name", "description": "Get weather using OpenWeatherMap API for city parameter"}}]}}"""

def planner(question):
    return f"""
    Analyze this task and determine what specialized agents are needed: {question}.
    Make sure the Names are Single Word like WebSearchAgent, MathAgent...
    Select appropriate model for each agent:
    - Simple tasks (only for summarization) -> gpt-4.1-nano
    - Medium tasks (planning, web search, API calls) -> gpt-4.1-mini
    - Complex tasks (analysis, synthesis) -> gpt-4.1
    - Very Complex tasks (multi-step reasoning) -> gpt-5.1
    Return ONLY valid JSON in this exact format:
    {{"agents":[{{"name":"AgentName","role":"RoleDescription","instructions":"What agent should do","model":"gpt-4.1-mini"}}],"plan":["step1","step2"]}}
    """

def synthesize(results):
    return f"Based on these agent results, provide ONLY the final answer (In Plain Text):\n{results}\n\n Final answer:"