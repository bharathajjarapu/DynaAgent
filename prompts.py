"""Prompts and strict JSON schemas for the coordinator and workers."""

import json
from datetime import datetime


def record(**props):
    """Strict JSON-schema object where every property is required."""
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def array(item):
    """JSON-schema array of `item`."""
    return {"type": "array", "items": item}


STR, BOOL, INT = {"type": "string"}, {"type": "boolean"}, {"type": "integer"}
TASK = record(id=STR, goal=STR, agent=STR, role=STR, instructions=STR, tools=array(STR), smart=BOOL, deps=array(STR))
SPEC = record(name=STR, description=STR, source=STR, env=STR)
PLAN = record(direct=BOOL, answer=STR, tools=array(SPEC), tasks=array(TASK))
REVIEW = record(answer=STR, score=INT, gaps=STR, tasks=array(TASK))
CODE = record(code=STR)
BALLOT = record(choice=STR, reason=STR)

RULES = """Task rules:
- Each task is one atomic step done by one worker agent. Ids are short: t1, t2, ...
- agent: single PascalCase word (e.g. WebResearcher, MathSolver). Give tasks that need the same expertise the same agent name, role and instructions so the agent is reused.
- tools: only names from the available or newly requested tools; [] if none needed.
- smart: true only for hard reasoning, deep analysis or tricky math; false otherwise.
- deps: ids whose output this task needs. Tasks that do not need each other must not depend on each other so they run in parallel.
- Use as few tasks as possible. One task is right for simple lookups or a single tool use."""


def now():
    """Current local time, so dated search results are trusted."""
    return f"Now is {datetime.now().astimezone():%A, %B %d, %Y %H:%M %Z}. Trust dated results up to now."


def coordinator():
    """Coordinator system prompt."""
    return f"You are the DynaAgent coordinator. You decompose tasks, design specialist agents, and synthesize their results. Be concise. {now()}"


def worker(name, role, instructions, tools):
    """Worker persona system prompt."""
    use = f" You have these tools: {', '.join(tools)}. Use them for anything current, external or computational; never claim you lack access." if tools else ""
    return f"You are {name}, a {role}. {instructions}{use} {now()}"


def plan(question, tools, examples, limit):
    """Ask for a direct answer or a task DAG plus any new tools."""
    past = "\n".join(f"- {example['question'][:200]}\n  {json.dumps(example['tasks'])[:800]}" for example in examples) or "None"
    return f"""Task: {question}

Available tools:
{tools}

Plans that worked for similar past tasks:
{past}

Decide:
1. direct=true with a short answer if this is a greeting, casual chat, or something you can answer correctly right now without tools, research or multi-step work. Then tools=[] and tasks=[].
2. Otherwise direct=false, answer="", and plan a dependency graph of at most {limit} tasks.

New tools: request one only for real-time or external data that web search handles poorly (weather, prices, exchange rates, specific public APIs). Never request a tool that an available tool already covers. Make new tools generic with parameters (e.g. exchange_rate(base, target), not usd_to_gbp). Prefer free no-auth APIs. name is snake_case; env is the name of an API key env var only if no free option exists, else "". Usually tools=[].

{RULES}"""


def work(question, task, deps):
    """Worker prompt for one task, with the outputs it depends on."""
    context = "\n\n".join(f"[{key}] {output}" for key, output in deps.items()) or "None"
    return f"""Overall request: {question}

Your task: {task['goal']}

Results from earlier tasks:
{context}

Do your task fully and accurately. Use your tools when they help. Do not introduce yourself, ask questions, or offer further help. Use markdown, no LaTeX. Return only the result."""


def results(tasks, outputs):
    """All task outputs, labelled by id and agent."""
    return "\n\n".join(
        f"[{task['id']}] {task['agent']}: {task['goal']}{' (tools used: ' + ', '.join(task['used']) + ')' if task.get('used') else ''}\n"
        f"{outputs.get(task['id'], '')[:4000]}"
        for task in tasks
    )


def propose(question, tasks, outputs):
    """Ask an agent for its candidate final answer."""
    return f"""Task: {question}

Worker results:
{results(tasks, outputs)}

From your expertise, write the best complete final answer to the user using these results, in at most 300 words. No preamble. Markdown, no LaTeX."""


def ballot(question, candidates, tally):
    """Ask an agent to vote for one candidate answer."""
    shown = f"\n\nVotes so far: {tally}" if tally else ""
    return f"""Task: {question}

Candidate final answers:

{candidates}{shown}

Vote for the single candidate that answers the task most correctly and completely. Judge content, not style.
choice: the candidate letter only. reason: one short sentence."""


def review(question, tasks, outputs, agents, turn, room, consensus=""):
    """Ask for the final answer, a 0-10 score, gaps and follow-up tasks."""
    voted = f"\n\nConsensus answer chosen by agent vote (base your answer on it; fix only clear errors or gaps):\n{consensus}" if consensus else ""
    return f"""Task: {question}

Worker results (round {turn}):
{results(tasks, outputs)}{voted}

Existing agents: {', '.join(agents)}

1. answer: the final answer to the user, written from these results. Answer the actual request directly and completely. No preamble. Markdown, no LaTeX.
2. score: 0-10, how fully and correctly the results answer the request.
3. gaps: what is missing or doubtful, "" if nothing.
4. tasks: if score < 8 and more work would fix the gaps, up to {room} follow-up tasks with new ids (deps may reference existing ids). Otherwise [].

{RULES}"""


def codegen(spec):
    """Ask for a stdlib-only Python tool function."""
    env = f"Read the API key with os.environ.get({spec['env']!r}); never hardcode secrets." if spec["env"] else "Use a free no-auth API."
    return f"""Write a Python tool function.
Name: {spec['name']}
Purpose: {spec['description']}
Suggested source: {spec['source'] or 'any reliable free API'}

Requirements:
- def {spec['name']}(...) -> str with type hints (str, int, float, bool) on every parameter.
- Standard library only: urllib.request, urllib.parse, json, os. Put imports inside the function. Use timeout=10 on requests and send a User-Agent header.
- {env}
- Catch all errors and return them as a string starting with "Error:". Return compact, readable text.
- No prints, no top-level side effects.
Return the code only in the `code` field."""
