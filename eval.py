"""Benchmark DynaAgent (or a single-model baseline) on GPQA-Diamond, MuSR and IFEval.

uv run eval.py musr --limit 30
uv run eval.py gpqa --limit 30 --single          # single LLM baseline, needs HF_TOKEN (gated dataset)
uv run eval.py ifeval --limit 50                 # writes responses; score with the official IFEval checker
"""

import argparse
import ast
import asyncio
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import llm
from core import LETTERS, DynaAgent

SETS = {
    "gpqa": ("Idavidrein/gpqa", "gpqa_diamond", "train"),
    "musr": ("TAUR-Lab/MuSR", "default", "murder_mysteries"),
    "ifeval": ("google/IFEval", "default", "train"),
}


def rows(dataset, config, split, limit):
    """Fetch `limit` rows from the Hugging Face dataset server."""
    found, headers = [], {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.getenv("HF_TOKEN") else {}
    while len(found) < limit:
        query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": len(found), "length": min(100, limit - len(found))})
        request = urllib.request.Request(f"https://datasets-server.huggingface.co/rows?{query}", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                page = [entry["row"] for entry in json.load(response)["rows"]]
        except urllib.error.HTTPError as error:
            sys.exit(f"{dataset}: HTTP {error.code}. Gated datasets need HF_TOKEN set and the terms accepted on huggingface.co/datasets/{dataset}.")
        if not page:
            break
        found += page
    return found


def item(kind, index, row):
    """Build (prompt, gold letter or None) for one row."""
    if kind == "ifeval":
        return row["prompt"], None
    if kind == "gpqa":
        question = row["Question"]
        choices = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        random.Random(index).shuffle(choices)
        gold = choices.index(row["Correct Answer"])
    else:
        question = f"{row['narrative']}\n\n{row['question']}"
        choices = ast.literal_eval(row["choices"]) if isinstance(row["choices"], str) else row["choices"]
        gold = int(row["answer_index"])
    options = "\n".join(f"({LETTERS[slot]}) {choice.strip()}" for slot, choice in enumerate(choices))
    prompt = f"{question}\n\n{options}\n\nThink it through, then finish with a final line exactly: Answer: <letter>"
    return prompt, LETTERS[gold]


def letter(text):
    """Last "Answer: X" letter in the response."""
    found = re.findall(r"Answer:\s*\**\(?([A-J])\b", text)
    return found[-1] if found else None


async def solve(prompt, single):
    """Answer with DynaAgent, or one plain model call for the baseline."""
    if single:
        budget = llm.Budget()
        try:
            return await llm.respond(prompt, budget=budget), budget, []
        except (llm.LLMError, llm.BudgetExceeded) as error:
            return f"Error: {error}", budget, []
    agent = DynaAgent(memory=None)  # no long-term memory, so items don't leak into each other
    return await agent.execute(prompt), agent.budget, agent.votes


async def main():
    """Run a benchmark and print accuracy, voting and cost."""
    parser = argparse.ArgumentParser()
    parser.add_argument("bench", choices=SETS)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--split", help="MuSR split: murder_mysteries, object_placements, team_allocation")
    parser.add_argument("--single", action="store_true", help="single-model baseline instead of DynaAgent")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    dataset, config, split = SETS[args.bench]
    data = rows(dataset, config, args.split or split, args.limit)
    gate = asyncio.Semaphore(args.concurrency)
    mode = "single" if args.single else "dynaagent"
    os.makedirs("results", exist_ok=True)
    path = f"results/{args.bench}_{mode}_{int(time.time())}.jsonl"

    async def grade(index, row):
        """Solve and score one row."""
        prompt, gold = item(args.bench, index, row)
        async with gate:
            answer, budget, votes = await solve(prompt, args.single)
        guess = letter(answer) if gold else None
        print(f"[{index + 1}/{len(data)}] {'-' if gold is None else 'ok' if guess == gold else 'x '} {budget.summary()}", flush=True)
        return {"prompt": prompt, "response": answer, "gold": gold, "pred": guess,
                "calls": budget.used_calls, "tokens": budget.used_tokens, "cost": budget.used_cost, "seconds": budget.elapsed(), "votes": votes}

    results = await asyncio.gather(*(grade(index, row) for index, row in enumerate(data)))
    with open(path, "w") as file:
        file.writelines(json.dumps(result) + "\n" for result in results)

    count = max(len(results), 1)
    total = {key: sum(result[key] for result in results) for key in ("tokens", "calls", "cost", "seconds")}
    print(f"\n{args.bench} · {mode} · model {llm.MODEL} · n={len(results)} · saved {path}")
    if args.bench == "ifeval":
        print("Score with the official checker (google-research instruction_following_eval) using input_response_data=" + path)
    else:
        accuracy = sum(result["pred"] == result["gold"] for result in results) / count
        print(f"accuracy {accuracy:.1%}")
    votes = [vote for result in results for vote in result["votes"]]
    if votes:
        ballots = sum(vote["voters"] for vote in votes)
        print(f"votes {len(votes)} ({votes[0]['mode']}) · self-vote rate {sum(vote['self_votes'] for vote in votes) / ballots:.0%}"
              f" · first vote won {sum(vote['first_won'] for vote in votes) / len(votes):.0%}")
    print(f"tokens {total['tokens']:,} · calls {total['calls']} · cost ${total['cost']:.4f} · avg {total['seconds'] / count:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
