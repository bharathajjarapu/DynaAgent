"""DynaAgent: plans a task DAG, generates worker agents, runs them in parallel, votes, and refines until convergence."""

import asyncio
import graphlib
import json
import os
import random
import re
import sqlite3
from collections import Counter

import llm
import prompts
from llm import Budget, BudgetExceeded, LLMError
from tools import BUILTIN, forge

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynaagent.db")
ROUNDS = 3  # initial DAG + up to 2 refinement rounds
GOOD = 8  # review score that ends refinement
THETA = 1  # stop when the score improves by less than this (convergence)
VOTE = os.getenv("DYNA_VOTE", "anonymous")  # anonymous | identified | off
VOTERS = 5  # max agents that propose and vote
LETTERS = "ABCDEFGHIJ"


class Memory:
    """Long-term memory in SQLite: high-scoring plans (FTS5) and generated tools."""

    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(
            """
            create virtual table if not exists plans using fts5(question, plan unindexed, score unindexed);
            create table if not exists tools(name text primary key, description text, code text, env text);
            """
        )

    def recall(self, question):
        """Top 2 past plans scoring 7+ for keyword-similar questions (BM25)."""
        # ponytail: keyword recall, swap for embeddings if reworded questions get missed.
        words = {word for word in re.findall(r"\w+", question.lower()) if len(word) > 2}
        if not words:
            return []
        rows = self.db.execute(
            "select question, plan from plans where plans match ? and score >= 7 order by bm25(plans) limit 2",
            (" OR ".join(f'"{word}"' for word in sorted(words)[:40]),),
        )
        return [{"question": question, "tasks": json.loads(plan)} for question, plan in rows]

    def remember(self, question, tasks, score):
        """Store a plan that worked."""
        with self.db:
            self.db.execute("insert into plans values (?, ?, ?)", (question, json.dumps(tasks), score))

    def tools(self):
        """All saved generated tools."""
        return self.db.execute("select name, description, code, env from tools").fetchall()

    def save(self, name, description, code, env):
        """Store a generated tool."""
        with self.db:
            self.db.execute("insert or replace into tools values (?, ?, ?, ?)", (name, description, code, env))


class DynaAgent:
    """Coordinator: Task Analyzer, Agent Generator, Orchestrator and Convergence Detector."""

    def __init__(self, memory=DB, on=None, **limits):
        self.on = on or (lambda who, text, code="": None)  # event callback for the UI
        self.history = []  # (question, answer) chat turns
        self.limits = limits  # Budget overrides, e.g. calls=20
        self.budget = Budget(**limits)
        self.memory = Memory(memory) if memory else None
        self.votes = []  # per-query voting stats
        self.unsaved = {}  # generated tools persist after their first successful call
        self.tools = {tool.name: tool for tool in BUILTIN}
        for name, description, code, env in self.memory.tools() if self.memory else []:
            try:
                self.tools[name] = forge(name, description, code, env)
            except (ValueError, SyntaxError):
                pass

    def context(self, question):
        """Question plus the last 4 chat turns."""
        if not self.history:
            return question
        turns = "\n\n".join(f"User: {asked}\nAnswer: {answer[:1500]}" for asked, answer in self.history[-4:])
        return f"Chat so far:\n\n{turns}\n\nNew user message: {question}"

    def catalog(self):
        """Available tools as a prompt list."""
        return "\n".join(
            f"- {tool.name}: {tool.spec.get('description', 'web search for current information')}" for tool in self.tools.values()
        )

    async def ask(self, prompt, schema):
        """Coordinator call with structured output."""
        return await llm.respond(prompt, budget=self.budget, model=llm.SMART, system=prompts.coordinator(), schema=schema)

    async def build(self, spec):
        """Generate, validate and register a new tool (paper Alg. 3)."""
        name = spec["name"]
        if name in self.tools:
            return
        self.on("Agent", f"Creating tool `{name}`. {spec['description']}")
        try:
            code = (await self.ask(prompts.codegen(spec), prompts.CODE))["code"]
            self.tools[name] = forge(name, spec["description"], code, spec["env"])
        except (LLMError, ValueError, SyntaxError, KeyError) as error:
            self.on("Agent", f"Failed to create `{name}`: {error}")
            return
        self.unsaved[name] = (spec["description"], code, spec["env"])
        self.on("Agent", f"Created `{name}` (sandboxed).", code)

    def clean(self, tasks, seen, room):
        """Keep valid tasks: new ids, known tools and deps, at most `room`."""
        kept, ids = [], {task["id"] for task in tasks}
        for task in tasks[: max(room, 0)]:
            if task["id"] in seen or not task["goal"]:
                continue
            task["tools"] = [name for name in task["tools"] if name in self.tools]
            task["deps"] = [dep for dep in task["deps"] if dep != task["id"] and (dep in seen or dep in ids)]
            seen.add(task["id"])
            kept.append(task)
        return kept

    def hire(self, tasks, agents):
        """Reuse an agent with the same name, else generate a new one (paper Alg. 2)."""
        for task in tasks:
            if task["agent"] in agents:
                agent = agents[task["agent"]]
                task["tools"] = list(dict.fromkeys(agent["tools"] + task["tools"]))  # a reused agent keeps its tools
                agent["tools"] = task["tools"]
                self.on("Agent", f"Reusing `{task['agent']}` for `{task['id']}`.")
            else:
                agents[task["agent"]] = {"role": task["role"], "instructions": task["instructions"], "tools": task["tools"]}
                tools = f" Tools: {', '.join(task['tools'])}." if task["tools"] else ""
                self.on("Agent", f"Created `{task['agent']}`, a {task['role']}.{tools}", task["instructions"])

    async def work(self, query, task, agent, outputs):
        """Run one task with its agent's persona and tools."""
        self.on("Agent", f"Passed `{task['id']}` to `{task['agent']}`.", task["goal"])
        used = []

        def on_tool(name, args, result):
            """Show a tool call and persist a new tool once it works."""
            used.append(name)
            if name in self.unsaved and not result.startswith("Error") and self.memory:
                self.memory.save(name, *self.unsaved.pop(name))
            self.on(task["agent"], f"Used `{name}`.", args)
            if result:
                self.on(task["agent"], f"`{name}` returned:", result)

        try:
            output = await llm.respond(
                prompts.work(query, task, {dep: outputs[dep] for dep in task["deps"] if dep in outputs}),
                budget=self.budget,
                model=llm.SMART if task["smart"] else llm.MODEL,
                system=prompts.worker(task["agent"], agent["role"], agent["instructions"], task["tools"]),
                tools=[self.tools[name] for name in task["tools"]],
                on_tool=on_tool,
            )
        except (LLMError, BudgetExceeded) as error:
            output = f"Error: {error}"
        self.on(task["agent"], output or "(no output)")
        task["used"] = list(dict.fromkeys(used))  # shown to the reviewer
        return output

    async def run(self, query, tasks, agents, outputs):
        """Run each task as soon as its deps finish; independent tasks run in parallel."""
        lookup = {task["id"]: task for task in tasks}
        graph = graphlib.TopologicalSorter({task["id"]: [dep for dep in task["deps"] if dep in lookup] for task in tasks})
        try:
            graph.prepare()
        except graphlib.CycleError:
            self.on("Agent", "Plan had a dependency cycle; running its tasks independently.")
            graph = graphlib.TopologicalSorter({key: [] for key in lookup})
            graph.prepare()
        running = {}
        while graph.is_active():
            for key in graph.get_ready():
                task = lookup[key]
                running[asyncio.create_task(self.work(query, task, agents[task["agent"]], outputs))] = key
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for job in done:
                key = running.pop(job)
                outputs[key] = job.result()
                graph.done(key)

    async def vote(self, query, tasks, outputs, agents):
        """Agents propose final answers then vote; plurality wins, ties go to the first-voted."""
        # anonymous: authors hidden, shuffled order per voter, parallel. identified: authors shown, sequential with a live tally.
        names = list(agents)[:VOTERS]
        if VOTE not in ("anonymous", "identified") or len(names) < 2:
            return ""

        def persona(name):
            """System prompt for an agent's persona, without tools."""
            return prompts.worker(name, agents[name]["role"], agents[name]["instructions"], [])

        drafts = await asyncio.gather(*(
            llm.respond(prompts.propose(query, tasks, outputs), budget=self.budget, system=persona(name)) for name in names
        ), return_exceptions=True)
        for draft in drafts:
            if isinstance(draft, BudgetExceeded):
                raise draft
        candidates = [(name, draft) for name, draft in zip(names, drafts) if isinstance(draft, str) and draft.strip()]
        if len(candidates) < 2:
            return candidates[0][1] if candidates else ""
        self.on("Agent", f"Voting ({VOTE}): {len(candidates)} candidate answers from {', '.join(f'`{name}`' for name, _ in candidates)}.")
        identified = VOTE == "identified"
        ballots = []  # (voter, candidate index) in the order cast

        async def cast(voter):
            """Collect one agent's vote."""
            order = list(range(len(candidates))) if identified else random.sample(range(len(candidates)), len(candidates))
            listing = "\n\n".join(
                f"({LETTERS[slot]}){f' by {candidates[index][0]}' if identified else ''}\n{candidates[index][1][:6000]}"
                for slot, index in enumerate(order)
            )
            tally = ", ".join(f"{candidates[index][0]}: {count}" for index, count in Counter(index for _, index in ballots).items()) if identified else ""
            try:
                pick = await llm.respond(prompts.ballot(query, listing, tally), budget=self.budget,
                                         system=persona(voter), schema=prompts.BALLOT)
                index = order[LETTERS.index(pick["choice"].strip("() ").upper()[:1])]
            except (LLMError, ValueError, IndexError, KeyError, TypeError):
                return
            ballots.append((voter, index))
            self.on(voter, f"Voted for `{candidates[index][0]}`: {pick['reason']}")

        if identified:
            for voter, _ in candidates:
                await cast(voter)
        else:
            await asyncio.gather(*(cast(voter) for voter, _ in candidates))
        self.budget.check()
        if not ballots:
            return ""
        tally = Counter(index for _, index in ballots)
        best = max(tally.values())
        winner = next(index for _, index in ballots if tally[index] == best)
        stats = {"mode": VOTE, "voters": len(ballots), "winner": candidates[winner][0],
                 "self_votes": sum(voter == candidates[index][0] for voter, index in ballots),
                 "first_won": ballots[0][1] == winner}
        self.votes.append(stats)
        result = ", ".join(f"`{candidates[index][0]}` {count}" for index, count in tally.most_common())
        self.on("Agent", f"Vote result: {result} → `{candidates[winner][0]}` wins. Self-votes {stats['self_votes']}/{len(ballots)}.",
                  candidates[winner][1])
        return candidates[winner][1]

    async def execute(self, question):
        """Answer one question: plan, build tools, run the DAG, vote, review, refine."""
        self.budget = Budget(**self.limits)
        self.votes = []
        query = self.context(question)
        tasks, agents, outputs = [], {}, {}
        answer, score = "", 0
        try:
            examples = self.memory.recall(question) if self.memory else []
            plan = await self.ask(prompts.plan(query, self.catalog(), examples, self.budget.agents), prompts.PLAN)
            if plan["direct"] and plan["answer"]:
                answer = plan["answer"]
            else:
                await asyncio.gather(*(self.build(spec) for spec in plan["tools"]))
                new = self.clean(plan["tasks"], set(), self.budget.agents)
                if not new:
                    new = [{"id": "t1", "goal": question, "agent": "Solver", "role": "problem solver",
                            "instructions": "Solve the task.", "tools": list(self.tools), "smart": False, "deps": []}]
                self.on("Agent", f"Planned {len(new)} task(s).", json.dumps(
                    [{key: task[key] for key in ("id", "agent", "goal", "deps")} for task in new], indent=2))
                last = -THETA
                for turn in range(1, ROUNDS + 1):
                    tasks += new
                    self.hire(new, agents)
                    await self.run(query, new, agents, outputs)
                    consensus = await self.vote(query, tasks, outputs, agents)
                    review = await self.ask(
                        prompts.review(query, tasks, outputs, agents, turn, self.budget.agents - len(tasks), consensus), prompts.REVIEW
                    )
                    answer, score = review["answer"], review["score"]
                    gaps = f" Gaps: {review['gaps']}" if review["gaps"] else ""
                    self.on("Agent", f"Round {turn} score {score}/10.{gaps}")
                    new = self.clean(review["tasks"], set(outputs), self.budget.agents - len(tasks))
                    if score >= GOOD or not new or score - last < THETA:
                        break
                    last = score
                if self.memory and score >= GOOD:
                    self.memory.remember(question, [{key: task[key] for key in ("goal", "agent", "role", "tools", "deps")} for task in tasks], score)
        except BudgetExceeded as error:
            self.on("Agent", f"Budget stop: {error}.")
            answer = answer or "\n\n".join(output for output in outputs.values() if not output.startswith("Error:"))
            answer = answer or f"The budget ran out ({error}) before an answer was ready. Try again with a larger budget."
        except (LLMError, KeyError, TypeError) as error:  # KeyError/TypeError: provider ignored the JSON schema
            self.on("Agent", f"LLM error: {error!r}")
        answer = answer or "Sorry, I could not produce an answer."
        self.on("Agent", answer)
        self.on("Budget", self.budget.summary())
        self.history.append((question, answer))
        return answer
