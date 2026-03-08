import sys
import json
import asyncio

from dotenv import load_dotenv
from agents import Agent, Runner

from tools import build, calculator, search
from prompts import analyze, judge, note, planner, route, synthesize, work

load_dotenv()

# Chitti 2.0
class DynaAgent:
    def __init__(self):
        self.agents = {}
        self.info = {}
        self.mem = {"notes": [], "outs": {}}
        self.tool_map = {"calculator": calculator, "search": search}
        self.tools = list(self.tool_map.values())
        self.coordinator = Agent(
            name="Agent",
            instructions="You are a coordinator that plans and synthesizes results",
            tools=self.tools,
            model="gpt-5.4",
        )

    def pick(self, names=None):
        if not names:
            return self.tools

        out = []
        for name in names:
            tool = self.tool_map.get(name)
            if tool:
                out.append(tool)

        return out if out else self.tools

    def create(
        self,
        name: str,
        role: str,
        instructions: str,
        model: str = "gpt-5-mini",
        tool_names=None,
    ):
        tools = self.pick(tool_names)

        if name in self.agents:
            agent = self.agents[name]
            agent.instructions = f"You are a {role}. {instructions}"
            agent.tools = tools
            agent.model = model
            self.info[name] = {"role": role, "tools": tool_names or []}
            return agent

        agent = Agent(
            name=name,
            instructions=f"You are a {role}. {instructions}",
            tools=tools,
            model=model,
        )
        self.agents[name] = agent
        self.info[name] = {"role": role, "tools": tool_names or []}
        return agent

    def snap(self):
        note_text = "\n".join(self.mem["notes"][-2:])
        out_text = "\n".join(
            [f"- {name}: {out}" for name, out in self.mem["outs"].items()]
        )

        return f"Shared Notes:\n{note_text or 'None'}\n\nKnown Outputs:\n{out_text or 'None'}"

    def fmt(self, text):
        if text is None:
            return ""

        text = str(text)
        raw = text.strip()
        if not raw:
            return ""

        if raw[0] in ["{", "["]:
            try:
                return json.dumps(json.loads(raw), indent=2)
            except Exception:
                return text

        return text

    def parsejson(self, text, default=None):
        if text is None:
            return default

        raw = str(text)
        if "{" not in raw or "}" not in raw:
            return default

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            return default

    def callid(self, raw):
        if isinstance(raw, dict):
            return raw.get("call_id", "")

        return getattr(raw, "call_id", "")

    def toolname(self, raw):
        if isinstance(raw, dict):
            name = raw.get("name") or raw.get("tool_name")
            kind = raw.get("type", "")
        else:
            name = getattr(raw, "name", None) or getattr(raw, "tool_name", None)
            kind = getattr(raw, "type", "")

        if name:
            return name

        if "web_search" in kind or "search" in kind:
            return "search"

        if kind.endswith("_call"):
            kind = kind[:-5]
        if kind.endswith("_output"):
            kind = kind[:-7]

        return kind.replace("_", " ") or "tool"

    def toolin(self, raw):
        if isinstance(raw, dict):
            text = raw.get("arguments") or raw.get("input") or raw.get("query") or ""
        else:
            text = (
                getattr(raw, "arguments", None)
                or getattr(raw, "input", None)
                or getattr(raw, "query", None)
                or ""
            )

        return self.fmt(text)

    def toolkind(self, raw):
        if isinstance(raw, dict):
            kind = raw.get("type", "")
        else:
            kind = getattr(raw, "type", "")

        return kind.replace("_", " ") or "tool"

    def read(self, items):
        out = []
        ids = {}
        last = None

        for item in items:
            kind = getattr(item, "type", "")

            if kind == "tool_call_item":
                raw = item.raw_item
                name = self.toolname(raw)
                call_id = self.callid(raw)
                one = {
                    "kind": "tool",
                    "name": name,
                    "inp": self.toolin(raw),
                    "out": "",
                    "proc": self.toolkind(raw),
                }
                if call_id:
                    ids[call_id] = one
                out.append(one)
                last = one

            elif kind == "tool_call_output_item":
                raw = item.raw_item
                call_id = self.callid(raw)
                one = ids.get(call_id, last)
                if one:
                    one["name"] = one["name"] or self.toolname(raw)
                    one["out"] = self.fmt(item.output)
                else:
                    out.append(
                        {
                            "kind": "tool",
                            "name": self.toolname(raw),
                            "inp": "",
                            "out": self.fmt(item.output),
                            "proc": self.toolkind(raw),
                        }
                    )

        return out

    def show(self, who: str, text: str = "", code: str = ""):
        print(f"{who}: {text}")
        if code:
            print(code)
        print()

    async def run(self, name: str, agent, question: str, rnd: int):
        prompt = work(name, self.info[name]["role"], question, self.snap(), rnd)
        self.show("Agent", f"Passed the task to `{name}`.")

        try:
            result = await Runner.run(agent, prompt)
            item = {
                "agent": name,
                "output": result.final_output,
                "trace": self.read(result.new_items),
            }

            for msg in item["trace"]:
                if msg.get("inp"):
                    self.show(name, f"Used `{msg['name']}` with this input.", msg["inp"])
                else:
                    self.show(name, f"Used `{msg['name']}`.")
                if msg.get("out"):
                    self.show(name, f"`{msg['name']}` returned this output.", msg["out"])

            self.show(name, item["output"])
            return item
        except Exception as e:
            item = {"agent": name, "output": f"Error: {str(e)}", "trace": []}
            self.show(name, item["output"])
            return item

    async def pickpath(self, question: str):
        try:
            result = await Runner.run(self.coordinator, route(question))
            text = result.final_output

            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                data = json.loads(text[start:end])
                return bool(data.get("direct")), data.get("answer", "")
        except Exception:
            pass

        return False, ""

    async def execute(self, question: str):
        direct, answer = await self.pickpath(question)
        if direct and answer:
            self.show("Agent", answer)
            return answer

        tool_result = await Runner.run(self.coordinator, analyze(question))
        tool_text = tool_result.final_output

        try:
            tool_data = self.parsejson(tool_text, {})
            if tool_data.get("needs_new_tools") and tool_data.get("tools"):
                pending = [
                    tool_spec
                    for tool_spec in tool_data["tools"]
                    if tool_spec["name"] not in self.tool_map
                ]
                for tool_spec in pending:
                    desc = tool_spec.get("description", "").strip()
                    text = f"I need to create `{tool_spec['name']}`."
                    if desc:
                        text = f"{text} {desc}"
                    self.show("Agent", text)
                    tool, code, _ = await build(self.coordinator, self.tools, tool_spec)
                    if tool:
                        self.tool_map[tool_spec["name"]] = tool
                        self.coordinator.tools = self.tools
                        self.show("Agent", f"Created `{tool_spec['name']}`.", code)
                    else:
                        self.show("Agent", f"Failed to create `{tool_spec['name']}`.")
        except Exception as e:
            self.show("Agent", f"Tool creation skipped: {e}")

        plan_result = await Runner.run(self.coordinator, planner(question))
        plan_text = plan_result.final_output

        try:
            plan_data = self.parsejson(plan_text, None)
            if not plan_data:
                raise ValueError("No JSON found")
        except Exception:
            plan_data = {
                "agents": [
                    {
                        "name": "Solver",
                        "role": "Problem Solver",
                        "instructions": "Solve the task",
                        "model": "gpt-5-mini",
                        "tools": ["calculator", "search"],
                    }
                ],
                "plan": ["Analyze and solve"],
                "maxrounds": 1,
            }

        agents_created = plan_data.get("agents", [])
        if not agents_created:
            agents_created = [
                {
                    "name": "Solver",
                    "role": "Problem Solver",
                    "instructions": "Solve the task",
                    "model": "gpt-5-mini",
                    "tools": ["calculator", "search"],
                }
            ]

        for spec in agents_created:
            name = spec.get("name", "Worker")
            role = spec.get("role", "Assistant")
            instructions = spec.get("instructions", "Help solve the task")
            model = spec.get("model", "gpt-5-mini")
            tool_names = spec.get("tools")
            self.create(name, role, instructions, model, tool_names)
            tool_text = f" Tools: {', '.join(tool_names)}." if tool_names else ""
            self.show("Agent", f"Created `{name}`.{tool_text}", instructions)

        maxrounds = plan_data.get("maxrounds", plan_data.get("max_rounds", plan_data.get("rounds", 2)))
        try:
            maxrounds = int(maxrounds)
        except Exception:
            maxrounds = 2
        maxrounds = 1 if maxrounds < 1 else 3 if maxrounds > 3 else maxrounds

        results = []
        for rnd in range(1, maxrounds + 1):
            jobs = [
                self.run(agent_name, agent, question, rnd)
                for agent_name, agent in list(self.agents.items())
            ]
            results = await asyncio.gather(*jobs)

            for item in results:
                self.mem["outs"][item["agent"]] = item["output"]

            if rnd < maxrounds:
                results_text = "\n".join(
                    [f"- {r['agent']}: {r['output']}" for r in results]
                )
                done = False
                try:
                    check = await Runner.run(self.coordinator, judge(question, results_text, rnd))
                    text = check.final_output
                    if "{" in text and "}" in text:
                        start = text.find("{")
                        end = text.rfind("}") + 1
                        data = json.loads(text[start:end])
                        done = bool(data.get("done"))
                except Exception:
                    done = rnd > 1

                if done:
                    break

                msg = await Runner.run(self.coordinator, note(question, results_text, rnd))
                note_text = msg.final_output
                self.mem["notes"].append(note_text)
                self.show("Agent", note_text)

        results = "\n".join([f"- {r['agent']}: {r['output']}" for r in results])

        try:
            final = await Runner.run(self.coordinator, synthesize(results))
            answer = final.final_output
            self.show("Agent", answer)
        except Exception as e:
            answer = results if results else "No results available"
            self.show("Agent", f"Synthesis failed: {e}")
            self.show("Agent", answer)

        return answer


async def main():
    dynaagent = DynaAgent()

    if sys.stdin.isatty():
        question = input("Question: ").strip()
    else:
        question = sys.stdin.read().strip()

    if not question:
        question = "What is 25 multiplied by 4?"
        print(f"Using default question: {question}")

    await dynaagent.execute(question)


if __name__ == "__main__":
    asyncio.run(main())