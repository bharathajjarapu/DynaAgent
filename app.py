import asyncio
import json

import streamlit as st
from agents import Agent, Runner
from dotenv import load_dotenv

from prompts import analyze, judge, note, planner, route, synthesize, work
from tools import build, calculator, search

load_dotenv()

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
            except:
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
        except:
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

    def pack(self, results):
        out = []

        for item in results:
            lines = [f"Agent: {item['agent']}"]

            for msg in item.get("trace", []):
                if msg["kind"] == "tool":
                    lines.append(f"Tool: {msg['name']}")
                    if msg["inp"]:
                        lines.append(f"Input: {msg['inp']}")
                    if msg["out"]:
                        lines.append(f"Output: {msg['out']}")

            lines.append(f"Output: {item['output']}")
            out.append("\n".join(lines))

        return "\n\n".join(out)

    async def check(self, question: str, results_text: str, rnd: int):
        try:
            result = await Runner.run(self.coordinator, judge(question, results_text, rnd))
            text = result.final_output

            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                data = json.loads(text[start:end])
                done = bool(data.get("done"))
                note_text = data.get("note", "")
                return done, note_text
        except:
            pass

        if rnd > 1:
            return True, ""

        return False, "Fill gaps, verify facts, and improve the weak parts."

    async def pickpath(self, question: str):
        try:
            result = await Runner.run(self.coordinator, route(question))
            text = result.final_output

            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                data = json.loads(text[start:end])
                direct = bool(data.get("direct"))
                answer = data.get("answer", "")
                return direct, answer
        except:
            pass

        return False, ""

    async def run(self, name: str, agent, question: str, rnd: int, callback=None):
        prompt = work(name, self.info[name]["role"], question, self.snap(), rnd)
        if callback:
            callback("dispatch", {"rnd": rnd, "name": name, "prompt": prompt})

        try:
            result = await Runner.run(agent, prompt)
            item = {
                "agent": name,
                "output": result.final_output,
                "trace": self.read(result.new_items),
            }
            if callback:
                callback("agent_run", {"rnd": rnd, "name": name, "step": "done", "data": item})
            return item
        except Exception as e:
            item = {"agent": name, "output": f"Error: {str(e)}", "trace": []}
            if callback:
                callback("agent_run", {"rnd": rnd, "name": name, "step": "done", "data": item})
            return item

    async def execute(self, question: str, callback=None):
        if callback:
            callback("route", None)

        direct, answer = await self.pickpath(question)
        if direct and answer:
            if callback:
                callback("direct", answer)
                callback("answer", answer)
            return answer, []

        # Tool Analysis
        if callback:
            callback("tool_analysis", None)

        tool_result = await Runner.run(self.coordinator, analyze(question))
        tool_text = tool_result.final_output
        if callback:
            callback("tool_text", tool_text)

        try:
            tool_data = self.parsejson(tool_text, {})

            if tool_data.get("needs_new_tools") and tool_data.get("tools"):
                made = []
                pending = [
                    tool_spec
                    for tool_spec in tool_data["tools"]
                    if tool_spec["name"] not in self.tool_map
                ]

                if pending and callback:
                    callback("tool_needed", pending)

                for tool_spec in pending:
                    if callback:
                        callback(
                            "tool",
                            {
                                "name": tool_spec["name"],
                                "step": "making",
                                "desc": tool_spec.get("description", ""),
                            },
                        )
                    tool, code, prompt = await build(self.coordinator, self.tools, tool_spec)
                    if tool:
                        self.tool_map[tool_spec["name"]] = tool
                        self.coordinator.tools = self.tools
                        made.append(tool_spec)
                        if callback:
                            callback(
                                "tool",
                                {
                                    "name": tool_spec["name"],
                                    "step": "made",
                                    "desc": tool_spec.get("description", ""),
                                    "prompt": prompt,
                                    "code": code,
                                },
                            )
                    elif callback:
                        callback(
                            "tool",
                            {
                                "name": tool_spec["name"],
                                "step": "failed",
                                "desc": tool_spec.get("description", ""),
                                "prompt": prompt,
                                "code": code,
                            },
                        )

                if callback:
                    callback("tools", made)
        except:
            pass

        # Agent Planning
        if callback:
            callback("planning", None)

        plan_result = await Runner.run(self.coordinator, planner(question))
        plan_text = plan_result.final_output

        try:
            if "{" in plan_text and "}" in plan_text:
                start = plan_text.find("{")
                end = plan_text.rfind("}") + 1
                plan_data = json.loads(plan_text[start:end])
            else:
                raise ValueError("No JSON")
        except:
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
                "maxrounds": 1,
            }
        if callback:
            callback("plan", {"text": plan_text, "data": plan_data})

        agents_list = plan_data.get("agents", [])
        if not agents_list:
            agents_list = [
                {
                    "name": "Solver",
                    "role": "Problem Solver",
                    "instructions": "Solve",
                    "model": "gpt-5-mini",
                    "tools": ["calculator", "search"],
                }
            ]

        if callback:
            callback("agents", agents_list)

        for spec in agents_list:
            name = spec.get("name", "Worker")
            role = spec.get("role", "Assistant")
            instructions = spec.get("instructions", "Help solve the task")
            model = spec.get("model", "gpt-5-mini")
            tool_names = spec.get("tools")
            self.create(name, role, instructions, model, tool_names)

        maxrounds = plan_data.get("maxrounds", plan_data.get("max_rounds", plan_data.get("rounds", 2)))
        try:
            maxrounds = int(maxrounds)
        except:
            maxrounds = 2
        maxrounds = 1 if maxrounds < 1 else 3 if maxrounds > 3 else maxrounds

        results = []
        for rnd in range(1, maxrounds + 1):
            if callback:
                callback("working", f"Round {rnd}")

            jobs = []
            for agent_name, agent in list(self.agents.items()):
                jobs.append(self.run(agent_name, agent, question, rnd, callback))

            results = await asyncio.gather(*jobs)
            if callback:
                callback("round", {"rnd": rnd, "items": results})

            for item in results:
                self.mem["outs"][item["agent"]] = item["output"]

            if rnd < maxrounds:
                if callback:
                    callback("judge", {"rnd": rnd})

                full_text = self.pack(results)
                done, note_text = await self.check(question, full_text, rnd)
                if done:
                    if callback:
                        callback("enough", {"rnd": rnd})
                    break

                msg = await Runner.run(self.coordinator, note(question, full_text, rnd))
                note_text = msg.final_output
                self.mem["notes"].append(note_text)
                if callback:
                    callback("note", {"rnd": rnd, "text": note_text})

        if callback:
            callback("finalizing", None)

        results_text = "\n".join([f"- {r['agent']}: {r['output']}" for r in results])

        try:
            final = await Runner.run(self.coordinator, synthesize(results_text))
            answer = final.final_output
        except:
            answer = results_text if results else "No results"

        if callback:
            callback("answer", answer)

        return answer, results


st.set_page_config(page_title="DynaAgent", layout="centered")

st.title("DynaAgent")

if "msgs" not in st.session_state:
    st.session_state.msgs = []

if "hist" not in st.session_state:
    st.session_state.hist = []

if "step" not in st.session_state:
    st.session_state.step = "Ready"

top_box = st.empty()
chat_box = st.empty()

def add(who, text, av, kind="text", name="", info=None):
    st.session_state.msgs.append(
        {"who": who, "text": text, "av": av, "kind": kind, "name": name, "info": info or {}}
    )


def codelang(text):
    raw = (text or "").strip()
    if not raw:
        return "text"
    if raw[0] in ["{", "["]:
        return "json"
    if raw.startswith("def ") or raw.startswith("import ") or "\ndef " in raw:
        return "python"
    return "text"


def detailtitle(item):
    return item["info"].get("title") or item["name"] or "Details"

def ask(text):
    if not st.session_state.hist:
        return text

    out = ["Chat so far:"]
    for item in st.session_state.hist[-4:]:
        out.append(f"User: {item['q']}\nAnswer: {item['a']}")
    out.append(f"New user message: {text}")
    return "\n\n".join(out)

def draw():
    if st.session_state.step in ["Ready", "Done"]:
        top_box.empty()
    else:
        top_box.info(st.session_state.step)

    with chat_box.container():
        for item in st.session_state.msgs:
            role = "user" if item["who"] == "You" else "assistant"

            with st.chat_message(role, avatar=item["av"]):
                if item["who"] != "You":
                    st.markdown(f"**{item['who']}**")

                st.markdown(item["text"])
                if item["kind"] == "code" and item["info"].get("code"):
                    with st.expander(detailtitle(item), expanded=False):
                        st.code(
                            item["info"]["code"],
                            language=item["info"].get("lang", codelang(item["info"]["code"])),
                        )

draw()

question = st.chat_input("Ask anything...")

if question:
    add("You", question, "🙂")
    st.session_state.step = "Working..."
    draw()

    def update(phase, data):
        if phase == "route":
            st.session_state.step = "Working..."

        elif phase == "direct":
            st.session_state.step = "Working..."

        elif phase == "tool_analysis":
            st.session_state.step = "Working..."

        elif phase == "tool_needed":
            for item in data:
                desc = item.get("description", "").strip()
                text = f"I need to create `{item['name']}`."
                if desc:
                    text = f"{text} {desc}"
                add("Agent", text, "🧠")

        elif phase == "tool":
            if data["step"] == "made":
                add(
                    "Agent",
                    f"Created `{data['name']}`.",
                    "🧠",
                    "code",
                    data["name"],
                    {
                        "code": data.get("code", ""),
                        "lang": "python",
                        "title": f"{data['name']} code",
                    },
                )
            elif data["step"] == "failed":
                add(
                    "Agent",
                    f"Failed to create `{data['name']}`.",
                    "🧠",
                )

        elif phase == "planning":
            st.session_state.step = "Working..."

        elif phase == "plan":
            st.session_state.step = "Working..."

        elif phase == "agents":
            for item in data:
                tools = item.get("tools", [])
                extra = f" Tools: {', '.join(tools)}." if tools else ""
                add(
                    "Agent",
                    f"Created `{item['name']}`.{extra}",
                    "🧠",
                    "code",
                    item["name"],
                    {
                        "code": item.get("instructions", ""),
                        "lang": "text",
                        "title": f"{item['name']} prompt",
                    },
                )

        elif phase == "dispatch":
            add("Agent", f"Passed the task to `{data['name']}`.", "🧠")

        elif phase == "working":
            st.session_state.step = "Working..."

        elif phase == "agent_run":
            if data["step"] == "done":
                item = data["data"]
                for msg in item["trace"]:
                    if msg.get("inp"):
                        add(
                            data["name"],
                            f"Used `{msg['name']}` with this input.",
                            "🤖",
                            "code",
                            msg["name"],
                            {
                                "code": msg.get("inp", ""),
                                "lang": codelang(msg.get("inp", "")),
                                "title": f"{msg['name']} input",
                            },
                        )
                    else:
                        add(data["name"], f"Used `{msg['name']}`.", "🤖")
                    if msg.get("out"):
                        add(
                            data["name"],
                            f"`{msg['name']}` returned this output.",
                            "🤖",
                            "code",
                            msg["name"],
                            {
                                "code": msg.get("out", ""),
                                "lang": codelang(msg.get("out", "")),
                                "title": f"{msg['name']} output",
                            },
                        )
                add(data["name"], item["output"], "🤖")

        elif phase == "judge":
            st.session_state.step = "Working..."

        elif phase == "enough":
            st.session_state.step = "Working..."

        elif phase == "note":
            add("Agent", data["text"], "🧠")

        elif phase == "finalizing":
            st.session_state.step = "Working..."

        elif phase == "answer":
            st.session_state.step = "Done"
            add("Agent", data, "🧠")

        draw()

    async def go():
        dynaagent = DynaAgent()
        return await dynaagent.execute(ask(question), update)

    answer, _ = asyncio.run(go())
    st.session_state.hist.append({"q": question, "a": answer})