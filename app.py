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
            name="Coordinator",
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
            callback("agent_run", {"rnd": rnd, "name": name, "step": "start"})

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
            if "{" in tool_text and "}" in tool_text:
                start = tool_text.find("{")
                end = tool_text.rfind("}") + 1
                tool_data = json.loads(tool_text[start:end])

                if tool_data.get("needs_new_tools") and tool_data.get("tools"):
                    made = []

                    for tool_spec in tool_data["tools"]:
                        if tool_spec["name"] in self.tool_map:
                            continue

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
st.caption("Research multi-agent app")

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

def ask(text):
    if not st.session_state.hist:
        return text

    out = ["Chat so far:"]
    for item in st.session_state.hist[-4:]:
        out.append(f"User: {item['q']}\nAnswer: {item['a']}")
    out.append(f"New user message: {text}")
    return "\n\n".join(out)

def draw():
    top_box.info(st.session_state.step)

    with chat_box.container():
        for item in st.session_state.msgs:
            role = "user" if item["who"] == "You" else "assistant"

            with st.chat_message(role, avatar=item["av"]):
                if item["who"] != "You":
                    st.markdown(f"**{item['who']}**")

                if item["kind"] == "call":
                    with st.expander(f"Tool input - {item['name']}", expanded=False):
                        st.caption("Input")
                        if item["text"]:
                            st.code(item["text"])
                elif item["kind"] == "tool":
                    with st.expander(f"Tool call - {item['name']}", expanded=False):
                        if item["info"].get("proc"):
                            st.caption(f"Process: {item['info']['proc']}")
                        if item["info"].get("inp"):
                            st.caption("Input")
                            st.code(item["info"]["inp"])
                        if item["info"].get("out"):
                            st.caption("Output")
                            st.code(item["info"]["out"])
                elif item["kind"] == "agent":
                    st.markdown(item["text"])
                    with st.expander(f"Agent setup - {item['name']}", expanded=False):
                        st.caption(f"Role: {item['info'].get('role', '')}")
                        st.caption(f"Model: {item['info'].get('model', '')}")
                        tools = item["info"].get("tools", [])
                        if tools:
                            st.caption(f"Tools: {', '.join(tools)}")
                        if item["info"].get("ins"):
                            st.caption("Prompt")
                            st.code(item["info"]["ins"])
                elif item["kind"] == "make":
                    st.markdown(item["text"])
                    with st.expander(f"Tool setup - {item['name']}", expanded=False):
                        if item["info"].get("desc"):
                            st.caption("Description")
                            st.write(item["info"]["desc"])
                        if item["info"].get("prompt"):
                            st.caption("Prompt")
                            st.code(item["info"]["prompt"])
                        if item["info"].get("code"):
                            st.caption("Code")
                            st.code(item["info"]["code"], language="python")
                elif item["kind"] == "out":
                    with st.expander(f"Tool output - {item['name']}", expanded=False):
                        st.caption("Output")
                        st.code(item["text"])
                else:
                    st.markdown(item["text"])

draw()

question = st.chat_input("Ask anything...")

if question:
    box = {"made": False, "tool_done": False}

    add("You", question, "🙂")
    st.session_state.step = "Starting"
    draw()

    def update(phase, data):
        if phase == "route":
            st.session_state.step = "Checking path"
            add("Coordinator", "Checking if this needs research or a direct reply.", "🧠")

        elif phase == "direct":
            st.session_state.step = "Direct reply"
            add("Coordinator", "Direct reply is enough here.", "🧠")

        elif phase == "tool_analysis":
            st.session_state.step = "Analyzing tools"
            add("Coordinator", "Checking what tools this task needs.", "🧠")

        elif phase == "tool":
            box["made"] = True
            if data["step"] == "making":
                add(
                    "Tools",
                    f"Creating `{data['name']}`.",
                    "🛠️",
                    "make",
                    data["name"],
                    {"desc": data.get("desc", "")},
                )
            elif data["step"] == "made":
                add(
                    "Tools",
                    f"Created `{data['name']}`.",
                    "🛠️",
                    "make",
                    data["name"],
                    {
                        "desc": data.get("desc", ""),
                        "prompt": data.get("prompt", ""),
                        "code": data.get("code", ""),
                    },
                )
            elif data["step"] == "failed":
                add(
                    "Tools",
                    f"Failed to create `{data['name']}`.",
                    "🛠️",
                    "make",
                    data["name"],
                    {
                        "desc": data.get("desc", ""),
                        "prompt": data.get("prompt", ""),
                        "code": data.get("code", ""),
                    },
                )

        elif phase == "planning":
            st.session_state.step = "Planning agents"
            if not box["tool_done"]:
                if not box["made"]:
                    add("Tools", "No new tools needed.", "🛠️")
                box["tool_done"] = True
            add("Coordinator", "Planning the agent team.", "🧠")

        elif phase == "plan":
            plan = data["data"].get("plan", [])
            text = "\n".join([f"- {item}" for item in plan])
            text = text or "Plan ready."
            maxrounds = data["data"].get("maxrounds", data["data"].get("max_rounds", data["data"].get("rounds", 1)))
            text = f"{text}\n\nMax rounds: {maxrounds}"
            add("Coordinator", text, "🧠")

        elif phase == "agents":
            for item in data:
                add(
                    item["name"],
                    f"Created `{item['name']}`.",
                    "🤖",
                    "agent",
                    item["name"],
                    {
                        "role": item.get("role", ""),
                        "model": item.get("model", ""),
                        "tools": item.get("tools", []),
                        "ins": item.get("instructions", ""),
                    },
                )

        elif phase == "working":
            st.session_state.step = f"{data} running"
            add("Coordinator", f"{data} started.", "🧠")

        elif phase == "agent_run":
            if data["step"] == "start":
                add(data["name"], f"Working in round {data['rnd']}.", "🤖")
            elif data["step"] == "done":
                item = data["data"]
                for msg in item["trace"]:
                    add(
                        data["name"],
                        f"Used `{msg['name']}`.",
                        "🤖",
                        "tool",
                        msg["name"],
                        {
                            "proc": msg.get("proc", ""),
                            "inp": msg.get("inp", ""),
                            "out": msg.get("out", ""),
                        },
                    )
                add(data["name"], item["output"], "🤖")

        elif phase == "judge":
            st.session_state.step = "Checking round"
            add("Coordinator", f"Checking round {data['rnd']}.", "🧠")

        elif phase == "enough":
            add("Coordinator", f"Round {data['rnd']} is enough.", "🧠")

        elif phase == "note":
            add("Coordinator", data["text"], "🧠")

        elif phase == "finalizing":
            st.session_state.step = "Finalizing"
            add("Coordinator", "Writing the final answer.", "🧠")

        elif phase == "answer":
            st.session_state.step = "Done"
            add("Answer", data, "✅")

        draw()

    async def go():
        dynaagent = DynaAgent()
        return await dynaagent.execute(ask(question), update)

    answer, _ = asyncio.run(go())
    st.session_state.hist.append({"q": question, "a": answer})
