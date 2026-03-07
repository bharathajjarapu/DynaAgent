import asyncio
import json

import streamlit as st
from agents import Agent, Runner
from dotenv import load_dotenv

from prompts import analyze, note, planner, synthesize, work
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
            model="gpt-4.1-mini",
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
        model: str = "gpt-4.1-mini",
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

    async def run(self, name: str, agent, question: str, rnd: int):
        prompt = work(name, self.info[name]["role"], question, self.snap(), rnd)

        try:
            result = await Runner.run(agent, prompt)
            return {"agent": name, "output": result.final_output}
        except Exception as e:
            return {"agent": name, "output": f"Error: {str(e)}"}

    async def execute(self, question: str, callback=None):
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

                        tool = await build(self.coordinator, self.tools, tool_spec)
                        if tool:
                            self.tool_map[tool_spec["name"]] = tool
                            self.coordinator.tools = self.tools
                            made.append(tool_spec)

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
                        "model": "gpt-4.1-mini",
                        "tools": ["calculator", "search"],
                    }
                ],
                "rounds": 1,
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
                    "model": "gpt-4.1-mini",
                    "tools": ["calculator", "search"],
                }
            ]

        if callback:
            callback("agents", agents_list)

        for spec in agents_list:
            name = spec.get("name", "Worker")
            role = spec.get("role", "Assistant")
            instructions = spec.get("instructions", "Help solve the task")
            model = spec.get("model", "gpt-4.1-mini")
            tool_names = spec.get("tools")
            self.create(name, role, instructions, model, tool_names)

        rounds = plan_data.get("rounds", 2)
        try:
            rounds = int(rounds)
        except:
            rounds = 2
        rounds = 1 if rounds < 1 else 2 if rounds > 2 else rounds

        results = []
        for rnd in range(1, rounds + 1):
            if callback:
                callback("working", f"Round {rnd}")

            jobs = []
            for agent_name, agent in list(self.agents.items()):
                jobs.append(self.run(agent_name, agent, question, rnd))

            results = await asyncio.gather(*jobs)
            if callback:
                callback("round", {"rnd": rnd, "items": results})

            for item in results:
                self.mem["outs"][item["agent"]] = item["output"]

            if rnd < rounds:
                results_text = "\n".join(
                    [f"- {r['agent']}: {r['output']}" for r in results]
                )
                msg = await Runner.run(self.coordinator, note(question, results_text, rnd))
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

question = st.text_input("Question", placeholder="Ask anything...")

if st.button("Submit"):
    if question:
        top_box = st.empty()
        tool_box = st.empty()
        plan_box = st.empty()
        agent_box = st.empty()
        work_box = st.empty()
        ans_box = st.empty()

        mem = {
            "step": "Starting",
            "tool": "",
            "tools": [],
            "plan": None,
            "agents": [],
            "runs": [],
            "notes": {},
            "ans": "",
        }

        def draw():
            top_box.info(mem["step"])

            with tool_box.container():
                st.subheader("Tools")
                if mem["tool"]:
                    with st.expander("Tool analysis", expanded=False):
                        st.text(mem["tool"])
                if mem["tools"]:
                    for item in mem["tools"]:
                        st.write(f"- {item['name']}")
                elif mem["tool"]:
                    st.write("No new tools")

            with plan_box.container():
                st.subheader("Plan")
                if mem["plan"]:
                    data = mem["plan"]["data"]
                    for item in data.get("plan", []):
                        st.write(f"- {item}")
                    st.caption(f"Rounds: {data.get('rounds', 1)}")
                    with st.expander("Raw plan", expanded=False):
                        st.text(mem["plan"]["text"])

            with agent_box.container():
                st.subheader("Agents")
                for item in mem["agents"]:
                    st.markdown(f"**{item['name']}**")
                    st.caption(f"{item['role']} - {item.get('model', 'gpt-4.1-mini')}")
                    if item.get("tools"):
                        st.write(f"Tools: {', '.join(item['tools'])}")

            with work_box.container():
                st.subheader("Run")
                for item in mem["runs"]:
                    with st.expander(f"Round {item['rnd']}", expanded=True):
                        for out in item["items"]:
                            st.markdown(f"**{out['agent']}**")
                            st.text(out["output"])
                        if item["rnd"] in mem["notes"]:
                            st.markdown("**Note**")
                            st.text(mem["notes"][item["rnd"]])

            with ans_box.container():
                if mem["ans"]:
                    st.divider()
                    st.subheader("Answer")
                    st.write(mem["ans"])

        def update(phase, data):
            if phase == "tool_analysis":
                mem["step"] = "Analyzing tools"

            elif phase == "tool_text":
                mem["tool"] = data

            elif phase == "tools":
                mem["tools"] = data

            elif phase == "planning":
                mem["step"] = "Planning agents"

            elif phase == "plan":
                mem["plan"] = data

            elif phase == "agents":
                mem["agents"] = data

            elif phase == "working":
                mem["step"] = f"{data} running"

            elif phase == "round":
                mem["runs"].append(data)

            elif phase == "note":
                mem["notes"][data["rnd"]] = data["text"]

            elif phase == "finalizing":
                mem["step"] = "Finalizing"

            elif phase == "answer":
                mem["ans"] = data
                mem["step"] = "Done"

            draw()

        async def run():
            dynaagent = DynaAgent()
            await dynaagent.execute(question, update)

        draw()
        asyncio.run(run())
    else:
        st.warning("Enter a question")
