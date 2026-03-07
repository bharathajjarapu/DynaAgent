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

        try:
            if "{" in tool_text and "}" in tool_text:
                start = tool_text.find("{")
                end = tool_text.rfind("}") + 1
                tool_data = json.loads(tool_text[start:end])

                if tool_data.get("needs_new_tools") and tool_data.get("tools"):
                    if callback:
                        callback("creating_tools", tool_data["tools"])

                    for tool_spec in tool_data["tools"]:
                        if tool_spec["name"] in self.tool_map:
                            continue

                        tool = await build(self.coordinator, self.tools, tool_spec)
                        if tool:
                            self.tool_map[tool_spec["name"]] = tool
                            self.coordinator.tools = self.tools
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
            jobs = []
            for agent_name, agent in list(self.agents.items()):
                if callback:
                    callback("working", f"Round {rnd}: {agent_name}")
                jobs.append(self.run(agent_name, agent, question, rnd))

            results = await asyncio.gather(*jobs)

            for item in results:
                self.mem["outs"][item["agent"]] = item["output"]

            if rnd < rounds:
                results_text = "\n".join(
                    [f"- {r['agent']}: {r['output']}" for r in results]
                )
                msg = await Runner.run(self.coordinator, note(question, results_text, rnd))
                self.mem["notes"].append(msg.final_output)

        if callback:
            callback("finalizing", None)

        results_text = "\n".join([f"- {r['agent']}: {r['output']}" for r in results])

        try:
            final = await Runner.run(self.coordinator, synthesize(results_text))
            answer = final.final_output
        except:
            answer = results_text if results else "No results"

        return answer, results


st.set_page_config(page_title="DynaAgent", layout="centered")

st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .agent {
        background: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }
    .agent-name {
        font-weight: 600;
        color: #000;
    }
    .agent-role {
        color: #666;
        font-size: 0.9rem;
    }
    .tool {
        background: #f8f9fa;
        padding: 0.75rem;
        margin: 0.4rem 0;
        border-radius: 4px;
        border: 1px solid #dee2e6;
    }
    .tool-name {
        font-weight: 600;
        color: #495057;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("DynaAgent")

question = st.text_input("Question", placeholder="Ask anything...")

if st.button("Submit"):
    if question:
        tool_area = st.empty()
        planning_area = st.empty()
        agents_area = st.empty()
        working_area = st.empty()
        result_area = st.empty()

        def update(phase, data):
            if phase == "tool_analysis":
                tool_area.info("Analyzing tools needed")

            elif phase == "creating_tools":
                tool_area.empty()
                with tool_area.container():
                    st.subheader("Tools Created")
                    for tool in data:
                        st.markdown(
                            f"""
                        <div class="tool">
                            <div class="tool-name">🔧 {tool["name"]}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

            elif phase == "planning":
                if not data:
                    tool_area.empty()
                planning_area.info("Planning agents")

            elif phase == "agents":
                planning_area.empty()
                with agents_area.container():
                    st.subheader("Agents")
                    for agent in data:
                        model_display = agent.get("model", "gpt-4.1-mini")
                        st.markdown(
                            f"""
                        <div class="agent">
                            <div class="agent-name">{agent["name"]}</div>
                            <div class="agent-role">{agent["role"]} • {model_display}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

            elif phase == "working":
                working_area.info(f"{data} working")

            elif phase == "finalizing":
                working_area.info("Finalizing")

        async def run():
            dynaagent = DynaAgent()
            answer, results = await dynaagent.execute(question, update)

            working_area.empty()

            with result_area.container():
                st.divider()
                st.subheader("Answer")
                st.write(answer)

                with st.expander("Details"):
                    for r in results:
                        st.markdown(f"**{r['agent']}**")
                        st.text(
                            r["output"][:400] + "..."
                            if len(r["output"]) > 400
                            else r["output"]
                        )

        asyncio.run(run())
