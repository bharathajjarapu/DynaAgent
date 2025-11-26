import asyncio
import json

import streamlit as st
from agents import Agent, Runner
from dotenv import load_dotenv

from tools import calculator, search

load_dotenv()

class DynaAgent:
    def __init__(self):
        self.agents = {}
        self.tools = [calculator, search]
        self.coordinator = Agent(
            name="Coordinator",
            instructions="You are a coordinator that plans and synthesizes results",
            tools=self.tools,
            model="gpt-4.1-mini",
        )

    def create(self, name: str, role: str, instructions: str):
        agent = Agent(
            name=name,
            instructions=f"You are a {role}. {instructions}",
            tools=self.tools,
            model="gpt-4.1-mini",
        )
        self.agents[name] = agent
        return agent

    async def execute(self, question: str, callback=None):
        if callback:
            callback("planning", None)

        plan_prompt = f'Analyze this task and determine what specialized agents are needed: {question}. Make sure the Names are Single Word like WebSearchAgent, MathAgent... Return ONLY valid JSON in this exact format: {{"agents":[{{"name":"AgentName","role":"RoleDescription","instructions":"What agent should do"}}],"plan":["step1","step2"]}}'

        plan_result = await Runner.run(self.coordinator, plan_prompt)
        plan_text = plan_result.final_output

        try:
            if "{" in plan_text and "}" in plan_text:
                start = plan_text.find("{")
                end = plan_text.rfind("}") + 1
                json_str = plan_text[start:end]
                plan_data = json.loads(json_str)
            else:
                raise ValueError("No JSON")
        except:
            plan_data = {
                "agents": [
                    {
                        "name": "Solver",
                        "role": "Problem Solver",
                        "instructions": "Solve the task",
                    }
                ],
            }

        agents_list = plan_data.get("agents", [])
        if not agents_list:
            agents_list = [
                {"name": "Solver", "role": "Problem Solver", "instructions": "Solve"}
            ]

        if callback:
            callback("agents", agents_list)

        for spec in agents_list:
            name = spec.get("name", "Worker")
            role = spec.get("role", "Assistant")
            instructions = spec.get("instructions", "Help solve the task")
            self.create(name, role, instructions)

        results = []
        for agent_name, agent in list(self.agents.items()):
            if callback:
                callback("working", agent_name)

            try:
                result = await Runner.run(agent, question)
                output = result.final_output
                results.append({"agent": agent_name, "output": output})
            except Exception as e:
                results.append({"agent": agent_name, "output": f"Error: {str(e)}"})

        if callback:
            callback("finalizing", None)

        results_summary = "\n".join([f"- {r['agent']}: {r['output']}" for r in results])
        synthesis_prompt = f"Based on these agent results, provide ONLY the final answer:\n{results_summary}\n\nFinal answer:"

        try:
            final = await Runner.run(self.coordinator, synthesis_prompt)
            answer = final.final_output
        except:
            answer = results_summary if results else "No results"

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
</style>
""",
    unsafe_allow_html=True,
)

st.title("DynaAgent")

question = st.text_input("Question", placeholder="Ask anything...")

if st.button("Submit"):
    if question:
        planning_area = st.empty()
        agents_area = st.empty()
        working_area = st.empty()
        result_area = st.empty()

        def update(phase, data):
            if phase == "planning":
                planning_area.info("Planning")

            elif phase == "agents":
                planning_area.empty()
                with agents_area.container():
                    st.subheader("Agents")
                    for agent in data:
                        st.markdown(
                            f"""
                        <div class="agent">
                            <div class="agent-name">{agent["name"]}</div>
                            <div class="agent-role">{agent["role"]}</div>
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
