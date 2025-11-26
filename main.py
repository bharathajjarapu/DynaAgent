import asyncio
import json
import sys

from agents import Agent
from dotenv import load_dotenv

from tools import calculator, search

load_dotenv()

# Chitti 2.0
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

    async def execute(self, question: str):
        print(f"\n=== Question ===\n{question}\n")

        # Planning
        print("=== Planning ===")
        plan_prompt = f'Analyze this task and determine what specialized agents are needed: {question}. Make sure the Names are Single Word like WebSearchAgent, MathAgent... Return ONLY valid JSON in this exact format: {{"agents":[{{"name":"AgentName","role":"RoleDescription","instructions":"What agent should do"}}],"plan":["step1","step2"]}}'

        from agents import Runner

        plan_result = await Runner.run(self.coordinator, plan_prompt)
        plan_text = plan_result.final_output
        print(f"Plan: {plan_text}\n")

        # Parse plan
        try:
            if "{" in plan_text and "}" in plan_text:
                start = plan_text.find("{")
                end = plan_text.rfind("}") + 1
                json_str = plan_text[start:end]
                plan_data = json.loads(json_str)
            else:
                raise ValueError("No JSON found")
        except:
            plan_data = {
                "agents": [
                    {
                        "name": "Solver",
                        "role": "Problem Solver",
                        "instructions": "Solve the given task using available tools",
                    }
                ],
                "plan": ["Analyze question", "Execute solution"],
            }

        # Creating agents
        print("=== Creating Agents ===")
        agents_created = plan_data.get("agents", [])
        if not agents_created:
            agents_created = [
                {
                    "name": "Solver",
                    "role": "Problem Solver",
                    "instructions": "Solve the task",
                }
            ]

        for spec in agents_created:
            name = spec.get("name", "Worker")
            role = spec.get("role", "Assistant")
            instructions = spec.get("instructions", "Help solve the task")
            self.create(name, role, instructions)
            print(f"✓ Created: {name} ({role})")
        print()

        # Executing
        print("=== Executing ===")
        results = []
        agent_items = list(self.agents.items())
        for agent_name, agent in agent_items:
            print(f"Running: {agent_name}...")
            try:
                await asyncio.sleep(2)
                result = await Runner.run(agent, question)
                output = result.final_output
                results.append({"agent": agent_name, "output": output})
                print(f"✓ {agent_name}: {output[:100]}...\n")
            except Exception as e:
                print(f"✗ {agent_name} failed: {e}")
                results.append({"agent": agent_name, "output": f"Error: {str(e)}"})

        # Synthesis
        print("=== Synthesis ===")
        results_summary = "\n".join([f"- {r['agent']}: {r['output']}" for r in results])
        synthesis_prompt = f"Based on these agent results, provide ONLY the final answer (no JSON, no formatting, just the direct answer):\n{results_summary}\n\nFinal answer:"

        try:
            await asyncio.sleep(2)
            final = await Runner.run(self.coordinator, synthesis_prompt)
            answer = final.final_output
            print(f"Final Answer: {answer}\n")
        except Exception as e:
            print(f"Synthesis failed: {e}")
            answer = results_summary if results else "No results available"
            print(f"Final Answer (from raw results): {answer}\n")

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