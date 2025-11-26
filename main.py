import sys
import json
import asyncio

from dotenv import load_dotenv
from agents import Agent, Runner

from tools import build, calculator, search
from prompts import analyze, planner, synthesize

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

    def create(
        self, name: str, role: str, instructions: str, model: str = "gpt-4.1-mini"
    ):
        agent = Agent(
            name=name,
            instructions=f"You are a {role}. {instructions}",
            tools=self.tools,
            model=model,
        )
        self.agents[name] = agent
        return agent

    async def execute(self, question: str):
        print(f"\n=== Question ===\n{question}\n")

        # Tool Analysis
        print("=== Tool Analysis ===")
        tool_result = await Runner.run(self.coordinator, analyze(question))
        tool_text = tool_result.final_output
        print(f"Tool Analysis: {tool_text}\n")

        try:
            if "{" in tool_text and "}" in tool_text:
                start = tool_text.find("{")
                end = tool_text.rfind("}") + 1
                tool_data = json.loads(tool_text[start:end])

                if tool_data.get("needs_new_tools") and tool_data.get("tools"):
                    print("=== Creating Tools ===")
                    for tool_spec in tool_data["tools"]:
                        print(f"Creating: {tool_spec['name']}...")
                        success = await build(self.coordinator, self.tools, tool_spec)
                        if success:
                            self.coordinator.tools = self.tools
                            print(f"✓ Created: {tool_spec['name']}")
                    print()
        except Exception as e:
            print(f"Tool creation skipped: {e}\n")

        # Agent Planning
        print("=== Agent Planning ===")
        plan_result = await Runner.run(self.coordinator, planner(question))
        plan_text = plan_result.final_output
        print(f"Plan: {plan_text}\n")

        try:
            if "{" in plan_text and "}" in plan_text:
                start = plan_text.find("{")
                end = plan_text.rfind("}") + 1
                plan_data = json.loads(plan_text[start:end])
            else:
                raise ValueError("No JSON found")
        except Exception:
            plan_data = {
                "agents": [
                    {
                        "name": "Solver",
                        "role": "Problem Solver",
                        "instructions": "Solve the task",
                    }
                ],
                "plan": ["Analyze and solve"],
            }

        # Creating Agents
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
            model = spec.get("model", "gpt-4.1-mini")
            self.create(name, role, instructions, model)
            print(f"✓ Created: {name} ({role})")
        print()

        # Executing
        print("=== Executing ===")
        results = []
        for agent_name, agent in list(self.agents.items()):
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
        results = "\n".join([f"- {r['agent']}: {r['output']}" for r in results])

        try:
            await asyncio.sleep(2)
            final = await Runner.run(self.coordinator, synthesize(results))
            answer = final.final_output
            print(f"Final Answer: {answer}\n")
        except Exception as e:
            print(f"Synthesis failed: {e}")
            answer = results if results else "No results available"
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
