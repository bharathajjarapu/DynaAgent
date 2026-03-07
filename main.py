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

    async def run(self, name: str, agent, question: str, rnd: int):
        print(f"Running: {name} (Round {rnd})...")
        prompt = work(name, self.info[name]["role"], question, self.snap(), rnd)

        try:
            if rnd > 1:
                await asyncio.sleep(2)
            result = await Runner.run(agent, prompt)
            output = result.final_output
            print(f"✓ {name}: {output[:100]}...\n")
            return {"agent": name, "output": output}
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            return {"agent": name, "output": f"Error: {str(e)}"}

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
        print(f"\n=== Question ===\n{question}\n")

        print("=== Route Check ===")
        direct, answer = await self.pickpath(question)
        if direct and answer:
            print("Direct reply selected\n")
            print(f"Final Answer: {answer}\n")
            return answer

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
                        if tool_spec["name"] in self.tool_map:
                            print(f"✓ Reusing: {tool_spec['name']}")
                            continue
                        print(f"Creating: {tool_spec['name']}...")
                        tool, code, prompt = await build(self.coordinator, self.tools, tool_spec)
                        if tool:
                            self.tool_map[tool_spec["name"]] = tool
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
                        "model": "gpt-5-mini",
                        "tools": ["calculator", "search"],
                    }
                ],
                "plan": ["Analyze and solve"],
                "maxrounds": 1,
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
            reusing = name in self.agents
            self.create(name, role, instructions, model, tool_names)
            action = "Reused" if reusing else "Created"
            print(f"✓ {action}: {name} ({role})")
        print()

        maxrounds = plan_data.get("maxrounds", plan_data.get("max_rounds", plan_data.get("rounds", 2)))
        try:
            maxrounds = int(maxrounds)
        except Exception:
            maxrounds = 2
        maxrounds = 1 if maxrounds < 1 else 3 if maxrounds > 3 else maxrounds

        print("=== Executing ===")
        results = []
        for rnd in range(1, maxrounds + 1):
            print(f"--- Round {rnd} ---")
            jobs = [
                self.run(agent_name, agent, question, rnd)
                for agent_name, agent in list(self.agents.items())
            ]
            results = await asyncio.gather(*jobs)

            for item in results:
                self.mem["outs"][item["agent"]] = item["output"]

            if rnd < maxrounds:
                print("=== Checking Round ===")
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
                    print("Round is enough\n")
                    break

                print("=== Coordinator Note ===")
                msg = await Runner.run(self.coordinator, note(question, results_text, rnd))
                note_text = msg.final_output
                self.mem["notes"].append(note_text)
                print(f"{note_text}\n")

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
