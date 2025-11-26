# DynaAgent

A simple dynamic multi-agent system that creates specialized AI agents on-the-fly to solve problems.

## What It Does

Instead of pre-defining agents, DynaAgent:
1. Analyzes your question
2. Creates specialized agents dynamically
3. Executes tasks using tool calling
4. Synthesizes results into a final answer

## Quick Start

### Install
```bash
pip install openai-agents python-dotenv
```

### Configure
Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

### Run
```bash
python main.py
```

## Example

**Input:**
```
What is 25 multiplied by 4?
```

**Output:**
```
=== Question ===
What is 25 multiplied by 4?

=== Planning ===
Plan: Creates CalculatorAgent

=== Creating Agents ===
✓ Created: CalculatorAgent (Calculator)

=== Executing ===
Running: CalculatorAgent...
🔧 Tool: calculator(*, 25.0, 4.0)
   Result: 100.0
✓ CalculatorAgent: 25 multiplied by 4 is 100.

=== Synthesis ===
Final Answer: 100
```

## How It Works

```
Question → Planning → Agent Creation → Execution → Synthesis → Answer
```

1. **Planning** - Coordinator analyzes question and plans agents
2. **Creation** - Specialized agents created with specific roles
3. **Execution** - Agents run and use tools as needed
4. **Synthesis** - Results combined into final answer

## Features

✅ **Dynamic Agent Generation** - Agents created based on task needs
✅ **Tool Calling** - calculator, search, analyze, request_agent
✅ **Multi-Agent Coordination** - Multiple agents work together
✅ **Runtime Agent Creation** - New agents spawned during execution
✅ **Simple Code** - Clean, minimal implementation

## Available Tools

- `calculator(operation, a, b)` - Math operations (+, -, *, /)
- `search(query)` - Web search simulation
- `analyze(data)` - Text statistics
- `request_agent(name, role, task)` - Create new agent during execution

## Usage Examples

### Interactive Mode
```bash
python main.py
# Enter your question when prompted
```

### Piped Input
```bash
echo "What is 15 * 8?" | python main.py
```

### Different Questions
```bash
echo "What is 100 divided by 5?" | python main.py
echo "Calculate 45 + 33" | python main.py
```

## Code Structure

```python
# Tools (4 functions)
calculator()     # Math operations
search()         # Web search
analyze()        # Text analysis
request_agent()  # Dynamic agent creation

# Core Class
class DynaAgent:
    create()     # Creates new agent
    execute()    # Main execution flow
```

## Requirements

- Python 3.10+
- openai-agents >= 0.6.1
- python-dotenv >= 1.2.1
- OpenAI API key

## Notes

- Uses `gpt-4o-mini` model
- 2-second delays between API calls (rate limiting)
- Free tier: 3 requests per minute
- Tool usage is printed for visibility

## Architecture

```
DynaAgent
    ├── Coordinator (plans & synthesizes)
    └── Dynamic Agents (created on-demand)
            ├── Agent 1 (specialized role)
            ├── Agent 2 (specialized role)
            └── ...
```

Each agent has access to all tools and executes independently.

## Example Scenarios

**Simple Math:**
- Question: "What is 25 * 4?"
- Agents: CalculatorAgent
- Tools Used: calculator(*, 25, 4)
- Answer: 100

**Multi-Agent:**
- Question: "Calculate 12 * 8 and analyze the result"
- Agents: CalculatorAgent, TextAnalysisAgent
- Tools Used: calculator, analyze
- Answer: 96 with analysis

**Dynamic Creation:**
- Question: Complex multi-step task
- Agents: Initial agents + runtime spawned agents
- Tools Used: Multiple tools as needed
- Answer: Comprehensive solution

## Troubleshooting

**Rate Limit Error?**
- Wait 20-30 seconds between runs
- Free tier allows 3 requests per minute

**No Output?**
- Check `.env` file exists
- Verify OPENAI_API_KEY is set correctly
- Ensure internet connection

**Import Error?**
- Run: `pip install openai-agents python-dotenv`

## License

MIT