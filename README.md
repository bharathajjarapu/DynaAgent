# DynaAgent

A dynamic multi-agent system that creates specialized AI agents on-the-fly to solve problems.

## Overview

DynaAgent analyzes questions, creates specialized agents dynamically, executes tasks using tool calling, and synthesizes results into answers.

## Installation

```bash
pip install openai-agents python-dotenv
```

## Configuration

Create `.env` file:

```
OPENAI_API_KEY=your_key_here
```

## Usage

```bash
python main.py
```

## Architecture

```
Question -> Planning -> Agent Creation -> Execution -> Synthesis -> Answer
```

Components:
- Coordinator: Plans agent creation and synthesizes results
- Dynamic Agents: Created on-demand with specialized roles
- Tools: calculator, search, analyze, request_agent

## Example

Input:
```
What is 25 multiplied by 4?
```

Output:
```
Plan: Creates CalculatorAgent
Created: CalculatorAgent (Calculator)
Tool: calculator(*, 25.0, 4.0) = 100.0
Result: 25 multiplied by 4 is 100.
Final Answer: 100
```

## Tools

- `calculator(operation, a, b)` - Math operations (+, -, *, /)
- `search(query)` - Web search
- `analyze(data)` - Text analysis
- `request_agent(name, role, task)` - Create new agent at runtime

## Code Structure

```python
# Tools
calculator()
search()
analyze()
request_agent()

# Core
class DynaAgent:
    create()
    execute()
```

## Requirements

- Python 3.10+
- openai-agents >= 0.6.1
- python-dotenv >= 1.2.1
- OpenAI API key

## Notes

- Model: gpt-4o-mini
- Rate limiting: 2-second delays between API calls
- Free tier: 3 requests per minute

## License

Proprietary - All rights reserved