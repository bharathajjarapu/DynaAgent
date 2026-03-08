import asyncio
import os
import sys

from core import DynaAgent

os.system("")  # Enable ANSI escape sequences on Windows Native CMD/Powershell

GREEN, CYAN, MAGENTA, DIM, BOLD, RESET = "\033[92m", "\033[96m", "\033[95m", "\033[2m", "\033[1m", "\033[0m"


def show(who: str, text: str = "", code: str = ""):
    """Print one agent event."""
    color = CYAN if who == "Agent" else DIM if who == "Budget" else MAGENTA
    name = "DynaAgent" if who == "Agent" else who
    print(f"{BOLD}{color}❯ {name}{RESET} {DIM}•{RESET} {text}")
    if code:
        print(DIM + "\n".join("    " + line for line in code.split("\n")) + RESET)
    print()


async def main():
    """Answer piped stdin once, or chat interactively."""
    agent = DynaAgent(on=show)

    if not sys.stdin.isatty():
        await agent.execute(sys.stdin.read().strip() or "What is 25 multiplied by 4?")
        return

    print(f"{BOLD}{CYAN}❯ DynaAgent{RESET} {DIM}•{RESET} {BOLD}Hi I am DynaAgent!{RESET} {DIM}Ask me Anything, or Type 'exit' or 'quit' to Stop.{RESET}\n")
    while True:
        try:
            question = input(f"{BOLD}{GREEN}❯ You{RESET} {DIM}•{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Exiting...{RESET}")
            break
        if question.lower() in ("exit", "quit"):
            break
        if question:
            await agent.execute(question)


if __name__ == "__main__":
    asyncio.run(main())
