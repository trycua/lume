"""Example demonstrating the ComputerAgent capabilities with the Omni provider."""

import asyncio
import logging
import traceback
import signal

from computer import Computer

# Import the unified agent class and types
from agent import ComputerAgent, LLMProvider, LLM, AgentLoop

# Import utility functions
from utils import load_dotenv_files, handle_sigint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_agent_example():
    """Run example of using the ComputerAgent with OpenAI and Omni provider."""
    print("\n=== Example: ComputerAgent with OpenAI and Omni provider ===")

    try:
        # Create Computer instance with default parameters
        computer = Computer(verbosity=logging.DEBUG)

        # Create agent with loop and provider
        agent = ComputerAgent(
            computer=computer,
            loop=AgentLoop.OPENAI,
            # loop=AgentLoop.ANTHROPIC,
            # loop=AgentLoop.OMNI,
            model=LLM(provider=LLMProvider.OPENAI),  # No model name for Operator CUA
            # model=LLM(provider=LLMProvider.OPENAI, name="gpt-4.5-preview"),
            # model=LLM(provider=LLMProvider.ANTHROPIC, name="claude-3-7-sonnet-20250219"),
            save_trajectory=True,
            only_n_most_recent_images=3,
            verbosity=logging.DEBUG,
        )

        tasks = [
            "Look for a repository named trycua/cua on GitHub.",
            "Check the open issues, open the most recent one and read it.",
            "Clone the repository in users/lume/projects if it doesn't exist yet.",
            "Open the repository with an app named Cursor (on the dock, black background and white cube icon).",
            "From Cursor, open Composer if not already open.",
            "Focus on the Composer text area, then write and submit a task to help resolve the GitHub issue.",
        ]

        for i, task in enumerate(tasks):
            print(f"\nExecuting task {i}/{len(tasks)}: {task}")
            async for result in agent.run(task):
                # print(result)
                pass

            print(f"\n✅ Task {i+1}/{len(tasks)} completed: {task}")

    except Exception as e:
        logger.error(f"Error in run_omni_agent_example: {e}")
        traceback.print_exc()
        raise
    finally:
        # Clean up resources
        if computer and computer._initialized:
            try:
                # await computer.stop()
                pass
            except Exception as e:
                logger.warning(f"Error stopping computer: {e}")


def main():
    """Run the Anthropic agent example."""
    try:
        load_dotenv_files()

        # Register signal handler for graceful exit
        signal.signal(signal.SIGINT, handle_sigint)

        asyncio.run(run_agent_example())
    except Exception as e:
        print(f"Error running example: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
