# pip install python-dotenv
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import argparse
import textwrap
from openai import OpenAI
from store_agent import run_agent
from erc3 import ERC3, ApiException
import traceback

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run store agent benchmark')
parser.add_argument('--skip', type=int, default=0, help='Number of tasks to skip from the beginning')
args = parser.parse_args()

client = OpenAI()
core = ERC3()
MODEL_ID = "gpt-5.1"

# Start session with metadata
res = core.start_session(
    benchmark="store",
    workspace="my",
    name="Simple SGR Agent",
    architecture="NextStep SGR Agent with OpenAI")

status = core.session_status(res.session_id)
print(f"Session has {len(status.tasks)} tasks")

for i, task in enumerate(status.tasks):
    # Skip first N tasks if requested
    if i < args.skip:
        print(f"Skipping Task #{i}: {task.task_id}")
        continue
    
    print("="*40)
    print(f"Starting Task #{i}: {task.task_id} ({task.spec_id}): {task.task_text}")
    # start the task
    core.start_task(task)
    try:
        stats = run_agent(MODEL_ID, core, task)
    except Exception as e:
        print(f"Agent run failed: {e}")
        stats = None

    # Log LLM usage statistics
    if stats:
        try:
            core.log_llm(
                task_id=task.task_id,
                model="gpt-5.1", #stats['model'],
                duration_sec=stats.get('duration_sec'),
                usage=stats.get('usage')
            )
        except Exception as e:
            print(f"Failed to log LLM stats: {e}")

    try:
        result = core.complete_task(task)
        if result.eval:
            explain = textwrap.indent(result.eval.logs, "  ")
            print(f"\nSCORE: {result.eval.score}\n{explain}\n")
    except ApiException as e:
        print(f"Failed to complete task {task.task_id}: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"Unexpected error completing task {task.task_id}: {e}")
        traceback.print_exc()

core.submit_session(res.session_id)
