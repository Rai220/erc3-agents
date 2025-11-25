# pip install python-dotenv
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import textwrap
from openai import OpenAI
from store_agent import run_agent
from erc3 import ERC3

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
    print("="*40)
    print(f"Starting Task #{i}: {task.task_id} ({task.spec_id}): {task.task_text}")
    # start the task
    core.start_task(task)
    try:
        run_agent(MODEL_ID, core, task)
    except Exception as e:
        print(e)
    result = core.complete_task(task)
    if result.eval:
        explain = textwrap.indent(result.eval.logs, "  ")
        print(f"\nSCORE: {result.eval.score}\n{explain}\n")

core.submit_session(res.session_id)











