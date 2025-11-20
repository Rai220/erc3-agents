# ERC3 Sample Agents

Sample agents demonstrating how to participate in the ERC3: AI Agents in Action competition.

## Getting Started

### 1. Get Your API Key

To use these agents, you'll need an ERC3 API key:

1. Visit https://erc.timetoact-group.at/
2. Enter the email address you used during registration
3. Your API key will be displayed

Note: If you haven't registered yet, https://www.timetoact-group.at/events/enterprise-rag-challenge-part-3 and allow 24 hours for your
registration to be processed.

### 2. Prerequisites

All agents require:
- ERC3 SDK - for connecting to the platform and accessing benchmarks
- ERC3_API_KEY - your competition API key
- LLM API Key - such as OPENAI_API_KEY or equivalent (depending on the agent)

## Running an Agent

Here's an example of running the sgr-agent-store (a simple agent that solves the store benchmark):

```
# Set up your environment variables
export OPENAI_API_KEY=sk-...
export ERC3_API_KEY=key-...

# Navigate to the agent directory
cd sgr-agent-store

# Activate your virtual environment (optional but recommended)
# python3 -m venv venv
# source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the agent
python3 main.py
```

this should look like this:

![res/console.png](res/console.png)

you can also go to the [Web UI]() and view your running session there:

![res/session.png](res/session.png)

Task execution view allows to see interactions of agent with the Benchmark in more detail:

![res/log.png](res/log.png)



# Available Agents

- sgr-agent-store - A simple agent implementation for the store benchmark. It relies on [Schema-Guided Reasoning](https://abdullin.com/schema-guided-reasoning/) to provide adaptive thinking capabilities with a single recursive prompt and gpt-4o.

# Resources

- https://www.timetoact-group.at/events/enterprise-rag-challenge-part-3
- https://erc.timetoact-group.at/

# Support

You can ask questions in the discord channel (you get a link to that with the registration email)

