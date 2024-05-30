# AutoGen 🤖

[AutoGen](https://microsoft.github.io/autogen/) is a framework that enables development of LLM applications using multiple agents that can converse with each other to solve tasks. 

*AutoGen is powered by collaborative research studies from Microsoft, Penn State University, and University of Washington.*

## Set up

```bash
conda create -n autogen python=3.11 pip
```

```bash
conda activate autogen
```

```bash
pip install pyautogen packaging python-dotenv
```

## Working environment

You need to create a file called `.env` which should include your OpenAI-API:

```bash
OPENAI_API_KEY = 'ENTER-YOUR-KEY'
```