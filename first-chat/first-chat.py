import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import autogen
from autogen import AssistantAgent, UserProxyAgent

llm_config = {"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}
assistant = AssistantAgent("assistant", llm_config=llm_config)

user_proxy = UserProxyAgent(
    "user_proxy", code_execution_config={"executor": autogen.coding.LocalCommandLineCodeExecutor(work_dir="coding")}
)

# Start the chat
user_proxy.initiate_chat(
    assistant,
    message="Plot a chart of NVDA and TESLA stock price change YTD.",
)