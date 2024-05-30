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
    message="You are an experienced data scientist and you want to create a regression model using the california census data to be able to predict the median house price in any district, given relevant predictor variables. The data is provided in this GitHub-repo: https://raw.githubusercontent.com/kirenz/data-science-projects/master/end-to-end/datasets/housing/housing.csv ",
)