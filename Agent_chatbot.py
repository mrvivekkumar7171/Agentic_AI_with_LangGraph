from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain import hub
import requests
import random
import string
import os

os.environ["LANGCHAIN_PROJECT"] = "Agent ChatBot"

load_dotenv()
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f'http://api.weatherstack.com/current?access_key=aeac120a947e88eb9970efa2148e7275&query={city}'
    response = requests.get(url)
    return response.json()

llm = ChatOpenAI()

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=10
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

def generate_userid():
    # generate 4 random letters (upper + lower case allowed)
    letters = ''.join(random.choice(string.ascii_letters) for _ in range(4))
    # generate 4 random digits
    numbers = str(random.randint(1000, 9999))
    return letters + numbers

config = {
    'run_name': f'{generate_userid()}',
    'tags': ['llm app', 'report_generation','summarization'],
    'metadata': {'model1': 'gpt-4o-mini','temperature1': 0.7,'model2': 'gpt-4o','temperature2': 0.5,'parser':'StrOutputParser'}
}

# Step 5: Invoke
response = agent_executor.invoke({"input": "Identify the birthplace city of Mahendra Singh Dhoni (search) and give its current temperature."}, config=config)
print(response)

print(response['output'])