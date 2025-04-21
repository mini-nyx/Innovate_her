# Imports
import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults


os.chdir(r"D:\Aishwarya\Work\MachineHack\Chubb\files")  # Change to the directory where your .env file is located
load_dotenv()  # Load variables from .env file

print(os.getcwd())  # Print the current working directory to confirm it's correct
# Now you can access them like this:
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

print("OpenAI Key:", openai_key)
print("Tavily Key:", tavily_key)