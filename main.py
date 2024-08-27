import langchain
import google.generativeai as genai  
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain , SimpleSequentialChain,SequentialChain
from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools
countries = ["Japan", "France", "United States", "Australia", "Egypt", "Argentina"]
def practice():
  prompt = PromptTemplate.from_template("What is the capital of {place}?")
  llm = ChatGoogleGenerativeAI(
      api_key=os.environ['OPENAI_API_KEY'],
      model="gemini-1.5-pro",
      temperature=0,
      max_tokens=None,
      timeout=None,
      max_retries=2,
      # other params...
  )
  chain = LLMChain(llm=llm, prompt=prompt)
  for city in countries:
    output = chain.run(city)
    print(output)
    import time 
    time.sleep(2)

# a llm to get name of an e commerce store from a product name 
prompt = PromptTemplate.from_template("What is the name of the e commerce store that sells {product}?")
llm = ChatGoogleGenerativeAI(
  api_key=os.environ['OPENAI_API_KEY'],
  model="gemini-1.5-pro",
  temperature=0,
  max_tokens=None,
  timeout=None,
  max_retries=2,
  # other params...
)
chain1 = LLMChain(llm=llm, prompt=prompt)




# llm to get comma seperated name of products from an e commerce store name
prompt = PromptTemplate.from_template("What are the names of the products at {store}?")

llm = ChatGoogleGenerativeAI(
  api_key=os.environ['OPENAI_API_KEY'],
  model="gemini-1.5-pro",
  temperature=0,
  max_tokens=None,
  timeout=None,
  max_retries=2,
  # other params...
)
chain2 = LLMChain(llm=llm, prompt=prompt)


# create overall sequential chain   
# overall_chain = SimpleSequentialChain(
#   chains=[chain1,chain2], verbose=True
# )
# overall_chain.run("Logitech G gaming mouse ")


# example of a sequential chain
# This is an LLMChain to write a synopsis given a title of a play.
llm = ChatGoogleGenerativeAI(
  api_key=os.environ['OPENAI_API_KEY'],
  model="gemini-1.5-pro",
  temperature=0,
  max_tokens=None,
  timeout=None,
  max_retries=2,
  # other params...
)
template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Era: {era}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title", "era"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")


# This is an LLMChain to write a review of a play given a synopsis.
llm = ChatGoogleGenerativeAI(
  api_key=os.environ['OPENAI_API_KEY'],
  model="gemini-1.5-pro",
  temperature=0,
  max_tokens=None,
  timeout=None,
  max_retries=2,
  # other params...
)
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template,output_key="review")
# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SimpleSequentialChain

overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["era", "title"],
    # Here we return multiple variables
    output_variables=["synopsis", "review"],
    verbose=True
)
# print(overall_chain({"era":"Renaissance", "title":"The Tempest"}))

#agent


llm = ChatGoogleGenerativeAI(
  api_key=os.environ['OPENAI_API_KEY'],
  model="gemini-1.5-pro",
  temperature=0,
  max_tokens=None,
  timeout=None,
  max_retries=2,
  # other params...
)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
output = agent.run("How old varun dhawan will be in 2056")
print(output)