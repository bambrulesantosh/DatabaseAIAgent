import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


key="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"
# model = ChatOpenAI(
#   api_key = key,
#   base_url="https://openrouter.ai/api/v1",
#   model="google/gemma-3-4b-it:free",
# )
model = ChatOpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
  model="llama3.1:8b",
)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df =pd.read_csv('./data/salaries_2023.csv').fillna(value=0)
# print(df.head())


from langchain_experimental.agents.agent_toolkits import(
  create_pandas_dataframe_agent,
  create_csv_agent)

agent  = create_pandas_dataframe_agent(
  llm = model,
  df=df,
  verbose=True,
  allow_dangerous_code=True,
  handle_parsing_errors=True, # Add this line
    max_execution_time=1000,
    max_iterations=150,
    include_df_in_prompt = False,

)

# prefix = "You are working with a Pandas DataFrame in Python. The name of the dataframe is `df`.\n\n",

# res= agent.invoke("What is the average salary?")
# print(res)
#
# res= agent.invoke("how many rows are there in the dataframe?")
# print(res)

CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names, then answer the question.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result,reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
"""
QUESTION = "Which grade has the highest average base salary, and compare the average female pay vs male pay?"

res = agent.invoke(CSV_PROMPT_PREFIX + QUESTION + CSV_PROMPT_SUFFIX)

print(f"Final result: {res["output"]}")