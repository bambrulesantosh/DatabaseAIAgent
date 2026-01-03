from sqlite3 import connect



# For Loading data

# with open('./db/chinook.sql', 'r',  encoding="utf8" ) as sql_file:
#     sql_script = sql_file.read()
#
# db = connect('./db/chinook.db')
# cursor = db.cursor()
# cursor.executescript(sql_script)
# db.commit()
# db.close()
# print(len(db.get_usable_table_names()), db.get_usable_table_names())



from  langchain_ollama.llms import  OllamaLLM
llm =OllamaLLM(model = "llama3.1:8b")

# for chunk  in llm.stream("Hi how are you ?"):
#     print(chunk,end="")

from langchain_community.utilities import  SQLDatabase
db = SQLDatabase.from_uri("sqlite:///./db/chinook.db", sample_rows_in_table_info = 3)
from langchain_classic.agents import AgentType
from langchain_community.agent_toolkits import create_sql_agent
agent_executor = create_sql_agent(llm, db = db, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)
agent_executor.invoke("How many different Artists are in the database?")