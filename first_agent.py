from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
key="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"









#
# client = OpenAI(
#   base_url="https://openrouter.ai/api/v1",
#   api_key="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",
# )
# completion = client.chat.completions.create(
#   extra_headers={
#     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
#     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
#   },
#   extra_body={},
#   model="meta-llama/llama-3.2-3b-instruct:free",
#   messages=[
#     {
#       "role": "user",
#       "content": "What is the meaning of life?"
#     }
#   ]
# )
# print(completion.choices[0].message.content)


llm = ChatOpenAI(
  api_key = key,
  base_url="https://openrouter.ai/api/v1",
  model="google/gemma-3-4b-it:free",
)


messages = [
    SystemMessage(
        content="You are a helpful assistant who is extremely competent as a Computer Scientist! Your name is Rob."
    ),
    HumanMessage(content="who was the very first computer scientist?"),
]

def first_agent(message):
  res = llm.invoke(message)
  return res

def run_agent():
  print("Simple AI Agent: Type 'exit' to quit")
  while True:
    user_input =input("You: ")
    if user_input.lower() =="exit":
      print("Good bye")
      break
    print("AI agent is thinking....")
    messages = [HumanMessage(user_input)]
    response = first_agent(messages)
    print("AI agent:getting the response....")
    print(f"AI agent: {response.content}")

if __name__ == "__main__":
  run_agent()


