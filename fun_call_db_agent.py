import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
from openai import OpenAI
import helpers
from sqlalchemy import create_engine
from helpers import (
    get_avg_salary_and_female_count_for_division,
    get_total_overtime_pay_for_department,
    get_total_longevity_pay_for_grade,
    get_employee_count_by_gender_in_department,
    get_employees_with_overtime_above,
)




client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

database_file_path = "./db/salary.db"
engine = create_engine(f"sqlite:///{database_file_path}")
file_url = "./data/salaries_2023.csv"
os.makedirs(os.path.dirname(database_file_path), exist_ok=True)
df = pd.read_csv(file_url).fillna(value=0)
df.to_sql("salaries_2023", con=engine, if_exists="replace", index=False)


def run_conversation(
    query="""What is the average salary and the count of female employees
    #                   in the ABS 85 Administrative Services division?""",
):
    messages = [
        # {
        #     "role": "user",
        #     "content": """What is the average salary and the count of female employees
        #               in the ABS 85 Administrative Services division?""",
        # },
        {
            "role": "user",
            "content": query,
        },
        # {
        #     "role": "user", # gives error request too large
        #     "content": """How many employees have overtime pay above 5000?""",
        # },
    ]

    # Call the model with the conversation and available functions
    response = client.chat.completions.create(
        model='llama3.2:3b',
        messages = messages,
        tools=helpers.tools_sql,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    print(response_message.model_dump_json(indent=2))
    print("tool calls: ", response_message.tool_calls)
    # Example calls to the functions
    tool_calls = response_message.tool_calls
    if tool_calls:
        # Step 3: call the function
        available_functions = {
            "get_avg_salary_and_female_count_for_division": get_avg_salary_and_female_count_for_division,
            "get_total_overtime_pay_for_department": get_total_overtime_pay_for_department,
            "get_total_longevity_pay_for_grade": get_total_longevity_pay_for_grade,
            "get_employee_count_by_gender_in_department": get_employee_count_by_gender_in_department,
            "get_employees_with_overtime_above": get_employees_with_overtime_above,
        }
        messages.append(response_message)  # extend conversation with assistant's reply

        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            if function_name == "get_employees_with_overtime_above":
                function_response = function_to_call(amount=function_args.get("amount"))
            elif function_name == "get_total_longevity_pay_for_grade":
                function_response = function_to_call(grade=function_args.get("grade"))
            else:
                function_response = function_to_call(**function_args)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                }
            )  # extend conversation with function responses
            second_response = client.chat.completions.create(
                model='llama3.2:3b',
                messages = messages,
            )  # get a new response from the model where it can see the function response

    return second_response

if __name__ == "__main__":
    res = (
        run_conversation(query="""What is the total longevity pay for employees grade 'M3'?""")
        .choices[0]
        .message.content
    )
    print(res)
    division_name = "ABS 85 Administrative Services"
    department_name = "Alcohol Beverage Services"
    grade = "M3"
    overtime_amount = 5000

    total_longevity_pay = get_total_longevity_pay_for_grade(grade)
    print(total_longevity_pay)