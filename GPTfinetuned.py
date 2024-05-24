import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


def query(message):
    response = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0125:oregon-state-university-center-of-business-analytics:test1:9OMrd0YJ",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are a helpful chatbot for Portland General Electric (PGE). You answer questions regarding PGE and PGE services."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": message
            }
        ]
        }
    ],
    temperature=0.1,
    max_tokens=450,
    top_p=0.6,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response.model_dump()

# response = query("Can I plug my generator directly into an outlet?")['choices'][0]['message']['content']
# print(response)


df = pd.read_csv("C:\\Users\\adria\\OneDrive\\Desktop\\OSU\\BANA 577 Capstone\\TestQuestions.csv")

print(df.head())
row_num = 0
for index, row in df.iterrows():
    row_num += 1
    print(row_num)
    response = query(row['Questions'])['choices'][0]['message']['content']

    df.at[index, 'Output'] = response
    print(row['Questions'])
    print(response)
    print("")

df.to_csv(r".\\TestQuestionsOutputGPTfinetuned.csv", index=False)