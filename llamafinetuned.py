import requests
import pandas as pd
import re
import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("hf_token")

API_URL = "https://lqrnv42wmh9orsli.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": f"Bearer {hf_token}",
	"Content-Type": "application/json" 
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

df = pd.read_csv("C:\\Users\\adria\\OneDrive\\Desktop\\Github repos\\BANA577-Capstone\\Model output - 30 Questions.csv")

print(df.head())
row_num = 0
for index, row in df.iterrows():
    row_num += 1
    print(row_num)

    question = ("You are a chatbot for Portland General Electric (PGE) You help answer questions regarding PGE and it's services. Make your answer concise and to the point. Only answer the question. Question: " + str(row['Question']))

    response = query({
	"inputs": question,
	"parameters": {}
})
    response = response[0].get('generated_text')
    response = re.sub(r'(?<!\.)\s*[^.!?]*$', '', str(response))


    match = re.search(r'Answer:\s*(.*)', response)
    if match is not None:
        response = match.group(1)
    else:
        response = response
        print("WARNING, THIS ROW NEEDS CLEANING",row_num)
    print(response)
    df.at[index, 'Output'] = response
    print("")

df.to_csv(r".\\TestQuestionsOutputLlamafinetuned.csv", index=False)