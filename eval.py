from groq import Groq
import os

os.environ["GROQ_API_KEY"] = "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"

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    # api_key="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",
)

import csv

aime_data = []

def get_aime2024_recaption():
    with open("aime2024_recaption.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            aime_data.append(row)

get_aime2024_recaption()

for i, row in enumerate(aime_data):
    print(f"[ ID ]: \n{row['ID']}")


# # client = Groq()
# completion = client.chat.completions.create(
#     model="qwen-qwq-32b",
#     messages=[
#         {
#             "role": "user",
#             "content": ""
#         },
#     ],
#     temperature=0.6,
#     max_completion_tokens=4096,
#     top_p=0.95,
#     stream=False,
#     stop=None,
# )

# print(completion.choices[0].message)
