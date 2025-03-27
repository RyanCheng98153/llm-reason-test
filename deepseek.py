import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import re
import csv
import argparse

def generate(text_generator, problem) -> str:
    temperature = 0.6
    
    prompt = f"{problem}\n\nPlease reason step by step, put your final answer within \\boxed{{}}. Don't think too much."
    # prompt = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}} and recap your reasoning in a few sentences within \\ideas{{}}.\n\n"
      
    # Generate response using the pipeline
    generated_text = text_generator(
        prompt,
        max_new_tokens=1024,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=1
    )[0]['generated_text']
    
    match = re.findall(r'\\boxed{(.*?)\}', generated_text)
    
    return {
        "prompt": prompt,
        "generated_text": generated_text.lstrip(prompt),
        "generated_answer": match
    }

def recap(text_generator, generated_text) -> dict:
    # problem = "Solve the equation $2x + 3 = 7$."
    # # problem = "Solve the equation $x^3 + y^3 = 1024, x+y=8, x*y=?$"
    # temperature = 0.6
    temperature = 0.6
    
    # recap_prompt = f"{generated_text.lstrip(prompt)} \nRecap your steps below in a numbered list format:\n\n[Recap]:\n1."
    recap_prompt = f"{generated_text} \nRecap your steps below in a structured format:\n\n[Recap]:\n[Step 1]: "

    # Generate recap using the pipeline
    generated_recap = text_generator(
        recap_prompt,
        max_new_tokens=128,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=1
    )[0]['generated_text']
    
    # Extract final answer using regex all result below \[Recap\]:
    recap = generated_recap.split("[Recap]:")[1]
    
    return {
        "recap_process": generated_recap.lstrip(recap_prompt.rstrip("1.")),
        "recap": recap
    }
    
def generate_recap_aime2024(text_generator):
    # Load AIME 2024 dataset (assuming it's on Hugging Face)
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train", cache_dir="./.cache/datasets")

    # aime_dataset_structure
    # {
    # "ID": "2024-I-1",
    # "Problem": "Problem statement...",
    # "Solution": "Detailed solution...",
    # "Answer": "Numerical answer"
    # }
    
    # write aim2024 recaption to csv file
    with open("aime2024_recaption.csv", "w") as csvfile:
        titles = ["ID", "Problem", "Ground Truth", "Recap Answer", "Recap", "Prompt", "Generated Text", "Solution", "Recap Process"]
        # writer = csv.writer(csvfile)
        # writer.writerow(titles)
        
        writer = csv.DictWriter(csvfile, titles)
        writer.writeheader()
        
        for i, data in enumerate(dataset):
            print(f"Processing {i+1}/{len(dataset)}: {data['ID']}")
            
            id = data["ID"]
            problem = data["Problem"]
            answer = data["Answer"]
            solution = data["Solution"]
            
            generate_result = generate(text_generator, problem)
            recap_result = recap(text_generator, generate_result["generated_text"])
            
            writer.writerow({
                "ID": id,
                "Problem": problem,
                "Ground Truth": answer,
                "Generate Answer": generate_result["generated_answer"],
                "Recap Process": recap_result["recap_process"],
                "Recap": recap_result["recap"],
                "Prompt": generate_result["prompt"],
                "Generated Text": generate_result["generated_text"],
                "Solution": solution
            })

def read_aime2024_recaption(line: int):
    with open("aime2024_recaption.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i == line:
                print(f"[ ID ]: \n{row['ID']}")
                print(f"[ Problem ]: \n{row['Problem']}")
                print(f"[ Ground Truth ]: \n{row['Ground Truth']}")
                print(f"[ Recap Process ]: \n{row['Recap Process']}")
                print(f"[ Recap ]: \n{row['Recap']}")
                print(f"[ Recap Answer ]: \n{row['Answer']}")
                print(f"[ Prompt ]: \n{row['Prompt']}")
                print(f"[ Generated Text ]: \n{row['Generated Text']}")
                print(f"[ Solution ]: \n{row['Solution']}")
                break
        
def evaluate_aime2024():
    # Load AIME 2024 dataset (assuming it's on Hugging Face)
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train", cache_dir="./.cache/datasets")

    # aime_dataset_structure
    # {
    # "ID": "2024-I-1",
    # "Problem": "Problem statement...",
    # "Solution": "Detailed solution...",
    # "Answer": "Numerical answer"
    # }
    
    correct = 0
    total = len(dataset)
    
    for i, data in enumerate(dataset):
        problem = data["Problem"]
        answer = data["Answer"]
        
        print(f"Problem: {problem}")
        print(f"Answer: {answer}")


# Load DeepSeek-R1 model and tokenizer
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    cache_dir="./.cache/tokenizers"
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True, 
    cache_dir="./.cache/models",
    torch_dtype=torch.float16  # Use FP16 instead of FP8
)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Create a text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1
)

import sys

# Run evaluation
if __name__ == "__main__":
    # test_recap()
    generate_recap_aime2024(text_generator)
    # read_aime2024_recaption(int(sys.argv[1]))
    
    # evaluate_aime2024()
