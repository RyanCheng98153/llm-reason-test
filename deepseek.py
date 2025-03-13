import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import re
import argparse

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

# Evaluation function
def evaluate_math500(text_generator, num_samples=500, temperature=0.6):
    # Load Math 500 dataset (assuming it's on Hugging Face)
    dataset = load_dataset("HuggingFaceH4/MATH-500", cache_dir="./.cache/datasets")

    correct = 0
    total = min(num_samples, len(dataset))
    
    for i in range(total):
        problem = dataset[i]["question"]
        answer = dataset[i]["answer"]
        
        # Construct the prompt following DeepSeek's recommendation
        prompt = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

        # Generate response using the pipeline
        generated_text = text_generator(
            prompt,
            max_new_tokens=512,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=1
        )[0]['generated_text']

        # Extract final answer using regex
        match = re.search(r"\\boxed{(.*?)}", generated_text)
        predicted_answer = match.group(1) if match else None

        # Check correctness
        if predicted_answer and predicted_answer.strip() == answer.strip():
            correct += 1

        print(f"Sample {i+1}/{total}: {'Correct' if predicted_answer == answer else 'Incorrect'}")

    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy

def recap(text_generator, problem) -> dict:
    problem = "Solve the equation $2x + 3 = 7$."
    # problem = "Solve the equation $x^3 + y^3 = 1024, x+y=8, x*y=?$"
    temperature = 0.6
    
    prompt = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    # prompt = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}} and recap your reasoning in a few sentences within \\ideas{{}}.\n\n"
      
    # Generate response using the pipeline
    generated_text = text_generator(
        prompt,
        max_new_tokens=512,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=1
    )[0]['generated_text']

    recap_prompt = f"{generated_text} \nRecap your steps below\n\n[Recap]:\n"
    
    # Generate recap using the pipeline
    generated_recap = text_generator(
        recap_prompt,
        max_new_tokens=128,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=1
    )[0]['generated_text']
    
    # Extract the process of recap by stripping the prompt and recap result
    recap_process = generated_recap.split("[Recap]:")[0].split(recap_prompt)
    
    # Extract final answer using regex all result below \[Recap\]:
    recap = generated_recap.split("[Recap]:")[1]
    
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "recap_process": recap_process,
        "recap": recap
    }

def evaluate_aime2024():
    # Load AIME 2024 dataset (assuming it's on Hugging Face)
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train", cache_dir="./.cache/datasets")
    evaluate_math500(text_generator, dataset, num_samples=50, temperature=0.6)

# Run evaluation
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to evaluate")
    # parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    # args = parser.parse_args()
    # python deepseek.py --num_samples 50 --temperature 0.6
    # evaluate_math500(text_generator, num_samples=50, temperature=0.6)
    
    recaption = recap(text_generator, "Solve the equation $2x + 3 = 7$.")
    print("[Prompt]:")
    print(recaption["prompt"])
    
    print(" ================ ")
    print("[Generated Text]:")
    print(recaption["generated_text"])

    print(" ================ ")
    print("[Recap Process]:")
    print(recaption["recap_process"])
    
    print(" ================ ")
    print("[Recap Result]:")
    print(recaption["recap"])
    
    # evaluate_aime2024()