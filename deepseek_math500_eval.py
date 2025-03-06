import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import argparse

# Load DeepSeek-R1 model and tokenizer
MODEL_NAME = "deepseek-ai/DeepSeek-R1"  # Change if using another DeepSeek-R1 model


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, dir="./.cache/tokenizers")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True, dir="./.cache/models")

# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", dir="./.cache/models")

# Load Math 500 dataset (assuming it's on Hugging Face)
dataset = load_dataset("deepseek-ai/math500", split="test", cache_dir="./.cache/datasets")

# Evaluation function
def evaluate_math500(model, tokenizer, dataset, num_samples=500, temperature=0.6):
    correct = 0
    total = min(num_samples, len(dataset))
    
    for i in range(total):
        problem = dataset[i]["question"]
        answer = dataset[i]["answer"]
        
        # Construct the prompt following DeepSeek's recommendation
        prompt = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        
        # Tokenization
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, temperature=temperature)
        
        # Decode response
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

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

# Run evaluation
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    args = parser.parse_args()

    evaluate_math500(model, tokenizer, dataset, num_samples=args.num_samples, temperature=args.temperature)
# python deepseek_math500_eval.py --num_samples 50 --temperature 0.6