from groq import Groq
import os

os.environ["GROQ_API_KEY"] = "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"

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    # api_key="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",
)

# client = Groq()
completion = client.chat.completions.create(
    model="qwen-qwq-32b",
    messages=[
        {
            "role": "user",
            "content": "Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations: \\[\\log_2\\left({x \\over yz}\\right) = {1 \\over 2}\\] \\[\\log_2\\left({y \\over xz}\\right) = {1 \\over 3}\\] \\[\\log_2\\left({z \\over xy}\\right) = {1 \\over 4}\\] Then the value of $\\left|\\log_2(x^4y^3z^2)\\right|$ is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$. Please reason step by step, put your final answer within \\boxed{}. Don't think too much."
        },
    ],
    temperature=0.6,
    max_completion_tokens=4096,
    top_p=0.95,
    stream=False,
    stop=None,
)

print(completion.choices[0].message)
