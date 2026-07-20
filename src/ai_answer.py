import dspy

lm = dspy.LM(
    "openai/Qwen/Qwen3-0.6B",
    api_base="http://localhost:8000/v1",
    api_key="none"
)

dspy.configure(lm=lm)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Who created Star Wars"
    }
]

print(lm(messages = messages)[0])