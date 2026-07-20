import dspy


# faire une autre classe qui recois une liste de question
# elle traite ensuite tout les questions une par une et stock
# le resultat dans un json similaire a asnwered questions
class AnswerBot(dspy.Signature):
    """
    Answer the question with the given data.
    """
    question: str = dspy.InputField(desc="Question to answer")
    data: str = dspy.InputField(desc="Data linked to the question.")
    answer: str = dspy.OutputField()


lm = dspy.LM(
    "openai/Qwen/Qwen3-0.6B",
    api_base="http://localhost:8000/v1",
    api_key="_"
)

dspy.configure(lm=lm)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "What all the planets of the solar system"
    }
]

# to do this for all n
reasoning_bot = dspy.ChainOfThought(AnswerBot)
result = reasoning_bot(
        data="### Using API Endpoints\n\nLoading a LoRA Adapter:\n\nTo dynamically load a LoRA adapter, send a POST request to the `/v1/load_lora_adapter` endpoint with the necessary\ndetails of the adapter to be loaded. The request payload should include the name and path to the LoRA adapter.\n\nExample request to load a LoRA adapter:\n\n```bash\ncurl -X POST http://localhost:8000/v1/load_lora_adapter \\\n-H \"Content-Type: application/json\" \\\n-d '{\n    \"lora_name\": \"sql_adapter\",\n    \"lora_path\": \"/path/to/sql-lora-adapter\"\n}'\n```\n\nUpon a successful request, the API will respond with a `200 OK` status code from `vllm serve`, and `curl` returns the response body: `Success: LoRA adapter 'sql_adapter' added successfully`. If an error occurs, such as if the adapter\ncannot be found or loaded, an appropriate error message will be returned.\n\nUnloading a LoRA Adapter:\n\nTo unload a LoRA adapter that has been previously loaded, send a POST request to the `/v1/unload_lora_adapter` endpoint\nwith the name or ID of the adapter to be unloaded.\n\nUpon a successful request, the API responds with a `200 OK` status code from `vllm serve`, and `curl` returns the response body: `Success: LoRA adapter 'sql_adapter' removed successfully`.\n\nExample request to unload a LoRA adapter:\n\n```bash\ncurl -X POST http://localhost:8000/v1/unload_lora_adapter \\\n-H \"Content-Type: application/json\" \\\n-d '{\n    \"lora_name\": \"sql_adapter\"\n}'\n```"
        ,
        question="What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?")
print(result.answer)
