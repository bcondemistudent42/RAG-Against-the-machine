import dspy


class AnswerBot(dspy.Signature):
    """
    Write a classical haiku given the provided inputs.
    """
    question: str = dspy.InputField(desc="Question to answer")
    data: str = dspy.InputField(desc="Data linked to the questions.")
    answer: str = dspy.OutputField()


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
        "content": "What all the planets of the solar system"
    }
]

# print(lm(messages = messages))

reasoning_bot = dspy.ChainOfThought(AnswerBot)
result = reasoning_bot(
        data="### Using API Endpoints\n\nLoading a LoRA Adapter:\n\nTo dynamically load a LoRA adapter, send a POST request to the `/v1/load_lora_adapter` endpoint with the necessary\ndetails of the adapter to be loaded. The request payload should include the name and path to the LoRA adapter.\n\nExample request to load a LoRA adapter:\n\n```bash\ncurl -X POST http://localhost:8000/v1/load_lora_adapter \\\n-H \"Content-Type: application/json\" \\\n-d '{\n    \"lora_name\": \"sql_adapter\",\n    \"lora_path\": \"/path/to/sql-lora-adapter\"\n}'\n```\n\nUpon a successful request, the API will respond with a `200 OK` status code from `vllm serve`, and `curl` returns the response body: `Success: LoRA adapter 'sql_adapter' added successfully`. If an error occurs, such as if the adapter\ncannot be found or loaded, an appropriate error message will be returned.\n\nUnloading a LoRA Adapter:\n\nTo unload a LoRA adapter that has been previously loaded, send a POST request to the `/v1/unload_lora_adapter` endpoint\nwith the name or ID of the adapter to be unloaded.\n\nUpon a successful request, the API responds with a `200 OK` status code from `vllm serve`, and `curl` returns the response body: `Success: LoRA adapter 'sql_adapter' removed successfully`.\n\nExample request to unload a LoRA adapter:\n\n```bash\ncurl -X POST http://localhost:8000/v1/unload_lora_adapter \\\n-H \"Content-Type: application/json\" \\\n-d '{\n    \"lora_name\": \"sql_adapter\"\n}'\n```"
        ,
        question="What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?")
print(result.answer)

# "if self.use_marlin:\n            return torch.ops.vllm.fused_marlin_moe(\n                x,\n                layer.w13_weight,\n                layer.w2_weight,\n                None,\n                None,\n                layer.w13_weight_scale,\n                layer.w2_weight_scale,\n                router_logits,\n                topk_weights,\n                topk_ids,\n                global_scale1=layer.w13_weight_scale_2,\n                global_scale2=layer.w2_weight_scale_2,\n                quant_type_id=scalar_types.float4_e2m1f.id,\n                apply_router_weight_on_input=apply_router_weight_on_input,\n                global_num_experts=global_num_experts,\n                expert_map=expert_map)\n\n        # FlashInfer fused experts path\n        if self.fused_experts is not None:\n            assert is_valid_flashinfer_cutlass_fused_moe(\n                x, layer.w13_weight, layer.w2_weight), (\n                    \"Flashinfer CUTLASS Fused MoE not applicable!\")\n\n            return self.fused_experts(\n                hidden_states=x,\n                w1=layer.w13_weight,\n                w2=layer.w2_weight,\n                topk_weights=topk_weights,\n                topk_ids=topk_ids,\n                inplace=False,  # TODO(shuw): fix later, now output is high prec\n                activation=activation,\n                global_num_experts=global_num_experts,\n                expert_map=expert_map,\n                w1_scale=layer.w13_blockscale_swizzled,\n                w2_scale=layer.w2_blockscale_swizzled,\n                apply_router_weight_on_input=apply_router_weight_on_input,\n            )\n\n        assert expert_map is None, (\"Expert Parallelism / expert_map \"\n                                    \"is currently not supported for \"\n                                    \"CompressedTensorsW4A4MoeMethod.\")\n        from vllm.model_executor.layers.fused_moe.cutlass_moe import (\n            cutlass_moe_fp4)"
# "### Using API Endpoints\n\nLoading a LoRA Adapter:\n\nTo dynamically load a LoRA adapter, send a POST request to the `/v1/load_lora_adapter` endpoint with the necessary\ndetails of the adapter to be loaded. The request payload should include the name and path to the LoRA adapter.\n\nExample request to load a LoRA adapter:\n\n```bash\ncurl -X POST http://localhost:8000/v1/load_lora_adapter \\\n-H \"Content-Type: application/json\" \\\n-d '{\n    \"lora_name\": \"sql_adapter\",\n    \"lora_path\": \"/path/to/sql-lora-adapter\"\n}'\n```\n\nUpon a successful request, the API will respond with a `200 OK` status code from `vllm serve`, and `curl` returns the response body: `Success: LoRA adapter 'sql_adapter' added successfully`. If an error occurs, such as if the adapter\ncannot be found or loaded, an appropriate error message will be returned.\n\nUnloading a LoRA Adapter:\n\nTo unload a LoRA adapter that has been previously loaded, send a POST request to the `/v1/unload_lora_adapter` endpoint\nwith the name or ID of the adapter to be unloaded.\n\nUpon a successful request, the API responds with a `200 OK` status code from `vllm serve`, and `curl` returns the response body: `Success: LoRA adapter 'sql_adapter' removed successfully`.\n\nExample request to unload a LoRA adapter:\n\n```bash\ncurl -X POST http://localhost:8000/v1/unload_lora_adapter \\\n-H \"Content-Type: application/json\" \\\n-d '{\n    \"lora_name\": \"sql_adapter\"\n}'\n```"
