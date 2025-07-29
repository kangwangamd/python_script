import ray
from ray import serve
from ray.serve.llm import LLMConfig, LLMServer, LLMRouter
from ray.serve.schema import LoggingConfig
import os
import argparse
import time
import logging
import uvicorn

def main():
    model_path = "/scratch1/huggingface_models/Meta-Llama-3.1-8B-Instruct"
    accelerator_type = "H100" # Could be AMD-Instinct-MI300X-OAM and H100

    # Create LLMConfig object
    server_config = LLMConfig(
        llm_engine="vLLM", #SGLang
        model_loading_config=dict(
            model_id=model_path,
            model_source=model_path,
        ),
        experimental_configs={
            # Maximum batching
            "stream_batching_interval_ms": 50,
        },
        accelerator_type=accelerator_type,
        engine_kwargs=dict( # vLLM args. ToDo: SGLang
            # model = "/scratch1/huggingface_models/Meta-Llama-3.1-8B-Instruct",
            model_path = "/scratch1/huggingface_models/Meta-Llama-3.1-8B-Instruct",
            gpu_memory_utilization=0.9, # mem_fraction_static = 0.5
            tensor_parallel_size=8, # tp_size = 8
            cuda_graph_max_bs = 64, # DIS to vllm 0.9.0.1
            # swap_space=16,
            dtype="float16",
            #chunked_prefill_size=-1, # enable_chunked_prefill=False,
            #disable_radix_cache=True, # enable_prefix_caching=False,
            # context_length=8192, #
            max_model_len=8192,
            # max_num_batched_tokens=8192,
            kv_cache_dtype="auto",
            # max_num_seqs=64,
            # max_seq_len_to_capture=8192,
            # disable_log_requests=True,
        ),
    )

    # Configure LLMRouter with explicit settings
    router_config = LLMConfig(
        model_loading_config=dict(
            model_id=model_path,
            model_source=model_path,
        ),
        experimental_configs={
            "num_router_replicas": 16,
        },
    )


    print(f"[DEBUG] server_config={server_config}")
    # Deploy the LLMServer. name_prefix must be "vLLM" to help parsing correctly in utils.sh
    print(f"[DEBUG] LLMServer.as_deployment...\n")
    deployment = LLMServer.as_deployment(server_config.get_serve_options(name_prefix="YourName:")).bind(server_config)
    print(f"[DEBUG] LLMRouter.as_deployment...\n")
    # LLMRouter.as_deployment() -> serve.Deploymnet
    # LLMRouter.as_deployment().bind -> Application
    llm_app = LLMRouter.as_deployment([router_config]).bind([deployment])

    # Run the serve deployment
    print(f"[DEBUG] serve.start...\n")
    serve.start(http_options={"host": "0.0.0.0", "port": 8123})
    logging_config = LoggingConfig(log_level="WARNING")
    print(f"[DEBUG] serve.run...\n")
    # api.py: run()-_run()->_run_many()->build_app()
    serve.run(llm_app, logging_config=logging_config)

    while True:
        time.sleep(100)

if __name__ == "__main__":
    main()

