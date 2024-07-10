#!/bin/bash
# external variables needed:
# TP=1 | 8 - how many GPUs

TP=8
echo TP=$TP

MODEL_PATH=/data/models--NousResearch--Meta-Llama-3-70B/snapshots/fdbf133d993d2b8157e518bfc8bc4570b2d31c45/

export HIP_FORCE_DEV_KERNARG=1
export ROCBLAS_LAYER=6

IL=100
OL=100
for BS in 1 4 8 16 32 64 128 256
do
        echo '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        echo Input_Length=$IL
        echo Ouput_Length=$OL
        echo Batch_Size=$BS
        python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL_PATH --input-len $IL --output-len $OL --batch-size $BS --tensor-parallel-size $TP --dtype float16 --worker-use-ray 2>&1 | tee latency_llama3_gemm-$BS-$IL-$OL.log
done


IL=2048
OL=128
for BS in 1 4 8 16 32 64 128 256
do
        echo '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        echo Input_Length=$IL
        echo Ouput_Length=$OL
        echo Batch_Size=$BS
        python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL_PATH --input-len $IL --output-len $OL --batch-size $BS --tensor-parallel-size $TP --dtype float16 --worker-use-ray 2>&1 | tee latency_llama3_gemm-$BS-$IL-$OL.log
done
