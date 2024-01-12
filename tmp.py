for i in range(0, 256):
    print('HIP_VISIBLE_DEVICES=0 hipblaslt-bench --transA N --transB T -m 32 -n 64 -k 4800000 --alpha 1 --a_type bf16_r --lda 32 --b_type bf16_r --ldb 64 --beta 0 --c_type bf16_r --ldc 32 --d_type bf16_r --ldd 32 --compute_type f32_r --algo_method index --api_method cpp --solution_index 21779 --splitk',i)
