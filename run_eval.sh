model=./quantized_model/llama-2-7b.c4.n128.L2048.LDLQ.2bit.g128.Hsort.seq
ori_model=meta-llama/Llama-2-7b-hf

python eval.py --model vllm \
  --model_args pretrained="${model}",tokenizer=${ori_model},dtype=bfloat16,gpu_memory_utilization=0.8,max_model_len=2048,enforce_eager=True \
  --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa \
  --batch_size auto