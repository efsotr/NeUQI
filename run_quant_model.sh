model=meta-llama/Llama-2-7b-hf
nbits=2
group_size=128

python quant_model_sequential.py \
  --model_path $model --dataset c4 --test_dataset c4,wikitext2 \
  --batch_size 4 --dtype bfloat16 --nbits $nbits --group_size $group_size \
  --method LDLQ --enable_H_reorder True --enable_H_diag_weight True \
  --param_init_method_diag grid2_s_best_zp \
  --result_dir ./result --log_dir ./log --save_dir ./quantized_model
