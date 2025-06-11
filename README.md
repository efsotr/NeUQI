# NeUQI: Near-Optimal Uniform Quantization Parameter Initialization

<p align="center">
<a href="https://arxiv.org/abs/2505.17595">
 <img src="https://img.shields.io/badge/arXiv-2505.17595-B31B1B.svg?logo=arxiv&logoColor=white" alt="arXiv"/>
</a>
</p>

> ⚠️ **Warning**  
> It is highly recommended to use Anaconda to replicate the exact environment. Other versions may not work correctly, and their results are not guaranteed. In general, newer versions are less risky than older ones.  
>  
> All experiments were conducted on a single NVIDIA A40 GPU. Although memory usage has been optimized, we do not guarantee that GPUs with less memory will avoid out-of-memory (OOM) issues. Identical results are only ensured on the A40 unless other factors interfere.  
>  
> This project uses the relatively new version of `torch.compile` and FlashAttention 2 to accelerate execution and reduce memory consumption. Older versions of PyTorch or FlashAttention may be incompatible—version mismatches can lead to failed optimizations or difficult debugging.
>  
> The current format of quantized weight and the current implementation of quantized linear forward pass is for idea validation only and does not include advanced acceleration. For optimized implementations, refer to [BitBLAS](https://github.com/microsoft/BitBLAS), [lut-gemm](https://github.com/naver-aics/lut-gemm), and [flute](https://github.com/HanGuo97/flute).  
>  
> The vLLM integration is only tested on single-GPU inference. Tensor-parallel multi-GPU inference is untested and not guaranteed to be correct.  
>  
> For distillation, only lightweight distillation is verified. Full compatibility with `transformers` and FSDP is limited and may require manual source code modifications.


> 📝 **Note**  
> For a given combination of method and hyperparameters, all generated files—including results, logs, and models—use the same short name as either a prefix or a directory name.
>
> `group_size = -1` indicates channel-wise quantization, whereas `group_size = 128` indicates group-wise quantization with a group size of 128.


## Environment Setup

We highly recommend using **Anaconda** to manage the environment. Please follow the steps below to set up the project:

1. Create and activate the environment using the provided `environment.yml` file:

   ```bash
   conda env create -f environment.yml -n nips_test
   conda activate nips_test
   ```

2. Install the optimized module:

   ```bash
   cd optimized_module
   pip install .
   ```

You're now ready to use NeUQI!

## Calibration Stage

For each of the following commands, the names of the result and log files are determined by the specified arguments, such as `--method`, `--dataset`, and others.

To facilitate file management and benchmarking, a unique **short name** is generated based on the same set of arguments.

You can obtain this **short name** by running the following command:

```bash
python arguments.py [same arguments as below]
```

The generated **short name** will be used consistently as a prefix or directory name for all associated output files.
**Note:** `[shortname]` serves as a placeholder for the actual generated value.

* **Result files** will have a `.json` extension (e.g., `[shortname].json`)
* **Log files** will have a `.log` extension (e.g., `[shortname].log`)
* **Model files** will be stored in a directory named after the short name (e.g., `[shortname]/`)

### Example Commands

In the following examples, variables like `$model`, `$nbits`, and `$group_size` are placeholders. Here's an example of how you might define them:

```bash
model=meta-llama/Llama-2-7b-hf
nbits=4
group_size=128
```

Replace these with the values specific to your experiment.

#### 1. **LDLQ** — Equivalent to GPTQ but faster and more accurate (preferred)

```bash
python quant_model_sequential.py \
  --model_path $model --dataset c4 --test_dataset c4,wikitext2 \
  --batch_size 4 --dtype bfloat16 --nbits $nbits --group_size $group_size \
  --method LDLQ --enable_H_reorder True \
  --result_dir ./result --log_dir ./log --save_dir ./quantized_model
```

#### 2. **MagR (+LDLQ)**

```bash
python quant_model_sequential.py \
  --model_path $model --dataset c4 --test_dataset c4,wikitext2 \
  --batch_size 4 --dtype bfloat16 --nbits $nbits --group_size $group_size \
  --method LDLQ --enable_H_reorder True --enable_H_diag_weight True \
  --enable_magr True --enable_magr_only True \
  --result_dir ./result --log_dir ./log --save_dir ./quantized_model
```

#### 3. **NeUQI (+LDLQ)**

```bash
python quant_model_sequential.py \
  --model_path $model --dataset c4 --test_dataset c4,wikitext2 \
  --batch_size 4 --dtype bfloat16 --nbits $nbits --group_size $group_size \
  --method LDLQ --enable_H_reorder True --enable_H_diag_weight True \
  --param_init_method_diag grid2_s_best_zp \
  --result_dir ./result --log_dir ./log --save_dir ./quantized_model
```

#### 4. **NeUQI\_int (+LDLQ)**

```bash
python quant_model_sequential.py \
  --model_path $model --dataset c4 --test_dataset c4,wikitext2 \
  --batch_size 4 --dtype bfloat16 --nbits $nbits --group_size $group_size \
  --method LDLQ --enable_H_reorder True --enable_H_diag_weight True \
  --param_init_method_diag grid_s_zp \
  --result_dir ./result --log_dir ./log --save_dir ./quantized_model
```

## Distillation Example

To distill a quantized model, enter the `llmtrainer` directory and run the following:

```bash
export WANDB_PROJECT="Quant"
export TORCH_LOGS="recompiles_verbose,graph_breaks"

train() {
    bash ./runs/$1.sh outputs/$2
}

export quant="llama-2-7b.c4.n128.L2048.LDLQ.2bit.g128.Hsort.seq"
export CUDA_VISIBLE_DEVICES=0
export PORT=$((30000 + $CUDA_VISIBLE_DEVICES))
export DATASET=c4
export nsamples=256
export num_train_epochs=1

export MODEL_NAME=meta-llama/Llama-2-7b-hf
export QUANT_MODEL_NAME="../quantized_model/"$quant

export LR=3e-4
exper_name=$quant.lr$LR.$DATASET.n$nsamples.${num_train_epochs}epochs
train quant_distill $exper_name
```

For more detailed adjustments, you can modify `runs/quant_distill.sh` accordingly.

## Evaluation Example

To evaluate a quantized model, run:

```bash
model=./quantized_model/llama-2-7b.c4.n128.L2048.LDLQ.2bit.g128.Hsort.seq
ori_model=meta-llama/Llama-2-7b-hf

python eval.py --model vllm \
  --model_args pretrained="${model}",tokenizer=${ori_model},dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=2048,enforce_eager=True \
  --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa \
  --batch_size auto
```

## Citing this work

```
@article{lin2025neuqi,
  title={NeUQI: Near-Optimal Uniform Quantization Parameter Initialization},
  author={Lin, Li and Hu, Xinyu and Wan, Xiaojun},
  journal={arXiv preprint arXiv:2505.17595},
  year={2025}
}
```
