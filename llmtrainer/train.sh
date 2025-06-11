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
