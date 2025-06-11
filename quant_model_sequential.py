import gc
import json
import math
import os
import sys
from contextlib import contextmanager
from functools import partial
from typing import cast

import torch
from torch import nn
from tqdm import trange
from transformers import AutoModelForCausalLM, HfArgumentParser, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer

import quant
from arguments import Args, QuantArgs, get_short_name
from datautils import get_loaders
from optimized_module.ops.quantize.quant_linear import QuantizedLinear
from quant import set_layer_idx, set_layer_name
from utils import check_save_path, to_json


@torch.compile
def cross_entropy(*args, **kwargs):
    return torch.nn.functional.cross_entropy(*args, **kwargs)

SAFETENSOR = ".safetensors"
torch.set_float32_matmul_precision("highest")

def init(args: Args, quant_args: QuantArgs, log_suffix = ""):
    path = get_short_name(args, quant_args)
    log_path = os.path.join(args.log_dir, path + log_suffix + ".log")
    check_save_path(log_path)
    log_file = open(log_path, "w")
    args.result_path = os.path.join(args.result_dir, path + ".json")
    args.save_path = os.path.join(args.save_dir, path)
    args.tmp_path = os.path.join(args.tmp_dir, path)
    check_save_path(args.result_path)
    check_save_path(args.save_path)
    sys.stdout, sys.stderr = log_file, log_file

@contextmanager
def to_cuda(modules: nn.Module | list[nn.Module], *args: nn.Module):
    if isinstance(modules, nn.Module):
        modules = [modules] + list(args)
    for module in modules:
        module.cuda()
    try:
        yield
    finally:
        for module in modules:
            module.cpu()

@contextmanager
def hook(hooks, *args):
    try:
        yield
    finally:
        if not isinstance(hooks, list | tuple):
            hooks = [hooks] + list(args)
        for hook in hooks:
            hook.remove()

@torch.no_grad()
def Quant(args: Args, quant_args: QuantArgs, model, train_dataloader):
    quant_loss_dict = {}
    if quant_args.nbits >= 16:
        return quant_loss_dict

    model.config.quantization_config = quant_args.to_config_dict()

    device = "cuda"
    hidden_size = model.config.hidden_size
    inps = torch.zeros((args.nsamples, args.seqlen, hidden_size), device=device, dtype=args.torch_dtype)
    cache = {"idx": 0, "kwargs": None}

    def catcher_pre_hook(self, args, kwargs):
        hidden_states = args[0]
        bsz = hidden_states.size(0)
        inps[cache["idx"]: cache["idx"] + bsz] = hidden_states
        cache["idx"] += bsz
        if cache["kwargs"] is None:
            cache["kwargs"] = kwargs
        raise ValueError

    with hook([model.model.layers[0].register_forward_pre_hook(catcher_pre_hook, with_kwargs=True)]):
        with to_cuda([model.model.embed_tokens, model.model.rotary_emb]):
            for batch in train_dataloader:
                try:
                    model(batch.to(device))
                except ValueError:
                    pass

    tot_tokens = args.nsamples * args.seqlen
    for i in trange(model.config.num_hidden_layers, desc="Quant (transformer block)"):
        set_layer_idx(i)
        layer = cast(LlamaDecoderLayer, model.model.layers[i])
        layer.self_attn = cast(LlamaAttention, layer.self_attn)
        subset = [
            [(layer.self_attn.q_proj, "self_attn.q_proj"), (layer.self_attn.k_proj, "self_attn.k_proj"), (layer.self_attn.v_proj, "self_attn.v_proj")],
            [(layer.self_attn.o_proj, "self_attn.o_proj")],
            [(layer.mlp.gate_proj, "mlp.gate_proj"), (layer.mlp.up_proj, "mlp.up_proj")],
            [(layer.mlp.down_proj, "mlp.down_proj")]
        ]
        subset = subset[::-1]

        with to_cuda(layer):
            while len(subset) > 0:
                linear_layers = subset.pop(-1)
                module, _ = linear_layers[0]
                H_acc = torch.zeros((module.in_features, module.in_features), device=device, dtype=torch.float64)
                def accumulate_H(self, args, H_acc):
                    inp = cast(torch.Tensor, args[0])
                    inp = inp.view(-1, inp.size(-1)).float()
                    H_acc += inp.T @ inp
                    raise ValueError

                with hook([module.register_forward_pre_hook(partial(accumulate_H, H_acc=H_acc))]):
                    for j in range(0, args.nsamples, args.batch_size):
                        sub = slice(j, j + args.batch_size)
                        try:
                            layer(inps[sub], **cache["kwargs"])
                        except ValueError:
                            pass

                H_acc = H_acc.div_(tot_tokens).float()

                # try:
                quantizer = quant.QuantMethodMap[quant_args.method](quant_args, H_acc)
                # except Exception as e:
                #     metadata = {
                #         "i": i,
                #         "layer_name": str(linear_layers[0][1]),
                #     }
                #     model.save_pretrained(args.tmp_path)
                #     cache["kwargs"]["inps"] = inps
                #     save_file(cache["kwargs"], os.path.join(args.tmp_path, "inps.safetensors"))
                #     json.dump(metadata, open(os.path.join(args.tmp_path, "recover_metadata.json"), "w"), indent=4)
                #     raise e

                for linear_layer, layer_name in linear_layers:
                    set_layer_name(layer_name)
                    # try:
                    quantized_linear_params, quant_loss = quantizer(linear_layer.weight.data)
                    # except Exception as e:
                    #     metadata = {
                    #         "i": i,
                    #         "layer_name": layer_name,
                    #     }
                    #     model.save_pretrained(args.tmp_path)
                    #     cache["kwargs"]["inps"] = inps
                    #     save_file(cache["kwargs"], os.path.join(args.tmp_path, "inps.safetensors"))
                    #     json.dump(metadata, open(os.path.join(args.tmp_path, "recover_metadata.json"), "w"), indent=4)
                    #     traceback.print_exc()
                    #     raise e

                    if linear_layer.bias is not None:
                        quantized_linear_params["bias"] = linear_layer.bias
                    quantized_linear = QuantizedLinear(linear_layer.in_features,
                                                        linear_layer.out_features,
                                                        bias=linear_layer.bias is not None,
                                                        bits=quant_args.nbits,
                                                        group_size=quantizer.quantizer.group_size,
                                                        dtype=args.torch_dtype).cuda()
                    quantized_linear.load_state_dict(quantized_linear_params)
                    sub_module_name, param_name = layer_name.split(".")
                    del layer._modules[sub_module_name]._modules[param_name]
                    layer._modules[sub_module_name]._modules[param_name] = quantized_linear
                    for loss_name, loss_value in quant_loss.items():
                        quant_loss_dict[f"layer.{i}.{layer_name}/{loss_name}"] = loss_value

                    torch.cuda.empty_cache()
                    gc.collect()

            for j in range(0, args.nsamples, args.batch_size):
                sub = slice(j, j + args.batch_size)
                inps[sub] = layer(inps[sub], **cache["kwargs"])[0]

    return quant_loss_dict

@torch.no_grad()
def evalPPL(args: Args, model: LlamaForCausalLM, testloader: torch.Tensor, dataset_name):
    device = "cuda"
    nsamples = testloader.numel() // args.seqlen
    testloader = testloader.view(-1)[: nsamples * args.seqlen].view(nsamples, args.seqlen).to(device)

    print(f"Evaluating dataset {dataset_name}, nsamples {nsamples}, seqlen {args.seqlen}")

    hidden_size = model.config.hidden_size
    inps = torch.zeros((nsamples, args.seqlen, hidden_size), device=device, dtype=args.torch_dtype)
    cache = {"idx": 0, "kwargs": None}

    def catcher_pre_hook(self, args, kwargs):
        hidden_states = args[0]
        bsz = hidden_states.size(0)
        inps[cache["idx"]: cache["idx"] + bsz] = hidden_states
        cache["idx"] += bsz
        if cache["kwargs"] is None:
            cache["kwargs"] = kwargs
        raise ValueError

    with hook([model.model.layers[0].register_forward_pre_hook(catcher_pre_hook, with_kwargs=True)]):
        with to_cuda([model.get_input_embeddings(), model.model.rotary_emb]):
            for i in range(0, nsamples, args.batch_size):
                try:
                    sub = slice(i, i +args.batch_size)
                    model(testloader[sub])
                except ValueError:
                    pass

    for i in trange(model.config.num_hidden_layers, desc="eval ppl (transformer block)"):
        layer = cast(LlamaDecoderLayer, model.model.layers[i])
        with to_cuda(layer):
            for j in range(0, nsamples, args.batch_size):
                sub = slice(j, j + args.batch_size)
                inps[sub] = layer(inps[sub], **cache["kwargs"])[0]

    with to_cuda(model.model.norm):
        inps = model.model.norm(inps)

    nlls = []
    lm_head = model.get_output_embeddings()
    with to_cuda(lm_head):
        for i in range(0, nsamples, args.batch_size):
            sub = slice(i, i + args.batch_size)
            hidden_states = inps[sub]
            logits = lm_head(hidden_states)
            labels = torch.nn.functional.pad(testloader[sub], (0, 1), value=-100)[..., 1:].contiguous()
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            nlls.append(cross_entropy(logits, labels, reduction="sum").item())

    log_ppl = torch.tensor(nlls).sum() / (nsamples * (args.seqlen - 1))
    return math.exp(log_ppl)


def main(args: Args, quant_args: QuantArgs):

    train_dataloader = get_loaders(args.dataset,
                                   nsamples=args.nsamples,
                                   seqlen=args.seqlen,
                                   model_path=args.model_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=args.torch_dtype,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    quant_loss = Quant(args, quant_args, model, train_dataloader)
    if quant_args.nbits < 16:
        model.save_pretrained(args.save_path)

    for i in range(model.config.num_hidden_layers):
        model.model.layers[i].forward = torch.compile(model.model.layers[i].forward)
    model.model.norm.forward = torch.compile(model.model.norm.forward)

    result = {
        "quant_args": quant_args,
        "args": args,
    }
    for dataset in args.test_dataset:
        testloader = get_loaders(
            dataset,
            seqlen=args.seqlen,
            model_path=args.model_path,
            eval_mode=True,
        )
        ppl = evalPPL(args, model, testloader, dataset)
        print(dataset, " ppl:", ppl, flush=True)
        result[f"{dataset}_ppl"] = ppl

    result["quant_loss"] = quant_loss

    json.dump(result, open(args.result_path, "w"), default=to_json, indent=4)


if __name__ == "__main__":
    args, quant_args = HfArgumentParser((Args, QuantArgs)).parse_args_into_dataclasses()
    init(args, quant_args)
    main(args, quant_args)
