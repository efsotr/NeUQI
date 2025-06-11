import os
import json
import time
from contextlib import contextmanager
from safetensors import safe_open

def set_device(num: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(num)

def set_torch_log(msg: str):
    os.environ["TORCH_LOGS"] = msg

def set_mirror():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def show_cuda_mem(reset=False):
    """
    Show the current CUDA memory usage.
    Args:
        reset (bool): If True, reset the peak memory stats after showing the memory usage.
    """
    import torch
    GB_convert = lambda x: x / 1024 ** 3
    print("MA", round(GB_convert(torch.cuda.memory_allocated()), 6), "GB", 
          "MaxMA", round(GB_convert(torch.cuda.max_memory_allocated()), 6), "GB", 
          "CA", round(GB_convert(torch.cuda.memory_reserved()), 6), "GB", 
          "MaxCA", round(GB_convert(torch.cuda.max_memory_reserved()), 6), "GB", flush=True)
    if reset:
        torch.cuda.reset_peak_memory_stats()

def show_model_size(model):
    """
    Show the size of the model in GB.
    Args:
        model (torch.nn.Module): The model to show the size of.
    """
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 3

def zero_grad(model):
    for param in model.parameters():
        param.grad = None

def collect():
    import torch
    import gc
    for i in range(10):
        gc.collect()
    torch.cuda.empty_cache()

@contextmanager
def time_counter(name, with_cuda=False):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        if with_cuda:
            import torch
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        print(f"[{name}] elapsed time: {end_time - start_time}s", flush=True)

def get_input_kwargs(batch_size, seq_len, vocab_size, with_fa2_kwargs=True, device="cuda"):
    import torch
    shape = (batch_size, seq_len)
    input_ids = torch.randint(0, vocab_size, shape, device=device).view(1, -1)
    labels = input_ids
    position_ids = torch.arange(0, shape[1], device=device).repeat(shape[0], 1).view(1, -1)
    kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
    }

    if with_fa2_kwargs:
        cu_seq_lens = torch.nn.functional.pad(position_ids.eq(0).nonzero(as_tuple=True)[1], 
                                            pad=(0, 1), value=position_ids.size(-1)).to(torch.int32)
        max_length = position_ids.max().item() + 1
        kwargs = {
            **kwargs,
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "max_length_q": max_length,
            "max_length_k": max_length
        }

    return kwargs

def forward(model, kwargs):
    return model(**kwargs).loss

def divider(len=64):
    print('-' * len, flush=True)

def torch_assert_close(tensor_a,
                       tensor_b,
                       rtol=1e-2,
                       atol=1e-3,
                       max_mismatched_ratio=0.001,
                       verbose=False):
    import torch

    # Compute the absolute difference between the two tensors
    diff = torch.abs(tensor_a - tensor_b)

    # Compute the maximum allowable difference for each element
    max_diff = atol + rtol * torch.abs(tensor_b)

    # Identify elements where the difference exceeds the maximum allowable difference
    mismatched = diff > max_diff

    # Count the number of mismatched elements
    num_mismatched = mismatched.sum().item()

    # Calculate the total number of elements in the tensor
    total_elements = tensor_a.numel()

    # Compute the allowed mismatched elements based on the ratio
    max_allowed_mismatched = int(total_elements * max_mismatched_ratio)

    # Print debug information about the mismatch
    if verbose:
        print(f"Number of mismatched elements: {num_mismatched} / {total_elements} "
              f"(allowed: {max_allowed_mismatched})")

    # Check if the number of mismatched elements exceeds the allowed threshold
    if num_mismatched > max_allowed_mismatched:
        raise AssertionError(
            f"Too many mismatched elements: {num_mismatched} > {max_allowed_mismatched} "
            f"({max_mismatched_ratio * 100:.2f}% allowed, but get {num_mismatched / total_elements * 100:.2f}%). "
            f"Greatest absolute difference: {diff.max().item()}, "
            f"Greatest relative difference: {(diff / (torch.abs(tensor_b) + 1e-12)).max().item()}.")
    else:
        return True

def get_total_activation(saved_tensors):
    tot_act = 0
    for tensor, name in saved_tensors:
        if name[0] == "<" and tensor.device.type == "cuda":
            tot_act += tensor.numel() * tensor.element_size()
    return tot_act / 1024 ** 3

def track_pt_module(out, params = None):
    from torch import Tensor, is_tensor
    from collections import deque, OrderedDict
    SAVED_PREFIX = "_saved_"

    if params is None:
        params = {}
    params = dict(zip(map(lambda t: t.data_ptr(), params.values()), params.keys()))
    
    def track(out):
        in_degree = {}
        def dfs(x):
            if x in in_degree:
                in_degree[x] += 1
                return
            
            in_degree[x] = 1
            for nxt, _ in x.next_functions:
                if nxt is not None:
                    dfs(nxt)
        dfs(out)

        used_gpu_mem = OrderedDict()
        def add_tensor(t: Tensor, name: str):
            if t.device.type == "meta":
                used_gpu_mem[id(t)] = (t, name)
                return
            
            if t.data_ptr() in params:
                name = params[t.data_ptr()]
            if t.data_ptr() not in used_gpu_mem:
                used_gpu_mem[t.data_ptr()] = (t, name)

        q = deque()
        q.append(out)
        while len(q) > 0:
            x = q.popleft()
            name = str(x)
            
            for attr in dir(x):
                if attr.startswith(SAVED_PREFIX):
                    v = getattr(x, attr)
                    if is_tensor(v):
                        add_tensor(v, name)
                    if isinstance(v, tuple):
                        for t in v:
                            if is_tensor(t):
                                add_tensor(t, name)

            if hasattr(x, "saved_tensors"):
                for t in x.saved_tensors:
                    if is_tensor(t):
                        add_tensor(t, name)
            
            if hasattr(x, "variable"):
                if is_tensor(x.variable):
                    add_tensor(x.variable, name)
            
            for nxt, _ in x.next_functions:
                if nxt is not None:
                    in_degree[nxt] -= 1
                    if in_degree[nxt] == 0:
                        q.append(nxt)

        return list(OrderedDict(used_gpu_mem).values())

    return track(out.grad_fn)

def size_to_str(size: tuple):
    return "(" + ", ".join(map(str, size)) + ")"

def used_mem_to_json(path, used_mem):
    import json
    out = [(ex[1], ex[0].numel() * ex[0].element_size(), size_to_str(ex[0].shape)) for ex in used_mem]
    total = {"parameter": sum(ex[1] for ex in out if ex[0] != "") / 1024 ** 3, 
             "activation": sum(ex[1] for ex in out if ex[0] == "") / 1024 ** 3}
    json.dump({"total": total, "out": out}, open(path, "w"), indent=4)

class safe_open_dir:

    def __init__(self, dirpath, framework="pt", device="cpu"):
        if dirpath.endswith(".safetensors") or os.path.exists(os.path.join(dirpath, "model.safetensors")):
            safetensor_path = os.path.join(dirpath, "model.safetensors") if not dirpath.endswith(".safetensors") else dirpath
            weight = safe_open(safetensor_path, framework=framework, device=device)
            self.weights = {"model.safetensors": weight}
            self.weight_map = {k: "model.safetensors" for k in weight.keys()}
            return

        metadata = json.load(open(os.path.join(dirpath, "model.safetensors.index.json")))
        weight_files = list(set(metadata["weight_map"].values()))
        weights = {}
        for weight_file in weight_files:
            weights[weight_file] = safe_open(os.path.join(dirpath, weight_file), framework=framework, device=device)

        self.weight_map = metadata["weight_map"]
        self.weights = weights

    def keys(self, filter_fn=lambda key: True):
        return list(filter(filter_fn, self.weight_map.keys()))

    def get_tensor(self, key):
        return self.weights[self.weight_map[key]].get_tensor(key)

    def get_state_dict(self, filter_fn = lambda key: True):
        return {key: self.get_tensor(key) for key in self.keys(filter_fn)}