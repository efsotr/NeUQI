import json
import os

import torch
from safetensors import safe_open


class safe_open_dir:

    def __init__(self, dirpath, framework="pt", device="cpu"):
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

def check_save_path(save_path: str):
    save_path = os.path.abspath(save_path)
    save_dir = os.path.dirname(save_path)
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Check Error: {repr(e)}")

    if os.path.exists(save_path):
        assert os.access(save_path, os.W_OK)
    else:
        assert os.access(save_dir, os.W_OK)

def diff(x, y, d=11):
    x = x.float()
    y = y.float()
    magnitude = torch.maximum(x.abs(), y.abs())
    abs_dt = (x - y).abs().div_(magnitude.masked_fill_(magnitude == 0, 1e-8))
    for i in range(0, d+1):
        print(f" <= 2^-{i}:", (abs_dt <= 2 ** -i).sum() / x.numel())

def to_json(x):
    try:
        for attr in ["tolist", "to_list", "to_dict"]:
            if hasattr(x, attr):
                return getattr(x, attr)()
        if hasattr(x, "__dict__"):
            return x.__dict__
    except Exception:
        pass
    return repr(x)
