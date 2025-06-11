import gzip
import json
from typing import Optional


suffixs = [".gz", ".json", ".pt"]
def get_base(filepath: str):
    for suffix in suffixs:
        filepath = filepath.removesuffix(suffix)
    return filepath

def safe_load(filepath: Optional[str]):
    if filepath is None:
        return None
    if filepath.endswith(".json.gz"):
        return json.load(gzip.open(filepath, "rt"))
    if filepath.endswith(".pt.gz"):
        import torch
        return torch.load(gzip.open(filepath, "rb"))
    if filepath.endswith(".json"):
        return json.load(open(filepath))
    if filepath.endswith(".pt"):
        import torch
        return torch.load(filepath)
    raise NotImplementedError(filepath)

def to_json(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise NotImplementedError(obj)

def safe_save(obj, filepath: str):
    if filepath.endswith(".json.gz"):
        return json.dump(obj, gzip.open(filepath, "wt", compresslevel=1), default=to_json)
    if filepath.endswith(".pt.gz"):
        import torch
        return torch.save(obj, gzip.open(filepath, "wb", compresslevel=1))
    if filepath.endswith(".json"):
        return json.dump(obj, open(filepath, "w"), indent=2, default=to_json)
    if filepath.endswith(".pt"):
        import torch
        return torch.save(obj, filepath)
    raise NotImplementedError(filepath)
