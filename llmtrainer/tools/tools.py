import logging
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="[%(asctime)s,%(msecs)d] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s]   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def cpu_memory():
    import psutil
    memory_info = psutil.Process().memory_info()
    logger.info(f"CPU Memory Usage: {memory_info.rss / 1024 ** 3} GB")

def gpu_memory():
    import torch
    logger.info(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
    logger.info(f"MAXGPU Memory Usage: {torch.cuda.max_memory_allocated() / 1024 ** 3} GB")
    torch.cuda.reset_peak_memory_stats()

def clear_memory():
    import gc

    import torch
    for _ in range(10):
        gc.collect()
        torch.cuda.empty_cache()

def diff(x, y, lim = 8):
    import torch
    x = x.detach().to(torch.float64)
    y = y.detach().to(torch.float64)
    for i in range(1, lim+1):
        print(f"<=2^-{i}", (((x - y).abs() / torch.maximum(torch.maximum(x.abs(), y.abs()), torch.tensor(1e-9))) <= 2 ** -i).to(torch.float64).mean())

def tensor_mem(t):
    return t.numel() * t.element_size() / 1024 ** 3

def model_mem(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 3

def track_pt_module(out, params = None):
    from collections import OrderedDict, deque

    from torch import Tensor, is_tensor
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
            if t.is_floating_point():
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
