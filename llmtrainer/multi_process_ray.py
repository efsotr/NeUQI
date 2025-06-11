import os
import traceback
from functools import reduce, wraps
from typing import Any, Callable, Iterable, List

import ray
import ray.util.queue
from tqdm import tqdm


ray.init(ignore_reinit_error=True)

def multi_process(
    inputs: Iterable[Any],
    process_fn: Callable[[List[Any],], List[Any]],
    num_workers: int,
    num_cpus: int = 0,
    num_gpus: int = 0,
    reduce_fn=lambda results: reduce(lambda x, y: x + y, results, []),
    worker_ids=None,
    extra_args=(),
):
    inputs = list(inputs)

    @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)
    def remote_process_fn(*args):
        try:
            return process_fn(*args)
        except Exception:
            traceback.print_exc()
            os._exit(1)

    _step = len(inputs) // num_workers
    remainder = len(inputs) % num_workers
    step = lambda i: _step + (i < remainder)
    futures = []
    start = 0
    for i in range(num_workers):
        part_inputs = inputs[start: start + step(i)]
        args = (part_inputs,) + (() if worker_ids is None else (worker_ids[i],)) + extra_args
        futures.append(remote_process_fn.remote(*args))
        start += step(i)

    results = ray.get(futures)
    return reduce_fn(results)

def multi_process_stream(
    inputs: Iterable[Any],
    process_fn: Callable[[List[Any],], List[Any]],
    num_workers: int,
    num_cpus: int = 0,
    num_gpus: int = 0,
    reduce_fn=lambda results: reduce(lambda x, y: x + y, results, []),
    worker_ids=None,
    inputs_len=None,
    enable_tqdm=True,
    extra_args=(),
):
    input_queue = ray.util.queue.Queue(maxsize=num_workers)

    @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)
    def remote_process_fn(*args):
        try:
            return process_fn(*args)
        except Exception:
            traceback.print_exc()
            os._exit(1)

    futures = [remote_process_fn.remote(*((input_queue,) + (() if worker_ids is None else (worker_ids[i],)) + extra_args))
               for i in range(num_workers)]

    if inputs_len is None:
        inputs_len = len(inputs)

    pbar = tqdm(total=inputs_len, disable=not enable_tqdm)

    for i, input in enumerate(inputs):
        input_queue.put(input)
        if i >= num_workers:
            pbar.update(1)

    for _ in range(num_workers):
        input_queue.put(None)
        pbar.update(1)

    results = ray.get(futures)
    return reduce_fn(results)

def multi_process_decorator(multi_process_fn, **kwargs):
    def decorator(process_fn):
        @wraps(process_fn)
        def wrapper(inputs):
            return multi_process_fn(
                inputs,
                process_fn=process_fn,
                **kwargs
            )
        return wrapper
    return decorator
