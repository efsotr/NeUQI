import copy
import random
from dataclasses import dataclass
from typing import Any, Callable, List, MutableSequence, cast

from sortedcontainers import SortedList


@dataclass
class Pair:
    size: int
    ptr: List[Any]

def batch_samples(
    samples: List[Any],
    len_fn: Callable[..., int],
    batch_max_size: int,
    enable_sort: bool = False,
    enable_shuffle: bool = True,
    seed: int = 1234
):
    if enable_sort:
        samples = sorted(samples, key=len_fn, reverse=True)
    elif enable_shuffle:
        samples = copy.deepcopy(samples)
        random.seed(seed)
        random.shuffle(samples)

    batchs = cast(List[List[Any]], [])
    pq = cast(MutableSequence[Pair], SortedList([], key=lambda x: x.size))
    for samp in samples:
        samp_len = len_fn(samp)
        if len(pq) == 0 or pq[-1].size < samp_len:
            batchs.append([])
            p = Pair(size=batch_max_size, ptr=batchs[-1])
        else:
            p = pq.pop(-1)
        p.size -= samp_len
        p.ptr.append(samp)
        pq.add(p)

    return batchs
