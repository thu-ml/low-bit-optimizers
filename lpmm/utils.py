import torch
import torch.nn.functional as F

import numpy as np
from typing import OrderedDict
import json
import os

from lpmm.functional import vectorwise_dequant, vectorwise_quant, _max_reduce_except_dim


def empty_cache(ratio):
    if ratio is None:
        return
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if reserved > 0 and allocated / reserved < ratio:
        torch.cuda.empty_cache()


def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated


def compute_tensor_bytes(tensors):
    """Compute the bytes used by a list of tensors"""
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    ret = 0
    for x in tensors:
        if x.dtype in [torch.float32, torch.int]:
            ret += np.prod(x.size()) * 4 
        elif x.dtype in [torch.bfloat16, torch.float16, torch.int16]:
            ret += np.prod(x.size()) * 2
        elif x.dtype in [torch.int8]:
            ret += np.prod(x.size())

    return ret


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def min_fn(a, b):
    return a < b


def max_fn(a, b):
    return a > b


def get_metric_fn(metric_op):
    if metric_op == 'min':
        return min_fn
    elif metric_op == 'max':
        return max_fn
    else:
        raise NotImplementedError


def sqnr(x, qx):
    Ps = torch.norm(x)
    Pn = torch.norm(x-qx)
    return 20 * torch.log10(Ps/Pn)


def relerr(x, qx):
    abs_error = torch.abs(x - qx)
    rel_error = abs_error.norm() / torch.abs(x).norm()
    return rel_error


def jsd(x, qx):
    x = x.flatten()
    qx = qx.flatten()
    m = 0.5 * (x + qx)
    jsd = 0.5 * (F.kl_div(x, m) + F.kl_div(qx, m))
    return jsd


def abserr(x, qx):
    return torch.abs(x - qx).mean()


def get_metric_from_q_and_dq(x, op, average, **kwargs):
    metric_fn_map = {
        'snqr': sqnr,
        'relerr': relerr,
        'abserr': abserr,
    }
    metric_fn = metric_fn_map['relerr']
    total_metric = 0.
    for _ in range(average):
        qx, md = vectorwise_quant(x, **kwargs)
        x_hat = vectorwise_dequant(qx, **md)
        total_metric += metric_fn(op(x), op(x_hat))
    total_metric /= average
    return total_metric

