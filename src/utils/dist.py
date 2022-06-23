import torch
import torch.distributed as torch_dist

import logging
logger = logging.getLogger(__name__)

def is_distributed():
    return torch_dist.is_initialized()

def get_world_size():
    if not torch_dist.is_initialized():
        return 1
    return torch_dist.get_world_size()

def get_rank():
    if not torch_dist.is_initialized():
        return 0
    return torch_dist.get_rank()

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch_dist.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size

def dist_log(info):
    if get_rank() < 1: logger.info(info)