import os
import time
import random
import torch
import numpy as np

from .dist import is_distributed, get_world_size

def init_seed(seed, np_fix=False, cuda_deterministic=True):
    random.seed(seed)
    torch.manual_seed(seed)

    if np_fix:
        np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def create_save_dir(opt):
    if is_distributed() and get_world_size() > 1:
        torch.distributed.barrier()

    opt.res_dir = os.path.join(
        opt.res_root_dir, "_".join([opt.dset_name, opt.model_name, time.strftime("%Y_%m_%d_%H_%M_%S")]))

    if not os.path.exists(opt.res_dir):
        os.makedirs(opt.res_dir, exist_ok=True)

    opt.log = os.path.join(opt.res_dir, 'model')
    opt.save_model = os.path.join(opt.res_dir, 'model')

    return opt

def set_lr(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * decay_factor


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


