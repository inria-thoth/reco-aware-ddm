# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
from typing import Any, List
from socket import gethostname
import torch
import torch.distributed

from . import training_stats


_sync_device = None

#----------------------------------------------------------------------------


def fetch_env_var(*var_names) -> str:
    for name in var_names:
        try:
            return os.environ[name]
        except KeyError:
            continue
    raise KeyError(f"No variable among {var_names} could be fetched.")


def init(do_init: bool = True) -> int:
    global _sync_device

    if not do_init:
        if torch.cuda.device_count() != 1:
            print("Distributed training must be initialized when multiple devices "
                  "are present, but `do_init` was set to False.")
            raise RuntimeError("Distributed training must be initialized when multiple devices are present")
        print("Distributed training will not be initialized.")
        torch.cuda.set_device(0)
        return 0

    try:
        master_addr = fetch_env_var("MASTER_ADDR")
        master_port = fetch_env_var("MASTER_PORT")
        rank = int(fetch_env_var("RANK", "SLURM_PROCID"))
        local_rank = int(fetch_env_var("LOCAL_RANK", "SLURM_LOCALID"))
        world_size = int(fetch_env_var("WORLD_SIZE"))
    except KeyError as e:
        print(f"Falling back to single GPU mode. Error: {e}")
        master_addr = 'localhost'
        master_port = '29512'
        rank = 0
        local_rank = 0
        world_size = 1

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    if world_size > 1:
        backend = 'gloo' if os.name == 'nt' else 'nccl'
        print(f"Distributed initialization at rank {rank} of {world_size} "
              f"(rank {local_rank} on {gethostname()} with {torch.cuda.device_count()} GPUs allocated).")
        torch.distributed.init_process_group(backend=backend, init_method='env://')
    torch.cuda.set_device(local_rank)
    sync_device = torch.device('cuda') if get_world_size() > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)

    return local_rank


def get_local_rank():
    if torch.distributed.is_initialized():
        return int(fetch_env_var("LOCAL_RANK", "SLURM_LOCALID"))
    return 0


def get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


#----------------------------------------------------------------------------


def _simple_gather_all_tensors(result: torch.Tensor, group: Any, world_size: int) -> List[torch.Tensor]:
    with torch.no_grad():
        gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_result, result, group)
    # to propagate autograd graph from local rank
    gathered_result[torch.distributed.get_rank(group)] = result
    return gathered_result

