import os
from datetime import timedelta

import torch


def init_distributed_env(args):
    # Initialize the distributed environment
    args.world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', 1)))
    args.distributed = args.world_size > 1
    args.rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
    args.local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))
    args.gpu = args.local_rank

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=1))
        obtained_world_size = torch.distributed.get_world_size()
        assert obtained_world_size == args.world_size, f"{obtained_world_size} != {args.world_size}"
        print(f"Initializing the environment with {args.world_size} processes / Process rank: {args.rank} / Local rank: {args.local_rank}")
        setup_for_distributed(args.local_rank == 0)  # print via one process per node
    args.effective_batch_size = args.batch_size * args.world_size
    print(f"# processes: {args.world_size} / batch size: {args.batch_size} / effective batch size: {args.effective_batch_size}")


def is_main_proc(local_rank=None, shared_fs=True):
    assert shared_fs or local_rank is not None
    main_proc = not torch.distributed.is_initialized() or (torch.distributed.get_rank() == 0 if shared_fs else local_rank == 0)
    return main_proc


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def wait_for_other_procs():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def reduce_tensor(tensor, average=False):
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    if average:
        rt /= world_size
    return rt
