"""SP (Sequence Parallelism) group management.

Works alongside tp_utils.py which manages TP groups.
SP splits the sequence dimension across ranks that have the same tp_rank.

With world_size=8, TP=4, SP=2:
  TP groups: [0,1,2,3], [4,5,6,7]
  SP groups: [0,4], [1,5], [2,6], [3,7]

Each rank's identity:
  tp_rank = global_rank % tp_degree
  sp_rank = global_rank // tp_degree
"""

import torch.distributed as dist

_SP_GROUP = None
_SP_RANK = 0
_SP_DEGREE = 1


def init_sp_group(tp_degree):
    """Create SP groups orthogonal to existing TP groups.

    Must be called after dist.init_process_group().
    """
    global _SP_GROUP, _SP_RANK, _SP_DEGREE

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    sp_degree = world_size // tp_degree

    if sp_degree <= 1:
        _SP_DEGREE = 1
        _SP_RANK = 0
        _SP_GROUP = None
        return

    tp_rank = rank % tp_degree

    # SP groups: same tp_rank across SP partitions
    for tp_i in range(tp_degree):
        ranks = list(range(tp_i, world_size, tp_degree))
        grp = dist.new_group(ranks)
        if tp_i == tp_rank:
            _SP_GROUP = grp
            _SP_RANK = rank // tp_degree
            _SP_DEGREE = sp_degree

    print(f"[SP] Initialized: sp_rank={_SP_RANK}/{_SP_DEGREE}, "
          f"tp_rank={tp_rank}/{tp_degree}, global_rank={rank}")


def get_sp_group():
    return _SP_GROUP


def get_sp_rank():
    return _SP_RANK


def get_sp_degree():
    return _SP_DEGREE
