"""SP-aware WanModel forward pass.

Patches WanModel.forward() for sequence-parallel execution.
Each rank holds seq_len//world_size tokens. Time embeddings and
modulations are sharded to match. The head gathers back to full.
"""

import math
import torch
import torch.distributed as dist
from models.parallel_state import get_sp_rank, get_sp_degree
from models.tp_utils import get_tp_rank, get_tp_world_size


def make_sp_forward(model):
    """Replace model.forward with SP-aware version."""
    from wan.modules.model import sinusoidal_embedding_1d

    sp_degree = get_sp_degree()
    tp_degree = get_tp_world_size()
    world_size = sp_degree * tp_degree
    sp_rank = get_sp_rank()
    tp_rank = get_tp_rank()
    world_rank = dist.get_rank()

    if sp_degree <= 1:
        return model

    def sp_forward(x, t, context, seq_len, y=None):
        if model.model_type == 'i2v':
            assert y is not None

        device = model.patch_embedding.weight.device
        if model.freqs.device != device:
            model.freqs = model.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Patch embedding (replicated — all ranks see same input)
        x = [model.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len

        # Round seq_len up to multiple of world_size for even sharding
        seq_len_padded = ((seq_len + world_size - 1) // world_size) * world_size

        # Pad to seq_len_padded: [B, seq_len_padded, dim]
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len_padded - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        L = seq_len_padded

        # Time embedding (compute full, then shard)
        if t.dim() == 1:
            t = t.expand(t.size(0), L)
        bt = t.size(0)
        t_flat = t.flatten()
        e = model.time_embedding(
            sinusoidal_embedding_1d(model.freq_dim, t_flat)
            .unflatten(0, (bt, L)).to(torch.bfloat16))
        e0 = model.time_projection(e).unflatten(2, (6, model.dim))

        # Shard x and e0 to this rank's portion
        L_local = L // world_size
        local_start = world_rank * L_local
        x = x[:, local_start:local_start + L_local].contiguous()
        e0_local = e0[:, local_start:local_start + L_local].contiguous()

        # Context (replicated)
        context_lens = None
        context = model.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # Run transformer blocks on SP-sharded x
        kwargs = dict(
            e=e0_local,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=model.freqs,
            context=context,
            context_lens=context_lens)

        for block in model.blocks:
            x = block(x, **kwargs)

        # Gather x back to full for head
        B = x.shape[0]
        x_full = torch.empty(B, L, model.dim, dtype=x.dtype, device=x.device)
        x_flat = x.reshape(B * L_local, model.dim).contiguous()
        x_full_flat = x_full.reshape(B * L, model.dim)
        dist.all_gather_into_tensor(x_full_flat, x_flat)
        x = x_full_flat.reshape(B, L, model.dim)

        # Head (on full sequence)
        x = model.head(x, e)

        # Unpatchify
        x = model.unpatchify(x, grid_sizes)
        return [u.to(torch.bfloat16) for u in x]

    model.forward = sp_forward
    return model
