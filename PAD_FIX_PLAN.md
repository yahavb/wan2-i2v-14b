# exp/b-pad — eliminate the `pad` copy NEFFs (the 3 biggest DMA offenders)

## Evidence (verified, run 7, instruction-level NTFF)
Three of the five heaviest NEFFs are standalone `pad` ops compiled to DMA-only
copy NEFFs — zero matmul, 71–79% stalled:

| NEFF       | time   | op                              | DMA        | MB  |
|------------|--------|---------------------------------|------------|-----|
| 99d2fe0b   | 69.6ms | `%pad.3 = pad(%Arg_0,%const)`   | static 94% | 545 |
| 264417b4   | 62.1ms | `%pad.3 = pad(%Arg_0,%const)`   | static 92% | 238 |
| cd58ccf3   | 33.1ms | `%pad.3 = pad(%Arg_0,%const)`   | static 96% | 137 |

Combined ≈ 165 ms of per-step time spent purely zero-padding tensors.

## Why fix C (8192→2048) did nothing
C shrank the pad *amount*. But these NEFFs are **memory-latency bound** (71–79%
stalled), not byte bound — the cost is the standalone copy + DMA setup, not the
pad width. Shrinking 8192→6144 trimmed ~25% of bytes on an op dominated by
stall, so latency didn't move. The fix must **eliminate the materialized pad**,
not resize it.

## Source of the pad
`models/sp_attention.py:70-73` — pads K/V seq from 4200 to a multiple of
SELF_ATTN_SEQLEN_MULTIPLE via `F.pad`, every self-attn call:
```python
pad_k = (SELF_ATTN_SEQLEN_MULTIPLE - l_k % SELF_ATTN_SEQLEN_MULTIPLE) % SELF_ATTN_SEQLEN_MULTIPLE
if pad_k > 0:
    k_nki = F.pad(k_nki, (0, pad_k))
    v_nki = F.pad(v_nki, (0, 0, 0, pad_k))
```
(also the `seq_len_padded` pad in `sp_model.py:52-55`.)

## Candidate fixes (need /neuron-nki-writing + correctness proof — NOT yet written)
1. **Pad inside the kernel, in SBUF.** The kernel already loads K/V in 512/128
   tiles (`kernels/self_attention.py:79-89`) and masks the tail to -inf
   (`mask[:, l_k:]`). Make it accept the true `seq_k=4200` and zero-fill the
   final partial tile in SBUF instead of requiring a pre-padded HBM input. Removes
   the F.pad copy entirely. Risk: partial-tile loads + DMA bounds in NKI.
2. **Fuse pad into the producer.** Have the op that writes K/V emit directly into
   a pre-zeroed padded buffer (preallocate once, write the 4200 valid rows), so
   no separate pad NEFF is emitted. Lower kernel risk, but the alloc must be
   correct and reused.

## Validation protocol (mandatory before merge to main)
- Provenance block must confirm branch=exp/b-pad + the kernel flag.
- Compare clean median vs the B baseline (67.5s) — must improve, not regress.
- Step [5] DMA analysis must show the `pad` NEFFs gone or shrunk.
- Eyeball /tmp/wan2_i2v_output.mp4 vs baseline — a kernel/pad change can corrupt
  output while timings stay stable.
