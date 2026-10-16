# torch.compile scope on Neuron: Wan2.2-I2V-A14B DiT

Goal: show how the **scope** of `torch.compile` (where graph boundaries fall) and the
`fullgraph` flag change performance, with everything else fixed: same kernels, same math,
same spec (368x480, 21 frames, 10 steps, batched CFG, TP8 on 2 trn2 devices), same node.
Every variant is `torch.compile`; none is eager.

## 1. Why torch.compile matters here (measured)

| Path | What runs | Denoise | Total |
|---|---|---|---|
| Eager DiT, `attention_cte` kernel | ~570 small per-op NEFFs, dispatched one op at a time from Python; only `block.ffn` etc. compiled | 25.2 s | 30.6 s |
| Fullgraph DiT, same `attention_cte` kernel | one compiled graph per transformer block | 6.3 s | 8.3 s |

Same attention kernel in both; the 4x on denoise comes from the compile scope. What a
whole-block graph buys on Neuron:

1. **One NEFF launch per block instead of hundreds of per-op launches.** Each launch is a
   host round trip; per-op dispatch leaves the device idle between ops.
2. **Fusion across ops.** Norms, modulation, RoPE, residual adds and casts fuse with their
   neighbours instead of each writing its result to HBM and reading it back.
3. **Collectives inside the graph.** The TP all-reduces (`funcol.all_reduce`) are scheduled
   by the compiler next to the matmuls that produce them, not issued from Python.
4. **NKI kernels inside the graph.** `attention_cte`, `output_projection_cte` and the fused
   adaln run as custom calls inside the block NEFF, with no graph break around them.
5. **Compile once, reuse 400 times.** All 40 blocks share one NEFF (weights are graph
   inputs), so one compile serves 40 blocks x 10 steps.
6. **`fullgraph=True` is a guarantee.** If anything would break the graph (an import, a
   data-dependent branch, an unsupported op), compilation fails loudly instead of silently
   splitting the block.

## 2. The code that defines scope and config today

`models/fast_dit.py:49` — the compile config used by every DiT unit:

```python
COMPILE_KW = dict(backend="neuron", dynamic=False, fullgraph=True)
```

`models/fast_dit.py:487-493` — the scope: one graph per unit, one unit per transformer block:

```python
self.time = torch.compile(_FastTime(model), **COMPILE_KW)
self.patch = torch.compile(_FastPatch(model.patch_embedding), **COMPILE_KW)
self.text = torch.compile(_FastText(model.text_embedding), **COMPILE_KW)
self.blocks = [torch.compile(_FastBlock(b), **COMPILE_KW) for b in model.blocks]
self.cross_kv = [torch.compile(_FastCrossKV(b), **COMPILE_KW) for b in model.blocks]
self.head = torch.compile(
    _FastHead(model.head, self.grid, model.patch_size, model.out_dim, guide_scale), **COMPILE_KW)
```

`_FastBlock.forward` (same file) is the region inside one graph: norm1 + modulation, QKV,
QK-RMS norm with its all-reduce, RoPE, self-attention, O-proj with all-reduce, norm3 +
gated residual, cross-attention, O-proj with all-reduce, norm2 + residual, FFN with
all-reduce, gated residual.

The VAE (`models/vae_static21.py:40, 316-317`) and the text encoder
(`models/fast_t5.py:28, 106-107`) use the same `COMPILE_KW`; this study varies only the DiT.

## 3. The ladder (switches: `FAST_SCOPE`, `FAST_ATTN_IMPL`, kernel flags; to be added)

One run per step, same spec, same node. Left to right, each step changes one thing.
Steps 1-4: everything native PyTorch, only the compile scope and `fullgraph` change.
Steps 5-7: on the best scope, kernels are added.

| Step | Attention | Other kernels | Compile scope | `fullgraph` | Graphs per denoise step |
|---|---|---|---|---|---|
| 1 | native `F.scaled_dot_product_attention` | none | sub-layer pieces, by explicit `graph_break()` (3 per block) | `False` | 120 |
| 2 | native SDPA | none | sub-layer pieces, each its own compiled unit | `True` | 120 |
| 3 | native SDPA | none | one graph per transformer block | `True` | 40 |
| 4 | native SDPA | none | all 40 blocks in one graph | `True` | 1 |
| 5 | legacy NKI `wan_flash_self_attn` | none | block | `True` | 40 |
| 6 | nkilib `attention_cte` | none | block | `True` | 40 |
| 7 | nkilib `attention_cte` | `output_projection_cte` + fused adaln | block | `True` | 40 |

What each step shows:

- 1 -> 2: same 120 boundaries, `fullgraph` False vs True. The flag does not change the
  graphs; it decides whether an unplanned break is an error (`True`) or silent (`False`).
- 2 -> 3: 120 -> 40 graphs per step. The cost of boundaries: launches, host gaps, and
  intermediates written to HBM and read back.
- 3 -> 4: 40 -> 1. Whether fewer boundaries still pay once graphs are large, and the price
  in compile time.
- 3 -> 5 -> 6: same scope, native attention vs our hand-written NKI kernel vs nkilib
  `attention_cte`.
- 6 -> 7: the remaining staging kernels on top (step 7 is today's best, 7.2 s total).

### Trade-offs per variant (expected; the runs will confirm or correct)

**`block` — one graph per block, `fullgraph=True`**
- Gains: one launch per block; full fusion inside the block; all-reduces scheduled next to
  their matmuls; one NEFF reused by all 40 blocks, so compile is moderate (minutes).
- Costs: 40 launches per step plus a host gap before each; no fusion across block
  boundaries; every op in the block must be traceable (no Python-side branches, prints,
  imports, `.item()`), or compilation fails.
- Best for: production latency with a compile time you can live with.

**`block_breaks` — same region, `fullgraph=False`, explicit `graph_break()`s**
- Gains: tolerant: code that cannot be traced (logging, data-dependent Python, a library
  call dynamo does not support) just splits the graph instead of failing; easy to bring up
  a new model.
- Costs: 3 launches per block (120 per step), each with a host gap and dynamo guard
  checks; no fusion across a break; intermediates written to HBM and read back at every
  break; more NEFFs to compile and load. Silent: a new break added by a code change goes
  unnoticed unless you check the dynamo counters.
- Best for: bring-up and debugging, not latency.

**`sublayer` — the same pieces as separate `fullgraph=True` units**
- Gains: the same pieces as `block_breaks`, but every boundary is deliberate and visible
  in the code; nothing breaks silently; each piece is profiled on its own (one NEFF per
  sub-layer), which makes per-sub-layer attribution easy.
- Costs: the same launches, gaps and HBM round trips as `block_breaks`.
- Best for: profiling and per-sub-layer tuning; shows that the boundaries, not the flag,
  set the cost.

**`model` — all 40 blocks in one graph, `fullgraph=True`**
- Gains: one launch per step, no gaps between blocks; the compiler can overlap the end of
  one block with the start of the next.
- Costs: a much bigger graph: the 40 blocks are inlined, so compile time grows a lot
  (possibly tens of minutes) and the NEFF is larger to load; one failure anywhere fails the
  whole model; a profile is one huge NEFF, harder to read. The per-block launch gaps it
  removes are already small in `block`, so the gain is likely small.
- Best for: squeezing the last per-step overhead once compile time is acceptable.

| | `block` | `block_breaks` | `sublayer` | `model` |
|---|---|---|---|---|
| Launches per step | ~40 | ~120 | ~120 | 1 |
| Fusion | within block | within piece | within piece | across blocks |
| HBM round trips at boundaries | per block | per piece | per piece | none between blocks |
| Compile time | moderate | moderate (more NEFFs) | moderate (more NEFFs) | longest |
| Fails loudly on untraceable code | yes | no (splits silently) | yes | yes |
| Profile granularity | per block | per piece | per piece | whole model |
| Expected latency | best or near-best | worst | ~same as `block_breaks` | ~`block`, maybe slightly better |

Planned code difference (shown side by side once implemented):

```python
# block_breaks: same compiled region, fullgraph=False, explicit breaks inside forward()
y = self_attention(...)
torch._dynamo.graph_break()        # graph 1 ends here
y = cross_attention(...)
torch._dynamo.graph_break()        # graph 2 ends here
y = ffn(...)                       # graph 3
torch.compile(_FastBlock(b), backend="neuron", dynamic=False, fullgraph=False)

# sublayer: each piece is its own fullgraph unit
self.sa  = [torch.compile(_SelfAttnPart(b), **COMPILE_KW) for b in model.blocks]
self.ca  = [torch.compile(_CrossAttnPart(b), **COMPILE_KW) for b in model.blocks]
self.ffn = [torch.compile(_FFNPart(b), **COMPILE_KW) for b in model.blocks]
```

## 4. How the variants show up in a Neuron Explorer system trace (expected)

| Variant | Executions per step | Execution length | Gaps between executions |
|---|---|---|---|
| `block` | ~40 (+ a few tiny: time, patch, head) | ~13.6 ms each | short: host launch only; all-reduces inside each execution |
| `block_breaks` / `sublayer` | ~120 | shorter pieces | one gap before each piece (host back in Python, dynamo guard checks, next launch); extra DMA at piece boundaries (intermediates to and from HBM); pieces sum to more than one block |
| `model` | 1 | ~550 ms | almost none |

`block_breaks` and `sublayer` should look nearly identical: same boundaries.

Three numbers read from each trace:
1. Executions per denoise step.
2. Device busy fraction = total NEFF time / step wall time.
3. Median gap between consecutive executions.

Step time ≈ total NEFF time + total gap time; breaks raise both.

## 5. Results (to fill in after the runs)

| Step | Graphs/step (dynamo) | Break reasons | Warm-up compile | Graph NEFF time | Denoise | s/step | Total | Exec/step (trace) | Busy fraction | Median gap |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | | |
| 2 | | | | | | | | | | |
| 3 | | | | | | | | | | |
| 4 | | | | | | | | | | |
| 5 | | | | | | | | | | |
| 6 | | | | | | | | | | |
| 7 | | | | | | | | | | |

## 6. System traces (Neuron Explorer screenshots, to add)

- Step 1: _pending_
- Step 2: _pending_
- Step 3: _pending_
- Step 4: _pending_
- Step 5: _pending_
- Step 6: _pending_
- Step 7: _pending_
