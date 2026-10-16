# torch.compile scope on Neuron: Wan2.2-I2V-A14B DiT

Goal: show how the **scope** of `torch.compile` (where graph boundaries fall) and the
`fullgraph` flag change performance, with everything else fixed: same kernels, same math,
same spec (368x480, 21 frames, 10 steps, batched CFG, TP8 on 2 trn2 devices), same node.
Every variant is `torch.compile`; none is eager.

## Narrative: from graph breaks to one graph

### 1. A piece of the model that cannot be captured: RoPE

RoPE (rotary position embedding) tells attention where each token sits. Attention by
itself only compares contents; it has no idea which video token is in which frame, row or
column. RoPE rotates pairs of query/key channels by an angle that depends on the token's
(frame, row, column) position, so the attention score between two tokens depends on how
far apart they are. For video it is 3-D: one set of frequencies per axis. Without it the
model could not tell neighbouring pixels from distant ones.

The original Wan code (`wan/modules/model.py`, `rope_apply`) is written for eager GPU
execution:

```python
for i, (f, h, w) in enumerate(grid_sizes.tolist()):          # tensor values -> Python ints
    x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
    freqs_i = torch.cat([...expand(f, h, w, -1)...]).reshape(seq_len, 1, -1)
    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)       # complex multiply
```

Why it cannot be compiled as part of one graph on Neuron:
- `grid_sizes.tolist()` turns tensor values into Python numbers in the middle of the
  forward pass; the compiler needs the whole computation as a graph of tensor ops, so it
  has to stop the graph there.
- `torch.view_as_complex`, complex multiply and float64: complex numbers and float64 are
  not supported on Neuron at all.
- The angle grid is rebuilt on every call.

Our first Neuron version (`wan/modules/rope_neuron.py`, `rope_apply_neuron`) removed the
complex math, but still calls `grid_sizes.tolist()`, builds the cos/sin grids on the CPU
and copies them to the device (`.to(device)`) on every call, twice per block.

A second example of the same kind: the old TP all-reduce (`models/tp_utils.py:96`) is
decorated `@torch.compiler.disable` "to force a graph break", so every all-reduce ran
outside the compiled graph.

### 2. Why `fullgraph=False` exists

`torch.compile(..., fullgraph=False)` handles code like this automatically: dynamo
compiles everything it can and splits the graph at each line it cannot capture. That lets
people compile a model without rewriting it. The price: one compiled graph per piece,
with Python running between the pieces.

### 3. What that costs: step 1

Step 1 reproduces the cost cleanly. Each transformer block is split into 3 graphs with two
explicit `torch._dynamo.graph_break()` calls, compiled with `fullgraph=False` (in the
ladder RoPE itself is already fixed, so the breaks are placed on purpose at the
sub-layer boundaries, and the measurement isolates the cost of the breaks):

```python
x = self.self_part(x, e, cos, sin)
torch._dynamo.graph_break()          # graph 1 ends
x = self.cross_part(x, ck, cv)
torch._dynamo.graph_break()          # graph 2 ends
return self.ffn_part(x, e)           # graph 3
```

In the system trace (one denoise step = 120 launches):
- `neuron_rt`: many short `kbl_exec_wait` bars, one per launch. After each one comes a
  fixed-cost handoff: the host returns to Python, dynamo checks its guards, and the next
  launch is set up, while the device sits idle.
- `neuron_hw`: `nc_exec_running` is chopped into many short pieces with extra scattered
  work between them.
- `host_to_device_transfers`: a continuous stream through the whole denoise.
- Measured: 0.88 s per step.

### 4. The fix: `fullgraph=True`, which means changing the code

`fullgraph=True` refuses to compile anything it cannot capture, so every such line has
to be rewritten. For RoPE (`models/fast_dit.py`): the video grid is fixed for a run, so
the cos/sin tables are built once at start-up and stay on the device; the per-call part is
plain real-valued tensor math:

```python
# once, at start-up (_rope_tables): grid known ahead of time, cos/sin moved to device once
ang = torch.cat([fr[0][:f]...expand(f, h, w, -1), fr[1][:h]..., fr[2][:w]...], dim=-1)
self.cos, self.sin = torch.cos(ang).to(device), torch.sin(ang).to(device)

# every call, inside the graph (_rope): no .tolist(), no complex numbers, no host copies
xp = x.float().reshape(B, L, n, d // 2, 2)
xr, xi = xp[..., 0], xp[..., 1]
out = torch.stack([xr * cos - xi * sin, xr * sin + xi * cos], dim=-1)
```

The all-reduce gets the same treatment: `funcol.all_reduce(x, "sum", group)`, a
collective the compiler schedules inside the graph, instead of the
`@torch.compiler.disable` version. Other rewrites of the same kind: the timestep
embedding computed once per step as [1, 1, 6, C] instead of per token, and the NKI kernel
wrappers built before tracing starts.

### 5. The impact: step 3

One graph per block, `fullgraph=True` (`COMPILE_KW = dict(backend="neuron",
dynamic=False, fullgraph=True)`; `torch.compile(_FastBlock(b), **COMPILE_KW)`):
- 40 launches per step instead of 120.
- `neuron_rt`: 10 clean, identical groups, one per denoise step.
- `neuron_hw`: one solid, continuous lane; the device stays busy.
- `host_to_device_transfers`: one compact block per step.
- Measured: 0.84 s per step (−5%).
- All 40 blocks are identical, so they share one compiled graph: compile once, run 400
  times per video.

### 6. One more step: the whole model as one graph (step 4)

Move the 40-block loop into one module and compile it once; dynamo unrolls the loop at
trace time:

```python
class _FastModel(nn.Module):
    def forward(self, x, e, cos, sin, *kv):
        for i, b in enumerate(self.blocks):
            x = b(x, e, cos, sin, kv[2 * i], kv[2 * i + 1])
        return x

self.whole = torch.compile(_FastModel(fbs), **COMPILE_KW)
```

- 1 launch per step: one `kbl_exec_wait` and one `nc_exec_running` per denoise step.
- Measured: 0.85 s per step, no faster than step 3: the per-block handoffs it removes
  were already small.
- Costs: warm-up (which includes compiling) 45 s -> 117 s, and the single large NEFF
  takes about 1.3 s to load at process start.

Conclusion: one graph per block is the sweet spot. Past that, the gains come from the
kernels inside the graph: native SDPA 8.4 s, our hand-written NKI attention 16.9 s,
nkilib `attention_cte` 6.3 s, plus o_proj and adaln kernels 5.8 s (denoise, same scope).

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
| 2 | native SDPA | none | 3 compiled units per block: the self-attention sub-layer, the cross-attention sub-layer, the FFN | `True` | 120 |
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

Scope variants by ladder step: `block_breaks` = step 1, `sublayer` = step 2, `block` = step 3, `model` = step 4.

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

## 5. Results (job on pod `wan2-i2v-14b-k6zcb`, branch `exp/compile-scope` @ `0d435e0`)

Same node, one job, VAE (tiled) and UMT5 identical in every step. Denoise and total are
medians of 3 clean runs. "Graph types" is dynamo's `unique_graphs`: distinct graphs
compiled in the process (the 40 blocks share one graph), not launches per step.
Warm-up denoise is the first denoise in a fresh cache and includes compiling.

| Step | Attention | Scope | `fullgraph` | Graph types | Graph breaks | Warm-up denoise | Block NEFF | Denoise | s/step | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SDPA | block + 2 breaks | `False` | 12 | 2 (user-inserted) | 37.1 s | in traces | 8.8 s | 0.88 | 9.6 s |
| 2 | SDPA | 3 units per block | `True` | 12 | 0 | 36.1 s | in traces | 8.8 s | 0.88 | 9.7 s |
| 3 | SDPA | block | `True` | 10 | 0 | 45.4 s | in traces | 8.4 s | 0.84 | 9.3 s |
| 4 | SDPA | whole model | `True` | 10 | 0 | 116.7 s | in traces | 8.5 s | 0.85 | 9.4 s |
| 5 | legacy NKI (in-graph) | block | `True` | 10 | 0 | 163.5 s | 41.63 ms | 16.9 s | 1.69 | 17.8 s |
| 6 | `attention_cte` | block | `True` | 10 | 0 | 52.1 s | 14.96 ms | 6.3 s | 0.63 | 7.2 s |
| 7 | cte + o_proj + adaln | block | `True` | 10 | 0 | 39.1 s | 13.62 ms | 5.8 s | 0.58 | 6.7 s |

What the numbers say:

- 1 vs 2: same 120 boundaries, `fullgraph` False vs True: 8.8 s both. The flag does not
  change speed; the boundaries do.
- 2 vs 3: 120 -> 40 graphs per step: -0.4 s denoise (-5%).
- 3 vs 4: 40 -> 1 graph: no gain (8.5 vs 8.4 s), and warm-up denoise (compile) 45 s -> 117 s.
- 3 vs 5 vs 6 (same block scope): native SDPA 8.4 s, our hand-written NKI kernel 16.9 s
  (2x slower than native), nkilib `attention_cte` 6.3 s. The kernel choice is worth up to
  2.7x; the scope about 5%.
- 6 vs 7: o_proj + adaln kernels, -0.5 s.

## 6. System traces (Neuron Explorer screenshots, to add)

- Step 1: _pending_
- Step 2: _pending_
- Step 3: _pending_
- Step 4: _pending_
- Step 5: _pending_
- Step 6: _pending_
- Step 7: _pending_
