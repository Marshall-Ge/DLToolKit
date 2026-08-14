---
name: aicas-vlm-optimizer
description: >
  Guide an agent to discover and implement NOVEL operator-level and
  system-level inference optimizations for Qwen3-VL.  The agent should
  pursue creative, research-grade approaches — not rehash standard
  FlashAttention/FlashDecode recipes.  Forbidden: pruning, caching tricks,
  image resize, model structure changes, hyperparameter scanning.
---

# AICAS VLM Optimizer

You are an agent tasked with inventing **novel** optimizations for Qwen3-VL
inference.  Standard tricks (vanilla FlashAttention, basic CUDA Graph, simple
kernel fusion) are the floor, not the ceiling.  Push beyond them.

## CRITICAL RULES

1. **Write code**, not configs.  If your next action is changing a number, stop.
2. **Be novel.**  If the optimization already exists in `my_kernel/` or is a
   textbook recipe, push further.  Ask: "what would a top systems researcher do?"
3. **No pruning, no caching tricks, no image resize, no model structure changes.**
4. **Verify accuracy** — 100% answer match after every change.
5. **Profile first, then cut the true hotspot.**  Do not keep pushing attention
   kernels when `torch.profiler` / stage timers show GEMM/GEMV is larger.

---

## PROFILE-DRIVEN OPERATING MODE

For this repo family, the current profiling evidence says:

- Steady-state decode bottleneck is usually **decode-side linear / GEMM / GEMV**
  inside CUDA-graph replay, not FlashAttention itself.
- A representative `128-token` run showed:
  - decode replay wall time dominated by `graph_replay`
  - operator profile led by `ampere_bf16*gemm` + `gemvx`
  - flashdecode kernel itself much smaller than the linear stack
- `flashdecode_v2` hit cost is tiny (`~0.025 ms/call`), so if replay is still
  slow, look at **QKV / O-proj / MLP**, not at attention first.

### What to prioritize now

1. **Decode QKV path**
   - packed `qkv` linear
   - q/k prep boundary
   - split/reshape/copy overhead around bridge fastpaths
2. **Decode MLP path**
   - gate/up shared-input reuse
   - down-proj writeback path
   - remove intermediate global-memory traffic
3. **Decode output projection**
   - `o_proj` small-GEMM/matvec path
   - only keep if it wins on real benchmark, not microbench alone
4. **Graph-internal hot path**
   - optimize kernels that run *inside* `graph.replay()`
   - ignore one-time capture cost unless it repeats in benchmark

### What to de-prioritize

- Pure window/bucket knob tuning without code changes
- Reworking FlashAttention again if profiler still shows GEMM/GEMV on top
- Warmup/token replay hacks
- Any profiling method that synchronizes CUDA events during graph capture

### Required workflow for every iteration

1. Add or use **capture-safe** profiling.
2. Identify whether the bottleneck is:
   - `prefill_forward`
   - `graph_replay`
   - `slot_capture`
   - or a specific operator family (`gemm`, `gemvx`, `flash`, `copy/index`)
3. Only then choose one operator family and patch it.
4. Re-run:
   - a short correctness smoke
   - a short throughput benchmark
   - compare against current stable path

If profile says **linear dominates**, the default next move should be:
**“deeper decode-side linear fusion or matvec specialization”**.

## NOVEL OPTIMIZATION DIRECTIONS

The following are research-grade ideas the agent should explore.  Each one
goes beyond standard practice.  Pick the one that best fits the current
bottleneck.

### Direction 1: Speculative Decoding

**Idea:** Generate multiple candidate tokens in one forward pass, verify in
parallel, accept the correct prefix.  Turns sequential decode into partially
parallel decode.

**Why it's novel for AICAS:** Nobody in the competition does this because it
requires careful engineering.  But it can 2-4x decode throughput.

**Approach:**
- Build a lightweight draft head (single linear layer on top of hidden states)
  that predicts the next K tokens simultaneously
- Run one forward pass with K candidate tokens
- Verify against the real model's logits
- Accept the longest correct prefix
- Net effect: generate K tokens in ~1.2 forward passes instead of K

```python
# Draft: predict next K tokens from last hidden state
# hidden: [1, 1, 1536] from layer N
draft_logits = draft_head(hidden)  # [1, K, vocab_size]
draft_tokens = draft_logits.argmax(-1)  # [1, K]

# Verify: run all K+1 positions in one forward pass
verify_input = torch.cat([current_token, draft_tokens], dim=1)  # [1, K+1]
verify_logits = model.forward(verify_input, past_key_values=kv)

# Accept longest matching prefix
real_tokens = verify_logits[:, :-1, :].argmax(-1)
match = (real_tokens == draft_tokens).cumprod(dim=1)
accepted = match.sum().item()
```

**Key challenge:** The draft head must be very fast (<5% overhead) and
reasonably accurate (>60% token match rate).  Train it offline on a few
hundred examples or distill from the model itself.

### Direction 2: Chunked Prefill with Compute-Memory Overlap

**Idea:** Break the prefill phase into chunks and overlap the attention
computation of chunk N with the KV write-back of chunk N-1 using separate
CUDA streams.

**Why it's novel:** Standard prefill processes the entire sequence at once.
For long sequences (1000+ tokens), this creates a huge memory spike and
wastes opportunities for pipelining.

**Approach:**
```python
stream_compute = torch.cuda.Stream()
stream_memory = torch.cuda.Stream()

chunk_size = 256
for i in range(0, seq_len, chunk_size):
    chunk = input_ids[:, i:i+chunk_size]

    with torch.cuda.stream(stream_compute):
        # Compute attention for this chunk
        attn_out = attention(chunk, kv_cache, chunk_offset=i)

    with torch.cuda.stream(stream_memory):
        stream_memory.wait_stream(stream_compute)
        # Write KV to cache while next chunk computes
        kv_cache.append(attn_out.key, attn_out.value)
```

**Benefit:** Reduces peak memory by chunk_size/seq_len factor.  Overlaps
compute and memory ops.  Critical for TTFT on long prompts.

### Direction 3: Persistent Kernel Decode

**Idea:** Instead of launching one kernel per decode step (standard CUDA Graph
replays a pre-captured graph), write a SINGLE persistent kernel that stays
resident on the GPU and executes the entire decode loop without returning to
the CPU.

**Why it's novel:** Even CUDA Graph has CPU-side replay overhead (~3-5μs per
step).  A persistent kernel eliminates ALL CPU involvement during decode.

**Approach:**
```cuda
__global__ void persistent_decode_kernel(
    /* model weights, KV cache, token buffer, max_steps */
) {
    // Thread block stays alive for the entire decode loop
    for (int step = 0; step < max_steps; step++) {
        // 1. Attention: Q×K^T → softmax → ×V
        //    (cooperatively across thread blocks)
        // 2. MLP: gate + up + silu + down
        // 3. LM head: hidden → logits → argmax
        // 4. Write new token to buffer
        // 5. Update KV cache in-place
        // 6. grid.sync() between steps
        __grid_sync();  // cooperative groups
        if (is_eos(new_token)) break;
    }
}
```

**Key challenge:** Requires cooperative groups for grid-wide sync.  Model must
fit in GPU registers + shared memory + L2 for this to outperform CUDA Graph.
Start with just the attention + argmax in persistent mode.

### Direction 4: Warp-Specialized Attention

**Idea:** Within a single kernel, assign different warps to different roles:
some warps read K, some read V, some compute softmax, some do the
reduction.  Producer-consumer pattern via shared memory.

**Why it's novel:** Standard FlashAttention uses all warps identically.
Warp specialization can better hide memory latency by having reader warps
prefetch while compute warps are busy.

**Approach (for decode attention):**
```
Warp 0-3: "K readers" — stream K tiles from HBM into shared memory
Warp 4-7: "V readers" — stream V tiles from HBM into shared memory
Warp 8-11: "Computers" — compute QK^T and softmax on the K tiles
Warp 12-15: "Accumulators" — multiply softmax weights by V tiles
```

**Synchronization:** Use `__syncwarp()` and shared memory barriers between
producer and consumer warps.  Double-buffer the shared memory tiles.

**Performance target:** 20-40% improvement over standard FlashDecode by
fully overlapping HBM reads with compute.

### Direction 5: Cross-Layer KV Sharing

**Idea:** Adjacent transformer layers often produce similar KV
representations.  Share KV cache across pairs of layers, computing KV only
for odd layers and reusing for even layers (or vice versa).

**Why it's novel:** This is NOT pruning — all parameters are preserved and
all layers execute their attention.  We only share the KV *cache*, not the
weights.  The attention computation still uses all query heads.

**Approach:**
```python
# Layer i and layer i+1 share the same KV
for i in range(0, num_layers, 2):
    layer_a = model.layers[i]
    layer_b = model.layers[i + 1]

    # Layer A: compute KV normally
    q_a, k_a, v_a = layer_a.self_attn.project(hidden)
    kv_cache.update(k_a, v_a, layer_idx=i)

    # Layer A: normal attention
    out_a = attention(q_a, k_a, v_a)

    # Layer B: compute only Q, reuse K/V from layer A
    q_b = layer_b.self_attn.q_proj(hidden_b)
    out_b = attention(q_b, k_a, v_a)  # shared KV!
```

**Accuracy risk:** Must calibrate which layer pairs can share without degrading
answers.  Run `--num-samples 50` accuracy check.  If accuracy drops > 1%,
don't share that pair.

### Direction 6: FP8 Compute for Memory-Bound Operators

**Idea:** For memory-bound operations (decode attention, GEMV), converting
to FP8 halves memory traffic and doubles effective bandwidth, without needing
to quantize model weights permanently.

**Why it's novel:** This is NOT quantization.  Weights stay in BF16/FP16
on disk and in the model.  We cast to FP8 on-the-fly inside the kernel,
compute in FP8, and accumulate in FP32.

**Approach:**
```python
@triton.jit
def fp8_decode_attention(Q, K, V, Out, ...):
    # Load K in BF16, cast to FP8 in registers
    k_bf16 = tl.load(K_ptr + offsets)
    k_fp8 = k_bf16.to(tl.float8e4m3fn)

    # Compute QK^T in FP8 (with FP32 accumulator)
    qk = tl.dot(q_fp8, tl.trans(k_fp8), acc=tl.zeros(..., dtype=tl.float32))

    # Softmax in FP32
    ...

    # V×weights in FP8
    v_fp8 = v_bf16.to(tl.float8e4m3fn)
    out = tl.dot(weights_fp8, v_fp8, acc=tl.zeros(..., dtype=tl.float32))
```

**Benefit:** 2x memory throughput for decode attention.  Requires Hopper (H100)
or Ada (RTX 4090) for native FP8.  On Ampere (A100), simulate with INT8.

### Direction 7: Asynchronous Vision-Language Pipeline

**Idea:** The vision encoder and language model prefill are completely
independent until the merge point.  Run them on separate CUDA streams with
true overlap.

**Why it's novel:** Current code runs vision → merge → LM sequentially.  But
vision produces image embeddings while LM only needs text embeddings for the
first few layers.  We can start LM prefill on text tokens while vision is
still running.

**Approach:**
```python
stream_vision = torch.cuda.Stream()
stream_lm = torch.cuda.Stream()
event_vision_done = torch.cuda.Event()

# Start vision encoding (async)
with torch.cuda.stream(stream_vision):
    image_embeds = model.visual(pixel_values, grid_thw=grid)
    event_vision_done.record()

# Start LM prefill on text-only tokens (parallel!)
with torch.cuda.stream(stream_lm):
    text_hidden = model.language_model.embed_tokens(text_token_ids)
    for layer in model.language_model.layers[:N_early_layers]:
        text_hidden = layer(text_hidden)

    # Wait for vision to finish, then merge
    event_vision_done.wait()
    merged = merge(text_hidden, image_embeds)

    # Continue LM prefill with merged input
    for layer in model.language_model.layers[N_early_layers:]:
        merged = layer(merged)
```

**Benefit:** If vision takes 30% of TTFT and text prefill takes 70%, overlap
saves up to 30% of TTFT.

### Direction 8: Adaptive Kernel Dispatch

**Idea:** Instead of using the same kernel for all sequence lengths, build a
dispatch table that selects the optimal kernel *configuration* (tile size,
number of warps, pipeline depth) for each specific shape at runtime.

**Why it's novel:** Standard FlashAttention uses fixed tile sizes.  But
optimal tiling depends on seq_len, head_dim, and hardware (L2 size, SM count,
register file).  An adaptive dispatcher can be 20-50% faster.

**Approach:**
1. Write multiple variants of each kernel with different BLOCK sizes
2. Offline: benchmark each variant on each shape → build a lookup table
3. Runtime: look up the best variant for the current shape, dispatch to it

```python
# Auto-tune at install time
DISPATCH_TABLE = {}
for seq_len in [512, 640, 768, 896, 1024, 1152, 1280]:
    best_time = float('inf')
    for block_q in [64, 128]:
        for block_kv in [32, 64, 128]:
            for num_warps in [4, 8]:
                t = benchmark_kernel(seq_len, block_q, block_kv, num_warps)
                if t < best_time:
                    best_time = t
                    DISPATCH_TABLE[seq_len] = (block_q, block_kv, num_warps)

# Runtime dispatch
def fast_attention(q, k, v):
    S = k.shape[-2]
    nearest = min(DISPATCH_TABLE.keys(), key=lambda x: abs(x - S))
    config = DISPATCH_TABLE[nearest]
    return attention_kernel[config](q, k, v)
```

### Direction 9: Decode Batch Coalescing

**Idea:** Decode generates one token at a time.  The GEMV (matrix-vector)
is severely underutilizing the GPU.  Speculatively batch multiple decode
steps by predicting likely tokens and verifying.

**This differs from Direction 1 (speculative decoding)** in that it doesn't
need a draft model — it uses the model's own top-K predictions:

```python
# Get top-K candidates for positions 1..K
logits = model.forward(token, past_kv)
top_k_tokens = logits.topk(K).indices  # [1, K]

# For each candidate, assume it's correct and predict the next
# Run all K candidates as a batch of size K
batch_input = top_k_tokens.view(K, 1)
batch_logits = model.forward(batch_input, past_kv_expanded)

# Verify: did position 0's prediction match the real next token?
# Accept the chain that matches
```

**Benefit:** Converts K sequential GEMV into 1 batched GEMM, which is far
more efficient.

### Direction 10: Memory-Mapped KV Cache with Prefetching

**Idea:** For very long sequences, KV cache exceeds L2 cache and causes
thrashing.  Implement software prefetching in attention kernels to hide
HBM latency.

**Approach:**
```cuda
// Inside decode attention kernel:
// While computing attention for KV block N,
// prefetch KV block N+1 from HBM into L2
for (int block = 0; block < num_kv_blocks; block++) {
    // Prefetch next block
    if (block + 1 < num_kv_blocks) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(k_ptr + (block+1) * BLOCK_SIZE));
        asm volatile("prefetch.global.L2 [%0];" :: "l"(v_ptr + (block+1) * BLOCK_SIZE));
    }
    // Compute attention on current block
    compute_attention(k_ptr + block * BLOCK_SIZE,
                     v_ptr + block * BLOCK_SIZE);
}
```

---

## How to Pick a Direction

1. **Profile** to find the #1 bottleneck
2. Match it:

| Bottleneck | Best Direction |
|---|---|
| Decode attention latency | 4 (Warp-Specialized) or 6 (FP8) |
| Decode throughput (tok/s) | 1 (Speculative) or 9 (Batch Coalescing) |
| TTFT on long prompts | 2 (Chunked Prefill) or 7 (Async Pipeline) |
| TTFT on image-heavy inputs | 7 (Async Vision-LM Pipeline) |
| CPU launch overhead | 3 (Persistent Kernel) |
| Suboptimal kernel perf | 8 (Adaptive Dispatch) |
| Memory bandwidth bound | 6 (FP8) or 10 (Prefetching) |
| KV cache memory pressure | 5 (Cross-Layer Sharing) or 10 (Prefetching) |

3. **Implement** a proof-of-concept for the top candidate
4. **Benchmark** and verify accuracy
5. If it works, polish.  If not, try the next direction.

## Qwen3-VL Architecture Reference

```
LM Decoder: 28 layers
  hidden=1536, heads=16q/8kv, head_dim=128, intermediate=8960
  MLP: gate[1536→8960] + up[1536→8960] + silu + down[8960→1536]

Vision Encoder: 32 blocks
  hidden=1280, heads=16, head_dim=80
  MLP: fc1[1280→5120] + GELU + fc2[5120→1280]

Decode attention shapes:
  Q: [1, 16, 1, 128]  K/V: [1, 8, S, 128]  S∈[500,1300]

Prefill attention shapes:
  Q: [1, 16, S, 128]  K/V: [1, 8, S, 128]  S∈[500,4096]
```
