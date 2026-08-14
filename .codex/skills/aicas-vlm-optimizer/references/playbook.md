# AICAS VLM Optimizer — Novel Kernel Engineering Playbook

This playbook provides implementation depth for each novel direction in
SKILL.md.  It is NOT a config tuning guide.

---

## Implementation Guide: Speculative Decoding (Direction 1)

### Phase 1: Build the Draft Head

The draft head must predict the next K tokens from a single hidden state.
It runs AFTER the last LM layer, BEFORE lm_head.

```python
import torch
import torch.nn as nn

class DraftHead(nn.Module):
    """Predict next K tokens from last hidden state."""
    def __init__(self, hidden_size=1536, vocab_size=151936, K=4):
        super().__init__()
        self.K = K
        # Lightweight: one linear per speculative position
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(K)
        ])

    def forward(self, hidden):
        # hidden: [1, 1, 1536]
        return torch.stack([h(hidden) for h in self.heads], dim=2)
        # output: [1, 1, K, vocab_size]
```

### Phase 2: Train the Draft Head

```python
# Collect training data: run model on 200 examples, capture hidden states
# and actual next-K tokens at each decode step
#
# hidden_states: [N, 1536]  (last layer hidden at each step)
# targets:       [N, K]     (actual next K tokens)

optimizer = torch.optim.Adam(draft_head.parameters(), lr=1e-3)
for epoch in range(10):
    logits = draft_head(hidden_states)  # [N, 1, K, vocab]
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
    )
    loss.backward()
    optimizer.step()
```

Target: >60% acceptance rate on first speculative token.

### Phase 3: Verification Loop

```python
def speculative_generate(model, draft_head, input_ids, kv, max_new_tokens, K=4):
    generated = []
    for _ in range(max_new_tokens):
        # 1. Get hidden state from last layer
        out = model.forward(input_ids, past_key_values=kv, output_hidden_states=True)
        hidden = out.hidden_states[-1][:, -1:, :]

        # 2. Draft K candidates
        draft_logits = draft_head(hidden)  # [1, 1, K, vocab]
        draft_tokens = draft_logits.squeeze().argmax(dim=-1)  # [K]

        # 3. Verify: feed all K+1 tokens through model at once
        verify_input = torch.cat([
            out.logits[:, -1:, :].argmax(-1),  # real token 0
            draft_tokens.unsqueeze(0)           # draft tokens 1..K
        ], dim=1)  # [1, K+1]

        verify_out = model.forward(verify_input, past_key_values=kv)
        verify_tokens = verify_out.logits.argmax(-1)  # [1, K+1]

        # 4. Find longest matching prefix
        real_first = verify_tokens[0, 0]
        accepted = [real_first]
        for j in range(K):
            if draft_tokens[j] == verify_tokens[0, j]:
                accepted.append(verify_tokens[0, j + 1])
            else:
                accepted.append(verify_tokens[0, j])
                break

        generated.extend(accepted)
        # Update KV cache for accepted tokens only
        ...

        if any(t == eos_token_id for t in accepted):
            break

    return generated
```

---

## Implementation Guide: Warp-Specialized Attention (Direction 4)

### Architecture

```
Block (128 threads = 4 warps):
  Warp 0: K-reader  — loads K tiles from HBM → shared memory buffer A
  Warp 1: V-reader  — loads V tiles from HBM → shared memory buffer B
  Warp 2: QK-compute — reads K from buffer A, computes QK^T, online softmax
  Warp 3: VAccum    — reads V from buffer B, accumulates weighted sum

Double buffering:
  While Warp 2 computes on buffer A[0], Warp 0 loads into buffer A[1]
  While Warp 3 accumulates from buffer B[0], Warp 1 loads into buffer B[1]
```

### CUDA Skeleton

```cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>

__device__ __shared__ half k_buf[2][BLOCK_KV][HEAD_DIM];
__device__ __shared__ half v_buf[2][BLOCK_KV][HEAD_DIM];
__device__ __shared__ float qk_buf[BLOCK_KV];
__device__ __shared__ float softmax_buf[BLOCK_KV];

__global__ void warp_specialized_decode_attn(
    const half* Q,   // [1, num_heads, 1, head_dim]
    const half* K,    // [1, num_kv_heads, seq_kv, head_dim]
    const half* V,    // [1, num_kv_heads, seq_kv, head_dim]
    half* Out,        // [1, num_heads, 1, head_dim]
    int seq_kv, float scale
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int head_id = blockIdx.x;
    int kv_head = head_id / GQA_RATIO;

    // Load query into registers (all warps need it)
    half q_reg[HEAD_DIM / 32];  // distributed across lanes
    load_q_distributed(Q, head_id, q_reg, lane_id);

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float accumulator[HEAD_DIM / 32] = {0};

    int buf_idx = 0;
    for (int kv_start = 0; kv_start < seq_kv; kv_start += BLOCK_KV) {
        // Warp 0: load K tile
        if (warp_id == 0) {
            load_kv_tile(K, kv_head, kv_start, k_buf[buf_idx], lane_id, seq_kv);
        }
        // Warp 1: load V tile
        if (warp_id == 1) {
            load_kv_tile(V, kv_head, kv_start, v_buf[buf_idx], lane_id, seq_kv);
        }
        __syncthreads();  // all warps wait for load

        // Warp 2: compute QK^T
        if (warp_id == 2) {
            compute_qk(q_reg, k_buf[buf_idx], qk_buf, scale, lane_id);
        }
        __syncthreads();

        // Warp 3: online softmax + accumulate V
        if (warp_id == 3) {
            online_softmax_accumulate(
                qk_buf, v_buf[buf_idx], accumulator,
                &running_max, &running_sum, lane_id);
        }
        __syncthreads();

        buf_idx ^= 1;  // double buffer flip
    }

    // Final: normalize and write output
    if (warp_id == 3) {
        normalize_and_store(accumulator, running_sum, Out, head_id, lane_id);
    }
}
```

---

## Implementation Guide: Async Vision-LM Pipeline (Direction 7)

### Profiling to Estimate Overlap

```python
# Measure vision time
t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)
t0.record()
image_embeds = model.model.visual(pixel_values, grid_thw=grid)
t1.record()
torch.cuda.synchronize()
vision_ms = t0.elapsed_time(t1)

# Measure LM prefill time
t0.record()
output = model.model.language_model(input_embeds, ...)
t1.record()
torch.cuda.synchronize()
lm_ms = t0.elapsed_time(t1)

print(f"Vision: {vision_ms:.1f}ms, LM: {lm_ms:.1f}ms")
print(f"Potential TTFT saving: {min(vision_ms, lm_ms):.1f}ms")
```

### Implementation

```python
def async_prefill(model, pixel_values, grid_thw, text_input_ids, device):
    stream_v = torch.cuda.Stream(device)
    stream_t = torch.cuda.Stream(device)
    event_v = torch.cuda.Event()

    # Vision on stream_v
    with torch.cuda.stream(stream_v):
        vis_output = model.model.visual(pixel_values, grid_thw=grid_thw)
        image_embeds = vis_output.pooler_output
        event_v.record()

    # Text embedding + early LM layers on stream_t (parallel)
    with torch.cuda.stream(stream_t):
        text_embeds = model.model.language_model.embed_tokens(text_input_ids)
        # Run first N layers that don't need image tokens
        hidden = text_embeds
        for layer in model.model.language_model.layers[:SPLIT_LAYER]:
            hidden = layer(hidden, ...)

        # Wait for vision, merge, continue
        event_v.wait(stream_t)
        merged = merge_image_text(hidden, image_embeds, ...)
        for layer in model.model.language_model.layers[SPLIT_LAYER:]:
            merged = layer(merged, ...)

    return merged
```

**Choosing SPLIT_LAYER:** Profile per-layer latency.  Pick the layer N where
`sum(layer[0:N]) ≈ vision_time`.  This maximizes overlap.

---

## Implementation Guide: FP8 Decode Attention (Direction 6)

### Triton FP8 Attention Kernel

```python
@triton.jit
def fp8_decode_attn_kernel(
    Q, K, V, Out,
    stride_qh, stride_kh, stride_kd, stride_vh, stride_vd,
    seq_kv, scale,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    head_id = tl.program_id(0)
    kv_head = head_id // 2  # GQA

    # Load Q in original precision
    q = tl.load(Q + head_id * stride_qh + tl.arange(0, HEAD_DIM))

    # Online softmax variables
    m_prev = float('-inf')
    l_prev = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for kv_start in range(0, seq_kv, BLOCK_KV):
        offs = kv_start + tl.arange(0, BLOCK_KV)
        mask = offs < seq_kv

        # Load K in bf16, cast to fp8 in registers
        k = tl.load(K + kv_head * stride_kh + offs[:, None] * stride_kd
                     + tl.arange(0, HEAD_DIM)[None, :],
                     mask=mask[:, None], other=0.0)
        k_fp8 = k.to(tl.float8e4m3fn)
        q_fp8 = q.to(tl.float8e4m3fn)

        # QK^T in FP8 with FP32 accumulation
        qk = tl.sum(q_fp8[None, :] * k_fp8, axis=1) * scale

        # Online softmax update
        m_curr = tl.maximum(m_prev, tl.max(qk, axis=0))
        p = tl.exp(qk - m_curr)
        p = tl.where(mask, p, 0.0)
        l_curr = tl.exp(m_prev - m_curr) * l_prev + tl.sum(p)

        # Load V, cast to fp8
        v = tl.load(V + kv_head * stride_vh + offs[:, None] * stride_vd
                     + tl.arange(0, HEAD_DIM)[None, :],
                     mask=mask[:, None], other=0.0)

        # Accumulate weighted V in FP32
        acc = acc * tl.exp(m_prev - m_curr) + tl.sum(p[:, None] * v, axis=0)

        m_prev = m_curr
        l_prev = l_curr

    # Normalize
    out = (acc / l_prev).to(Out.dtype.element_ty)
    tl.store(Out + head_id * HEAD_DIM + tl.arange(0, HEAD_DIM), out)
```

**Hardware requirement:** FP8 native on Hopper/Ada.  On Ampere, use INT8
(`tl.int8`) instead with similar pattern.

---

## Implementation Guide: Adaptive Kernel Dispatch (Direction 8)

### Offline Autotuning Script

```python
import triton

# Write your attention kernel with configurable BLOCK sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 32}, num_warps=4),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=4),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=8),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 128}, num_warps=8),
    ],
    key=['seq_q', 'seq_kv'],
)
@triton.jit
def attention_kernel(Q, K, V, Out, seq_q, seq_kv, ...):
    ...

# Benchmark each config for each shape
SHAPES = [(1, s) for s in range(500, 1400, 50)]
for seq_q, seq_kv in SHAPES:
    # Triton autotune will find the best config automatically
    attention_kernel(q, k, v, out, seq_q, seq_kv)
```

Save the dispatch table for runtime use.  No searching at inference time.

---

## Validation Protocol

After implementing any direction:

```bash
# 1. Smoke test (must not crash)
python benchmark.py --num-samples 1 --output smoke.json ...

# 2. Accuracy test (must be 100% match)
python benchmark.py --num-samples 50 --output novel.json ...
python -c "
import json
b = json.load(open('baseline.json'))
n = json.load(open('novel.json'))
match = sum(1 for x,y in zip(b['answers'], n['answers']) if x==y)
total = len(b['answers'])
print(f'{match}/{total} match ({match/total*100:.1f}%)')
assert match == total, 'ACCURACY REGRESSION — revert!'
"

# 3. Performance measurement
# Compare TTFT and throughput vs baseline
```

---

## Checklist

- [ ] I picked a direction from SKILL.md based on profiling data
- [ ] I wrote NEW kernel/system code (not modified a config)
- [ ] My optimization preserves bit-identical outputs (within FP tolerance)
- [ ] I have a fallback path for unsupported shapes
- [ ] 100% accuracy match on 50 samples
- [ ] I measured and recorded TTFT Δ and throughput Δ
