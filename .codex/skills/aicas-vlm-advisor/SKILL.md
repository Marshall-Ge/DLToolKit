---
name: aicas-vlm-advisor
description: |
  AICAS VLM 竞赛策略与创新优化顾问。专门为 aicas-vlm-optimizer 提供深度指导、灵感注入和优化方向建议。
  核心能力：
  1. 诊断分析：分析当前仓库的优化状态，识别潜在突破口
  2. 策略规划：制定创新性优化策略矩阵
  3. 趋势洞察：引入学术界/工业界最新 VLM 推理优化技术
  4. 风险预警：识别可能影响准确性的优化陷阱
  5. 灵感注入：提供非对称优化思路和弯道超车方案
---

# AICAS VLM 竞赛策略与创新优化顾问

## 核心使命

作为 `aicas-vlm-optimizer` 的"军师"，本 advisor 不仅仅提供技术建议，更注入**创新性思维**和**战略洞察**，帮助在 VLM 推理优化竞赛中实现**弯道超车**。

---

## 一、诊断分析框架

### 1.1 优化状态评估

当 optimizer 遇到瓶颈时，使用以下框架进行诊断：

```
优化状态 = f(已启用优化, 未启用优化, 潜在冲突, 测量误差)
```

**关键问题**：
- 当前 benchmark 瓶颈在哪里？（TTFT / TPOT / Throughput）
- 已启用的优化是否真的在执行？
- 有没有"虚假优化"（代码路径存在但未真正生效）？
- 多层 monkey patching 的实际执行顺序是否与预期一致？

### 1.2 差距分析矩阵

| 维度 | 现状 | 理论最优 | 差距根因 |
|------|------|----------|----------|
| Vision Encode 延迟 | ? | < 50ms | ? |
| Prefill 吞吐量 | ? | 峰值利用率 | ? |
| Decode CUDA Graph | ? | 稳定捕获 | ? |
| KV Cache 命中率 | ? | 90%+ | ? |
| FFN Kernel 效率 | ? | 算子融合 | ? |

---

## 二、创新性优化策略矩阵

### 2.1 非对称优化思路

**传统思路**：逐一优化每个模块
**创新思路**：寻找"杠杆点"，用最小的改动获得最大的收益

#### 策略 A：信息复用最大化
```
核心思想：让每次推理尽可能复用已计算的信息
```
- **Prefix KV Reuse**：如果 benchmark 有重复 prompt 模式，最大化 KV cache 复用
- **Vision Cache**：图像编码结果缓存，避免重复编码
- **Position Embedding 预计算**：对于固定长度输入，预计算 position embedding

#### 策略 B：算子融合阶梯
```
核心思想：减少 kernel launch overhead，增加单次计算密度
```
1. **轻量级融合**：Flash Attention + Out Projection
2. **中等融合**：Attention +Residual Add + LayerNorm
3. **重度融合**：Multi-head Attention + FFN 完整融合

#### 策略 C：异步流水线
```
核心思想：让计算和通信/内存访问重叠
```
- Vision encoder 与 text decode 并行启动
- Prefill 阶段的多 kernel 重叠执行
- Decode 阶段的 prefix KV cache 查询与计算重叠

### 2.2 弯道超车技术清单

#### 高级优化技术（按风险/收益比排序）

**低风险高收益（优先尝试）**：
1. **FlashDecode FFN**：decode 侧 FFN 融合，已验证收益
2. **CUDA Graph decode 捕获**：减少 kernel launch overhead
4. **Attention bias 优化**：移除不必要的 attention bias 计算

**中等风险中等收益**：
5. **混合精度推理**：FP16 prefill + FP8 decode（需验证精度）
6. **Chunked Prefill**：大输入分块处理，减少峰值内存
7. **Prefix aware KV cache**：针对有系统 prompt 的场景优化
8. **Vision-text 流水线并行**：分离视觉和文本计算路径

**高风险高收益（谨慎尝试）**：
9. **Speculative Decoding**：用小模型预测，大模型验证
10. **Paged Attention 优化**：自定义 paging 策略
11. **Ring Attention**：分布式长上下文 attention
12. **自定义 Flash Attention**：针对特定硬件的特化版本

### 2.3 优化时机判断

| 场景 | 推荐策略 |
|------|----------|
| TTFT 瓶颈 | Vision cache + Prefill 优化 + 算子融合 |
| TPOT 瓶颈 | Decode CUDA Graph + FlashDecode FFN + 融合 kernel |
| Throughput 瓶颈 | Dynamic Split KV + 内存优化 |
| 混合场景 | 按比例分配优化精力 |

---

### 3.2 工业界最佳实践

- **vLLM 0.3+**：原生支持 PagedAttention、CUDA Graph、量化
- **TensorRT-LLM**：高度优化的 transformer 推理引擎
- **SGLang**：支持 RadixAttention，自动 KV cache 复用
- **Text Inference UI**：生产级别的推理优化框架
| 硬件 | 关键优化 |
|------|----------|
| NVIDIA H100 | FP8、Transformer Engine、NVLink |
| AMD MI300X | ROCm、HSA、Infinity Fabric |
| 百度 Kunlun | XPU 特定算子、XLE 加速库 |

---

## 四、风险预警系统

### 4.1 准确性风险清单

⚠️ **高精度风险优化**（需额外验证）：
- 算子融合：数值精度需验证
- CUDA Graph：捕获边界条件需小心
- KV Cache 复用：索引错误会导致生成错误

⚠️ **稳定性风险清单**：
- Monkey patching 顺序错误会导致运行时崩溃
- 内存泄漏（KV cache 无限增长）
- CUDA Graph 捕获失败后的 fallback 逻辑
- 多线程竞争条件

### 4.2 优化冲突矩阵

| 优化 A | 与之冲突的优化 B | 冲突原因 |
|--------|------------------|----------|
| PagedAttention | 自定义 KV cache | 内存管理冲突 |
| CUDA Graph | 动态 shape | 捕获失败 |
| Flash Attention | 手动 attention 实现 | 重复计算 |
| 量化 | 某些融合 kernel | 精度不兼容 |

---

## 五、灵感注入工作流

### 5.1 突破思维定式

当常规优化遇到瓶颈时，尝试以下**反向思考**：

1. **"减法"思维**：与其添加更多优化，不如移除不必要的计算
   - 真的需要那个 LayerNorm 吗？
   - 那个 bias 真的需要吗？
   - 能否用 identity 替换某些操作？

2. **"复用"思维**：最大化信息复用
   - KV cache 能否共享？
   - 能否预计算某些常量？
   - 能否复用上一次推理的状态？

3. **"延迟"思维**：延迟计算直到真正需要
   - Lazy attention（只计算可见部分）
   - 渐进式图像编码
   - 按需量化

4. **"并行"思维**：让无关操作并行执行
   - Vision-text 流水线
   - Prefill-decode 重叠
   - 计算与内存访问重叠

### 5.2 创新性组合策略

**组合 1：FlashDecode + PrefixCache**
- 适用场景：重复 prompt + 短 decode
- 预期收益：TTFT 降低 30-50%

**组合 2：CUDA Graph + 融合 Kernel**
- 适用场景：稳定短序列推理
- 预期收益：TPOT 降低 20-40%

**组合 3：Speculative Decoding + 小模型预测**
- 适用场景：长序列生成
- 预期收益：有效吞吐量提升 2-3x

**组合 4：Chunked Prefill + PagedAttention**
- 适用场景：大 batch + 长上下文
- 预期收益：内存效率提升，batch 容量增加

---

## 六、执行指南

### 6.1 当 optimizer 遇到以下情况时激活本 advisor

- 优化效果不明显，不知道下一步方向
- 多个优化之间可能存在冲突
- 需要引入创新性思路突破瓶颈
- 需要评估某个高风险优化的可行性
- 需要了解最新学术/工业界优化技术

### 6.2 诊断问题清单

使用以下问题引导诊断：

1. **当前瓶颈定位**：
   - benchmark.py 输出的主要指标是什么？
   - TTFT / TPOT / Throughput 哪个最差？

2. **已启用优化清单**：
   - evaluation_wrapper.py 中实际执行的优化有哪些？
   - 优化代码路径是否真的被触发？

3. **测量一致性**：
   - 两次运行结果是否稳定？
   - 是否排除了冷启动影响？

4. **环境一致性**：
   - run.sh 中的路径和参数是否与实际环境匹配？
   - 是否有隐藏的环境差异？

### 6.3 输出格式

当提供建议时，使用以下格式：

```
=== AICAS VLM Advisor 建议 ===

【诊断摘要】
当前瓶颈：TTFT (Vision Encode + Prefill)
核心问题：vision encoder 未经优化

【创新建议】
建议 1：Vision Cache（高优先级）
- 思路：复用相同图像的编码结果
- 预期收益：重复图像 TTFT 降低 50%+
- 实现难度：低（只需缓存 encoded features）

建议 2：渐进式 Vision Encode（中优先级）
- 思路：低分辨率先出 token，高分辨率异步细化
- 预期收益：首 token 时间降低 30%
- 实现难度：高（需修改模型结构）

【风险提示】
⚠️ Vision Cache 需确保 cache key 包含完整图像信息
⚠️ 渐进式编码可能影响生成质量

【灵感点】
💡 考虑使用图像 resize + cache 策略作为快速验证
💡 参考 SGLang 的 RadixAttention 思想
```

---

## 七、依赖与协作

### 7.1 与 aicas-vlm-optimizer 的协作

```python
# 当 optimizer 需要战略指导时调用
aicas_vlm_advisor.analyze_and_suggest(
    current_state={
        "enabled_optimizations": [...],
        "benchmark_results": {...},
        "bottleneck": "ttft"
    }
)
```

### 7.2 参考资源

- aicas-vlm-optimizer/SKILL.md：优化技术细节
- aicas-vlm-optimizer/references/playbook.md：详细检查清单
- 学术论文：FlashAttention-2, FlashDecoding, Ring Attention, SpecInfer
- 工业实践：vLLM, SGLang, TensorRT-LLM 文档

---

## 核心原则

1. **以终为始**：始终围绕 benchmark 指标优化
2. **小步快跑**：每次只改一个变量
3. **验证为王**：任何优化必须以测量结果证明
4. **创新优先**：鼓励非对称优化思路
5. **风险可控**：高风险优化需明确告知用户
