# ESRGAN(repro) vs Real-ESRGAN(exe / ncnn-vulkan) 差异对比与 Shader 对齐清单

目标：把 Unity 内的 ESRGAN 复刻路径（repro）在“处理逻辑 + shader 计算路径 + 性能特征”上尽可能贴近 ref 目录下编译出来的 realesrgan-ncnn-vulkan（exe），重点聚焦卷积相关 shader（目前最大可疑性能差异点）。

## 1. 两条路径是什么

- **realesrgan(exe) 路径**：Unity 侧只做 PNG 编解码与进程调用，核心计算由 ncnn-vulkan 完成。入口：[RealEsrganNcnnVulkanRunner.cs](file:///e:/Projects/AIImage/Assets/Scripts/RealEsrganNcnnVulkanRunner.cs)
- **ESRGAN(repro) 路径**：Unity 侧用 `ComputeShader` + `RenderTexture(Texture2DArray pack4)` 复刻 ncnn 计算图。入口：[RealEsrganNcnnReproRunner.cs](file:///e:/Projects/AIImage/Assets/Scripts/RealEsrganNcnnReproRunner.cs)，算子集合：[NcnnCompute.compute](file:///e:/Projects/AIImage/Assets/Resources/NcnnCompute.compute)

## 2. ncnn-vulkan 关键事实（决定性能上限的点）

对比 ncnn 原版代码可见：当满足条件（尤其是 **3x3 s1 d1**，且 `num_input >= 16 && num_output >= 16`），ncnn-vulkan 会启用 **Winograd 变换 + GEMM** 的专用管线，而不是直接 3x3 空间域卷积：

- 入口逻辑（选择 winograd/coopmat 等）：[convolution_vulkan.cpp](file:///e:/Projects/AIImage/ref/ncnn-master/src/layer/vulkan/convolution_vulkan.cpp#L178-L205)
- Winograd23 transform input（pack4）：[convolution_pack4_3x3s1d1_winograd23_transform_input.comp](file:///e:/Projects/AIImage/ref/ncnn-master/src/layer/vulkan/shader/convolution_pack4_3x3s1d1_winograd23_transform_input.comp)
- Winograd23/43 会对 **weight 做预变换**（CPU 侧或 pipeline 创建阶段），并将权重按更适合 GEMM 的 layout/packing 存储：[convolution_vulkan.cpp](file:///e:/Projects/AIImage/ref/ncnn-master/src/layer/vulkan/convolution_vulkan.cpp#L210-L257)

结论：exe 的大头性能来自 **卷积路径本身就是完全不同的算法与权重布局**（Winograd/GEMM/coopmat/fp16），而 repro 当前主要是空间域直接卷积（即使做了 groupshared tile，也仍是 O(9 * Cin) 的直接算）。

## 3. repro vs ncnn 的主要差异（按“优先级/影响范围”排序）

### A. 算法路径差异（最高优先级）

- **ncnn**：3x3 s1 d1 pack4 优先走 Winograd23/43（transform input → gemm → transform output），可选 cooperative matrix（子组矩阵乘）+ fp16 storage/arith。
- **repro**：`NcnnConv3x3Pack4` 是空间域 3x3 直接卷积（每输出像素做 9 * InPacks 次 dot），无法达到 ncnn 的数量级性能。

对齐动作（要做）：
- 新增 Winograd23（先）/Winograd43（后）的 3 个 kernel：`transform_input / gemm / transform_output`，并在 ESRGAN 复刻图中对满足条件的 3x3 s1 d1 卷积切到 Winograd 管线。

### B. 权重布局与预处理差异（高优先级）

- **ncnn**：对 kernel 做 Winograd 预变换并重排（layout 改成 GEMM 友好），并可选择 fp16 packed。
- **repro**：权重基本按原始 pack4 layout 存在 `_ConvW4`，在 shader 内按 `op, ip, k` 逐次读取。

对齐动作（要做）：
- 在 C# 权重加载阶段，为 Winograd 卷积生成 `weight_tm`（16 or 36 个分量）并按 ncnn 的 pack4 方式写入 `ComputeBuffer`。

### C. 数值类型与硬件路径差异（高优先级）

- **ncnn**：大量使用 `sfpvec4/afpvec4`（fp16 storage / fp16 arithmetic 可选），并依赖 Vulkan 的子组/coopmat 特性（如果硬件支持）。
- **repro**：Unity `RenderTextureFormat.ARGBHalf` 是 fp16 存储，但 HLSL 的运算大多是 `float/float4`（可能升为 fp32），且没有使用 Vulkan subgroup/coopmat。

对齐动作（要做）：
- 先做 fp16 storage + fp32 算术的 Winograd baseline（稳定优先）。
- 再评估把关键路径改成 `half/half4`（fp16 arithmetic）是否在 Vulkan/D3D11 都可稳定编译与提速。

### D. Kernel 粒度与数据复用策略（中优先级）

- **ncnn**：transform kernel 一次处理 4x4→16 或 6x6→36 块，GEMM kernel 具有更高算术强度；全程大量复用中间结果。
- **repro**：每个输出像素独立做 9 个采样，算术强度较低，访存比例高。

对齐动作（要做）：
- Winograd 的中间张量使用 `ComputeBuffer`（而不是 RT），避免 RT 读写与格式转换开销。

### E. 同步/Barrier 与可变控制流（低优先级，但会造成灾难性退化）

- 在 repro 中，一旦出现 **每 ip 一次 barrier** 或者 barrier 处在“varying flow”，会导致严重的性能倒退甚至编译失败。
- 近期回归案例：`NcnnConv1x1Pack4` 曾引入 per-ip barrier，导致耗时显著增加；现已去除（保持无 barrier 的直读版本）。

## 4. 已做的对齐修复（避免 160s 回退）

- `NcnnConv1x1Pack4` 已移除 per-ip barrier 与 shared 权重路径，恢复为无 barrier 的直读权重版本（避免灾难性同步开销）。实现：[NcnnCompute.compute](file:///e:/Projects/AIImage/Assets/Resources/NcnnCompute.compute#L175-L298)

## 5. 下一步对齐实现路线（逐条落地）

1) **Winograd23（pack4）最小闭环**
   - 新增 kernel：`Winograd23TransformInputPack4` / `Winograd23GemmPack4` / `Winograd23TransformOutputPack4`
   - 先只覆盖 ESRGAN 中占比最高的卷积：`3x3 s1 d1 pad=1`，`InPacks>=4 && OutPacks>=4`
   - C#：加载时预生成 `weight_tm23` 并缓存到 `ConvPack`
2) **对齐 ncnn 的 tile/block 定义**
   - 参考 ncnn 的 block_x/block_y（输出按 2x2 tile 分块）与 transform kernel 的 local_size（ncnn winograd23 pack4 transform_input local_size=8x8x1）
3) **fp16 storage +（可选）fp16 arithmetic**
   - 逐步引入 `half/half4`，并保留 fallback 到 float 路径，避免跨后端回归。
4) **（可选）Winograd43/coopmat**
   - Winograd43 与 cooperative matrix 能进一步拉近与 exe 的性能，但对硬件/驱动要求更高，放在 winograd23 稳定后推进。



