# wholeBrainSeg_Large_UNEST_segmentation pack4 交接记录

日期：2026-06-10

## 目标

本交接文档给新的 Codex 对话使用，目标不是重新证明 wholeBrain 模型能跑，而是继续把 `wholeBrainSeg_Large_UNEST_segmentation` 的复刻路径往 pack4 主链收敛，并在不引发 OOM / 重启的前提下继续对齐。

## 当前结论

- `wholeBrainSeg_Large_UNEST_segmentation` 这条 MONAI 模型在 Unity 里已经稳定跑通。
- 当前最可信的安全基线是：
  - `E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\wholeBrainSeg_Large_UNEST_segmentation_unity\chris_t1_run46_full_compare_attnfix2_safe\summary.txt`
- 其中关键结果是：
  - `compare_labelmap_equal_ratio=0.998750372`
  - `mismatch_count=11427`
- 这表示当前工作重点已经不是“整体跑不通”或“主逻辑完全错误”，而是继续把复刻路径往 pack4 真实执行对齐。

## 安全守卫现状

- 已验证低内存主动中止守卫是有效的：
  - `E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\wholeBrainSeg_Large_UNEST_segmentation_unity\chris_t1_run47_probe_guard_abort_check\runtime_debug.log`
- 这个 run 是故意把 private memory limit 压低，确认能在危险前主动 abort，而不是把机器拖死。
- 现有安全相关代码位置：
  - `E:\Projects\AIImage\Assets\Editor\NcnnDebugRunner.cs`
  - `E:\Projects\AIImage\Assets\Scripts\MONAINcnnReproRunner.cs`

## 现有关键安全参数

这些默认值已经在当前流程中被证明是有意义的，不要随便移除：

- `AIIMAGE_MONAI_CLEAR_TEMP_POOL_EACH_PATCH=1`
- `AIIMAGE_MONAI_TEMP_POOL_CLEAR_INTERVAL=1`
- `AIIMAGE_MONAI_YIELD_INTERVAL=1`
- `AIIMAGE_MONAI_MANAGED_CLEANUP_INTERVAL=1`
- `AIIMAGE_MONAI_RESOURCE_SNAPSHOT_INTERVAL=1`
- `AIIMAGE_MONAI_ABORT_PRIVATE_MEMORY_MB=8192`
- `AIIMAGE_BATCH_TIMEOUT_MINUTES=10`

相关代码锚点：

- `E:\Projects\AIImage\Assets\Editor\NcnnDebugRunner.cs`
  - `RunMonaiDebugBatch`
  - `RunBatchBlocking`
  - `MonaiAbortPrivateMemoryMbEnvVar`
  - `MonaiYieldIntervalEnvVar`
- `E:\Projects\AIImage\Assets\Scripts\MONAINcnnReproRunner.cs`
  - `RunSlidingWindowInferenceAsync`
  - `RunSlidingWindowProbeInferenceAsync`
  - `MaybeRunSlidingWindowMaintenanceAsync`
  - `ThrowIfSlidingWindowMemoryLimitExceeded`

## 当前 wholeBrain 已经解决过的关键问题

- attention flatten / reshape 相关对齐问题已经修到很后面，`run46` 已经不是早期那种大面积错图。
- sliding window 45 patch 全量跑通。
- 模型加载、baseline tensor 输入、labelmap compare、summary / runtime log 都已成体系。

## 后续线程不要重新做的事

- 不要重新从 “这个模型能不能在 Unity 跑起来” 开始。
- 不要先去关掉内存守卫再跑大 case。
- 不要把主精力放回老的 compute buffer 调试路径，除非为了快速隔离单个 layer。

## 当前更像瓶颈的方向

根据现有仓库状态，后续 pack4 收敛优先看这些 layer / 路径：

- `NcnnGemmLayerRepro`
  - `E:\Projects\AIImage\Assets\Scripts\NcnnLayers\NcnnGemmLayerRepro.cs`
  - 文档状态：`pack4 RT = 无`，`cmd = 占位`
- `NcnnLayerNormLayerRepro`
  - `E:\Projects\AIImage\Assets\Scripts\NcnnLayers\NcnnLayerNormLayerRepro.cs`
  - 文档状态：`pack4 RT = 无`，`cmd = 占位`
- `NcnnMatMulLayerRepro`
  - `E:\Projects\AIImage\Assets\Scripts\NcnnLayers\NcnnMatMulLayerRepro.cs`
  - 文档状态：`pack4 RT = 无`，`cmd = 占位`
- `NcnnMultiHeadAttentionLayerRepro`
  - `E:\Projects\AIImage\Assets\Scripts\NcnnLayers\NcnnMultiHeadAttentionLayerRepro.cs`
  - 文档状态：`pack4 RT = 无`，`cmd = 占位`
- `NcnnSoftmaxLayerRepro`
  - `E:\Projects\AIImage\Assets\Scripts\NcnnLayers\NcnnSoftmaxLayerRepro.cs`
  - 文档状态：仅部分 `axis==0` pack4 RT / cmd RT
- `NcnnPermuteLayerRepro`
  - `E:\Projects\AIImage\Assets\Scripts\NcnnLayers\NcnnPermuteLayerRepro.cs`
  - 文档状态：当前仅部分 `dims=3` pack4
- `NcnnInstanceNormLayerRepro`
  - `E:\Projects\AIImage\Assets\Scripts\NcnnLayers\NcnnInstanceNormLayerRepro.cs`
  - 文档状态：部分 pack4，依赖 affine / texture path

参考总表：

- `E:\Projects\AIImage\Documents\Ncnn_Vulkan_Layer_复刻覆盖统计_2026-06-05.md`

## 推荐的推进顺序

1. 先读取 `run46` 的 `summary.txt` 和 `runtime_debug.log`，确认当前基线。
2. 用 `probe_only` 或 `max_patches=1` 做小步实验，不要一上来跑满 45 patch。
3. 每次只针对一个 layer 家族推进 pack4，避免多变量一起动。
4. 一旦单层 pack4 路径接通，先做：
   - probe
   - 小 patch compare
   - 再做全量 compare
5. 如果某层 pack4 路径导致资源波动明显，优先保留安全 fallback，不要硬顶到整机不稳定。

## 推荐调试入口

统一优先用：

- `E:\Projects\AIImage\Tools\MonaiToNCNN\RunMonaiUnityDebug.bat`

默认 Unity 方式：

```text
C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe -batchmode -quit -projectPath E:\Projects\AIImage -executeMethod NcnnDebugRunner.RunMonaiDebugBatch
```

## 给新对话的建议指令

可以直接这样开新线程：

```text
继续 AIImage 工程里的 wholeBrainSeg_Large_UNEST_segmentation pack4 收敛。以 run46 全量对齐结果为基线，保持现有内存守卫开启，优先从 Gemm / LayerNorm / MatMul / MultiHeadAttention / Softmax / Permute 这些 pack4 覆盖不完整的层继续推进，并用 Unity batchmode 做 probe、小 patch、再全量 compare 的节奏验证。
```
