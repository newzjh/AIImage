# CodeFormer 单脸修复流程

更新时间：2026-06-01 01:29 +08:00

## 1. 目的与范围

这份文档用于把当前对话里围绕 `CodeFormer` 单脸修复链路做过的核对、修正、验证和后续建议整理成可交接状态，供其他 IDE 工具或新对话直接接着做。

本文默认把用户口述的“`0.3jpg`”统一按仓库内实际文件 `ref/CodeFormer-ncnn-main/data/03.jpg` 记录。

本文覆盖：

- Unity 复刻链路与官方 `ref/CodeFormer-ncnn-main` 的单脸修复对齐
- 单脸对齐后 `512x512` 人脸进入 CodeFormer 之后的修复本体
- 最终回贴相关的已确认问题
- 显存/资源追踪、50+ 压力测试、移动端判断

本文不再依赖聊天附件里的原始长日志；关键结论已尽量消化到正文。

## 2. 参考实现与关键入口

官方参考：

- `ref/CodeFormer-ncnn-main/src/face.cpp`
- `ref/CodeFormer-ncnn-main/src/pipeline.cpp`
- `ref/CodeFormer-ncnn-main` 依赖的推理框架来自 `ref/ncnn-master`

Unity 当前关键入口：

- `Assets/Scripts/CodeFormerNcnnReproRunner2.cs`
  - `ProcessAsync`：整条 CodeFormer 单脸/多脸入口
  - `ConvertClippedPack4ToTexture2D`：最终 `out -> RGB` 转图
  - `PasteAlignedFaceInPlace`：最终回贴入口
- `Assets/Scripts/NcnnFaceRegionGenerator.cs`
  - `GenerateAsync`：人脸检出、landmark、对齐前置
  - `preferTexturePathForFaceDetector`
  - `useArgbFloatForDetector`
- `Assets/Scripts/NcnnFaceRegionPaster.cs`
  - `PasteAlignedFaceWithSoftMask`：自定义软 mask 回贴
- `Assets/Scripts/NcnnCompute/NcnnOps.cs`
  - `PackRgbToPack4Gfpgan`
- `Assets/Resources/NcnnCompute.compute`
  - `NcnnPackRgbToPack4Gfpgan`
- `Assets/Scripts/NcnnCompute/NcnnRepro2.cs`
  - `RentTempArray`
  - `InferResult.Dispose`
  - `GetOrConvertToBuffer.exact / physical / trimmed`
- `Assets/Scripts/NcnnCompute/NcnnGpuResourceTracker.cs`
  - GPU buffer / RenderTexture 追踪
- `Assets/Editor/NcnnDebugRunner.cs`
  - `RunCodeFormerDebug`
  - `RunCodeFormerStressBatch`

已有的上一份相关文档：

- `Documents/CodeFormer_ncnn_yolov7_人脸检出与回贴对齐排查总结.md`

那份文档偏 detector/多脸/回贴对齐背景；本文偏单脸 CodeFormer 修复本体和当前显存状态。

## 3. 当前结论

截至本文写入，结论可以概括为：

1. 单脸 CodeFormer 修复本体已经基本对齐官方 `CodeFormer-ncnn`。
2. 最关键的差异点不是网络结构整体错了，而是输入/输出方向语义和最终转图、回贴细节。
3. 最终“回贴后右上偏 1 像素”的问题，用户已验证修好。
4. 当前没有再复现出 `RenderTexture` 泄露；最新 60 次压力测试后 `live_buffers=0`、`live_rts=0`。
5. 当前显存压力的主体不是 RT 泄露，而是 `ComputeBuffer` 侧的大量中间转换。
6. 现阶段不建议优先把核心 pack4 链路改成 `ARGB32` 或 `R11G11B10`；收益和风险不成比例。

## 4. 本对话已完成的关键修正与核对

### 4.1 先把单脸问题从 detector / paste 中剥离出来

这轮排查里，先把关注点压缩到：

- 对齐后的 `00_face512.png`
- CodeFormer encoder / generator 本体
- `16_out_rgb.png`

也就是尽量不让 detector 和最终 paste 混入变量。这样做之后，问题从“大链路哪里都可能错”缩小成了“CodeFormer 单脸本体和最终转图哪里还不一致”。

### 4.2 单脸输出大幅靠近官方的核心修正：encoder 输入补 `flipY`

落地点：

- `Assets/Scripts/NcnnCompute/NcnnOps.cs`
- `Assets/Resources/NcnnCompute.compute`
- `Assets/Scripts/CodeFormerNcnnReproRunner2.cs`

当前状态：

- `PackRgbToPack4Gfpgan` 已支持 `flipY`
- `CodeFormerNcnnReproRunner2` 里 encoder 输入当前走的是：
  - `PackRgbToPack4Gfpgan(face512, ..., flipY: true)`

这一步是把单脸修复输出拉回官方的最关键修正之一。此前历史对比日志中，官方与 Unity 的 `out` 前 32 个 float 的 mean abs diff 已降到约 `0.0075`，说明网络本体差异已经很小。

### 4.3 最终 `out -> 16_out_rgb.png` 的方向与数值语义已修正

落地点：

- `Assets/Scripts/CodeFormerNcnnReproRunner2.cs` 的 `ConvertClippedPack4ToTexture2D`

当前做法：

- 先把 `clipTex` 用 `Pack4ToBufferCHW` 读回成 CHW float
- 再按官方语义做 `[-1, 1] -> [0, 1]` 映射：
  - `value * 0.5 + 0.5`
- 转回 `Texture2D`
- 同时做了行方向修正

这一步之后，`16_out_rgb.png` 的方向已经不再是此前那种“内容基本对但上下语义不对”的状态。

### 4.4 中间 blob 已证实基本对齐，问题不在大部分网络主体

这轮曾经做过一段很关键的中间 blob 对照：

- 以同一个 `00_face512.png` 作为输入
- 比较官方风格 ncnn 流程和 Unity 复刻流程的 generator 中间 blob

结论是：

- `1383 / 1425 / 1454 / 1459 / out_pack4` 这类关键节点的统计已经非常接近
- 说明此前那种明显“眼线/五官被拉坏”的大偏差，不是整条网络从前面就算错了

这一步非常重要，因为它把问题从“整个网络实现可能有误”缩小成了“少数输入输出语义和后处理问题”。

### 4.5 最终回贴右上偏 1 像素问题已修复

落地点：

- `Assets/Scripts/NcnnFaceRegionPaster.cs`

处理方式：

- 去掉了自定义 paster 中额外的 `tx/ty + 1` 补偿

状态：

- 用户已明确验证：“偏移 1 个像素的问题我已验证修好了”

注意：

- 这个问题是在 Unity 自定义 paster 路径里修的
- 不是官方 `pipeline.cpp` 本身有问题

### 4.6 为继续追 RT 分配点，补了更细的 GPU 追踪标签

这部分是当前工作树里仍然保留的追踪增强，主要用于继续追 `NcnnRepro2` 内部 `512x512x1 ARGBHalf` 申请点。

落地点：

- `Assets/Scripts/NcnnCompute/NcnnGpuResourceTracker.cs`
  - 新增 `UpdateTextureLabel`
- `Assets/Scripts/NcnnCompute/NcnnRepro2.cs`
  - `RentTempArray(...)` 现在会自动带上 `CallerMemberName / CallerLineNumber`
  - 申请标签形如 `NcnnRepro2.RentTempArray(GenerateAsync:233)|new`
- `Assets/Scripts/CodeFormerNcnnReproRunner2.cs`
  - 对几个关键 RT 额外打标签：
    - `CodeFormer.encoderInput`
    - `CodeFormer.generatorOut`
    - `CodeFormer.clipTex`

这些修改主要是调试辅助，不是业务逻辑修正本身。

### 4.7 `InferResult.Dispose()` 的 buffer 释放追踪已补齐

落地点：

- `Assets/Scripts/NcnnCompute/NcnnRepro2.cs`

处理方式：

- `InferResult.Dispose()` 现在会对 `_tempOwned` 里的 `ComputeBuffer` 先调用 `NcnnGpuResourceTracker.ReleaseBuffer(...)`

意义：

- 压力测试里 buffer 统计终于可信了
- 之前看起来像“资源没释放”的部分，很可能只是 tracker 没记完整

## 5. 压力测试与显存结论

### 5.1 已新增 60 次压力测试入口

落地点：

- `Assets/Editor/NcnnDebugRunner.cs`

入口：

- 菜单：`Tools/AIImage/Run CodeFormer Stress (60x)`
- 方法：`RunCodeFormerStressBatch()`

支持的环境变量：

- `AIIMAGE_DEBUG_INPUT`
- `AIIMAGE_STRESS_INPUT_DIR`
- `AIIMAGE_STRESS_COUNT`
- `AIIMAGE_FACE_BUFFER_PATH`
- `AIIMAGE_FACE_PROB_THRESHOLD`
- `AIIMAGE_FACE_NMS_THRESHOLD`

其中：

- `AIIMAGE_FACE_BUFFER_PATH=buffer` 表示 detector 强制走 buffer path
- `AIIMAGE_FACE_BUFFER_PATH=texture` 或不设，默认走 texture path

### 5.2 最新可信压力测试结果

最新可信结果目录：

- `C:\Users\hc\AppData\Local\Temp\YanQi\AIImage\AIImage_CodeFormerStress_20260601_011713\stress_summary.txt`
- `C:\Users\hc\AppData\Local\Temp\YanQi\AIImage\AIImage_CodeFormerStress_20260601_011713\stress_gpu_resources.txt`

关键结论：

- `60/60` 全部成功
- 没有报错
- `live_buffers = 0`
- `live_rts = 0`
- 没有看到 lingering 的 `512x512x1 ARGBHalf` RT

峰值：

- `peak_total_mb = 773.875`
- `peak_buffers_mb = 715.250`
- `peak_rts_mb = 216.625`
- `peak_buffer_count = 44`
- `peak_rt_count = 38`

压力测试输入来自：

- `ref/CodeFormer-ncnn-main/data/02.png`
- `ref/CodeFormer-ncnn-main/data/02_pro.png`
- `ref/CodeFormer-ncnn-main/data/03.jpg`
- `ref/CodeFormer-ncnn-main/data/03_pro.png`

### 5.3 如何解读这些数字

当前最重要的判断：

- 真正值得信的，是 `NcnnGpuResourceTracker` 的 `peak_* / live_*`
- Unity Editor 里看到的 `gfx_mb` 不能直接当作“我们自己泄露了多少”

原因：

- Editor 自身有额外图形内存与池化
- Vulkan / Unity 临时 RT 池会污染肉眼看到的总量
- 即使 `gfx_mb` 首轮到 `3.3GB+`，也不等于我们应用级别有同量 live RT 没释放

所以当前更合理的结论是：

- RT 泄露不是主问题
- 内存主体在 `ComputeBuffer`

### 5.4 关于“追 `NcnnRepro2` 内部 512x512x1 ARGBHalf 申请点”的当前状态

当前状态可以直接下结论：

- 这个点已经被追过一轮
- 最新 60 次压力测试后没有看到它以 live RT 形式留存

也就是说：

- 之前怀疑的“`512x512x1 ARGBHalf` RT 一直挂着不释放”，在当前代码状态下没有再被证明为 leak
- 当前保留的追踪标签代码，更多是为了以后再复查时更快定位，而不是因为它还处于未解状态

## 6. 现在到底“尽量用了 texture path”了吗

结论要实话实说：

- 核心推理链路已经大量使用了 `pack4 RenderTexture`
- 但当前流程还不是“几乎全 texture path”

原因：

1. 最终 `out -> Texture2D` 仍然会回读到 CPU。
2. `NcnnRepro2` 内部仍然存在不少 `GetOrConvertToBuffer.exact / physical / trimmed`。
3. 压力测试里 `peak_buffers_mb` 明显高于 `peak_rts_mb`，说明 buffer 才是当前大头。

所以更准确的描述是：

- “主要主链尽量贴近 pack4 RT”
- “但为了兼容某些 blob 访问、输出转换、物理布局处理，仍有不少 ComputeBuffer”

这也解释了为什么当前移动端压力首先不是“再压一点 RT 格式”，而是“减少不必要的 buffer 物化和转换”。

## 7. 对移动端的判断

### 7.1 现阶段不建议优先改成 `ARGB32` 或 `R11G11B10`

原因：

- `ARGB32` 对 CodeFormer 这类中间 signed feature 来说精度风险太大
- `R11G11B10` 只有 3 通道，而且通常按正值语义使用，不适合通用 `pack4` signed 中间特征
- 即便改了 RT 格式，也不一定能打到当前真正的大头，因为当前峰值主体在 `ComputeBuffer`

换句话说：

- 这不是当前“最划算的第一刀”

### 7.2 更值得优先做的移动端降内存方向

建议优先级：

1. 先找大头 `ComputeBuffer` 转换。
2. 再看 detector 能不能从 `ARGBFloat` 安全降到 `ARGBHalf`。
3. 最后再考虑是否启用 temp pool 减少抖动。

可直接关注的点：

- `Assets/Scripts/NcnnCompute/NcnnRepro2.cs`
  - `GetOrConvertToBuffer.exact`
  - `GetOrConvertToBuffer.physical`
  - `GetOrConvertToBuffer.trimmed`
- `Assets/Scripts/NcnnFaceRegionGenerator.cs`
  - `useArgbFloatForDetector = true`
  - detector 输入当前仍默认 `ARGBFloat`
- `Assets/Scripts/CodeFormerNcnnReproRunner2.cs`
  - `enableTempPool = false`

### 7.3 当前更像是“峰值过高”而不是“稳定泄露”

结合最新 tracker 结果：

- 没有证据表明 RT/Buffer 在一轮轮累计泄露
- 更像是单次处理过程中的瞬时峰值偏高

这对移动端策略意味着：

- 当前首要目标是压峰值
- 不是先花大力气追“已经无法复现的 RT leak”

## 8. 当前仍然建议继续做的事

### 8.1 先做 buffer 大头定位与减法

建议直接从 `stress_gpu_resources.txt` 里重复出现的大 buffer conversion 标签入手，特别是：

- `NcnnRepro2.GetOrConvertToBuffer.exact`
- `NcnnRepro2.GetOrConvertToBuffer.physical`
- `NcnnRepro2.GetOrConvertToBuffer.trimmed`

目标不是先“全改 texture”，而是先回答：

- 哪些转换是 CodeFormer 这条链路里真正必须的
- 哪些只是为了 debug / dump / 通用接口兼容而发生

### 8.2 做 detector `ARGBFloat -> ARGBHalf` 的 A/B 测试

当前 detector 入口默认：

- `Assets/Scripts/NcnnFaceRegionGenerator.cs`
  - `useArgbFloatForDetector = true`

这是一个风险相对可控、但潜在节省明确的实验点。建议做法：

- 单独切成 `false`
- 比较 `proposal_count / picked_count / landmark`
- 用 `02.png`、`03.jpg` 和之前问题样本回归

### 8.3 `enableTempPool` 可以评估，但别在正确性未稳定前盲开

当前默认：

- `Assets/Scripts/CodeFormerNcnnReproRunner2.cs`
  - `enableTempPool = false`

建议：

- 正确性已经基本对齐后，可以做一次仅开 pool 的 A/B 压测
- 重点看峰值、抖动、以及是否引入旧纹理残留

### 8.4 如果未来 paste 又出现细小偏移，优先回看官方 `pipeline.cpp`

当前 1 像素偏移已修好，但如果未来再有：

- 先核对 `trans_matrix_inv`
- 再核对是否错误重复实现了官方 `tx += 1 / ty += 1` 语义
- 最后再看采样中心是按像素角还是像素中心实现

## 9. 当前工作树里值得注意的未提交调试增强

截至写本文时，工作树里和这轮追踪直接相关、但偏“调试辅助”的改动主要有：

- `Assets/Scripts/CodeFormerNcnnReproRunner2.cs`
  - 给关键 RT 打 tracker label
- `Assets/Scripts/NcnnCompute/NcnnGpuResourceTracker.cs`
  - 新增 `UpdateTextureLabel`
- `Assets/Scripts/NcnnCompute/NcnnRepro2.cs`
  - `RentTempArray` 调用点标签
  - `_tempOwned` buffer release tracker 补齐

如果新对话继续追显存问题，请先利用这些标签，不要一上来就删掉。

## 10. 快速复现建议

如果要继续接手排查，推荐顺序：

1. 先看本文和 `Documents/CodeFormer_ncnn_yolov7_人脸检出与回贴对齐排查总结.md`。
2. 用 `Tools/AIImage/Run CodeFormer Debug` 在 `03.jpg` 上重跑单脸。
3. 先确认 dump 里的：
   - `00_face512.png`
   - `16_out_rgb.png`
   - `16_out_chw_preview.txt`
   - `17_full_output.png`
   - `generator_stats.txt`
4. 再跑 `Tools/AIImage/Run CodeFormer Stress (60x)`。
5. 优先看：
   - `stress_summary.txt`
   - `stress_gpu_resources.txt`
6. 如果目标是移动端降峰值，先追 buffer conversion，而不是先改 RT 格式。

## 11. 一句话交接结论

当前单脸 CodeFormer 修复本体已经基本对齐官方；最终 1 像素回贴偏移也已修好；最新压力测试没有证明存在 RT 泄露。下一步最值得做的不是改 `ARGB32/R11G11B10`，而是继续压 `ComputeBuffer` 峰值，并验证 detector 是否能安全从 `ARGBFloat` 降到 `ARGBHalf`。
