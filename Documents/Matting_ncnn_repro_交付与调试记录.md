# Matting NCNN Repro 交付与调试记录

更新时间：2026-06-01

## 1. 目标

当前工作目标是基于：

- `ref/ncnn_matting-main`
- `ref/MODNet-master`

在 Unity 中复刻一条可运行、可调试、可对照官方结果的 matting 流程。

约束：

- 不改现有 `NcnnRepro` / `NcnnRepro2`
- 新增独立实现 `MatterNcnnReproRunner` / `NcnnRepro3`
- 优先走 `pack4 RenderTexture`
- 如结果正确性与 pack4 texture 路径冲突，先保证正确性，再逐步把子模块切回 texture 版

## 2. 当前关键文件

核心运行：

- `Assets/Scripts/MatterNcnnReproRunner.cs`
- `Assets/Scripts/NcnnCompute/NcnnRepro3.cs`
- `Assets/Scripts/NcnnCompute/NcnnOps.cs`
- `Assets/Resources/NcnnCompute.compute`

调试入口：

- `Assets/Editor/NcnnDebugRunner.cs`
  - `RunMattingDebugMenu`
  - `RunMattingDebugBatch`

模型与样例：

- `Assets/StreamingAssets/Matting/matting.param`
- `Assets/StreamingAssets/Matting/matting.bin`
- `ref/ncnn_matting-main/test_img.jpg`
- `ref/ncnn_matting-main/test_result.jpg`
- `ref/ncnn_matting-main/windows/matting.dll`
- `ref/ncnn_matting-main/windows/matting_gui.exe`

## 3. 已完成工作

### 3.1 新建独立 runner 与图执行器

已新增：

- `MatterNcnnReproRunner`
- `NcnnRepro3`

`NcnnRepro3` 当前只覆盖这份 matting 模型实际用到的 layer 集合：

- `Input`
- `Split`
- `Convolution`
- `BinaryOp`
- `ReLU`
- `Pooling`
- `MaxPoolingInd`
- `MaxUnPooling`
- `Interp`
- `Concat`

### 3.2 补齐底层算子与 shader 能力

已在 `NcnnOps.cs` / `NcnnCompute.compute` 增量补充：

- 卷积内置 `sigmoid`
- 通用 pack4 `Interp`
- `MaxPoolingIndPack4`
- `MaxUnPoolingPack4`

注意：

- 这些新增 kernel 目前并不都在默认路径中使用
- 默认路径会优先选择“已验证正确”的实现

### 3.3 前处理与后处理

`MatterNcnnReproRunner` 当前默认：

- 固定输入到 `512x512`
- `useArgbFloatTensor = true`
- `enableWinograd23 = false`
- `forceBufferConvolution = false`
- `useTextureMaxPoolingInd = false`
- 启用前景清理：
  - 最大前景连通域
  - 小半径闭运算

背景合成色与官方示例对齐：

- `Color32(120, 255, 155, 255)`

### 3.4 官方 DLL 静默对照链路已打通

已确认 `ref/ncnn_matting-main/windows/matting.dll` 可直接静默调用：

- 导出函数：
  - `Init`
  - `MakeUpFile`

Python/ctypes 可直接跑：

```python
import os, ctypes
root = r"E:\Projects\AIImage\ref\ncnn_matting-main\windows"
os.chdir(root)
os.add_dll_directory(root)
dll = ctypes.WinDLL(os.path.join(root, "matting.dll"))
dll.Init()
dll.MakeUpFile(r"E:\Projects\AIImage\ref\ncnn_matting-main\test_img.jpg".encode("utf-8"))
```

输出固定写到：

- `ref/ncnn_matting-main/windows/seg_out.jpg`

## 4. 当前最优结果配置

截至本记录，默认正确性最优配置是：

- `ARGBFloat`
- `enableWinograd23 = false`
- `forceBufferConvolution = false`
- `useTextureMaxPoolingInd = false`
- `MaxPoolingInd` 使用 CPU/buffer 参考实现
- `MaxUnPooling` 使用 CPU/buffer 参考实现
- 最终 alpha 启用前景清理

最新静默批跑结果：

- 对 `ref/ncnn_matting-main/test_result.jpg`
  - `mean_abs_rgb = 3.1120`
  - `max_abs_rgb = 215`

说明：

- 背景脏块已基本压下
- 主干卷积 pack4 texture 路径在关闭 winograd 后可对齐
- 剩余 texture 版问题集中在 `MaxPoolingInd`

## 5. 已定位出的关键结论

### 5.1 Winograd23 会显著拉坏这份 matting 模型

在这份图上，`enableWinograd23=true` 会导致结果恶化。

结论：

- matting 默认应固定 `false`

### 5.2 `ARGBFloat` 是当前对齐官方所需

这份模型在 `ARGBHalf` 下曾出现明显饱和和背景脏块。

结论：

- 在完全对齐官方前，不要回切 `ARGBHalf`
- 等 texture `MaxPoolingInd` 修好后，再做 half 回归

### 5.3 当前 pack4 卷积主链基本已对齐

通过 `conv_compare` 做过逐层对照：

- 前半段 `Conv_20/22/23/...`
- 后半段 `Conv_153 ... Conv_236`

在最佳配置下，很多层已收敛到 `1e-6` 量级。

结论：

- 当前主要误差不再是主干卷积

### 5.4 `MaxPoolingInd` 的 CPU 参考实现是对的

`maxpool_compare` 已验证：

- `MaxPool_19`
- `MaxPool_41`

CPU 参考版输出与 reference 一致。

### 5.5 texture 版 `MaxPoolingInd` 目前仍不正确

已经试过两类 texture 方案：

1. 单 kernel 直接同时求 pooled value 和 indices
2. 两阶段：
   - `PoolingPack4`
   - `MaxPoolingIndicesFromValuePack4`

两类方案在当前工程里都会导致结果明显退化：

- `mean_abs_rgb` 会回到大约 `58.x`

并且 `maxpool_compare` 明确显示：

- `MaxPool_19` / `MaxPool_41` 输出本身就和 CPU 参考大幅偏离

结论：

- 目前还不能把 `MaxPoolingInd` 默认切回 texture 版

## 6. 现有调试方法

### 6.1 Unity 静默批跑

```powershell
$env:AIIMAGE_DEBUG_INPUT='E:\Projects\AIImage\ref\ncnn_matting-main\test_img.jpg'
& 'C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe' `
  -batchmode `
  -projectPath 'E:\Projects\AIImage' `
  -executeMethod NcnnDebugRunner.RunMattingDebugBatch `
  -logFile 'E:\Projects\AIImage\Logs\matting-test.log'
```

关键日志关注：

- `Matting Debug result`
- `Matting Debug compare`
- `dump=...`

### 6.2 批跑产物目录

合成结果目录：

- `%LOCALAPPDATA%\Temp\YanQi\AIImage\AIImage_MattingRepro_*`

中间 blob 统计目录：

- `%LOCALAPPDATA%\Temp\YanQi\AIImage\AIImage_MattingBlobDump_*`

其中常见文件：

- `17_composite.png`
- `18_matte.png`
- `blob_stats.txt`
- `conv_compare.txt`

### 6.3 官方 DLL 对照

官方静默输出：

- `ref/ncnn_matting-main/windows/seg_out.jpg`

可以直接和 Unity 输出做图像差异对照。

### 6.4 常用离线分析

当前用过且有效的离线分析包括：

- 从官方 `test_result.jpg` 反推 alpha 近似真值
- 比较 `our alpha` 与 `official estimated alpha`
- `largest connected component`
- 二值 support 清理
- 灰度 alpha 闭运算试验

## 7. 当前实现状态

### 默认正确性路径

- `MaxPooling`：pack4 texture
- `MaxPoolingInd`：CPU 参考实现
- `MaxUnPooling`：CPU 参考实现
- `Convolution`：pack4 texture
- `Winograd23`：关闭
- `TensorTextureFormat`：`ARGBFloat`

### 已实现但暂不默认启用

- texture `MaxPoolingInd`
- texture `MaxUnPooling`
- 全层 `conv_compare`
- `maxpool_compare`

这些能力保留用于继续调试。

## 8. 下一步建议

下一步最值得做的是只专攻一件事：

### 修正 texture `MaxPoolingInd`

建议顺序：

1. 给 `PoolingPack4` 单独补一个对照路径，确认是 pool value 本身错，还是只在 `MaxPoolingInd` 组合使用时错
2. 如确认是 pool value 错，单独重写 pack4 max-pooling shader
3. 修好后再把 `MaxPoolingIndicesFromValuePack4` 接回去
4. 让 `maxpool_compare mean_abs=0`
5. 再观察最终误差是否仍维持在 `3.x`

只有 texture `MaxPoolingInd` 对齐后，才建议继续做：

### 回切 `ARGBHalf`

回切顺序建议：

1. 保持 `Winograd23=false`
2. 只把 `TensorTextureFormat` 改成 `ARGBHalf`
3. 跑同一张 `test_img.jpg`
4. 如果误差明显反弹或背景噪点回归，则不能直接回切

## 9. 需要明确给下一个对话/IDE 工具的事实

1. 当前“结果最正确”的版本已经存在，不要误以为工程还停留在 `58.x` 的错误状态。
2. `enableWinograd23` 已固定不应开启。
3. `ARGBFloat` 当前不是随意选择，而是为了对齐官方。
4. `MaxPoolingInd` 的 texture 版已经尝试过两种做法，但仍未对齐，当前默认不能切回。
5. 如果继续做 texture `MaxPoolingInd`，必须优先依赖现有的：
   - `maxpool_compare`
   - `conv_compare`
   - Unity batch debug
   否则很容易反复退化到 `58.x`。

