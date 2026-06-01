# CLIP ncnn repro 交接与调试记录

## 目标

在 Unity 工程内基于 `clip(ref\\cnn_mobileclip-main)` 和 `ncnn(ref\\ncnn-master)` 复刻一套可用的 CLIP 流程，
输出至少 8 类标签：

- Portrait
- Landscape
- Night
- Food
- Pet
- Architecture
- Document
- Group
- Photo

要求：

- 新写 `ClipNcnnReproRunner`
- 不改旧的 `NcnnRepro / NcnnRepro2 / NcnnRepro3`
- 如有必要新写 `NcnnRepro4`
- 尽量使用 `pack4 RenderTexture`
- 已知 `Winograd23` 在 pack4 纹理卷积里更容易出问题，优先普通 direct conv

## 当前结论

### 1. 当前默认模型级别

当前代码默认使用 `S1`，不是 `S0`。

位置：

- [ClipNcnnReproRunner.cs](/E:/Projects/AIImage/Assets/Scripts/ClipNcnnReproRunner.cs)
  `public ClipModelLevel modelLevel = ClipModelLevel.S1;`

实际验证也全部基于 `S1`：

- `P1010096.JPG`
- `02.png`
- `03.jpg`

所以：

- `S1` 现在是有实际用途的
- 不能直接删
- `S0` 目前保留为可选，但没有重新做完整对齐验证

如果未来真的想删 `S1`，要先用 `S0` 重新跑完一遍对齐和性能回归。

### 2. 当前默认运行策略

为了先把结果完全对齐到官方，当前 `ClipNcnnReproRunner` 图像编码器默认使用“严格对齐官方”的路径：

- `ForceBufferConvolutionAll = true`
- `ForceBufferBinaryOpAll = true`
- `ForceBufferGeluAll = true`
- `EnableDepthWiseTextureConvolution = false`
- `EnableConv1x1TextureConvolution = false`
- `EnableWinograd23 = false`

这意味着：

- 当前结果优先正确性
- 不是最大化 pack4 覆盖率
- 目前是一个“对齐官方优先”的稳定版本

位置：

- [ClipNcnnReproRunner.cs](/E:/Projects/AIImage/Assets/Scripts/ClipNcnnReproRunner.cs)

## 已完成的主要工作

### 新增/修改的核心文件

- [ClipNcnnReproRunner.cs](/E:/Projects/AIImage/Assets/Scripts/ClipNcnnReproRunner.cs)
- [MobileClipSimpleTokenizer.cs](/E:/Projects/AIImage/Assets/Scripts/MobileClipSimpleTokenizer.cs)
- [NcnnRepro4.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro4.cs)
- [NcnnOps.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnOps.cs)
- [NcnnCompute.compute](/E:/Projects/AIImage/Assets/Resources/NcnnCompute.compute)
- [NcnnDebugRunner.cs](/E:/Projects/AIImage/Assets/Editor/NcnnDebugRunner.cs)
- [MainView.cs](/E:/Projects/AIImage/Assets/Scripts/MainView.cs)
- [NcnnBinReader.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnBinReader.cs)

### 关键修复

#### A. 文本侧跑通

- tokenizer 资源已补齐到 `Assets/StreamingAssets/Clip`
- text encoder / projection runner 可稳定输出有限值 embedding

#### B. 图像输入预处理方向修复

问题：

- 官方 `in0[0] = 0.643137`
- Unity 侧一度得到 `0.0588`
- 后来确认是 `Y` 方向翻转了

原因：

- `PackRgbToPack4` 默认没有 `flipY`
- Unity `Graphics.Blit` 后读取方向和官方 `stbi_load + from_pixels_resize` 不一致

修复：

- `NcnnPackRgbToPack4` shader 加入 `_FlipY`
- `NcnnOps.PackRgbToPack4(..., flipY)`
- CLIP 输入调用时显式传 `flipY = true`

相关文件：

- [NcnnCompute.compute](/E:/Projects/AIImage/Assets/Resources/NcnnCompute.compute)
- [NcnnOps.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnOps.cs)
- [ClipNcnnReproRunner.cs](/E:/Projects/AIImage/Assets/Scripts/ClipNcnnReproRunner.cs)

#### C. ncnn raw float 权重读取错位修复

这是本次最关键的根因之一。

现象：

- `blob in0` 已经对齐官方
- 但 `blob 1` 从第一层 `conv_12` 开始就明显跑偏
- 我用独立 CPU 参考把 `conv_12` 单独算了一遍，发现只要按我们旧的权重读取方式读，结果就会稳定复现错误

真正原因：

- ncnn 的 raw float 权重块前面有一个 4-byte flag
- 对于 raw float，这个 flag 可能就是 `0`
- 旧代码把这个 `0` 当成了第一个权重值
- 整段 weight 全部错位 1 个 float

修复点：

- [NcnnBinReader.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnBinReader.cs)
  `ReadNcnnArrayAsFloat32()` raw-float 分支改成严格按官方 `ModelBin::load()` 语义读
- [NcnnRepro4.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro4.cs)
  `ReadClipArrayAsFloat32()` 改成直接走 `ReadNcnnMatAsFloat32()`

修复后验证：

- Unity `blob 1`：
  `-16.1711, -1.3372, -1.1996 ...`
- 官方 `blob 1`：
  `-16.1752, -1.37855, -1.20681 ...`

已经进入非常接近的范围。

#### D. Debug 和 blob dump 系统

为了定位问题，补了很多调试能力：

- 指定 blob dump
- texture/blob 任意导出
- 图像 `in0` 直接导出
- 卷积 compare 日志
- source / weight / bias finite/nan 统计

这部分主要在：

- [ClipNcnnReproRunner.cs](/E:/Projects/AIImage/Assets/Scripts/ClipNcnnReproRunner.cs)
- [NcnnRepro4.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro4.cs)

## MainView 集成

已在 [MainView.cs](/E:/Projects/AIImage/Assets/Scripts/MainView.cs) 增加按钮：

- `CLIP标签`

行为：

- 读取当前选中的历史图，如果没有则回退到原图
- 调用 `ClipNcnnReproRunner.ProcessAsync()`
- 通过 toast 显示 top1 / top3
- 在 Console 打印完整结果

当前不会改图，只做分类。

## Editor 批量静默调试

已在 [NcnnDebugRunner.cs](/E:/Projects/AIImage/Assets/Editor/NcnnDebugRunner.cs) 增加：

- 菜单：
  `Tools/AIImage/Run CLIP Directory Debug`
- batch 入口：
  `NcnnDebugRunner.RunClipDirectoryDebugBatch`

### 用法

设置环境变量：

- `AIIMAGE_CLIP_INPUT_DIR`
  指向要递归跑的目录
- `AIIMAGE_CLIP_MODEL`
  可选，`S0` 或 `S1`
- `AIIMAGE_CLIP_ENABLE_DUMP`
  可选，`1/true` 时为每张图打开 dump

示例：

```powershell
$env:AIIMAGE_CLIP_INPUT_DIR='E:\photos\2008-1-9广州南沙天后宫'
$env:AIIMAGE_CLIP_MODEL='S1'
"C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe" `
  -batchmode `
  -projectPath E:\Projects\AIImage `
  -executeMethod NcnnDebugRunner.RunClipDirectoryDebugBatch `
  -logFile E:\Projects\AIImage\Logs\clip_dir_batch.log
```

输出：

- `Logs\AIImage_ClipDirBatch-*`
- `summary.tsv`
- `gpu_resource_stats.txt`

`summary.tsv` 包含：

- image
- status
- elapsed_ms
- best_label
- best_prob
- top3
- gpu_summary
- rt_count
- managed_mb
- gfx_driver_mb
- dump

这能直接拿来做：

- 性能对比
- 长目录批跑稳定性检查
- `RenderTexture` / `ComputeBuffer` 泄漏趋势排查

## 当前结果对齐情况

### P1010096.JPG

Unity 当前：

- Landscape `0.762628`
- Photo `0.231002`
- Portrait `0.000590`

官方参考：

- Landscape `0.757469`
- Photo `0.236261`
- Portrait `0.000581604`

已经非常接近，并且 top1 正确。

### 02.png

Unity 当前 top3：

- Photo `0.518233`
- Group `0.459419`
- Night `0.015329`

官方参考 top3：

- Photo `0.521865`
- Group `0.455433`
- Night `0.0158496`

### 03.jpg

Unity 当前 top3：

- Photo `0.866485`
- Portrait `0.090744`
- Document `0.030716`

官方参考 top3：

- Photo `0.87546`
- Portrait `0.085979`
- Document `0.0281151`

## pack4 rendertexture 没能继续使用的原因

这里分“已经确认的问题”和“还没精确到 shader 行级别的问题”。

### 1. depthwise pack4 texture conv

现象：

- 在 pack4 主路径下，最早的失败点会回到非常早的 `ConvolutionDepthWise` 层
- 例如早期 block 的 `convdw_226 / convdw_227` 等
- source finite，但 texture 路径输出会直接变 NaN

结论：

- 这是图像编码器里最不稳定的一类 pack4 texture 算子
- 不是模型权重读取问题
- 在修复权重读取和 flipY 后仍然存在

当前处理：

- 禁用图像侧 `depthwise texture conv`
- 回退到 reference/buffer 路径

### 2. GELU pack4 texture

现象：

- pack4 主路径下，`blob 1` 已经 finite 且接近官方
- 但一到 `gelu_103` 对应 `blob 2` 就开始出现 NaN

结论：

- GELU 的 texture path 仍然不稳定
- 问题已经独立于输入和权重读取

当前处理：

- 图像侧 `GELU` 固定走 buffer/CPU 精确实现

### 3. 1x1 pack4 texture conv

现象：

- 单独保留 `1x1 texture conv` 时，前半段不一定立刻 NaN
- 但会在后半段累计出明显数值漂移
- 最终在 `bn_8 / gemm_0 / attention` 这一带把值推爆或者推到 NaN

结论：

- `1x1 texture conv` 当前虽然不像 depthwise 那样“很早就全 NaN”
- 但数值一致性不够，累积误差大

当前处理：

- 图像侧 `1x1 texture conv` 禁用
- 回退到 reference/buffer 路径

### 4. 还能继续用的 pack4

目前实际保留得比较稳的是：

- 输入 pack4
- 一部分普通 texture 流程
- 其它非关键 texture helper

但为了“完全对齐官方”，当前默认把图像侧主干算子切到了更保守的 reference 路径。

## 遗留问题

### A. 结果已对齐，但 pack4 覆盖率没有恢复到理想状态

当前版本优先“结果正确”，不是“pack4 覆盖最大”。

后续若要继续优化 pack4 覆盖率，建议顺序：

1. 先单独修 `GeluPack4`
2. 再单独修 `ConvDepthWisePack4`
3. 最后再收 `Conv1x1Pack4`

### B. 还没重新验证 S0

当前对齐验证全是 `S1`。

如果想减包体，可以后续：

1. 先把 `ClipNcnnReproRunner.modelLevel` 改成 `S0`
2. 跑 `P1010096.JPG / 02.png / 03.jpg`
3. 再跑目录批量
4. 如果分布和速度都满意，再删 `S1`

### C. 图像侧严格官方路径目前偏 buffer/reference

当前代码更像：

- “正确版”

而不是：

- “最高 pack4 利用率版”

## 推荐给下一个对话的起点

如果下一个对话要继续做 pack4 优化，不要从头重查。

建议直接从这里开始：

1. 保持当前默认配置不动，先保证结果回归不退化
2. 新开一个开关，只单独恢复 `GeluPack4`
3. 用 `P1010096.JPG` 对比 `blob 1 -> blob 2`
4. 如果 `blob 2` 仍然 NaN，就只修 `NcnnGeluPack4`
5. 然后再单独恢复 `depthwise texture conv`
6. 最后恢复 `1x1 texture conv`

重点参考文件：

- [ClipNcnnReproRunner.cs](/E:/Projects/AIImage/Assets/Scripts/ClipNcnnReproRunner.cs)
- [NcnnRepro4.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro4.cs)
- [NcnnOps.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnOps.cs)
- [NcnnCompute.compute](/E:/Projects/AIImage/Assets/Resources/NcnnCompute.compute)
- [NcnnDebugRunner.cs](/E:/Projects/AIImage/Assets/Editor/NcnnDebugRunner.cs)

## 本次最终推荐状态

如果目标是“先交付一个完全可用、与官方结果足够接近的版本”，当前状态已经可用：

- `S1` 保留
- `MainView` 可直接调用
- `Editor` 可整目录静默批跑
- `P1010096.JPG` 已稳定回到 `Landscape`
- `02.png / 03.jpg` top1/top3 已贴近官方

