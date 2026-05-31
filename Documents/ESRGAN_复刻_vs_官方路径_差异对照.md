## 目标
对比“ESRGAN(复刻)”与两条“官方/原版”路径（外部进程 realesrgan-ncnn-vulkan.exe、native realesrgan_unity.dll）的实现差异，按差异点给出对应代码位置，便于逐项对齐效果与耗时。

## 路径概览

### 官方路径 A：外部进程（exe）
- Unity 侧负责：输入缩放/编码 PNG → 启动进程 → 读取输出 PNG →（必要时）回缩放。
- 代码入口：[RealEsrganNcnnVulkanRunner.ProcessAsync](file:///e:/Projects/AIImage/Assets/Scripts/RealEsrganNcnnVulkanRunner.cs#L32-L200)

### 官方路径 B：native（realesrgan_unity.dll）
- Unity 侧负责：输入缩放 → 取 RGBA bytes → 调 native →（必要时）回缩放。
- 代码入口：[RealEsrganNcnnNativeRunner.ProcessAsync](file:///e:/Projects/AIImage/Assets/Scripts/RealEsrganNcnnNativeRunner.cs#L59-L213)
- native 侧负责：tile 切分/预处理/推理/裁剪拼接/alpha，核心实现：[realesrgan_unity.cpp](file:///e:/Projects/AIImage/RealESRGAN/Native/realesrgan_unity.cpp#L425-L629)

### 复刻路径：Unity Compute（ESRGAN(复刻)）
- Unity 侧负责：输入缩放 → tile 切分 → pack4 特征图 → 按 param 解释执行算子 → tile 输出拼接 → readback。
- 代码入口：[RealEsrganNcnnReproRunner.ProcessAsync](file:///e:/Projects/AIImage/Assets/Scripts/RealEsrganNcnnReproRunner.cs#L136-L209)
- 关键算子/核函数：[NcnnCompute.compute](file:///e:/Projects/AIImage/Assets/Resources/NcnnCompute.compute#L1-L120)、调度封装：[NcnnOps](file:///e:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnOps.cs#L85-L220)

## 关键差异清单（带代码点）

### 1) 输入来源与颜色空间/归一化链路
- 外部进程：Unity 先 `EncodeToPNG()` 写盘，再由 exe 读 PNG 解码成 RGB，再做归一化/推理。Unity 侧 PNG 写盘位置/输入缩放见：[RealEsrganNcnnVulkanRunner.cs:L114-L162](file:///e:/Projects/AIImage/Assets/Scripts/RealEsrganNcnnVulkanRunner.cs#L114-L162)
- native：Unity 侧传 RGBA bytes，native 侧把 tile RGBA → RGB，并做 `norm=1/255`：见 [realesrgan_unity.cpp:L503-L508](file:///e:/Projects/AIImage/RealESRGAN/Native/realesrgan_unity.cpp#L503-L508)
- 复刻：Unity compute 直接从 `Texture2D<float4> _NcnnIn` 取值写入 pack4（未显式乘 `1/255`，依赖 Unity 纹理取样返回 0..1）：[NcnnCompute.compute:L24-L94](file:///e:/Projects/AIImage/Assets/Resources/NcnnCompute.compute#L24-L94)

对齐建议：
- 若出现“同图不同 tile 内部颜色偏差”，优先验证输入是否在三条路径中处于同一色彩空间（sRGB/Linear）与同一数值域（0..1 或 0..255）。native 明确是 0..1，复刻目前依赖 Unity 的纹理读法。

### 2) tile 策略（默认 tileSize 行为）与 pad/预处理边界
- native：tileSize <= 0 时 native 内部会设默认 tile（实现里固定到 256/并限制 32..512），并使用 `prepadding` 做 tile 输入扩边；随后会在输出端裁掉 `pad*factor`：见 [realesrgan_unity.cpp:L445-L579](file:///e:/Projects/AIImage/RealESRGAN/Native/realesrgan_unity.cpp#L445-L579)
- 复刻：tileSize 默认 128（并按显存大小 AutoTileSize），tilePad 默认 10，tile 输入扩边后进入网络，输出按 `tileOutOriginX/Y` 计算在大图内拼接：[RealEsrganNcnnReproRunner.cs:L251-L295](file:///e:/Projects/AIImage/Assets/Scripts/RealEsrganNcnnReproRunner.cs#L251-L295)
- 外部进程：tileSize 通过 `-t` 参数交给 exe 决定（exe 的“0=auto”语义可能与 native/复刻不一致）：[RealEsrganNcnnVulkanRunner.cs:L166-L178](file:///e:/Projects/AIImage/Assets/Scripts/RealEsrganNcnnVulkanRunner.cs#L166-L178)
- 官方 exe 的 auto tileSize 规则（Real-ESRGAN-ncnn-vulkan）：当 `-t 0` 时按 Vulkan heap budget 做阈值选择 `200/100/64/32`，见 [main.cpp:L776-L795](file:///e:/Projects/AIImage/ref/Real-ESRGAN-ncnn-vulkan/src/main.cpp#L776-L795)
- 官方 RealESRGAN 的 tile 切分与 prepadding：按 `tilesize/prepadding` 做 y 方向分块、构造输入 tile、并复用 blob/staging allocator，见 [realesrgan.cpp:L214-L260](file:///e:/Projects/AIImage/ref/Real-ESRGAN-ncnn-vulkan/src/realesrgan.cpp#L214-L260)

对齐建议：
- 先统一“默认 tileSize=0 的语义”：在三条路径里把最终使用的 tileSize 打印出来（或固定一个 tileSize 做 AB）。
- 对效果敏感时，先固定 `tilePad/prepadding=10`，再做 tileSize 改动；否则“tile 数量变化”会放大误差定位难度。

### 3) 边界采样（clamp/reflect）与 tile 输入构造
- native：tile RGBA→RGB 的边界处理是 clamp（超出原图边界就钳制到边缘像素），见 `rgba_to_rgb_tile_clamp(...)` 调用：[realesrgan_unity.cpp:L503-L505](file:///e:/Projects/AIImage/RealESRGAN/Native/realesrgan_unity.cpp#L503-L505)
- 复刻：tile 输入由 `NcnnPackRgbToPack4` 按 offset 直接从源纹理取像素，边界已改为 clamp：[NcnnCompute.compute:L80-L94](file:///e:/Projects/AIImage/Assets/Resources/NcnnCompute.compute#L80-L94)

### 4) 网络执行方式：原版 ncnn 的“图级执行” vs 复刻的“逐层解释执行”
- 官方（exe/native）：由 ncnn 网络一次性执行（Vulkan 下使用 ncnn 的 shader/pipeline、blob/workspace allocator、light mode 等），复刻无法逐项复刻其“层融合/内存复用/pack/精度”细节。
- 官方（Real-ESRGAN-ncnn-vulkan）额外有 preproc/postproc Vulkan pipeline（local size 32×32×3），并根据 fp16/int8 组合选择不同 spv：见 [realesrgan.cpp:L117-L163](file:///e:/Projects/AIImage/ref/Real-ESRGAN-ncnn-vulkan/src/realesrgan.cpp#L117-L163)
- 复刻：在 C# 里解析 `.param` 并逐层做：
  - `Split/Concat`（Concat 走 `Graphics.CopyTexture` 多 slice copy）
  - `Convolution`（pack4+3x3/1x1 compute kernel）
  - `Eltwise/BinaryOp`（Add）
  - `Interp`（2x/0.5，按 `resize_type` 选择 nearest/bilinear）
  代码见：[RealEsrganNcnnReproRunner.ForwardPack4](file:///e:/Projects/AIImage/Assets/Scripts/RealEsrganNcnnReproRunner.cs#L327-L458)

对齐建议：
- 性能差距大时，优先从“分配/回收/同步”对齐：ncnn Vulkan 用 allocator 池 + light mode，复刻需要尽量复用 RenderTexture2DArray（且要保证 GPU 完成后再复用）。

### 5) 内存分配与同步：ncnn allocator vs Unity RenderTexture 临时分配
- 官方（native）：Vulkan 下为每次调用获取 `blob_allocator/staging_allocator` 并在 scope 内复用（类似池化）：[realesrgan_unity.cpp:L471-L479](file:///e:/Projects/AIImage/RealESRGAN/Native/realesrgan_unity.cpp#L471-L479)
- 复刻：大量 `RenderTexture.GetTemporary(Texture2DArray)`，并提供 Vulkan 下的“带 fence 的安全复用池”（防止未完成就复用，同时降低碎片）：[RealEsrganNcnnReproRunner.cs:L593-L733](file:///e:/Projects/AIImage/Assets/Scripts/RealEsrganNcnnReproRunner.cs#L593-L733)

对齐建议：
- 如果仍出现“相邻块内部差异”：需要排查是否存在“未写清/未同步就读”的 hazard（尤其 Vulkan 下）。复刻侧的复用池如果 fence 不可用，需要退回到“至少跨 2 帧再复用”的策略。

### 6) Vulkan 路径的 CPU↔GPU 往返
- native：当前实现对每个 tile 做多次 `submit_and_wait()`（in upload、网络执行、out download），这会显著拉低耗时，且与 exe 的实现细节可能不同：[realesrgan_unity.cpp:L521-L557](file:///e:/Projects/AIImage/RealESRGAN/Native/realesrgan_unity.cpp#L521-L557)
- 外部进程：是否同样按 tile 往返取决于 exe 的实现；但从耗时看，它很可能有更好的 tile 策略/更少同步点。
- 复刻：在 tile 内完全 GPU 执行，最终只 readback 一次（整张图）：[RealEsrganNcnnReproRunner.cs:L297-L308](file:///e:/Projects/AIImage/Assets/Scripts/RealEsrganNcnnReproRunner.cs#L297-L308)

### 6.1) Interp 的精确定义（影响 tile 边界一致性）
- ncnn 的 Interp：`resize_type=1` 为 nearest，`resize_type=2` 为 bilinear；bilinear 的坐标映射为 `fx = (dx + 0.5) * (w/outw) - 0.5`，并在边界处钳制到 `sx in [0, w-2]`（末端 `fx=1`）：见 [interp.cpp:L56-L90](file:///e:/Projects/AIImage/ref/ncnn-master/src/layer/interp.cpp#L56-L90) 与 [interp.cpp:L514-L555](file:///e:/Projects/AIImage/ref/ncnn-master/src/layer/interp.cpp#L514-L555)
- 复刻：Unity compute 的 bilinear Interp 已按上述边界规则对齐（避免末端 `x0/x1` 都钳到 `w-1` 导致边缘取样差异）：见 [NcnnCompute.compute:L323-L409](file:///e:/Projects/AIImage/Assets/Resources/NcnnCompute.compute#L323-L409)

### 7) alpha 处理差异
- native：会扫描输入 alpha，若存在透明度则额外做 alpha upscale：[realesrgan_unity.cpp:L611-L625](file:///e:/Projects/AIImage/RealESRGAN/Native/realesrgan_unity.cpp#L611-L625)
- 外部进程：取决于 exe；Unity 侧不显式处理 alpha。
- 复刻：目前只处理 RGB（三通道），输出 alpha 固定 1（见 unpack/blit 代码路径）：[NcnnCompute.compute:L96-L104](file:///e:/Projects/AIImage/Assets/Resources/NcnnCompute.compute#L96-L104)

## 当前观测到的耗时差异（示例日志）与可能原因
你提供的同图耗时：
- Real-ESRGAN(exe) ~35s
- Real-ESRGAN(native) ~102s
- ESRGAN(repro) ~145s

高概率原因组合：
1. tile 数量不同：native 默认 tile 更小会导致 tile 数量激增，尤其在每 tile 多次 submit_and_wait 的情况下会指数级放大。
2. 同步点不同：native 每 tile 多次 `submit_and_wait()`；exe 可能减少了同步或使用更优的自动 tile；复刻只有一次 readback，但层级执行导致 GPU dispatch 数量与中间写回更多。
3. 中间张量分配/释放成本：复刻如果不能稳定复用中间 RenderTexture，会导致 Vulkan 分配碎片与峰值抖动，进而变慢甚至 OOM。

## 下一步建议（按收益/成本排序）
1. 固定对比用的 tileSize/tilePad/prepadding：让三条路径在“同 tileSize”下对比耗时与效果，缩小变量。
2. 在复刻里对关键中间 blob 做轻量 hash/统计（非整图 readback）：用于定位“块内差异从哪一层开始产生”。
3. 如果目标是逼近 ncnn Vulkan 性能，需要进一步对齐：
   - pack/precision（fp16 路径）
   - layer fusion（conv+activation 等）
   - 运行时内存复用（allocator 语义）
