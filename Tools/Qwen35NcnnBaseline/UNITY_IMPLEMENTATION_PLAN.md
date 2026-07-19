# Unity C#/ComputeShader 完整实现方案

## 结论

最终路径使用镜像中已转换的 ncnn param/bin，不在移动端加载 GGUF，也不接入 llama.cpp。Unsloth
GGUF 可以作为另一条离线导出来源，但它与当前自研 ncnn 图执行器、PDF 的 moduleop/cache 契约
不一致，而且指定镜像没有提供 GGUF，因此不作为本方案运行时格式。

外部金标 CLI 的实测峰值约为 5.8 GiB private memory，模型文件为 3.18 GiB。这个 FP32/现有存储
版本适合桌面正确性阶段，不能直接宣称适配普通 Android/iOS 设备。移动发布前必须完成权重共享、
量化和资产分片，否则即使算子正确也不满足端侧可用性。

## 模型契约

- Decoder: 869 layers / 1181 blobs。
- 普通输入: `in0` hidden、`in1` mask、`in2/in3` RoPE cos/sin。
- Full Attention: 6 组 `cache_kN/cache_vN` 及对应输出。
- Linear Attention: 18 组 `cache_convN`、18 组 `cache_gdrN` 及对应输出。
- 拓扑: `ShortConv + GatedDeltaRule` 三组后接一个 cache SDPA，共重复六次。
- Vision: patch 16、patch dim 768、spatial merge 2、最多 49152 patches。
- LM Head 和 token embedding 共享 `qwen3.5_embed_token.ncnn.bin`，C# 加载器必须只解析/持有一份。

## Unity 模块

在 `Packages/com.aiimage.inference.unitygpu` 与 kernels package 中实现：

1. `NcnnLayerTypes` 增加 `ShortConv`、`GatedDeltaRule`，factory 注册对应 C# layer repro。
2. `NcnnShortConvLayerRepro.cs` 和 `NcnnGatedDeltaRuleLayerRepro.cs` 只实现 RenderTexture 与
   CommandBuffer 路径；严格模式下没有 texture path 就直接报错。
3. `NcnnCompute.compute` 增加 FP32 ShortConv、GDR decay/read/update/project kernel。
4. `Qwen35ModelContract.cs` 校验六个网络、共享权重别名和 52 个 decoder 输入。
5. `Qwen35ByteLevelBpeTokenizer.cs` 对齐 `qwen35_tokenizer.py` 的 byte map、最长特殊 token 匹配、
   merges 和 UTF-8 decode。
6. `Qwen35VisionPreprocessor.cs` 对齐 255.5 分母、RGB 平面、temporal duplicate、2x2 merge 重排
   和二维 vision RoPE。
7. `Qwen35Runner.cs` 负责 system/user chat template、视觉 embedding 注入、prefill/decode、greedy
   采样和 cache 生命周期；不通过原生 DLL 或 P/Invoke 调用参考 CLI。

## 纹理布局

所有 activation、临时量和 cache 使用 texture-backed `ComputeTexture`，禁止正常推理期间创建临时
ComputeBuffer 或通过 buffer materialization 绕过缺失算子。

| 逻辑张量 | 纹理布局 |
| --- | --- |
| Hidden `[seq, dim]` | RGBAFloat 2D，width=`ceil(dim/4)`，height=`seq` |
| Q/K/V `[seq, head, dim]` | RGBAFloat 2DArray，width=`ceil(dim/4)`，height=`seq`，slice=`head` |
| GDR state `[head, kdim, vdim]` | RGBAFloat 2DArray，width=`ceil(vdim/4)`，height=`kdim`，slice=`head` |
| ShortConv state `[kernel, groups]` | RGBAFloat 2D，width=`ceil(groups/4)`，height=`kernel` |
| KV cache `[head, capacity, dim]` | RGBAFloat 2DArray，width=`ceil(dim/4)`，height=`capacity`，slice=`head` |

GDR recurrence 有 token 依赖。prefill 先以每 token 有序 dispatch 实现正确性，state 用 ping-pong RT；
decode 为单 token dispatch。核心 L2Norm、decay、outer-product update 和 projection 全部 FP32，禁止
half 累加。后续优化只能在逐层金标通过后合并 kernel。

不可变权重可以作为固定常量资源加载，但共享 embedding/head 不能复制。移动量化产物应按层/词表
tile 分片，避免单张纹理高度超过设备上限；activation/cache 仍保持 texture-only。

## 跨平台发布

- Windows/macOS: 先完成完整 FP32 正确性和逐层 dump。
- Android: Unity Vulkan Compute，模型使用 Play Asset Delivery 或首次启动下载到
  `persistentDataPath`，不能把 3.18 GiB 直接塞入基础 APK。
- iOS: Unity Metal Compute，使用 On-Demand Resources/远程资源包；不要依赖 Android
  `StreamingAssets` 的文件访问语义。
- 两端都要查询 RGBAFloat、Texture2DArray、最大纹理尺寸和 slice 数；不满足即明确判定设备不支持，
  不能静默回退 ComputeBuffer。

移动目标产物应从山东大学镜像原始权重离线派生：优先 int8 weight-only，进一步评估 int4 groupwise。
建议首版包体目标小于 1.5 GiB、运行峰值小于 3 GiB；达不到时应缩小词表/模型或提升最低设备档位，
而不是隐藏内存风险。

## 验收顺序

1. Python 10 项 NumPy 测试必须通过。
2. 外部图片 CLI 报告 `valid=true`，四个固定 OCR 标记命中。
3. 按 `qwen35_0_8b_compare_manifest.json` 对六个网络逐层比较逻辑解包 FP32 值。
4. `ShortConv/GDR` 使用 `atol=rtol=2e-5`；norm/SDPA/GDR reduction 上限
   `atol=3e-4, rtol=5e-4`；Gemm/Conv 上限 `atol=5e-4, rtol=8e-4`。
5. Prefill 一次执行与逐 token decode 的 hidden/cache 结果一致。
6. 最终 logits top-1 相同、cosine similarity 至少 0.99999，OCR 四个标记仍命中。
7. 打开 `DisallowInferenceTempComputeBuffers`、`DisallowBufferToTextureMaterialization` 等严格开关，
   检查正常推理没有 `GetBufferData`、`Pack4ToBuffer*` 或临时 ComputeBuffer fallback。
8. `dotnet build AIImage.sln -v minimal` 通过，再分别做 Android Vulkan 与 iOS Metal 真机验证。

逐层比对阶段允许显式 debug readback；readback 代码必须由调试开关保护，不能进入正常推理路径。

