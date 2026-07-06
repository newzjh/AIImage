# ncnn Vulkan Layer 复刻覆盖统计

更新时间：2026-07-06

## 统计口径

- 官方基线目录：`ref/ncnn-master/src/layer/vulkan`
- 当前复刻入口目录：`Assets/Scripts/NcnnLayers`
- 当前注册入口来源：`Assets/Scripts/NcnnLayers/NcnnLayerFactoryRepro.cs`
- 本文的“已覆盖”含义：
  - 有对应 `LayerRepro` 入口
  - 已进入 `NcnnLayerFactoryRepro` 注册表
  - 当前 `Assembly-CSharp.csproj` 可编译
- 本文的“实现状态”口径：
  - `Buffer(legacy)` 只表示当前代码里仍然存在的兼容/真值路径，不代表长期目标
  - `RT` 表示 `ExecuteRenderTexturePath` 的 pack4 `RenderTexture` 路径
  - `Cmd` 表示 `ExecuteCommandBuffer` 的 pack4 `ComputeTexture` 路径
- 本文状态标签：
  - `legacy完整`：仍有完整 compute-buffer/CPU 兼容实现，但这条路已是遗留路径
  - `legacy别名`：buffer 侧只做 alias / shape reinterpretation，不承担真实算子计算
  - `真实`：该路径已有真实 pack4 shader / texture 计算
  - `部分`：部分 shape / 特化分支走真实 pack4，其他情况仍回退到 legacy 或 placeholder
  - `别名`：只透传纹理/shape/contract，不做真实算子计算
  - `材质化`：由 buffer 结果或常量直接材质化为纹理，不是该层自己的真实 RT shader
  - `占位`：只发布 shape-correct placeholder，保证链路可编译/可串接
  - `无`：该路径当前没有独立实现，实际会回到别的路径

## 最新结论

- 官方 Vulkan layer 数量仍为 `64`
- 当前工程对官方 `64/64` 全部有复刻入口，覆盖率仍为 `100%`
- 当前 `NcnnLayerFactoryRepro` 注册入口数为 `77`
- 当前 `Ncnn*LayerRepro.cs` 实现文件数为 `68`
- 相比官方 Vulkan 口径，仓内额外扩展了 `13` 个入口：
  - `Input`
  - `pnnx.Expression`
  - `aten::to`
  - `Convolution3D`
  - `Deconvolution3D`
  - `Pooling3D`
  - `Embed`
  - `ExpandDims`
  - `MatMul`
  - `MaxPoolingInd`
  - `MaxUnPooling`
  - `Squeeze`
  - `Tile`

## 这轮需要特别记录的变化

- `Reshape / ExpandDims / Flatten / Squeeze` 的 shape-changing alias 路径，已经从“只改 `textureShapes[name]`”收紧成“真正统一契约生效”：
  - 每个 blob 会拿到自己的 wrapper
  - wrapper 里同时携带 `logicalShape + storageShape + layoutKind`
  - 生命周期通过共享 owner 串起来
  - 下游已经开始按这个 contract 消费，而不是只看名字旁边的 shape 侧表
- 当前这套统一契约的关键实现位于：
  - `Assets/Scripts/NcnnCompute/NcnnRepro.cs` 里的 `RepoVkTensorContract`
  - `CreateTextureAlias / CreateCmdTensorAlias`
  - `GetTextureContract / GetCmdTensorContract`
  - `TryGetExistingTextureContract`
- 结果上，alias-compatible 的 `reshape` 现在已经能把 `storageShape` 往下游传递，避免不必要的物理重排；这点已经重新用真实 runner 跑过 CLIP 和 Matting 路径验证。

## 代码与验证基线

- 当前轮已通过：`dotnet build Assembly-CSharp.csproj -v minimal`
- 当前轮已复跑真实静默 runner：
  - CLIP 目录批处理：`Documents/ClipCompareInput`
  - Matting 单图批处理：`ref/ncnn_matting-main/test_img.jpg`
- 已知结果保持一致：
  - CLIP 三张样例的 `best_label / best_prob / top3` 与前一轮稳定结果一致
  - Matting 仍为 `mean_abs_rgb=2.9897`、`max_abs_rgb=204`

## 官方 Vulkan 64 层当前状态

| 官方 Vulkan layer | 当前入口 | Buffer(legacy) | RT | Cmd | 备注 |
| --- | --- | --- | --- | --- | --- |
| `absval_vulkan` | `NcnnUnaryOpAliasLayerRepro(AbsVal)` | legacy完整 | 真实 | 真实 | 走共享 `UnaryOpPack4` 路径 |
| `batchnorm_vulkan` | `NcnnBatchNormLayerRepro` | legacy完整 | 真实 | 真实 | BatchNorm pack4 路径已接通 |
| `binaryop_vulkan` | `NcnnBinaryOpLayerRepro` | legacy完整 | 部分 | 部分 | 标量、同 shape、常见广播主路径已是真实 pack4；其余仍有 fallback / placeholder |
| `cast_vulkan` | `NcnnCastLayerRepro` | legacy完整 | 无 | 占位 | same-type 或部分 dtype bridge 可 Noop；通用 cmd 仍是 placeholder |
| `celu_vulkan` | `NcnnPointwiseFormulaLayerRepro(CELU)` | legacy完整 | 真实 | 真实 | 共享 pointwise 公式层 |
| `clip_vulkan` | `NcnnClipLayerRepro` | legacy完整 | 真实 | 真实 | Clip pack4 shader 已接通 |
| `concat_vulkan` | `NcnnConcatLayerRepro` | legacy完整 | 部分 | 部分 | 常见 pack4 concat 已接通；其余仍回 legacy / placeholder |
| `convolution_vulkan` | `NcnnConvolutionLayerRepro` | legacy完整 | 部分 | 部分 | `1x1 / 3x3 / Winograd / 常见 pack4` 已接通，其余仍有 fallback |
| `convolution1d_vulkan` | `NcnnConvolution1DLayerRepro` | legacy完整 | 无 | 占位 | 当前只有 legacy 真值路径 |
| `convolutiondepthwise_vulkan` | `NcnnConvolutionDepthWiseLayerRepro` | legacy完整 | 部分 | 部分 | 常见 pack4 / depthwise 特化已接通 |
| `crop_vulkan` | `NcnnCropLayerRepro` | legacy完整 | 部分 | 部分 | 支持常见 pack4 crop；identity 分支会 alias |
| `deconvolution_vulkan` | `NcnnDeconvolutionLayerRepro` | legacy完整 | 无 | 占位 | 目前仍以 legacy 真值为主 |
| `deconvolutiondepthwise_vulkan` | `NcnnDeconvolutionDepthWiseLayerRepro` | legacy完整 | 无 | 占位 | 目前仍以 legacy 真值为主 |
| `deepcopy_vulkan` | `NcnnDeepCopyLayerRepro` | legacy完整 | 真实 | 真实 | `CopyPack4 / cmd CopyPack4` 可用 |
| `dequantize_vulkan` | `NcnnDequantizeLayerRepro` | legacy完整 | 无 | 占位 | 当前只有 legacy 真值路径 |
| `dropout_vulkan` | `NcnnDropoutLayerRepro` | legacy完整 | 真实 | 真实 | `scale != 1` 时走 pointwise pack4 |
| `eltwise_vulkan` | `NcnnEltwiseLayerRepro` | legacy完整 | 真实 | 真实 | 多输入累积 pack4 已接通 |
| `elu_vulkan` | `NcnnPointwiseFormulaLayerRepro(ELU)` | legacy完整 | 真实 | 真实 | 共享 pointwise 公式层 |
| `erf_vulkan` | `NcnnPointwiseFormulaLayerRepro(Erf)` | legacy完整 | 真实 | 真实 | 共享 pointwise 公式层 |
| `flatten_vulkan` | `NcnnFlattenLayerRepro` | legacy完整 | 无 | 别名 | RT 仍回 legacy；cmd 现在是统一契约 alias |
| `gelu_vulkan` | `NcnnGeluLayerRepro` | legacy完整 | 真实 | 真实 | Gelu pack4 shader 已接通 |
| `gemm_vulkan` | `NcnnGemmLayerRepro` | legacy完整 | 部分 | 部分 | 部分 pack4 纹理/GEMM 特化已接通；通用 cmd 仍可回 placeholder |
| `groupnorm_vulkan` | `NcnnGroupNormLayerRepro` | legacy完整 | 部分 | 部分 | `affine + supported pack4` 可走真实路径，否则仍回 legacy 或 like-input |
| `hardsigmoid_vulkan` | `NcnnPointwiseFormulaLayerRepro(HardSigmoid)` | legacy完整 | 真实 | 真实 | 共享 pointwise 公式层 |
| `hardswish_vulkan` | `NcnnPointwiseFormulaLayerRepro(HardSwish)` | legacy完整 | 真实 | 真实 | 共享 pointwise 公式层 |
| `innerproduct_vulkan` | `NcnnInnerProductLayerRepro` | legacy完整 | 部分 | 部分 | 标量 1D/2D texture A 路径已接通，其余仍回 legacy / placeholder |
| `instancenorm_vulkan` | `NcnnInstanceNormLayerRepro` | legacy完整 | 部分 | 部分 | 依赖 `GroupNormPack4Tex` 的可支持分支已接通 |
| `interp_vulkan` | `NcnnInterpLayerRepro` | legacy完整 | 部分 | 部分 | `2x / down2 / 常见 linear` 已接通，其余仍有 fallback |
| `layernorm_vulkan` | `NcnnLayerNormLayerRepro` | legacy完整 | 无 | 占位 | 当前只有 legacy 真值路径 |
| `lrn_vulkan` | `NcnnLRNLayerRepro` | legacy完整 | 无 | 占位 | 当前只有 legacy 真值路径 |
| `memorydata_vulkan` | `NcnnMemoryDataLayerRepro` | legacy完整 | 材质化 | 材质化 | 常量 buffer 可直接材质化为纹理；不是该层自己的真实 RT shader |
| `mish_vulkan` | `NcnnPointwiseFormulaLayerRepro(Mish)` | legacy完整 | 真实 | 真实 | 共享 pointwise 公式层 |
| `multiheadattention_vulkan` | `NcnnMultiHeadAttentionLayerRepro` | legacy完整 | 部分 | 部分 | 无 mask / 无 kv-cache 的 pack4 特化已接通；`kv_cache` 仍未实现 |
| `noop_vulkan` | `NcnnNoopLayerRepro` | legacy别名 | 别名 | 别名 | 纯 passthrough；已接入统一契约 |
| `normalize_vulkan` | `NcnnNormalizeLayerRepro` | legacy完整 | 无 | 占位 | 当前只有 legacy 真值路径 |
| `packing_vulkan` | `NcnnPackingLayerRepro` | legacy完整 | 无 | 占位 | 当前只有 legacy 真值路径 |
| `padding_vulkan` | `NcnnPaddingLayerRepro` | legacy完整 | 部分 | 部分 | 常见 2D/3D pack4 padding 已接通，其余回 legacy |
| `permute_vulkan` | `NcnnPermuteLayerRepro` | legacy完整 | 部分 | 部分 | 常见 `dims=3/4` pack4 已接通；identity 分支会 alias |
| `pixelshuffle_vulkan` | `NcnnPixelShuffleLayerRepro` | legacy完整 | 真实 | 部分 | RT 已是真实 pack4；cmd 在不满足条件时仍回 placeholder |
| `pooling_vulkan` | `NcnnPoolingLayerRepro` | legacy完整 | 部分 | 部分 | 常见 2D / global / adaptive 分支已接通，其余回 legacy |
| `prelu_vulkan` | `NcnnPReLULayerRepro` | legacy完整 | 部分 | 部分 | 常见 pack4 slope 路径已接通 |
| `priorbox_vulkan` | `NcnnPriorBoxLayerRepro` | legacy完整 | 无 | 占位 | 当前只有 legacy 真值路径 |
| `quantize_vulkan` | `NcnnQuantizeLayerRepro` | legacy完整 | 无 | 占位 | 当前只有 legacy 真值路径 |
| `reduction_vulkan` | `NcnnReductionLayerRepro` | legacy完整 | 部分 | 部分 | 标量 2D 与常见空间 reduction pack4 已接通，通用 cmd 仍有 placeholder |
| `relu_vulkan` | `NcnnReLULayerRepro` | legacy完整 | 真实 | 真实 | `LeakyReluPack4` 路径已接通 |
| `reorg_vulkan` | `NcnnReorgLayerRepro` | legacy完整 | 真实 | 部分 | RT 已是真实 pack4；cmd 在不满足条件时仍回 placeholder |
| `requantize_vulkan` | `NcnnRequantizeLayerRepro` | legacy完整 | 无 | 占位 | 当前只有 legacy 真值路径 |
| `reshape_vulkan` | `NcnnReshapeLayerRepro` | legacy完整 | 部分 | 部分 | 既有真实 pack4 reshape/window/attention 特化，也有基于统一契约的 alias-compatible 零重排分支 |
| `rmsnorm_vulkan` | `NcnnRMSNormLayerRepro` | legacy完整 | 无 | 占位 | 当前只有 legacy 真值路径 |
| `rotaryembed_vulkan` | `NcnnRotaryEmbedLayerRepro` | legacy完整 | 无 | 别名 | RT 仍回 legacy；cmd 当前只做 passthrough/alias |
| `scale_vulkan` | `NcnnScaleLayerRepro` | legacy完整 | 部分 | 部分 | 常见静态标量 scale 已接通，其余回 legacy / placeholder |
| `sdpa_vulkan` | `NcnnSdpaLayerRepro` | legacy完整 | 部分 | 部分 | 无 mask / 无 kv-cache / 无 int8-scale 的 pack4 特化已接通，其余仍回 legacy / placeholder |
| `selu_vulkan` | `NcnnPointwiseFormulaLayerRepro(SELU)` | legacy完整 | 真实 | 真实 | 共享 pointwise 公式层 |
| `shrink_vulkan` | `NcnnPointwiseFormulaLayerRepro(Shrink)` | legacy完整 | 真实 | 真实 | 共享 pointwise 公式层 |
| `shufflechannel_vulkan` | `NcnnShuffleChannelLayerRepro` | legacy完整 | 真实 | 部分 | RT 已是真实 pack4；cmd 在不满足条件时仍回 copy / placeholder |
| `sigmoid_vulkan` | `NcnnSigmoidLayerRepro` | legacy完整 | 真实 | 真实 | Sigmoid pack4 shader 已接通 |
| `slice_vulkan` | `NcnnSliceLayerRepro` | legacy完整 | 真实 | 部分 | RT 已有真实 pack4 slice；cmd 在不满足条件时仍回 placeholder；identity 分支会 alias |
| `softmax_vulkan` | `NcnnSoftmaxLayerRepro` | legacy完整 | 部分 | 部分 | channel、scalar2D、CDHW 的常见 pack4 分支已接通 |
| `softplus_vulkan` | `NcnnPointwiseFormulaLayerRepro(Softplus)` | legacy完整 | 真实 | 真实 | 共享 pointwise 公式层 |
| `split_vulkan` | `NcnnSplitLayerRepro` | legacy别名 | 别名 | 别名 | 纯分发引用；已接入统一契约 |
| `swish_vulkan` | `NcnnSwishLayerRepro` | legacy完整 | 真实 | 真实 | Swish pack4 shader 已接通 |
| `tanh_vulkan` | `NcnnUnaryOpAliasLayerRepro(TanH)` | legacy完整 | 真实 | 真实 | 走共享 `UnaryOpPack4` 路径 |
| `unaryop_vulkan` | `NcnnUnaryOpLayerRepro` | legacy完整 | 真实 | 真实 | UnaryOp pack4 shader 已接通 |
| `unfold_vulkan` | `NcnnUnfoldLayerRepro` | legacy完整 | 无 | 占位 | 当前只有 legacy 真值路径 |

## 仓内额外 13 层当前状态

| 仓内扩展层 | 当前入口 | Buffer(legacy) | RT | Cmd | 备注 |
| --- | --- | --- | --- | --- | --- |
| `Input` | `NcnnInputLayerRepro` | 无 | 无 | 无 | 输入层本身不执行 |
| `pnnx.Expression` | `NcnnPnnxExpressionLayerRepro` | legacy完整 | 真实 | 真实 | 只支持常量表达式；动态表达式仍不支持 |
| `aten::to` | `NcnnAtenToLayerRepro` | legacy别名 | 别名 | 别名 | 当前仅按 dtype-preserving bridge 处理，内部复用 `Noop` |
| `Convolution3D` | `NcnnConvolution3DLayerRepro` | legacy完整 | 部分 | 部分 | `dims=4` 的 pack4 CDHW 路径已接通 |
| `Deconvolution3D` | `NcnnDeconvolution3DLayerRepro` | legacy完整 | 部分 | 部分 | `dims=4` 的 pack4 CDHW 路径已接通 |
| `Pooling3D` | `NcnnPooling3DLayerRepro` | legacy完整 | 部分 | 部分 | 3D pack4 pooling 已有真实路径，其余仍可回 legacy |
| `Embed` | `NcnnEmbedLayerRepro` | legacy完整 | 无 | 占位 | 当前只有 legacy 真值路径 |
| `ExpandDims` | `NcnnExpandDimsLayerRepro` | legacy别名 | 别名 | 别名 | 形状变化 alias 已接入统一契约 |
| `MatMul` | `NcnnMatMulLayerRepro` | legacy完整 | 部分 | 部分 | attention / VISTA 相关 pack4 特化已接通，通用 cmd 仍可能 placeholder |
| `MaxPoolingInd` | `NcnnMaxPoolingIndLayerRepro` | legacy完整 | 真实 | 真实 | 值输出与 index 输出都可走纹理路径 |
| `MaxUnPooling` | `NcnnMaxUnPoolingLayerRepro` | legacy完整 | 部分 | 部分 | 主路径可用，但仍保留 CPU 辅助 / fallback |
| `Squeeze` | `NcnnSqueezeLayerRepro` | legacy完整 | 无 | 别名 | cmd 当前是统一契约 alias |
| `Tile` | `NcnnTileLayerRepro` | legacy完整 | 别名 | 占位 | `tiles<=1` 仅 alias；真正 tile 仍无 RT/cmd 真实路径 |

## Sentis 对比：Sentis 有，而当前复刻 ncnn 还缺的

对比来源：Unity 官方 `Sentis / AI Inference 2.6.1` 的 Supported ONNX operators 页面  
链接：<https://docs.unity.cn/Packages/com.unity.ai.inference@2.6/manual/supported-operators.html>

对比时做了两点约束：

- 已经能明确映射到现有通用层的，不再重复算“缺口”
  - 例如 `Conv / ConvTranspose / Gemm / MatMul / Relu / Softmax / Slice / Tile / Unsqueeze / Squeeze / Transpose / Pad / Reduce* / UnaryOp / BinaryOp`
- `pnnx.Expression` 与 `MemoryData` 只算“项目化常量输入能力”
  - 不算通用 ONNX `Constant / ConstantOfShape / Shape / Size / Range` 全覆盖

### 高优先级缺口

| 类别 | Sentis 已支持，但当前 ncnn 复刻仍缺 | 备注 |
| --- | --- | --- |
| Shape / 元信息 | `Shape`, `Size`, `Range`, `ConstantOfShape`, `Expand` | 当前没有通用 shape tensor 流与按 shape 构造张量的层 |
| 索引 / 选择 | `ArgMax`, `ArgMin`, `Where`, `TopK`, `NonZero`, `OneHot`, `CumSum`, `Compress` | 这些在当前 `NcnnLayerTypes` 里都没有直接入口 |
| Gather / Scatter | `Gather`, `GatherElements`, `GatherND`, `ScatterElements`, `ScatterND`, `Scatter` | 当前没有通用 gather/scatter 家族 |
| 采样 / 检测 | `GridSample`, `RoiAlign`, `NonMaxSuppression` | 这类对视觉模型接入价值高，但当前没有对应层 |
| 序列 / 采样随机 | `LSTM`, `Bernoulli`, `Multinomial`, `RandomNormal`, `RandomNormalLike`, `RandomUniform`, `RandomUniformLike` | 当前完全缺层 |
| 频域 / 音频 | `DFT`, `STFT`, `MelWeightMatrix`, `BlackmanWindow`, `HammingWindow`, `HannWindow` | 当前完全缺层 |

### Sentis-only 优化层缺口

这些不是 ONNX 原始算子本名，而是 Sentis 优化模型时可能生成的内部层；当前 ncnn 复刻也没有对应入口：

- `BroadcastArgs`
- `MoveDim`
- `Narrow`
- `Select`
- `SliceSet`
- `Atan2`
- `NotEqual`
- `RandomChoice`
- `Square`
- `FloorDiv`
- `TrueDiv`
- `Relu6`
- `ScaleBias`
- `Dense`
- `MatMul2D`
- `DequantizeUint8`

### 不是缺口、但容易误判的项

下面这些在 Sentis 支持表里能看到，但当前工程已经有等价或近似覆盖，所以不计入“当前缺口”：

- `Conv / ConvTranspose`
  - 已对应到 `Convolution / Convolution1D / Convolution3D / Deconvolution / Deconvolution3D`
- `Unsqueeze / Squeeze / Transpose / Slice / Tile / Reshape`
  - 已对应到 `ExpandDims / Squeeze / Permute / Slice / Tile / Reshape`
- `DepthToSpace`
  - 已对应到 `PixelShuffle`
- `Pad`
  - 已对应到 `Padding`
- `Reduce*`
  - 已由 `Reduction` 覆盖一部分，虽然 RT/cmd 还不是全量
- `Unary / Binary / Pointwise` 大家族
  - 已由 `UnaryOp / UnaryOpAlias / BinaryOp / PointwiseFormula / Clip / Swish / Sigmoid / GELU / PReLU` 等共享覆盖大量常见操作

## 后续推进优先级建议

如果按“最可能直接提升新模型导入成功率”的顺序排：

1. `Shape / Size / Range / ConstantOfShape / Expand`
2. `Gather / GatherElements / GatherND / Where / TopK / OneHot / NonZero`
3. `ScatterElements / ScatterND / Scatter`
4. `GridSample / RoiAlign / NonMaxSuppression`
5. `LSTM` 与随机采样类
6. 频域 / 音频算子

原因：

- 现在的主缺口已经不再是经典卷积、激活、归一化，而是 shape tensor、索引选择、gather/scatter 这些更“图编排化”的算子
- 当前 attention 相关的 `Reshape / MatMul / SDPA / MultiHeadAttention` 已经比 2026-06-05 那一版明显前进，因此下一阶段更值得补图结构类算子

## 当前仍待继续确认的点

- `Buffer(legacy)` 路径后续会删除，所以这列更多是“迁移过程中的真值/兼容基线”，不是最终目标
- 仍需继续把更多 `Cmd` 侧的 placeholder 收敛为真实 `ComputeTexture` pack4 路径
- `InnerProduct / Gemm / MatMul / MultiHeadAttention / SDPA / Reduction / Softmax / Reshape` 这些层已经有部分真实 cmd/RT 特化，但仍不是全覆盖
- `RotaryEmbed / Flatten / Squeeze / Tile` 这类层在 cmd/RT 侧仍有较明显的“alias-only / placeholder-only”尾巴

## 本次状态

- 本轮已重新对照当前代码与官方 `ref/ncnn-master/src/layer/vulkan`
- 本轮已按最新代码状态更新官方 `64` 层与仓内额外 `13` 层的表述
- 本轮已补入 Sentis 官方支持而当前 ncnn 复刻仍缺的主要缺口
