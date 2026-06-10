# ncnn Vulkan Layer 复刻覆盖统计

更新时间：2026-06-05

## 统计口径

- 官方基线目录：`ref/ncnn-master/src/layer/vulkan`
- 统计对象：仅统计官方有 Vulkan 实现的 layer / operator
- 当前工程复刻入口：`Assets/Scripts/NcnnLayers`
- 当前注册入口来源：`Assets/Scripts/NcnnLayers/NcnnLayerFactoryRepro.cs`
- 本文的“已复刻”表示：
  - 已有对应 `LayerRepro` 入口
  - 已进入 `NcnnLayerFactoryRepro` 注册表
  - 已进入当前 `Assembly-CSharp.csproj` 编译
- 本文的“未复刻”表示：
  - 官方 `src/layer/vulkan/*.cpp` 有对应层
  - 但本工程没有对应复刻入口

## 总结

- 官方 Vulkan layer 数量：64
- 当前已复刻入口数量：64
- 当前未复刻入口数量：0
- 当前入口覆盖率：100%

说明：

- 这里的 100% 仅代表“层级入口覆盖完成”。
- 不代表每个 layer 的 `LoadLayer`、`ExecuteBuffer`、pack4 `RenderTexture` 路径、`ExecuteCommandBuffer` 都已经和官方语义完全对齐。
- 后续真正还要继续收敛的重点，是各 layer 的实现完整度、shader 完整度，以及 compute buffer / pack4 rendertexture 两条路径的一致性验证。

## 路径迁移提示

- 当前已在 `Assets/Scripts/NcnnLayers/NcnnBaseLayerRepro.cs` 以及各个 `LayerRepro` 类上补充统一迁移提示。
- compute buffer 路径当前仍保留，主要作为兼容路径 / 真值路径参考；后续应尽量减少新增 compute-buffer-only 分支。
- 近阶段优先迁移到 pack4 `RenderTexture` 执行路径。
- 长期目标是迁移到基于 `ComputeTexture` 的 `ExecuteCommandBuffer` pack4 RT 路径，以支持 async compute，以及 command buffer 内创建临时 RT。

## 路径统计口径

- 本节“路径使用情况”按当前工程内 `Assets/Scripts/NcnnLayers` 下的 `LayerRepro` 类统计。
- 当前 `LayerRepro` 类数量：63。
- 之所以不是 64，是因为官方部分 Vulkan layer 共享同一个复刻类：
  - `NcnnPointwiseFormulaLayerRepro` 对应 `CELU / ELU / Erf / HardSigmoid / HardSwish / Mish / SELU / Shrink / Softplus`
  - `NcnnUnaryOpAliasLayerRepro` 对应 `AbsVal / TanH` 等 alias unary 项
- 状态定义：
  - `compute buffer`
    - `完整`：已有真实 compute buffer 计算逻辑，可作为当前真值路径
    - `别名/重解释`：主要做 buffer / shape alias 或重解释，不承担真实算子计算
    - `无`：该层本身不执行或当前没有 compute buffer 路径
  - `pack4 RT (ExecuteBuffer)`
    - `真实`：已有真实 pack4 `RenderTexture` 执行逻辑
    - `部分`：仅部分配置 / 特化分支走 pack4 RT，其余仍回 compute buffer
    - `别名`：只保留 / 传递已有纹理链，不做真实 RT 算子执行
    - `无`：当前没有 pack4 RT 路径
  - `cmd pack4 RT (ExecuteCommandBuffer)`
    - `真实`：已有真实 `ComputeTexture` pack4 RT command-buffer 执行逻辑
    - `部分`：仅部分配置走真实 cmd pack4 RT，其余仍走 copy / placeholder
    - `别名/拷贝`：只做 alias、shape 重解释或 texture copy，不是该层的真实 RT shader 执行
    - `材质化`：由 buffer 结果直接材质化为 `ComputeTexture`，不是该层自有 shader 路径
    - `占位`：仅发布 shape-correct placeholder，保证 command buffer 链路可编译
    - `无`：该层本身不执行

## 当前 LayerRepro 路径覆盖汇总

- `compute buffer`：`完整 58`，`别名/重解释 4`，`无 1`
- `pack4 RT (ExecuteBuffer)`：`真实 17`，`部分 16`，`别名 5`，`无 25`
- `cmd pack4 RT (ExecuteCommandBuffer)`：`真实 13`，`部分 19`，`别名/拷贝 7`，`材质化 1`，`占位 22`，`无 1`
- 从迁移优先级看：
  - 已具备真实或部分 `pack4 RT` 的类共有 `33`
  - 已具备真实、部分、别名/拷贝或材质化的 command-buffer 侧非纯占位类共有 `40`
  - 仍是 command-buffer 纯占位的类共有 `22`，这些是后续补 `ComputeTexture` pack4 RT 的优先收敛对象

## 当前 LayerRepro 路径使用情况

| LayerRepro | compute buffer | pack4 RT (`ExecuteBuffer`) | cmd pack4 RT (`ExecuteCommandBuffer`) | 备注 |
| --- | --- | --- | --- | --- |
| `NcnnBatchNormLayerRepro` | 完整 | 真实 | 真实 | BatchNorm pack4 shader 已接通。 |
| `NcnnBinaryOpLayerRepro` | 完整 | 部分 | 部分 | pack4 RT 覆盖标量/同形/广播主路径，仍保留 buffer / placeholder 兜底。 |
| `NcnnCastLayerRepro` | 完整 | 无 | 占位 | 当前只有 compute buffer 真值路径。 |
| `NcnnClipLayerRepro` | 完整 | 真实 | 真实 | Clip pack4 shader 已接通。 |
| `NcnnConcatLayerRepro` | 完整 | 部分 | 部分 | 当前仅 channel-axis 的 exact pack4 concat 走 RT。 |
| `NcnnConvolution1DLayerRepro` | 完整 | 无 | 占位 | cmd 仅发布 shape-correct placeholder。 |
| `NcnnConvolutionDepthWiseLayerRepro` | 完整 | 部分 | 部分 | 仅部分 pack4 kernel / 特化分支接通。 |
| `NcnnConvolutionLayerRepro` | 完整 | 部分 | 部分 | `1x1 / 3x3 / depthwise` 等 pack4 分支已接，其他配置仍回 buffer / placeholder。 |
| `NcnnCropLayerRepro` | 完整 | 部分 | 部分 | `dims=3` 且可 pack4 时走真实 RT，其他情况回 buffer / copy。 |
| `NcnnDeconvolutionDepthWiseLayerRepro` | 完整 | 无 | 占位 | cmd 目前仅保留输出 shape。 |
| `NcnnDeconvolutionLayerRepro` | 完整 | 无 | 占位 | cmd 目前仅保留输出 shape。 |
| `NcnnDeepCopyLayerRepro` | 完整 | 真实 | 真实 | `CopyPack4 / cmd CopyPack4` 均可用。 |
| `NcnnDequantizeLayerRepro` | 完整 | 无 | 占位 | 当前只有 buffer 真值路径。 |
| `NcnnDropoutLayerRepro` | 完整 | 真实 | 真实 | `scale!=1` 时可直接走 pointwise pack4。 |
| `NcnnEltwiseLayerRepro` | 完整 | 真实 | 真实 | 多输入 pack4 累积路径已接通。 |
| `NcnnEmbedLayerRepro` | 完整 | 无 | 占位 | 当前只有 buffer 真值路径。 |
| `NcnnExpandDimsLayerRepro` | 别名/重解释 | 别名 | 别名/拷贝 | 主要保留 buffer / texture alias 与 shape 重解释。 |
| `NcnnFlattenLayerRepro` | 完整 | 无 | 别名/拷贝 | cmd 当前只做 shape 重解释。 |
| `NcnnGeluLayerRepro` | 完整 | 真实 | 真实 | Gelu pack4 shader 已接通。 |
| `NcnnGemmLayerRepro` | 完整 | 无 | 占位 | cmd 目前仅保留输出 shape。 |
| `NcnnGroupNormLayerRepro` | 完整 | 部分 | 部分 | pack4 RT 依赖 affine + GroupNorm texture path 开关。 |
| `NcnnInnerProductLayerRepro` | 完整 | 无 | 占位 | cmd 目前仅保留输出 shape。 |
| `NcnnInputLayerRepro` | 无 | 无 | 无 | 输入层本身不做执行。 |
| `NcnnInstanceNormLayerRepro` | 完整 | 部分 | 部分 | pack4 RT 依赖 affine + GroupNorm texture path 开关。 |
| `NcnnInterpLayerRepro` | 完整 | 部分 | 部分 | `2x / down2 / 通用 linear` pack4 已接，其他 resize 类型仍回 buffer / placeholder。 |
| `NcnnLayerNormLayerRepro` | 完整 | 无 | 占位 | 当前只有 buffer 真值路径。 |
| `NcnnLRNLayerRepro` | 完整 | 无 | 占位 | 当前只有 buffer 真值路径。 |
| `NcnnMatMulLayerRepro` | 完整 | 无 | 占位 | cmd 目前仅保留输出 shape。 |
| `NcnnMaxPoolingIndLayerRepro` | 完整 | 真实 | 真实 | 已具备 texture 输出与 index texture 输出。 |
| `NcnnMaxUnPoolingLayerRepro` | 完整 | 部分 | 部分 | buffer 路径完整；texture / cmd pack4 路径仍有 CPU 辅助或 placeholder 兜底。 |
| `NcnnMemoryDataLayerRepro` | 完整 | 无 | 材质化 | cmd 通过 `PublishCmdTensorBufferOutput` 材质化为 `ComputeTexture`。 |
| `NcnnMultiHeadAttentionLayerRepro` | 完整 | 无 | 占位 | cmd 目前仅保留输出 shape。 |
| `NcnnNoopLayerRepro` | 别名/重解释 | 别名 | 别名/拷贝 | 主要保留 buffer / texture alias。 |
| `NcnnNormalizeLayerRepro` | 完整 | 无 | 占位 | 当前只有 buffer 真值路径。 |
| `NcnnPackingLayerRepro` | 完整 | 无 | 占位 | cmd 目前仅保留输出 shape。 |
| `NcnnPaddingLayerRepro` | 完整 | 部分 | 部分 | 当前仅 simple pack4 padding 走 RT / cmd RT。 |
| `NcnnPermuteLayerRepro` | 完整 | 部分 | 部分 | 当前仅 `dims=3` 的 pack4 permute 走 RT / cmd RT。 |
| `NcnnPixelShuffleLayerRepro` | 完整 | 真实 | 部分 | `ExecuteBuffer` 已有真实 pack4 RT；cmd 遇到不满足 pack4 条件时仍 placeholder。 |
| `NcnnPointwiseFormulaLayerRepro` | 完整 | 真实 | 真实 | 覆盖 `CELU / ELU / Erf / HardSigmoid / HardSwish / Mish / SELU / Shrink / Softplus`。 |
| `NcnnPoolingLayerRepro` | 完整 | 部分 | 部分 | 常见 2D pack4 pooling 已接，其他配置仍回 buffer / placeholder。 |
| `NcnnPReLULayerRepro` | 完整 | 部分 | 部分 | 当前仅标量 slope 走 pack4 RT / cmd RT。 |
| `NcnnPriorBoxLayerRepro` | 完整 | 无 | 占位 | cmd 仅发布输出 shape。 |
| `NcnnQuantizeLayerRepro` | 完整 | 无 | 占位 | 当前只有 buffer 真值路径。 |
| `NcnnReductionLayerRepro` | 完整 | 无 | 占位 | cmd 目前仅保留输出 shape。 |
| `NcnnReLULayerRepro` | 完整 | 真实 | 真实 | `LeakyReluPack4` 路径已接通。 |
| `NcnnReorgLayerRepro` | 完整 | 真实 | 部分 | `ExecuteBuffer` 已有真实 pack4 RT；cmd 遇到不满足 pack4 条件时仍 placeholder。 |
| `NcnnRequantizeLayerRepro` | 完整 | 无 | 占位 | 当前只有 buffer 真值路径。 |
| `NcnnReshapeLayerRepro` | 别名/重解释 | 别名 | 别名/拷贝 | 优先保留 texture alias；cmd 仅 alias / copy，不做真实 layer shader。 |
| `NcnnRMSNormLayerRepro` | 完整 | 无 | 占位 | 当前只有 buffer 真值路径。 |
| `NcnnRotaryEmbedLayerRepro` | 完整 | 无 | 别名/拷贝 | cmd 当前只透传 shape / 引用。 |
| `NcnnScaleLayerRepro` | 完整 | 部分 | 部分 | 当前仅静态单标量 scale 走 pack4 RT / cmd RT。 |
| `NcnnSdpaLayerRepro` | 完整 | 无 | 占位 | cmd 目前仅保留输出与 cache shape。 |
| `NcnnShuffleChannelLayerRepro` | 完整 | 真实 | 部分 | `ExecuteBuffer` 已有真实 pack4 RT；cmd 不满足 pack4 条件时走 copy fallback。 |
| `NcnnSigmoidLayerRepro` | 完整 | 真实 | 真实 | Sigmoid pack4 shader 已接通。 |
| `NcnnSliceLayerRepro` | 完整 | 真实 | 部分 | `ExecuteBuffer` 已有真实 pack4 slice；cmd 不满足 pack4 条件时仍 placeholder。 |
| `NcnnSoftmaxLayerRepro` | 完整 | 部分 | 部分 | 当前仅 `axis==0` 的 channel softmax 走 pack4 RT / cmd RT。 |
| `NcnnSplitLayerRepro` | 别名/重解释 | 别名 | 别名/拷贝 | 主要复制引用并保留 texture alias。 |
| `NcnnSqueezeLayerRepro` | 完整 | 无 | 别名/拷贝 | cmd 当前只做 shape 重解释。 |
| `NcnnSwishLayerRepro` | 完整 | 真实 | 真实 | Swish pack4 shader 已接通。 |
| `NcnnTileLayerRepro` | 完整 | 别名 | 占位 | 仅 passthrough 分支保留 texture alias，真正 tile 仍无 RT / cmd RT 实现。 |
| `NcnnUnaryOpAliasLayerRepro` | 完整 | 真实 | 真实 | 当前承接 Vulkan 统计里的 `AbsVal / TanH` 等 alias unary。 |
| `NcnnUnaryOpLayerRepro` | 完整 | 真实 | 真实 | UnaryOp pack4 shader 已接通。 |
| `NcnnUnfoldLayerRepro` | 完整 | 无 | 占位 | cmd 目前仅保留输出 shape。 |

## 官方 Vulkan Layer 对照表

| 官方 Vulkan layer | 当前复刻入口 |
| --- | --- |
| `absval_vulkan` | `NcnnUnaryOpAliasLayerRepro(AbsVal)` |
| `batchnorm_vulkan` | `NcnnBatchNormLayerRepro` |
| `binaryop_vulkan` | `NcnnBinaryOpLayerRepro` |
| `cast_vulkan` | `NcnnCastLayerRepro` |
| `celu_vulkan` | `NcnnPointwiseFormulaLayerRepro(CELU)` |
| `clip_vulkan` | `NcnnClipLayerRepro` |
| `concat_vulkan` | `NcnnConcatLayerRepro` |
| `convolution_vulkan` | `NcnnConvolutionLayerRepro` |
| `convolution1d_vulkan` | `NcnnConvolution1DLayerRepro` |
| `convolutiondepthwise_vulkan` | `NcnnConvolutionDepthWiseLayerRepro` |
| `crop_vulkan` | `NcnnCropLayerRepro` |
| `deconvolution_vulkan` | `NcnnDeconvolutionLayerRepro` |
| `deconvolutiondepthwise_vulkan` | `NcnnDeconvolutionDepthWiseLayerRepro` |
| `deepcopy_vulkan` | `NcnnDeepCopyLayerRepro` |
| `dequantize_vulkan` | `NcnnDequantizeLayerRepro` |
| `dropout_vulkan` | `NcnnDropoutLayerRepro` |
| `eltwise_vulkan` | `NcnnEltwiseLayerRepro` |
| `elu_vulkan` | `NcnnPointwiseFormulaLayerRepro(ELU)` |
| `erf_vulkan` | `NcnnPointwiseFormulaLayerRepro(Erf)` |
| `flatten_vulkan` | `NcnnFlattenLayerRepro` |
| `gelu_vulkan` | `NcnnGeluLayerRepro` |
| `gemm_vulkan` | `NcnnGemmLayerRepro` |
| `groupnorm_vulkan` | `NcnnGroupNormLayerRepro` |
| `hardsigmoid_vulkan` | `NcnnPointwiseFormulaLayerRepro(HardSigmoid)` |
| `hardswish_vulkan` | `NcnnPointwiseFormulaLayerRepro(HardSwish)` |
| `innerproduct_vulkan` | `NcnnInnerProductLayerRepro` |
| `instancenorm_vulkan` | `NcnnInstanceNormLayerRepro` |
| `interp_vulkan` | `NcnnInterpLayerRepro` |
| `layernorm_vulkan` | `NcnnLayerNormLayerRepro` |
| `lrn_vulkan` | `NcnnLRNLayerRepro` |
| `memorydata_vulkan` | `NcnnMemoryDataLayerRepro` |
| `mish_vulkan` | `NcnnPointwiseFormulaLayerRepro(Mish)` |
| `multiheadattention_vulkan` | `NcnnMultiHeadAttentionLayerRepro` |
| `noop_vulkan` | `NcnnNoopLayerRepro` |
| `normalize_vulkan` | `NcnnNormalizeLayerRepro` |
| `packing_vulkan` | `NcnnPackingLayerRepro` |
| `padding_vulkan` | `NcnnPaddingLayerRepro` |
| `permute_vulkan` | `NcnnPermuteLayerRepro` |
| `pixelshuffle_vulkan` | `NcnnPixelShuffleLayerRepro` |
| `pooling_vulkan` | `NcnnPoolingLayerRepro` |
| `prelu_vulkan` | `NcnnPReLULayerRepro` |
| `priorbox_vulkan` | `NcnnPriorBoxLayerRepro` |
| `quantize_vulkan` | `NcnnQuantizeLayerRepro` |
| `reduction_vulkan` | `NcnnReductionLayerRepro` |
| `relu_vulkan` | `NcnnReLULayerRepro` |
| `reorg_vulkan` | `NcnnReorgLayerRepro` |
| `requantize_vulkan` | `NcnnRequantizeLayerRepro` |
| `reshape_vulkan` | `NcnnReshapeLayerRepro` |
| `rmsnorm_vulkan` | `NcnnRMSNormLayerRepro` |
| `rotaryembed_vulkan` | `NcnnRotaryEmbedLayerRepro` |
| `scale_vulkan` | `NcnnScaleLayerRepro` |
| `sdpa_vulkan` | `NcnnSdpaLayerRepro` |
| `selu_vulkan` | `NcnnPointwiseFormulaLayerRepro(SELU)` |
| `shrink_vulkan` | `NcnnPointwiseFormulaLayerRepro(Shrink)` |
| `shufflechannel_vulkan` | `NcnnShuffleChannelLayerRepro` |
| `sigmoid_vulkan` | `NcnnSigmoidLayerRepro` |
| `slice_vulkan` | `NcnnSliceLayerRepro` |
| `softmax_vulkan` | `NcnnSoftmaxLayerRepro` |
| `softplus_vulkan` | `NcnnPointwiseFormulaLayerRepro(Softplus)` |
| `split_vulkan` | `NcnnSplitLayerRepro` |
| `swish_vulkan` | `NcnnSwishLayerRepro` |
| `tanh_vulkan` | `NcnnUnaryOpAliasLayerRepro(TanH)` |
| `unaryop_vulkan` | `NcnnUnaryOpLayerRepro` |
| `unfold_vulkan` | `NcnnUnfoldLayerRepro` |

## 当前未复刻清单

当前按官方 Vulkan 目录统计，未复刻项为 0。

## 仓库内额外存在、但不属于官方 Vulkan 统计口径的层

以下层当前在本工程里有实现或注册，但不在 `ref/ncnn-master/src/layer/vulkan` 统计口径内：

- `Input`
- `Embed`
- `ExpandDims`
- `MatMul`
- `MaxPoolingInd`
- `MaxUnPooling`
- `Squeeze`
- `Tile`

这些项不计入本次“官方 Vulkan layer 复刻覆盖率”。

## 当前仍待继续推进的工作

- 逐层确认 `LoadLayer` 是否完整对齐官方参数与权重读取语义
- 逐层确认 `ExecuteBuffer` 的 compute buffer 路径是否可作为真值路径
- 逐层确认 pack4 `RenderTexture` 路径是否与 compute buffer 路径结果一致
- 逐层确认 `ExecuteCommandBuffer` 至少具备可编译、可接入 async compute command buffer 的执行形态
- 补齐仍可能缺失的 shader、特化分支、pack4 辅助 kernel
- 最后再统一进入静默模式逐层正确性比对

## 本次状态

- 本轮已完成 `Assembly-CSharp.csproj` 编译打通
- 本轮未执行任何 layer 正确性测试
- 本轮未拉起 Unity
- 本轮未进行静默 runner 验证
