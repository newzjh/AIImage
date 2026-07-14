using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnPermuteLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnPermuteLayerRepro() : base(NcnnLayerTypes.Permute, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var orderType = layer.GetInt(0, 0);
            if (TryGetPermuteTextureInput(
                    owner,
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var srcTex,
                    out var srcShape)
                && (CanUseLinearMatPermute(srcTex, srcShape, orderType, out _, out _)
                    || CanUsePack4Permute(srcTex, srcShape, orderType, out _, out _)))
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            var orderType = layer.GetInt(0, 0);
            var srcTensor = owner.GetReadableTensorInput(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            if (srcTensor == null || srcTensor.buffer == null)
                throw new InvalidOperationException("Permute source not found: " + layer.bottomNames[0]);

            var dims = Mathf.Clamp(srcTensor.dims, 2, 4);
            var axes = NcnnRepro.ResolvePermuteAxes(dims, orderType, layer.name);
            var outShape = NcnnRepro.ResolvePermuteShape(srcTensor, dims, axes);
            var outTensor = owner.RentTempTensorBuffer(outShape.dims, outShape.w, outShape.h, outShape.d, outShape.c);
            owner.Ops.Permute(srcTensor.buffer, dims, srcTensor.w, srcTensor.h, srcTensor.d, srcTensor.c, orderType, outTensor.buffer);

            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: outShape.dims <= 3,
                textureBlobs,
                textureShapes,
                bufferBlobs,
                bufferRefs,
                bufferViews,
                tempOwned);
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;
            var orderType = layer.GetInt(0, 0);

            if (!TryGetPermuteTextureInput(owner, layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
            {
                throw new InvalidOperationException("Permute render-texture path requires supported pack4 input: " + layer.name);
            }

            var canUseLinearMat = CanUseLinearMatPermute(srcTex, srcShape, orderType, out var axes, out var outShape);
            var canUsePack4 = !canUseLinearMat && CanUsePack4Permute(srcTex, srcShape, orderType, out axes, out outShape);
            if (!canUseLinearMat && !canUsePack4)
                throw new InvalidOperationException("Permute render-texture path requires supported pack4 input: " + layer.name);

            if (TryExecuteDirectAttentionPermuteAlias(owner, layer, srcTex, srcShape, orderType, outShape, textureBlobs, textureShapes))
            {
                owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
                return;
            }

            if (orderType == 0)
            {
                var storageShape = NcnnRepro.GetTextureStorageShape(srcTex, srcShape);
                textureBlobs[layer.topNames[0]] = NcnnRepro.CreateTextureAlias(srcTex, srcShape, storageShape);
                textureShapes[layer.topNames[0]] = srcShape;
            }
            else
            {
                if (canUseLinearMat)
                {
                    var storageShape = NcnnRepro.ResolveLinearMatStorageShape(outShape);
                    var outRt = owner.RentTempMat(storageShape.w, storageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
                    owner.Ops.PermuteLinearMat2D(srcTex.texture, srcShape.w, srcShape.h, axes, outRt);
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, storageShape);
                }
                else if (srcShape.dims == 4)
                {
                    var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
                    var outSlices = Mathf.Max(1, outShape.d) * outPacks;
                    var outRt = owner.RentTempArray(outShape.w, outShape.h, outSlices, NcnnRepro.ResolveTensorTextureFormat(outShape.dims));
                    owner.Ops.PermutePack4Cdhw(
                        srcTex.texture,
                        srcShape.w,
                        srcShape.h,
                        srcShape.d,
                        srcShape.c,
                        axes,
                        outShape.w,
                        outShape.h,
                        outShape.d,
                        outShape.c,
                        outRt);
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape);
                }
                else
                {
                    var outChannels = srcShape.dims == 2 ? 1 : outShape.c;
                    var outPacks = Mathf.Max(1, Mathf.CeilToInt(outChannels / 4f));
                    var outRt = owner.RentTempArray(outShape.w, outShape.h, outPacks, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PermutePack4(srcTex.texture, srcShape.w, srcShape.h, srcShape.dims == 2 ? 1 : srcShape.c, axes, outShape.w, outShape.h, outChannels, outRt);
                    if (srcShape.dims == 2)
                    {
                        NcnnRepro.SetTextureBlob(
                            textureBlobs,
                            textureShapes,
                            layer.topNames[0],
                            outRt,
                            outShape,
                            new NcnnRepro.BufferShape(3, outShape.w, outShape.h, 1, 1));
                    }
                    else
                    {
                        NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape);
                    }
                }
            }
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var orderType = layer.GetInt(0, 0);
            if (TryExecuteCommandBufferPack4Linear2DTranspose(owner, layer, src, srcShape, orderType, blobs, shapes, cmd))
            {
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }
            var canUseLinearMat = CanUseLinearMatPermute(src, srcShape, orderType, out var axes, out var outShape);
            var canUsePack4 = !canUseLinearMat && CanUsePack4Permute(src, srcShape, orderType, out axes, out outShape);
            if (canUseLinearMat || canUsePack4)
            {
                if (TryExecuteDirectAttentionPermuteAlias(owner, layer, src, srcShape, orderType, outShape, blobs, shapes))
                {
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }

                if (orderType == 0)
                {
                    var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
                    blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorAlias(src, srcShape, storageShape);
                    shapes[layer.topNames[0]] = srcShape;
                }
                else
                {
                    if (canUseLinearMat)
                    {
                        var storageShape = NcnnRepro.ResolveLinearMatStorageShape(outShape);
                        var outMat = owner.RentTempMat(cmd, storageShape.w, storageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
                        owner.Ops.PermuteLinearMat2D(cmd, src.texture, srcShape.w, srcShape.h, axes, outMat);
                        blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(outMat, outShape, storageShape, owned: true);
                    }
                    else
                    {
                        var outChannels = srcShape.dims == 2 ? 1 : outShape.c;
                        var outPacks = Mathf.Max(1, Mathf.CeilToInt(outChannels / 4f));
                        var outDepth = srcShape.dims == 4 ? Mathf.Max(1, outShape.d) * outPacks : outPacks;
                        var outFormat = srcShape.dims == 4 ? NcnnRepro.ResolveTensorTextureFormat(outShape.dims) : RenderTextureFormat.ARGBHalf;
                        var outArr = owner.RentTempArray(cmd, outShape.w, outShape.h, outDepth, outFormat);
                        if (srcShape.dims == 4)
                        {
                            owner.Ops.PermutePack4Cdhw(
                                cmd,
                                src.texture,
                                srcShape.w,
                                srcShape.h,
                                srcShape.d,
                                srcShape.c,
                                axes,
                                outShape.w,
                                outShape.h,
                                outShape.d,
                                outShape.c,
                                outArr);
                        }
                        else
                        {
                            owner.Ops.PermutePack4(cmd, src.texture, srcShape.w, srcShape.h, srcShape.dims == 2 ? 1 : srcShape.c, axes, outShape.w, outShape.h, outChannels, outArr);
                        }
                        var storageShape = srcShape.dims == 2
                            ? new NcnnRepro.BufferShape(3, outShape.w, outShape.h, 1, 1)
                            : outShape;
                        blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                        {
                            texture = outArr,
                            width = outShape.w,
                            height = outShape.h,
                            packs = outPacks,
                            refs = 1,
                            owned = true,
                            hasLogicalShape = true,
                            logicalShape = outShape,
                            hasStorageShape = true,
                            storageShape = storageShape
                        };
                    }
                    shapes[layer.topNames[0]] = outShape;
                }
            }
            else
            {
                owner.DebugLog?.Invoke(
                    "[CmdPlaceholder][Permute]"
                    + " | layer=" + layer.name
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | orderType=" + orderType);
                NcnnRepro.ResolveCmdTextureLayout(srcShape, out var width, out var height, out var packs);
                owner.PublishCmdTensorLikeInput(cmd, layer.topNames[0], width, height, packs, blobs, shapes, srcShape);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool TryExecuteCommandBufferPack4Linear2DTranspose(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnRepro.CmdTensorRef src,
            NcnnRepro.BufferShape srcShape,
            int orderType,
            System.Collections.Generic.Dictionary<string, NcnnRepro.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> shapes,
            CommandBuffer cmd)
        {
            if (owner == null || layer == null || src == null || src.texture == null
                || orderType != 1
                || !NcnnRepro.IsPack4LinearMatTexture(src, srcShape))
            {
                return false;
            }

            var unpacked = owner.RentTempArray(cmd, srcShape.w, srcShape.h, 1, src.texture.format);
            try
            {
                owner.Ops.ReshapePack4ToPack4(
                    cmd,
                    src.texture,
                    srcShape.w,
                    srcShape.h,
                    srcShape.d,
                    srcShape.c,
                    srcShape.dims,
                    srcShape.w,
                    srcShape.h,
                    1,
                    1,
                    2,
                    unpacked,
                    inputPack4Linear: true);

                var outputShape = new NcnnRepro.BufferShape(2, srcShape.h, srcShape.w, 1, 1);
                var output = owner.RentTempArray(cmd, outputShape.w, outputShape.h, 1, src.texture.format);
                owner.Ops.PermutePack4(
                    cmd,
                    unpacked,
                    srcShape.w,
                    srcShape.h,
                    1,
                    new Vector4Int(1, 0, 2, 0),
                    outputShape.w,
                    outputShape.h,
                    1,
                    output);
                var storageShape = new NcnnRepro.BufferShape(3, outputShape.w, outputShape.h, 1, 1);
                blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(output, outputShape, storageShape, owned: true);
                if (shapes != null)
                    shapes[layer.topNames[0]] = outputShape;
                owner.DebugLog?.Invoke(
                    "[CmdTexture][PermutePack4Linear2DTranspose]"
                    + " | layer=" + layer.name
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h
                    + " | dst=d" + outputShape.dims + ":" + outputShape.w + "x" + outputShape.h);
                return true;
            }
            finally
            {
                owner.ReturnTempArray(cmd, unpacked);
            }
        }

        private static bool TryGetPermuteTextureInput(
            NcnnRepro owner,
            string inputName,
            System.Collections.Generic.Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            System.Collections.Generic.Dictionary<string, ComputeBuffer> bufferBlobs,
            System.Collections.Generic.Dictionary<string, NcnnTensorBuffer> bufferViews,
            out NcnnRepro.TensorRef srcTex,
            out NcnnRepro.BufferShape srcShape)
        {
            srcTex = null;
            srcShape = default;

            if (owner != null
                && owner.TryGetPack4Texture(inputName, textureBlobs, textureShapes, bufferBlobs, bufferViews, out srcTex, out srcShape))
            {
                return true;
            }

            if (!NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, inputName, out srcTex, out srcShape))
                return false;

            return srcTex != null
                && srcTex.texture != null
                && srcShape.dims == 2
                && srcShape.w > 0
                && srcShape.h > 0
                && srcTex.width == srcShape.w
                && srcTex.height == srcShape.h
                && srcTex.packs == 1;
        }

        private static bool TryExecuteDirectAttentionPermuteAlias(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnRepro.TensorRef src,
            NcnnRepro.BufferShape srcShape,
            int orderType,
            NcnnRepro.BufferShape outShape,
            System.Collections.Generic.Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> textureShapes)
        {
            if (owner?.Model?.layers == null || layer == null || src == null || src.texture == null || orderType != 2)
                return false;
            if (srcShape.dims != 3 || outShape.dims != 3)
                return false;

            var storageShape = NcnnRepro.GetTextureStorageShape(src, srcShape);
            if (!IsDirectAttentionPackedStorage(src, srcShape, storageShape))
                return false;

            var isQkvPrep = srcShape.w == storageShape.w
                && srcShape.h == storageShape.c
                && srcShape.c == storageShape.h
                && outShape.w == storageShape.w
                && outShape.h == storageShape.h
                && outShape.c == storageShape.c
                && HasSingleConsumerOfType(owner.Model, layer.topNames, NcnnLayerTypes.SDPA);
            var isContextFlatten = srcShape.w == storageShape.w
                && srcShape.h == storageShape.h
                && srcShape.c == storageShape.c
                && outShape.w == storageShape.w
                && outShape.h == storageShape.c
                && outShape.c == storageShape.h
                && HasSingleContextFlattenConsumer(owner.Model, layer.topNames);
            if (!isQkvPrep && !isContextFlatten)
                return false;

            textureBlobs[layer.topNames[0]] = NcnnRepro.CreateTextureAlias(src, outShape, storageShape);
            textureShapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[Attention][PermuteAlias]"
                + " | layer=" + layer.name
                + " | mode=" + (isQkvPrep ? "qkv" : "context")
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | storage=d" + storageShape.dims + ":" + storageShape.w + "x" + storageShape.h + "x" + storageShape.d + "x" + storageShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static bool TryExecuteDirectAttentionPermuteAlias(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnRepro.CmdTensorRef src,
            NcnnRepro.BufferShape srcShape,
            int orderType,
            NcnnRepro.BufferShape outShape,
            System.Collections.Generic.Dictionary<string, NcnnRepro.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> shapes)
        {
            if (owner?.Model?.layers == null || layer == null || src == null || src.texture == null || orderType != 2)
                return false;
            if (srcShape.dims != 3 || outShape.dims != 3)
                return false;

            var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
            if (!IsDirectAttentionPackedStorage(src, srcShape, storageShape))
                return false;

            var isQkvPrep = srcShape.w == storageShape.w
                && srcShape.h == storageShape.c
                && srcShape.c == storageShape.h
                && outShape.w == storageShape.w
                && outShape.h == storageShape.h
                && outShape.c == storageShape.c
                && HasSingleConsumerOfType(owner.Model, layer.topNames, NcnnLayerTypes.SDPA);
            var isContextFlatten = srcShape.w == storageShape.w
                && srcShape.h == storageShape.h
                && srcShape.c == storageShape.c
                && outShape.w == storageShape.w
                && outShape.h == storageShape.c
                && outShape.c == storageShape.h
                && HasSingleContextFlattenConsumer(owner.Model, layer.topNames);
            if (!isQkvPrep && !isContextFlatten)
                return false;

            blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorAlias(src, outShape, storageShape);
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[CmdAttention][PermuteAlias]"
                + " | layer=" + layer.name
                + " | mode=" + (isQkvPrep ? "qkv" : "context")
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | storage=d" + storageShape.dims + ":" + storageShape.w + "x" + storageShape.h + "x" + storageShape.d + "x" + storageShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static bool IsDirectAttentionPackedStorage(
            NcnnRepro.TensorRef src,
            NcnnRepro.BufferShape srcShape,
            NcnnRepro.BufferShape storageShape)
        {
            return src != null
                && src.texture != null
                && !NcnnRepro.IsStrictLinearMatTexture(src)
                && storageShape.dims == 3
                && storageShape.d == 1
                && storageShape.w > 0
                && storageShape.h > 0
                && storageShape.c > 1
                && src.width == storageShape.w
                && src.height == storageShape.h
                && src.packs == Mathf.Max(1, Mathf.CeilToInt(storageShape.c / 4f));
        }

        private static bool IsDirectAttentionPackedStorage(
            NcnnRepro.CmdTensorRef src,
            NcnnRepro.BufferShape srcShape,
            NcnnRepro.BufferShape storageShape)
        {
            return src != null
                && src.texture != null
                && !NcnnRepro.IsStrictLinearMatTexture(src)
                && storageShape.dims == 3
                && storageShape.d == 1
                && storageShape.w > 0
                && storageShape.h > 0
                && storageShape.c > 1
                && src.width == storageShape.w
                && src.height == storageShape.h
                && src.packs == Mathf.Max(1, Mathf.CeilToInt(storageShape.c / 4f));
        }

        private static bool HasSingleConsumerOfType(NcnnParamModel model, string[] topNames, NcnnLayerTypeKey type)
        {
            if (model == null || topNames == null || topNames.Length == 0)
                return false;

            var consumer = FindSingleConsumer(model, topNames[0]);
            return consumer != null && consumer.type == type;
        }

        private static bool HasSingleContextFlattenConsumer(NcnnParamModel model, string[] topNames)
        {
            if (model == null || topNames == null || topNames.Length == 0)
                return false;

            var reshape = FindSingleConsumer(model, topNames[0]);
            if (reshape == null
                || reshape.type != NcnnLayerTypes.Reshape
                || reshape.topNames == null
                || reshape.topNames.Length != 1)
                return false;

            var next = FindSingleConsumer(model, reshape.topNames[0]);
            return next != null && next.type == NcnnLayerTypes.InnerProduct;
        }

        private static NcnnParamModel.Layer FindSingleConsumer(NcnnParamModel model, string blobName)
        {
            if (model?.layers == null || string.IsNullOrWhiteSpace(blobName))
                return null;

            NcnnParamModel.Layer found = null;
            for (var i = 0; i < model.layers.Count; i++)
            {
                var candidate = model.layers[i];
                if (candidate?.bottomNames == null)
                    continue;
                for (var j = 0; j < candidate.bottomNames.Length; j++)
                {
                    if (!string.Equals(candidate.bottomNames[j], blobName, StringComparison.Ordinal))
                        continue;
                    if (found != null)
                        return null;
                    found = candidate;
                    break;
                }
            }

            return found;
        }

        private static bool CanUsePack4Permute(NcnnRepro.TensorRef srcTex, NcnnRepro.BufferShape srcShape, int orderType, out Vector4Int axes, out NcnnRepro.BufferShape outShape)
        {
            axes = default;
            outShape = default;
            if (srcTex == null || srcTex.texture == null)
                return false;
            if (NcnnRepro.IsStrictLinearMatTexture(srcTex))
                return false;
            return CanUsePack4PermuteCore(srcShape, orderType, out axes, out outShape);
        }

        private static bool CanUsePack4Permute(NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape srcShape, int orderType, out Vector4Int axes, out NcnnRepro.BufferShape outShape)
        {
            axes = default;
            outShape = default;
            if (src == null || src.texture == null)
                return false;
            if (NcnnRepro.IsStrictLinearMatTexture(src))
                return false;
            return CanUsePack4PermuteCore(srcShape, orderType, out axes, out outShape);
        }

        private static bool CanUseLinearMatPermute(NcnnRepro.TensorRef srcTex, NcnnRepro.BufferShape srcShape, int orderType, out Vector4Int axes, out NcnnRepro.BufferShape outShape)
        {
            axes = default;
            outShape = default;
            if (srcTex == null || srcTex.texture == null || !NcnnRepro.IsStrictLinearMatTexture(srcTex))
                return false;

            var storageShape = NcnnRepro.GetTextureStorageShape(srcTex, srcShape);
            if (srcShape.dims != 2 || storageShape.dims != 2 || srcTex.packs != 1 || storageShape.w != srcTex.width || storageShape.h != srcTex.height)
                return false;

            return CanUsePack4PermuteCore(srcShape, orderType, out axes, out outShape);
        }

        private static bool CanUseLinearMatPermute(NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape srcShape, int orderType, out Vector4Int axes, out NcnnRepro.BufferShape outShape)
        {
            axes = default;
            outShape = default;
            if (src == null || src.texture == null || !NcnnRepro.IsStrictLinearMatTexture(src))
                return false;

            var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
            if (srcShape.dims != 2 || storageShape.dims != 2 || src.packs != 1 || storageShape.w != src.width || storageShape.h != src.height)
                return false;

            return CanUsePack4PermuteCore(srcShape, orderType, out axes, out outShape);
        }

        private static bool CanUsePack4PermuteCore(NcnnRepro.BufferShape srcShape, int orderType, out Vector4Int axes, out NcnnRepro.BufferShape outShape)
        {
            axes = default;
            outShape = default;
            if (srcShape.dims == 2)
            {
                if (orderType == 0)
                {
                    axes = new Vector4Int(0, 1, 2, 0);
                    outShape = srcShape;
                    return true;
                }

                if (orderType == 1)
                {
                    axes = new Vector4Int(1, 0, 2, 0);
                    outShape = new NcnnRepro.BufferShape(2, srcShape.h, srcShape.w, 1, 1);
                    return true;
                }

                return false;
            }

            if (srcShape.dims == 3 && srcShape.d == 1)
            {
                axes = NcnnRepro.ResolvePermuteAxes(3, orderType, "PermutePack4");
                outShape = NcnnRepro.ResolvePermuteShape(srcShape, 3, axes);
                return outShape.dims == 3 && outShape.d == 1;
            }

            if (srcShape.dims == 4)
            {
                axes = NcnnRepro.ResolvePermuteAxes(4, orderType, "PermutePack4CDHW");
                outShape = NcnnRepro.ResolvePermuteShape(srcShape, 4, axes);
                return outShape.dims == 4;
            }

            return false;
        }
    }
}
