using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnPermuteLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnPermuteLayerRepro() : base(NcnnLayerTypes.Permute, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
        public override void ExecuteComputeBufferPath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
            var axes = NcnnGraphSession.ResolvePermuteAxes(dims, orderType, layer.name);
            var outShape = NcnnGraphSession.ResolvePermuteShape(srcTensor, dims, axes);
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

        public override void ExecuteRenderTexturePath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                var storageShape = NcnnGraphSession.GetTextureStorageShape(srcTex, srcShape);
                textureBlobs[layer.topNames[0]] = NcnnGraphSession.CreateTextureAlias(srcTex, srcShape, storageShape);
                textureShapes[layer.topNames[0]] = srcShape;
            }
            else
            {
                if (canUseLinearMat)
                {
                    var storageShape = NcnnGraphSession.ResolveLinearMatStorageShape(outShape);
                    var outRt = owner.RentTempMat(storageShape.w, storageShape.h, NcnnGraphSession.ResolveLinearMatTextureFormat());
                    owner.Ops.PermuteLinearMat2D(srcTex.texture, srcShape.w, srcShape.h, axes, outRt);
                    NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, storageShape);
                }
                else if (srcShape.dims == 4)
                {
                    if (TryExecuteRenderTextureCdhwToFoldD(owner, layer, srcTex, srcShape, axes, outShape, textureBlobs, textureShapes)
                        || TryExecuteRenderTextureFoldDToCdhw(owner, layer, srcTex, srcShape, axes, outShape, textureBlobs, textureShapes))
                    {
                        owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
                        return;
                    }

                    var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
                    var outSlices = Mathf.Max(1, outShape.d) * outPacks;
                    var outRt = owner.RentTempArray(outShape.w, outShape.h, outSlices, NcnnGraphSession.ResolveTensorTextureFormat(outShape.dims));
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
                    NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape);
                }
                else
                {
                    var outChannels = srcShape.dims == 2 ? 1 : outShape.c;
                    var outPacks = Mathf.Max(1, Mathf.CeilToInt(outChannels / 4f));
                    var outRt = owner.RentTempArray(outShape.w, outShape.h, outPacks, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PermutePack4(srcTex.texture, srcShape.w, srcShape.h, srcShape.dims == 2 ? 1 : srcShape.c, axes, outShape.w, outShape.h, outChannels, outRt);
                    if (srcShape.dims == 2)
                    {
                        NcnnGraphSession.SetTextureBlob(
                            textureBlobs,
                            textureShapes,
                            layer.topNames[0],
                            outRt,
                            outShape,
                            new NcnnGraphSession.BufferShape(3, outShape.w, outShape.h, 1, 1));
                    }
                    else
                    {
                        NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape);
                    }
                }
            }
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var sourceContract = NcnnGraphSession.GetCmdTensorContract(src);
            var srcShape = sourceContract.LogicalShape;
            var orderType = layer.GetInt(0, 0);
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
                    blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorAlias(src, srcShape, sourceContract.StorageShape);
                    if (shapes != null)
                        shapes[layer.topNames[0]] = srcShape;
                }
                else
                {
                    if (canUseLinearMat)
                    {
                        var storageShape = NcnnGraphSession.ResolveLinearMatStorageShape(outShape);
                        var outMat = owner.RentTempMat(cmd, storageShape.w, storageShape.h, NcnnGraphSession.ResolveLinearMatTextureFormat());
                        owner.Ops.PermuteLinearMat2D(cmd, src.texture, srcShape.w, srcShape.h, axes, outMat);
                        blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(outMat, outShape, storageShape, owned: true);
                    }
                    else
                    {
                        if (srcShape.dims == 4)
                        {
                            if (TryExecuteCommandBufferCdhwToFoldD(owner, layer, src, srcShape, axes, outShape, blobs, shapes, cmd)
                                || TryExecuteCommandBufferFoldDToCdhw(owner, layer, src, srcShape, axes, outShape, blobs, shapes, cmd))
                            {
                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                return;
                            }

                            var outPacks4D = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
                            var outDepth4D = Mathf.Max(1, outShape.d) * outPacks4D;
                            var outArr4D = owner.RentTempArray(cmd, outShape.w, outShape.h, outDepth4D, NcnnGraphSession.ResolveTensorTextureFormat(outShape.dims));
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
                                outArr4D);
                            blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(outArr4D, outShape, outShape, owned: true, blobName: layer.topNames[0]);
                        }
                        else
                        {
                            var outChannels = srcShape.dims == 2 ? 1 : outShape.c;
                            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outChannels / 4f));
                            var outArr = owner.RentTempArray(cmd, outShape.w, outShape.h, outPacks, RenderTextureFormat.ARGBHalf);
                            owner.Ops.PermutePack4(cmd, src.texture, srcShape.w, srcShape.h, srcShape.dims == 2 ? 1 : srcShape.c, axes, outShape.w, outShape.h, outChannels, outArr);
                            var storageShape = srcShape.dims == 2
                                ? new NcnnGraphSession.BufferShape(3, outShape.w, outShape.h, 1, 1)
                                : outShape;
                            blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(outArr, outShape, storageShape, owned: true, blobName: layer.topNames[0]);
                        }
                    }
                    if (shapes != null)
                        shapes[layer.topNames[0]] = outShape;
                }
            }
            else
            {
                throw new InvalidOperationException(
                    "Permute command-buffer Pack4 profile is unsupported; placeholder publication is prohibited"
                    + " | layer=" + layer.name
                    + " | logical=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | storage=d" + sourceContract.StorageShape.dims + ":" + sourceContract.StorageShape.w + "x" + sourceContract.StorageShape.h + "x" + sourceContract.StorageShape.d + "x" + sourceContract.StorageShape.c
                    + " | layout=" + sourceContract.LayoutKind
                    + " | orderType=" + orderType);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool TryGetPermuteTextureInput(
            NcnnGraphSession owner,
            string inputName,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.TensorRef> textureBlobs,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.BufferShape> textureShapes,
            System.Collections.Generic.Dictionary<string, ComputeBuffer> bufferBlobs,
            System.Collections.Generic.Dictionary<string, NcnnTensorBuffer> bufferViews,
            out NcnnGraphSession.TensorRef srcTex,
            out NcnnGraphSession.BufferShape srcShape)
        {
            srcTex = null;
            srcShape = default;

            if (owner != null
                && owner.TryGetPack4Texture(inputName, textureBlobs, textureShapes, bufferBlobs, bufferViews, out srcTex, out srcShape))
            {
                return true;
            }

            if (!NcnnGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, inputName, out srcTex, out srcShape))
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
            NcnnGraphSession owner,
            NcnnParamModel.Layer layer,
            NcnnGraphSession.TensorRef src,
            NcnnGraphSession.BufferShape srcShape,
            int orderType,
            NcnnGraphSession.BufferShape outShape,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.TensorRef> textureBlobs,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.BufferShape> textureShapes)
        {
            if (owner?.Model?.layers == null || layer == null || src == null || src.texture == null || orderType != 2)
                return false;
            if (srcShape.dims != 3 || outShape.dims != 3)
                return false;

            var storageShape = NcnnGraphSession.GetTextureStorageShape(src, srcShape);
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

            textureBlobs[layer.topNames[0]] = NcnnGraphSession.CreateTextureAlias(src, outShape, storageShape);
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
            NcnnGraphSession owner,
            NcnnParamModel.Layer layer,
            NcnnGraphSession.CmdTensorRef src,
            NcnnGraphSession.BufferShape srcShape,
            int orderType,
            NcnnGraphSession.BufferShape outShape,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.BufferShape> shapes)
        {
            if (owner?.Model?.layers == null || layer == null || src == null || src.texture == null || orderType != 2)
                return false;
            if (srcShape.dims != 3 || outShape.dims != 3)
                return false;

            var storageShape = NcnnGraphSession.GetCmdStorageShape(src, srcShape);
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

            blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorAlias(src, outShape, storageShape);
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
            NcnnGraphSession.TensorRef src,
            NcnnGraphSession.BufferShape srcShape,
            NcnnGraphSession.BufferShape storageShape)
        {
            return src != null
                && src.texture != null
                && !NcnnGraphSession.IsStrictLinearMatTexture(src)
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
            NcnnGraphSession.CmdTensorRef src,
            NcnnGraphSession.BufferShape srcShape,
            NcnnGraphSession.BufferShape storageShape)
        {
            return src != null
                && src.texture != null
                && !NcnnGraphSession.IsStrictLinearMatTexture(src)
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

        private static bool TryExecuteRenderTextureCdhwToFoldD(
            NcnnGraphSession owner,
            NcnnParamModel.Layer layer,
            NcnnGraphSession.TensorRef src,
            NcnnGraphSession.BufferShape srcShape,
            Vector4Int axes,
            NcnnGraphSession.BufferShape outShape,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.TensorRef> textureBlobs,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.BufferShape> textureShapes)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;
            if (!IsExactCdhwStorage(src, srcShape))
                return false;
            if (CanUseExactCdhwStorageShape(outShape))
                return false;
            if (!TryResolveFoldDStorageShape(outShape, out var foldStorageShape))
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outRt = owner.RentTempArray(foldStorageShape.w, foldStorageShape.h, outPacks, NcnnGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.PermutePack4CdhwFoldD(
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
                outRt);
            NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, foldStorageShape);
            owner.DebugLog?.Invoke(
                "[PermuteFoldD][RT] cdhw->foldD"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                + " | storage=d" + foldStorageShape.dims + ":" + foldStorageShape.w + "x" + foldStorageShape.h + "x" + foldStorageShape.d + "x" + foldStorageShape.c);
            return true;
        }

        private static bool TryExecuteRenderTextureFoldDToCdhw(
            NcnnGraphSession owner,
            NcnnParamModel.Layer layer,
            NcnnGraphSession.TensorRef src,
            NcnnGraphSession.BufferShape srcShape,
            Vector4Int axes,
            NcnnGraphSession.BufferShape outShape,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.TensorRef> textureBlobs,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.BufferShape> textureShapes)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;
            if (!IsFoldDStorage(src, srcShape))
                return false;
            if (!CanUseExactCdhwStorageShape(outShape))
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = Mathf.Max(1, outShape.d) * outPacks;
            var outRt = owner.RentTempArray(outShape.w, outShape.h, outSlices, NcnnGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.PermutePack4FoldDToCdhw(
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
                outRt);
            NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, outShape);
            owner.DebugLog?.Invoke(
                "[PermuteFoldD][RT] foldD->cdhw"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static bool TryExecuteCommandBufferCdhwToFoldD(
            NcnnGraphSession owner,
            NcnnParamModel.Layer layer,
            NcnnGraphSession.CmdTensorRef src,
            NcnnGraphSession.BufferShape srcShape,
            Vector4Int axes,
            NcnnGraphSession.BufferShape outShape,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.BufferShape> shapes,
            CommandBuffer cmd)
        {
            if (owner == null || layer == null || src == null || src.texture == null || cmd == null)
                return false;
            if (!IsExactCdhwStorage(src, srcShape))
                return false;
            if (CanUseExactCdhwStorageShape(outShape))
                return false;
            if (!TryResolveFoldDStorageShape(outShape, out var foldStorageShape))
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outRt = owner.RentTempArray(cmd, foldStorageShape.w, foldStorageShape.h, outPacks, NcnnGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.PermutePack4CdhwFoldD(
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
                outRt);
            blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(outRt, outShape, foldStorageShape, owned: true, blobName: layer.topNames[0]);
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[PermuteFoldD][Cmd] cdhw->foldD"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                + " | storage=d" + foldStorageShape.dims + ":" + foldStorageShape.w + "x" + foldStorageShape.h + "x" + foldStorageShape.d + "x" + foldStorageShape.c);
            return true;
        }

        private static bool TryExecuteCommandBufferFoldDToCdhw(
            NcnnGraphSession owner,
            NcnnParamModel.Layer layer,
            NcnnGraphSession.CmdTensorRef src,
            NcnnGraphSession.BufferShape srcShape,
            Vector4Int axes,
            NcnnGraphSession.BufferShape outShape,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, NcnnGraphSession.BufferShape> shapes,
            CommandBuffer cmd)
        {
            if (owner == null || layer == null || src == null || src.texture == null || cmd == null)
                return false;
            if (!IsFoldDStorage(src, srcShape))
                return false;
            if (!CanUseExactCdhwStorageShape(outShape))
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = Mathf.Max(1, outShape.d) * outPacks;
            var outRt = owner.RentTempArray(cmd, outShape.w, outShape.h, outSlices, NcnnGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.PermutePack4FoldDToCdhw(
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
                outRt);
            blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(outRt, outShape, outShape, owned: true, blobName: layer.topNames[0]);
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[PermuteFoldD][Cmd] foldD->cdhw"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static bool IsExactCdhwStorage(NcnnGraphSession.TensorRef tensor, NcnnGraphSession.BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null || logicalShape.dims != 4)
                return false;
            var storageShape = NcnnGraphSession.GetTextureStorageShape(tensor, logicalShape);
            var expectedPacks = Mathf.Max(1, Mathf.CeilToInt(logicalShape.c / 4f));
            return ShapesEqual(storageShape, logicalShape)
                && tensor.width == logicalShape.w
                && tensor.height == logicalShape.h
                && tensor.packs == expectedPacks
                && Mathf.Max(1, tensor.texture.volumeDepth) == Mathf.Max(1, logicalShape.d) * expectedPacks;
        }

        private static bool IsExactCdhwStorage(NcnnGraphSession.CmdTensorRef tensor, NcnnGraphSession.BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null || logicalShape.dims != 4)
                return false;
            var storageShape = NcnnGraphSession.GetCmdStorageShape(tensor, logicalShape);
            var expectedPacks = Mathf.Max(1, Mathf.CeilToInt(logicalShape.c / 4f));
            return ShapesEqual(storageShape, logicalShape)
                && tensor.width == logicalShape.w
                && tensor.height == logicalShape.h
                && tensor.packs == expectedPacks
                && Mathf.Max(1, tensor.texture.depth) == Mathf.Max(1, logicalShape.d) * expectedPacks;
        }

        private static bool IsFoldDStorage(NcnnGraphSession.TensorRef tensor, NcnnGraphSession.BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null || !TryResolveFoldDStorageShape(logicalShape, out var foldStorageShape))
                return false;
            var storageShape = NcnnGraphSession.GetTextureStorageShape(tensor, logicalShape);
            var expectedPacks = Mathf.Max(1, Mathf.CeilToInt(logicalShape.c / 4f));
            return ShapesEqual(storageShape, foldStorageShape)
                && tensor.width == foldStorageShape.w
                && tensor.height == foldStorageShape.h
                && tensor.packs == expectedPacks
                && Mathf.Max(1, tensor.texture.volumeDepth) == expectedPacks;
        }

        private static bool IsFoldDStorage(NcnnGraphSession.CmdTensorRef tensor, NcnnGraphSession.BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null || !TryResolveFoldDStorageShape(logicalShape, out var foldStorageShape))
                return false;
            var storageShape = NcnnGraphSession.GetCmdStorageShape(tensor, logicalShape);
            var expectedPacks = Mathf.Max(1, Mathf.CeilToInt(logicalShape.c / 4f));
            return ShapesEqual(storageShape, foldStorageShape)
                && tensor.width == foldStorageShape.w
                && tensor.height == foldStorageShape.h
                && tensor.packs == expectedPacks
                && Mathf.Max(1, tensor.texture.depth) == expectedPacks;
        }

        private static bool TryResolveFoldDStorageShape(NcnnGraphSession.BufferShape logicalShape, out NcnnGraphSession.BufferShape storageShape)
        {
            storageShape = default;
            if (logicalShape.dims != 4
                || logicalShape.w <= 0
                || logicalShape.h <= 0
                || logicalShape.d <= 0
                || logicalShape.c <= 0)
            {
                return false;
            }

            var foldedHeight = checked(logicalShape.h * logicalShape.d);
            var outPacks = Mathf.Max(1, Mathf.CeilToInt(logicalShape.c / 4f));
            if (foldedHeight > GetMaxTextureSizeSafe() || outPacks > GetMaxTextureArraySlicesSafe())
                return false;

            storageShape = new NcnnGraphSession.BufferShape(4, logicalShape.w, foldedHeight, 1, logicalShape.c);
            return true;
        }

        private static bool CanUseExactCdhwStorageShape(NcnnGraphSession.BufferShape logicalShape)
        {
            if (logicalShape.dims != 4
                || logicalShape.w <= 0
                || logicalShape.h <= 0
                || logicalShape.d <= 0
                || logicalShape.c <= 0)
            {
                return false;
            }

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(logicalShape.c / 4f));
            var outSlices = checked(Mathf.Max(1, logicalShape.d) * outPacks);
            return logicalShape.w <= GetMaxTextureSizeSafe()
                && logicalShape.h <= GetMaxTextureSizeSafe()
                && outSlices <= GetMaxTextureArraySlicesSafe();
        }

        private static bool ShapesEqual(NcnnGraphSession.BufferShape a, NcnnGraphSession.BufferShape b)
        {
            return a.dims == b.dims
                && a.w == b.w
                && a.h == b.h
                && a.d == b.d
                && a.c == b.c;
        }

        private static int GetMaxTextureArraySlicesSafe()
        {
            try
            {
                return Mathf.Max(1, SystemInfo.maxTextureArraySlices);
            }
            catch
            {
                return 2048;
            }
        }

        private static int GetMaxTextureSizeSafe()
        {
            try
            {
                return Mathf.Max(1, SystemInfo.maxTextureSize);
            }
            catch
            {
                return 16384;
            }
        }

        private static bool CanUsePack4Permute(NcnnGraphSession.TensorRef srcTex, NcnnGraphSession.BufferShape srcShape, int orderType, out Vector4Int axes, out NcnnGraphSession.BufferShape outShape)
        {
            axes = default;
            outShape = default;
            if (srcTex == null || srcTex.texture == null)
                return false;
            if (NcnnGraphSession.IsStrictLinearMatTexture(srcTex))
                return false;
            return CanUsePack4PermuteCore(srcShape, orderType, out axes, out outShape);
        }

        private static bool CanUsePack4Permute(NcnnGraphSession.CmdTensorRef src, NcnnGraphSession.BufferShape srcShape, int orderType, out Vector4Int axes, out NcnnGraphSession.BufferShape outShape)
        {
            axes = default;
            outShape = default;
            if (src == null || src.texture == null)
                return false;
            if (NcnnGraphSession.IsStrictLinearMatTexture(src))
                return false;
            return CanUsePack4PermuteCore(srcShape, orderType, out axes, out outShape);
        }

        private static bool CanUseLinearMatPermute(NcnnGraphSession.TensorRef srcTex, NcnnGraphSession.BufferShape srcShape, int orderType, out Vector4Int axes, out NcnnGraphSession.BufferShape outShape)
        {
            axes = default;
            outShape = default;
            if (srcTex == null || srcTex.texture == null || !NcnnGraphSession.IsStrictLinearMatTexture(srcTex))
                return false;

            var storageShape = NcnnGraphSession.GetTextureStorageShape(srcTex, srcShape);
            if (srcShape.dims != 2 || storageShape.dims != 2 || srcTex.packs != 1 || storageShape.w != srcTex.width || storageShape.h != srcTex.height)
                return false;

            return CanUsePack4PermuteCore(srcShape, orderType, out axes, out outShape);
        }

        private static bool CanUseLinearMatPermute(NcnnGraphSession.CmdTensorRef src, NcnnGraphSession.BufferShape srcShape, int orderType, out Vector4Int axes, out NcnnGraphSession.BufferShape outShape)
        {
            axes = default;
            outShape = default;
            if (src == null || src.texture == null || !NcnnGraphSession.IsStrictLinearMatTexture(src))
                return false;

            var storageShape = NcnnGraphSession.GetCmdStorageShape(src, srcShape);
            if (srcShape.dims != 2 || storageShape.dims != 2 || src.packs != 1 || storageShape.w != src.width || storageShape.h != src.height)
                return false;

            return CanUsePack4PermuteCore(srcShape, orderType, out axes, out outShape);
        }

        private static bool CanUsePack4PermuteCore(NcnnGraphSession.BufferShape srcShape, int orderType, out Vector4Int axes, out NcnnGraphSession.BufferShape outShape)
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
                    outShape = new NcnnGraphSession.BufferShape(2, srcShape.h, srcShape.w, 1, 1);
                    return true;
                }

                return false;
            }

            if (srcShape.dims == 3 && srcShape.d == 1)
            {
                axes = NcnnGraphSession.ResolvePermuteAxes(3, orderType, "PermutePack4");
                outShape = NcnnGraphSession.ResolvePermuteShape(srcShape, 3, axes);
                return outShape.dims == 3 && outShape.d == 1;
            }

            if (srcShape.dims == 4)
            {
                axes = NcnnGraphSession.ResolvePermuteAxes(4, orderType, "PermutePack4CDHW");
                outShape = NcnnGraphSession.ResolvePermuteShape(srcShape, 4, axes);
                return outShape.dims == 4;
            }

            return false;
        }
    }
}
