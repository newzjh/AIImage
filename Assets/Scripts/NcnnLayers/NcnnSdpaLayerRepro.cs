using System;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnSdpaLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnSdpaLayerRepro()
            : base(NcnnLayerTypes.SDPA, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            owner._extraPacks[layer.name] = new NcnnRepro.SdpaPack
            {
                attnMask = layer.GetInt(5, 0) != 0,
                scale = layer.GetFloat(6, 0f),
                kvCache = layer.GetInt(7, 0) != 0,
                int8ScaleTerm = layer.GetInt(18, 0) != 0
            };
            return default;
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.SdpaPack sp)
                throw new InvalidOperationException("SDPA pack not found: " + layer.name);
            if (sp.int8ScaleTerm)
                throw new InvalidOperationException("SDPA int8_scale_term path is not implemented in repro: " + layer.name);

            var queryBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var keyBuf = owner.GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var valueBuf = owner.GetOrConvertToBuffer(layer.bottomNames[2], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var queryView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            var keyView = NcnnRepro.TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
            var valueView = NcnnRepro.TryGetBufferView(layer.bottomNames[2], bufferBlobs, bufferViews);
            if (queryBuf == null || keyBuf == null || valueBuf == null || queryView == null || keyView == null || valueView == null)
                throw new InvalidOperationException("SDPA input missing: " + layer.name);
            if (queryView.dims != 3 || keyView.dims != 3 || valueView.dims != 3)
                throw new InvalidOperationException("SDPA expects dims=3 tensors: " + layer.name);

            var embedDim = queryView.w;
            var srcSeqLen = queryView.h;
            var numHeads = queryView.c;
            var curSeqLen = keyView.h;
            var numGroup = keyView.c;
            var outEmbedDim = valueView.w;
            var pastSeqLen = 0;
            if (numGroup <= 0 || numHeads <= 0 || embedDim <= 0 || outEmbedDim <= 0)
                throw new InvalidOperationException("SDPA invalid shapes: " + layer.name);
            if (numHeads % numGroup != 0)
                throw new InvalidOperationException("SDPA requires num_heads divisible by num_group: " + layer.name);

            ComputeBuffer maskBuf = null;
            NcnnTensorBuffer attnMaskView = null;
            if (sp.attnMask)
            {
                maskBuf = owner.GetOrConvertToBuffer(layer.bottomNames[3], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                attnMaskView = NcnnRepro.TryGetBufferView(layer.bottomNames[3], bufferBlobs, bufferViews);
                if (maskBuf == null || attnMaskView == null)
                    throw new InvalidOperationException("SDPA attention mask input missing: " + layer.name);
            }

            ComputeBuffer keyAllBuf = keyBuf;
            ComputeBuffer valueAllBuf = valueBuf;
            if (sp.kvCache)
            {
                var pastKeyBottom = sp.attnMask ? 4 : 3;
                var pastValueBottom = sp.attnMask ? 5 : 4;
                var pastKeyBuf = owner.GetOrConvertToBuffer(layer.bottomNames[pastKeyBottom], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                var pastValueBuf = owner.GetOrConvertToBuffer(layer.bottomNames[pastValueBottom], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                var pastKeyView = NcnnRepro.TryGetBufferView(layer.bottomNames[pastKeyBottom], bufferBlobs, bufferViews);
                var pastValueView = NcnnRepro.TryGetBufferView(layer.bottomNames[pastValueBottom], bufferBlobs, bufferViews);
                if (pastKeyBuf == null || pastValueBuf == null || pastKeyView == null || pastValueView == null)
                    throw new InvalidOperationException("SDPA kv cache inputs missing: " + layer.name);
                pastSeqLen = pastKeyView.h;

                var dstKeyCount = embedDim * (pastSeqLen + curSeqLen) * numGroup;
                var dstValueCount = outEmbedDim * (pastSeqLen + curSeqLen) * numGroup;
                keyAllBuf = owner.RentTempBuffer(dstKeyCount, sizeof(float));
                valueAllBuf = owner.RentTempBuffer(dstValueCount, sizeof(float));
                tempOwned.Add(keyAllBuf);
                tempOwned.Add(valueAllBuf);

                for (var q = 0; q < numGroup; q++)
                {
                    var pastKeyOffset = q * pastSeqLen * embedDim;
                    var curKeyOffset = q * curSeqLen * embedDim;
                    var dstKeyOffset = q * (pastSeqLen + curSeqLen) * embedDim;
                    owner.Ops.CopyBufPartial(pastKeyBuf, pastKeyOffset, keyAllBuf, pastSeqLen * embedDim, dstKeyOffset);
                    owner.Ops.CopyBufPartial(keyBuf, curKeyOffset, keyAllBuf, curSeqLen * embedDim, dstKeyOffset + pastSeqLen * embedDim);

                    var pastValueOffset = q * pastSeqLen * outEmbedDim;
                    var curValueOffset = q * curSeqLen * outEmbedDim;
                    var dstValueOffset = q * (pastSeqLen + curSeqLen) * outEmbedDim;
                    owner.Ops.CopyBufPartial(pastValueBuf, pastValueOffset, valueAllBuf, pastSeqLen * outEmbedDim, dstValueOffset);
                    owner.Ops.CopyBufPartial(valueBuf, curValueOffset, valueAllBuf, curSeqLen * outEmbedDim, dstValueOffset + pastSeqLen * outEmbedDim);
                }
            }

            var dstSeqLen = pastSeqLen + curSeqLen;
            var scale = sp.scale == 0f ? 1f / Mathf.Sqrt(embedDim) : sp.scale;
            if (dstSeqLen > 4096)
                throw new InvalidOperationException("SDPA dst_seqlen exceeds current repro shader limit 4096: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(3, outEmbedDim, srcSeqLen, 1, numHeads);
            var canUseFastPath = !sp.kvCache && ResolveSdpaFastPathEnabled();
            if (canUseFastPath)
            {
                owner.Ops.SdpaAttentionFast(
                    queryBuf,
                    keyAllBuf,
                    valueAllBuf,
                    null,
                    srcSeqLen,
                    dstSeqLen,
                    embedDim,
                    outEmbedDim,
                    numHeads,
                    numGroup,
                    0,
                    0,
                    0,
                    0,
                    scale,
                    outTensor.buffer);
            }
            else
            {
                var scoreBuf = owner.RentTempBuffer(numHeads * srcSeqLen * dstSeqLen, sizeof(float));
                tempOwned.Add(scoreBuf);

                owner.Ops.SdpaQkBuf(
                    queryBuf,
                    keyAllBuf,
                    maskBuf,
                    srcSeqLen,
                    dstSeqLen,
                    embedDim,
                    numHeads,
                    numGroup,
                    attnMaskView?.dims ?? 0,
                    attnMaskView?.w ?? 0,
                    attnMaskView?.h ?? 0,
                    attnMaskView?.c ?? 0,
                    scale,
                    scoreBuf);
                owner.Ops.SdpaSoftmaxBuf(scoreBuf, srcSeqLen, dstSeqLen, numHeads);
                owner.Ops.SdpaQkvBuf(scoreBuf, valueAllBuf, srcSeqLen, dstSeqLen, outEmbedDim, numHeads, numGroup, outTensor.buffer);
            }

            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: true,
                textureBlobs,
                textureShapes,
                bufferBlobs,
                bufferRefs,
                bufferViews,
                tempOwned);

            if (sp.kvCache && layer.topNames.Length >= 3)
            {
                PublishTensor(owner, layer.topNames[1], 3, embedDim, dstSeqLen, 1, numGroup, keyAllBuf, true, textureBlobs, textureShapes, bufferBlobs, bufferRefs, bufferViews, tempOwned);
                PublishTensor(owner, layer.topNames[2], 3, outEmbedDim, dstSeqLen, 1, numGroup, valueAllBuf, true, textureBlobs, textureShapes, bufferBlobs, bufferRefs, bufferViews, tempOwned);
            }

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (TryExecuteRenderTexturePath(owner, layer, context))
                return;

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (TryExecuteCommandBufferTexturePath(owner, layer, context))
                return;

            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.SdpaPack sp)
                throw new InvalidOperationException("SDPA pack not found: " + layer.name);

            var queryShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var keyShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[1]);
            var valueShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[2]);
            if (queryShape.dims != 3 || keyShape.dims != 3 || valueShape.dims != 3)
                throw new InvalidOperationException("SDPA expects dims=3 tensors: " + layer.name);

            var dstSeqLen = keyShape.h;
            if (sp.kvCache)
            {
                var pastKeyBottom = sp.attnMask ? 4 : 3;
                var pastKeyShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[pastKeyBottom]);
                dstSeqLen += pastKeyShape.h;
            }

            var outShape = new NcnnRepro.BufferShape(3, valueShape.w, queryShape.h, 1, queryShape.c);
            owner.PublishCmdPlaceholder(cmd, layer.topNames[0], outShape, blobs, shapes);
            if (sp.kvCache && layer.topNames.Length >= 3)
            {
                owner.PublishCmdPlaceholder(
                    cmd,
                    layer.topNames[1],
                    new NcnnRepro.BufferShape(3, keyShape.w, dstSeqLen, 1, keyShape.c),
                    blobs,
                    shapes);
                owner.PublishCmdPlaceholder(
                    cmd,
                    layer.topNames[2],
                    new NcnnRepro.BufferShape(3, valueShape.w, dstSeqLen, 1, valueShape.c),
                    blobs,
                    shapes);
            }
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static void PublishTensor(
            NcnnRepro owner,
            string topName,
            int dims,
            int w,
            int h,
            int d,
            int c,
            ComputeBuffer buffer,
            bool preferTexture,
            System.Collections.Generic.Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            System.Collections.Generic.Dictionary<string, ComputeBuffer> bufferBlobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferRef> bufferRefs,
            System.Collections.Generic.Dictionary<string, NcnnTensorBuffer> bufferViews,
            System.Collections.Generic.List<IDisposable> tempOwned)
        {
            var tensor = owner.RentTempTensorBuffer(dims, w, h, d, c);
            owner.Ops.CopyBuf(buffer, tensor.buffer, tensor.buffer.count);
            owner.PublishTensorBufferOutput(topName, tensor, preferTexture, textureBlobs, textureShapes, bufferBlobs, bufferRefs, bufferViews, tempOwned);
        }

        private static bool TryExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (!owner.EnableAttentionMatMulPack4Specializations)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (!TryResolveRtPlan(owner, layer, context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out var plan))
                return false;

            RenderTexture queryScaled = null;
            RenderTexture keyTransposed = null;
            RenderTexture scores = null;
            RenderTexture output = null;

            try
            {
                queryScaled = owner.RentTempArray(plan.queryStorageShape.w, plan.queryStorageShape.h, plan.querySlices, plan.pack4TextureFormat);
                owner.Ops.BinaryOpScalarPack4(plan.query.texture, plan.scale, plan.querySlices, 2, queryScaled);

                keyTransposed = owner.RentTempArray(plan.keyTransposedStorageShape.w, plan.keyTransposedStorageShape.h, plan.keyTransposedSlices, plan.pack4TextureFormat);
                owner.Ops.PermutePack4Cdhw(
                    plan.key.texture,
                    plan.keyShape.w,
                    plan.keyShape.h,
                    plan.keyShape.d,
                    plan.keyShape.c,
                    new Vector4Int(1, 0, 2, 3),
                    plan.keyTransposedShape.w,
                    plan.keyTransposedShape.h,
                    plan.keyTransposedShape.d,
                    plan.keyTransposedShape.c,
                    keyTransposed);

                scores = owner.RentTempArray(plan.scoresStorageShape.w, plan.scoresStorageShape.h, plan.scoresSlices, plan.pack4TextureFormat);
                owner.Ops.MatMulPack4Cdhw(
                    queryScaled,
                    plan.queryShape.h,
                    plan.queryShape.w,
                    plan.queryShape.d,
                    plan.queryShape.c,
                    keyTransposed,
                    plan.keyTransposedShape.h,
                    plan.keyTransposedShape.w,
                    plan.keyTransposedShape.d,
                    plan.keyTransposedShape.c,
                    false,
                    plan.scoresShape.d,
                    plan.scoresShape.c,
                    scores);

                ReturnTemp(owner, ref keyTransposed);
                ReturnTemp(owner, ref queryScaled);

                output = owner.RentTempArray(plan.outputStorageShape.w, plan.outputStorageShape.h, plan.outputSlices, plan.pack4TextureFormat);
                owner.Ops.SoftmaxPack4Cdhw(scores, plan.scoresShape.w, plan.scoresShape.h, plan.scoresShape.d, plan.scoresShape.c, scores);
                owner.Ops.MatMulPack4Cdhw(
                    scores,
                    plan.scoresShape.h,
                    plan.scoresShape.w,
                    plan.scoresShape.d,
                    plan.scoresShape.c,
                    plan.value.texture,
                    plan.valueShape.h,
                    plan.valueShape.w,
                    plan.valueShape.d,
                    plan.valueShape.c,
                    false,
                    plan.outputShape.d,
                    plan.outputShape.c,
                    output);

                ReturnTemp(owner, ref scores);

                NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, plan.outputShape, plan.outputStorageShape);
                output = null;

                owner.Consume(
                    context.textureBlobs,
                    context.bufferBlobs,
                    context.bufferRefs,
                    context.bufferViews,
                    context.remaining,
                    layer.bottomNames,
                    context.pinnedNames);
                return true;
            }
            finally
            {
                ReturnTemp(owner, ref queryScaled);
                ReturnTemp(owner, ref keyTransposed);
                ReturnTemp(owner, ref scores);
                ReturnTemp(owner, ref output);
            }
        }

        private static bool TryExecuteCommandBufferTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (!owner.EnableAttentionMatMulPack4Specializations)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (!TryResolveCmdRtPlan(owner, layer, context.blobs, context.shapes, out var plan))
                return false;

            var cmd = context.commandBuffer;
            ComputeTexture queryScaled = null;
            ComputeTexture keyTransposed = null;
            ComputeTexture scores = null;
            ComputeTexture output = null;

            try
            {
                queryScaled = owner.RentTempArray(cmd, plan.queryStorageShape.w, plan.queryStorageShape.h, plan.querySlices, plan.pack4TextureFormat);
                owner.Ops.BinaryOpScalarPack4(cmd, plan.query.texture, plan.scale, plan.querySlices, 2, queryScaled);

                keyTransposed = owner.RentTempArray(cmd, plan.keyTransposedStorageShape.w, plan.keyTransposedStorageShape.h, plan.keyTransposedSlices, plan.pack4TextureFormat);
                owner.Ops.PermutePack4Cdhw(
                    cmd,
                    plan.key.texture,
                    plan.keyShape.w,
                    plan.keyShape.h,
                    plan.keyShape.d,
                    plan.keyShape.c,
                    new Vector4Int(1, 0, 2, 3),
                    plan.keyTransposedShape.w,
                    plan.keyTransposedShape.h,
                    plan.keyTransposedShape.d,
                    plan.keyTransposedShape.c,
                    keyTransposed);

                scores = owner.RentTempArray(cmd, plan.scoresStorageShape.w, plan.scoresStorageShape.h, plan.scoresSlices, plan.pack4TextureFormat);
                owner.Ops.MatMulPack4Cdhw(
                    cmd,
                    queryScaled,
                    plan.queryShape.h,
                    plan.queryShape.w,
                    plan.queryShape.d,
                    plan.queryShape.c,
                    keyTransposed,
                    plan.keyTransposedShape.h,
                    plan.keyTransposedShape.w,
                    plan.keyTransposedShape.d,
                    plan.keyTransposedShape.c,
                    false,
                    plan.scoresShape.d,
                    plan.scoresShape.c,
                    scores);

                owner.ReturnTempArray(cmd, keyTransposed);
                keyTransposed = null;
                owner.ReturnTempArray(cmd, queryScaled);
                queryScaled = null;

                output = owner.RentTempArray(cmd, plan.outputStorageShape.w, plan.outputStorageShape.h, plan.outputSlices, plan.pack4TextureFormat);
                owner.Ops.SoftmaxPack4Cdhw(cmd, scores, plan.scoresShape.w, plan.scoresShape.h, plan.scoresShape.d, plan.scoresShape.c, scores);
                owner.Ops.MatMulPack4Cdhw(
                    cmd,
                    scores,
                    plan.scoresShape.h,
                    plan.scoresShape.w,
                    plan.scoresShape.d,
                    plan.scoresShape.c,
                    plan.value.texture,
                    plan.valueShape.h,
                    plan.valueShape.w,
                    plan.valueShape.d,
                    plan.valueShape.c,
                    false,
                    plan.outputShape.d,
                    plan.outputShape.c,
                    output);

                owner.ReturnTempArray(cmd, scores);
                scores = null;

                context.blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                {
                    texture = output,
                    width = plan.outputStorageShape.w,
                    height = plan.outputStorageShape.h,
                    packs = plan.outputPacks,
                    refs = 1,
                    owned = true,
                    hasLogicalShape = true,
                    logicalShape = plan.outputShape,
                    hasStorageShape = true,
                    storageShape = plan.outputStorageShape
                };
                context.shapes[layer.topNames[0]] = plan.outputShape;
                output = null;

                owner.ConsumeCmd(cmd, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return true;
            }
            finally
            {
                ReturnTemp(owner, cmd, ref queryScaled);
                ReturnTemp(owner, cmd, ref keyTransposed);
                ReturnTemp(owner, cmd, ref scores);
                ReturnTemp(owner, cmd, ref output);
            }
        }

        private static bool TryResolveRtPlan(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            System.Collections.Generic.Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            System.Collections.Generic.Dictionary<string, ComputeBuffer> bufferBlobs,
            System.Collections.Generic.Dictionary<string, NcnnTensorBuffer> bufferViews,
            out SdpaRtPlan plan)
        {
            plan = default;
            if (!TryResolveCommonPlan(owner, layer, out var common))
                return false;
            if (common.sp.attnMask || common.sp.kvCache || common.sp.int8ScaleTerm)
                return false;
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var query, out var queryShape)
                || !owner.TryGetPack4Texture(layer.bottomNames[1], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var key, out var keyShape)
                || !owner.TryGetPack4Texture(layer.bottomNames[2], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var value, out var valueShape))
            {
                return false;
            }

            if (!TryValidateTextureInputs(query, queryShape, key, keyShape, value, valueShape, common, out var outputShape, out var outputStorageShape))
                return false;

            plan = new SdpaRtPlan(
                common.scale,
                query,
                queryShape,
                key,
                keyShape,
                value,
                valueShape,
                outputShape,
                outputStorageShape);
            return true;
        }

        private static bool TryResolveCmdRtPlan(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            System.Collections.Generic.Dictionary<string, NcnnRepro.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> shapes,
            out SdpaCmdRtPlan plan)
        {
            plan = default;
            if (!TryResolveCommonPlan(owner, layer, out var common))
                return false;
            if (common.sp.attnMask || common.sp.kvCache || common.sp.int8ScaleTerm)
                return false;
            if (!TryGetExistingCmdTexture(blobs, shapes, layer.bottomNames[0], out var query, out var queryShape)
                || !TryGetExistingCmdTexture(blobs, shapes, layer.bottomNames[1], out var key, out var keyShape)
                || !TryGetExistingCmdTexture(blobs, shapes, layer.bottomNames[2], out var value, out var valueShape))
            {
                return false;
            }

            if (!TryValidateTextureInputs(query, queryShape, key, keyShape, value, valueShape, common, out var outputShape, out var outputStorageShape))
                return false;

            plan = new SdpaCmdRtPlan(
                common.scale,
                query,
                queryShape,
                key,
                keyShape,
                value,
                valueShape,
                outputShape,
                outputStorageShape);
            return true;
        }

        private static bool TryResolveCommonPlan(NcnnRepro owner, NcnnParamModel.Layer layer, out SdpaCommonPlan plan)
        {
            plan = default;
            if (owner == null || layer == null)
                return false;
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.SdpaPack sp)
                return false;

            plan = new SdpaCommonPlan(sp);
            return true;
        }

        private static bool TryValidateTextureInputs(
            object queryTex,
            NcnnRepro.BufferShape queryShape,
            object keyTex,
            NcnnRepro.BufferShape keyShape,
            object valueTex,
            NcnnRepro.BufferShape valueShape,
            in SdpaCommonPlan common,
            out NcnnRepro.BufferShape outputShape,
            out NcnnRepro.BufferShape outputStorageShape)
        {
            outputShape = default;
            outputStorageShape = default;
            if (queryTex == null || keyTex == null || valueTex == null)
                return false;
            if (queryShape.dims != 3 || keyShape.dims != 3 || valueShape.dims != 3)
                return false;
            if (queryShape.d != 1 || keyShape.d != 1 || valueShape.d != 1)
                return false;
            if (queryShape.w <= 0 || queryShape.h <= 0 || queryShape.c <= 0)
                return false;
            if (keyShape.w <= 0 || keyShape.h <= 0 || keyShape.c <= 0 || valueShape.w <= 0 || valueShape.h <= 0 || valueShape.c <= 0)
                return false;
            if (queryShape.w != keyShape.w)
                return false;
            if (keyShape.h != valueShape.h)
                return false;
            if (queryShape.c != valueShape.c)
                return false;
            if ((queryShape.c % keyShape.c) != 0)
                return false;
            if (keyShape.h > 4096)
                return false;

            outputShape = new NcnnRepro.BufferShape(3, valueShape.w, queryShape.h, 1, queryShape.c);
            outputStorageShape = outputShape;
            return true;
        }

        private static bool TryGetExistingCmdTexture(
            System.Collections.Generic.Dictionary<string, NcnnRepro.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> shapes,
            string blobName,
            out NcnnRepro.CmdTensorRef texture,
            out NcnnRepro.BufferShape shape)
        {
            texture = null;
            shape = default;
            if (blobs == null || string.IsNullOrWhiteSpace(blobName))
                return false;
            if (!blobs.TryGetValue(blobName, out texture) || texture == null || texture.texture == null)
            {
                texture = null;
                return false;
            }

            shape = NcnnRepro.GetCmdShape(shapes, blobs, blobName);
            return shape.dims >= 1 && shape.dims <= 4;
        }

        private static void ReturnTemp(NcnnRepro owner, ref RenderTexture texture)
        {
            if (owner == null || texture == null)
                return;
            owner.ReturnTempArray(texture);
            texture = null;
        }

        private static void ReturnTemp(NcnnRepro owner, CommandBuffer cmd, ref ComputeTexture texture)
        {
            if (owner == null || cmd == null || texture == null)
                return;
            owner.ReturnTempArray(cmd, texture);
            texture = null;
        }

        private readonly struct SdpaCommonPlan
        {
            public readonly NcnnRepro.SdpaPack sp;
            public readonly float scale;

            public SdpaCommonPlan(NcnnRepro.SdpaPack sp)
            {
                this.sp = sp;
                scale = sp.scale;
            }
        }

        private readonly struct SdpaRtPlan
        {
            public readonly float scale;
            public readonly NcnnRepro.TensorRef query;
            public readonly NcnnRepro.BufferShape queryShape;
            public readonly NcnnRepro.BufferShape queryStorageShape;
            public readonly int querySlices;
            public readonly NcnnRepro.TensorRef key;
            public readonly NcnnRepro.BufferShape keyShape;
            public readonly NcnnRepro.TensorRef value;
            public readonly NcnnRepro.BufferShape valueShape;
            public readonly NcnnRepro.BufferShape keyTransposedShape;
            public readonly NcnnRepro.BufferShape keyTransposedStorageShape;
            public readonly int keyTransposedSlices;
            public readonly NcnnRepro.BufferShape scoresShape;
            public readonly NcnnRepro.BufferShape scoresStorageShape;
            public readonly int scoresSlices;
            public readonly NcnnRepro.BufferShape outputShape;
            public readonly NcnnRepro.BufferShape outputStorageShape;
            public readonly int outputPacks;
            public readonly int outputSlices;
            public readonly RenderTextureFormat pack4TextureFormat;

            public SdpaRtPlan(
                float scale,
                NcnnRepro.TensorRef query,
                NcnnRepro.BufferShape queryShape,
                NcnnRepro.TensorRef key,
                NcnnRepro.BufferShape keyShape,
                NcnnRepro.TensorRef value,
                NcnnRepro.BufferShape valueShape,
                NcnnRepro.BufferShape outputShape,
                NcnnRepro.BufferShape outputStorageShape)
            {
                this.scale = Mathf.Approximately(scale, 0f) ? 1f / Mathf.Sqrt(Mathf.Max(1, queryShape.w)) : scale;
                this.query = query;
                this.queryShape = queryShape;
                queryStorageShape = queryShape;
                querySlices = Mathf.Max(1, queryShape.d * Mathf.CeilToInt(queryShape.c / 4f));
                this.key = key;
                this.keyShape = keyShape;
                this.value = value;
                this.valueShape = valueShape;
                keyTransposedShape = new NcnnRepro.BufferShape(4, keyShape.h, keyShape.w, Mathf.Max(1, keyShape.d), keyShape.c);
                keyTransposedStorageShape = keyTransposedShape;
                keyTransposedSlices = Mathf.Max(1, keyTransposedShape.d * Mathf.CeilToInt(keyTransposedShape.c / 4f));
                scoresShape = new NcnnRepro.BufferShape(4, keyShape.h, queryShape.h, Mathf.Max(1, queryShape.d), queryShape.c);
                scoresStorageShape = scoresShape;
                scoresSlices = Mathf.Max(1, scoresShape.d * Mathf.CeilToInt(scoresShape.c / 4f));
                this.outputShape = outputShape;
                this.outputStorageShape = outputStorageShape;
                outputPacks = Mathf.Max(1, Mathf.CeilToInt(outputStorageShape.c / 4f));
                outputSlices = Mathf.Max(1, outputStorageShape.d * outputPacks);
                pack4TextureFormat = NcnnRepro.ResolveTensorTextureFormat(4);
            }
        }

        private readonly struct SdpaCmdRtPlan
        {
            public readonly float scale;
            public readonly NcnnRepro.CmdTensorRef query;
            public readonly NcnnRepro.BufferShape queryShape;
            public readonly NcnnRepro.BufferShape queryStorageShape;
            public readonly int querySlices;
            public readonly NcnnRepro.CmdTensorRef key;
            public readonly NcnnRepro.BufferShape keyShape;
            public readonly NcnnRepro.CmdTensorRef value;
            public readonly NcnnRepro.BufferShape valueShape;
            public readonly NcnnRepro.BufferShape keyTransposedShape;
            public readonly NcnnRepro.BufferShape keyTransposedStorageShape;
            public readonly int keyTransposedSlices;
            public readonly NcnnRepro.BufferShape scoresShape;
            public readonly NcnnRepro.BufferShape scoresStorageShape;
            public readonly int scoresSlices;
            public readonly NcnnRepro.BufferShape outputShape;
            public readonly NcnnRepro.BufferShape outputStorageShape;
            public readonly int outputPacks;
            public readonly int outputSlices;
            public readonly RenderTextureFormat pack4TextureFormat;

            public SdpaCmdRtPlan(
                float scale,
                NcnnRepro.CmdTensorRef query,
                NcnnRepro.BufferShape queryShape,
                NcnnRepro.CmdTensorRef key,
                NcnnRepro.BufferShape keyShape,
                NcnnRepro.CmdTensorRef value,
                NcnnRepro.BufferShape valueShape,
                NcnnRepro.BufferShape outputShape,
                NcnnRepro.BufferShape outputStorageShape)
            {
                this.scale = Mathf.Approximately(scale, 0f) ? 1f / Mathf.Sqrt(Mathf.Max(1, queryShape.w)) : scale;
                this.query = query;
                this.queryShape = queryShape;
                queryStorageShape = queryShape;
                querySlices = Mathf.Max(1, queryShape.d * Mathf.CeilToInt(queryShape.c / 4f));
                this.key = key;
                this.keyShape = keyShape;
                this.value = value;
                this.valueShape = valueShape;
                keyTransposedShape = new NcnnRepro.BufferShape(4, keyShape.h, keyShape.w, Mathf.Max(1, keyShape.d), keyShape.c);
                keyTransposedStorageShape = keyTransposedShape;
                keyTransposedSlices = Mathf.Max(1, keyTransposedShape.d * Mathf.CeilToInt(keyTransposedShape.c / 4f));
                scoresShape = new NcnnRepro.BufferShape(4, keyShape.h, queryShape.h, Mathf.Max(1, queryShape.d), queryShape.c);
                scoresStorageShape = scoresShape;
                scoresSlices = Mathf.Max(1, scoresShape.d * Mathf.CeilToInt(scoresShape.c / 4f));
                this.outputShape = outputShape;
                this.outputStorageShape = outputStorageShape;
                outputPacks = Mathf.Max(1, Mathf.CeilToInt(outputStorageShape.c / 4f));
                outputSlices = Mathf.Max(1, outputStorageShape.d * outputPacks);
                pack4TextureFormat = NcnnRepro.ResolveTensorTextureFormat(4);
            }
        }

        private static bool ResolveSdpaFastPathEnabled()
        {
            try
            {
                var env = Environment.GetEnvironmentVariable("AIIMAGE_SD_SDPA_FASTPATH");
                if (string.IsNullOrWhiteSpace(env))
                    return true;

                env = env.Trim();
                if (string.Equals(env, "0", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(env, "false", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(env, "off", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(env, "no", StringComparison.OrdinalIgnoreCase))
                    return false;

                if (string.Equals(env, "1", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(env, "true", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(env, "on", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(env, "yes", StringComparison.OrdinalIgnoreCase))
                    return true;

                if (float.TryParse(env, NumberStyles.Float, CultureInfo.InvariantCulture, out var numeric))
                    return numeric > 0f;
            }
            catch
            {
            }

            return true;
        }
    }
}
