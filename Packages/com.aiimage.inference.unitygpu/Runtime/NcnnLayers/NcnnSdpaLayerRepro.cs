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
                causal = layer.GetInt(8, 0) != 0,
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

            if (owner != null && owner.StrictTextureInference)
                throw new InvalidOperationException("SDPA texture execution plan rejected; refusing ComputeBuffer fallback: layer=" + (layer?.name ?? string.Empty));

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

            if (sp.kvCache)
                throw new NotSupportedException("CommandBuffer SDPA kv-cache is not implemented"
                    + " | layer=" + layer.name
                    + " | rejectedFallback=buffer-materialization");

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
            throw new NotSupportedException("CommandBuffer SDPA requires Pack4 Q/K/V, an optional Pack4 scalar mask, and no kv-cache"
                + " | layer=" + layer.name
                + " | q=d" + queryShape.dims + ":" + queryShape.w + "x" + queryShape.h + "x" + queryShape.d + "x" + queryShape.c
                + " | k=d" + keyShape.dims + ":" + keyShape.w + "x" + keyShape.h + "x" + keyShape.d + "x" + keyShape.c
                + " | v=d" + valueShape.dims + ":" + valueShape.w + "x" + valueShape.h + "x" + valueShape.d + "x" + valueShape.c
                + " | rejectedFallback=placeholder-or-buffer-materialization");
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
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (!TryResolveRtPlan(owner, layer, context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out var plan))
                return false;

            owner.DebugLog?.Invoke(
                "[Pack4SDPA] layer=" + layer.name
                + " | q=d" + plan.queryShape.dims + ":" + plan.queryShape.w + "x" + plan.queryShape.h + "x" + plan.queryShape.d + "x" + plan.queryShape.c
                + " | k=d" + plan.keyShape.dims + ":" + plan.keyShape.w + "x" + plan.keyShape.h + "x" + plan.keyShape.d + "x" + plan.keyShape.c
                + " | v=d" + plan.valueShape.dims + ":" + plan.valueShape.w + "x" + plan.valueShape.h + "x" + plan.valueShape.d + "x" + plan.valueShape.c
                + " | scale=" + plan.scale.ToString("G9", CultureInfo.InvariantCulture));

            RenderTexture queryScaled = null;
            RenderTexture keyTransposed = null;
            RenderTexture scores = null;
            RenderTexture weights = null;
            RenderTexture output = null;
            RenderTexture keyCache = null;
            RenderTexture valueCache = null;

            try
            {
                var keyInput = plan.key.texture;
                var valueInput = plan.value.texture;
                if (plan.hasPastCache)
                {
                    var keySlices = Mathf.Max(1, plan.keyShape.d * Mathf.CeilToInt(plan.keyShape.c / 4f));
                    var valueSlices = Mathf.Max(1, plan.valueShape.d * Mathf.CeilToInt(plan.valueShape.c / 4f));
                    var keyStorageHeight = Mathf.Max(plan.keyShape.h, owner.AttentionKvCacheTextureCapacity);
                    var valueStorageHeight = Mathf.Max(plan.valueShape.h, owner.AttentionKvCacheTextureCapacity);
                    keyCache = owner.RentTempArray(plan.keyShape.w, keyStorageHeight, keySlices, plan.pack4TextureFormat);
                    valueCache = owner.RentTempArray(plan.valueShape.w, valueStorageHeight, valueSlices, plan.pack4TextureFormat);
                    owner.Ops.ConcatSequencePack4Cdhw(plan.pastKey.texture, plan.key.texture, plan.pastKeyShape.h, plan.keyCurrentShape.h, keyCache);
                    owner.Ops.ConcatSequencePack4Cdhw(plan.pastValue.texture, plan.value.texture, plan.pastValueShape.h, plan.valueCurrentShape.h, valueCache);
                    keyInput = keyCache;
                    valueInput = valueCache;
                }

                if (ResolveSdpaFastPathEnabled() || plan.hasAttnMask || plan.causal)
                {
                    output = owner.RentTempArray(plan.outputStorageShape.w, plan.outputStorageShape.h, plan.outputSlices, plan.pack4TextureFormat);
                    owner.Ops.SdpaAttentionPack4Cdhw(
                        plan.query.texture,
                        keyInput,
                        valueInput,
                        plan.queryShape.h,
                        plan.keyShape.h,
                        plan.queryShape.w,
                        plan.valueShape.w,
                        plan.queryShape.c,
                        plan.keyShape.c,
                        plan.scale,
                        output,
                        plan.attnMask != null ? plan.attnMask.texture : null,
                        plan.causal);

                    NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, plan.outputShape, plan.outputStorageShape);
                    output = null;
                    PublishKvCache(owner, layer, context, plan, ref keyCache, ref valueCache);

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

                queryScaled = owner.RentTempArray(plan.queryStorageShape.w, plan.queryStorageShape.h, plan.querySlices, plan.pack4TextureFormat);
                owner.Ops.BinaryOpScalarPack4(plan.query.texture, plan.scale, plan.querySlices, 2, queryScaled);

                keyTransposed = owner.RentTempArray(plan.keyTransposedStorageShape.w, plan.keyTransposedStorageShape.h, plan.keyTransposedSlices, plan.pack4TextureFormat);
                owner.Ops.PermutePack4Cdhw(
                    keyInput,
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

                weights = owner.RentTempArray(plan.scoresStorageShape.w, plan.scoresStorageShape.h, plan.scoresSlices, plan.pack4TextureFormat);
                owner.Ops.SoftmaxPack4Cdhw(scores, plan.scoresShape.w, plan.scoresShape.h, plan.scoresShape.d, plan.scoresShape.c, weights);
                ReturnTemp(owner, ref scores);

                output = owner.RentTempArray(plan.outputStorageShape.w, plan.outputStorageShape.h, plan.outputSlices, plan.pack4TextureFormat);
                owner.Ops.MatMulPack4Cdhw(
                    weights,
                    plan.scoresShape.h,
                    plan.scoresShape.w,
                    plan.scoresShape.d,
                    plan.scoresShape.c,
                    valueInput,
                    plan.valueShape.h,
                    plan.valueShape.w,
                    plan.valueShape.d,
                    plan.valueShape.c,
                    false,
                    plan.outputShape.d,
                    plan.outputShape.c,
                    output);

                ReturnTemp(owner, ref weights);

                NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, plan.outputShape, plan.outputStorageShape);
                output = null;
                PublishKvCache(owner, layer, context, plan, ref keyCache, ref valueCache);

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
                ReturnTemp(owner, ref weights);
                ReturnTemp(owner, ref output);
                ReturnTemp(owner, ref keyCache);
                ReturnTemp(owner, ref valueCache);
            }
        }

        private static bool TryExecuteCommandBufferTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (!TryResolveCmdRtPlan(owner, layer, context.blobs, context.shapes, out var plan))
                return false;

            var cmd = context.commandBuffer;
            ComputeTexture queryScaled = null;
            ComputeTexture keyTransposed = null;
            ComputeTexture scores = null;
            ComputeTexture weights = null;
            ComputeTexture output = null;
            ComputeTexture keyCache = null;
            ComputeTexture valueCache = null;

            try
            {
                var keyInput = plan.key.texture;
                var valueInput = plan.value.texture;
                if (plan.hasPastCache)
                {
                    var keySlices = Mathf.Max(1, plan.keyShape.d * Mathf.CeilToInt(plan.keyShape.c / 4f));
                    var valueSlices = Mathf.Max(1, plan.valueShape.d * Mathf.CeilToInt(plan.valueShape.c / 4f));
                    var keyStorageHeight = Mathf.Max(plan.keyShape.h, owner.AttentionKvCacheTextureCapacity);
                    var valueStorageHeight = Mathf.Max(plan.valueShape.h, owner.AttentionKvCacheTextureCapacity);
                    keyCache = owner.RentTempArray(cmd, plan.keyShape.w, keyStorageHeight, keySlices, plan.pack4TextureFormat);
                    valueCache = owner.RentTempArray(cmd, plan.valueShape.w, valueStorageHeight, valueSlices, plan.pack4TextureFormat);
                    owner.Ops.ConcatSequencePack4Cdhw(cmd, plan.pastKey.texture, plan.key.texture, plan.pastKeyShape.h, plan.keyCurrentShape.h, keyCache);
                    owner.Ops.ConcatSequencePack4Cdhw(cmd, plan.pastValue.texture, plan.value.texture, plan.pastValueShape.h, plan.valueCurrentShape.h, valueCache);
                    keyInput = keyCache;
                    valueInput = valueCache;
                }

                if (ResolveSdpaFastPathEnabled() || plan.hasAttnMask || plan.causal)
                {
                    output = owner.RentTempArray(cmd, plan.outputStorageShape.w, plan.outputStorageShape.h, plan.outputSlices, plan.pack4TextureFormat);
                    owner.Ops.SdpaAttentionPack4Cdhw(
                        cmd,
                        plan.query.texture,
                        keyInput,
                        valueInput,
                        plan.queryShape.h,
                        plan.keyShape.h,
                        plan.queryShape.w,
                        plan.valueShape.w,
                        plan.queryShape.c,
                        plan.keyShape.c,
                        plan.scale,
                        output,
                        plan.attnMask != null ? plan.attnMask.texture : null,
                        plan.causal);

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
                    PublishKvCache(owner, layer, context, plan, ref keyCache, ref valueCache);

                    owner.ConsumeCmd(cmd, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                    return true;
                }

                queryScaled = owner.RentTempArray(cmd, plan.queryStorageShape.w, plan.queryStorageShape.h, plan.querySlices, plan.pack4TextureFormat);
                owner.Ops.BinaryOpScalarPack4(cmd, plan.query.texture, plan.scale, plan.querySlices, 2, queryScaled);

                keyTransposed = owner.RentTempArray(cmd, plan.keyTransposedStorageShape.w, plan.keyTransposedStorageShape.h, plan.keyTransposedSlices, plan.pack4TextureFormat);
                owner.Ops.PermutePack4Cdhw(
                    cmd,
                    keyInput,
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

                weights = owner.RentTempArray(cmd, plan.scoresStorageShape.w, plan.scoresStorageShape.h, plan.scoresSlices, plan.pack4TextureFormat);
                owner.Ops.SoftmaxPack4Cdhw(cmd, scores, plan.scoresShape.w, plan.scoresShape.h, plan.scoresShape.d, plan.scoresShape.c, weights);
                owner.ReturnTempArray(cmd, scores);
                scores = null;

                output = owner.RentTempArray(cmd, plan.outputStorageShape.w, plan.outputStorageShape.h, plan.outputSlices, plan.pack4TextureFormat);
                owner.Ops.MatMulPack4Cdhw(
                    cmd,
                    weights,
                    plan.scoresShape.h,
                    plan.scoresShape.w,
                    plan.scoresShape.d,
                    plan.scoresShape.c,
                    valueInput,
                    plan.valueShape.h,
                    plan.valueShape.w,
                    plan.valueShape.d,
                    plan.valueShape.c,
                    false,
                    plan.outputShape.d,
                    plan.outputShape.c,
                    output);

                owner.ReturnTempArray(cmd, weights);
                weights = null;

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
                PublishKvCache(owner, layer, context, plan, ref keyCache, ref valueCache);

                owner.ConsumeCmd(cmd, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return true;
            }
            finally
            {
                ReturnTemp(owner, cmd, ref queryScaled);
                ReturnTemp(owner, cmd, ref keyTransposed);
                ReturnTemp(owner, cmd, ref scores);
                ReturnTemp(owner, cmd, ref weights);
                ReturnTemp(owner, cmd, ref output);
                ReturnTemp(owner, cmd, ref keyCache);
                ReturnTemp(owner, cmd, ref valueCache);
            }
        }

        private static void PublishKvCache(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnLayerBufferContext context,
            in SdpaRtPlan plan,
            ref RenderTexture keyCache,
            ref RenderTexture valueCache)
        {
            if (layer.topNames == null || layer.topNames.Length < 3)
                return;

            if (plan.hasPastCache)
            {
                var keyCapacityStorageShape = new NcnnRepro.BufferShape(3, plan.keyShape.w, keyCache.height, plan.keyShape.d, plan.keyShape.c);
                var valueCapacityStorageShape = new NcnnRepro.BufferShape(3, plan.valueShape.w, valueCache.height, plan.valueShape.d, plan.valueShape.c);
                NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[1], keyCache, plan.keyShape, keyCapacityStorageShape);
                NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[2], valueCache, plan.valueShape, valueCapacityStorageShape);
                keyCache = null;
                valueCache = null;
                return;
            }

            var keyStorageShape = NcnnRepro.GetTextureStorageShape(plan.key, plan.keyCurrentShape);
            var valueStorageShape = NcnnRepro.GetTextureStorageShape(plan.value, plan.valueCurrentShape);
            context.textureBlobs[layer.topNames[1]] = NcnnRepro.CreateTextureAlias(plan.key, plan.keyShape, keyStorageShape);
            context.textureBlobs[layer.topNames[2]] = NcnnRepro.CreateTextureAlias(plan.value, plan.valueShape, valueStorageShape);
            context.textureShapes[layer.topNames[1]] = plan.keyShape;
            context.textureShapes[layer.topNames[2]] = plan.valueShape;
        }

        private static void PublishKvCache(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnLayerCommandBufferContext context,
            in SdpaCmdRtPlan plan,
            ref ComputeTexture keyCache,
            ref ComputeTexture valueCache)
        {
            if (layer.topNames == null || layer.topNames.Length < 3)
                return;

            if (plan.hasPastCache)
            {
                var keyCapacityStorageShape = new NcnnRepro.BufferShape(3, plan.keyShape.w, keyCache.height, plan.keyShape.d, plan.keyShape.c);
                var valueCapacityStorageShape = new NcnnRepro.BufferShape(3, plan.valueShape.w, valueCache.height, plan.valueShape.d, plan.valueShape.c);
                context.blobs[layer.topNames[1]] = NcnnRepro.CreateCmdTensorRef(keyCache, plan.keyShape, keyCapacityStorageShape, owned: true);
                context.blobs[layer.topNames[2]] = NcnnRepro.CreateCmdTensorRef(valueCache, plan.valueShape, valueCapacityStorageShape, owned: true);
                keyCache = null;
                valueCache = null;
            }
            else
            {
                var keyStorageShape = NcnnRepro.GetCmdStorageShape(plan.key, plan.keyCurrentShape);
                var valueStorageShape = NcnnRepro.GetCmdStorageShape(plan.value, plan.valueCurrentShape);
                context.blobs[layer.topNames[1]] = NcnnRepro.CreateCmdTensorAlias(plan.key, plan.keyShape, keyStorageShape);
                context.blobs[layer.topNames[2]] = NcnnRepro.CreateCmdTensorAlias(plan.value, plan.valueShape, valueStorageShape);
            }

            if (context.shapes != null)
            {
                context.shapes[layer.topNames[1]] = plan.keyShape;
                context.shapes[layer.topNames[2]] = plan.valueShape;
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
            if (common.sp.int8ScaleTerm)
                return false;
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var query, out var queryShape)
                || !owner.TryGetPack4Texture(layer.bottomNames[1], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var key, out var keyShape)
                || !owner.TryGetPack4Texture(layer.bottomNames[2], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var value, out var valueShape))
            {
                return false;
            }

            NcnnRepro.TensorRef pastKey = null;
            NcnnRepro.TensorRef pastValue = null;
            NcnnRepro.BufferShape pastKeyShape = default;
            NcnnRepro.BufferShape pastValueShape = default;
            var attentionKeyShape = keyShape;
            var attentionValueShape = valueShape;
            if (common.sp.kvCache)
            {
                var pastKeyBottom = common.sp.attnMask ? 4 : 3;
                var pastValueBottom = common.sp.attnMask ? 5 : 4;
                var hasPastKey = layer.bottomNames.Length > pastKeyBottom
                    && NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[pastKeyBottom], out pastKey, out pastKeyShape);
                var hasPastValue = layer.bottomNames.Length > pastValueBottom
                    && NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[pastValueBottom], out pastValue, out pastValueShape);
                if (hasPastKey != hasPastValue)
                    return false;
                if (hasPastKey)
                {
                    if (!TryValidatePastCache(keyShape, valueShape, pastKey, pastKeyShape, pastValue, pastValueShape))
                        return false;
                    attentionKeyShape = new NcnnRepro.BufferShape(3, keyShape.w, pastKeyShape.h + keyShape.h, 1, keyShape.c);
                    attentionValueShape = new NcnnRepro.BufferShape(3, valueShape.w, pastValueShape.h + valueShape.h, 1, valueShape.c);
                }
            }

            if (!TryValidateTextureInputs(query, queryShape, key, attentionKeyShape, value, attentionValueShape, common, out var outputShape, out var outputStorageShape))
                return false;

            NcnnRepro.TensorRef attnMask = null;
            NcnnRepro.BufferShape attnMaskShape = default;
            if (common.sp.attnMask)
            {
                if (layer.bottomNames.Length <= 3
                    || !owner.TryGetPack4Texture(layer.bottomNames[3], textureBlobs, textureShapes, bufferBlobs, bufferViews, out attnMask, out attnMaskShape)
                    || attnMaskShape.dims != 2
                    || attnMaskShape.w != attentionKeyShape.h
                    || attnMaskShape.h != queryShape.h
                    || attnMask.packs != 1
                    || NcnnRepro.IsStrictLinearMatTexture(attnMask))
                    return false;
            }

            plan = new SdpaRtPlan(
                common.scale,
                common.sp.causal,
                query,
                queryShape,
                key,
                keyShape,
                value,
                valueShape,
                pastKey,
                pastKeyShape,
                pastValue,
                pastValueShape,
                attentionKeyShape,
                attentionValueShape,
                attnMask,
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
            if (common.sp.int8ScaleTerm)
                return false;
            if (!TryGetExistingCmdTexture(blobs, shapes, layer.bottomNames[0], out var query, out var queryShape)
                || !TryGetExistingCmdTexture(blobs, shapes, layer.bottomNames[1], out var key, out var keyShape)
                || !TryGetExistingCmdTexture(blobs, shapes, layer.bottomNames[2], out var value, out var valueShape))
            {
                return false;
            }

            NcnnRepro.CmdTensorRef pastKey = null;
            NcnnRepro.CmdTensorRef pastValue = null;
            NcnnRepro.BufferShape pastKeyShape = default;
            NcnnRepro.BufferShape pastValueShape = default;
            var attentionKeyShape = keyShape;
            var attentionValueShape = valueShape;
            if (common.sp.kvCache)
            {
                var pastKeyBottom = common.sp.attnMask ? 4 : 3;
                var pastValueBottom = common.sp.attnMask ? 5 : 4;
                var hasPastKey = layer.bottomNames.Length > pastKeyBottom
                    && TryGetExistingCmdTexture(blobs, shapes, layer.bottomNames[pastKeyBottom], out pastKey, out pastKeyShape);
                var hasPastValue = layer.bottomNames.Length > pastValueBottom
                    && TryGetExistingCmdTexture(blobs, shapes, layer.bottomNames[pastValueBottom], out pastValue, out pastValueShape);
                if (hasPastKey != hasPastValue)
                    return false;
                if (hasPastKey)
                {
                    if (!TryValidatePastCache(keyShape, valueShape, pastKey, pastKeyShape, pastValue, pastValueShape))
                        return false;
                    attentionKeyShape = new NcnnRepro.BufferShape(3, keyShape.w, pastKeyShape.h + keyShape.h, 1, keyShape.c);
                    attentionValueShape = new NcnnRepro.BufferShape(3, valueShape.w, pastValueShape.h + valueShape.h, 1, valueShape.c);
                }
            }

            if (!TryValidateTextureInputs(query, queryShape, key, attentionKeyShape, value, attentionValueShape, common, out var outputShape, out var outputStorageShape))
                return false;

            NcnnRepro.CmdTensorRef attnMask = null;
            NcnnRepro.BufferShape attnMaskShape = default;
            if (common.sp.attnMask)
            {
                if (layer.bottomNames.Length <= 3
                    || !TryGetExistingCmdTexture(blobs, shapes, layer.bottomNames[3], out attnMask, out attnMaskShape)
                    || attnMaskShape.dims != 2
                    || attnMaskShape.w != attentionKeyShape.h
                    || attnMaskShape.h != queryShape.h
                    || attnMask.packs != 1
                    || NcnnRepro.IsStrictLinearMatTexture(attnMask))
                    return false;
            }

            plan = new SdpaCmdRtPlan(
                common.scale,
                common.sp.causal,
                query,
                queryShape,
                key,
                keyShape,
                value,
                valueShape,
                pastKey,
                pastKeyShape,
                pastValue,
                pastValueShape,
                attentionKeyShape,
                attentionValueShape,
                attnMask,
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

        private static bool TryValidatePastCache(
            NcnnRepro.BufferShape currentKeyShape,
            NcnnRepro.BufferShape currentValueShape,
            object pastKey,
            NcnnRepro.BufferShape pastKeyShape,
            object pastValue,
            NcnnRepro.BufferShape pastValueShape)
        {
            if (pastKey == null || pastValue == null)
                return false;
            if (pastKeyShape.dims != 3 || pastValueShape.dims != 3 || pastKeyShape.h <= 0 || pastValueShape.h <= 0)
                return false;
            return pastKeyShape.w == currentKeyShape.w
                && pastKeyShape.d == currentKeyShape.d
                && pastKeyShape.c == currentKeyShape.c
                && pastValueShape.w == currentValueShape.w
                && pastValueShape.d == currentValueShape.d
                && pastValueShape.c == currentValueShape.c
                && pastKeyShape.h == pastValueShape.h;
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
            public readonly bool causal;
            public readonly NcnnRepro.TensorRef query;
            public readonly NcnnRepro.BufferShape queryShape;
            public readonly NcnnRepro.BufferShape queryStorageShape;
            public readonly int querySlices;
            public readonly NcnnRepro.TensorRef key;
            public readonly NcnnRepro.BufferShape keyCurrentShape;
            public readonly NcnnRepro.BufferShape keyShape;
            public readonly NcnnRepro.TensorRef value;
            public readonly NcnnRepro.BufferShape valueCurrentShape;
            public readonly NcnnRepro.BufferShape valueShape;
            public readonly NcnnRepro.TensorRef pastKey;
            public readonly NcnnRepro.BufferShape pastKeyShape;
            public readonly NcnnRepro.TensorRef pastValue;
            public readonly NcnnRepro.BufferShape pastValueShape;
            public readonly bool hasPastCache;
            public readonly NcnnRepro.TensorRef attnMask;
            public readonly bool hasAttnMask;
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
                bool causal,
                NcnnRepro.TensorRef query,
                NcnnRepro.BufferShape queryShape,
                NcnnRepro.TensorRef key,
                NcnnRepro.BufferShape keyShape,
                NcnnRepro.TensorRef value,
                NcnnRepro.BufferShape valueShape,
                NcnnRepro.TensorRef pastKey,
                NcnnRepro.BufferShape pastKeyShape,
                NcnnRepro.TensorRef pastValue,
                NcnnRepro.BufferShape pastValueShape,
                NcnnRepro.BufferShape attentionKeyShape,
                NcnnRepro.BufferShape attentionValueShape,
                NcnnRepro.TensorRef attnMask,
                NcnnRepro.BufferShape outputShape,
                NcnnRepro.BufferShape outputStorageShape)
            {
                this.scale = Mathf.Approximately(scale, 0f) ? 1f / Mathf.Sqrt(Mathf.Max(1, queryShape.w)) : scale;
                this.causal = causal;
                this.query = query;
                this.queryShape = queryShape;
                queryStorageShape = queryShape;
                querySlices = Mathf.Max(1, queryShape.d * Mathf.CeilToInt(queryShape.c / 4f));
                this.key = key;
                keyCurrentShape = keyShape;
                this.keyShape = attentionKeyShape;
                this.value = value;
                valueCurrentShape = valueShape;
                this.valueShape = attentionValueShape;
                this.pastKey = pastKey;
                this.pastKeyShape = pastKeyShape;
                this.pastValue = pastValue;
                this.pastValueShape = pastValueShape;
                hasPastCache = pastKey != null && pastValue != null;
                this.attnMask = attnMask;
                hasAttnMask = attnMask != null && attnMask.texture != null;
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
            public readonly bool causal;
            public readonly NcnnRepro.CmdTensorRef query;
            public readonly NcnnRepro.BufferShape queryShape;
            public readonly NcnnRepro.BufferShape queryStorageShape;
            public readonly int querySlices;
            public readonly NcnnRepro.CmdTensorRef key;
            public readonly NcnnRepro.BufferShape keyCurrentShape;
            public readonly NcnnRepro.BufferShape keyShape;
            public readonly NcnnRepro.CmdTensorRef value;
            public readonly NcnnRepro.BufferShape valueCurrentShape;
            public readonly NcnnRepro.BufferShape valueShape;
            public readonly NcnnRepro.CmdTensorRef pastKey;
            public readonly NcnnRepro.BufferShape pastKeyShape;
            public readonly NcnnRepro.CmdTensorRef pastValue;
            public readonly NcnnRepro.BufferShape pastValueShape;
            public readonly bool hasPastCache;
            public readonly NcnnRepro.CmdTensorRef attnMask;
            public readonly bool hasAttnMask;
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
                bool causal,
                NcnnRepro.CmdTensorRef query,
                NcnnRepro.BufferShape queryShape,
                NcnnRepro.CmdTensorRef key,
                NcnnRepro.BufferShape keyShape,
                NcnnRepro.CmdTensorRef value,
                NcnnRepro.BufferShape valueShape,
                NcnnRepro.CmdTensorRef pastKey,
                NcnnRepro.BufferShape pastKeyShape,
                NcnnRepro.CmdTensorRef pastValue,
                NcnnRepro.BufferShape pastValueShape,
                NcnnRepro.BufferShape attentionKeyShape,
                NcnnRepro.BufferShape attentionValueShape,
                NcnnRepro.CmdTensorRef attnMask,
                NcnnRepro.BufferShape outputShape,
                NcnnRepro.BufferShape outputStorageShape)
            {
                this.scale = Mathf.Approximately(scale, 0f) ? 1f / Mathf.Sqrt(Mathf.Max(1, queryShape.w)) : scale;
                this.causal = causal;
                this.query = query;
                this.queryShape = queryShape;
                queryStorageShape = queryShape;
                querySlices = Mathf.Max(1, queryShape.d * Mathf.CeilToInt(queryShape.c / 4f));
                this.key = key;
                keyCurrentShape = keyShape;
                this.keyShape = attentionKeyShape;
                this.value = value;
                valueCurrentShape = valueShape;
                this.valueShape = attentionValueShape;
                this.pastKey = pastKey;
                this.pastKeyShape = pastKeyShape;
                this.pastValue = pastValue;
                this.pastValueShape = pastValueShape;
                hasPastCache = pastKey != null && pastValue != null;
                this.attnMask = attnMask;
                hasAttnMask = attnMask != null && attnMask.texture != null;
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
