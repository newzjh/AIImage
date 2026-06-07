using System;
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

            var scoreBuf = owner.RentTempBuffer(numHeads * srcSeqLen * dstSeqLen, sizeof(float));
            var outTensor = owner.RentTempTensorBuffer(3, outEmbedDim, srcSeqLen, 1, numHeads);
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
#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
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
    }
}
