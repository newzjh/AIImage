using System;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnRotaryEmbedLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnRotaryEmbedLayerRepro()
            : base(NcnnLayerTypes.RotaryEmbed, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            owner._extraPacks[layer.name] = new NcnnRepro.RotaryEmbedPack
            {
                interleaved = layer.GetInt(0, 0) != 0
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.RotaryEmbedPack rp)
                throw new InvalidOperationException("RotaryEmbed pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            var cosBuf = owner.GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var sinBuf = owner.GetOrConvertToBuffer(layer.bottomNames[2], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var cosView = NcnnRepro.TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
            var sinView = NcnnRepro.TryGetBufferView(layer.bottomNames[2], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || srcView.dims != 3)
                throw new InvalidOperationException("RotaryEmbed expects dims=3 source: " + layer.name);
            if (cosBuf == null || sinBuf == null || cosView == null || sinView == null)
                throw new InvalidOperationException("RotaryEmbed cache input missing: " + layer.name);

            var embedDim = srcView.w;
            var seqLen = srcView.h;
            var numHeads = srcView.c;
            if ((embedDim & 1) != 0)
                throw new InvalidOperationException("RotaryEmbed requires even embed_dim: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(3, srcView.w, srcView.h, 1, srcView.c);
            owner.Ops.RotaryEmbedBuf(srcBuf, embedDim, seqLen, numHeads, rp.interleaved, cosBuf, sinBuf, outTensor.buffer);
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

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
            blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorAlias(src, srcShape, storageShape);
            if (shapes != null)
                shapes[layer.topNames[0]] = srcShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
