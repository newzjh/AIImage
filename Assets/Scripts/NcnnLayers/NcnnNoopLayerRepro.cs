namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnNoopLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnNoopLayerRepro()
            : base(NcnnLayerTypes.Noop, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var aliasedBuffer = false;
            if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var buf) && buf != null)
            {
                bufferBlobs[layer.topNames[0]] = buf;
                if (bufferRefs.TryGetValue(layer.bottomNames[0], out var bufferRef) && bufferRef != null)
                {
                    bufferRefs[layer.topNames[0]] = bufferRef;
                    bufferRef.refs++;
                }

                var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                if (srcView != null)
                    bufferViews[layer.topNames[0]] = srcView;
                aliasedBuffer = true;
            }

            if (textureBlobs.TryGetValue(layer.bottomNames[0], out var srcTex) && srcTex != null && srcTex.texture != null)
            {
                var srcShape = NcnnRepro.GetTextureShape(textureShapes, srcTex, layer.bottomNames[0]);
                textureBlobs[layer.topNames[0]] = srcTex;
                textureShapes[layer.topNames[0]] = srcShape;
                srcTex.refs++;
            }
            else if (!aliasedBuffer)
            {
                var src = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                var srcShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                textureBlobs[layer.topNames[0]] = src;
                textureShapes[layer.topNames[0]] = srcShape;
                src.refs++;
            }

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            blobs[layer.topNames[0]] = src;
            src.refs++;
            if (shapes != null)
                shapes[layer.topNames[0]] = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
