namespace NcnnCompute
{
    public sealed class NcnnFlattenLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnFlattenLayerRepro()
            : base(NcnnLayerTypes.Flatten, supportsBufferPath: true, supportsCommandBufferPath: true)
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
                    bufferViews[layer.topNames[0]] = srcView.Reshape(1, srcView.elementCount);
            }
            else
            {
                var src = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                textureBlobs[layer.topNames[0]] = src;
                textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(1, src.width * src.height * src.packs * 4, 1, 1, 1);
                src.refs++;
            }

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            blobs[layer.topNames[0]] = src;
            src.refs++;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
        }
    }
}
