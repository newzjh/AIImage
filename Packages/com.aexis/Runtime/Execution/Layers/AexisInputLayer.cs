namespace Aexis.Execution
{
    // Input is a graph-boundary alias. It must never turn a fixed Buffer input into a
    // texture just to satisfy the first consumer.
    public sealed class AexisInputLayer : AexisBaseLayer
    {
        public AexisInputLayer() : base(AexisLayerTypes.Input, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (layer == null || context == null || layer.topNames == null || layer.topNames.Length == 0)
                return;

            var inputName = layer.name;
            if (string.IsNullOrWhiteSpace(inputName))
                return;

            if (context.bufferBlobs.TryGetValue(inputName, out var inputBuffer) && inputBuffer != null)
            {
                context.bufferRefs.TryGetValue(inputName, out var inputBufferRef);
                var inputView = AexisGraphSession.TryGetBufferView(inputName, context.bufferBlobs, context.bufferViews);
                for (var i = 0; i < layer.topNames.Length; i++)
                {
                    var topName = layer.topNames[i];
                    if (string.IsNullOrWhiteSpace(topName) || topName == inputName)
                        continue;

                    context.bufferBlobs[topName] = inputBuffer;
                    if (inputBufferRef != null)
                    {
                        context.bufferRefs[topName] = inputBufferRef;
                        inputBufferRef.refs++;
                    }
                    if (inputView != null)
                        context.bufferViews[topName] = inputView;
                }

                return;
            }

            if (!AexisGraphSession.TryGetExistingTexture(
                    context.textureBlobs,
                    context.textureShapes,
                    inputName,
                    out var inputTexture,
                    out var logicalShape))
                return;

            var storageShape = AexisGraphSession.GetTextureStorageShape(inputTexture, logicalShape);
            for (var i = 0; i < layer.topNames.Length; i++)
            {
                var topName = layer.topNames[i];
                if (string.IsNullOrWhiteSpace(topName) || topName == inputName)
                    continue;

                context.textureBlobs[topName] = AexisGraphSession.CreateTextureAlias(inputTexture, logicalShape, storageShape);
                context.textureShapes[topName] = logicalShape;
            }
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            ExecuteBuffer(owner, layer, context);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (layer == null || context == null || layer.topNames == null || layer.topNames.Length == 0)
                return;

            var inputName = layer.name;
            if (string.IsNullOrWhiteSpace(inputName)
                || !AexisGraphSession.TryGetCmdShape(context.shapes, context.blobs, inputName, out var logicalShape)
                || !context.blobs.TryGetValue(inputName, out var inputTexture)
                || inputTexture == null
                || inputTexture.texture == null)
                return;

            var storageShape = AexisGraphSession.GetCmdStorageShape(inputTexture, logicalShape);
            for (var i = 0; i < layer.topNames.Length; i++)
            {
                var topName = layer.topNames[i];
                if (string.IsNullOrWhiteSpace(topName) || topName == inputName)
                    continue;

                context.blobs[topName] = AexisGraphSession.CreateCmdTensorAlias(inputTexture, logicalShape, storageShape);
                context.shapes[topName] = logicalShape;
            }
        }
    }
}
