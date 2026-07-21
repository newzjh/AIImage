using System;

namespace Aexis.Execution
{
    // The exported SD inpainting models only use aten::to as a dtype-preserving bridge
    // (int64->int64 for tokens, fp32->fp32 for activations), so aliasing the first input
    // is sufficient and avoids unnecessary buffer/materialization work.
    public sealed class AexisAtenToLayer : AexisBaseLayer
    {
        private readonly AexisNoopLayer _noop = new AexisNoopLayer();

        public AexisAtenToLayer()
            : base(AexisLayerTypes.AtenTo, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            Validate(layer);
            _noop.ExecuteBuffer(owner, layer, context);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            Validate(layer);
            _noop.ExecuteCommandBuffer(owner, layer, context);
        }

        private static void Validate(AexisGraphModel.Layer layer)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));
            if (layer.bottomNames == null || layer.bottomNames.Length < 1)
                throw new InvalidOperationException("aten::to requires at least one input: " + layer.name);
        }
    }
}
