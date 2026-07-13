using System;

namespace NcnnCompute
{
    // The exported SD inpainting models only use aten::to as a dtype-preserving bridge
    // (int64->int64 for tokens, fp32->fp32 for activations), so aliasing the first input
    // is sufficient and avoids unnecessary buffer/materialization work.
    public sealed class NcnnAtenToLayerRepro : NcnnBaseLayerRepro
    {
        private readonly NcnnNoopLayerRepro _noop = new NcnnNoopLayerRepro();

        public NcnnAtenToLayerRepro()
            : base(NcnnLayerTypes.AtenTo, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            Validate(layer);
            _noop.ExecuteBuffer(owner, layer, context);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            Validate(layer);
            _noop.ExecuteCommandBuffer(owner, layer, context);
        }

        private static void Validate(NcnnParamModel.Layer layer)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));
            if (layer.bottomNames == null || layer.bottomNames.Length < 1)
                throw new InvalidOperationException("aten::to requires at least one input: " + layer.name);
        }
    }
}
