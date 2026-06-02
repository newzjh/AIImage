using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnConvolutionDepthWiseLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnConvolutionDepthWiseLayerRepro() : base(NcnnLayerTypes.ConvolutionDepthWise, supportsBufferPath: true, supportsCommandBufferPath: false) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br) => owner.LoadConvolutionFamilyLayer(layer, br);
        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteConvolutionFamilyBufferLayer(layer, context);
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context) => owner.ExecuteConvolutionFamilyCommandBufferLayer(layer, context);
    }
}
