using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnEltwiseLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnEltwiseLayerRepro() : base(NcnnLayerTypes.Eltwise, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteEltwiseBufferLayer(layer, context);
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context) => owner.ExecuteEltwiseCommandBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal void ExecuteEltwiseBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var indexBlobs = context.indexBlobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            do
            {
                                    var a = GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                    var b = GetOrMaterializeTexture(layer.bottomNames[1], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                    if (a.width != b.width || a.height != b.height || a.packs != b.packs)
                                        throw new InvalidOperationException("Eltwise shape mismatch: " + layer.name);
                                    var coeff = ParseEltwiseCoeff(layer);
                                    var isTargetSftResidualLayer = string.IsNullOrEmpty(CodeFormerTargetSftResidualLayer)
                                        ? CodeFormerSftResidualLayers.Contains(layer.name)
                                        : string.Equals(layer.name, CodeFormerTargetSftResidualLayer, StringComparison.Ordinal);
                                    if (CodeFormerSftMulScale != 1f && isTargetSftResidualLayer)
                                        coeff = (coeff.coeffA, coeff.coeffB * CodeFormerSftMulScale);
                                    var outRt = RentTempArray(a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                                    _ops.AddPack4(a.texture, b.texture, coeff.coeffA, coeff.coeffB, a.packs, outRt);
                                    textureBlobs[layer.topNames[0]] = new TensorRef
                                    {
                                        texture = outRt,
                                        width = a.width,
                                        height = a.height,
                                        packs = a.packs,
                                        refs = 1,
                                        owned = true
                                    };
                                    var aShape = GetTextureShape(textureShapes, a, layer.bottomNames[0]);
                                    textureShapes[layer.topNames[0]] = new BufferShape(3, a.width, a.height, 1, aShape.c);
                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }

        internal void ExecuteEltwiseCommandBufferLayer(NcnnParamModel.Layer l, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            do
            {
                                    var a = GetCmdTensor(blobs, l.bottomNames[0]);
                                    var b = GetCmdTensor(blobs, l.bottomNames[1]);
                                    var coeff = ParseEltwiseCoeff(l);
                                    var outArr = RentTempArray(cmd, a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                                    _ops.AddPack4(cmd, a.texture, b.texture, coeff.coeffA, coeff.coeffB, a.packs, outArr);
                                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = a.width, height = a.height, packs = a.packs, refs = 1, owned = true };
                                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
