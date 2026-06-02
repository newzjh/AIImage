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

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                                var a = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                var b = owner.GetOrMaterializeTexture(layer.bottomNames[1], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                if (a.width != b.width || a.height != b.height || a.packs != b.packs)
                                                    throw new InvalidOperationException("Eltwise shape mismatch: " + layer.name);
                                                var coeff = NcnnRepro.ParseEltwiseCoeff(layer);
                                                var isTargetSftResidualLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftResidualLayer)
                                                    ? NcnnRepro.CodeFormerSftResidualLayers.Contains(layer.name)
                                                    : string.Equals(layer.name, owner.CodeFormerTargetSftResidualLayer, StringComparison.Ordinal);
                                                if (owner.CodeFormerSftMulScale != 1f && isTargetSftResidualLayer)
                                                    coeff = (coeff.coeffA, coeff.coeffB * owner.CodeFormerSftMulScale);
                                                var outRt = owner.RentTempArray(a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                                                owner.Ops.AddPack4(a.texture, b.texture, coeff.coeffA, coeff.coeffB, a.packs, outRt);
                                                textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
                                                {
                                                    texture = outRt,
                                                    width = a.width,
                                                    height = a.height,
                                                    packs = a.packs,
                                                    refs = 1,
                                                    owned = true
                                                };
                                                var aShape = NcnnRepro.GetTextureShape(textureShapes, a, layer.bottomNames[0]);
                                                textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, a.width, a.height, 1, aShape.c);
                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
                        var cmd = context.commandBuffer;
                        var blobs = context.blobs;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;

                        do
                        {
                                                var a = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var b = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[1]);
                                                var coeff = NcnnRepro.ParseEltwiseCoeff(layer);
                                                var outArr = owner.RentTempArray(cmd, a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                                                owner.Ops.AddPack4(cmd, a.texture, b.texture, coeff.coeffA, coeff.coeffB, a.packs, outArr);
                                                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = a.width, height = a.height, packs = a.packs, refs = 1, owned = true };
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
