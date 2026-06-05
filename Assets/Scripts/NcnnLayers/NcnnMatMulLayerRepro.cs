using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnMatMulLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnMatMulLayerRepro() : base(NcnnLayerTypes.MatMul, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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
                                                var aBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var bBuf = owner.GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var aView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                var bView = NcnnRepro.TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
                                                if (aBuf == null || bBuf == null || aView == null || bView == null)
                                                    throw new InvalidOperationException("MatMul source not found: " + layer.name);

                                                var outTensor = owner.RunMatMulLayer(aBuf, aView, bBuf, bView, layer.GetInt(0, 0) != 0);
                                                bufferBlobs[layer.topNames[0]] = outTensor.buffer;
                                                bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], outTensor.buffer);
                                                bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outTensor.buffer, outTensor.dims, outTensor.w, outTensor.h, outTensor.d, outTensor.c, false);
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

            var a = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var b = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[1]);
            owner.PublishCmdTensorLikeInput(cmd, layer.topNames[0], Mathf.Max(1, b.width), Mathf.Max(1, a.height), 1, blobs);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
        }
    }
}
