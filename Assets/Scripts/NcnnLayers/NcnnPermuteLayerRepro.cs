using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnPermuteLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnPermuteLayerRepro() : base(NcnnLayerTypes.Permute, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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
                                                var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                if (srcBuf == null)
                                                    throw new InvalidOperationException("Permute source not found: " + layer.bottomNames[0]);

                                                var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                if (srcTensor == null)
                                                    throw new InvalidOperationException("Permute shape not resolved: " + layer.name);

                                                var orderType = layer.GetInt(0, 0);
                                                var dims = Mathf.Clamp(srcTensor.dims, 2, 4);
                                                var axes = NcnnRepro.ResolvePermuteAxes(dims, orderType, layer.name);
                                                var outShape = NcnnRepro.ResolvePermuteShape(srcTensor, dims, axes);
                                                var outBuf = owner.RentTempBuffer(outShape.w * outShape.h * outShape.d * outShape.c, sizeof(float));
                                                owner.Ops.Permute(srcBuf, dims, srcTensor.w, srcTensor.h, srcTensor.d, srcTensor.c, orderType, outBuf);

                                                bufferBlobs[layer.topNames[0]] = outBuf;
                                                bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], outBuf);
                                                bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, outShape.dims, outShape.w, outShape.h, outShape.d, outShape.c, false);
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

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            owner.CopyCmdTensor(cmd, src, layer.topNames[0], blobs, src.width, src.height, src.packs);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
        }
    }
}
