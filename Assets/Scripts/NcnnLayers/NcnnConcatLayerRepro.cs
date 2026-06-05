using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnConcatLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnConcatLayerRepro() : base(NcnnLayerTypes.Concat, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            var partBuffers = new ComputeBuffer[layer.bottomNames.Length];
            var partViews = new NcnnTensorBuffer[layer.bottomNames.Length];
            for (var i = 0; i < layer.bottomNames.Length; i++)
            {
                partBuffers[i] = owner.GetOrConvertToBuffer(layer.bottomNames[i], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                partViews[i] = NcnnRepro.TryGetBufferView(layer.bottomNames[i], bufferBlobs, bufferViews);
                if (partBuffers[i] == null || partViews[i] == null)
                    throw new InvalidOperationException("Concat source not found: " + layer.name + " | " + layer.bottomNames[i]);
            }

            var firstView = partViews[0];
            var positiveAxis = layer.GetInt(0, 0);
            if (positiveAxis < 0)
                positiveAxis += firstView.dims;
            if (positiveAxis < 0 || positiveAxis >= firstView.dims)
                throw new InvalidOperationException("Concat axis out of range: " + layer.name);

            var tensorAxis = NcnnRepro.MapNcnnAxisToTensorAxis(firstView.dims, positiveAxis);
            var outW = firstView.w;
            var outH = firstView.h;
            var outD = firstView.d;
            var outC = firstView.c;

            for (var i = 0; i < partViews.Length; i++)
            {
                var v = partViews[i];
                if (v.dims != firstView.dims)
                    throw new InvalidOperationException("Concat dims mismatch: " + layer.name);

                if (tensorAxis != 0 && v.w != firstView.w)
                    throw new InvalidOperationException("Concat width mismatch: " + layer.name);
                if (tensorAxis != 1 && v.h != firstView.h)
                    throw new InvalidOperationException("Concat height mismatch: " + layer.name);
                if (firstView.dims == 4 && tensorAxis != 2 && v.d != firstView.d)
                    throw new InvalidOperationException("Concat depth mismatch: " + layer.name);
                var channelAxis = firstView.dims == 4 ? 3 : 2;
                if (tensorAxis != channelAxis && v.c != firstView.c)
                    throw new InvalidOperationException("Concat channel mismatch: " + layer.name);

                if (i == 0)
                    continue;

                if (tensorAxis == 0) outW += v.w;
                else if (tensorAxis == 1) outH += v.h;
                else if (tensorAxis == 2 && firstView.dims == 4) outD += v.d;
                else outC += v.c;
            }

            var outCount = outW * outH * outD * outC;
            var outData = new float[outCount];
            var dstAxisOffset = 0;

            for (var i = 0; i < partViews.Length; i++)
            {
                var v = partViews[i];
                var srcData = NcnnRepro.ReadFloatBuffer(partBuffers[i]);

                for (var c = 0; c < v.c; c++)
                {
                    var dstC = tensorAxis == (firstView.dims == 4 ? 3 : 2) ? dstAxisOffset + c : c;
                    for (var z = 0; z < v.d; z++)
                    {
                        var dstDLocal = tensorAxis == 2 && firstView.dims == 4 ? dstAxisOffset + z : z;
                        for (var y = 0; y < v.h; y++)
                        {
                            var dstYLocal = tensorAxis == 1 ? dstAxisOffset + y : y;
                            for (var x = 0; x < v.w; x++)
                            {
                                var dstXLocal = tensorAxis == 0 ? dstAxisOffset + x : x;
                                var srcIndex = (((c * v.d) + z) * v.h + y) * v.w + x;
                                var dstIndex = (((dstC * outD) + dstDLocal) * outH + dstYLocal) * outW + dstXLocal;
                                outData[dstIndex] = srcData[srcIndex];
                            }
                        }
                    }
                }

                if (tensorAxis == 0) dstAxisOffset += v.w;
                else if (tensorAxis == 1) dstAxisOffset += v.h;
                else if (tensorAxis == 2 && firstView.dims == 4) dstAxisOffset += v.d;
                else dstAxisOffset += v.c;
            }

            var outBuf = owner.RentTempBuffer(outCount, sizeof(float));
            outBuf.SetData(outData);
            var outTensor = new NcnnTensorBuffer(outBuf, firstView.dims, outW, outH, outD, outC, false);

            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: firstView.dims <= 3 && tensorAxis == (firstView.dims == 3 ? 2 : -1),
                textureBlobs,
                textureShapes,
                bufferBlobs,
                bufferRefs,
                bufferViews,
                tempOwned);
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
                        var cmd = context.commandBuffer;
                        var blobs = context.blobs;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;

                        do
                        {
                                                var parts = new NcnnRepro.CmdTensorRef[layer.bottomNames.Length];
                                                var sumP = 0;
                                                var w = 0;
                                                var h = 0;
                                                for (var i = 0; i < layer.bottomNames.Length; i++)
                                                {
                                                    var tr = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[i]);
                                                    parts[i] = tr;
                                                    w = tr.width;
                                                    h = tr.height;
                                                    sumP += tr.packs;
                                                }

                                                var outArr = owner.RentTempArray(cmd, w, h, sumP, RenderTextureFormat.ARGBHalf);
                                                var off = 0;
                                                for (var i = 0; i < parts.Length; i++)
                                                {
                                                    owner.Ops.CopyPack4(cmd, parts[i].texture, 0, outArr, off, parts[i].packs);
                                                    off += parts[i].packs;
                                                }

                                                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = w, height = h, packs = sumP, refs = 1, owned = true };
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
