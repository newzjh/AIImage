using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnMemoryDataLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnMemoryDataLayerRepro() : base(NcnnLayerTypes.MemoryData, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var w = layer.GetInt(0, 0);
                                        var h = layer.GetInt(1, 0);
                                        var d = layer.GetInt(11, 0);
                                        var c = layer.GetInt(2, 0);
                                        var loadType = layer.GetInt(21, 1);

                                        phaseSw.Restart();
                                        var a = NcnnRepro.ReadClipMatAsFloat32(br, w, h, d, c, loadType);
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        phaseSw.Restart();
                                        var buf = new ComputeBuffer(a.Length, sizeof(float), ComputeBufferType.Structured);
                                        buf.SetData(a);
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;

                                        var dims = 1;
                                        if (h > 0) dims = 2;
                                        if (c > 0) dims = d > 0 ? 4 : 3;
                                        owner._memoryData[layer.name] = new NcnnRepro.MemoryDataPack
                                        {
                                            data = buf,
                                            dims = dims,
                                            w = Mathf.Max(1, w),
                                            h = Mathf.Max(1, h),
                                            d = Mathf.Max(1, d),
                                            c = Mathf.Max(1, c)
                                        };
                                        return new NcnnRepro.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }
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
                                                if (!owner._memoryData.TryGetValue(layer.name, out var mp) || mp.data == null)
                                                    throw new InvalidOperationException("MemoryData not found: " + layer.name);
                                                var tensor = new NcnnTensorBuffer(mp.data, mp.dims, mp.w, mp.h, mp.d, mp.c, false);
                                                owner.PublishTensorBufferOutput(
                                                    layer.topNames[0],
                                                    tensor,
                                                    preferTexture: mp.dims <= 3,
                                                    textureBlobs,
                                                    textureShapes,
                                                    bufferBlobs,
                                                    bufferRefs,
                                                    bufferViews,
                                                    tempOwned);
                                                continue;
                        } while (false);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;

            if (!owner._memoryData.TryGetValue(layer.name, out var mp) || mp.data == null)
                throw new InvalidOperationException("MemoryData not found: " + layer.name);

            owner.PublishCmdTensorBufferOutput(
                cmd,
                layer.topNames[0],
                new NcnnTensorBuffer(mp.data, mp.dims, mp.w, mp.h, mp.d, mp.c, false),
                preferTexture: true,
                blobs,
                shapes);
        }
    }
}
