using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnMemoryDataLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnMemoryDataLayerRepro() : base(NcnnLayerTypes.MemoryData, supportsBufferPath: true, supportsCommandBufferPath: false) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br) => owner.LoadMemoryDataLayer(layer, br);
        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteMemoryDataBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal LayerLoadMetrics LoadMemoryDataLayer(NcnnParamModel.Layer layer, NcnnBinReader br)
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
                            var a = ReadClipMatAsFloat32(br, w, h, d, c, loadType);
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
                            _memoryData[layer.name] = new MemoryDataPack
                            {
                                data = buf,
                                dims = dims,
                                w = Mathf.Max(1, w),
                                h = Mathf.Max(1, h),
                                d = Mathf.Max(1, d),
                                c = Mathf.Max(1, c)
                            };
                            return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        internal void ExecuteMemoryDataBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                    if (!_memoryData.TryGetValue(layer.name, out var mp) || mp.data == null)
                                        throw new InvalidOperationException("MemoryData not found: " + layer.name);
                                    bufferBlobs[layer.topNames[0]] = mp.data;
                                    bufferRefs[layer.topNames[0]] = new BufferRef
                                    {
                                        buffer = mp.data,
                                        refs = 1,
                                        owned = false
                                    };
                                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(mp.data, mp.dims, mp.w, mp.h, mp.d, mp.c, false);
                                    continue;
            } while (false);
        }
    }
}
