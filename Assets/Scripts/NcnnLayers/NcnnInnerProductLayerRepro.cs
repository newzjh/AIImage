using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnInnerProductLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnInnerProductLayerRepro() : base(NcnnLayerTypes.InnerProduct, supportsBufferPath: true, supportsCommandBufferPath: false) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br) => owner.LoadInnerProductLayer(layer, br);
        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteInnerProductBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal LayerLoadMetrics LoadInnerProductLayer(NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

                            var ip = new InnerProductPack();
                            ip.outFeatures = layer.GetInt(0, 0);
                            ip.biasTerm = layer.GetInt(1, 0);
                            ip.weightSize = layer.GetInt(2, 0);
                            ip.inFeatures = ip.outFeatures > 0 ? ip.weightSize / ip.outFeatures : 0;

                            phaseSw.Restart();
                            var w = ReadPackedOrRawWeightArray(br, ip.weightSize, layer.name);
                            var b = ip.biasTerm != 0 ? br.ReadFloat32Array(ip.outFeatures) : new float[ip.outFeatures];
                            phaseSw.Stop();
                            readMs += phaseSw.ElapsedMilliseconds;

                            phaseSw.Restart();
                            ip.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                            ip.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                            ip.w.SetData(w);
                            ip.b.SetData(b);
                            phaseSw.Stop();
                            uploadMs += phaseSw.ElapsedMilliseconds;

                            _innerProduct[layer.name] = ip;
                            return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        internal void ExecuteInnerProductBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                    if (!_innerProduct.TryGetValue(layer.name, out var ip))
                                        throw new InvalidOperationException("InnerProduct not found: " + layer.name);

                                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                    if (srcBuf == null)
                                        throw new InvalidOperationException("InnerProduct source not found: " + layer.bottomNames[0]);

                                    var srcTensor = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                    var rows = srcTensor != null && srcTensor.dims == 2 && srcTensor.w == ip.inFeatures ? srcTensor.h : 1;
                                    var outBuf = RentTempBuffer(ip.outFeatures * rows, sizeof(float));
                                    if (rows > 1)
                                        _ops.InnerProduct2D(srcBuf, rows, ip.inFeatures, ip.w, ip.b, ip.outFeatures, outBuf);
                                    else
                                        _ops.InnerProduct(srcBuf, ip.inFeatures, ip.w, ip.b, ip.outFeatures, outBuf);

                                    bufferBlobs[layer.topNames[0]] = outBuf;
                                    bufferViews[layer.topNames[0]] = rows > 1
                                        ? new NcnnTensorBuffer(outBuf, 2, ip.outFeatures, rows, 1, 1, false)
                                        : new NcnnTensorBuffer(outBuf, 1, ip.outFeatures, 1, 1, 1, false);
                                    tempOwned.Add(outBuf);
                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
