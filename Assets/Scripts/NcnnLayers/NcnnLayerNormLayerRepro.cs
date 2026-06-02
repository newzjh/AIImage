using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnLayerNormLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnLayerNormLayerRepro() : base(NcnnLayerTypes.LayerNorm, supportsBufferPath: true, supportsCommandBufferPath: false) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br) => owner.LoadLayerNormLayer(layer, br);
        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteLayerNormBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal LayerLoadMetrics LoadLayerNormLayer(NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

                            var lp = new LayerNormPack();
                            lp.affineSize = layer.GetInt(0, 0);
                            lp.eps = layer.GetFloat(1, 1e-5f);
                            lp.affine = layer.GetInt(2, 1) != 0;

                            float[] gamma = null;
                            float[] beta = null;
                            if (lp.affine && lp.affineSize > 0)
                            {
                                phaseSw.Restart();
                                gamma = br.ReadNcnnMatAsFloat32(lp.affineSize, 0, 0, 0, 1);
                                beta = br.ReadNcnnMatAsFloat32(lp.affineSize, 0, 0, 0, 1);
                                phaseSw.Stop();
                                readMs += phaseSw.ElapsedMilliseconds;

                                phaseSw.Restart();
                                lp.gamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
                                lp.beta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
                                lp.gamma.SetData(gamma);
                                lp.beta.SetData(beta);
                                phaseSw.Stop();
                                uploadMs += phaseSw.ElapsedMilliseconds;
                            }

                            _layerNorm[layer.name] = lp;
                            return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        internal void ExecuteLayerNormBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                    if (!_layerNorm.TryGetValue(layer.name, out var lp))
                                        throw new InvalidOperationException("LayerNorm not found: " + layer.name);
                                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                    var srcView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                    if (srcBuf == null || srcView == null || srcView.dims != 2)
                                        throw new InvalidOperationException("LayerNorm expects dims=2 buffer input: " + layer.name);
                                    var outBuf = RentTempBuffer(srcBuf.count, sizeof(float));
                                    _ops.CopyBuf(srcBuf, outBuf, srcBuf.count);
                                    _ops.LayerNorm2DInplace(outBuf, srcView.h, srcView.w, lp.eps, lp.affine, lp.gamma, lp.beta);
                                    bufferBlobs[layer.topNames[0]] = outBuf;
                                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
                                    tempOwned.Add(outBuf);
                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
