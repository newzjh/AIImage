using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnGroupNormLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnGroupNormLayerRepro() : base(NcnnLayerTypes.GroupNorm, supportsBufferPath: true, supportsCommandBufferPath: false) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var gp = new NcnnRepro.GroupNormPack();
                                        gp.group = layer.GetInt(0, 1);
                                        gp.channels = layer.GetInt(1, 0);
                                        gp.eps = layer.GetFloat(2, 1e-5f);
                                        gp.affine = layer.GetInt(3, 1) != 0;

                                        float[] gamma = null;
                                        float[] beta = null;
                                        if (gp.affine && gp.channels > 0)
                                        {
                                            phaseSw.Restart();
                                            gamma = br.ReadNcnnMatAsFloat32(gp.channels, 0, 0, 0, 1);
                                            beta = br.ReadNcnnMatAsFloat32(gp.channels, 0, 0, 0, 1);
                                            phaseSw.Stop();
                                            readMs += phaseSw.ElapsedMilliseconds;

                                            phaseSw.Restart();
                                            gp.gamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
                                            gp.beta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
                                            gp.gamma.SetData(gamma);
                                            gp.beta.SetData(beta);
                                            phaseSw.Stop();
                                            uploadMs += phaseSw.ElapsedMilliseconds;
                                        }

                                        owner._groupNorm[layer.name] = gp;
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
                                                if (!owner._groupNorm.TryGetValue(layer.name, out var gp))
                                                    throw new InvalidOperationException("GroupNorm not found: " + layer.name);
                                                if (owner.EnableGroupNormTexturePath
                                                    && owner.UseNcnnStyleGroupNorm
                                                    && owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                                                    && NcnnRepro.CanUseGroupNormPack4Path(srcTex, srcShape, gp))
                                                {
                                                    var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
                                                    var statsBuf = owner.RentTempBuffer(gp.group, sizeof(float) * 4);
                                                    try
                                                    {
                                                        owner.Ops.GroupNormPack4(srcTex.texture, srcShape.w, srcShape.h, srcShape.c, srcTex.packs, gp.group, gp.eps, gp.gamma, gp.beta, statsBuf, outRt);
                                                        NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
                                                        outRt = null;
                                                    }
                                                    finally
                                                    {
                                                        owner.ReturnTempBuffer(statsBuf);
                                                        if (outRt != null)
                                                            owner.ReturnTempArray(outRt);
                                                    }

                                                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                    continue;
                                                }

                                                var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                if (srcBuf == null || srcView == null)
                                                    throw new InvalidOperationException("GroupNorm source not found: " + layer.name);
                                                var outBuf = owner.RentTempBuffer(srcBuf.count, sizeof(float));
                                                owner.Ops.CopyBuf(srcBuf, outBuf, srcBuf.count);
                                                var spatial = srcBuf.count / Mathf.Max(1, gp.channels);
                                                owner.Ops.GroupNormInplace(outBuf, spatial, 1, gp.channels, gp.group, gp.eps, gp.affine, gp.gamma, gp.beta);
                                                bufferBlobs[layer.topNames[0]] = outBuf;
                                                bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
                                                tempOwned.Add(outBuf);
                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
