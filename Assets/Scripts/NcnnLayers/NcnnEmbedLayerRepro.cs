using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnEmbedLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnEmbedLayerRepro() : base(NcnnLayerTypes.Embed, supportsBufferPath: true, supportsCommandBufferPath: false) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var ep = new NcnnRepro.EmbedPack();
                                        ep.numOutput = layer.GetInt(0, 0);
                                        ep.inputDim = layer.GetInt(1, 0);
                                        ep.biasTerm = layer.GetInt(2, 0);
                                        ep.weightSize = layer.GetInt(3, 0);

                                        phaseSw.Restart();
                                        var w = NcnnRepro.ReadClipArrayAsFloat32(br, ep.weightSize, 0);
                                        float[] b = null;
                                        if (ep.biasTerm != 0)
                                            b = br.ReadNcnnMatAsFloat32(ep.numOutput, 0, 0, 0, 1);
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        phaseSw.Restart();
                                        ep.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                                        ep.w.SetData(w);
                                        if (b != null)
                                        {
                                            ep.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                                            ep.b.SetData(b);
                                        }
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;

                                        owner._embed[layer.name] = ep;
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
                                                if (!owner._embed.TryGetValue(layer.name, out var ep) || ep.w == null)
                                                    throw new InvalidOperationException("Embed not found: " + layer.name);
                                                if (!bufferBlobs.TryGetValue(layer.bottomNames[0], out var indicesBuf) || indicesBuf == null)
                                                    throw new InvalidOperationException("Embed input buffer not found: " + layer.bottomNames[0]);

                                                var words = indicesBuf.count;
                                                var outBuf = owner.RentTempBuffer(words * ep.numOutput, sizeof(float));
                                                owner.Ops.Embed(indicesBuf, words, ep.w, ep.b, ep.numOutput, ep.inputDim, ep.biasTerm != 0, outBuf);
                                                bufferBlobs[layer.topNames[0]] = outBuf;
                                                bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], outBuf);
                                                bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, 2, ep.numOutput, words, 1, 1, false);
                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
