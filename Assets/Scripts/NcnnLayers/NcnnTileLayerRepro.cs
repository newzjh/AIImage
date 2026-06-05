using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnTileLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnTileLayerRepro() : base(NcnnLayerTypes.Tile, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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

                        do
                        {
                                                var hasAxis = layer.intParams != null && layer.intParams.ContainsKey(0);
                                                var hasTiles = layer.intParams != null && layer.intParams.ContainsKey(1);
                                                var axis = layer.GetInt(0, 0);
                                                var tiles = layer.GetInt(1, 1);
                                                var isPassthrough = (!hasAxis && !hasTiles) || tiles <= 1;

                                                if (isPassthrough)
                                                {
                                                    if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var tileBuf) && tileBuf != null)
                                                    {
                                                        bufferBlobs[layer.topNames[0]] = tileBuf;
                                                        if (bufferRefs.TryGetValue(layer.bottomNames[0], out var tileRef) && tileRef != null)
                                                        {
                                                            bufferRefs[layer.topNames[0]] = tileRef;
                                                            tileRef.refs++;
                                                        }

                                                        var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                        if (srcView != null)
                                                            bufferViews[layer.topNames[0]] = srcView;
                                                    }
                                                    else
                                                    {
                                                        var src = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                        textureBlobs[layer.topNames[0]] = src;
                                                        textureShapes[layer.topNames[0]] = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                                        src.refs++;
                                                    }

                                                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                    continue;
                                                }

                                                var srcBuf2 = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var srcView2 = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                if (srcBuf2 == null || srcView2 == null)
                                                    throw new InvalidOperationException("Tile source not found: " + layer.name);

                                                if (axis < 0)
                                                    axis += srcView2.dims;
                                                if (axis < 0 || axis >= srcView2.dims)
                                                    throw new InvalidOperationException("Tile axis out of range: " + layer.name);

                                                var tensorAxis = NcnnRepro.MapNcnnAxisToTensorAxis(srcView2.dims, axis);
                                                var outW = srcView2.w;
                                                var outH = srcView2.h;
                                                var outD = srcView2.d;
                                                var outC = srcView2.c;
                                                if (tensorAxis == 0) outW *= tiles;
                                                else if (tensorAxis == 1) outH *= tiles;
                                                else if (tensorAxis == 2 && srcView2.dims == 4) outD *= tiles;
                                                else if (tensorAxis == 2 || tensorAxis == 3) outC *= tiles;

                                                var outBuf = owner.RentTempBuffer(outW * outH * outD * outC, sizeof(float));
                                                owner.Ops.Tile(srcBuf2, srcView2.dims, srcView2.w, srcView2.h, srcView2.d, srcView2.c, tensorAxis, tiles, outW, outH, outD, outC, outBuf);

                                                bufferBlobs[layer.topNames[0]] = outBuf;
                                                bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], outBuf);
                                                bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcView2.dims, outW, outH, outD, outC, false);
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
            var hasAxis = layer.intParams != null && layer.intParams.ContainsKey(0);
            var hasTiles = layer.intParams != null && layer.intParams.ContainsKey(1);
            var tiles = layer.GetInt(1, 1);
            if ((!hasAxis && !hasTiles) || tiles <= 1)
            {
                blobs[layer.topNames[0]] = src;
                src.refs++;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            owner.CopyCmdTensor(cmd, src, layer.topNames[0], blobs, src.width, src.height, Mathf.Max(1, src.packs * tiles));
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
        }
    }
}
