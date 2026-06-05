using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
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
                                                    var hasTexture = textureBlobs.TryGetValue(layer.bottomNames[0], out var tileTex) && tileTex != null && tileTex.texture != null;
                                                    var tileTexShape = hasTexture ? NcnnRepro.GetTextureShape(textureShapes, tileTex, layer.bottomNames[0]) : default;
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

                                                        if (hasTexture)
                                                        {
                                                            textureBlobs[layer.topNames[0]] = tileTex;
                                                            textureShapes[layer.topNames[0]] = tileTexShape;
                                                            tileTex.refs++;
                                                        }
                                                    }
                                                    else
                                                    {
                                                        var src = hasTexture ? tileTex : owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                        textureBlobs[layer.topNames[0]] = src;
                                                        textureShapes[layer.topNames[0]] = hasTexture ? tileTexShape : NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
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

                                                var outTensor = owner.RentTempTensorBuffer(srcView2.dims, outW, outH, outD, outC);
                                                owner.Ops.Tile(srcBuf2, srcView2.dims, srcView2.w, srcView2.h, srcView2.d, srcView2.c, tensorAxis, tiles, outW, outH, outD, outC, outTensor.buffer);

                                                owner.PublishTensorBufferOutput(
                                                    layer.topNames[0],
                                                    outTensor,
                                                    preferTexture: srcView2.dims <= 3,
                                                    textureBlobs,
                                                    textureShapes,
                                                    bufferBlobs,
                                                    bufferRefs,
                                                    bufferViews,
                                                    tempOwned);
                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var hasAxis = layer.intParams != null && layer.intParams.ContainsKey(0);
            var hasTiles = layer.intParams != null && layer.intParams.ContainsKey(1);
            var tiles = layer.GetInt(1, 1);
            if ((!hasAxis && !hasTiles) || tiles <= 1)
            {
                blobs[layer.topNames[0]] = src;
                if (shapes != null)
                    shapes[layer.topNames[0]] = srcShape;
                src.refs++;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            var axis = layer.GetInt(0, 0);
            if (axis < 0)
                axis += srcShape.dims;
            if (axis < 0 || axis >= srcShape.dims)
                throw new InvalidOperationException("Tile axis out of range: " + layer.name);

            var tensorAxis = NcnnRepro.MapNcnnAxisToTensorAxis(srcShape.dims, axis);
            var outShape = srcShape;
            if (tensorAxis == 0) outShape = new NcnnRepro.BufferShape(srcShape.dims, srcShape.w * tiles, srcShape.h, srcShape.d, srcShape.c);
            else if (tensorAxis == 1) outShape = new NcnnRepro.BufferShape(srcShape.dims, srcShape.w, srcShape.h * tiles, srcShape.d, srcShape.c);
            else if (tensorAxis == 2 && srcShape.dims == 4) outShape = new NcnnRepro.BufferShape(srcShape.dims, srcShape.w, srcShape.h, srcShape.d * tiles, srcShape.c);
            else outShape = new NcnnRepro.BufferShape(srcShape.dims, srcShape.w, srcShape.h, srcShape.d, srcShape.c * tiles);

            owner.PublishCmdPlaceholder(cmd, layer.topNames[0], outShape, blobs, shapes);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
