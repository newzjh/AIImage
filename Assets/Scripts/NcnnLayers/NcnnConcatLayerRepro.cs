using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
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
                        var indexBlobs = context.indexBlobs;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;
                        var tempOwned = context.tempOwned;

                        do
                        {
                                                var firstBufferView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                if (firstBufferView != null && firstBufferView.dims == 1)
                                                {
                                                    var concatAxis = layer.GetInt(0, 0);
                                                    if (concatAxis != 0)
                                                        throw new InvalidOperationException("Concat dims=1 only supports axis=0: " + layer.name);

                                                    var totalCount = 0;
                                                    for (var i = 0; i < layer.bottomNames.Length; i++)
                                                    {
                                                        var partView = NcnnRepro.TryGetBufferView(layer.bottomNames[i], bufferBlobs, bufferViews);
                                                        if (partView == null || partView.dims != 1)
                                                            throw new InvalidOperationException("Concat dims=1 source missing: " + layer.name + " | " + layer.bottomNames[i]);
                                                        totalCount += partView.w;
                                                    }

                                                    var outBuf = owner.RentTempBuffer(totalCount, sizeof(float));
                                                    var dstOffset = 0;
                                                    for (var i = 0; i < layer.bottomNames.Length; i++)
                                                    {
                                                        var partBuf = owner.GetOrConvertToBuffer(layer.bottomNames[i], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                        var partView = NcnnRepro.TryGetBufferView(layer.bottomNames[i], bufferBlobs, bufferViews);
                                                        if (partBuf == null || partView == null)
                                                            throw new InvalidOperationException("Concat dims=1 buffer source missing: " + layer.name + " | " + layer.bottomNames[i]);
                                                        owner.Ops.CopyBufPartial(partBuf, 0, outBuf, partView.w, dstOffset);
                                                        dstOffset += partView.w;
                                                    }

                                                    bufferBlobs[layer.topNames[0]] = outBuf;
                                                    bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], outBuf);
                                                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, 1, totalCount, 1, 1, 1, false);
                                                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                    continue;
                                                }

                                                if (firstBufferView != null && firstBufferView.dims == 2)
                                                {
                                                    var concatAxis = layer.GetInt(0, 0);
                                                    if (concatAxis != 0)
                                                        throw new InvalidOperationException("Concat dims=2 only supports axis=0: " + layer.name);

                                                    var totalRows = 0;
                                                    var outWidth = firstBufferView.w;
                                                    for (var i = 0; i < layer.bottomNames.Length; i++)
                                                    {
                                                        var partView = NcnnRepro.TryGetBufferView(layer.bottomNames[i], bufferBlobs, bufferViews);
                                                        if (partView == null || partView.dims != 2)
                                                            throw new InvalidOperationException("Concat dims=2 source missing: " + layer.name + " | " + layer.bottomNames[i]);
                                                        if (partView.w != outWidth)
                                                            throw new InvalidOperationException("Concat dims=2 width mismatch: " + layer.name);
                                                        totalRows += partView.h;
                                                    }

                                                    var outCount = outWidth * totalRows;
                                                    var outBuf = owner.RentTempBuffer(outCount, sizeof(float));
                                                    var dstOffset = 0;
                                                    for (var i = 0; i < layer.bottomNames.Length; i++)
                                                    {
                                                        var partBuf = owner.GetOrConvertToBuffer(layer.bottomNames[i], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                        var partView = NcnnRepro.TryGetBufferView(layer.bottomNames[i], bufferBlobs, bufferViews);
                                                        if (partBuf == null || partView == null)
                                                            throw new InvalidOperationException("Concat dims=2 buffer source missing: " + layer.name + " | " + layer.bottomNames[i]);
                                                        var partCount = partView.w * partView.h;
                                                        owner.Ops.CopyBufPartial(partBuf, 0, outBuf, partCount, dstOffset);
                                                        dstOffset += partCount;
                                                    }

                                                    bufferBlobs[layer.topNames[0]] = outBuf;
                                                    bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], outBuf);
                                                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, 2, outWidth, totalRows, 1, 1, false);
                                                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                    continue;
                                                }

                                                var first = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                var axis = layer.GetInt(0, 0);
                                                if (axis != 0)
                                                    throw new InvalidOperationException("Concat only supports channel axis for texture tensors: " + layer.name);

                                                var totalPacks = 0;
                                                var totalLogicalChannels = 0;
                                                var canStayTexture = true;
                                                for (var i = 0; i < layer.bottomNames.Length; i++)
                                                {
                                                    var tr = owner.GetOrMaterializeTexture(layer.bottomNames[i], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                    if (tr.width != first.width || tr.height != first.height)
                                                        throw new InvalidOperationException("Concat shape mismatch: " + layer.name);
                                                    var logicalShape = NcnnRepro.GetTextureShape(textureShapes, tr, layer.bottomNames[i]);
                                                    if (i < layer.bottomNames.Length - 1 && (logicalShape.c % 4) != 0)
                                                        canStayTexture = false;
                                                    totalPacks += tr.packs;
                                                    totalLogicalChannels += logicalShape.c;
                                                }

                                                if (canStayTexture)
                                                {
                                                    var outRt = owner.RentTempArray(first.width, first.height, totalPacks, RenderTextureFormat.ARGBHalf);
                                                    var packOffset = 0;
                                                    for (var i = 0; i < layer.bottomNames.Length; i++)
                                                    {
                                                        var part = owner.GetOrMaterializeTexture(layer.bottomNames[i], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                        owner.Ops.CopyPack4(part.texture, 0, outRt, packOffset, part.packs);
                                                        packOffset += part.packs;
                                                    }

                                                    textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
                                                    {
                                                        texture = outRt,
                                                        width = first.width,
                                                        height = first.height,
                                                        packs = totalPacks,
                                                        refs = 1,
                                                        owned = true
                                                    };
                                                    textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, first.width, first.height, 1, totalLogicalChannels);
                                                }
                                                else
                                                {
                                                    var featureSize = first.width * first.height;
                                                    var outCount = featureSize * totalLogicalChannels;
                                                    var outBuf = owner.RentTempBuffer(outCount, sizeof(float));
                                                    var dstOffset = 0;
                                                    for (var i = 0; i < layer.bottomNames.Length; i++)
                                                    {
                                                        var partBuf = owner.GetOrConvertToBuffer(layer.bottomNames[i], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                        var partView = NcnnRepro.TryGetBufferView(layer.bottomNames[i], bufferBlobs, bufferViews);
                                                        if (partBuf == null || partView == null)
                                                            throw new InvalidOperationException("Concat buffer fallback source not found: " + layer.name + " | " + layer.bottomNames[i]);

                                                        var partCount = partView.w * partView.h * partView.d * partView.c;
                                                        owner.Ops.CopyBufPartial(partBuf, 0, outBuf, partCount, dstOffset);
                                                        dstOffset += partCount;
                                                    }

                                                    bufferBlobs[layer.topNames[0]] = outBuf;
                                                    bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], outBuf);
                                                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, 3, first.width, first.height, 1, totalLogicalChannels, false);
                                                }

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
