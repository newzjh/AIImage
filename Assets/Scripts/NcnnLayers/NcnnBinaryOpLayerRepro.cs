using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnBinaryOpLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnBinaryOpLayerRepro() : base(NcnnLayerTypes.BinaryOp, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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
                                                var opType = layer.GetInt(0, 0);
                                                var withScalar = layer.GetInt(1, 0);
                                                var scalarB = layer.GetFloat(2, 0f);

                                                if (withScalar != 0)
                                                {
                                                    if (owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var aTex, out var aTexShape))
                                                    {
                                                        var outRt = owner.RentTempArray(aTex.width, aTex.height, aTex.packs, RenderTextureFormat.ARGBHalf);
                                                        owner.Ops.BinaryOpScalarPack4(aTex.texture, scalarB, aTex.packs, opType, outRt);
                                                        NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, aTexShape);
                                                    }
                                                    else
                                                    {
                                                        var aBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                        var aView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                        if (aBuf == null)
                                                            throw new InvalidOperationException("BinaryOp source not found: " + layer.name);

                                                        var outBuf = owner.RentTempBuffer(aBuf.count, sizeof(float));
                                                        owner.Ops.BinaryOpScalarBuf(aBuf, scalarB, aBuf.count, opType, outBuf);
                                                        bufferBlobs[layer.topNames[0]] = outBuf;
                                                        if (aView != null)
                                                            bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, aView.dims, aView.w, aView.h, aView.d, aView.c, false);
                                                        tempOwned.Add(outBuf);
                                                    }
                                                }
                                                else
                                                {
                                                    var isTargetSftAddLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftAddLayer)
                                                        ? NcnnRepro.CodeFormerSftAddLayers.Contains(layer.name)
                                                        : string.Equals(layer.name, owner.CodeFormerTargetSftAddLayer, StringComparison.Ordinal);
                                                    var isTargetSftMulLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftMulLayer)
                                                        ? NcnnRepro.CodeFormerSftMulLayers.Contains(layer.name)
                                                        : string.Equals(layer.name, owner.CodeFormerTargetSftMulLayer, StringComparison.Ordinal);
                                                    var isCodeFormerSftMul = opType == 2 && isTargetSftMulLayer;

                                                    if (!owner.ForceBufferBinaryOpAll
                                                        && owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var aTex, out var aTexShape)
                                                        && owner.TryGetPack4Texture(layer.bottomNames[1], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var bTex, out var bTexShape)
                                                        && NcnnRepro.CanUseExactPack4BinaryPath(aTex, aTexShape, bTex, bTexShape))
                                                    {
                                                        RenderTexture scaledBTexture = null;
                                                        RenderTexture finalTexture = null;
                                                        try
                                                        {
                                                            var rhsTexture = bTex.texture;
                                                            if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                                                            {
                                                                scaledBTexture = owner.RentTempArray(bTex.width, bTex.height, bTex.packs, RenderTextureFormat.ARGBHalf);
                                                                owner.Ops.ScalePack4(bTex.texture, owner.CodeFormerSftAddScale, bTex.packs, scaledBTexture);
                                                                rhsTexture = scaledBTexture;
                                                            }

                                                            finalTexture = owner.RentTempArray(aTex.width, aTex.height, aTex.packs, RenderTextureFormat.ARGBHalf);
                                                            if (isCodeFormerSftMul && owner.CodeFormerBypassSftMul)
                                                            {
                                                                owner.Ops.CopyPack4(aTex.texture, 0, finalTexture, 0, aTex.packs);
                                                            }
                                                            else
                                                            {
                                                                owner.Ops.BinaryOpPack4(aTex.texture, rhsTexture, aTex.packs, opType, finalTexture);
                                                                if (isCodeFormerSftMul && owner.CodeFormerSftMulScale != 1f)
                                                                {
                                                                    var scaledOutTexture = owner.RentTempArray(aTex.width, aTex.height, aTex.packs, RenderTextureFormat.ARGBHalf);
                                                                    owner.Ops.ScalePack4(finalTexture, owner.CodeFormerSftMulScale, aTex.packs, scaledOutTexture);
                                                                    owner.ReturnTempArray(finalTexture);
                                                                    finalTexture = scaledOutTexture;
                                                                }
                                                            }

                                                            NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, aTexShape);
                                                            finalTexture = null;
                                                        }
                                                        finally
                                                        {
                                                            if (scaledBTexture != null)
                                                                owner.ReturnTempArray(scaledBTexture);
                                                            if (finalTexture != null)
                                                                owner.ReturnTempArray(finalTexture);
                                                        }
                                                    }
                                                    else
                                                    {
                                                        var aBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                        var aView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                        if (aBuf == null)
                                                            throw new InvalidOperationException("BinaryOp source not found: " + layer.name);

                                                        var bBuf = owner.GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                        var bView = NcnnRepro.TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
                                                        if (bBuf == null)
                                                            throw new InvalidOperationException("BinaryOp second source not found: " + layer.name);

                                                        // ncnn reduction + binary-op chains in CodeFormer frequently mix [h,w] with [1,w] or [h,1].
                                                        // Expanding the smaller 2d tensor explicitly avoids ambiguous modulo-based broadcasting.
                                                        if (aView != null && bView != null && aView.dims == 2 && bView.dims == 2 && aBuf.count != bBuf.count)
                                                        {
                                                            if (owner.TryExpand2DBroadcastBuffer(bBuf, bView, aView, out var expandedB, out var expandedBView))
                                                            {
                                                                bBuf = expandedB;
                                                                bView = expandedBView;
                                                                tempOwned.Add(expandedB);
                                                            }
                                                            else if (owner.TryExpand2DBroadcastBuffer(aBuf, aView, bView, out var expandedA, out var expandedAView))
                                                            {
                                                                aBuf = expandedA;
                                                                aView = expandedAView;
                                                                tempOwned.Add(expandedA);
                                                            }
                                                        }
                                                        else if (aView != null && bView != null && aView.dims == 1 && bView.dims == 2)
                                                        {
                                                            if (owner.TryExpand1DTo2DBroadcastBuffer(aBuf, aView, bView, out var expandedA, out var expandedAView))
                                                            {
                                                                aBuf = expandedA;
                                                                aView = expandedAView;
                                                                tempOwned.Add(expandedA);
                                                            }
                                                        }
                                                        else if (aView != null && bView != null && aView.dims == 2 && bView.dims == 1)
                                                        {
                                                            if (owner.TryExpand1DTo2DBroadcastBuffer(bBuf, bView, aView, out var expandedB, out var expandedBView))
                                                            {
                                                                bBuf = expandedB;
                                                                bView = expandedBView;
                                                                tempOwned.Add(expandedB);
                                                            }
                                                        }
                                                        else if (aView != null && bView != null && aView.dims == 3 && bView.dims == 3)
                                                        {
                                                            if (owner.TryExpand3DBroadcastBuffer(bBuf, bView, aView, out var expandedB, out var expandedBView))
                                                            {
                                                                bBuf = expandedB;
                                                                bView = expandedBView;
                                                                tempOwned.Add(expandedB);
                                                            }
                                                            else if (owner.TryExpand3DBroadcastBuffer(aBuf, aView, bView, out var expandedA, out var expandedAView))
                                                            {
                                                                aBuf = expandedA;
                                                                aView = expandedAView;
                                                                tempOwned.Add(expandedA);
                                                            }
                                                        }

                                                        if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                                                        {
                                                            var scaledB = owner.RentTempBuffer(bBuf.count, sizeof(float));
                                                            owner.Ops.CopyBuf(bBuf, scaledB, bBuf.count);
                                                            owner.Ops.MulScalarInplace(scaledB, owner.CodeFormerSftAddScale, scaledB.count);
                                                            bBuf = scaledB;
                                                            tempOwned.Add(scaledB);
                                                        }

                                                        var broadcast = NcnnRepro.ResolveBinaryBroadcast(aView, bView, aBuf.count, bBuf.count, layer.name);
                                                        var outBuf = owner.RentTempBuffer(broadcast.total, sizeof(float));
                                                        owner.Ops.BinaryOpBuf(aBuf, bBuf, broadcast.total, opType, outBuf, broadcast.mode, broadcast.size);
                                                        if (isCodeFormerSftMul)
                                                        {
                                                            if (owner.CodeFormerBypassSftMul)
                                                            {
                                                                owner.Ops.CopyBuf(aBuf, outBuf, broadcast.total);
                                                            }
                                                            else if (owner.CodeFormerSftMulScale != 1f)
                                                            {
                                                                owner.Ops.MulScalarInplace(outBuf, owner.CodeFormerSftMulScale, outBuf.count);
                                                            }
                                                        }
                                                        bufferBlobs[layer.topNames[0]] = outBuf;
                                                        if (broadcast.outputView != null)
                                                            bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, broadcast.outputView.dims, broadcast.outputView.w, broadcast.outputView.h, broadcast.outputView.d, broadcast.outputView.c, false);
                                                        tempOwned.Add(outBuf);
                                                    }
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
                                                var opType = layer.GetInt(0, 0);
                                                var withScalar = layer.GetInt(1, 0);
                                                var scalarB = layer.GetFloat(2, 0f);
                                                var a = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var outArr = owner.RentTempArray(cmd, a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                                                if (withScalar != 0)
                                                {
                                                    owner.Ops.BinaryOpScalarPack4(cmd, a.texture, scalarB, a.packs, opType, outArr);
                                                }
                                                else
                                                {
                                                    var b = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[1]);
                                                    if (a.width != b.width || a.height != b.height || a.packs != b.packs)
                                                        throw new InvalidOperationException("BinaryOp broadcast not supported: " + layer.name);
                                                    owner.Ops.BinaryOpPack4(cmd, a.texture, b.texture, a.packs, opType, outArr);
                                                }
                                                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = a.width, height = a.height, packs = a.packs, refs = 1, owned = true };
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
