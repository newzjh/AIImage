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

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteBinaryOpBufferLayer(layer, context);
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context) => owner.ExecuteBinaryOpCommandBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal void ExecuteBinaryOpBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                        if (TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var aTex, out var aTexShape))
                                        {
                                            var outRt = RentTempArray(aTex.width, aTex.height, aTex.packs, RenderTextureFormat.ARGBHalf);
                                            _ops.BinaryOpScalarPack4(aTex.texture, scalarB, aTex.packs, opType, outRt);
                                            SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, aTexShape);
                                        }
                                        else
                                        {
                                            var aBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                            var aView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                            if (aBuf == null)
                                                throw new InvalidOperationException("BinaryOp source not found: " + layer.name);

                                            var outBuf = RentTempBuffer(aBuf.count, sizeof(float));
                                            _ops.BinaryOpScalarBuf(aBuf, scalarB, aBuf.count, opType, outBuf);
                                            bufferBlobs[layer.topNames[0]] = outBuf;
                                            if (aView != null)
                                                bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, aView.dims, aView.w, aView.h, aView.d, aView.c, false);
                                            tempOwned.Add(outBuf);
                                        }
                                    }
                                    else
                                    {
                                        var isTargetSftAddLayer = string.IsNullOrEmpty(CodeFormerTargetSftAddLayer)
                                            ? CodeFormerSftAddLayers.Contains(layer.name)
                                            : string.Equals(layer.name, CodeFormerTargetSftAddLayer, StringComparison.Ordinal);
                                        var isTargetSftMulLayer = string.IsNullOrEmpty(CodeFormerTargetSftMulLayer)
                                            ? CodeFormerSftMulLayers.Contains(layer.name)
                                            : string.Equals(layer.name, CodeFormerTargetSftMulLayer, StringComparison.Ordinal);
                                        var isCodeFormerSftMul = opType == 2 && isTargetSftMulLayer;

                                        if (!ForceBufferBinaryOpAll
                                            && TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var aTex, out var aTexShape)
                                            && TryGetPack4Texture(layer.bottomNames[1], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var bTex, out var bTexShape)
                                            && CanUseExactPack4BinaryPath(aTex, aTexShape, bTex, bTexShape))
                                        {
                                            RenderTexture scaledBTexture = null;
                                            RenderTexture finalTexture = null;
                                            try
                                            {
                                                var rhsTexture = bTex.texture;
                                                if (CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                                                {
                                                    scaledBTexture = RentTempArray(bTex.width, bTex.height, bTex.packs, RenderTextureFormat.ARGBHalf);
                                                    _ops.ScalePack4(bTex.texture, CodeFormerSftAddScale, bTex.packs, scaledBTexture);
                                                    rhsTexture = scaledBTexture;
                                                }

                                                finalTexture = RentTempArray(aTex.width, aTex.height, aTex.packs, RenderTextureFormat.ARGBHalf);
                                                if (isCodeFormerSftMul && CodeFormerBypassSftMul)
                                                {
                                                    _ops.CopyPack4(aTex.texture, 0, finalTexture, 0, aTex.packs);
                                                }
                                                else
                                                {
                                                    _ops.BinaryOpPack4(aTex.texture, rhsTexture, aTex.packs, opType, finalTexture);
                                                    if (isCodeFormerSftMul && CodeFormerSftMulScale != 1f)
                                                    {
                                                        var scaledOutTexture = RentTempArray(aTex.width, aTex.height, aTex.packs, RenderTextureFormat.ARGBHalf);
                                                        _ops.ScalePack4(finalTexture, CodeFormerSftMulScale, aTex.packs, scaledOutTexture);
                                                        ReturnTempArray(finalTexture);
                                                        finalTexture = scaledOutTexture;
                                                    }
                                                }

                                                SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, aTexShape);
                                                finalTexture = null;
                                            }
                                            finally
                                            {
                                                if (scaledBTexture != null)
                                                    ReturnTempArray(scaledBTexture);
                                                if (finalTexture != null)
                                                    ReturnTempArray(finalTexture);
                                            }
                                        }
                                        else
                                        {
                                            var aBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                            var aView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                            if (aBuf == null)
                                                throw new InvalidOperationException("BinaryOp source not found: " + layer.name);

                                            var bBuf = GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                            var bView = TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
                                            if (bBuf == null)
                                                throw new InvalidOperationException("BinaryOp second source not found: " + layer.name);

                                            // ncnn reduction + binary-op chains in CodeFormer frequently mix [h,w] with [1,w] or [h,1].
                                            // Expanding the smaller 2d tensor explicitly avoids ambiguous modulo-based broadcasting.
                                            if (aView != null && bView != null && aView.dims == 2 && bView.dims == 2 && aBuf.count != bBuf.count)
                                            {
                                                if (TryExpand2DBroadcastBuffer(bBuf, bView, aView, out var expandedB, out var expandedBView))
                                                {
                                                    bBuf = expandedB;
                                                    bView = expandedBView;
                                                    tempOwned.Add(expandedB);
                                                }
                                                else if (TryExpand2DBroadcastBuffer(aBuf, aView, bView, out var expandedA, out var expandedAView))
                                                {
                                                    aBuf = expandedA;
                                                    aView = expandedAView;
                                                    tempOwned.Add(expandedA);
                                                }
                                            }
                                            else if (aView != null && bView != null && aView.dims == 1 && bView.dims == 2)
                                            {
                                                if (TryExpand1DTo2DBroadcastBuffer(aBuf, aView, bView, out var expandedA, out var expandedAView))
                                                {
                                                    aBuf = expandedA;
                                                    aView = expandedAView;
                                                    tempOwned.Add(expandedA);
                                                }
                                            }
                                            else if (aView != null && bView != null && aView.dims == 2 && bView.dims == 1)
                                            {
                                                if (TryExpand1DTo2DBroadcastBuffer(bBuf, bView, aView, out var expandedB, out var expandedBView))
                                                {
                                                    bBuf = expandedB;
                                                    bView = expandedBView;
                                                    tempOwned.Add(expandedB);
                                                }
                                            }
                                            else if (aView != null && bView != null && aView.dims == 3 && bView.dims == 3)
                                            {
                                                if (TryExpand3DBroadcastBuffer(bBuf, bView, aView, out var expandedB, out var expandedBView))
                                                {
                                                    bBuf = expandedB;
                                                    bView = expandedBView;
                                                    tempOwned.Add(expandedB);
                                                }
                                                else if (TryExpand3DBroadcastBuffer(aBuf, aView, bView, out var expandedA, out var expandedAView))
                                                {
                                                    aBuf = expandedA;
                                                    aView = expandedAView;
                                                    tempOwned.Add(expandedA);
                                                }
                                            }

                                            if (CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                                            {
                                                var scaledB = RentTempBuffer(bBuf.count, sizeof(float));
                                                _ops.CopyBuf(bBuf, scaledB, bBuf.count);
                                                _ops.MulScalarInplace(scaledB, CodeFormerSftAddScale, scaledB.count);
                                                bBuf = scaledB;
                                                tempOwned.Add(scaledB);
                                            }

                                            var broadcast = ResolveBinaryBroadcast(aView, bView, aBuf.count, bBuf.count, layer.name);
                                            var outBuf = RentTempBuffer(broadcast.total, sizeof(float));
                                            _ops.BinaryOpBuf(aBuf, bBuf, broadcast.total, opType, outBuf, broadcast.mode, broadcast.size);
                                            if (isCodeFormerSftMul)
                                            {
                                                if (CodeFormerBypassSftMul)
                                                {
                                                    _ops.CopyBuf(aBuf, outBuf, broadcast.total);
                                                }
                                                else if (CodeFormerSftMulScale != 1f)
                                                {
                                                    _ops.MulScalarInplace(outBuf, CodeFormerSftMulScale, outBuf.count);
                                                }
                                            }
                                            bufferBlobs[layer.topNames[0]] = outBuf;
                                            if (broadcast.outputView != null)
                                                bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, broadcast.outputView.dims, broadcast.outputView.w, broadcast.outputView.h, broadcast.outputView.d, broadcast.outputView.c, false);
                                            tempOwned.Add(outBuf);
                                        }
                                    }

                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }

        internal void ExecuteBinaryOpCommandBufferLayer(NcnnParamModel.Layer l, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            do
            {
                                    var opType = l.GetInt(0, 0);
                                    var withScalar = l.GetInt(1, 0);
                                    var scalarB = l.GetFloat(2, 0f);
                                    var a = GetCmdTensor(blobs, l.bottomNames[0]);
                                    var outArr = RentTempArray(cmd, a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                                    if (withScalar != 0)
                                    {
                                        _ops.BinaryOpScalarPack4(cmd, a.texture, scalarB, a.packs, opType, outArr);
                                    }
                                    else
                                    {
                                        var b = GetCmdTensor(blobs, l.bottomNames[1]);
                                        if (a.width != b.width || a.height != b.height || a.packs != b.packs)
                                            throw new InvalidOperationException("BinaryOp broadcast not supported: " + l.name);
                                        _ops.BinaryOpPack4(cmd, a.texture, b.texture, a.packs, opType, outArr);
                                    }
                                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = a.width, height = a.height, packs = a.packs, refs = 1, owned = true };
                                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
