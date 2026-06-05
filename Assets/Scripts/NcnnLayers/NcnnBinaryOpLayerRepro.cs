using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
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
                                                    if (!owner.ShouldForceCurrentLayerBufferPath()
                                                        && NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var aTex, out var aTexShape)
                                                        && aTexShape.dims <= 3)
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

                                                    NcnnRepro.TensorRef aTex = null;
                                                    NcnnRepro.TensorRef bTex = null;
                                                    NcnnRepro.BufferShape aTexShape = default;
                                                    NcnnRepro.BufferShape bTexShape = default;
                                                    var canUseTextureBinary = !owner.ForceBufferBinaryOpAll
                                                        && owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out aTex, out aTexShape)
                                                        && owner.TryGetPack4Texture(layer.bottomNames[1], textureBlobs, textureShapes, bufferBlobs, bufferViews, out bTex, out bTexShape);

                                                    if (canUseTextureBinary
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
                                                    else if (!owner.ForceBufferBinaryOpAll
                                                             && !isTargetSftAddLayer
                                                             && !isTargetSftMulLayer
                                                             && TryResolvePack4BufferScalar(
                                                                 owner,
                                                                 layer.bottomNames[0],
                                                                 layer.bottomNames[1],
                                                                 textureBlobs,
                                                                 textureShapes,
                                                                 bufferBlobs,
                                                                 bufferViews,
                                                                 out var scalarTexture,
                                                                 out var scalarTextureShape,
                                                                 out var scalarBuffer,
                                                                 out var scalarIsA))
                                                    {
                                                        RenderTexture finalTexture = null;
                                                        try
                                                        {
                                                            finalTexture = owner.RentTempArray(scalarTexture.width, scalarTexture.height, scalarTexture.packs, RenderTextureFormat.ARGBHalf);
                                                            owner.Ops.BinaryOpPack4BufferScalar(scalarTexture.texture, scalarBuffer, scalarTexture.packs, opType, scalarIsA, finalTexture);
                                                            NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, scalarTextureShape);
                                                            finalTexture = null;
                                                        }
                                                        finally
                                                        {
                                                            if (finalTexture != null)
                                                                owner.ReturnTempArray(finalTexture);
                                                        }
                                                    }
                                                    else if (canUseTextureBinary
                                                             && !isTargetSftAddLayer
                                                             && !isTargetSftMulLayer
                                                             && TryResolvePack4SpatialBroadcast(aTex, aTexShape, bTex, bTexShape, out var broadcastMode, out var outShape, out var outWidth, out var outHeight, out var outPacks))
                                                    {
                                                        RenderTexture finalTexture = null;
                                                        try
                                                        {
                                                            finalTexture = owner.RentTempArray(outWidth, outHeight, outPacks, RenderTextureFormat.ARGBHalf);
                                                            owner.Ops.BinaryOpPack4Broadcast(aTex.texture, bTex.texture, outPacks, opType, broadcastMode, finalTexture);
                                                            NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, outShape);
                                                            finalTexture = null;
                                                        }
                                                        finally
                                                        {
                                                            if (finalTexture != null)
                                                                owner.ReturnTempArray(finalTexture);
                                                        }
                                                    }
                                                    else
                                                    {
                                                        using var aReadable = owner.GetReadableTensorInput(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                        var aBuf = aReadable?.buffer;
                                                        var aView = aReadable;
                                                        if (aBuf == null || aView == null)
                                                            throw new InvalidOperationException("BinaryOp source not found: " + layer.name);

                                                        using var bReadable = owner.GetReadableTensorInput(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                        var bBuf = bReadable?.buffer;
                                                        var bView = bReadable;
                                                        if (bBuf == null || bView == null)
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
                                                        var outTensor = broadcast.outputView != null
                                                            ? new NcnnTensorBuffer(outBuf, broadcast.outputView.dims, broadcast.outputView.w, broadcast.outputView.h, broadcast.outputView.d, broadcast.outputView.c, true, owner.ReturnTempBuffer)
                                                            : new NcnnTensorBuffer(outBuf, 1, outBuf.count, 1, 1, 1, true, owner.ReturnTempBuffer);
                                                        owner.PublishTensorBufferOutput(
                                                            layer.topNames[0],
                                                            outTensor,
                                                            preferTexture: broadcast.outputView != null && broadcast.outputView.dims <= 3,
                                                            textureBlobs,
                                                            textureShapes,
                                                            bufferBlobs,
                                                            bufferRefs,
                                                            bufferViews,
                                                            tempOwned);
                                                    }
                                                }

                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }

        private static bool TryResolvePack4SpatialBroadcast(
            NcnnRepro.TensorRef aTex,
            NcnnRepro.BufferShape aShape,
            NcnnRepro.TensorRef bTex,
            NcnnRepro.BufferShape bShape,
            out int broadcastMode,
            out NcnnRepro.BufferShape outShape,
            out int outWidth,
            out int outHeight,
            out int outPacks)
        {
            broadcastMode = 0;
            outShape = default;
            outWidth = 0;
            outHeight = 0;
            outPacks = 0;

            if (aTex == null || bTex == null || aTex.texture == null || bTex.texture == null)
                return false;
            if (aShape.dims != 3 || bShape.dims != 3)
                return false;
            if (aShape.c != bShape.c || aTex.packs != bTex.packs)
                return false;

            var aIsScalarSpatial = aShape.w == 1 && aShape.h == 1 && aTex.width == 1 && aTex.height == 1;
            var bIsScalarSpatial = bShape.w == 1 && bShape.h == 1 && bTex.width == 1 && bTex.height == 1;
            var aIsOutputSpatial = aShape.w == aTex.width && aShape.h == aTex.height;
            var bIsOutputSpatial = bShape.w == bTex.width && bShape.h == bTex.height;

            if (aIsScalarSpatial && !bIsScalarSpatial && bIsOutputSpatial)
            {
                broadcastMode = 1;
                outWidth = bTex.width;
                outHeight = bTex.height;
                outPacks = bTex.packs;
                outShape = new NcnnRepro.BufferShape(3, bShape.w, bShape.h, 1, bShape.c);
                return true;
            }

            if (bIsScalarSpatial && !aIsScalarSpatial && aIsOutputSpatial)
            {
                broadcastMode = 2;
                outWidth = aTex.width;
                outHeight = aTex.height;
                outPacks = aTex.packs;
                outShape = new NcnnRepro.BufferShape(3, aShape.w, aShape.h, 1, aShape.c);
                return true;
            }

            return false;
        }

        private static bool TryResolvePack4BufferScalar(
            NcnnRepro owner,
            string aName,
            string bName,
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            out NcnnRepro.TensorRef texture,
            out NcnnRepro.BufferShape textureShape,
            out ComputeBuffer scalar,
            out bool scalarIsA)
        {
            texture = null;
            textureShape = default;
            scalar = null;
            scalarIsA = false;

            if (owner == null)
                return false;

            if (TryGetSingleElementBuffer(bName, bufferBlobs, bufferViews, out var bScalar)
                && owner.TryGetPack4Texture(aName, textureBlobs, textureShapes, bufferBlobs, bufferViews, out var aTex, out var aShape))
            {
                texture = aTex;
                textureShape = aShape;
                scalar = bScalar;
                scalarIsA = false;
                return true;
            }

            if (TryGetSingleElementBuffer(aName, bufferBlobs, bufferViews, out var aScalar)
                && owner.TryGetPack4Texture(bName, textureBlobs, textureShapes, bufferBlobs, bufferViews, out var bTex, out var bShape))
            {
                texture = bTex;
                textureShape = bShape;
                scalar = aScalar;
                scalarIsA = true;
                return true;
            }

            return false;
        }

        private static bool TryGetSingleElementBuffer(
            string name,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            out ComputeBuffer scalar)
        {
            scalar = null;
            if (string.IsNullOrEmpty(name) || bufferBlobs == null)
                return false;
            if (!bufferBlobs.TryGetValue(name, out var buffer) || buffer == null || buffer.count < 1)
                return false;

            if (bufferViews != null && bufferViews.TryGetValue(name, out var view) && view != null)
            {
                if (view.elementCount != 1)
                    return false;
            }
            else if (buffer.count != 1)
            {
                return false;
            }

            scalar = buffer;
            return true;
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
                        var cmd = context.commandBuffer;
                        var blobs = context.blobs;
                        var shapes = context.shapes;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;

                        do
                        {
                                                var opType = layer.GetInt(0, 0);
                                                var withScalar = layer.GetInt(1, 0);
                                                var scalarB = layer.GetFloat(2, 0f);
                                                var a = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var aShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
                                                if (withScalar != 0)
                                                {
                                                    var outArr = owner.RentTempArray(cmd, a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                                                    owner.Ops.BinaryOpScalarPack4(cmd, a.texture, scalarB, a.packs, opType, outArr);
                                                    blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = a.width, height = a.height, packs = a.packs, refs = 1, owned = true };
                                                    if (shapes != null)
                                                        shapes[layer.topNames[0]] = aShape;
                                                }
                                                else
                                                {
                                                    var b = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[1]);
                                                    var bShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[1]);
                                                    if (CanUseExactCmdBinaryPath(a, aShape, b, bShape))
                                                    {
                                                        var outArr = owner.RentTempArray(cmd, a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                                                        owner.Ops.BinaryOpPack4(cmd, a.texture, b.texture, a.packs, opType, outArr);
                                                        blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = a.width, height = a.height, packs = a.packs, refs = 1, owned = true };
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = aShape;
                                                    }
                                                    else if (TryResolveCmdSpatialBroadcast(a, aShape, b, bShape, out var broadcastMode, out var outShape, out var outWidth, out var outHeight, out var outPacks))
                                                    {
                                                        var outArr = owner.RentTempArray(cmd, outWidth, outHeight, outPacks, RenderTextureFormat.ARGBHalf);
                                                        owner.Ops.BinaryOpPack4Broadcast(cmd, a.texture, b.texture, outPacks, opType, broadcastMode, outArr);
                                                        blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = outWidth, height = outHeight, packs = outPacks, refs = 1, owned = true };
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = outShape;
                                                    }
                                                    else
                                                    {
                                                        var fallbackShape = ResolveCmdOutputShape(aShape, bShape);
                                                        NcnnRepro.ResolveCmdTextureLayout(fallbackShape, out var width, out var height, out var packs);
                                                        owner.PublishCmdTensorLikeInput(cmd, layer.topNames[0], width, height, packs, blobs, shapes, fallbackShape);
                                                        owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                        continue;
                                                    }
                                                }
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }

        private static bool CanUseExactCmdBinaryPath(NcnnRepro.CmdTensorRef a, NcnnRepro.BufferShape aShape, NcnnRepro.CmdTensorRef b, NcnnRepro.BufferShape bShape)
        {
            return a != null
                && b != null
                && a.texture != null
                && b.texture != null
                && aShape.dims == 3
                && bShape.dims == 3
                && aShape.w == bShape.w
                && aShape.h == bShape.h
                && aShape.c == bShape.c
                && a.width == b.width
                && a.height == b.height
                && a.packs == b.packs;
        }

        private static bool TryResolveCmdSpatialBroadcast(
            NcnnRepro.CmdTensorRef a,
            NcnnRepro.BufferShape aShape,
            NcnnRepro.CmdTensorRef b,
            NcnnRepro.BufferShape bShape,
            out int broadcastMode,
            out NcnnRepro.BufferShape outShape,
            out int outWidth,
            out int outHeight,
            out int outPacks)
        {
            broadcastMode = 0;
            outShape = default;
            outWidth = 0;
            outHeight = 0;
            outPacks = 0;

            if (a == null || b == null || a.texture == null || b.texture == null)
                return false;
            if (aShape.dims != 3 || bShape.dims != 3)
                return false;
            if (aShape.c != bShape.c || a.packs != b.packs)
                return false;

            var aIsScalarSpatial = aShape.w == 1 && aShape.h == 1 && a.width == 1 && a.height == 1;
            var bIsScalarSpatial = bShape.w == 1 && bShape.h == 1 && b.width == 1 && b.height == 1;
            var aIsOutputSpatial = aShape.w == a.width && aShape.h == a.height;
            var bIsOutputSpatial = bShape.w == b.width && bShape.h == b.height;

            if (aIsScalarSpatial && !bIsScalarSpatial && bIsOutputSpatial)
            {
                broadcastMode = 1;
                outWidth = b.width;
                outHeight = b.height;
                outPacks = b.packs;
                outShape = new NcnnRepro.BufferShape(3, bShape.w, bShape.h, 1, bShape.c);
                return true;
            }

            if (bIsScalarSpatial && !aIsScalarSpatial && aIsOutputSpatial)
            {
                broadcastMode = 2;
                outWidth = a.width;
                outHeight = a.height;
                outPacks = a.packs;
                outShape = new NcnnRepro.BufferShape(3, aShape.w, aShape.h, 1, aShape.c);
                return true;
            }

            return false;
        }

        private static NcnnRepro.BufferShape ResolveCmdOutputShape(NcnnRepro.BufferShape aShape, NcnnRepro.BufferShape bShape)
        {
            var aCount = Mathf.Max(1, aShape.w * aShape.h * aShape.d * aShape.c);
            var bCount = Mathf.Max(1, bShape.w * bShape.h * bShape.d * bShape.c);
            if (aCount == bCount)
                return aShape;
            if (aCount < bCount && bCount % aCount == 0)
                return bShape;
            if (bCount < aCount && aCount % bCount == 0)
                return aShape;
            return aShape;
        }
    }
}
