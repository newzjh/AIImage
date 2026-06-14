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
            if (CanExecuteRenderTexturePath(owner, layer, context))
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            var opType = layer.GetInt(0, 0);
            var withScalar = layer.GetInt(1, 0);
            var scalarB = layer.GetFloat(2, 0f);

            if (withScalar != 0)
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
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            var isTargetSftAddLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftAddLayer)
                ? NcnnRepro.CodeFormerSftAddLayers.Contains(layer.name)
                : string.Equals(layer.name, owner.CodeFormerTargetSftAddLayer, StringComparison.Ordinal);
            var isTargetSftMulLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftMulLayer)
                ? NcnnRepro.CodeFormerSftMulLayers.Contains(layer.name)
                : string.Equals(layer.name, owner.CodeFormerTargetSftMulLayer, StringComparison.Ordinal);
            var isCodeFormerSftMul = opType == 2 && isTargetSftMulLayer;

            using var aReadable = owner.GetReadableTensorInput(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var aBufFallback = aReadable?.buffer;
            var aViewFallback = aReadable;
            if (aBufFallback == null || aViewFallback == null)
                throw new InvalidOperationException("BinaryOp source not found: " + layer.name);

            using var bReadable = owner.GetReadableTensorInput(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var bBufFallback = bReadable?.buffer;
            var bViewFallback = bReadable;
            if (bBufFallback == null || bViewFallback == null)
                throw new InvalidOperationException("BinaryOp second source not found: " + layer.name);

            if (aViewFallback.dims == 2 && bViewFallback.dims == 2 && aBufFallback.count != bBufFallback.count)
            {
                if (owner.TryExpand2DBroadcastBuffer(bBufFallback, bViewFallback, aViewFallback, out var expandedB, out var expandedBView))
                {
                    bBufFallback = expandedB;
                    bViewFallback = expandedBView;
                    tempOwned.Add(expandedB);
                }
                else if (owner.TryExpand2DBroadcastBuffer(aBufFallback, aViewFallback, bViewFallback, out var expandedA, out var expandedAView))
                {
                    aBufFallback = expandedA;
                    aViewFallback = expandedAView;
                    tempOwned.Add(expandedA);
                }
            }
            else if (aViewFallback.dims == 1 && bViewFallback.dims == 2)
            {
                if (owner.TryExpand1DTo2DBroadcastBuffer(aBufFallback, aViewFallback, bViewFallback, out var expandedA, out var expandedAView))
                {
                    aBufFallback = expandedA;
                    aViewFallback = expandedAView;
                    tempOwned.Add(expandedA);
                }
            }
            else if (aViewFallback.dims == 2 && bViewFallback.dims == 1)
            {
                if (owner.TryExpand1DTo2DBroadcastBuffer(bBufFallback, bViewFallback, aViewFallback, out var expandedB, out var expandedBView))
                {
                    bBufFallback = expandedB;
                    bViewFallback = expandedBView;
                    tempOwned.Add(expandedB);
                }
            }
            else if (aViewFallback.dims == 3 && bViewFallback.dims == 3)
            {
                if (owner.TryExpand3DBroadcastBuffer(bBufFallback, bViewFallback, aViewFallback, out var expandedB, out var expandedBView))
                {
                    bBufFallback = expandedB;
                    bViewFallback = expandedBView;
                    tempOwned.Add(expandedB);
                }
                else if (owner.TryExpand3DBroadcastBuffer(aBufFallback, aViewFallback, bViewFallback, out var expandedA, out var expandedAView))
                {
                    aBufFallback = expandedA;
                    aViewFallback = expandedAView;
                    tempOwned.Add(expandedA);
                }
            }

            if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
            {
                var scaledB = owner.RentTempBuffer(bBufFallback.count, sizeof(float));
                owner.Ops.CopyBuf(bBufFallback, scaledB, bBufFallback.count);
                owner.Ops.MulScalarInplace(scaledB, owner.CodeFormerSftAddScale, scaledB.count);
                bBufFallback = scaledB;
                tempOwned.Add(scaledB);
            }

            var broadcast = NcnnRepro.ResolveBinaryBroadcast(aViewFallback, bViewFallback, aBufFallback.count, bBufFallback.count, layer.name);
            var outBufFallback = owner.RentTempBuffer(broadcast.total, sizeof(float));
            owner.Ops.BinaryOpBuf(aBufFallback, bBufFallback, broadcast.total, opType, outBufFallback, broadcast.mode, broadcast.size);
            if (isCodeFormerSftMul)
            {
                if (owner.CodeFormerBypassSftMul)
                {
                    owner.Ops.CopyBuf(aBufFallback, outBufFallback, broadcast.total);
                }
                else if (owner.CodeFormerSftMulScale != 1f)
                {
                    owner.Ops.MulScalarInplace(outBufFallback, owner.CodeFormerSftMulScale, outBufFallback.count);
                }
            }

            var outTensorFallback = broadcast.outputView != null
                ? new NcnnTensorBuffer(outBufFallback, broadcast.outputView.dims, broadcast.outputView.w, broadcast.outputView.h, broadcast.outputView.d, broadcast.outputView.c, true, owner.ReturnTempBuffer)
                : new NcnnTensorBuffer(outBufFallback, 1, outBufFallback.count, 1, 1, 1, true, owner.ReturnTempBuffer);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensorFallback,
                preferTexture: broadcast.outputView != null && broadcast.outputView.dims <= 3,
                textureBlobs,
                textureShapes,
                bufferBlobs,
                bufferRefs,
                bufferViews,
                tempOwned);
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var opType = layer.GetInt(0, 0);
            var withScalar = layer.GetInt(1, 0);
            var scalarB = layer.GetFloat(2, 0f);
            var scalarTextureFormat = NcnnRepro.ResolveTensorTextureFormat(2);

            if (withScalar != 0)
            {
                if (owner.ShouldForceCurrentLayerBufferPath()
                    || !NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var aTexScalar, out var aTexShapeScalar)
                    || !CanUseScalarLikeTexturePath(aTexScalar, aTexShapeScalar))
                {
                    throw new InvalidOperationException("BinaryOp render-texture scalar path requires existing texture input: " + layer.name);
                }

                var outRtScalar = owner.RentTempArray(
                    aTexScalar.width,
                    aTexScalar.height,
                    Mathf.Max(1, aTexScalar.texture.volumeDepth),
                    NcnnRepro.ResolveTensorTextureFormat(aTexShapeScalar.dims));
                owner.Ops.BinaryOpScalarPack4(aTexScalar.texture, scalarB, aTexScalar.packs, opType, outRtScalar);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRtScalar, aTexShapeScalar);
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

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

            if (canUseTextureBinary && NcnnRepro.CanUseExactPack4BinaryPath(aTex, aTexShape, bTex, bTexShape))
            {
                RenderTexture scaledBTexture = null;
                RenderTexture finalTexture = null;
                try
                {
                    var rhsTexture = bTex.texture;
                    if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                    {
                        var scaledBDepth = bTexShape.dims == 4 ? bTexShape.d * bTex.packs : bTex.packs;
                        scaledBTexture = owner.RentTempArray(bTex.width, bTex.height, scaledBDepth, RenderTextureFormat.ARGBHalf);
                        owner.Ops.ScalePack4(bTex.texture, owner.CodeFormerSftAddScale, bTex.packs, scaledBTexture);
                        rhsTexture = scaledBTexture;
                    }

                    var exactOutDepth = aTexShape.dims == 4 ? aTexShape.d * aTex.packs : aTex.packs;
                    finalTexture = owner.RentTempArray(aTex.width, aTex.height, exactOutDepth, RenderTextureFormat.ARGBHalf);
                    if (isCodeFormerSftMul && owner.CodeFormerBypassSftMul)
                    {
                        owner.Ops.CopyPack4(aTex.texture, 0, finalTexture, 0, aTex.packs);
                    }
                    else
                    {
                        owner.Ops.BinaryOpPack4(aTex.texture, rhsTexture, aTex.packs, opType, finalTexture);
                        if (isCodeFormerSftMul && owner.CodeFormerSftMulScale != 1f)
                        {
                            var scaledOutTexture = owner.RentTempArray(aTex.width, aTex.height, exactOutDepth, RenderTextureFormat.ARGBHalf);
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

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (TryResolveExactScalar2DTextureBinaryPath(textureBlobs, textureShapes, layer.bottomNames[0], layer.bottomNames[1], out aTex, out aTexShape, out bTex, out bTexShape))
            {
                RenderTexture scaledBTexture = null;
                RenderTexture finalTexture = null;
                try
                {
                    var rhsTexture = bTex.texture;
                    if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                    {
                        scaledBTexture = owner.RentTempArray(bTex.width, bTex.height, bTex.packs, scalarTextureFormat);
                        owner.Ops.ScalePack4(bTex.texture, owner.CodeFormerSftAddScale, bTex.packs, scaledBTexture);
                        rhsTexture = scaledBTexture;
                    }

                    finalTexture = owner.RentTempArray(aTex.width, aTex.height, aTex.packs, scalarTextureFormat);
                    if (isCodeFormerSftMul && owner.CodeFormerBypassSftMul)
                    {
                        owner.Ops.CopyPack4(aTex.texture, 0, finalTexture, 0, aTex.packs);
                    }
                    else
                    {
                        owner.Ops.BinaryOpPack4(aTex.texture, rhsTexture, aTex.packs, opType, finalTexture);
                        if (isCodeFormerSftMul && owner.CodeFormerSftMulScale != 1f)
                        {
                            var scaledOutTexture = owner.RentTempArray(aTex.width, aTex.height, aTex.packs, scalarTextureFormat);
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

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (TryResolveScalarSingleBroadcastTextureBinaryPath(
                    textureBlobs,
                    textureShapes,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    out var scalarBroadA,
                    out var scalarBroadAShape,
                    out var scalarBroadB,
                    out var scalarBroadBShape,
                    out var scalarBroadcastMode,
                    out var scalarBroadcastOutShape,
                    out var scalarBroadcastStorageShape))
            {
                RenderTexture scaledBTexture = null;
                RenderTexture finalTexture = null;
                try
                {
                    var rhsTexture = scalarBroadB.texture;
                    if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                    {
                        scaledBTexture = owner.RentTempArray(scalarBroadB.width, scalarBroadB.height, scalarBroadB.packs, scalarTextureFormat);
                        owner.Ops.ScalePack4(scalarBroadB.texture, owner.CodeFormerSftAddScale, scalarBroadB.packs, scaledBTexture);
                        rhsTexture = scaledBTexture;
                    }

                    finalTexture = owner.RentTempArray(
                        scalarBroadcastStorageShape.w,
                        scalarBroadcastStorageShape.h,
                        1,
                        scalarTextureFormat);

                    if (isCodeFormerSftMul && owner.CodeFormerBypassSftMul)
                    {
                        var zeroTexture = owner.RentTempArray(rhsTexture.width, rhsTexture.height, 1, scalarTextureFormat);
                        owner.Ops.BinaryOpScalarPack4(rhsTexture, 0f, 1, 2, zeroTexture);
                        owner.Ops.BinaryOpScalarSingleBroadcast(
                            scalarBroadA.texture,
                            zeroTexture,
                            scalarBroadcastStorageShape.w,
                            scalarBroadcastStorageShape.h,
                            0,
                            scalarBroadcastMode,
                            finalTexture);
                        owner.ReturnTempArray(zeroTexture);
                    }
                    else
                    {
                        owner.Ops.BinaryOpScalarSingleBroadcast(
                            scalarBroadA.texture,
                            rhsTexture,
                            scalarBroadcastStorageShape.w,
                            scalarBroadcastStorageShape.h,
                            opType,
                            scalarBroadcastMode,
                            finalTexture);
                        if (isCodeFormerSftMul && owner.CodeFormerSftMulScale != 1f)
                        {
                            var scaledOutTexture = owner.RentTempArray(finalTexture.width, finalTexture.height, 1, scalarTextureFormat);
                            owner.Ops.ScalePack4(finalTexture, owner.CodeFormerSftMulScale, 1, scaledOutTexture);
                            owner.ReturnTempArray(finalTexture);
                            finalTexture = scaledOutTexture;
                        }
                    }

                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, scalarBroadcastOutShape, scalarBroadcastStorageShape);
                    finalTexture = null;
                }
                finally
                {
                    if (scaledBTexture != null)
                        owner.ReturnTempArray(scaledBTexture);
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (!owner.ForceBufferBinaryOpAll
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4TextureChannelVectorBroadcast(
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    textureBlobs,
                    textureShapes,
                    out var vectorTextureTex,
                    out var vectorTextureTexShape,
                    out var channelVectorTextureTex,
                    out var channelVectorTextureIsA))
            {
                RenderTexture finalTexture = null;
                try
                {
                    var vectorOutDepth = vectorTextureTexShape.dims == 4 ? vectorTextureTexShape.d * vectorTextureTex.packs : vectorTextureTex.packs;
                    finalTexture = owner.RentTempArray(vectorTextureTex.width, vectorTextureTex.height, vectorOutDepth, RenderTextureFormat.ARGBHalf);
                    owner.Ops.BinaryOpPack4ChannelVectorTex(
                        vectorTextureTex.texture,
                        channelVectorTextureTex.texture,
                        vectorTextureTex.packs,
                        opType,
                        channelVectorTextureIsA,
                        finalTexture);
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, vectorTextureTexShape);
                    finalTexture = null;
                }
                finally
                {
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (!owner.ForceBufferBinaryOpAll
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4ChannelVectorBroadcast(
                    owner,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    textureBlobs,
                    textureShapes,
                    bufferBlobs,
                    bufferViews,
                    out var vectorTexture,
                    out var vectorTextureShape,
                    out var channelVectorView,
                    out var channelVectorIsA))
            {
                RenderTexture vectorBroadcastTexture = null;
                RenderTexture finalTexture = null;
                try
                {
                    var packedChannelView = channelVectorView.Reshape(3, 1, 1, 1, vectorTextureShape.c);
                    vectorBroadcastTexture = owner.MaterializeTextureFromBufferView(channelVectorView.buffer, packedChannelView);
                    if (vectorBroadcastTexture == null)
                        throw new InvalidOperationException("Failed to materialize BinaryOp channel vector texture: " + layer.name);

                    var vectorOutDepth = vectorTextureShape.dims == 4 ? vectorTextureShape.d * vectorTexture.packs : vectorTexture.packs;
                    finalTexture = owner.RentTempArray(vectorTexture.width, vectorTexture.height, vectorOutDepth, RenderTextureFormat.ARGBHalf);
                    if (channelVectorIsA)
                        owner.Ops.BinaryOpPack4Broadcast(vectorBroadcastTexture, vectorTexture.texture, vectorTexture.packs, opType, 1, finalTexture);
                    else
                        owner.Ops.BinaryOpPack4Broadcast(vectorTexture.texture, vectorBroadcastTexture, vectorTexture.packs, opType, 2, finalTexture);
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, vectorTextureShape);
                    finalTexture = null;
                }
                finally
                {
                    if (vectorBroadcastTexture != null)
                        owner.ReturnTempArray(vectorBroadcastTexture);
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (!owner.ForceBufferBinaryOpAll
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
                    var scalarOutDepth = scalarTextureShape.dims == 4 ? scalarTextureShape.d * scalarTexture.packs : scalarTexture.packs;
                    finalTexture = owner.RentTempArray(scalarTexture.width, scalarTexture.height, scalarOutDepth, RenderTextureFormat.ARGBHalf);
                    owner.Ops.BinaryOpPack4BufferScalar(scalarTexture.texture, scalarBuffer, scalarTexture.packs, opType, scalarIsA, finalTexture);
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, scalarTextureShape);
                    finalTexture = null;
                }
                finally
                {
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (canUseTextureBinary
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4SpatialBroadcast(aTex, aTexShape, bTex, bTexShape, out var broadcastMode, out var outShape, out var outWidth, out var outHeight, out var outPacks))
            {
                RenderTexture finalTexture = null;
                try
                {
                    var spatialOutDepth = outShape.dims == 4 ? outShape.d * outPacks : outPacks;
                    finalTexture = owner.RentTempArray(outWidth, outHeight, spatialOutDepth, RenderTextureFormat.ARGBHalf);
                    owner.Ops.BinaryOpPack4Broadcast(aTex.texture, bTex.texture, outPacks, opType, broadcastMode, finalTexture);
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, outShape);
                    finalTexture = null;
                }
                finally
                {
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            throw new InvalidOperationException("BinaryOp render-texture path unsupported config: " + layer.name);
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
            if ((aShape.dims != 3 && aShape.dims != 4) || aShape.dims != bShape.dims)
                return false;
            if (aShape.c != bShape.c || aTex.packs != bTex.packs)
                return false;

            var aIsScalarSpatial = aShape.w == 1 && aShape.h == 1 && aShape.d == 1 && aTex.width == 1 && aTex.height == 1;
            var bIsScalarSpatial = bShape.w == 1 && bShape.h == 1 && bShape.d == 1 && bTex.width == 1 && bTex.height == 1;
            var aIsOutputSpatial = aShape.w == aTex.width && aShape.h == aTex.height;
            var bIsOutputSpatial = bShape.w == bTex.width && bShape.h == bTex.height;

            if (aIsScalarSpatial && !bIsScalarSpatial && bIsOutputSpatial)
            {
                broadcastMode = 1;
                outWidth = bTex.width;
                outHeight = bTex.height;
                outPacks = bTex.packs;
                outShape = new NcnnRepro.BufferShape(bShape.dims, bShape.w, bShape.h, bShape.d, bShape.c);
                return true;
            }

            if (bIsScalarSpatial && !aIsScalarSpatial && aIsOutputSpatial)
            {
                broadcastMode = 2;
                outWidth = aTex.width;
                outHeight = aTex.height;
                outPacks = aTex.packs;
                outShape = new NcnnRepro.BufferShape(aShape.dims, aShape.w, aShape.h, aShape.d, aShape.c);
                return true;
            }

            return false;
        }

        private static bool TryResolvePack4ChannelVectorBroadcast(
            NcnnRepro owner,
            string aName,
            string bName,
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            out NcnnRepro.TensorRef texture,
            out NcnnRepro.BufferShape textureShape,
            out NcnnTensorBuffer channelVector,
            out bool channelVectorIsA)
        {
            texture = null;
            textureShape = default;
            channelVector = null;
            channelVectorIsA = false;

            if (owner == null)
                return false;

            if (TryGetChannelVectorBuffer(bName, bufferBlobs, bufferViews, out var bVector)
                && owner.TryGetPack4Texture(aName, textureBlobs, textureShapes, bufferBlobs, bufferViews, out var aTex, out var aShape)
                && aShape.dims == 3
                && bVector.elementCount == aShape.c)
            {
                texture = aTex;
                textureShape = aShape;
                channelVector = bVector;
                channelVectorIsA = false;
                return true;
            }

            if (TryGetChannelVectorBuffer(aName, bufferBlobs, bufferViews, out var aVector)
                && owner.TryGetPack4Texture(bName, textureBlobs, textureShapes, bufferBlobs, bufferViews, out var bTex, out var bShape)
                && bShape.dims == 3
                && aVector.elementCount == bShape.c)
            {
                texture = bTex;
                textureShape = bShape;
                channelVector = aVector;
                channelVectorIsA = true;
                return true;
            }

            return false;
        }

        private static bool TryResolvePack4TextureChannelVectorBroadcast(
            string aName,
            string bName,
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            out NcnnRepro.TensorRef texture,
            out NcnnRepro.BufferShape textureShape,
            out NcnnRepro.TensorRef channelVector,
            out bool channelVectorIsA)
        {
            texture = null;
            textureShape = default;
            channelVector = null;
            channelVectorIsA = false;

            if (TryGetExistingChannelVectorTexture(textureBlobs, textureShapes, bName, out var bVector, out var bLogicalShape, out _)
                && NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, aName, out var aTex, out var aShape)
                && (aShape.dims == 3 || aShape.dims == 4)
                && NcnnRepro.MatchesPack4TextureStorage(aTex, aShape)
                && MatchesChannelVectorWidth(bLogicalShape, aShape.c))
            {
                texture = aTex;
                textureShape = aShape;
                channelVector = bVector;
                channelVectorIsA = false;
                return true;
            }

            if (TryGetExistingChannelVectorTexture(textureBlobs, textureShapes, aName, out var aVector, out var aLogicalShape, out _)
                && NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, bName, out var bTex, out var bShape)
                && (bShape.dims == 3 || bShape.dims == 4)
                && NcnnRepro.MatchesPack4TextureStorage(bTex, bShape)
                && MatchesChannelVectorWidth(aLogicalShape, bShape.c))
            {
                texture = bTex;
                textureShape = bShape;
                channelVector = aVector;
                channelVectorIsA = true;
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

        private static bool TryResolveExactScalar2DTextureBinaryPath(
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            string aName,
            string bName,
            out NcnnRepro.TensorRef aTex,
            out NcnnRepro.BufferShape aShape,
            out NcnnRepro.TensorRef bTex,
            out NcnnRepro.BufferShape bShape)
        {
            aTex = null;
            aShape = default;
            bTex = null;
            bShape = default;

            if (!TryGetScalar2DTexture(textureBlobs, textureShapes, aName, out aTex, out aShape)
                || !TryGetScalar2DTexture(textureBlobs, textureShapes, bName, out bTex, out bShape))
            {
                return false;
            }

            return aShape.w == bShape.w
                && aShape.h == bShape.h
                && aShape.d == bShape.d
                && aShape.c == bShape.c
                && aTex.width == bTex.width
                && aTex.height == bTex.height
                && aTex.packs == bTex.packs;
        }

        private static bool TryResolveScalarSingleBroadcastTextureBinaryPath(
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            string aName,
            string bName,
            out NcnnRepro.TensorRef aTex,
            out NcnnRepro.BufferShape aShape,
            out NcnnRepro.TensorRef bTex,
            out NcnnRepro.BufferShape bShape,
            out int broadcastMode,
            out NcnnRepro.BufferShape outShape,
            out NcnnRepro.BufferShape storageShape)
        {
            aTex = null;
            aShape = default;
            bTex = null;
            bShape = default;
            broadcastMode = 0;
            outShape = default;
            storageShape = default;

            if (!TryGetScalarLikeTexture(textureBlobs, textureShapes, aName, out aTex, out aShape)
                || !TryGetScalarLikeTexture(textureBlobs, textureShapes, bName, out bTex, out bShape))
            {
                return false;
            }

            return TryResolveScalarSingleBroadcastShapes(aShape, bShape, out broadcastMode, out outShape, out storageShape);
        }

        private static bool TryGetScalar2DTexture(
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            string name,
            out NcnnRepro.TensorRef texture,
            out NcnnRepro.BufferShape shape)
        {
            if (!NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, name, out texture, out shape))
                return false;
            return CanUseScalar2DTexturePath(texture, shape);
        }

        private static bool TryGetScalarLikeTexture(
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            string name,
            out NcnnRepro.TensorRef texture,
            out NcnnRepro.BufferShape shape)
        {
            texture = null;
            shape = default;
            if (!NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, name, out texture, out shape))
                return false;
            return CanUseScalarLikeTexturePath(texture, shape);
        }

        private static bool TryGetExistingChannelVectorTexture(
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            string name,
            out NcnnRepro.TensorRef texture,
            out NcnnRepro.BufferShape logicalShape,
            out NcnnRepro.BufferShape storageShape)
        {
            texture = null;
            logicalShape = default;
            storageShape = default;
            if (!NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, name, out texture, out logicalShape))
                return false;
            storageShape = NcnnRepro.GetTextureStorageShape(texture, logicalShape);
            return CanUseChannelVectorTexturePath(texture, logicalShape, storageShape);
        }

        private static bool CanUseScalar2DTexturePath(NcnnRepro.TensorRef texture, NcnnRepro.BufferShape shape)
        {
            return texture != null
                && texture.texture != null
                && shape.dims == 2
                && shape.w > 0
                && shape.h > 0
                && texture.width == shape.w
                && texture.height == shape.h
                && texture.packs == 1;
        }

        private static bool CanUseScalarLikeTexturePath(NcnnRepro.TensorRef texture, NcnnRepro.BufferShape shape)
        {
            if (texture == null || texture.texture == null)
                return false;
            if (texture.packs == 1
                && ((shape.dims == 1 && shape.w > 0 && texture.width == shape.w && texture.height == 1)
                    || (shape.dims == 2 && shape.w > 0 && shape.h > 0 && texture.width == shape.w && texture.height == shape.h)))
            {
                return true;
            }

            return (shape.dims == 3 || shape.dims == 4)
                && shape.w > 0
                && shape.h > 0
                && shape.c > 0
                && NcnnRepro.MatchesPack4TextureStorage(texture, shape);
        }

        private static bool CanUseChannelVectorTexturePath(NcnnRepro.TensorRef texture, NcnnRepro.BufferShape logicalShape, NcnnRepro.BufferShape storageShape)
        {
            if (texture == null || texture.texture == null || texture.packs != 1)
                return false;
            if (logicalShape.dims != 1 && logicalShape.dims != 2)
                return false;
            if (logicalShape.w <= 0)
                return false;
            if (storageShape.w == logicalShape.w && storageShape.h == 1 && texture.width == logicalShape.w && texture.height == 1)
                return true;
            if (storageShape.w == 1 && storageShape.h == logicalShape.w && texture.width == 1 && texture.height == logicalShape.w)
                return true;
            return false;
        }

        private static bool CanUseScalar2DTexturePath(NcnnRepro.CmdTensorRef texture, NcnnRepro.BufferShape shape)
        {
            return texture != null
                && texture.texture != null
                && shape.dims == 2
                && shape.w > 0
                && shape.h > 0
                && texture.width == shape.w
                && texture.height == shape.h
                && texture.packs == 1;
        }

        private static bool CanUseScalarLikeTexturePath(NcnnRepro.CmdTensorRef texture, NcnnRepro.BufferShape shape)
        {
            if (texture == null || texture.texture == null)
                return false;
            if (texture.packs == 1
                && ((shape.dims == 1 && shape.w > 0 && texture.width == shape.w && texture.height == 1)
                    || (shape.dims == 2 && shape.w > 0 && shape.h > 0 && texture.width == shape.w && texture.height == shape.h)))
            {
                return true;
            }

            return (shape.dims == 3 || shape.dims == 4)
                && shape.w > 0
                && shape.h > 0
                && shape.c > 0
                && NcnnRepro.MatchesPack4TextureStorage(texture, shape);
        }

        private static bool CanUseChannelVectorTexturePath(NcnnRepro.CmdTensorRef texture, NcnnRepro.BufferShape logicalShape, NcnnRepro.BufferShape storageShape)
        {
            if (texture == null || texture.texture == null || texture.packs != 1)
                return false;
            if (logicalShape.dims != 1 && logicalShape.dims != 2)
                return false;
            if (logicalShape.w <= 0)
                return false;
            if (storageShape.w == logicalShape.w && storageShape.h == 1 && texture.width == logicalShape.w && texture.height == 1)
                return true;
            if (storageShape.w == 1 && storageShape.h == logicalShape.w && texture.width == 1 && texture.height == logicalShape.w)
                return true;
            return false;
        }

        private static bool MatchesChannelVectorWidth(NcnnRepro.BufferShape logicalShape, int channels)
        {
            return logicalShape.w == channels
                && (logicalShape.dims == 1 || (logicalShape.dims == 2 && logicalShape.h == 1));
        }

        private static bool TryResolveScalarSingleBroadcastCmdBinaryPath(
            NcnnRepro.CmdTensorRef aTex,
            NcnnRepro.BufferShape aShape,
            NcnnRepro.CmdTensorRef bTex,
            NcnnRepro.BufferShape bShape,
            out NcnnRepro.CmdTensorRef outATex,
            out NcnnRepro.CmdTensorRef outBTex,
            out int broadcastMode,
            out NcnnRepro.BufferShape outShape,
            out NcnnRepro.BufferShape storageShape)
        {
            outATex = null;
            outBTex = null;
            broadcastMode = 0;
            outShape = default;
            storageShape = default;

            if (!CanUseScalarLikeTexturePath(aTex, aShape) || !CanUseScalarLikeTexturePath(bTex, bShape))
                return false;
            if (!TryResolveScalarSingleBroadcastShapes(aShape, bShape, out broadcastMode, out outShape, out storageShape))
                return false;

            outATex = aTex;
            outBTex = bTex;
            return true;
        }

        private static bool TryResolveCmdPack4TextureChannelVectorBroadcast(
            NcnnRepro.CmdTensorRef a,
            NcnnRepro.BufferShape aShape,
            NcnnRepro.CmdTensorRef b,
            NcnnRepro.BufferShape bShape,
            out NcnnRepro.CmdTensorRef texture,
            out NcnnRepro.BufferShape textureShape,
            out NcnnRepro.CmdTensorRef channelVector,
            out bool channelVectorIsA)
        {
            texture = null;
            textureShape = default;
            channelVector = null;
            channelVectorIsA = false;

            if ((aShape.dims == 3 || aShape.dims == 4)
                && NcnnRepro.MatchesPack4TextureStorage(a, aShape)
                && CanUseChannelVectorTexturePath(b, bShape, NcnnRepro.GetCmdStorageShape(b, bShape))
                && MatchesChannelVectorWidth(bShape, aShape.c))
            {
                texture = a;
                textureShape = aShape;
                channelVector = b;
                channelVectorIsA = false;
                return true;
            }

            if ((bShape.dims == 3 || bShape.dims == 4)
                && NcnnRepro.MatchesPack4TextureStorage(b, bShape)
                && CanUseChannelVectorTexturePath(a, aShape, NcnnRepro.GetCmdStorageShape(a, aShape))
                && MatchesChannelVectorWidth(aShape, bShape.c))
            {
                texture = b;
                textureShape = bShape;
                channelVector = a;
                channelVectorIsA = true;
                return true;
            }

            return false;
        }

        private static bool TryResolveScalarSingleBroadcastShapes(
            NcnnRepro.BufferShape aShape,
            NcnnRepro.BufferShape bShape,
            out int broadcastMode,
            out NcnnRepro.BufferShape outShape,
            out NcnnRepro.BufferShape storageShape)
        {
            broadcastMode = 0;
            outShape = default;
            storageShape = default;

            static int GetRows(NcnnRepro.BufferShape shape) => shape.dims == 1 ? 1 : shape.h;
            static int GetCols(NcnnRepro.BufferShape shape) => shape.w;
            static int ElementCount(NcnnRepro.BufferShape shape) => Mathf.Max(1, shape.w) * Mathf.Max(1, shape.h) * Mathf.Max(1, shape.d) * Mathf.Max(1, shape.c);
            static bool IsScalarSingle(NcnnRepro.BufferShape shape) => shape.dims >= 1 && shape.dims <= 2 && ElementCount(shape) == 1;
            static bool CanUseScalarSingleOutput(NcnnRepro.BufferShape shape) => shape.dims == 1 || shape.dims == 2;
            static NcnnRepro.BufferShape ScalarSingleStorage(NcnnRepro.BufferShape shape)
            {
                var width = Mathf.Max(1, shape.w);
                var height = shape.dims == 1 ? 1 : Mathf.Max(1, shape.h);
                return new NcnnRepro.BufferShape(3, width, height, 1, 1);
            }
            static bool IsRowVector(NcnnRepro.BufferShape shape) => shape.dims == 2 && shape.h == 1 && shape.w > 0;
            static bool IsColumnVector(NcnnRepro.BufferShape shape) => shape.dims == 2 && shape.w == 1 && shape.h > 0;

            var aRows = GetRows(aShape);
            var aCols = GetCols(aShape);
            var bRows = GetRows(bShape);
            var bCols = GetCols(bShape);

            if (IsScalarSingle(aShape) && !IsScalarSingle(bShape) && CanUseScalarSingleOutput(bShape))
            {
                broadcastMode = 5;
                outShape = bShape;
                storageShape = ScalarSingleStorage(bShape);
                return true;
            }

            if (IsScalarSingle(bShape) && !IsScalarSingle(aShape) && CanUseScalarSingleOutput(aShape))
            {
                broadcastMode = 6;
                outShape = aShape;
                storageShape = ScalarSingleStorage(aShape);
                return true;
            }

            if (aShape.dims == 2 && bShape.dims == 1 && aRows == bCols && aCols > 1)
            {
                broadcastMode = 4;
                outShape = aShape;
                storageShape = new NcnnRepro.BufferShape(3, aShape.w, aShape.h, 1, 1);
                return true;
            }

            if (aShape.dims == 1 && bShape.dims == 2 && aCols == bRows && bCols > 1)
            {
                broadcastMode = 2;
                outShape = bShape;
                storageShape = new NcnnRepro.BufferShape(3, bShape.w, bShape.h, 1, 1);
                return true;
            }

            if (aShape.dims == 2 && IsRowVector(bShape) && aCols == bCols && aRows > 1)
            {
                broadcastMode = 3;
                outShape = aShape;
                storageShape = new NcnnRepro.BufferShape(3, aShape.w, aShape.h, 1, 1);
                return true;
            }

            if (IsRowVector(aShape) && bShape.dims == 2 && aCols == bCols && bRows > 1)
            {
                broadcastMode = 1;
                outShape = bShape;
                storageShape = new NcnnRepro.BufferShape(3, bShape.w, bShape.h, 1, 1);
                return true;
            }

            if (aShape.dims == 2 && IsColumnVector(bShape) && aRows == bRows && aCols > 1)
            {
                broadcastMode = 4;
                outShape = aShape;
                storageShape = new NcnnRepro.BufferShape(3, aShape.w, aShape.h, 1, 1);
                return true;
            }

            if (IsColumnVector(aShape) && bShape.dims == 2 && aRows == bRows && bCols > 1)
            {
                broadcastMode = 2;
                outShape = bShape;
                storageShape = new NcnnRepro.BufferShape(3, bShape.w, bShape.h, 1, 1);
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

        private static bool TryGetChannelVectorBuffer(
            string name,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            out NcnnTensorBuffer vector)
        {
            vector = null;
            if (string.IsNullOrEmpty(name) || bufferBlobs == null || bufferViews == null)
                return false;
            if (!bufferBlobs.TryGetValue(name, out var buffer) || buffer == null)
                return false;
            if (!bufferViews.TryGetValue(name, out var view) || view == null || view.buffer == null)
                return false;

            var isVector =
                view.dims == 1
                || (view.dims == 2 && (view.w == 1 || view.h == 1))
                || (view.dims == 3 && view.w == 1 && view.h == 1)
                || (view.dims == 4 && view.w == 1 && view.h == 1 && view.d == 1);
            if (!isVector)
                return false;
            if (view.elementCount <= 1)
                return false;

            vector = view;
            return true;
        }

        private static bool CanExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;

            var withScalar = layer.GetInt(1, 0);

            if (withScalar != 0)
            {
                return !owner.ShouldForceCurrentLayerBufferPath()
                    && NcnnRepro.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var scalarTex, out var scalarTexShape)
                    && CanUseScalarLikeTexturePath(scalarTex, scalarTexShape);
            }

            var isTargetSftAddLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftAddLayer)
                ? NcnnRepro.CodeFormerSftAddLayers.Contains(layer.name)
                : string.Equals(layer.name, owner.CodeFormerTargetSftAddLayer, StringComparison.Ordinal);
            var isTargetSftMulLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftMulLayer)
                ? NcnnRepro.CodeFormerSftMulLayers.Contains(layer.name)
                : string.Equals(layer.name, owner.CodeFormerTargetSftMulLayer, StringComparison.Ordinal);

            NcnnRepro.TensorRef aTex = null;
            NcnnRepro.TensorRef bTex = null;
            NcnnRepro.BufferShape aTexShape = default;
            NcnnRepro.BufferShape bTexShape = default;
            var canUseTextureBinary = !owner.ForceBufferBinaryOpAll
                && owner.TryGetPack4Texture(layer.bottomNames[0], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out aTex, out aTexShape)
                && owner.TryGetPack4Texture(layer.bottomNames[1], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out bTex, out bTexShape);

            if (canUseTextureBinary && NcnnRepro.CanUseExactPack4BinaryPath(aTex, aTexShape, bTex, bTexShape))
                return true;

            if (TryResolveExactScalar2DTextureBinaryPath(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    out _,
                    out _,
                    out _,
                    out _))
            {
                return true;
            }

            if (TryResolveScalarSingleBroadcastTextureBinaryPath(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    out _,
                    out _,
                    out _,
                    out _,
                    out _,
                    out _,
                    out _))
            {
                return true;
            }

            if (!owner.ForceBufferBinaryOpAll
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4TextureChannelVectorBroadcast(
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    context.textureBlobs,
                    context.textureShapes,
                    out _,
                    out _,
                    out _,
                    out _))
            {
                return true;
            }

            if (!owner.ForceBufferBinaryOpAll
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4ChannelVectorBroadcast(
                    owner,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out _,
                    out _,
                    out _,
                    out _))
            {
                return true;
            }

            if (!owner.ForceBufferBinaryOpAll
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4BufferScalar(
                    owner,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out _,
                    out _,
                    out _,
                    out _))
            {
                return true;
            }

            return canUseTextureBinary
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4SpatialBroadcast(aTex, aTexShape, bTex, bTexShape, out _, out _, out _, out _, out _);
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
                                                var isTargetSftAddLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftAddLayer)
                                                    ? NcnnRepro.CodeFormerSftAddLayers.Contains(layer.name)
                                                    : string.Equals(layer.name, owner.CodeFormerTargetSftAddLayer, StringComparison.Ordinal);
                                                var isTargetSftMulLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftMulLayer)
                                                    ? NcnnRepro.CodeFormerSftMulLayers.Contains(layer.name)
                                                    : string.Equals(layer.name, owner.CodeFormerTargetSftMulLayer, StringComparison.Ordinal);
                                                var isCodeFormerSftMul = opType == 2 && isTargetSftMulLayer;
                                                var scalarTextureFormat = NcnnRepro.ResolveTensorTextureFormat(2);
                                                var a = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var aShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
                                                if (withScalar != 0)
                                                {
                                                    if (!CanUseScalarLikeTexturePath(a, aShape))
                                                        break;
                                                    var outDepth = Mathf.Max(1, a.texture.depth);
                                                    var outArr = owner.RentTempArray(cmd, a.width, a.height, outDepth, NcnnRepro.ResolveTensorTextureFormat(aShape.dims));
                                                    owner.Ops.BinaryOpScalarPack4(cmd, a.texture, scalarB, a.packs, opType, outArr);
                                                    blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                                                    {
                                                        texture = outArr,
                                                        width = a.width,
                                                        height = a.height,
                                                        packs = a.packs,
                                                        refs = 1,
                                                        owned = true,
                                                        hasLogicalShape = true,
                                                        logicalShape = aShape,
                                                        hasStorageShape = true,
                                                        storageShape = NcnnRepro.GetCmdStorageShape(a, aShape)
                                                    };
                                                    if (shapes != null)
                                                        shapes[layer.topNames[0]] = aShape;
                                                }
                                                else
                                                {
                                                    var b = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[1]);
                                                    var bShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[1]);
                                                    if (CanUseExactCmdBinaryPath(a, aShape, b, bShape))
                                                    {
                                                        var outDepth = aShape.dims == 4 ? Mathf.Max(1, aShape.d) * a.packs : a.packs;
                                                        var outArr = owner.RentTempArray(cmd, a.width, a.height, outDepth, RenderTextureFormat.ARGBHalf);
                                                        owner.Ops.BinaryOpPack4(cmd, a.texture, b.texture, a.packs, opType, outArr);
                                                        blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                                                        {
                                                            texture = outArr,
                                                            width = a.width,
                                                            height = a.height,
                                                            packs = a.packs,
                                                            refs = 1,
                                                            owned = true,
                                                            hasLogicalShape = true,
                                                            logicalShape = aShape,
                                                            hasStorageShape = true,
                                                            storageShape = aShape
                                                        };
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = aShape;
                                                    }
                                                    else if (CanUseScalar2DTexturePath(a, aShape)
                                                        && CanUseScalar2DTexturePath(b, bShape)
                                                        && aShape.w == bShape.w
                                                        && aShape.h == bShape.h
                                                        && aShape.d == bShape.d
                                                        && aShape.c == bShape.c
                                                        && a.width == b.width
                                                        && a.height == b.height
                                                        && a.packs == b.packs)
                                                    {
                                                        ComputeTexture scaledB = null;
                                                        ComputeTexture finalOut = null;
                                                        try
                                                        {
                                                            var rhs = b.texture;
                                                            if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                                                            {
                                                                scaledB = owner.RentTempArray(cmd, b.width, b.height, b.packs, scalarTextureFormat);
                                                                owner.Ops.BinaryOpScalarPack4(cmd, b.texture, owner.CodeFormerSftAddScale, b.packs, 2, scaledB);
                                                                rhs = scaledB;
                                                            }

                                                            finalOut = owner.RentTempArray(cmd, a.width, a.height, a.packs, scalarTextureFormat);
                                                            if (isCodeFormerSftMul && owner.CodeFormerBypassSftMul)
                                                            {
                                                                owner.Ops.CopyPack4(cmd, a.texture, 0, finalOut, 0, a.packs);
                                                            }
                                                            else
                                                            {
                                                                owner.Ops.BinaryOpPack4(cmd, a.texture, rhs, a.packs, opType, finalOut);
                                                                if (isCodeFormerSftMul && owner.CodeFormerSftMulScale != 1f)
                                                                {
                                                                    var scaledOut = owner.RentTempArray(cmd, a.width, a.height, a.packs, scalarTextureFormat);
                                                                    owner.Ops.BinaryOpScalarPack4(cmd, finalOut, owner.CodeFormerSftMulScale, a.packs, 2, scaledOut);
                                                                    owner.ReturnTempArray(cmd, finalOut);
                                                                    finalOut = scaledOut;
                                                                }
                                                            }

                                                            blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                                                            {
                                                                texture = finalOut,
                                                                width = a.width,
                                                                height = a.height,
                                                                packs = a.packs,
                                                                refs = 1,
                                                                owned = true,
                                                                hasLogicalShape = true,
                                                                logicalShape = aShape,
                                                                hasStorageShape = true,
                                                                storageShape = aShape
                                                            };
                                                            if (shapes != null)
                                                                shapes[layer.topNames[0]] = aShape;
                                                            finalOut = null;
                                                        }
                                                        finally
                                                        {
                                                            if (scaledB != null)
                                                                owner.ReturnTempArray(cmd, scaledB);
                                                            if (finalOut != null)
                                                                owner.ReturnTempArray(cmd, finalOut);
                                                        }
                                                    }
                                                    else if (TryResolveScalarSingleBroadcastCmdBinaryPath(
                                                        a,
                                                        aShape,
                                                        b,
                                                        bShape,
                                                        out var scalarBroadCmdA,
                                                        out var scalarBroadCmdB,
                                                        out var scalarBroadcastCmdMode,
                                                        out var scalarBroadcastCmdOutShape,
                                                        out var scalarBroadcastCmdStorageShape))
                                                    {
                                                        ComputeTexture scaledB = null;
                                                        ComputeTexture finalOut = null;
                                                        try
                                                        {
                                                            var rhs = scalarBroadCmdB.texture;
                                                            if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                                                            {
                                                                scaledB = owner.RentTempArray(cmd, scalarBroadCmdB.width, scalarBroadCmdB.height, scalarBroadCmdB.packs, scalarTextureFormat);
                                                                owner.Ops.BinaryOpScalarPack4(cmd, scalarBroadCmdB.texture, owner.CodeFormerSftAddScale, scalarBroadCmdB.packs, 2, scaledB);
                                                                rhs = scaledB;
                                                            }

                                                            finalOut = owner.RentTempArray(cmd, scalarBroadcastCmdStorageShape.w, scalarBroadcastCmdStorageShape.h, 1, scalarTextureFormat);
                                                            if (isCodeFormerSftMul && owner.CodeFormerBypassSftMul)
                                                            {
                                                                var zeroTexture = owner.RentTempArray(cmd, scalarBroadCmdB.width, scalarBroadCmdB.height, 1, scalarTextureFormat);
                                                                owner.Ops.BinaryOpScalarPack4(cmd, scalarBroadCmdB.texture, 0f, 1, 2, zeroTexture);
                                                                owner.Ops.BinaryOpScalarSingleBroadcast(
                                                                    cmd,
                                                                    scalarBroadCmdA.texture,
                                                                    zeroTexture,
                                                                    scalarBroadcastCmdStorageShape.w,
                                                                    scalarBroadcastCmdStorageShape.h,
                                                                    0,
                                                                    scalarBroadcastCmdMode,
                                                                    finalOut);
                                                                owner.ReturnTempArray(cmd, zeroTexture);
                                                            }
                                                            else
                                                            {
                                                                owner.Ops.BinaryOpScalarSingleBroadcast(
                                                                    cmd,
                                                                    scalarBroadCmdA.texture,
                                                                    rhs,
                                                                    scalarBroadcastCmdStorageShape.w,
                                                                    scalarBroadcastCmdStorageShape.h,
                                                                    opType,
                                                                    scalarBroadcastCmdMode,
                                                                    finalOut);
                                                                if (isCodeFormerSftMul && owner.CodeFormerSftMulScale != 1f)
                                                                {
                                                                    var scaledOut = owner.RentTempArray(cmd, finalOut.width, finalOut.height, 1, scalarTextureFormat);
                                                                    owner.Ops.BinaryOpScalarPack4(cmd, finalOut, owner.CodeFormerSftMulScale, 1, 2, scaledOut);
                                                                    owner.ReturnTempArray(cmd, finalOut);
                                                                    finalOut = scaledOut;
                                                                }
                                                            }

                                                            blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                                                            {
                                                                texture = finalOut,
                                                                width = scalarBroadcastCmdStorageShape.w,
                                                                height = scalarBroadcastCmdStorageShape.h,
                                                                packs = 1,
                                                                refs = 1,
                                                                owned = true,
                                                                hasLogicalShape = true,
                                                                logicalShape = scalarBroadcastCmdOutShape,
                                                                hasStorageShape = true,
                                                                storageShape = scalarBroadcastCmdStorageShape
                                                            };
                                                            if (shapes != null)
                                                                shapes[layer.topNames[0]] = scalarBroadcastCmdOutShape;
                                                            finalOut = null;
                                                        }
                                                        finally
                                                        {
                                                            if (scaledB != null)
                                                                owner.ReturnTempArray(cmd, scaledB);
                                                            if (finalOut != null)
                                                                owner.ReturnTempArray(cmd, finalOut);
                                                        }
                                                    }
                                                    else if (!owner.ForceBufferBinaryOpAll
                                                        && !isTargetSftAddLayer
                                                        && !isTargetSftMulLayer
                                                        && TryResolveCmdPack4TextureChannelVectorBroadcast(
                                                            a,
                                                            aShape,
                                                            b,
                                                            bShape,
                                                            out var vectorCmdTexture,
                                                            out var vectorCmdTextureShape,
                                                            out var channelVectorCmdTexture,
                                                            out var channelVectorCmdIsA))
                                                    {
                                                        var outDepth = vectorCmdTextureShape.dims == 4
                                                            ? Mathf.Max(1, vectorCmdTextureShape.d) * vectorCmdTexture.packs
                                                            : vectorCmdTexture.packs;
                                                        var outArr = owner.RentTempArray(cmd, vectorCmdTexture.width, vectorCmdTexture.height, outDepth, RenderTextureFormat.ARGBHalf);
                                                        owner.Ops.BinaryOpPack4ChannelVectorTex(
                                                            cmd,
                                                            vectorCmdTexture.texture,
                                                            channelVectorCmdTexture.texture,
                                                            vectorCmdTexture.packs,
                                                            opType,
                                                            channelVectorCmdIsA,
                                                            outArr);
                                                        blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                                                        {
                                                            texture = outArr,
                                                            width = vectorCmdTexture.width,
                                                            height = vectorCmdTexture.height,
                                                            packs = vectorCmdTexture.packs,
                                                            refs = 1,
                                                            owned = true,
                                                            hasLogicalShape = true,
                                                            logicalShape = vectorCmdTextureShape,
                                                            hasStorageShape = true,
                                                            storageShape = vectorCmdTextureShape
                                                        };
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = vectorCmdTextureShape;
                                                    }
                                                    else if (TryResolveCmdSpatialBroadcast(a, aShape, b, bShape, out var broadcastMode, out var outShape, out var outWidth, out var outHeight, out var outPacks))
                                                    {
                                                        var outDepth = outShape.dims == 4 ? Mathf.Max(1, outShape.d) * outPacks : outPacks;
                                                        var outArr = owner.RentTempArray(cmd, outWidth, outHeight, outDepth, RenderTextureFormat.ARGBHalf);
                                                        owner.Ops.BinaryOpPack4Broadcast(cmd, a.texture, b.texture, outPacks, opType, broadcastMode, outArr);
                                                        blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                                                        {
                                                            texture = outArr,
                                                            width = outWidth,
                                                            height = outHeight,
                                                            packs = outPacks,
                                                            refs = 1,
                                                            owned = true,
                                                            hasLogicalShape = true,
                                                            logicalShape = outShape,
                                                            hasStorageShape = true,
                                                            storageShape = outShape
                                                        };
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = outShape;
                                                    }
                                                    else
                                                    {
                                                        var fallbackShape = ResolveCmdOutputShape(aShape, bShape);
                                                        owner.DebugLog?.Invoke(
                                                            "[CmdPlaceholder][BinaryOp]"
                                                            + " | layer=" + layer.name
                                                            + " | opType=" + opType.ToString(CultureInfo.InvariantCulture)
                                                            + " | resizeGuard=" + owner.DisallowBufferAccess.ToString()
                                                            + " | a=d" + aShape.dims + ":" + aShape.w + "x" + aShape.h + "x" + aShape.d + "x" + aShape.c
                                                            + " | aTex=" + a.width + "x" + a.height + "x" + a.packs
                                                            + " | b=d" + bShape.dims + ":" + bShape.w + "x" + bShape.h + "x" + bShape.d + "x" + bShape.c
                                                            + " | bTex=" + b.width + "x" + b.height + "x" + b.packs
                                                            + " | out=d" + fallbackShape.dims + ":" + fallbackShape.w + "x" + fallbackShape.h + "x" + fallbackShape.d + "x" + fallbackShape.c);
                                                        owner.DebugLog?.Invoke(
                                                            "[CmdPlaceholder][BinaryOpShape]"
                                                            + " | layer=" + layer.name
                                                            + " | aStorage=" + NcnnRepro.GetCmdStorageShape(a, aShape)
                                                            + " | bStorage=" + NcnnRepro.GetCmdStorageShape(b, bShape)
                                                            + " | aLogical=" + aShape
                                                            + " | bLogical=" + bShape
                                                            + " | outLogical=" + fallbackShape);
                                                        if (owner.DisallowBufferAccess || owner.DisallowBufferOutputs || owner.DisallowBufferToTextureMaterialization)
                                                        {
                                                            throw new InvalidOperationException(
                                                                "pack4-only guard: command-buffer BinaryOp placeholder disallowed"
                                                                + " | layer=" + layer.name
                                                                + " | opType=" + opType.ToString(CultureInfo.InvariantCulture));
                                                        }
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
                && (aShape.dims == 3 || aShape.dims == 4)
                && aShape.dims == bShape.dims
                && aShape.w == bShape.w
                && aShape.h == bShape.h
                && aShape.d == bShape.d
                && aShape.c == bShape.c
                && NcnnRepro.MatchesPack4TextureStorage(a, aShape)
                && NcnnRepro.MatchesPack4TextureStorage(b, bShape)
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
            if ((aShape.dims != 3 && aShape.dims != 4) || aShape.dims != bShape.dims)
                return false;
            if (aShape.c != bShape.c || a.packs != b.packs)
                return false;

            var aIsScalarSpatial = aShape.w == 1 && aShape.h == 1 && aShape.d == 1 && a.width == 1 && a.height == 1;
            var bIsScalarSpatial = bShape.w == 1 && bShape.h == 1 && bShape.d == 1 && b.width == 1 && b.height == 1;
            var aIsOutputSpatial = aShape.w == a.width && aShape.h == a.height;
            var bIsOutputSpatial = bShape.w == b.width && bShape.h == b.height;

            if (aIsScalarSpatial && !bIsScalarSpatial && bIsOutputSpatial)
            {
                broadcastMode = 1;
                outWidth = b.width;
                outHeight = b.height;
                outPacks = b.packs;
                outShape = new NcnnRepro.BufferShape(bShape.dims, bShape.w, bShape.h, bShape.d, bShape.c);
                return true;
            }

            if (bIsScalarSpatial && !aIsScalarSpatial && aIsOutputSpatial)
            {
                broadcastMode = 2;
                outWidth = a.width;
                outHeight = a.height;
                outPacks = a.packs;
                outShape = new NcnnRepro.BufferShape(aShape.dims, aShape.w, aShape.h, aShape.d, aShape.c);
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
