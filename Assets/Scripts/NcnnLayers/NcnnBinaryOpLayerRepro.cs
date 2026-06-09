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

            if (withScalar != 0)
            {
                if (owner.ShouldForceCurrentLayerBufferPath()
                    || !NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var aTexScalar, out var aTexShapeScalar)
                    || aTexShapeScalar.dims > 3)
                {
                    throw new InvalidOperationException("BinaryOp render-texture scalar path requires existing texture input: " + layer.name);
                }

                var outDepthScalar = aTexShapeScalar.dims == 4 ? aTexShapeScalar.d * aTexScalar.packs : aTexScalar.packs;
                var outRtScalar = owner.RentTempArray(aTexScalar.width, aTexScalar.height, outDepthScalar, RenderTextureFormat.ARGBHalf);
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
            var opType = layer.GetInt(0, 0);
            var withScalar = layer.GetInt(1, 0);

            if (withScalar != 0)
            {
                return !owner.ShouldForceCurrentLayerBufferPath()
                    && NcnnRepro.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out _, out var scalarTexShape)
                    && scalarTexShape.dims <= 3;
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
                && (aShape.dims == 3 || aShape.dims == 4)
                && aShape.dims == bShape.dims
                && aShape.w == bShape.w
                && aShape.h == bShape.h
                && aShape.d == bShape.d
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
