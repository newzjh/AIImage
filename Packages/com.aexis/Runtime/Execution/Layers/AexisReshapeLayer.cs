using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisReshapeLayer : AexisBaseLayer
    {
        public AexisReshapeLayer() : base(AexisLayerTypes.Reshape, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            // Preserve the older buffer-first behavior for generic models. The newer
            // pack4 reshape transforms are intended for explicit attention / VISTA
            // opt-in flows and should not silently change CLIP execution.
            if (context.bufferBlobs.TryGetValue(layer.bottomNames[0], out var existingBuffer)
                && existingBuffer != null
                && !ShouldAllowAnyPack4ReshapeSpecializations(owner))
            {
#pragma warning disable CS0618
                ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
                return;
            }

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
        public override void ExecuteComputeBufferPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            var shapeExpr = layer.GetString(6, null);
            var bottomShapes = BuildBottomShapes(owner, layer, textureBlobs, textureShapes, bufferBlobs, bufferViews, tempOwned, !string.IsNullOrWhiteSpace(shapeExpr));

            if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var reshapeBuf) && reshapeBuf != null)
            {
                var srcTensor = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                if (srcTensor != null
                    && TryResolveWindowPartitionOutput(owner, layer, srcTensor, out var partitionTensor))
                {
                    owner.PublishTensorBufferOutput(
                        layer.topNames[0],
                        partitionTensor,
                        preferTexture: true,
                        textureBlobs,
                        textureShapes,
                        bufferBlobs,
                        bufferRefs,
                        bufferViews,
                        tempOwned);
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (srcTensor != null
                    && TryResolveWindowUnpartitionOutput(owner, layer, srcTensor, out var unpartitionTensor))
                {
                    owner.PublishTensorBufferOutput(
                        layer.topNames[0],
                        unpartitionTensor,
                        preferTexture: true,
                        textureBlobs,
                        textureShapes,
                        bufferBlobs,
                        bufferRefs,
                        bufferViews,
                        tempOwned);
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (srcTensor != null
                    && TryResolveAttentionContextFlattenOutput(owner, layer, srcTensor, out var attentionFlattenTensor))
                {
                    owner.DebugLog?.Invoke(
                        "TryResolveAttentionContextFlattenOutput applied"
                        + " | layer=" + layer.name
                        + " | src=" + srcTensor.dims + ":" + srcTensor.w + "x" + srcTensor.h + "x" + srcTensor.d + "x" + srcTensor.c
                        + " | dst=" + attentionFlattenTensor.dims + ":" + attentionFlattenTensor.w + "x" + attentionFlattenTensor.h + "x" + attentionFlattenTensor.d + "x" + attentionFlattenTensor.c);
                    owner.PublishTensorBufferOutput(
                        layer.topNames[0],
                        attentionFlattenTensor,
                        preferTexture: true,
                        textureBlobs,
                        textureShapes,
                        bufferBlobs,
                        bufferRefs,
                        bufferViews,
                        tempOwned,
                        RenderTextureFormat.ARGBFloat);
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (srcTensor != null)
                {
                    // Some lowered attention reshapes perform a real data reorder and must publish
                    // the newly produced buffer instead of aliasing the original input buffer.
                    var attentionTensor = TryResolveImplicitAttentionReshape(owner, layer, srcTensor);
                    if (attentionTensor != null)
                    {
                        owner.PublishTensorBufferOutput(
                            layer.topNames[0],
                            attentionTensor,
                            preferTexture: false,
                            textureBlobs,
                            textureShapes,
                            bufferBlobs,
                            bufferRefs,
                            bufferViews,
                            tempOwned);
                        owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                        return;
                    }
                }

                bufferBlobs[layer.topNames[0]] = reshapeBuf;
                if (bufferRefs.TryGetValue(layer.bottomNames[0], out var reshapeRef) && reshapeRef != null)
                {
                    bufferRefs[layer.topNames[0]] = reshapeRef;
                    reshapeRef.refs++;
                }

                if (srcTensor != null)
                {
                    var outView = AexisGraphSession.ResolveReshapeTensor(srcTensor, layer, bottomShapes);
                    bufferViews[layer.topNames[0]] = outView;

                    if (textureBlobs.TryGetValue(layer.bottomNames[0], out var reshapeTex) && reshapeTex != null && reshapeTex.texture != null)
                    {
                        var srcShape = AexisGraphSession.GetTextureShape(textureShapes, reshapeTex, layer.bottomNames[0]);
                        var outShape = new AexisGraphSession.BufferShape(outView.dims, outView.w, outView.h, outView.d, outView.c);
                        var canAliasTexture = CanAliasTextureLayout(owner, srcShape, outShape);
                        if (canAliasTexture)
                        {
                            var storageShape = AexisGraphSession.GetTextureStorageShape(reshapeTex, srcShape);
                            textureBlobs[layer.topNames[0]] = AexisGraphSession.CreateTextureAlias(reshapeTex, outShape, storageShape);
                            textureShapes[layer.topNames[0]] = outShape;
                        }
                    }
                }
            }
            else
            {
                var src = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                var srcShape = AexisGraphSession.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
                var canAliasTexture = CanAliasTextureLayout(owner, srcShape, outShape);

                if (ShouldKeepVistaTailFeatureTextureAlias(owner, layer, srcShape, outShape))
                {
                    var storageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
                    textureBlobs[layer.topNames[0]] = AexisGraphSession.CreateTextureAlias(src, outShape, storageShape);
                    textureShapes[layer.topNames[0]] = outShape;
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (!canAliasTexture)
                {
                    var scratchTensor = owner.RentScratchTensorFromTexture(src, srcShape, layer.bottomNames[0]);
                    if (TryResolveWindowPartitionOutput(owner, layer, scratchTensor, out var partitionTensor))
                    {
                        owner.PublishTensorBufferOutput(
                            layer.topNames[0],
                            partitionTensor,
                            preferTexture: true,
                            textureBlobs,
                            textureShapes,
                            bufferBlobs,
                            bufferRefs,
                            bufferViews,
                            tempOwned);
                        scratchTensor.Dispose();
                        owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                        return;
                    }

                    if (TryResolveWindowUnpartitionOutput(owner, layer, scratchTensor, out var unpartitionTensor))
                    {
                        owner.PublishTensorBufferOutput(
                            layer.topNames[0],
                            unpartitionTensor,
                            preferTexture: true,
                            textureBlobs,
                            textureShapes,
                            bufferBlobs,
                            bufferRefs,
                            bufferViews,
                            tempOwned);
                        scratchTensor.Dispose();
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (TryResolveAttentionContextFlattenOutput(owner, layer, scratchTensor, out var attentionFlattenTensor))
                {
                    owner.DebugLog?.Invoke(
                        "TryResolveAttentionContextFlattenOutput applied"
                        + " | layer=" + layer.name
                        + " | src=" + scratchTensor.dims + ":" + scratchTensor.w + "x" + scratchTensor.h + "x" + scratchTensor.d + "x" + scratchTensor.c
                        + " | dst=" + attentionFlattenTensor.dims + ":" + attentionFlattenTensor.w + "x" + attentionFlattenTensor.h + "x" + attentionFlattenTensor.d + "x" + attentionFlattenTensor.c);
                    owner.PublishTensorBufferOutput(
                        layer.topNames[0],
                        attentionFlattenTensor,
                        preferTexture: true,
                        textureBlobs,
                        textureShapes,
                        bufferBlobs,
                        bufferRefs,
                        bufferViews,
                        tempOwned,
                        RenderTextureFormat.ARGBFloat);
                    scratchTensor.Dispose();
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                var attentionTensor = TryResolveImplicitAttentionReshape(owner, layer, scratchTensor);
                if (attentionTensor != null)
                {
                        owner.PublishTensorBufferOutput(
                            layer.topNames[0],
                            attentionTensor,
                            preferTexture: false,
                            textureBlobs,
                            textureShapes,
                            bufferBlobs,
                            bufferRefs,
                            bufferViews,
                            tempOwned);
                        scratchTensor.Dispose();
                        owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                        return;
                    }

                    var outView = AexisGraphSession.ResolveReshapeTensor(scratchTensor, layer, bottomShapes);
                    var outTensor = new AexisTensorBuffer(
                        scratchTensor.buffer,
                        outView.dims,
                        outView.w,
                        outView.h,
                        outView.d,
                        outView.c,
                        true,
                        owner.ReturnTempBuffer);
                    var textureFormatOverride = ShouldPromoteGemmPrepTexture(owner, layer, outView)
                        ? RenderTextureFormat.ARGBFloat
                        : (RenderTextureFormat?)null;
                    owner.PublishTensorBufferOutput(
                        layer.topNames[0],
                        outTensor,
                        preferTexture: outView.dims <= 3,
                        textureBlobs,
                        textureShapes,
                        bufferBlobs,
                        bufferRefs,
                        bufferViews,
                        tempOwned,
                        textureFormatOverride);
                    if (TryShouldKeepVistaTailFeatureTexture(owner, layer, srcShape, outView))
                    {
                        var aliasLogicalShape = new AexisGraphSession.BufferShape(outView.dims, outView.w, outView.h, outView.d, outView.c);
                        var storageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
                        textureBlobs[layer.topNames[0]] = AexisGraphSession.CreateTextureAlias(src, aliasLogicalShape, storageShape);
                        textureShapes[layer.topNames[0]] = aliasLogicalShape;
                    }
                }
                else
                {
                    var storageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
                    textureBlobs[layer.topNames[0]] = AexisGraphSession.CreateTextureAlias(src, outShape, storageShape);
                    textureShapes[layer.topNames[0]] = outShape;
                }
            }

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            var shapeExpr = layer.GetString(6, null);
            var bottomShapes = BuildBottomShapes(owner, layer, textureBlobs, textureShapes, bufferBlobs, bufferViews, tempOwned, !string.IsNullOrWhiteSpace(shapeExpr));
            if (!AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var src, out var srcShape))
                throw new InvalidOperationException("Reshape render-texture path requires existing texture input: " + layer.name);

            var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
            if (ShouldKeepVistaTailFeatureTextureAlias(owner, layer, srcShape, outShape))
            {
                var storageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
                textureBlobs[layer.topNames[0]] = AexisGraphSession.CreateTextureAlias(src, outShape, storageShape);
                textureShapes[layer.topNames[0]] = outShape;
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            // A d3 attention context packs channels into texture lanes. Flattening
            // it to d2 changes physical order and must use the texture reshape path.
            if (TryExecuteRenderTextureDirectAttentionQkvReshapeAlias(owner, layer, src, srcShape, bottomShapes, textureBlobs, textureShapes))
            {
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (ShouldAllowAttentionPack4ReshapeSpecializations(owner))
            {
                if (TryExecuteRenderTextureWindowPartition(owner, layer, src, srcShape, textureBlobs, textureShapes))
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (TryExecuteRenderTextureWindowUnpartition(owner, layer, src, srcShape, textureBlobs, textureShapes))
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (TryExecuteRenderTextureAttentionContextFlatten(owner, layer, src, srcShape, textureBlobs, textureShapes))
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (TryExecuteRenderTexturePack4ToScalar2DReshape(owner, layer, src, srcShape, bottomShapes, textureBlobs, textureShapes))
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (TryExecuteRenderTexturePack4ToPack4Reshape(owner, layer, src, srcShape, bottomShapes, textureBlobs, textureShapes))
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (TryExecuteRenderTexturePack4Linear2DToPack4Reshape(owner, layer, src, srcShape, bottomShapes, textureBlobs, textureShapes))
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (TryExecuteRenderTextureScalar2DToPack4Reshape(owner, layer, src, srcShape, textureBlobs, textureShapes))
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (TryExecuteRenderTextureImplicitAttentionReshape(owner, layer, src, srcShape, textureBlobs, textureShapes))
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }
            }

            if (ShouldAllowGenericPack4ReshapeSpecializations(owner))
            {
                if (TryExecuteRenderTextureLinearMat2DReshape(owner, layer, src, srcShape, bottomShapes, textureBlobs, textureShapes))
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (TryExecuteRenderTexturePack4ToScalar2DReshape(owner, layer, src, srcShape, bottomShapes, textureBlobs, textureShapes))
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (TryExecuteRenderTexturePack4ToPack4Reshape(owner, layer, src, srcShape, bottomShapes, textureBlobs, textureShapes))
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (TryExecuteRenderTexturePack4Linear2DToPack4Reshape(owner, layer, src, srcShape, bottomShapes, textureBlobs, textureShapes))
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (TryExecuteRenderTextureScalar2DToPack4Reshape(owner, layer, src, srcShape, textureBlobs, textureShapes))
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }
            }

            if (CanAliasLinearMatTextureLayout(src, srcShape, outShape))
            {
                var linearStorageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
                textureBlobs[layer.topNames[0]] = AexisGraphSession.CreateTextureAlias(src, outShape, linearStorageShape);
                textureShapes[layer.topNames[0]] = outShape;
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (!CanAliasTextureLayout(owner, srcShape, outShape))
                throw new InvalidOperationException("Reshape render-texture path only supports alias-compatible layout: " + layer.name);

            var fallbackStorageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
            textureBlobs[layer.topNames[0]] = AexisGraphSession.CreateTextureAlias(src, outShape, fallbackStorageShape);
            textureShapes[layer.topNames[0]] = outShape;
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }
        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var sourceContract = AexisGraphSession.GetCmdTensorContract(src);
            var srcShape = sourceContract.LogicalShape;
            var bottomShapes = BuildCmdBottomShapes(layer, blobs, shapes);
            var initialOutShape = !string.IsNullOrWhiteSpace(layer.GetString(6, null))
                ? AexisGraphSession.EvaluateReshapeShapeExpression(layer.GetString(6, null), bottomShapes, layer)
                : AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
            if (ShouldKeepVistaTailFeatureTextureAlias(owner, layer, srcShape, initialOutShape))
            {
                var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorAlias(src, initialOutShape, storageShape);
                if (shapes != null)
                    shapes[layer.topNames[0]] = initialOutShape;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            if (TryExecuteCommandBufferDirectAttentionQkvReshapeAlias(owner, layer, src, srcShape, bottomShapes, blobs, shapes))
            {
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            if (ShouldAllowAttentionPack4ReshapeSpecializations(owner))
            {
                if (TryExecuteCommandBufferWindowPartition(owner, layer, src, srcShape, blobs, shapes, cmd))
                {
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }

                if (TryExecuteCommandBufferWindowUnpartition(owner, layer, src, srcShape, blobs, shapes, cmd))
                {
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }

                if (TryExecuteCommandBufferAttentionContextFlatten(owner, layer, src, srcShape, blobs, shapes, cmd))
                {
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }

                if (TryExecuteCommandBufferPack4ToScalar2DReshape(owner, layer, src, srcShape, blobs, shapes, cmd))
                {
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }

                if (TryExecuteCommandBufferPack4ToPack4Reshape(owner, layer, src, srcShape, blobs, shapes, cmd))
                {
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }

                if (TryExecuteCommandBufferPack4Linear2DToPack4Reshape(owner, layer, src, srcShape, blobs, shapes, cmd))
                {
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }

                if (TryExecuteCommandBufferScalar2DToPack4Reshape(owner, layer, src, srcShape, blobs, shapes, cmd))
                {
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }

                if (TryExecuteCommandBufferImplicitAttentionReshape(owner, layer, src, srcShape, blobs, shapes, cmd))
                {
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }
            }

            if (ShouldAllowGenericPack4ReshapeSpecializations(owner))
            {
                if (TryExecuteCommandBufferPack4ToScalar2DReshape(owner, layer, src, srcShape, blobs, shapes, cmd))
                {
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }

                if (TryExecuteCommandBufferPack4ToPack4Reshape(owner, layer, src, srcShape, blobs, shapes, cmd))
                {
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }

                if (TryExecuteCommandBufferPack4Linear2DToPack4Reshape(owner, layer, src, srcShape, blobs, shapes, cmd))
                {
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }

                if (TryExecuteCommandBufferScalar2DToPack4Reshape(owner, layer, src, srcShape, blobs, shapes, cmd))
                {
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }
            }

            if (!TryResolveCmdOutputShape(owner, layer, blobs, shapes, src, out var outShape, out var outW, out var outH, out var outPacks))
            {
                throw new InvalidOperationException(
                    "Reshape command-buffer Pack4 requires static metadata or a supported texture shape tensor; descriptor alias fallback is prohibited"
                    + " | layer=" + layer.name
                    + " | logical=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | storage=d" + sourceContract.StorageShape.dims + ":" + sourceContract.StorageShape.w + "x" + sourceContract.StorageShape.h + "x" + sourceContract.StorageShape.d + "x" + sourceContract.StorageShape.c
                    + " | layout=" + sourceContract.LayoutKind);
            }

            if (CanAliasLinearMatTextureLayout(src, srcShape, outShape))
            {
                var linearStorageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorAlias(src, outShape, linearStorageShape);
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            var aliasStorageShape = ResolveTextureStorageShape(outShape);
            if (!CanAliasTextureLayout(owner, srcShape, outShape) || !MatchesCmdTextureStorageShape(src, aliasStorageShape))
                throw new InvalidOperationException("Reshape command-buffer path only supports alias-compatible layout or explicit pack4 specializations: " + layer.name);

            var fallbackStorageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
            blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorAlias(src, outShape, fallbackStorageShape);
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool TryExecuteCommandBufferPack4Linear2DToPack4Reshape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape srcShape,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.BufferShape> shapes,
            UnityEngine.Rendering.CommandBuffer cmd)
        {
            if (owner == null || layer == null || src == null || src.texture == null
                || !AexisGraphSession.IsPack4LinearMatTexture(src, srcShape))
                return false;

            var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, BuildCmdBottomShapes(layer, blobs, shapes));
            if (outShape.dims < 3 || GetShapeElementCount(srcShape) != GetShapeElementCount(outShape))
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = outShape.dims >= 4 ? Mathf.Max(1, outShape.d) * outPacks : outPacks;
            var storageShape = outShape;
            var outputHeight = outShape.h;
            var outputSlices = outSlices;
            if (outShape.dims == 4
                && outSlices > GetMaxTextureArraySlicesSafe()
                && outPacks <= GetMaxTextureArraySlicesSafe()
                && checked(outShape.h * outShape.d) <= GetMaxTextureSizeSafe())
            {
                outputHeight = checked(outShape.h * outShape.d);
                outputSlices = outPacks;
                storageShape = new AexisGraphSession.BufferShape(4, outShape.w, outputHeight, 1, outShape.c);
            }
            var outRt = owner.RentTempArray(cmd, outShape.w, outputHeight, outputSlices, AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.ReshapePack4ToPack4(cmd, src.texture, srcShape.w, srcShape.h, srcShape.d, srcShape.c, srcShape.dims, outShape.w, outShape.h, outShape.d, outShape.c, outShape.dims, outRt, inputPack4Linear: true);
            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outRt,
                width = outShape.w,
                height = outputHeight,
                packs = outPacks,
                refs = 1,
                owned = true,
                hasLogicalShape = true,
                logicalShape = outShape,
                hasStorageShape = true,
                storageShape = storageShape
            };
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            return true;
        }

        private static bool TryExecuteCommandBufferScalar2DToPack4Reshape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape srcShape,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.BufferShape> shapes,
            UnityEngine.Rendering.CommandBuffer cmd)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;
            if (!TryResolveScalar2DToPack4ReshapeShape(layer, srcShape, out var outShape))
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = outShape.dims >= 4 ? Mathf.Max(1, outShape.d) * outPacks : outPacks;
            var outRt = owner.RentTempArray(cmd, outShape.w, outShape.h, outSlices, AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
            if (AexisGraphSession.IsStrictLinearMatTexture(src))
            {
                owner.Ops.ReshapeLinearMatToPack4(
                    cmd,
                    src.texture,
                    srcShape.w,
                    srcShape.h,
                    outShape.w,
                    outShape.h,
                    outShape.d,
                    outShape.c,
                    outShape.dims,
                    outRt);
                owner.DebugLog?.Invoke(
                    "[CmdTexture][ReshapeScalar2DToPack4]"
                    + " | layer=" + layer.name
                    + " | strictLinear=1"
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                    + " | outFormat=" + outRt.format);
            }
            else
            {
                owner.Ops.ReshapeScalar2DToPack4(
                    cmd,
                    src.texture,
                    srcShape.w,
                    srcShape.h,
                    outShape.w,
                    outShape.h,
                    outShape.d,
                    outShape.c,
                    outShape.dims,
                    outRt);
            }
            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outRt,
                width = outShape.w,
                height = outShape.h,
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
            return true;
        }

        private static bool TryExecuteCommandBufferAttentionContextFlatten(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape srcShape,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.BufferShape> shapes,
            UnityEngine.Rendering.CommandBuffer cmd)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;
            if (!TryResolveAttentionContextFlattenShape(owner, layer, srcShape, out var outShape))
                return false;
            if (srcShape.dims != 4 || srcShape.w <= 0 || srcShape.h <= 0 || srcShape.d <= 0 || srcShape.c <= 0)
                return false;
            if (outShape.w != srcShape.w * srcShape.c)
                return false;

            var storageShape = ResolveAttentionContextFlattenStorageShape(outShape);
            var outPacks = Mathf.Max(1, Mathf.CeilToInt(Mathf.Max(1, storageShape.c) / 4f));
            var outRt = owner.RentTempArray(cmd, storageShape.w, storageShape.h, outPacks, RenderTextureFormat.ARGBFloat);
            owner.Ops.AttentionContextFlattenPack4(
                cmd,
                src.texture,
                srcShape.w,
                srcShape.h,
                srcShape.d,
                srcShape.c,
                outShape.c,
                outShape.dims,
                outRt);
            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outRt,
                width = storageShape.w,
                height = storageShape.h,
                packs = outPacks,
                refs = 1,
                owned = true,
                hasLogicalShape = true,
                logicalShape = outShape,
                hasStorageShape = true,
                storageShape = storageShape
            };
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[AttentionContextFlattenPack4][cmd] applied"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | dst=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static bool TryExecuteCommandBufferDirectAttentionQkvReshapeAlias(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape srcShape,
            IReadOnlyList<AexisGraphSession.BufferShape> bottomShapes,
            Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            Dictionary<string, AexisGraphSession.BufferShape> shapes)
        {
            if (owner?.Model?.layers == null || layer == null || src == null || src.texture == null)
                return false;
            if (srcShape.dims != 2)
                return false;

            var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
            if (storageShape.dims != 3 || storageShape.d != 1 || storageShape.c <= 1)
                return false;

            var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
            if (outShape.dims != 3 || outShape.w != storageShape.w || outShape.h != storageShape.c || outShape.c != storageShape.h)
                return false;

            var consumer = FindSingleConsumer(owner.Model, layer.topNames != null && layer.topNames.Length > 0 ? layer.topNames[0] : null);
            if (consumer == null || consumer.type != AexisLayerTypes.Permute || consumer.GetInt(0, -1) != 2)
                return false;

            blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorAlias(src, outShape, storageShape);
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[CmdAttentionQkv][ReshapeAlias]"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | storage=d" + storageShape.dims + ":" + storageShape.w + "x" + storageShape.h + "x" + storageShape.d + "x" + storageShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static bool TryExecuteCommandBufferDirectAttentionContextReshapeAlias(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape srcShape,
            IReadOnlyList<AexisGraphSession.BufferShape> bottomShapes,
            Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            Dictionary<string, AexisGraphSession.BufferShape> shapes)
        {
            if (owner?.Model?.layers == null || layer == null || src == null || src.texture == null)
                return false;
            if (srcShape.dims != 3)
                return false;

            var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
            if (storageShape.dims != 3 || storageShape.d != 1 || storageShape.c <= 1)
                return false;

            var producer = FindSingleProducer(owner.Model, layer.bottomNames != null && layer.bottomNames.Length > 0 ? layer.bottomNames[0] : null);
            if (producer == null || producer.type != AexisLayerTypes.Permute || producer.GetInt(0, -1) != 2)
                return false;

            var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
            if (outShape.dims != 2 || outShape.w != storageShape.w * storageShape.c || outShape.h != storageShape.h)
                return false;

            blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorAlias(src, outShape, storageShape);
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[CmdAttentionContext][ReshapeAlias]"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | storage=d" + storageShape.dims + ":" + storageShape.w + "x" + storageShape.h + "x" + storageShape.d + "x" + storageShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static bool TryExecuteCommandBufferPack4ToPack4Reshape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape srcShape,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.BufferShape> shapes,
            UnityEngine.Rendering.CommandBuffer cmd)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;

            var bottomShapes = BuildCmdBottomShapes(layer, blobs, shapes);
            var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
            if (!CanUsePack4ToPack4Reshape(owner, srcShape, outShape))
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = outShape.dims >= 4 ? Mathf.Max(1, outShape.d) * outPacks : outPacks;
            var outRt = owner.RentTempArray(cmd, outShape.w, outShape.h, outSlices, AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.ReshapePack4ToPack4(cmd, src.texture, srcShape.w, srcShape.h, srcShape.d, srcShape.c, srcShape.dims, outShape.w, outShape.h, outShape.d, outShape.c, outShape.dims, outRt);
            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outRt,
                width = outShape.w,
                height = outShape.h,
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
            return true;
        }

        private static bool TryExecuteCommandBufferImplicitAttentionReshape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape srcShape,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.BufferShape> shapes,
            UnityEngine.Rendering.CommandBuffer cmd)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;
            if (!TryResolveImplicitAttentionReshapeShape(owner, layer, srcShape, out var outShape))
                return false;
            if (srcShape.dims != 3 || outShape.dims != 4)
                return false;
            if (srcShape.w <= 0 || srcShape.h <= 0 || srcShape.c <= 0 || outShape.w <= 0 || outShape.d <= 0 || outShape.c <= 0)
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = Mathf.Max(1, outShape.d) * outPacks;
            var outRt = owner.RentTempArray(cmd, outShape.w, outShape.h, outSlices, AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.AttentionReshapePack4(
                cmd,
                src.texture,
                srcShape.w,
                srcShape.h,
                srcShape.c,
                outShape.w,
                outShape.d,
                outShape.c,
                outRt);
            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outRt,
                width = outShape.w,
                height = outShape.h,
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
            owner.DebugLog?.Invoke(
                "[AttentionReshapePack4][cmd] applied"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | dst=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static bool TryExecuteCommandBufferPack4ToScalar2DReshape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape srcShape,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.BufferShape> shapes,
            UnityEngine.Rendering.CommandBuffer cmd)
        {
            if (owner == null || layer == null || src == null || src.texture == null
                || AexisGraphSession.IsStrictLinearMatTexture(src))
                return false;

            var bottomShapes = BuildCmdBottomShapes(layer, blobs, shapes);
            var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
            if (!CanUsePack4ToScalar2DReshape(owner, layer, srcShape, outShape))
                return false;

            var storageShape = AexisGraphSession.ResolveLinearMatStorageShape(outShape);
            var outRt = owner.RentTempMat(cmd, storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
            owner.Ops.ReshapePack4ToLinearMat(
                cmd,
                src.texture,
                srcShape.w,
                srcShape.h,
                srcShape.d,
                srcShape.c,
                srcShape.dims,
                outRt,
                inputPack4Linear: AexisGraphSession.IsPack4LinearMatTexture(src, srcShape));
            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outRt,
                width = outRt.width,
                height = outRt.height,
                packs = 1,
                refs = 1,
                owned = true,
                hasLogicalShape = true,
                logicalShape = outShape,
                hasStorageShape = true,
                storageShape = storageShape
            };
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[CmdTexture][ReshapePack4ToLinearMat]"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                + " | storage=d" + storageShape.dims + ":" + storageShape.w + "x" + storageShape.h + "x" + storageShape.d + "x" + storageShape.c
                + " | outFormat=" + outRt.format);
            return true;
        }

        private static bool TryResolveCmdOutputShape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.BufferShape> shapes,
            AexisGraphSession.CmdTensorRef src,
            out AexisGraphSession.BufferShape outShape,
            out int outW,
            out int outH,
            out int outPacks)
        {
            outShape = AexisGraphSession.InferCmdShape(src);
            outW = src.width;
            outH = src.height;
            outPacks = src.packs;

            if (src == null || layer == null)
                return false;

            var bottomShapes = BuildCmdBottomShapes(layer, blobs, shapes);
            if (!string.IsNullOrWhiteSpace(layer.GetString(6, null)))
            {
                outShape = AexisGraphSession.EvaluateReshapeShapeExpression(layer.GetString(6, null), bottomShapes, layer);
                AexisGraphSession.ResolveCmdTextureLayout(outShape, out outW, out outH, out outPacks);
                return true;
            }

            outShape = AexisGraphSession.ResolveReshapeShape(bottomShapes[0], layer, bottomShapes);
            AexisGraphSession.ResolveCmdTextureLayout(outShape, out outW, out outH, out outPacks);
            return true;
        }

        private static System.Collections.Generic.List<AexisGraphSession.BufferShape> BuildBottomShapes(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            System.Collections.Generic.Dictionary<string, ComputeBuffer> bufferBlobs,
            System.Collections.Generic.Dictionary<string, AexisTensorBuffer> bufferViews,
            System.Collections.Generic.List<IDisposable> tempOwned,
            bool materializeAll)
        {
            var shapes = new System.Collections.Generic.List<AexisGraphSession.BufferShape>(layer.bottomNames.Length);
            for (var i = 0; i < layer.bottomNames.Length; i++)
            {
                var name = layer.bottomNames[i];
                if (bufferViews.TryGetValue(name, out var view) && view != null)
                {
                    shapes.Add(new AexisGraphSession.BufferShape(view.dims, view.w, view.h, view.d, view.c));
                    continue;
                }

                if (textureBlobs.TryGetValue(name, out var tr) && tr != null && tr.texture != null)
                {
                    shapes.Add(AexisGraphSession.GetTextureShape(textureShapes, tr, name));
                    continue;
                }

                if (materializeAll)
                {
                    owner.GetOrConvertToBuffer(name, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    if (bufferViews.TryGetValue(name, out view) && view != null)
                    {
                        shapes.Add(new AexisGraphSession.BufferShape(view.dims, view.w, view.h, view.d, view.c));
                        continue;
                    }
                }

            throw new InvalidOperationException("Reshape bottom shape unavailable: " + layer.name + " | " + name);
        }
        return shapes;
    }

        private static bool CanAliasTextureLayout(AexisGraphSession owner, AexisGraphSession.BufferShape srcShape, AexisGraphSession.BufferShape outShape)
        {
            if (srcShape.dims > 3 || outShape.dims > 3)
            {
                if (srcShape.dims == outShape.dims
                    && srcShape.w == outShape.w
                    && srcShape.h == outShape.h
                    && srcShape.d == outShape.d
                    && srcShape.c == outShape.c)
                {
                    // Paramless 4D reshapes are often just graph markers around later pack4
                    // window/attention transforms. Preserve the existing array texture when the
                    // logical shape is unchanged so pack4-only validation can stay on the RT path.
                    return true;
                }
                if (CanAliasVistaTailTextureLayout(owner, srcShape, outShape))
                    return true;
                return false;
            }

            // A 4D tensor flattened into 2D/3D often changes the logical row-major interpretation
            // even if the pack4 texture dimensions happen to match. Keep those cases on the buffer
            // path so downstream matrix-style consumers read the intended linear order.
            if (srcShape.dims != outShape.dims)
                return false;

            var srcCount = srcShape.w * srcShape.h * srcShape.d * srcShape.c;
            var outCount = outShape.w * outShape.h * outShape.d * outShape.c;
            if (srcCount != outCount)
                return false;

            if (srcShape.dims == outShape.dims
                && srcShape.w == outShape.w
                && srcShape.h == outShape.h
                && srcShape.d == outShape.d
                && srcShape.c == outShape.c)
            {
                return true;
            }

            if ((srcShape.dims == 3 && (srcShape.c % 4) != 0) || (outShape.dims == 3 && (outShape.c % 4) != 0))
                return false;

            ResolvePack4TextureLayout(srcShape, out var srcW, out var srcH, out var srcPacks);
            ResolvePack4TextureLayout(outShape, out var outW, out var outH, out var outPacks);
            return srcW == outW && srcH == outH && srcPacks == outPacks;
        }

        private static bool CanAliasLinearMatTextureLayout(AexisGraphSession.TensorRef src, AexisGraphSession.BufferShape srcShape, AexisGraphSession.BufferShape outShape)
        {
            return AexisGraphSession.IsStrictLinearMatTexture(src)
                && CanAliasLinearMatShape(srcShape, outShape)
                && GetShapeElementCount(outShape) <= Mathf.Max(1, src.texture.width) * Mathf.Max(1, src.texture.height);
        }

        private static bool CanAliasLinearMatTextureLayout(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape srcShape, AexisGraphSession.BufferShape outShape)
        {
            return AexisGraphSession.IsStrictLinearMatTexture(src)
                && CanAliasLinearMatShape(srcShape, outShape)
                && GetShapeElementCount(outShape) <= Mathf.Max(1, src.texture.width) * Mathf.Max(1, src.texture.height);
        }

        private static bool CanAliasLinearMatShape(AexisGraphSession.BufferShape srcShape, AexisGraphSession.BufferShape outShape)
        {
            if (srcShape.dims < 1 || outShape.dims < 1)
                return false;
            if (srcShape.dims > 4 || outShape.dims > 4)
                return false;
            return GetShapeElementCount(srcShape) == GetShapeElementCount(outShape);
        }

        private static int GetShapeElementCount(AexisGraphSession.BufferShape shape)
        {
            return checked(
                Mathf.Max(1, shape.w)
                * Mathf.Max(1, shape.h)
                * Mathf.Max(1, shape.d)
                * Mathf.Max(1, shape.c));
        }

        private static void ResolvePack4TextureLayout(AexisGraphSession.BufferShape shape, out int width, out int height, out int packs)
        {
            width = Mathf.Max(1, shape.w);
            height = 1;
            packs = 1;

            if (shape.dims == 2)
            {
                height = Mathf.Max(1, shape.h);
                return;
            }

            if (shape.dims == 3)
            {
                height = Mathf.Max(1, shape.h);
                packs = Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f));
                return;
            }

            if (shape.dims >= 4)
            {
                height = Mathf.Max(1, shape.h);
                packs = Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f));
            }
        }

        private static AexisGraphSession.BufferShape ResolveTextureStorageShape(AexisGraphSession.BufferShape logicalShape)
        {
            if (logicalShape.dims <= 1)
                return new AexisGraphSession.BufferShape(3, logicalShape.w, 1, 1, 1);
            if (logicalShape.dims == 2)
                return new AexisGraphSession.BufferShape(3, logicalShape.w, logicalShape.h, 1, 1);
            return logicalShape;
        }

        private static bool MatchesCmdTextureStorageShape(AexisGraphSession.CmdTensorRef tensor, AexisGraphSession.BufferShape storageShape)
        {
            if (tensor == null || tensor.texture == null)
                return false;
            if (tensor.width != storageShape.w || tensor.height != storageShape.h)
                return false;

            var expectedPacks = Mathf.Max(1, Mathf.CeilToInt(Mathf.Max(1, storageShape.c) / 4f));
            if (tensor.packs != expectedPacks)
                return false;

            var expectedDepth = storageShape.dims == 4
                ? Mathf.Max(1, storageShape.d) * expectedPacks
                : expectedPacks;
            return Mathf.Max(1, tensor.texture.depth) == expectedDepth;
        }

        private static bool CanAliasVistaTailTextureLayout(AexisGraphSession owner, AexisGraphSession.BufferShape srcShape, AexisGraphSession.BufferShape outShape)
        {
            if (owner == null || !owner.EnableVistaTailPack4Specializations)
                return false;

            if (srcShape.dims == 4 && outShape.dims == 4)
            {
                return srcShape.w == outShape.w
                    && srcShape.h == outShape.h
                    && srcShape.d == outShape.d
                    && srcShape.c == outShape.c;
            }

            if (srcShape.dims != 2 || outShape.dims != 4)
                return false;
            if (srcShape.h != 1 || srcShape.c != 1 || outShape.c != 1)
                return false;

            var srcCount = srcShape.w;
            var outCount = outShape.w * outShape.h * outShape.d * outShape.c;
            if (srcCount != outCount)
                return false;

            var expectedFlatW = outShape.w * outShape.h * outShape.d;
            return srcShape.w == expectedFlatW;
        }

        private static bool TryShouldKeepVistaTailFeatureTexture(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape srcShape,
            AexisTensorBuffer outView)
        {
            if (owner == null || !owner.EnableVistaTailPack4Specializations)
                return false;
            if (layer == null || outView == null)
                return false;
            if (!string.Equals(layer.name, "reshape_124", StringComparison.Ordinal))
                return false;
            if (srcShape.dims != 4 || outView.dims != 2)
                return false;
            if (srcShape.w <= 0 || srcShape.h <= 0 || srcShape.d <= 0 || srcShape.c <= 0)
                return false;
            return outView.w == srcShape.w * srcShape.h * srcShape.d
                && outView.h == srcShape.c;
        }

        private static bool ShouldKeepVistaTailFeatureTextureAlias(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape srcShape,
            AexisGraphSession.BufferShape outShape)
        {
            if (owner == null || !owner.EnableVistaTailPack4Specializations)
                return false;
            if (layer == null)
                return false;
            if (!string.Equals(layer.name, "reshape_124", StringComparison.Ordinal))
                return false;
            if (srcShape.dims != 4 || outShape.dims != 2)
                return false;
            if (srcShape.w <= 0 || srcShape.h <= 0 || srcShape.d <= 0 || srcShape.c <= 0)
                return false;
            return outShape.w == srcShape.w * srcShape.h * srcShape.d
                && outShape.h == srcShape.c;
        }

        private static bool CanExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;

            var shapeExpr = layer.GetString(6, null);
            var bottomShapes = BuildBottomShapes(owner, layer, context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, context.tempOwned, !string.IsNullOrWhiteSpace(shapeExpr));
            if (!AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var src, out var srcShape))
            {
                if (owner?.DebugLog != null && layer != null && string.Equals(layer.name, "reshape_491", StringComparison.Ordinal))
                    owner.DebugLog("[ReshapeRTDiag] layer=" + layer.name + " | hasExistingTexture=false");
                return false;
            }

            if (ShouldAllowAttentionPack4ReshapeSpecializations(owner))
            {
                if (TryResolveImplicitAttentionReshapeShape(owner, layer, srcShape, out _))
                    return src != null && src.texture != null;
                if (TryResolveWindowPartitionPattern(owner, layer, srcShape, out _, out _, out _, out _, out _, out _, out _))
                    return src != null && src.texture != null;
                if (TryResolveWindowUnpartitionPattern(owner, layer, srcShape, out _, out _, out _, out _, out _, out _, out _))
                    return src != null && src.texture != null;
                if (TryResolveAttentionContextFlattenShape(owner, layer, srcShape, out _))
                    return src != null && src.texture != null;
                var attentionBottomShapes = BuildBottomShapes(owner, layer, context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, context.tempOwned, materializeAll: false);
                var attentionOutShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, attentionBottomShapes);
                var canPack4ToScalar2D = CanUsePack4ToScalar2DReshape(owner, layer, srcShape, attentionOutShape);
                if (canPack4ToScalar2D)
                    return src != null && src.texture != null;
                var canPack4ToPack4 = CanUsePack4ToPack4Reshape(owner, srcShape, attentionOutShape);
                if (canPack4ToPack4)
                    return src != null && src.texture != null;
                if (TryResolveScalar2DToPack4ReshapeShape(layer, srcShape, out _))
                    return src != null && src.texture != null;

                if (owner?.DebugLog != null && layer != null && string.Equals(layer.name, "reshape_491", StringComparison.Ordinal))
                {
                    owner.DebugLog(
                        "[ReshapeRTDiag] layer=" + layer.name
                        + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                        + " | out=d" + attentionOutShape.dims + ":" + attentionOutShape.w + "x" + attentionOutShape.h + "x" + attentionOutShape.d + "x" + attentionOutShape.c
                        + " | canPack4ToScalar2D=" + canPack4ToScalar2D
                        + " | canPack4ToPack4=" + canPack4ToPack4
                        + " | attentionSpecializations=" + ShouldAllowAttentionPack4ReshapeSpecializations(owner));
                }
            }

            var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
            if (ShouldAllowGenericPack4ReshapeSpecializations(owner))
            {
                if (CanAliasLinearMatTextureLayout(src, srcShape, outShape))
                    return src != null && src.texture != null;

                var canGenericPack4ToScalar2D = CanUsePack4ToScalar2DReshape(owner, layer, srcShape, outShape);
                var canGenericPack4ToPack4 = CanUsePack4ToPack4Reshape(owner, srcShape, outShape);
                if (canGenericPack4ToScalar2D)
                    return src != null && src.texture != null;
                if (canGenericPack4ToPack4)
                    return src != null && src.texture != null;
                if (TryResolveScalar2DToPack4ReshapeShape(layer, srcShape, out _))
                    return src != null && src.texture != null;
                if (AexisGraphSession.IsPack4LinearMatTexture(src, srcShape) && outShape.dims >= 3 && GetShapeElementCount(srcShape) == GetShapeElementCount(outShape))
                    return src != null && src.texture != null;
            }

            if (owner?.DebugLog != null && layer != null && string.Equals(layer.name, "reshape_491", StringComparison.Ordinal))
            {
                owner.DebugLog(
                    "[ReshapeRTDiag] layer=" + layer.name
                    + " | aliasFallback src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                    + " | canAlias=" + CanAliasTextureLayout(owner, srcShape, outShape));
            }
            return (CanAliasLinearMatTextureLayout(src, srcShape, outShape)
                || CanAliasTextureLayout(owner, srcShape, outShape)) && src != null && src.texture != null;
        }

        private static bool TryExecuteRenderTextureWindowPartition(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;
            if (!TryResolveWindowPartitionPattern(owner, layer, srcShape, out var outShape, out var groupsA, out var groupsB, out var groupsC, out var tokensA, out var tokensB, out var tokensC))
                return false;

            var storageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
            if (storageShape.dims != 4)
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outRt = owner.RentTempArray(outShape.w, outShape.h, outPacks, RenderTextureFormat.ARGBHalf);
            owner.Ops.WindowPartitionPack4(
                src.texture,
                storageShape.w,
                storageShape.h,
                storageShape.d,
                storageShape.c,
                outShape.w,
                outShape.h,
                outShape.c,
                groupsA,
                groupsB,
                groupsC,
                tokensA,
                tokensB,
                tokensC,
                outRt);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, outShape);
            return true;
        }

        private static bool TryExecuteCommandBufferWindowPartition(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape srcShape,
            Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            Dictionary<string, AexisGraphSession.BufferShape> shapes,
            CommandBuffer cmd)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;
            if (!TryResolveWindowPartitionPattern(owner, layer, srcShape, out var outShape, out var groupsA, out var groupsB, out var groupsC, out var tokensA, out var tokensB, out var tokensC))
                return false;

            var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
            if (storageShape.dims != 4)
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outRt = owner.RentTempArray(cmd, outShape.w, outShape.h, outPacks, RenderTextureFormat.ARGBHalf);
            owner.Ops.WindowPartitionPack4(
                cmd,
                src.texture,
                storageShape.w,
                storageShape.h,
                storageShape.d,
                storageShape.c,
                outShape.w,
                outShape.h,
                outShape.c,
                groupsA,
                groupsB,
                groupsC,
                tokensA,
                tokensB,
                tokensC,
                outRt);
            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outRt,
                width = outShape.w,
                height = outShape.h,
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
            owner.DebugLog?.Invoke(
                "[WindowPartitionPack4][cmd] applied"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | dst=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static bool TryExecuteRenderTexturePack4Linear2DToPack4Reshape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            System.Collections.Generic.IReadOnlyList<AexisGraphSession.BufferShape> bottomShapes,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes)
        {
            if (owner == null || layer == null || src == null || src.texture == null
                || !AexisGraphSession.IsPack4LinearMatTexture(src, srcShape))
                return false;

            var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
            if (outShape.dims < 3 || GetShapeElementCount(srcShape) != GetShapeElementCount(outShape))
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = outShape.dims >= 4 ? Mathf.Max(1, outShape.d) * outPacks : outPacks;
            var storageShape = outShape;
            var outputHeight = outShape.h;
            var outputSlices = outSlices;
            if (outShape.dims == 4
                && outSlices > GetMaxTextureArraySlicesSafe()
                && outPacks <= GetMaxTextureArraySlicesSafe()
                && checked(outShape.h * outShape.d) <= GetMaxTextureSizeSafe())
            {
                outputHeight = checked(outShape.h * outShape.d);
                outputSlices = outPacks;
                storageShape = new AexisGraphSession.BufferShape(4, outShape.w, outputHeight, 1, outShape.c);
            }
            var outRt = owner.RentTempArray(outShape.w, outputHeight, outputSlices, AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.ReshapePack4ToPack4(src.texture, srcShape.w, srcShape.h, srcShape.d, srcShape.c, srcShape.dims, outShape.w, outShape.h, outShape.d, outShape.c, outShape.dims, outRt, inputPack4Linear: true);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, storageShape);
            return true;
        }

        private static int GetMaxTextureArraySlicesSafe()
        {
            try { return Mathf.Max(1, SystemInfo.maxTextureArraySlices); }
            catch { return 2048; }
        }

        private static int GetMaxTextureSizeSafe()
        {
            try { return Mathf.Max(1, SystemInfo.maxTextureSize); }
            catch { return 16384; }
        }

        private static bool TryExecuteRenderTextureLinearMat2DReshape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            System.Collections.Generic.IReadOnlyList<AexisGraphSession.BufferShape> bottomShapes,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes)
        {
            if (owner == null || layer == null || src == null || src.texture == null || !AexisGraphSession.IsStrictLinearMatTexture(src))
                return false;
            var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
            if (srcShape.dims > 2 || outShape.dims > 2 || GetShapeElementCount(srcShape) != GetShapeElementCount(outShape))
                return false;
            if (src.texture.width == outShape.w && src.texture.height == outShape.h)
                return false;

            var output = owner.RentTempMat(outShape.w, outShape.h, src.texture.format);
            owner.Ops.ReshapeLinearMat2D(src.texture, src.texture.width, src.texture.height, output);
            var storageShape = AexisGraphSession.ResolveLinearMatStorageShape(outShape);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], output, outShape, storageShape);
            return true;
        }

        private static bool TryExecuteRenderTextureScalar2DToPack4Reshape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;
            if (!TryResolveScalar2DToPack4ReshapeShape(layer, srcShape, out var outShape))
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = outShape.dims >= 4 ? Mathf.Max(1, outShape.d) * outPacks : outPacks;
            var outRt = owner.RentTempArray(outShape.w, outShape.h, outSlices, AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
            if (AexisGraphSession.IsStrictLinearMatTexture(src))
            {
                owner.Ops.ReshapeLinearMatToPack4(
                    src.texture,
                    srcShape.w,
                    srcShape.h,
                    outShape.w,
                    outShape.h,
                    outShape.d,
                    outShape.c,
                    outShape.dims,
                    outRt);
                owner.DebugLog?.Invoke(
                    "[Texture][ReshapeScalar2DToPack4]"
                    + " | layer=" + layer.name
                    + " | strictLinear=1"
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                    + " | outFormat=" + outRt.format);
            }
            else
            {
                owner.Ops.ReshapeScalar2DToPack4(
                    src.texture,
                    srcShape.w,
                    srcShape.h,
                    outShape.w,
                    outShape.h,
                    outShape.d,
                    outShape.c,
                    outShape.dims,
                    outRt);
            }
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, outShape);
            return true;
        }

        private static bool TryExecuteRenderTextureDirectAttentionQkvReshapeAlias(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            IReadOnlyList<AexisGraphSession.BufferShape> bottomShapes,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes)
        {
            if (owner?.Model?.layers == null || layer == null || src == null || src.texture == null)
                return false;
            if (srcShape.dims != 2)
                return false;

            var storageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
            if (storageShape.dims != 3 || storageShape.d != 1 || storageShape.c <= 1)
                return false;

            var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
            if (outShape.dims != 3 || outShape.w != storageShape.w || outShape.h != storageShape.c || outShape.c != storageShape.h)
                return false;

            var consumer = FindSingleConsumer(owner.Model, layer.topNames != null && layer.topNames.Length > 0 ? layer.topNames[0] : null);
            if (consumer == null || consumer.type != AexisLayerTypes.Permute || consumer.GetInt(0, -1) != 2)
                return false;

            textureBlobs[layer.topNames[0]] = AexisGraphSession.CreateTextureAlias(src, outShape, storageShape);
            textureShapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[AttentionQkv][ReshapeAlias]"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | storage=d" + storageShape.dims + ":" + storageShape.w + "x" + storageShape.h + "x" + storageShape.d + "x" + storageShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static bool TryExecuteRenderTextureDirectAttentionContextReshapeAlias(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            IReadOnlyList<AexisGraphSession.BufferShape> bottomShapes,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes)
        {
            if (owner?.Model?.layers == null || layer == null || src == null || src.texture == null)
                return false;
            if (srcShape.dims != 3)
                return false;

            var storageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
            if (storageShape.dims != 3 || storageShape.d != 1 || storageShape.c <= 1)
                return false;

            var producer = FindSingleProducer(owner.Model, layer.bottomNames != null && layer.bottomNames.Length > 0 ? layer.bottomNames[0] : null);
            if (producer == null || producer.type != AexisLayerTypes.Permute || producer.GetInt(0, -1) != 2)
                return false;

            var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
            if (outShape.dims != 2 || outShape.w != storageShape.w * storageShape.c || outShape.h != storageShape.h)
                return false;

            textureBlobs[layer.topNames[0]] = AexisGraphSession.CreateTextureAlias(src, outShape, storageShape);
            textureShapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[AttentionContext][ReshapeAlias]"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | storage=d" + storageShape.dims + ":" + storageShape.w + "x" + storageShape.h + "x" + storageShape.d + "x" + storageShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static bool TryExecuteRenderTextureAttentionContextFlatten(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;
            if (!TryResolveAttentionContextFlattenShape(owner, layer, srcShape, out var outShape))
                return false;
            if (srcShape.dims != 4 || srcShape.w <= 0 || srcShape.h <= 0 || srcShape.d <= 0 || srcShape.c <= 0)
                return false;
            if (outShape.w != srcShape.w * srcShape.c)
                return false;

            var storageShape = ResolveAttentionContextFlattenStorageShape(outShape);
            var outPacks = Mathf.Max(1, Mathf.CeilToInt(Mathf.Max(1, storageShape.c) / 4f));
            var outRt = owner.RentTempArray(storageShape.w, storageShape.h, outPacks, RenderTextureFormat.ARGBFloat);
            owner.Ops.AttentionContextFlattenPack4(
                src.texture,
                srcShape.w,
                srcShape.h,
                srcShape.d,
                srcShape.c,
                outShape.c,
                outShape.dims,
                outRt);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, storageShape);
            owner.DebugLog?.Invoke(
                "[AttentionContextFlattenPack4] applied"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | dst=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static bool TryExecuteRenderTexturePack4ToScalar2DReshape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            System.Collections.Generic.IReadOnlyList<AexisGraphSession.BufferShape> bottomShapes,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes)
        {
            if (owner == null || layer == null || src == null || src.texture == null
                || AexisGraphSession.IsStrictLinearMatTexture(src))
                return false;

            var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
            if (!CanUsePack4ToScalar2DReshape(owner, layer, srcShape, outShape))
                return false;

            var storageShape = AexisGraphSession.ResolveLinearMatStorageShape(outShape);
            var outRt = owner.RentTempMat(storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
            var inputPack4Linear = AexisGraphSession.IsPack4LinearMatTexture(src, srcShape);
            owner.Ops.ReshapePack4ToLinearMat(
                src.texture,
                srcShape.w,
                srcShape.h,
                srcShape.d,
                srcShape.c,
                srcShape.dims,
                outRt,
                inputPack4Linear: inputPack4Linear);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, storageShape);
            owner.DebugLog?.Invoke(
                "[Texture][ReshapePack4ToLinearMat]"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                + " | storage=d" + storageShape.dims + ":" + storageShape.w + "x" + storageShape.h + "x" + storageShape.d + "x" + storageShape.c
                + " | outFormat=" + outRt.format);
            return true;
        }

        private static bool TryExecuteRenderTexturePack4ToPack4Reshape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            System.Collections.Generic.IReadOnlyList<AexisGraphSession.BufferShape> bottomShapes,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;

            var outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer, bottomShapes);
            if (!CanUsePack4ToPack4Reshape(owner, srcShape, outShape))
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = outShape.dims >= 4 ? Mathf.Max(1, outShape.d) * outPacks : outPacks;
            var outRt = owner.RentTempArray(outShape.w, outShape.h, outSlices, AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.ReshapePack4ToPack4(src.texture, srcShape.w, srcShape.h, srcShape.d, srcShape.c, srcShape.dims, outShape.w, outShape.h, outShape.d, outShape.c, outShape.dims, outRt);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, outShape);
            return true;
        }

        private static bool TryExecuteRenderTextureWindowUnpartition(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;
            if (!TryResolveWindowUnpartitionPattern(owner, layer, srcShape, out var outShape, out var groupsA, out var groupsB, out var groupsC, out var tokensA, out var tokensB, out var tokensC))
                return false;

            var storageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
            if (storageShape.dims != 3)
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = Mathf.Max(1, outShape.d) * outPacks;
            var outRt = owner.RentTempArray(outShape.w, outShape.h, outSlices, AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.WindowUnpartitionPack4(
                src.texture,
                storageShape.w,
                storageShape.h,
                storageShape.c,
                outShape.w,
                outShape.h,
                outShape.d,
                outShape.c,
                groupsA,
                groupsB,
                groupsC,
                tokensA,
                tokensB,
                tokensC,
                outRt);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, outShape);
            return true;
        }

        private static bool TryExecuteCommandBufferWindowUnpartition(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape srcShape,
            Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            Dictionary<string, AexisGraphSession.BufferShape> shapes,
            CommandBuffer cmd)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;
            if (!TryResolveWindowUnpartitionPattern(owner, layer, srcShape, out var outShape, out var groupsA, out var groupsB, out var groupsC, out var tokensA, out var tokensB, out var tokensC))
                return false;

            var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
            if (storageShape.dims != 3)
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = Mathf.Max(1, outShape.d) * outPacks;
            var outRt = owner.RentTempArray(cmd, outShape.w, outShape.h, outSlices, AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.WindowUnpartitionPack4(
                cmd,
                src.texture,
                storageShape.w,
                storageShape.h,
                storageShape.c,
                outShape.w,
                outShape.h,
                outShape.d,
                outShape.c,
                groupsA,
                groupsB,
                groupsC,
                tokensA,
                tokensB,
                tokensC,
                outRt);
            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outRt,
                width = outShape.w,
                height = outShape.h,
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
            owner.DebugLog?.Invoke(
                "[WindowUnpartitionPack4][cmd] applied"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | dst=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static bool TryExecuteRenderTextureImplicitAttentionReshape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes)
        {
            if (owner == null || layer == null || src == null || src.texture == null)
                return false;
            if (!TryResolveImplicitAttentionReshapeShape(owner, layer, srcShape, out var outShape))
                return false;
            if (srcShape.dims != 3 || outShape.dims != 4)
                return false;
            if (srcShape.w <= 0 || srcShape.h <= 0 || srcShape.c <= 0 || outShape.w <= 0 || outShape.d <= 0 || outShape.c <= 0)
                return false;

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = Mathf.Max(1, outShape.d) * outPacks;
            var outRt = owner.RentTempArray(outShape.w, outShape.h, outSlices, AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.AttentionReshapePack4(
                src.texture,
                srcShape.w,
                srcShape.h,
                srcShape.c,
                outShape.w,
                outShape.d,
                outShape.c,
                outRt);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, outShape);
            owner.DebugLog?.Invoke(
                "[AttentionReshapePack4] applied"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | dst=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            return true;
        }

        private static AexisTensorBuffer TryResolveImplicitAttentionReshape(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisTensorBuffer src)
        {
            if (src == null)
                return null;
            if (!TryResolveImplicitAttentionReshapeShape(owner, layer, new AexisGraphSession.BufferShape(src.dims, src.w, src.h, src.d, src.c), out var outShape))
                return null;

            var source = AexisGraphSession.ReadFloatBuffer(src.buffer);
            var destination = new float[source.Length];
            var headDim = outShape.w;
            var tokens = outShape.h;
            var windows = outShape.d;
            var qkvHeadChannels = outShape.c;

            for (var window = 0; window < windows; window++)
            {
                for (var token = 0; token < tokens; token++)
                {
                    var srcBase = ((window * src.h) + token) * src.w;
                    for (var qkvHead = 0; qkvHead < qkvHeadChannels; qkvHead++)
                    {
                        var dstBase = (((qkvHead * windows) + window) * tokens + token) * headDim;
                        Array.Copy(
                            source,
                            srcBase + (qkvHead * headDim),
                            destination,
                            dstBase,
                            headDim);
                    }
                }
            }

            var outBuffer = owner.RentTempBuffer(destination.Length, sizeof(float));
            outBuffer.SetData(destination);
            return new AexisTensorBuffer(
                outBuffer,
                outShape.dims,
                outShape.w,
                outShape.h,
                outShape.d,
                outShape.c,
                true,
                owner.ReturnTempBuffer);
        }

        private static bool TryResolveImplicitAttentionReshapeShape(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisGraphSession.BufferShape srcShape, out AexisGraphSession.BufferShape outShape)
        {
            outShape = default;
            if (owner?.Model?.layers == null || layer == null)
                return false;
            if (!IsParamlessReshape(layer))
                return false;
            if (srcShape.dims != 3 || srcShape.w <= 0 || srcShape.h <= 0 || srcShape.c <= 0)
                return false;
            if (layer.topNames == null || layer.topNames.Length == 0 || string.IsNullOrWhiteSpace(layer.topNames[0]))
                return false;

            var permute = FindSingleConsumer(owner.Model, layer.topNames[0]);
            if (permute == null || permute.type != AexisLayerTypes.Permute || permute.GetInt(0, -1) != 0)
                return false;
            if (permute.topNames == null || permute.topNames.Length == 0 || string.IsNullOrWhiteSpace(permute.topNames[0]))
                return false;

            var slice = FindSingleConsumer(owner.Model, permute.topNames[0]);
            if (slice == null || slice.type != AexisLayerTypes.Slice || slice.topNames == null || slice.topNames.Length != 3)
                return false;
            var positiveAxis = slice.GetInt(1, 0);
            if (positiveAxis < 0)
                positiveAxis += srcShape.dims;
            if (positiveAxis != 0)
                return false;

            if (!TryInferAttentionHeadDim(owner.Model, slice, out var headDim))
                return false;
            if (headDim <= 0 || (srcShape.w % headDim) != 0)
                return false;

            var totalChannels = srcShape.w / headDim;
            if (totalChannels <= 0 || (totalChannels % 3) != 0)
                return false;

            outShape = new AexisGraphSession.BufferShape(4, headDim, srcShape.h, srcShape.c, totalChannels);
            return true;
        }

        private static bool TryResolveScalar2DToPack4ReshapeShape(
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape srcShape,
            out AexisGraphSession.BufferShape outShape)
        {
            outShape = default;
            if (layer == null)
                return false;
            if (srcShape.dims != 2 || srcShape.w <= 0 || srcShape.h <= 0)
                return false;

            outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer);
            if ((outShape.dims != 3 && outShape.dims != 4) || outShape.w <= 0 || outShape.h <= 0 || outShape.c <= 0)
                return false;

            var srcCount = srcShape.w * srcShape.h;
            var outCount = outShape.w * outShape.h * Mathf.Max(1, outShape.d) * outShape.c;
            if (srcCount != outCount)
                return false;

            return true;
        }

        private static bool TryResolveWindowPartitionOutput(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisTensorBuffer src,
            out AexisTensorBuffer outTensor)
        {
            outTensor = null;
            if (src == null || src.buffer == null)
                return false;
            if (!TryResolveWindowPartitionPattern(
                    owner,
                    layer,
                    new AexisGraphSession.BufferShape(src.dims, src.w, src.h, src.d, src.c),
                    out var outShape,
                    out var groupsA,
                    out var groupsB,
                    out var groupsC,
                    out var tokensA,
                    out var tokensB,
                    out var tokensC))
            {
                return false;
            }

            var axisA = src.c;
            var axisB = src.d;
            var axisC = src.h;
            var embedDim = src.w;
            var source = AexisGraphSession.ReadFloatBuffer(src.buffer);
            var destination = new float[source.Length];
            var outputGroups = outShape.c;
            var outputTokens = outShape.h;

            for (var groupA = 0; groupA < groupsA; groupA++)
            {
                for (var groupB = 0; groupB < groupsB; groupB++)
                {
                    for (var groupC = 0; groupC < groupsC; groupC++)
                    {
                        var outputGroup = ((groupA * groupsB) + groupB) * groupsC + groupC;
                        for (var tokenA = 0; tokenA < tokensA; tokenA++)
                        {
                            var inputA = groupA * tokensA + tokenA;
                            for (var tokenB = 0; tokenB < tokensB; tokenB++)
                            {
                                var inputB = groupB * tokensB + tokenB;
                                for (var tokenC = 0; tokenC < tokensC; tokenC++)
                                {
                                    var inputC = groupC * tokensC + tokenC;
                                    var outputToken = ((tokenA * tokensB) + tokenB) * tokensC + tokenC;
                                    var sourceBase = (((inputA * axisB) + inputB) * axisC + inputC) * embedDim;
                                    var destinationBase = ((outputGroup * outputTokens) + outputToken) * embedDim;
                                    System.Array.Copy(source, sourceBase, destination, destinationBase, embedDim);
                                }
                            }
                        }
                    }
                }
            }

            var outBuffer = owner.RentTempBuffer(destination.Length, sizeof(float));
            outBuffer.SetData(destination);
            outTensor = new AexisTensorBuffer(outBuffer, outShape.dims, outShape.w, outShape.h, outShape.d, outShape.c, true, owner.ReturnTempBuffer);
            return true;
        }

        private static bool TryResolveWindowPartitionPattern(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape srcShape,
            out AexisGraphSession.BufferShape outShape,
            out int groupsA,
            out int groupsB,
            out int groupsC,
            out int tokensA,
            out int tokensB,
            out int tokensC)
        {
            outShape = default;
            groupsA = 0;
            groupsB = 0;
            groupsC = 0;
            tokensA = 0;
            tokensB = 0;
            tokensC = 0;

            if (owner?.Model?.layers == null || layer == null)
                return false;
            if (srcShape.dims != 4 || srcShape.w <= 0 || srcShape.h <= 0 || srcShape.d <= 0 || srcShape.c <= 0)
                return false;
            if (srcShape.h != srcShape.d || srcShape.d != srcShape.c)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length == 0 || string.IsNullOrWhiteSpace(layer.bottomNames[0]))
                return false;

            outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer);
            if (outShape.dims != 3 || outShape.w != srcShape.w || outShape.h <= 0 || outShape.c <= 0)
                return false;
            if ((srcShape.c * srcShape.d * srcShape.h) != (outShape.c * outShape.h))
                return false;

            if (!TryMatchWindowPartitionProducerChain(owner.Model, layer))
                return false;
            if (!TryResolvePerfectCube(outShape.c, out var groupsEdge))
                return false;
            if (!TryResolvePerfectCube(outShape.h, out var tokensEdge))
                return false;
            if ((groupsEdge * tokensEdge) != srcShape.h)
                return false;

            groupsA = groupsB = groupsC = groupsEdge;
            tokensA = tokensB = tokensC = tokensEdge;
            return true;
        }

        private static bool TryResolveWindowUnpartitionOutput(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisTensorBuffer src,
            out AexisTensorBuffer outTensor)
        {
            outTensor = null;
            if (src == null || src.buffer == null)
                return false;
            if (!TryResolveWindowUnpartitionPattern(
                    owner,
                    layer,
                    new AexisGraphSession.BufferShape(src.dims, src.w, src.h, src.d, src.c),
                    out var outShape,
                    out var groupsA,
                    out var groupsB,
                    out var groupsC,
                    out var tokensA,
                    out var tokensB,
                    out var tokensC))
            {
                return false;
            }

            var axisA = outShape.c;
            var axisB = outShape.d;
            var axisC = outShape.h;
            var embedDim = src.w;
            var source = AexisGraphSession.ReadFloatBuffer(src.buffer);
            var destination = new float[source.Length];
            var inputTokens = src.h;

            for (var groupA = 0; groupA < groupsA; groupA++)
            {
                for (var groupB = 0; groupB < groupsB; groupB++)
                {
                    for (var groupC = 0; groupC < groupsC; groupC++)
                    {
                        var inputGroup = ((groupA * groupsB) + groupB) * groupsC + groupC;
                        for (var tokenA = 0; tokenA < tokensA; tokenA++)
                        {
                            var outputA = groupA * tokensA + tokenA;
                            for (var tokenB = 0; tokenB < tokensB; tokenB++)
                            {
                                var outputB = groupB * tokensB + tokenB;
                                for (var tokenC = 0; tokenC < tokensC; tokenC++)
                                {
                                    var outputC = groupC * tokensC + tokenC;
                                    var inputToken = ((tokenA * tokensB) + tokenB) * tokensC + tokenC;
                                    var sourceBase = ((inputGroup * inputTokens) + inputToken) * embedDim;
                                    var destinationBase = (((outputA * axisB) + outputB) * axisC + outputC) * embedDim;
                                    System.Array.Copy(source, sourceBase, destination, destinationBase, embedDim);
                                }
                            }
                        }
                    }
                }
            }

            var outBuffer = owner.RentTempBuffer(destination.Length, sizeof(float));
            outBuffer.SetData(destination);
            outTensor = new AexisTensorBuffer(outBuffer, outShape.dims, outShape.w, outShape.h, outShape.d, outShape.c, true, owner.ReturnTempBuffer);
            return true;
        }

        private static bool TryResolveWindowUnpartitionPattern(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape srcShape,
            out AexisGraphSession.BufferShape outShape,
            out int groupsA,
            out int groupsB,
            out int groupsC,
            out int tokensA,
            out int tokensB,
            out int tokensC)
        {
            outShape = default;
            groupsA = 0;
            groupsB = 0;
            groupsC = 0;
            tokensA = 0;
            tokensB = 0;
            tokensC = 0;

            if (owner?.Model?.layers == null || layer == null)
                return false;
            if (srcShape.dims != 3 || srcShape.w <= 0 || srcShape.h <= 0 || srcShape.c <= 0)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length == 0 || string.IsNullOrWhiteSpace(layer.bottomNames[0]))
                return false;

            outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer);
            if (outShape.dims != 4 || outShape.w != srcShape.w || outShape.h <= 0 || outShape.d <= 0 || outShape.c <= 0)
                return false;
            if (outShape.h != outShape.d || outShape.d != outShape.c)
                return false;
            if ((srcShape.c * srcShape.h) != (outShape.c * outShape.d * outShape.h))
                return false;
            if (!TryMatchWindowUnpartitionProducerChain(owner.Model, layer))
                return false;
            if (!TryResolvePerfectCube(srcShape.c, out var groupsEdge))
                return false;
            if (!TryResolvePerfectCube(srcShape.h, out var tokensEdge))
                return false;
            if (outShape.h != (groupsEdge * tokensEdge))
                return false;

            groupsA = groupsB = groupsC = groupsEdge;
            tokensA = tokensB = tokensC = tokensEdge;
            return true;
        }

        private static bool TryResolveAttentionContextFlattenOutput(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisTensorBuffer src,
            out AexisTensorBuffer outTensor)
        {
            outTensor = null;
            if (src == null || src.buffer == null)
                return false;
            if (!TryResolveAttentionContextFlattenShape(owner, layer, new AexisGraphSession.BufferShape(src.dims, src.w, src.h, src.d, src.c), out var outShape))
                return false;

            var source = AexisGraphSession.ReadFloatBuffer(src.buffer);
            var destination = new float[source.Length];
            var headDim = src.w;
            var tokens = src.h;
            var windows = src.d;
            var heads = src.c;

            for (var window = 0; window < windows; window++)
            {
                for (var token = 0; token < tokens; token++)
                {
                    var dstBase = ((window * tokens) + token) * outShape.w;
                    for (var dim = 0; dim < headDim; dim++)
                    {
                        for (var head = 0; head < heads; head++)
                        {
                            var srcIndex = ((((head * windows) + window) * tokens) + token) * headDim + dim;
                            var dstIndex = dstBase + dim * heads + head;
                            destination[dstIndex] = source[srcIndex];
                        }
                    }
                }
            }

            var outBuffer = owner.RentTempBuffer(destination.Length, sizeof(float));
            outBuffer.SetData(destination);
            outTensor = new AexisTensorBuffer(
                outBuffer,
                outShape.dims,
                outShape.w,
                outShape.h,
                outShape.d,
                outShape.c,
                true,
                owner.ReturnTempBuffer);
            return true;
        }

        private static bool TryResolveAttentionContextFlattenShape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape srcShape,
            out AexisGraphSession.BufferShape outShape)
        {
            outShape = default;
            if (owner?.Model?.layers == null || layer == null)
                return false;
            if (srcShape.dims != 4 || srcShape.w <= 0 || srcShape.h <= 0 || srcShape.d <= 0 || srcShape.c <= 0)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length == 0 || string.IsNullOrWhiteSpace(layer.bottomNames[0]))
                return false;

            var producer = FindSingleProducer(owner.Model, layer.bottomNames[0]);
            if (producer == null || producer.type != AexisLayerTypes.MatMul)
                return false;

            outShape = AexisGraphSession.ResolveReshapeShape(srcShape, layer);
            if (outShape.dims != 2 && outShape.dims != 3)
                return false;
            if (outShape.w != srcShape.w * srcShape.c)
                return false;

            var collapsesWindowsIntoTokens =
                outShape.h == srcShape.h * srcShape.d
                && (outShape.dims != 3 || outShape.c == 1);
            if (collapsesWindowsIntoTokens)
                return true;

            var preservesWindowAxis =
                outShape.dims == 3
                && outShape.h == srcShape.h
                && outShape.c == srcShape.d;
            return preservesWindowAxis;
        }

        private static AexisGraphSession.BufferShape ResolveAttentionContextFlattenStorageShape(AexisGraphSession.BufferShape logicalShape)
        {
            if (logicalShape.dims == 2)
                return new AexisGraphSession.BufferShape(3, logicalShape.w, logicalShape.h, 1, 1);
            if (logicalShape.dims == 3 && logicalShape.c == 1)
                return new AexisGraphSession.BufferShape(3, logicalShape.w, logicalShape.h, 1, 1);
            return logicalShape;
        }

        private static bool TryMatchWindowPartitionProducerChain(AexisGraphModel model, AexisGraphModel.Layer layer)
        {
            if (model?.layers == null || layer?.bottomNames == null || layer.bottomNames.Length == 0)
                return false;

            var permute0 = FindSingleProducer(model, layer.bottomNames[0]);
            if (permute0 == null || permute0.type != AexisLayerTypes.Permute || permute0.GetInt(0, -1) != 0)
                return false;
            if (permute0.bottomNames == null || permute0.bottomNames.Length == 0 || string.IsNullOrWhiteSpace(permute0.bottomNames[0]))
                return false;

            var reshapeMarker = FindSingleProducer(model, permute0.bottomNames[0]);
            if (reshapeMarker == null || reshapeMarker.type != AexisLayerTypes.Reshape || !IsParamlessReshape(reshapeMarker))
                return false;
            if (reshapeMarker.bottomNames == null || reshapeMarker.bottomNames.Length == 0 || string.IsNullOrWhiteSpace(reshapeMarker.bottomNames[0]))
                return false;

            var permute9 = FindSingleProducer(model, reshapeMarker.bottomNames[0]);
            return permute9 != null && permute9.type == AexisLayerTypes.Permute && permute9.GetInt(0, -1) == 9;
        }

        private static bool TryMatchWindowUnpartitionProducerChain(AexisGraphModel model, AexisGraphModel.Layer layer)
        {
            if (model?.layers == null || layer?.bottomNames == null || layer.bottomNames.Length == 0)
                return false;

            var permute0 = FindSingleProducer(model, layer.bottomNames[0]);
            if (permute0 == null || permute0.type != AexisLayerTypes.Permute || permute0.GetInt(0, -1) != 0)
                return false;
            if (permute0.bottomNames == null || permute0.bottomNames.Length == 0 || string.IsNullOrWhiteSpace(permute0.bottomNames[0]))
                return false;

            var reshapeMarker = FindSingleProducer(model, permute0.bottomNames[0]);
            return reshapeMarker != null
                && reshapeMarker.type == AexisLayerTypes.Reshape
                && IsParamlessReshape(reshapeMarker);
        }

        private static bool TryResolvePerfectCube(int value, out int edge)
        {
            edge = 0;
            if (value <= 0)
                return false;

            var candidate = Mathf.RoundToInt(Mathf.Pow(value, 1f / 3f));
            for (var delta = -1; delta <= 1; delta++)
            {
                var test = candidate + delta;
                if (test <= 0)
                    continue;
                if ((test * test * test) != value)
                    continue;
                edge = test;
                return true;
            }

            return false;
        }

        private static bool TryInferAttentionHeadDim(AexisGraphModel model, AexisGraphModel.Layer slice, out int headDim)
        {
            headDim = 0;
            if (model?.layers == null || slice?.topNames == null)
                return false;

            for (var i = 0; i < slice.topNames.Length; i++)
            {
                var branchTop = slice.topNames[i];
                if (string.IsNullOrWhiteSpace(branchTop))
                    continue;

                var reshape = FindSingleConsumer(model, branchTop);
                if (reshape == null || reshape.type != AexisLayerTypes.Reshape || !IsParamlessReshape(reshape))
                    continue;
                if (reshape.topNames == null || reshape.topNames.Length == 0 || string.IsNullOrWhiteSpace(reshape.topNames[0]))
                    continue;

                var matmul = FindSingleConsumer(model, reshape.topNames[0]);
                if (matmul == null || matmul.type != AexisLayerTypes.MatMul)
                    continue;
                if (matmul.topNames == null || matmul.topNames.Length == 0 || string.IsNullOrWhiteSpace(matmul.topNames[0]))
                    continue;

                var mul = FindSingleConsumer(model, matmul.topNames[0]);
                if (mul == null || mul.type != AexisLayerTypes.BinaryOp || mul.GetInt(0, -1) != 2)
                    continue;

                var scale = mul.GetFloat(2, 0f);
                if (!(scale > 0f))
                    continue;

                var inferred = Mathf.RoundToInt(1f / (scale * scale));
                if (inferred <= 0)
                    continue;

                var reconstructed = 1f / Mathf.Sqrt(inferred);
                if (Mathf.Abs(reconstructed - scale) > 1e-3f)
                    continue;

                headDim = inferred;
                return true;
            }

            return false;
        }

        private static bool IsParamlessReshape(AexisGraphModel.Layer layer)
        {
            if (layer == null)
                return false;
            if (!string.IsNullOrWhiteSpace(layer.GetString(6, null)))
                return false;
            return layer.GetInt(0, -233) == -233
                && layer.GetInt(1, -233) == -233
                && layer.GetInt(11, -233) == -233
                && layer.GetInt(2, -233) == -233;
        }

        private static AexisGraphModel.Layer FindSingleConsumer(AexisGraphModel model, string blobName)
        {
            if (model?.layers == null || string.IsNullOrWhiteSpace(blobName))
                return null;

            AexisGraphModel.Layer found = null;
            for (var i = 0; i < model.layers.Count; i++)
            {
                var candidate = model.layers[i];
                if (candidate?.bottomNames == null)
                    continue;
                for (var j = 0; j < candidate.bottomNames.Length; j++)
                {
                    if (!string.Equals(candidate.bottomNames[j], blobName, StringComparison.Ordinal))
                        continue;
                    if (found != null)
                        return null;
                    found = candidate;
                    break;
                }
            }

            return found;
        }

        private static AexisGraphModel.Layer FindEffectiveSingleConsumer(
            AexisGraphModel model,
            string blobName)
        {
            var consumer = FindSingleConsumer(model, blobName);
            var hopGuard = 0;
            while (consumer != null
                && hopGuard++ < 8
                && (consumer.type == AexisLayerTypes.AtenTo || consumer.type == AexisLayerTypes.Noop)
                && consumer.topNames != null
                && consumer.topNames.Length > 0
                && !string.IsNullOrWhiteSpace(consumer.topNames[0]))
            {
                consumer = FindSingleConsumer(model, consumer.topNames[0]);
            }

            return consumer;
        }

        private static bool ShouldPromoteGemmPrepTexture(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisTensorBuffer outView)
        {
            if (outView == null)
                return false;
            return ShouldPromoteGemmPrepTexture(
                owner,
                layer,
                new AexisGraphSession.BufferShape(outView.dims, outView.w, outView.h, outView.d, outView.c));
        }

        private static bool ShouldPromoteGemmPrepTexture(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisGraphSession.BufferShape outShape)
        {
            if (!ShouldAllowPack4LinearMatReshapeSpecializations(owner))
                return false;
            if (owner?.Model?.layers == null || layer?.topNames == null || layer.topNames.Length == 0)
                return false;
            if (outShape.dims != 2)
                return false;

            var consumer = FindEffectiveSingleConsumer(owner.Model, layer.topNames[0]);
            return consumer != null
                && consumer.type == AexisLayerTypes.Gemm
                && consumer.GetInt(5, 0) != 0;
        }

        private static bool CanUsePack4ToScalar2DReshape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape srcShape,
            AexisGraphSession.BufferShape outShape)
        {
            if (!ShouldAllowGenericPack4ReshapeSpecializations(owner))
                return false;
            if (srcShape.dims < 1 || srcShape.dims > 4)
                return false;
            if (outShape.dims < 1 || outShape.dims > 2)
                return false;
            if (outShape.w <= 0 || outShape.h <= 0)
                return false;
            if (srcShape.w <= 0 || srcShape.h <= 0 || srcShape.d <= 0 || srcShape.c <= 0)
                return false;

            var srcCount = srcShape.w * srcShape.h * srcShape.d * srcShape.c;
            var outCount = outShape.w * outShape.h;
            if (srcCount != outCount)
                return false;
            if (srcShape.dims == 1)
                return true;
            if (CanUseWidthPreservingPack4ToScalar2DReshape(srcShape, outShape))
                return true;

            var consumer = FindEffectiveSingleConsumer(owner?.Model, layer?.topNames != null && layer.topNames.Length > 0 ? layer.topNames[0] : null);
            if (consumer != null
                && (consumer.type == AexisLayerTypes.Permute
                    || consumer.type == AexisLayerTypes.Gemm
                    || consumer.type == AexisLayerTypes.InnerProduct
                    || consumer.type == AexisLayerTypes.RMSNorm
                    || consumer.type == AexisLayerTypes.Swish))
            {
                return true;
            }

            return CanUseCodeFormerStylePack4ToScalar2DReshape(owner, layer, srcShape, outShape);
        }

        private static bool CanUseWidthPreservingPack4ToScalar2DReshape(
            AexisGraphSession.BufferShape srcShape,
            AexisGraphSession.BufferShape outShape)
        {
            if (srcShape.dims != 3 && srcShape.dims != 4)
                return false;
            if (outShape.dims != 2)
                return false;
            if (srcShape.w <= 0 || srcShape.h <= 0 || srcShape.d <= 0 || srcShape.c <= 0)
                return false;
            if (outShape.w <= 0 || outShape.h <= 0)
                return false;

            var expectedH = srcShape.h * Mathf.Max(1, srcShape.d) * srcShape.c;
            if (outShape.w == srcShape.w && outShape.h == expectedH)
                return true;

            var expectedW = srcShape.w * srcShape.h * Mathf.Max(1, srcShape.d);
            return outShape.w == expectedW && outShape.h == srcShape.c;
        }

        private static bool CanUsePack4ToPack4Reshape(
            AexisGraphSession owner,
            AexisGraphSession.BufferShape srcShape,
            AexisGraphSession.BufferShape outShape)
        {
            if (!ShouldAllowGenericPack4ReshapeSpecializations(owner))
                return false;
            if (srcShape.dims < 3 || outShape.dims < 3)
                return false;
            if (srcShape.w <= 0 || srcShape.h <= 0 || srcShape.c <= 0 || outShape.w <= 0 || outShape.h <= 0 || outShape.c <= 0)
                return false;

            var srcCount = srcShape.w * srcShape.h * srcShape.d * srcShape.c;
            var outCount = outShape.w * outShape.h * outShape.d * outShape.c;
            if (srcCount != outCount)
                return false;

            return srcShape.dims != outShape.dims
                || srcShape.w != outShape.w
                || srcShape.h != outShape.h
                || srcShape.d != outShape.d
                || srcShape.c != outShape.c;
        }

        private static bool CanUseCodeFormerStylePack4ToScalar2DReshape(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape srcShape,
            AexisGraphSession.BufferShape outShape)
        {
            if (!ShouldAllowAttentionPack4ReshapeSpecializations(owner))
                return false;
            if (owner?.Model?.layers == null || layer?.topNames == null || layer.topNames.Length == 0)
                return false;
            if (srcShape.dims != 3 && srcShape.dims != 4)
                return false;
            if (outShape.dims != 2 || outShape.w <= 0 || outShape.h <= 0)
                return false;
            if (srcShape.w <= 0 || srcShape.h <= 0 || srcShape.c <= 0)
                return false;

            var expectedRows = srcShape.w * srcShape.h * Mathf.Max(1, srcShape.d);
            if (outShape.w != srcShape.c || outShape.h != expectedRows)
                return false;

            var consumer = FindSingleConsumer(owner.Model, layer.topNames[0]);
            return consumer != null
                && (consumer.type == AexisLayerTypes.Split
                    || consumer.type == AexisLayerTypes.Reduction);
        }

        private static bool ShouldAllowAttentionPack4ReshapeSpecializations(AexisGraphSession owner)
        {
            return owner != null && owner.EnableAttentionMatMulPack4Specializations;
        }

        private static bool ShouldAllowGenericPack4ReshapeSpecializations(AexisGraphSession owner)
        {
            return owner != null
                && (owner.ExecutionMode == AexisInferenceExecutionMode.ProductionTextureOnly
                    || owner.EnableAttentionMatMulPack4Specializations
                    || owner.DisallowBufferAccess
                    || owner.DisallowBufferOutputs
                    || owner.DisallowBufferToTextureMaterialization
                    || owner.DisallowInferenceTempComputeBuffers);
        }

        private static bool ShouldAllowPack4LinearMatReshapeSpecializations(AexisGraphSession owner)
        {
            return ShouldAllowGenericPack4ReshapeSpecializations(owner);
        }

        private static bool ShouldAllowAnyPack4ReshapeSpecializations(AexisGraphSession owner)
        {
            return owner != null
                && (owner.EnableAttentionMatMulPack4Specializations
                    || owner.EnableVistaTailPack4Specializations
                    || owner.DisallowBufferAccess
                    || owner.DisallowBufferOutputs
                    || owner.DisallowBufferToTextureMaterialization
                    || owner.DisallowInferenceTempComputeBuffers);
        }

        private static AexisGraphModel.Layer FindSingleProducer(AexisGraphModel model, string blobName)
        {
            if (model?.layers == null || string.IsNullOrWhiteSpace(blobName))
                return null;

            for (var i = 0; i < model.layers.Count; i++)
            {
                var candidate = model.layers[i];
                if (candidate?.topNames == null)
                    continue;
                for (var j = 0; j < candidate.topNames.Length; j++)
                {
                    if (string.Equals(candidate.topNames[j], blobName, StringComparison.Ordinal))
                        return candidate;
                }
            }

            return null;
        }

        private static System.Collections.Generic.List<AexisGraphSession.BufferShape> BuildCmdBottomShapes(
            AexisGraphModel.Layer layer,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.BufferShape> cmdShapes)
        {
            var bottomShapes = new System.Collections.Generic.List<AexisGraphSession.BufferShape>(layer.bottomNames.Length);
            for (var i = 0; i < layer.bottomNames.Length; i++)
            {
                bottomShapes.Add(AexisGraphSession.GetCmdShape(cmdShapes, blobs, layer.bottomNames[i]));
            }
            return bottomShapes;
        }
    }
}
