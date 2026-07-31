using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    public sealed class AexisDeepFillV2ContextualAttentionLayer : AexisBaseLayer
    {
        private const int SourcePacksPerDispatch = 1;

        public AexisDeepFillV2ContextualAttentionLayer()
            : base(AexisLayerTypes.DeepFillV2ContextualAttention, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            ExecuteRenderTexturePath(owner, layer, context);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            foreach (var _ in ExecuteRenderTexturePathIncremental(owner, layer, context))
            {
            }
        }

        public override IEnumerable<bool> ExecuteBufferIncremental(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            return ExecuteRenderTexturePathIncremental(owner, layer, context);
        }

        private IEnumerable<bool> ExecuteRenderTexturePathIncremental(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            if (layer == null) throw new ArgumentNullException(nameof(layer));
            if (context == null) throw new ArgumentNullException(nameof(context));
            if (layer.bottomNames == null || layer.bottomNames.Length != 2 || layer.topNames == null || layer.topNames.Length != 1)
                throw new InvalidOperationException("DeepFillV2 contextual attention requires feature+mask inputs and one output: " + layer.name);

            var feature = RequireTextureInput(context, layer.bottomNames[0], layer.name);
            var mask = RequireTextureInput(context, layer.bottomNames[1], layer.name);
            var featureShape = AexisGraphSession.GetTextureShape(context.textureShapes, feature, layer.bottomNames[0]);
            var maskShape = AexisGraphSession.GetTextureShape(context.textureShapes, mask, layer.bottomNames[1]);

            var ksize = layer.GetInt(0, 3);
            var rate = layer.GetInt(1, 2);
            var stride = layer.GetInt(2, 1);
            var softmaxScale = layer.GetFloat(3, 10f);
            var patchEpsilon = layer.GetFloat(4, 1e-4f);
            var maskDownsample = layer.GetInt(5, 8);
            ValidateProfile(layer, featureShape, maskShape, ksize, rate, stride, maskDownsample, patchEpsilon, softmaxScale);

            var matchW = featureShape.w / rate;
            var matchH = featureShape.h / rate;
            var sourceCount = matchW * matchH;
            var sourcePacks = Mathf.CeilToInt(sourceCount / 4f);
            var featurePacks = Mathf.CeilToInt(featureShape.c / 4f);
            var format = feature.texture.format;
            RenderTexture patchStats = null;
            RenderTexture scores = null;
            RenderTexture weights = null;
            RenderTexture output = null;
            try
            {
                patchStats = owner.RentTempArray(matchW, matchH, 1, format);
                scores = owner.RentTempArray(matchW, matchH, sourcePacks, format);
                weights = owner.RentTempArray(matchW, matchH, sourcePacks, format);
                LogStage(
                    owner,
                    layer,
                    "allocated"
                    + " | feature=" + feature.texture.width + "x" + feature.texture.height + "x" + feature.texture.volumeDepth
                    + " | feature_format=" + feature.texture.format
                    + " | score_weight=" + scores.width + "x" + scores.height + "x" + scores.volumeDepth
                    + " | format=" + scores.format
                    + " | max_array_slices=" + SystemInfo.maxTextureArraySlices
                    + " | async_compute=" + SystemInfo.supportsAsyncCompute);

                LogStage(owner, layer, "patch-stats-submit-begin");
                owner.Ops.DeepFillV2PatchStats(
                    feature.texture, mask.texture,
                    featureShape.w, featureShape.h, featureShape.c,
                    matchW, matchH, maskShape.w, maskShape.h,
                    maskDownsample, patchEpsilon, softmaxScale, patchStats);
                LogStage(owner, layer, "patch-stats-submit-end");
                foreach (var _ in WaitForGpuRetirement())
                    yield return true;
                LogStage(owner, layer, "patch-stats-resumed");
                yield return true;
                for (var sourcePackOffset = 0; sourcePackOffset < sourcePacks; sourcePackOffset += SourcePacksPerDispatch)
                {
                    var sourcePackCount = Mathf.Min(SourcePacksPerDispatch, sourcePacks - sourcePackOffset);
                    var logScoreChunk = ShouldLogChunk(sourcePackOffset, sourcePackCount, sourcePacks);
                    if (logScoreChunk)
                        LogStage(owner, layer, "scores-submit-begin | packs=" + sourcePackOffset + "+" + sourcePackCount + "/" + sourcePacks);
                    owner.Ops.DeepFillV2ScoresChunk(
                        feature.texture, patchStats,
                        featureShape.w, featureShape.h, featureShape.c,
                        matchW, matchH, maskShape.w, maskShape.h,
                        maskDownsample, patchEpsilon, softmaxScale,
                        sourcePackOffset, sourcePackCount, scores);
                    if (logScoreChunk)
                        LogStage(owner, layer, "scores-submit-end | packs=" + sourcePackOffset + "+" + sourcePackCount + "/" + sourcePacks);
                    foreach (var _ in WaitForGpuRetirement())
                        yield return true;
                    if (logScoreChunk)
                        LogStage(owner, layer, "scores-resumed | packs=" + sourcePackOffset + "+" + sourcePackCount + "/" + sourcePacks);
                    yield return true;
                }
                LogStage(owner, layer, "softmax-submit-begin");
                owner.Ops.DeepFillV2Softmax(
                    scores, patchStats,
                    featureShape.w, featureShape.h, featureShape.c,
                    matchW, matchH, maskShape.w, maskShape.h,
                    maskDownsample, patchEpsilon, softmaxScale, weights);
                LogStage(owner, layer, "softmax-submit-end");
                foreach (var _ in WaitForGpuRetirement())
                    yield return true;
                LogStage(owner, layer, "softmax-resumed");
                yield return true;

                // RentTempArray clears this output. Reconstruction then keeps
                // each source-pack contribution in one UAV; it never creates an
                // SRV/UAV texture dependency in a single dispatch.
                output = owner.RentTempArray(featureShape.w, featureShape.h, featurePacks, format);
                for (var sourcePackOffset = 0; sourcePackOffset < sourcePacks; sourcePackOffset += SourcePacksPerDispatch)
                {
                    var sourcePackCount = Mathf.Min(SourcePacksPerDispatch, sourcePacks - sourcePackOffset);
                    var logReconstructChunk = ShouldLogChunk(sourcePackOffset, sourcePackCount, sourcePacks);
                    if (logReconstructChunk)
                        LogStage(owner, layer, "reconstruct-submit-begin | packs=" + sourcePackOffset + "+" + sourcePackCount + "/" + sourcePacks);
                    owner.Ops.DeepFillV2ReconstructChunk(
                        feature.texture, weights,
                        featureShape.w, featureShape.h, featureShape.c,
                        matchW, matchH, maskShape.w, maskShape.h,
                        maskDownsample, patchEpsilon, softmaxScale,
                        sourcePackOffset, sourcePackCount,
                        sourcePackOffset == 0 ? null : output, output);
                    if (logReconstructChunk)
                        LogStage(owner, layer, "reconstruct-submit-end | packs=" + sourcePackOffset + "+" + sourcePackCount + "/" + sourcePacks);
                    foreach (var _ in WaitForGpuRetirement())
                        yield return true;
                    if (logReconstructChunk)
                        LogStage(owner, layer, "reconstruct-resumed | packs=" + sourcePackOffset + "+" + sourcePackCount + "/" + sourcePacks);
                    yield return true;
                }

                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, featureShape);
                output = null;

                // Attention spans multiple score and reconstruction dispatches.
                // Keep inputs and intermediates alive beyond graph liveness so a
                // following same-shaped convolution cannot alias an in-flight tile
                // renderer resource.
                context.tempOwned.Add(owner.RetainTextureRefUntilResultDispose(feature));
                context.tempOwned.Add(owner.RetainTextureRefUntilResultDispose(mask));
                context.tempOwned.Add(owner.DeferTempArrayReturn(patchStats));
                patchStats = null;
                context.tempOwned.Add(owner.DeferTempArrayReturn(scores));
                scores = null;
                context.tempOwned.Add(owner.DeferTempArrayReturn(weights));
                weights = null;
            }
            finally
            {
                if (patchStats != null) owner.ReturnTempArray(patchStats);
                if (scores != null) owner.ReturnTempArray(scores);
                if (weights != null) owner.ReturnTempArray(weights);
                if (output != null) owner.ReturnTempArray(output);
            }

            owner.Consume(
                context.textureBlobs,
                context.bufferBlobs,
                context.bufferRefs,
                context.bufferViews,
                context.remaining,
                layer.bottomNames,
                context.pinnedNames);
        }

        private static bool ShouldLogChunk(int sourcePackOffset, int sourcePackCount, int sourcePacks)
        {
            const int logInterval = 64;
            return sourcePackOffset == 0
                || sourcePackOffset + sourcePackCount >= sourcePacks
                || (sourcePackOffset + sourcePackCount) % logInterval == 0;
        }

        private static void LogStage(AexisGraphSession owner, AexisGraphModel.Layer layer, string stage)
        {
            owner.DebugLog?.Invoke(
                "[DeepFillV2][ContextualAttention]"
                + " | layer=" + layer.name
                + " | stage=" + stage);
        }

        // A frame yield alone only gives Unity's main loop a chance to run.  It
        // does not prove that the preceding compute dispatch has retired.  The
        // accumulator ping-pong below must not reuse an RT until its write has
        // completed on the graphics queue, particularly on tile-based GPUs.
        private static IEnumerable<bool> WaitForGpuRetirement()
        {
            // AsyncQueueSynchronisation cannot be queried on a graphics-only
            // device. CPUSynchronisation is specifically the fence type for this
            // graphics-queue polling path and works independently of async
            // compute support.
            if (!SystemInfo.supportsGraphicsFence)
            {
                yield return true;
                yield break;
            }

            var fence = Graphics.CreateGraphicsFence(
                GraphicsFenceType.CPUSynchronisation,
                SynchronisationStageFlags.AllGPUOperations);
            while (!fence.passed)
                yield return true;
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            if (layer == null) throw new ArgumentNullException(nameof(layer));
            if (context == null || context.commandBuffer == null) throw new ArgumentNullException(nameof(context));
            if (layer.bottomNames == null || layer.bottomNames.Length != 2 || layer.topNames == null || layer.topNames.Length != 1)
                throw new InvalidOperationException("DeepFillV2 contextual attention requires feature+mask inputs and one output: " + layer.name);

            var feature = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var mask = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[1]);
            var featureShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            var maskShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[1]);
            if (!AexisGraphSession.MatchesPack4TextureStorage(feature, featureShape)
                || !AexisGraphSession.MatchesPack4TextureStorage(mask, maskShape))
            {
                throw new InvalidOperationException(
                    "DeepFillV2 contextual attention requires exact Pack4 Texture2DArray inputs"
                    + " | layer=" + layer.name
                    + " | rejected_fallback=buffer-activation");
            }

            var ksize = layer.GetInt(0, 3);
            var rate = layer.GetInt(1, 2);
            var stride = layer.GetInt(2, 1);
            var softmaxScale = layer.GetFloat(3, 10f);
            var patchEpsilon = layer.GetFloat(4, 1e-4f);
            var maskDownsample = layer.GetInt(5, 8);
            ValidateProfile(layer, featureShape, maskShape, ksize, rate, stride, maskDownsample, patchEpsilon, softmaxScale);

            var matchW = featureShape.w / rate;
            var matchH = featureShape.h / rate;
            var sourcePacks = Mathf.CeilToInt(matchW * matchH / 4f);
            var featurePacks = Mathf.CeilToInt(featureShape.c / 4f);
            ComputeTexture patchStats = null;
            ComputeTexture scores = null;
            ComputeTexture weights = null;
            ComputeTexture output = null;
            try
            {
                patchStats = owner.RentTempArray(
                    context.commandBuffer, matchW, matchH, 1, feature.texture.format,
                    AexisGraphSession.GetStrictPack4ScratchIdentity(layer, "deepfill-patch-stats"));
                scores = owner.RentTempArray(
                    context.commandBuffer, matchW, matchH, sourcePacks, feature.texture.format,
                    AexisGraphSession.GetStrictPack4ScratchIdentity(layer, "deepfill-scores"));
                weights = owner.RentTempArray(
                    context.commandBuffer, matchW, matchH, sourcePacks, feature.texture.format,
                    AexisGraphSession.GetStrictPack4ScratchIdentity(layer, "deepfill-weights"));

                owner.Ops.DeepFillV2PatchStats(context.commandBuffer, feature.texture, mask.texture, featureShape.w, featureShape.h, featureShape.c, matchW, matchH, maskShape.w, maskShape.h, maskDownsample, patchEpsilon, softmaxScale, patchStats);
                for (var sourcePackOffset = 0; sourcePackOffset < sourcePacks; sourcePackOffset += SourcePacksPerDispatch)
                {
                    var sourcePackCount = Mathf.Min(SourcePacksPerDispatch, sourcePacks - sourcePackOffset);
                    owner.Ops.DeepFillV2ScoresChunk(context.commandBuffer, feature.texture, patchStats, featureShape.w, featureShape.h, featureShape.c, matchW, matchH, maskShape.w, maskShape.h, maskDownsample, patchEpsilon, softmaxScale, sourcePackOffset, sourcePackCount, scores);
                }
                owner.Ops.DeepFillV2Softmax(context.commandBuffer, scores, patchStats, featureShape.w, featureShape.h, featureShape.c, matchW, matchH, maskShape.w, maskShape.h, maskDownsample, patchEpsilon, softmaxScale, weights);

                output = owner.RentTempArray(
                    context.commandBuffer, featureShape.w, featureShape.h, featurePacks, feature.texture.format,
                    layer.topNames[0]);
                for (var sourcePackOffset = 0; sourcePackOffset < sourcePacks; sourcePackOffset += SourcePacksPerDispatch)
                {
                    var sourcePackCount = Mathf.Min(SourcePacksPerDispatch, sourcePacks - sourcePackOffset);
                    owner.Ops.DeepFillV2ReconstructChunk(context.commandBuffer, feature.texture, weights, featureShape.w, featureShape.h, featureShape.c, matchW, matchH, maskShape.w, maskShape.h, maskDownsample, patchEpsilon, softmaxScale, sourcePackOffset, sourcePackCount, sourcePackOffset == 0 ? null : output, output);
                }

                var featureStorage = AexisGraphSession.GetCmdStorageShape(feature, featureShape);
                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, featureShape, featureStorage, owned: true, blobName: layer.topNames[0]);
                if (context.shapes != null)
                    context.shapes[layer.topNames[0]] = featureShape;
                output = null;
            }
            finally
            {
                if (patchStats != null) owner.ReturnTempArray(context.commandBuffer, patchStats);
                if (scores != null) owner.ReturnTempArray(context.commandBuffer, scores);
                if (weights != null) owner.ReturnTempArray(context.commandBuffer, weights);
                if (output != null) owner.ReturnTempArray(context.commandBuffer, output);
            }

            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static AexisGraphSession.TensorRef RequireTextureInput(AexisLayerBufferContext context, string blob, string layerName)
        {
            if (context.textureBlobs == null || !context.textureBlobs.TryGetValue(blob, out var tensor) || tensor == null || tensor.texture == null)
            {
                throw new InvalidOperationException(
                    "DeepFillV2 contextual attention requires a direct Pack4 texture input"
                    + " | layer=" + layerName
                    + " | blob=" + blob
                    + " | rejected_fallback=buffer-activation");
            }
            return tensor;
        }

        private static void ValidateProfile(
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape featureShape,
            AexisGraphSession.BufferShape maskShape,
            int ksize,
            int rate,
            int stride,
            int maskDownsample,
            float patchEpsilon,
            float softmaxScale)
        {
            if (ksize != 3 || rate != 2 || stride != 1 || maskDownsample != 8)
                throw new InvalidOperationException("DeepFillV2 contextual attention only supports ksize=3, rate=2, stride=1, mask_downsample=8: " + layer.name);
            if (float.IsNaN(patchEpsilon) || float.IsInfinity(patchEpsilon) || patchEpsilon <= 0f
                || float.IsNaN(softmaxScale) || float.IsInfinity(softmaxScale) || softmaxScale <= 0f)
            {
                throw new InvalidOperationException("DeepFillV2 contextual attention requires finite positive patch_epsilon and softmax_scale: " + layer.name);
            }
            if (featureShape.dims != 3 || featureShape.w != 100 || featureShape.h != 128 || featureShape.d != 1 || featureShape.c != 96)
                throw new InvalidOperationException(
                    "DeepFillV2 contextual attention feature shape must be d3:100x128x96, got d"
                    + featureShape.dims + ":" + featureShape.w + "x" + featureShape.h + "x" + featureShape.c
                    + " | layer=" + layer.name);
            if (maskShape.dims != 3 || maskShape.w != 400 || maskShape.h != 512 || maskShape.d != 1 || maskShape.c != 1)
                throw new InvalidOperationException(
                    "DeepFillV2 contextual attention mask shape must be d3:400x512x1, got d"
                    + maskShape.dims + ":" + maskShape.w + "x" + maskShape.h + "x" + maskShape.c
                    + " | layer=" + layer.name);
        }
    }
}
