using System;
using UnityEngine;

namespace Aexis.Ncnn
{
    public sealed class NcnnDeepFillV2ContextualAttentionLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnDeepFillV2ContextualAttentionLayerRepro()
            : base(NcnnLayerTypes.DeepFillV2ContextualAttention, supportsBufferPath: false, supportsCommandBufferPath: false)
        {
        }

        public override void ExecuteBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            ExecuteRenderTexturePath(owner, layer, context);
        }

        public override void ExecuteRenderTexturePath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            if (layer == null) throw new ArgumentNullException(nameof(layer));
            if (context == null) throw new ArgumentNullException(nameof(context));
            if (layer.bottomNames == null || layer.bottomNames.Length != 2 || layer.topNames == null || layer.topNames.Length != 1)
                throw new InvalidOperationException("DeepFillV2 contextual attention requires feature+mask inputs and one output: " + layer.name);

            if (!owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var feature,
                    out var featureShape))
                throw new InvalidOperationException("DeepFillV2 contextual attention feature input is not a pack4 texture: " + layer.name);
            if (!owner.TryGetPack4Texture(
                    layer.bottomNames[1],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var mask,
                    out var maskShape))
                throw new InvalidOperationException("DeepFillV2 contextual attention mask input is not a pack4 texture: " + layer.name);

            var ksize = layer.GetInt(0, 3);
            var rate = layer.GetInt(1, 2);
            var stride = layer.GetInt(2, 1);
            var softmaxScale = layer.GetFloat(3, 10f);
            var patchEpsilon = layer.GetFloat(4, 1e-4f);
            var maskDownsample = layer.GetInt(5, 8);
            if (ksize != 3 || rate != 2 || stride != 1)
                throw new InvalidOperationException("DeepFillV2 contextual attention only supports ksize=3, rate=2, stride=1: " + layer.name);
            if (featureShape.dims != 3 || featureShape.w != 100 || featureShape.h != 128 || featureShape.c != 96)
                throw new InvalidOperationException(
                    "DeepFillV2 contextual attention feature shape must be d3:100x128x96, got d"
                    + featureShape.dims + ":" + featureShape.w + "x" + featureShape.h + "x" + featureShape.c
                    + " | layer=" + layer.name);
            if (maskShape.dims != 3 || maskShape.w != 400 || maskShape.h != 512 || maskShape.c != 1)
                throw new InvalidOperationException(
                    "DeepFillV2 contextual attention mask shape must be d3:400x512x1, got d"
                    + maskShape.dims + ":" + maskShape.w + "x" + maskShape.h + "x" + maskShape.c
                    + " | layer=" + layer.name);

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
                output = owner.RentTempArray(featureShape.w, featureShape.h, featurePacks, format);

                owner.Ops.DeepFillV2PatchStats(
                    feature.texture, mask.texture,
                    featureShape.w, featureShape.h, featureShape.c,
                    matchW, matchH, maskShape.w, maskShape.h,
                    maskDownsample, patchEpsilon, softmaxScale, patchStats);
                owner.Ops.DeepFillV2Scores(
                    feature.texture, patchStats,
                    featureShape.w, featureShape.h, featureShape.c,
                    matchW, matchH, maskShape.w, maskShape.h,
                    maskDownsample, patchEpsilon, softmaxScale, scores);
                owner.Ops.DeepFillV2Softmax(
                    scores, patchStats,
                    featureShape.w, featureShape.h, featureShape.c,
                    matchW, matchH, maskShape.w, maskShape.h,
                    maskDownsample, patchEpsilon, softmaxScale, weights);
                owner.Ops.DeepFillV2Reconstruct(
                    feature.texture, weights,
                    featureShape.w, featureShape.h, featureShape.c,
                    matchW, matchH, maskShape.w, maskShape.h,
                    maskDownsample, patchEpsilon, softmaxScale, output);

                NcnnGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, featureShape);
                output = null;
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
    }
}
