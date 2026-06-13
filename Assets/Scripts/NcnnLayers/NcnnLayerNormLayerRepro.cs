using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnLayerNormLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnLayerNormLayerRepro() : base(NcnnLayerTypes.LayerNorm, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var lp = new NcnnRepro.LayerNormPack();
                                        lp.affineSize = layer.GetInt(0, 0);
                                        lp.eps = layer.GetFloat(1, 1e-5f);
                                        lp.affine = layer.GetInt(2, 1) != 0;

                                        float[] gamma = null;
                                        float[] beta = null;
                                        if (lp.affine && lp.affineSize > 0)
                                        {
                                            phaseSw.Restart();
                                            gamma = br.ReadNcnnMatAsFloat32(lp.affineSize, 0, 0, 0, 1);
                                            beta = br.ReadNcnnMatAsFloat32(lp.affineSize, 0, 0, 0, 1);
                                            phaseSw.Stop();
                                            readMs += phaseSw.ElapsedMilliseconds;

                                            phaseSw.Restart();
                                            lp.gamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
                                            lp.beta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
                                            lp.gamma.SetData(gamma);
                                            lp.beta.SetData(beta);
                                            phaseSw.Stop();
                                            uploadMs += phaseSw.ElapsedMilliseconds;
                                        }

                                        owner._layerNorm[layer.name] = lp;
                                        return new NcnnRepro.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }
        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                                if (!owner._layerNorm.TryGetValue(layer.name, out var lp))
                                                    throw new InvalidOperationException("LayerNorm not found: " + layer.name);
                                                using var srcView = owner.GetReadableTensorInput(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                if (srcView == null || srcView.buffer == null)
                                                    throw new InvalidOperationException("LayerNorm source not found: " + layer.name);
                                                ResolveLayerNormRowsCols(srcView, lp.affineSize, layer.name, out var rows, out var cols);
                                                var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
                                                owner.Ops.CopyBuf(srcView.buffer, outTensor.buffer, srcView.buffer.count);
                                                owner.Ops.LayerNorm2DInplace(outTensor.buffer, rows, cols, lp.eps, lp.affine, lp.gamma, lp.beta);
                                                var textureFormatOverride = ShouldPromoteAttentionPrepTexture(owner, layer)
                                                    ? RenderTextureFormat.ARGBFloat
                                                    : (RenderTextureFormat?)null;
                                                owner.PublishTensorBufferOutput(
                                                    layer.topNames[0],
                                                    outTensor,
                                                    preferTexture: true,
                                                    textureBlobs,
                                                    textureShapes,
                                                    bufferBlobs,
                                                    bufferRefs,
                                                    bufferViews,
                                                    tempOwned,
                                                    textureFormatOverride);
                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            if (!owner._layerNorm.TryGetValue(layer.name, out var lp))
                throw new InvalidOperationException("LayerNorm not found: " + layer.name);
            if (!TryGetPack4WidthTexture(owner, layer, lp, textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
                throw new InvalidOperationException("LayerNorm render-texture path requires supported pack4 width-norm input: " + layer.name);

            var outFormat = ResolveLayerNormOutputFormat(owner, layer, srcShape);
            var outRt = owner.RentTempArray(
                srcTex.width,
                srcTex.height,
                srcShape.dims == 4 ? srcShape.d * srcTex.packs : srcTex.packs,
                outFormat);
            owner.Ops.LayerNormPack4WidthTex(
                srcTex.texture,
                srcShape.w,
                srcShape.h,
                srcShape.dims == 4 ? srcShape.d : 1,
                srcShape.c,
                srcTex.packs,
                lp.eps,
                lp.affine,
                lp.gamma,
                lp.beta,
                outRt);
            NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        private static void ResolveLayerNormRowsCols(NcnnTensorBuffer srcView, int affineSize, string layerName, out int rows, out int cols)
        {
            if (srcView == null)
                throw new ArgumentNullException(nameof(srcView));

            var w = srcView.w;
            var h = Mathf.Max(1, srcView.h);
            var d = Mathf.Max(1, srcView.d);
            var c = Mathf.Max(1, srcView.c);
            rows = 0;
            cols = 0;

            switch (srcView.dims)
            {
                case 1:
                {
                    rows = 1;
                    cols = w;
                    return;
                }
                case 2:
                {
                    rows = h;
                    cols = w;
                    return;
                }
                case 3:
                {
                    if (affineSize <= 0 || affineSize == w)
                    {
                        rows = h * c;
                        cols = w;
                        return;
                    }

                    if (affineSize == w * h)
                    {
                        rows = c;
                        cols = w * h;
                        return;
                    }

                    break;
                }
                case 4:
                {
                    if (affineSize <= 0 || affineSize == w)
                    {
                        rows = h * d * c;
                        cols = w;
                        return;
                    }

                    if (affineSize == w * h)
                    {
                        rows = d * c;
                        cols = w * h;
                        return;
                    }

                    if (affineSize == w * h * d)
                    {
                        rows = c;
                        cols = w * h * d;
                        return;
                    }

                    break;
                }
            }

            throw new InvalidOperationException(
                "Unsupported LayerNorm shape"
                + " | layer=" + layerName
                + " | dims=" + srcView.dims
                + " | w=" + srcView.w
                + " | h=" + srcView.h
                + " | d=" + srcView.d
                + " | c=" + srcView.c
                + " | affineSize=" + affineSize);
        }

        private static bool ShouldPromoteAttentionPrepTexture(NcnnRepro owner, NcnnParamModel.Layer layer)
        {
            if (owner?.Model?.layers == null || layer?.topNames == null || layer.topNames.Length == 0)
                return false;

            var reshape = FindSingleConsumer(owner.Model, layer.topNames[0]);
            if (reshape == null || reshape.type != NcnnLayerTypes.Reshape || reshape.topNames == null || reshape.topNames.Length == 0)
                return false;

            var gemm = FindSingleConsumer(owner.Model, reshape.topNames[0]);
            return gemm != null
                && gemm.type == NcnnLayerTypes.Gemm
                && gemm.GetInt(5, 0) != 0;
        }

        private static NcnnParamModel.Layer FindSingleConsumer(NcnnParamModel model, string blobName)
        {
            if (model?.layers == null || string.IsNullOrWhiteSpace(blobName))
                return null;

            NcnnParamModel.Layer found = null;
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

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            if (!owner._layerNorm.TryGetValue(layer.name, out var lp))
                throw new InvalidOperationException("LayerNorm not found: " + layer.name);

            if (CanUsePack4WidthCmdPath(src, srcShape, lp))
            {
                var outFormat = ResolveLayerNormOutputFormat(owner, layer, srcShape);
                var outArr = owner.RentTempArray(cmd, src.width, src.height, srcShape.dims == 4 ? srcShape.d * src.packs : src.packs, outFormat);
                owner.Ops.LayerNormPack4WidthTex(
                    cmd,
                    src.texture,
                    srcShape.w,
                    srcShape.h,
                    srcShape.dims == 4 ? srcShape.d : 1,
                    srcShape.c,
                    src.packs,
                    lp.eps,
                    lp.affine,
                    lp.gamma,
                    lp.beta,
                    outArr);
                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                {
                    texture = outArr,
                    width = src.width,
                    height = src.height,
                    packs = src.packs,
                    refs = 1,
                    owned = true,
                    hasLogicalShape = true,
                    logicalShape = srcShape,
                    hasStorageShape = true,
                    storageShape = srcShape
                };
                if (shapes != null)
                    shapes[layer.topNames[0]] = srcShape;
                owner.DebugLog?.Invoke(
                    "[CmdTexture][LayerNorm]"
                    + " | layer=" + layer.name
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | packs=" + src.packs
                    + " | outFormat=" + outArr.format);
            }
            else
            {
                owner.DebugLog?.Invoke(
                    "[CmdPlaceholder][LayerNorm]"
                    + " | layer=" + layer.name
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | packs=" + src.packs
                    + " | affine=" + (lp != null && lp.affine ? "1" : "0")
                    + " | affineSize=" + (lp != null ? lp.affineSize.ToString(CultureInfo.InvariantCulture) : "null"));
                owner.PublishCmdPlaceholder(cmd, layer.topNames[0], srcShape, blobs, shapes);
            }
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static RenderTextureFormat ResolveLayerNormOutputFormat(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnRepro.BufferShape srcShape)
        {
            if (ShouldPromoteAttentionPrepTexture(owner, layer))
                return RenderTextureFormat.ARGBFloat;
            return NcnnRepro.ResolveTensorTextureFormat(srcShape.dims);
        }

        private static bool TryGetPack4WidthTexture(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnRepro.LayerNormPack lp,
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            out NcnnRepro.TensorRef srcTex,
            out NcnnRepro.BufferShape srcShape)
        {
            srcTex = null;
            srcShape = default;
            if (lp == null)
                return false;
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out srcTex, out srcShape))
                return false;
            return CanUsePack4WidthPath(srcTex, srcShape, lp);
        }

        private static bool CanUsePack4WidthPath(NcnnRepro.TensorRef srcTex, NcnnRepro.BufferShape srcShape, NcnnRepro.LayerNormPack lp)
        {
            var logicalDepth = srcShape.dims == 4 ? Mathf.Max(1, srcShape.d) : 1;
            var expectedVolumeDepth = logicalDepth * Mathf.Max(1, srcTex?.packs ?? 0);
            return srcTex != null
                && srcTex.texture != null
                && lp != null
                && lp.affine
                && lp.gamma != null
                && lp.beta != null
                && (srcShape.dims == 3 || srcShape.dims == 4)
                && srcShape.w > 0
                && srcShape.w == lp.affineSize
                && srcShape.w == srcTex.width
                && srcShape.h == srcTex.height
                && (srcShape.dims != 4 || Mathf.Max(1, srcTex.texture.volumeDepth) == expectedVolumeDepth)
                && srcShape.c > 0
                && srcTex.packs == Mathf.CeilToInt(srcShape.c / 4f);
        }

        private static bool CanUsePack4WidthCmdPath(NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape srcShape, NcnnRepro.LayerNormPack lp)
        {
            return src != null
                && src.texture != null
                && lp != null
                && lp.affine
                && lp.gamma != null
                && lp.beta != null
                && (srcShape.dims == 3 || srcShape.dims == 4)
                && srcShape.w > 0
                && srcShape.w == lp.affineSize
                && srcShape.w == src.width
                && srcShape.h == src.height
                && (srcShape.dims != 4 || Mathf.Max(1, src.texture.depth) == Mathf.Max(1, srcShape.d) * src.packs)
                && srcShape.c > 0
                && src.packs == Mathf.CeilToInt(srcShape.c / 4f);
        }
    }
}
