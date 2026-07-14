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

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (!owner.ShouldForceCurrentLayerBufferPath()
                && owner._layerNorm.TryGetValue(layer.name, out var lp)
                && NcnnRepro.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var src, out var srcShape)
                && CanUsePack4WidthPath(src, srcShape, lp))
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
            if (CanUsePack4Linear2DPath(srcTex, srcShape, lp))
            {
                var storageShape = NcnnRepro.GetTextureStorageShape(srcTex, srcShape);
                var outMat = owner.RentTempArray(storageShape.w, storageShape.h, 1, srcTex.texture.format);
                owner.Ops.LayerNormPack4Linear2D(srcTex.texture, srcShape.w, srcShape.h, lp.eps, lp.affine, lp.gamma, lp.beta, outMat);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outMat, srcShape, storageShape);
                owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
                return;
            }

            var isStrictLinear = NcnnRepro.IsStrictLinearMatTexture(srcTex);
            var packCount = Mathf.Max(1, Mathf.CeilToInt(Mathf.Max(1, srcShape.c) / 4f));
            var logicalDepth = srcShape.dims == 4 ? Mathf.Max(1, srcShape.d) : 1;
            var sliceCount = logicalDepth * packCount;
            RenderTexture materializedInput = null;
            RenderTexture outRt = null;
            try
            {
                var kernelInput = MaterializePack4WidthInput(owner, srcTex, srcShape, outFormat, ref materializedInput);
                outRt = owner.RentTempArray(srcShape.w, srcShape.h, sliceCount, outFormat);
                owner.Ops.LayerNormPack4WidthTex(
                    kernelInput,
                    srcShape.w,
                    srcShape.h,
                    logicalDepth,
                    srcShape.c,
                    packCount,
                    lp.eps,
                    lp.affine,
                    lp.gamma,
                    lp.beta,
                    outRt);

                if (isStrictLinear)
                {
                    var storageShape = NcnnRepro.GetTextureStorageShape(srcTex, srcShape);
                    var outMat = owner.RentTempMat(storageShape.w, storageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
                    owner.Ops.ReshapePack4ToLinearMat(outRt, srcShape.w, srcShape.h, srcShape.d, srcShape.c, srcShape.dims, outMat);
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outMat, srcShape, storageShape);
                }
                else
                {
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape, srcShape);
                    outRt = null;
                }
            }
            finally
            {
                if (materializedInput != null)
                    owner.ReturnTempArray(materializedInput);
                if (outRt != null)
                    owner.ReturnTempArray(outRt);
            }
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
                if (CanUsePack4Linear2DPath(src, srcShape, lp))
                {
                    var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
                    var outMat = owner.RentTempArray(cmd, storageShape.w, storageShape.h, 1, src.texture.format);
                    owner.Ops.LayerNormPack4Linear2D(cmd, src.texture, srcShape.w, srcShape.h, lp.eps, lp.affine, lp.gamma, lp.beta, outMat);
                    blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(outMat, srcShape, storageShape, owned: true);
                    if (shapes != null)
                        shapes[layer.topNames[0]] = srcShape;
                    owner.DebugLog?.Invoke(
                        "[CmdTexture][LayerNormPack4Linear2D]"
                        + " | layer=" + layer.name
                        + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                        + " | storage=" + storageShape.w + "x" + storageShape.h
                        + " | outFormat=" + src.texture.format);
                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                    return;
                }

                var isStrictLinear = NcnnRepro.IsStrictLinearMatTexture(src);
                var packCount = Mathf.Max(1, Mathf.CeilToInt(Mathf.Max(1, srcShape.c) / 4f));
                var logicalDepth = srcShape.dims == 4 ? Mathf.Max(1, srcShape.d) : 1;
                var sliceCount = logicalDepth * packCount;
                ComputeTexture materializedInput = null;
                ComputeTexture outArr = null;
                try
                {
                    var kernelInput = MaterializePack4WidthInput(owner, cmd, src, srcShape, outFormat, ref materializedInput);
                    outArr = owner.RentTempArray(cmd, srcShape.w, srcShape.h, sliceCount, outFormat);
                    owner.Ops.LayerNormPack4WidthTex(
                        cmd,
                        kernelInput,
                        srcShape.w,
                        srcShape.h,
                        logicalDepth,
                        srcShape.c,
                        packCount,
                        lp.eps,
                        lp.affine,
                        lp.gamma,
                        lp.beta,
                        outArr);
                    if (isStrictLinear)
                    {
                        var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
                        var outMat = owner.RentTempMat(cmd, storageShape.w, storageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
                        owner.Ops.ReshapePack4ToLinearMat(cmd, outArr, srcShape.w, srcShape.h, srcShape.d, srcShape.c, srcShape.dims, outMat);
                        blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(outMat, srcShape, storageShape, owned: true);
                    }
                    else
                    {
                        blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(outArr, srcShape, srcShape, owned: true);
                        outArr = null;
                    }
                }
                finally
                {
                    if (materializedInput != null)
                        owner.ReturnTempArray(cmd, materializedInput);
                    if (outArr != null)
                        owner.ReturnTempArray(cmd, outArr);
                }
                if (shapes != null)
                    shapes[layer.topNames[0]] = srcShape;
                owner.DebugLog?.Invoke(
                    "[CmdTexture][LayerNorm]"
                    + " | layer=" + layer.name
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | packs=" + src.packs
                    + " | outFormat=" + outFormat);
            }
            else
            {
                throw new InvalidOperationException(BuildLayerNormUnsupportedMessage(layer.name, srcShape, src, lp));
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

            var inputName = layer.bottomNames[0];

            if (!NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, inputName, out srcTex, out srcShape))
                return false;

            return CanUsePack4WidthPath(srcTex, srcShape, lp);
        }

        private static string BuildLayerNormUnsupportedMessage(
            string layerName,
            NcnnRepro.BufferShape srcShape,
            NcnnRepro.CmdTensorRef src,
            NcnnRepro.LayerNormPack lp)
        {
            return "LayerNorm command-buffer pack4 path unsupported: " + layerName
                + " | reason=requires affine width-normalized texture input"
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | packs=" + (src != null ? src.packs.ToString(CultureInfo.InvariantCulture) : "null")
                + " | affine=" + (lp != null && lp.affine ? "1" : "0")
                + " | affineSize=" + (lp != null ? lp.affineSize.ToString(CultureInfo.InvariantCulture) : "null")
                + " | rejectedFallback=placeholder-or-buffer-materialization";
        }

        private static bool CanUsePack4WidthPath(NcnnRepro.TensorRef srcTex, NcnnRepro.BufferShape srcShape, NcnnRepro.LayerNormPack lp)
        {
            if (CanUsePack4Linear2DPath(srcTex, srcShape, lp))
                return true;

            if (srcTex != null && NcnnRepro.IsStrictLinearMatTexture(srcTex))
            {
                var storageShape = NcnnRepro.GetTextureStorageShape(srcTex, srcShape);
                return lp != null
                    && lp.affine
                    && lp.gamma != null
                    && lp.beta != null
                    && (srcShape.dims == 2 || srcShape.dims == 3 || srcShape.dims == 4)
                    && srcShape.w > 0
                    && srcShape.w == lp.affineSize
                    && srcShape.h > 0
                    && srcShape.c > 0
                    && storageShape.w == srcTex.width
                    && storageShape.h == srcTex.height
                    && srcTex.packs == 1;
            }

            var logicalDepth = srcShape.dims == 4 ? Mathf.Max(1, srcShape.d) : 1;
            var expectedVolumeDepth = logicalDepth * Mathf.Max(1, srcTex?.packs ?? 0);
            return srcTex != null
                && srcTex.texture != null
                && lp != null
                && lp.affine
                && lp.gamma != null
                && lp.beta != null
                && (srcShape.dims == 2 || srcShape.dims == 3 || srcShape.dims == 4)
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
            if (CanUsePack4Linear2DPath(src, srcShape, lp))
                return true;

            if (src != null && NcnnRepro.IsStrictLinearMatTexture(src))
            {
                var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
                return src.texture != null
                    && lp != null
                    && lp.affine
                    && lp.gamma != null
                    && lp.beta != null
                    && (srcShape.dims == 2 || srcShape.dims == 3 || srcShape.dims == 4)
                    && srcShape.w > 0
                    && srcShape.w == lp.affineSize
                    && srcShape.h > 0
                    && srcShape.c > 0
                    && storageShape.w == src.width
                    && storageShape.h == src.height
                    && src.packs == 1;
            }

            return src != null
                && src.texture != null
                && lp != null
                && lp.affine
                && lp.gamma != null
                && lp.beta != null
                && (srcShape.dims == 2 || srcShape.dims == 3 || srcShape.dims == 4)
                && srcShape.w > 0
                && srcShape.w == lp.affineSize
                && srcShape.w == src.width
                && srcShape.h == src.height
                && (srcShape.dims != 4 || Mathf.Max(1, src.texture.depth) == Mathf.Max(1, srcShape.d) * src.packs)
                && srcShape.c > 0
                && src.packs == Mathf.CeilToInt(srcShape.c / 4f);
        }

        private static bool CanUsePack4Linear2DPath(NcnnRepro.TensorRef srcTex, NcnnRepro.BufferShape srcShape, NcnnRepro.LayerNormPack lp)
        {
            return srcTex != null
                && srcTex.texture != null
                && lp != null
                && lp.affine
                && lp.gamma != null
                && lp.beta != null
                && srcShape.dims == 2
                && srcShape.w > 0
                && srcShape.h > 0
                && srcShape.w == lp.affineSize
                && NcnnRepro.IsPack4LinearMatTexture(srcTex, srcShape);
        }

        private static bool CanUsePack4Linear2DPath(NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape srcShape, NcnnRepro.LayerNormPack lp)
        {
            return src != null
                && src.texture != null
                && lp != null
                && lp.affine
                && lp.gamma != null
                && lp.beta != null
                && srcShape.dims == 2
                && srcShape.w > 0
                && srcShape.h > 0
                && srcShape.w == lp.affineSize
                && NcnnRepro.IsPack4LinearMatTexture(src, srcShape);
        }

        private static RenderTexture MaterializePack4WidthInput(
            NcnnRepro owner,
            NcnnRepro.TensorRef source,
            NcnnRepro.BufferShape logicalShape,
            RenderTextureFormat pack4Format,
            ref RenderTexture materialized)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (source == null || source.texture == null)
                throw new ArgumentNullException(nameof(source));

            if (!NcnnRepro.IsStrictLinearMatTexture(source))
                return source.texture;

            var storageShape = NcnnRepro.GetTextureStorageShape(source, logicalShape);
            var sliceCount = logicalShape.dims >= 4
                ? Mathf.Max(1, logicalShape.d) * Mathf.Max(1, Mathf.CeilToInt(Mathf.Max(1, logicalShape.c) / 4f))
                : Mathf.Max(1, Mathf.CeilToInt(Mathf.Max(1, logicalShape.c) / 4f));
            materialized = owner.RentTempArray(logicalShape.w, logicalShape.h, sliceCount, pack4Format);
            owner.Ops.ReshapeLinearMatToPack4(
                source.texture,
                storageShape.w,
                storageShape.h,
                logicalShape.w,
                logicalShape.h,
                logicalShape.d,
                logicalShape.c,
                logicalShape.dims,
                materialized);
            return materialized;
        }

        private static ComputeTexture MaterializePack4WidthInput(
            NcnnRepro owner,
            CommandBuffer cmd,
            NcnnRepro.CmdTensorRef source,
            NcnnRepro.BufferShape logicalShape,
            RenderTextureFormat pack4Format,
            ref ComputeTexture materialized)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (source == null || source.texture == null)
                throw new ArgumentNullException(nameof(source));

            if (!NcnnRepro.IsStrictLinearMatTexture(source))
                return source.texture;

            var storageShape = NcnnRepro.GetCmdStorageShape(source, logicalShape);
            var sliceCount = logicalShape.dims >= 4
                ? Mathf.Max(1, logicalShape.d) * Mathf.Max(1, Mathf.CeilToInt(Mathf.Max(1, logicalShape.c) / 4f))
                : Mathf.Max(1, Mathf.CeilToInt(Mathf.Max(1, logicalShape.c) / 4f));
            materialized = owner.RentTempArray(cmd, logicalShape.w, logicalShape.h, sliceCount, pack4Format);
            owner.Ops.ReshapeLinearMatToPack4(
                cmd,
                source.texture,
                storageShape.w,
                storageShape.h,
                logicalShape.w,
                logicalShape.h,
                logicalShape.d,
                logicalShape.c,
                logicalShape.dims,
                materialized);
            return materialized;
        }
    }
}
