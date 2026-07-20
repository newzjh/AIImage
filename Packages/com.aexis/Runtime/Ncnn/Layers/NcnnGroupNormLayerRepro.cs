using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnGroupNormLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnGroupNormLayerRepro() : base(NcnnLayerTypes.GroupNorm, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override NcnnGraphSession.LayerLoadMetrics LoadLayer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var gp = new NcnnGraphSession.GroupNormPack();
                                        gp.group = layer.GetInt(0, 1);
                                        gp.channels = layer.GetInt(1, 0);
                                        gp.eps = layer.GetFloat(2, 1e-5f);
                                        gp.affine = layer.GetInt(3, 1) != 0;

                                        float[] gamma = null;
                                        float[] beta = null;
                                        if (gp.affine && gp.channels > 0)
                                        {
                                            phaseSw.Restart();
                                            gamma = br.ReadNcnnMatAsFloat32(gp.channels, 0, 0, 0, 1);
                                            beta = br.ReadNcnnMatAsFloat32(gp.channels, 0, 0, 0, 1);
                                            phaseSw.Stop();
                                            readMs += phaseSw.ElapsedMilliseconds;

                                            phaseSw.Restart();
                                            gp.gamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
                                            gp.beta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
                                            gp.gamma.SetData(gamma);
                                            gp.beta.SetData(beta);
                                            phaseSw.Stop();
                                            uploadMs += phaseSw.ElapsedMilliseconds;
                                        }

                                        owner._groupNorm[layer.name] = gp;
                                        return new NcnnGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }
        public override void ExecuteBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (!owner._groupNorm.TryGetValue(layer.name, out var gp))
                throw new InvalidOperationException("GroupNorm not found: " + layer.name);

            if (owner.EnableGroupNormTexturePath
                && owner.UseNcnnStyleGroupNorm
                && NcnnGraphSession.TryGetExistingTexture(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.bottomNames[0],
                    out var srcTex,
                    out var srcShape)
                && (NcnnGraphSession.CanUseGroupNormPack4Path(srcTex, srcShape, gp)
                    || CanUseLinearMatPath(srcTex, srcShape, gp)))
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            if (!owner._groupNorm.TryGetValue(layer.name, out var gp))
                throw new InvalidOperationException("GroupNorm not found: " + layer.name);
            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("GroupNorm source not found: " + layer.name);
            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.CopyBuf(srcBuf, outTensor.buffer, srcBuf.count);
            var spatial = srcBuf.count / Mathf.Max(1, gp.channels);
            owner.Ops.GroupNormInplace(
                outTensor.buffer,
                spatial,
                1,
                gp.channels,
                gp.group,
                gp.eps,
                gp.affine,
                gp.gamma,
                gp.beta,
                owner.UseNcnnStyleGroupNorm);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: srcView.dims <= 4,
                textureBlobs,
                textureShapes,
                bufferBlobs,
                bufferRefs,
                bufferViews,
                tempOwned);
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;

            if (!owner._groupNorm.TryGetValue(layer.name, out var gp))
                throw new InvalidOperationException("GroupNorm not found: " + layer.name);
            if (!NcnnGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape))
            {
                throw new InvalidOperationException("GroupNorm render-texture path requires existing texture input: " + layer.name);
            }

            if (CanUseLinearMatPath(srcTex, srcShape, gp))
            {
                ExecuteLinearMatRenderTexturePath(owner, layer, gp, srcTex, srcShape, textureBlobs, textureShapes);
            }
            else if (NcnnGraphSession.CanUseGroupNormPack4Path(srcTex, srcShape, gp))
            {
                var outRt = owner.RentTempArray(
                    srcTex.width,
                    srcTex.height,
                    srcShape.dims == 4 ? srcShape.d * srcTex.packs : srcTex.packs,
                    RenderTextureFormat.ARGBHalf);
                var statsA = owner.RentTempArray(gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
                var statsB = owner.RentTempArray(gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
                var logicalD = srcShape.dims == 4 ? srcShape.d : 1;
                try
                {
                    owner.Ops.GroupNormPack4Tex(srcTex.texture, srcShape.w, srcShape.h, logicalD, srcShape.c, srcTex.packs, gp.group, gp.eps, gp.gamma, gp.beta, statsA, statsB, outRt);
                    NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
                    outRt = null;
                }
                finally
                {
                    if (statsA != null)
                        owner.ReturnTempArray(statsA);
                    if (statsB != null)
                        owner.ReturnTempArray(statsB);
                    if (outRt != null)
                        owner.ReturnTempArray(outRt);
                }
            }
            else
            {
                throw new InvalidOperationException(BuildUnsupportedMessage(layer.name, srcShape, srcTex != null ? srcTex.packs : 0, gp));
            }

            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = NcnnGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            if (!owner._groupNorm.TryGetValue(layer.name, out var gp))
                throw new InvalidOperationException("GroupNorm not found: " + layer.name);

            if (owner.UseNcnnStyleGroupNorm && CanUsePack4CmdPath(src, srcShape, gp))
            {
                var logicalDepth = srcShape.dims == 4 ? Mathf.Max(1, srcShape.d) : 1;
                var statsA = owner.RentTempArray(cmd, gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
                var statsB = owner.RentTempArray(cmd, gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
                var outDepth = srcShape.dims == 4 ? logicalDepth * src.packs : src.packs;
                var outArr = owner.RentTempArray(cmd, src.width, src.height, outDepth, RenderTextureFormat.ARGBHalf);
                owner.Ops.GroupNormPack4Tex(cmd, src.texture, srcShape.w, srcShape.h, logicalDepth, srcShape.c, src.packs, gp.group, gp.eps, gp.gamma, gp.beta, statsA, statsB, outArr);
                owner.ReturnTempArray(cmd, statsA);
                owner.ReturnTempArray(cmd, statsB);
                blobs[layer.topNames[0]] = new NcnnGraphSession.CmdTensorRef
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
            }
            else if (owner.UseNcnnStyleGroupNorm && CanUseLinearMatCmdPath(src, srcShape, gp))
            {
                ExecuteLinearMatCommandBufferPath(owner, layer, gp, src, srcShape, cmd, blobs, shapes);
            }
            else
            {
                throw new InvalidOperationException(BuildUnsupportedMessage(layer.name, srcShape, src != null ? src.packs : 0, gp));
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static void ExecuteLinearMatRenderTexturePath(
            NcnnGraphSession owner,
            NcnnParamModel.Layer layer,
            NcnnGraphSession.GroupNormPack gp,
            NcnnGraphSession.TensorRef srcTex,
            NcnnGraphSession.BufferShape srcShape,
            Dictionary<string, NcnnGraphSession.TensorRef> textureBlobs,
            Dictionary<string, NcnnGraphSession.BufferShape> textureShapes)
        {
            var storageShape = NcnnGraphSession.GetTextureStorageShape(srcTex, srcShape);
            var pack4Shape = BuildLinearMatPack4Shape(srcShape, gp);
            var packCount = Mathf.Max(1, Mathf.CeilToInt(pack4Shape.c / 4f));
            RenderTexture materializedInput = null;
            RenderTexture outArr = null;
            RenderTexture linearOut = null;
            var statsA = owner.RentTempArray(gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
            var statsB = owner.RentTempArray(gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
            try
            {
                materializedInput = owner.RentTempArray(pack4Shape.w, pack4Shape.h, packCount, RenderTextureFormat.ARGBFloat);
                owner.Ops.ReshapeLinearMatToPack4(
                    srcTex.texture,
                    storageShape.w,
                    storageShape.h,
                    pack4Shape.w,
                    pack4Shape.h,
                    pack4Shape.d,
                    pack4Shape.c,
                    pack4Shape.dims,
                    materializedInput);

                outArr = owner.RentTempArray(pack4Shape.w, pack4Shape.h, packCount, RenderTextureFormat.ARGBFloat);
                owner.Ops.GroupNormPack4Tex(
                    materializedInput,
                    pack4Shape.w,
                    pack4Shape.h,
                    1,
                    pack4Shape.c,
                    packCount,
                    gp.group,
                    gp.eps,
                    gp.gamma,
                    gp.beta,
                    statsA,
                    statsB,
                    outArr);

                linearOut = owner.RentTempMat(storageShape.w, storageShape.h, NcnnGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.ReshapePack4ToLinearMat(
                    outArr,
                    pack4Shape.w,
                    pack4Shape.h,
                    pack4Shape.d,
                    pack4Shape.c,
                    pack4Shape.dims,
                    linearOut);
                NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], linearOut, srcShape, storageShape);
                owner.DebugLog?.Invoke(
                    "[Texture][GroupNormLinearMat]"
                    + " | layer=" + layer.name
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | storage=d" + storageShape.dims + ":" + storageShape.w + "x" + storageShape.h + "x" + storageShape.d + "x" + storageShape.c
                    + " | pack4=d" + pack4Shape.dims + ":" + pack4Shape.w + "x" + pack4Shape.h + "x" + pack4Shape.d + "x" + pack4Shape.c
                    + " | packs=" + packCount.ToString(CultureInfo.InvariantCulture));
                linearOut = null;
            }
            finally
            {
                if (statsA != null)
                    owner.ReturnTempArray(statsA);
                if (statsB != null)
                    owner.ReturnTempArray(statsB);
                if (materializedInput != null)
                    owner.ReturnTempArray(materializedInput);
                if (outArr != null)
                    owner.ReturnTempArray(outArr);
                if (linearOut != null)
                    owner.ReturnTempArray(linearOut);
            }
        }

        private static void ExecuteLinearMatCommandBufferPath(
            NcnnGraphSession owner,
            NcnnParamModel.Layer layer,
            NcnnGraphSession.GroupNormPack gp,
            NcnnGraphSession.CmdTensorRef src,
            NcnnGraphSession.BufferShape srcShape,
            CommandBuffer cmd,
            Dictionary<string, NcnnGraphSession.CmdTensorRef> blobs,
            Dictionary<string, NcnnGraphSession.BufferShape> shapes)
        {
            var storageShape = NcnnGraphSession.GetCmdStorageShape(src, srcShape);
            var pack4Shape = BuildLinearMatPack4Shape(srcShape, gp);
            var packCount = Mathf.Max(1, Mathf.CeilToInt(pack4Shape.c / 4f));
            var statsA = owner.RentTempArray(cmd, gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
            var statsB = owner.RentTempArray(cmd, gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
            ComputeTexture materializedInput = null;
            ComputeTexture outArr = null;
            try
            {
                materializedInput = owner.RentTempArray(cmd, pack4Shape.w, pack4Shape.h, packCount, RenderTextureFormat.ARGBFloat);
                owner.Ops.ReshapeLinearMatToPack4(
                    cmd,
                    src.texture,
                    storageShape.w,
                    storageShape.h,
                    pack4Shape.w,
                    pack4Shape.h,
                    pack4Shape.d,
                    pack4Shape.c,
                    pack4Shape.dims,
                    materializedInput);

                outArr = owner.RentTempArray(cmd, pack4Shape.w, pack4Shape.h, packCount, RenderTextureFormat.ARGBFloat);
                owner.Ops.GroupNormPack4Tex(
                    cmd,
                    materializedInput,
                    pack4Shape.w,
                    pack4Shape.h,
                    1,
                    pack4Shape.c,
                    packCount,
                    gp.group,
                    gp.eps,
                    gp.gamma,
                    gp.beta,
                    statsA,
                    statsB,
                    outArr);

                var linearOut = owner.RentTempMat(cmd, storageShape.w, storageShape.h, NcnnGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.ReshapePack4ToLinearMat(
                    cmd,
                    outArr,
                    pack4Shape.w,
                    pack4Shape.h,
                    pack4Shape.d,
                    pack4Shape.c,
                    pack4Shape.dims,
                    linearOut);
                blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(linearOut, srcShape, storageShape, owned: true);
                if (shapes != null)
                    shapes[layer.topNames[0]] = srcShape;
                owner.DebugLog?.Invoke(
                    "[CmdTexture][GroupNormLinearMat]"
                    + " | layer=" + layer.name
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | storage=d" + storageShape.dims + ":" + storageShape.w + "x" + storageShape.h + "x" + storageShape.d + "x" + storageShape.c
                    + " | pack4=d" + pack4Shape.dims + ":" + pack4Shape.w + "x" + pack4Shape.h + "x" + pack4Shape.d + "x" + pack4Shape.c
                    + " | packs=" + packCount.ToString(CultureInfo.InvariantCulture));
            }
            finally
            {
                owner.ReturnTempArray(cmd, statsA);
                owner.ReturnTempArray(cmd, statsB);
                if (materializedInput != null)
                    owner.ReturnTempArray(cmd, materializedInput);
                if (outArr != null)
                    owner.ReturnTempArray(cmd, outArr);
            }
        }

        private static bool CanUsePack4CmdPath(NcnnGraphSession.CmdTensorRef src, NcnnGraphSession.BufferShape srcShape, NcnnGraphSession.GroupNormPack gp)
        {
            return src != null
                && src.texture != null
                && gp != null
                && gp.affine
                && gp.gamma != null
                && gp.beta != null
                && (srcShape.dims == 3 || srcShape.dims == 4)
                && srcShape.w == src.width
                && srcShape.h == src.height
                && (srcShape.dims != 4 || Mathf.Max(1, src.texture.depth) == Mathf.Max(1, srcShape.d) * src.packs)
                && srcShape.c == gp.channels
                && gp.channels > 0
                && gp.group > 0
                && gp.channels % gp.group == 0
                && src.packs == Mathf.CeilToInt(gp.channels / 4f);
        }

        private static bool CanUseLinearMatPath(NcnnGraphSession.TensorRef src, NcnnGraphSession.BufferShape srcShape, NcnnGraphSession.GroupNormPack gp)
        {
            if (src == null || src.texture == null || !NcnnGraphSession.IsStrictLinearMatTexture(src))
                return false;

            var storageShape = NcnnGraphSession.GetTextureStorageShape(src, srcShape);
            return gp != null
                && gp.affine
                && gp.gamma != null
                && gp.beta != null
                && srcShape.dims == 2
                && srcShape.w > 0
                && srcShape.h == gp.channels
                && gp.channels > 0
                && gp.group > 0
                && gp.channels % gp.group == 0
                && storageShape.dims == 2
                && storageShape.w == src.width
                && storageShape.h == src.height
                && src.packs == 1;
        }

        private static bool CanUseLinearMatCmdPath(NcnnGraphSession.CmdTensorRef src, NcnnGraphSession.BufferShape srcShape, NcnnGraphSession.GroupNormPack gp)
        {
            if (src == null || src.texture == null || !NcnnGraphSession.IsStrictLinearMatTexture(src))
                return false;

            var storageShape = NcnnGraphSession.GetCmdStorageShape(src, srcShape);
            return gp != null
                && gp.affine
                && gp.gamma != null
                && gp.beta != null
                && srcShape.dims == 2
                && srcShape.w > 0
                && srcShape.h == gp.channels
                && gp.channels > 0
                && gp.group > 0
                && gp.channels % gp.group == 0
                && storageShape.dims == 2
                && storageShape.w == src.width
                && storageShape.h == src.height
                && src.packs == 1;
        }

        private static NcnnGraphSession.BufferShape BuildLinearMatPack4Shape(NcnnGraphSession.BufferShape srcShape, NcnnGraphSession.GroupNormPack gp)
        {
            if (srcShape.dims != 2)
                throw new InvalidOperationException("Linear-mat GroupNorm requires dims=2 input.");
            if (gp == null || srcShape.h != gp.channels)
                throw new InvalidOperationException("Linear-mat GroupNorm requires height=channels.");
            return new NcnnGraphSession.BufferShape(3, srcShape.w, 1, 1, gp.channels);
        }

        private static string BuildUnsupportedMessage(
            string layerName,
            NcnnGraphSession.BufferShape srcShape,
            int packs,
            NcnnGraphSession.GroupNormPack gp)
        {
            return "GroupNorm pack4 path unsupported: " + layerName
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | packs=" + packs.ToString(CultureInfo.InvariantCulture)
                + " | affine=" + (gp != null && gp.affine ? "1" : "0")
                + " | channels=" + (gp != null ? gp.channels.ToString(CultureInfo.InvariantCulture) : "null")
                + " | group=" + (gp != null ? gp.group.ToString(CultureInfo.InvariantCulture) : "null")
                + " | supported=pack4_3d4d_or_linear_mat_dims2_height_equals_channels";
        }
    }
}
