using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnGroupNormLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnGroupNormLayerRepro() : base(NcnnLayerTypes.GroupNorm, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var gp = new NcnnRepro.GroupNormPack();
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
                                        return new NcnnRepro.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }
        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (!owner._groupNorm.TryGetValue(layer.name, out var gp))
                throw new InvalidOperationException("GroupNorm not found: " + layer.name);

            if (owner.EnableGroupNormTexturePath
                && owner.UseNcnnStyleGroupNorm
                && owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var srcTex,
                    out var srcShape)
                && NcnnRepro.CanUseGroupNormPack4Path(srcTex, srcShape, gp))
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

            if (!owner._groupNorm.TryGetValue(layer.name, out var gp))
                throw new InvalidOperationException("GroupNorm not found: " + layer.name);
            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
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

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            if (!owner._groupNorm.TryGetValue(layer.name, out var gp))
                throw new InvalidOperationException("GroupNorm not found: " + layer.name);
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                || !NcnnRepro.CanUseGroupNormPack4Path(srcTex, srcShape, gp))
            {
                throw new InvalidOperationException("GroupNorm render-texture path requires supported pack4 input: " + layer.name);
            }

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
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
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

            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            if (!owner._groupNorm.TryGetValue(layer.name, out var gp))
                throw new InvalidOperationException("GroupNorm not found: " + layer.name);

            if (owner.UseNcnnStyleGroupNorm && CanUsePack4CmdPath(src, srcShape, gp))
            {
                var statsA = owner.RentTempArray(cmd, gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
                var statsB = owner.RentTempArray(cmd, gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
                var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.GroupNormPack4Tex(cmd, src.texture, srcShape.w, srcShape.h, 1, srcShape.c, src.packs, gp.group, gp.eps, gp.gamma, gp.beta, statsA, statsB, outArr);
                owner.ReturnTempArray(cmd, statsA);
                owner.ReturnTempArray(cmd, statsB);
                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                {
                    texture = outArr,
                    width = src.width,
                    height = src.height,
                    packs = src.packs,
                    refs = 1,
                    owned = true
                };
                if (shapes != null)
                    shapes[layer.topNames[0]] = srcShape;
            }
            else
            {
                NcnnRepro.ResolveCmdTextureLayout(srcShape, out var width, out var height, out var packs);
                owner.PublishCmdTensorLikeInput(cmd, layer.topNames[0], width, height, packs, blobs, shapes, srcShape);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool CanUsePack4CmdPath(NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape srcShape, NcnnRepro.GroupNormPack gp)
        {
            return src != null
                && src.texture != null
                && gp != null
                && gp.affine
                && gp.gamma != null
                && gp.beta != null
                && srcShape.dims == 3
                && srcShape.w == src.width
                && srcShape.h == src.height
                && srcShape.c == gp.channels
                && gp.channels > 0
                && gp.group > 0
                && gp.channels % gp.group == 0
                && src.packs == Mathf.CeilToInt(gp.channels / 4f);
        }
    }
}
