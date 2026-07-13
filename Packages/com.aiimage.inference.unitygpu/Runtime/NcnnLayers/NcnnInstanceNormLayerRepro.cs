using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnInstanceNormLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnInstanceNormLayerRepro()
            : base(NcnnLayerTypes.InstanceNorm, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var gp = new NcnnRepro.GroupNormPack
            {
                group = Mathf.Max(1, layer.GetInt(0, 0)),
                channels = Mathf.Max(1, layer.GetInt(0, 0)),
                eps = layer.GetFloat(1, 0.001f),
                affine = layer.GetInt(2, 1) != 0
            };

            if (gp.affine)
            {
                phaseSw.Restart();
                var gamma = br.ReadNcnnMatAsFloat32(gp.channels, 0, 0, 0, 1);
                var beta = br.ReadNcnnMatAsFloat32(gp.channels, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                gp.gamma = NcnnRepro.NewBuffer(gamma);
                gp.beta = NcnnRepro.NewBuffer(beta);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;
            }

            owner._extraPacks[layer.name] = gp;
            return new NcnnRepro.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.GroupNormPack gp)
                throw new InvalidOperationException("InstanceNorm pack not found: " + layer.name);

            if (!owner.ShouldForceCurrentLayerBufferPath()
                && owner.EnableGroupNormTexturePath
                && owner.UseNcnnStyleGroupNorm
                && gp.affine
                && owner.TryGetPack4Texture(layer.bottomNames[0], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out var srcTex, out var srcShape)
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.GroupNormPack gp)
                throw new InvalidOperationException("InstanceNorm pack not found: " + layer.name);
            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || (srcView.dims != 3 && srcView.dims != 4))
                throw new InvalidOperationException("InstanceNorm expects dims=3/4 tensor input: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.CopyBuf(srcBuf, outTensor.buffer, srcBuf.count);
            var channelStats = owner.RentTempBuffer(srcView.c, sizeof(float) * 4);
            try
            {
                var spatial = srcView.w * srcView.h * Mathf.Max(1, srcView.d);
                owner.Ops.GroupNormInplace(outTensor.buffer, spatial, 1, srcView.c, srcView.c, gp.eps, gp.affine, gp.gamma, gp.beta, channelStats, true);
            }
            finally
            {
                owner.ReturnTempBuffer(channelStats);
            }

            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: true,
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.GroupNormPack gp)
                throw new InvalidOperationException("InstanceNorm pack not found: " + layer.name);
            if (!gp.affine
                || !owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                || !NcnnRepro.CanUseGroupNormPack4Path(srcTex, srcShape, gp))
            {
                throw new InvalidOperationException("InstanceNorm render-texture path requires affine supported pack4 input: " + layer.name);
            }

            var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcShape.dims == 4 ? srcShape.d * srcTex.packs : srcTex.packs, RenderTextureFormat.ARGBHalf);
            var statsA = owner.RentTempArray(gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
            var statsB = owner.RentTempArray(gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
            try
            {
                owner.Ops.GroupNormPack4Tex(srcTex.texture, srcShape.w, srcShape.h, srcShape.dims == 4 ? srcShape.d : 1, srcShape.c, srcTex.packs, gp.group, gp.eps, gp.gamma, gp.beta, statsA, statsB, outRt);
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
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.GroupNormPack gp)
                throw new InvalidOperationException("InstanceNorm pack not found: " + layer.name);

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
            }
            else
            {
                owner.DebugLog?.Invoke(
                    "[CmdPlaceholder][InstanceNorm]"
                    + " | layer=" + layer.name
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | packs=" + src.packs
                    + " | affine=" + (gp.affine ? "1" : "0")
                    + " | channels=" + gp.channels.ToString()
                    + " | group=" + gp.group.ToString());
                if (owner.DisallowBufferAccess || owner.DisallowBufferOutputs || owner.DisallowBufferToTextureMaterialization)
                {
                    throw new InvalidOperationException(
                        "pack4-only guard: command-buffer InstanceNorm placeholder disallowed"
                        + " | layer=" + layer.name
                        + " | dims=" + srcShape.dims
                        + " | shape=" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c);
                }
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
    }
}
