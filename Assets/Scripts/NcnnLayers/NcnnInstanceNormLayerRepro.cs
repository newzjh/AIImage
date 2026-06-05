using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
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

            if (owner.EnableGroupNormTexturePath
                && owner.UseNcnnStyleGroupNorm
                && gp.affine
                && owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                && NcnnRepro.CanUseGroupNormPack4Path(srcTex, srcShape, gp))
            {
                var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
                var statsA = owner.RentTempArray(gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
                var statsB = owner.RentTempArray(gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
                try
                {
                    owner.Ops.GroupNormPack4Tex(srcTex.texture, srcShape.w, srcShape.h, srcShape.c, srcTex.packs, gp.group, gp.eps, gp.gamma, gp.beta, statsA, statsB, outRt);
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

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || srcView.dims != 3)
                throw new InvalidOperationException("InstanceNorm expects dims=3 tensor input: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(3, srcView.w, srcView.h, 1, srcView.c);
            owner.Ops.CopyBuf(srcBuf, outTensor.buffer, srcBuf.count);
            var channelStats = owner.RentTempBuffer(srcView.c, sizeof(float) * 4);
            try
            {
                owner.Ops.GroupNormInplace(outTensor.buffer, srcView.w, srcView.h, srcView.c, srcView.c, gp.eps, gp.affine, gp.gamma, gp.beta, channelStats, true);
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
                var statsA = owner.RentTempArray(cmd, gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
                var statsB = owner.RentTempArray(cmd, gp.group, 1, 1, RenderTextureFormat.ARGBFloat);
                var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.GroupNormPack4Tex(cmd, src.texture, srcShape.w, srcShape.h, srcShape.c, src.packs, gp.group, gp.eps, gp.gamma, gp.beta, statsA, statsB, outArr);
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
                && srcShape.d == 1
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
