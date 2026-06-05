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

            if (gp.affine
                && owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                && NcnnRepro.CanUseGroupNormPack4Path(srcTex, srcShape, gp))
            {
                var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
                var stats = owner.RentTempBuffer(gp.group, sizeof(float) * 4);
                try
                {
                    owner.Ops.GroupNormPack4(srcTex.texture, srcShape.w, srcShape.h, srcShape.c, srcTex.packs, gp.group, gp.eps, gp.gamma, gp.beta, stats, outRt);
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
                    outRt = null;
                }
                finally
                {
                    owner.ReturnTempBuffer(stats);
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
            var stats = owner.RentTempBuffer(srcView.c, sizeof(float) * 4);
            try
            {
                owner.Ops.GroupNormInplace(outTensor.buffer, srcView.w, srcView.h, srcView.c, srcView.c, gp.eps, gp.affine, gp.gamma, gp.beta, stats, true);
            }
            finally
            {
                owner.ReturnTempBuffer(stats);
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
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            owner.CopyCmdTensor(cmd, src, layer.topNames[0], blobs, src.width, src.height, src.packs);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
        }
    }
}
