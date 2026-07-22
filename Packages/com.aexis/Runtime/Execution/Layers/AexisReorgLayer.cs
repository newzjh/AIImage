using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisReorgLayer : AexisBaseLayer
    {
        public AexisReorgLayer()
            : base(AexisLayerTypes.Reorg, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var stride = Mathf.Max(1, layer.GetInt(0, 1));
            var mode = layer.GetInt(1, 0);

            if (stride == 2
                && mode == 0
                && owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var srcTex,
                    out var srcShape)
                && srcShape.dims == 3
                && srcShape.w == srcTex.width
                && srcShape.h == srcTex.height)
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

            var stride = Mathf.Max(1, layer.GetInt(0, 1));
            var mode = layer.GetInt(1, 0);
            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || srcView.dims != 3)
                throw new InvalidOperationException("Reorg expects dims=3 source: " + layer.name);
            if (srcView.w % stride != 0 || srcView.h % stride != 0)
                throw new InvalidOperationException("Reorg requires divisible spatial size: " + layer.name);

            var outW = srcView.w / stride;
            var outH = srcView.h / stride;
            var outC = srcView.c * stride * stride;
            var outTensor = owner.RentTempTensorBuffer(3, outW, outH, 1, outC);
            owner.Ops.ReorgBuf(srcBuf, srcView.w, srcView.h, srcView.c, stride, mode, outTensor.buffer);
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

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            var stride = Mathf.Max(1, layer.GetInt(0, 1));
            var mode = layer.GetInt(1, 0);
            if (stride != 2 || mode != 0)
                throw new InvalidOperationException("Reorg render-texture path currently supports stride=2 and mode=0 only: " + layer.name);

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                || srcShape.dims != 3
                || srcShape.w != srcTex.width
                || srcShape.h != srcTex.height)
            {
                throw new InvalidOperationException("Reorg render-texture path requires supported pack4 input: " + layer.name);
            }

            var outW = srcShape.w / 2;
            var outH = srcShape.h / 2;
            var outC = srcShape.c * 4;
            var outPacks = Mathf.CeilToInt(outC / 4f);
            var outRt = owner.RentTempArray(outW, outH, outPacks, RenderTextureFormat.ARGBHalf);
            owner.Ops.ReorgPack4(srcTex.texture, srcTex.packs, outRt);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, new AexisGraphSession.BufferShape(3, outW, outH, 1, outC));
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var stride = Mathf.Max(1, layer.GetInt(0, 1));
            var mode = layer.GetInt(1, 0);
            if (stride != 2 || mode != 0)
            {
                var rejectedShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
                throw new InvalidOperationException(
                    "Reorg CommandBuffer Pack4 supports only stride=2 and mode=0"
                    + " | layer=" + layer.name
                    + " | stride=" + stride
                    + " | mode=" + mode
                    + " | input=d" + rejectedShape.dims + ":" + rejectedShape.w + "x" + rejectedShape.h + "x" + rejectedShape.d + "x" + rejectedShape.c
                    + " | rejected_fallback=placeholder");
            }
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var outArr = owner.RentTempArray(cmd, src.width / 2, src.height / 2, src.packs * 4, RenderTextureFormat.ARGBHalf);
            owner.Ops.ReorgPack4(cmd, src.texture, src.packs, outArr);
            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outArr,
                width = src.width / 2,
                height = src.height / 2,
                packs = src.packs * 4,
                refs = 1,
                owned = true
            };
            if (shapes != null)
                shapes[layer.topNames[0]] = new AexisGraphSession.BufferShape(3, srcShape.w / 2, srcShape.h / 2, 1, srcShape.c * 4);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
