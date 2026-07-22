using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisScaleLayer : AexisBaseLayer
    {
        public AexisScaleLayer()
            : base(AexisLayerTypes.Scale, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var sp = new AexisGraphSession.ScalePack
            {
                scaleDataSize = layer.GetInt(0, 0),
                dynamic = layer.GetInt(0, 0) == -233
            };
            sp.biasTerm = !sp.dynamic && layer.GetInt(1, 0) != 0;

            if (!sp.dynamic)
            {
                phaseSw.Restart();
                sp.scaleCpu = br.ReadTensorAsFloat32(sp.scaleDataSize, 0, 0, 0, 1);
                if (sp.biasTerm)
                    sp.biasCpu = br.ReadTensorAsFloat32(sp.scaleDataSize, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                sp.scale = AexisGraphSession.NewBuffer(sp.scaleCpu);
                if (sp.biasCpu != null)
                    sp.bias = AexisGraphSession.NewBuffer(sp.biasCpu);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;
            }

            owner._extraPacks[layer.name] = sp;
            return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.ScalePack sp)
                throw new InvalidOperationException("Scale pack not found: " + layer.name);

            if (!sp.dynamic
                && sp.scaleDataSize == 1
                && !sp.biasTerm
                && owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out _,
                    out _))
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.ScalePack sp)
                throw new InvalidOperationException("Scale pack not found: " + layer.name);

            ComputeBuffer scaleBuf;
            ComputeBuffer biasBuf;
            var scaleCount = sp.scaleDataSize;
            var hasBias = sp.biasTerm;

            if (sp.dynamic)
            {
                if (layer.bottomNames.Length < 2)
                    throw new InvalidOperationException("Dynamic scale input missing: " + layer.name);
                scaleBuf = owner.GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                if (scaleBuf == null)
                    throw new InvalidOperationException("Dynamic scale buffer missing: " + layer.name);
                var scaleView = AexisGraphSession.TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
                scaleCount = scaleView != null ? scaleView.elementCount : scaleBuf.count;
                if (layer.bottomNames.Length > 2)
                {
                    biasBuf = owner.GetOrConvertToBuffer(layer.bottomNames[2], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    hasBias = biasBuf != null;
                }
                else
                {
                    biasBuf = null;
                    hasBias = false;
                }
            }
            else
            {
                scaleBuf = sp.scale;
                biasBuf = sp.bias;
            }

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Scale source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.ScaleBuf(srcBuf, srcView, scaleBuf, scaleCount, hasBias, biasBuf, outTensor.buffer);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: srcView.dims <= 3,
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
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.ScalePack sp)
                throw new InvalidOperationException("Scale pack not found: " + layer.name);
            if (sp.dynamic || sp.scaleDataSize != 1 || sp.biasTerm)
                throw new InvalidOperationException("Scale render-texture path currently requires static scalar scale without bias: " + layer.name);

            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
                throw new InvalidOperationException("Scale render-texture path requires pack4 texture input: " + layer.name);

            var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, srcTex.texture.format);
            owner.Ops.ScalePack4(srcTex.texture, sp.scaleCpu[0], srcTex.packs, outRt);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.ScalePack sp)
                throw new InvalidOperationException("Scale pack not found: " + layer.name);

            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);

            if (!sp.dynamic && sp.scaleDataSize == 1 && !sp.biasTerm)
            {
                var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, src.texture.format);
                owner.Ops.PointwisePack4(cmd, src.texture, src.packs, AexisOps.PointwiseType.ScaleScalar, sp.scaleCpu[0], 0f, outArr);
                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(
                    outArr,
                    srcShape,
                    storageShape,
                    owned: true,
                    blobName: layer.topNames[0]);
                if (shapes != null)
                    shapes[layer.topNames[0]] = srcShape;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            throw new InvalidOperationException(
                "Scale input does not match the verified scalar CommandBuffer Pack4 profile"
                + " | layer=" + layer.name
                + " | dynamic=" + sp.dynamic
                + " | scaleDataSize=" + sp.scaleDataSize
                + " | biasTerm=" + sp.biasTerm
                + " | input=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | rejected_fallback=placeholder");
        }
    }
}
