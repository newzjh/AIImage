using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisMaxPoolingIndLayer : AexisBaseLayer
    {
        public AexisMaxPoolingIndLayer() : base(AexisLayerTypes.MaxPoolingInd, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var kernelW = layer.GetInt(1, 0);
            var kernelH = layer.GetInt(11, kernelW);
            var strideW = layer.GetInt(2, 1);
            var strideH = layer.GetInt(12, strideW);
            var padLeft = layer.GetInt(3, 0);
            var padRight = layer.GetInt(14, padLeft);
            var padTop = layer.GetInt(13, padLeft);
            var padBottom = layer.GetInt(15, padTop);

            if (owner.TryGetPack4Texture(
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
            var indexBlobs = context.indexBlobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            var kernelW = layer.GetInt(1, 0);
            var kernelH = layer.GetInt(11, kernelW);
            var strideW = layer.GetInt(2, 1);
            var strideH = layer.GetInt(12, strideW);
            var padLeft = layer.GetInt(3, 0);
            var padRight = layer.GetInt(14, padLeft);
            var padTop = layer.GetInt(13, padLeft);
            var padBottom = layer.GetInt(15, padTop);
            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || srcView.dims != 3)
                throw new InvalidOperationException("MaxPoolingInd expects dims=3 buffer input: " + layer.name);

            var outW = Mathf.Max(1, (srcView.w + padLeft + padRight - kernelW) / Mathf.Max(1, strideW) + 1);
            var outH = Mathf.Max(1, (srcView.h + padTop + padBottom - kernelH) / Mathf.Max(1, strideH) + 1);
            var outTensor = owner.RentTempTensorBuffer(3, outW, outH, 1, srcView.c);
            var idxTensor = owner.RentTempTensorBuffer(3, outW, outH, 1, srcView.c);
            owner.ApplyMaxPoolingIndCpu(srcBuf, srcView, kernelW, kernelH, strideW, strideH, padLeft, padTop, outTensor, idxTensor);
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
            indexBlobs[layer.topNames[1]] = new AexisGraphSession.IndexRef
            {
                buffer = idxTensor.buffer,
                view = idxTensor,
                width = outW,
                height = outH,
                packs = Mathf.Max(1, Mathf.CeilToInt(srcView.c / 4f)),
                sourceWidth = srcView.w,
                sourceHeight = srcView.h,
                kernelW = kernelW,
                kernelH = kernelH,
                strideW = strideW,
                strideH = strideH,
                padLeft = padLeft,
                padTop = padTop,
                refs = owner._blobUseCount.TryGetValue(layer.topNames[1], out var idxUseCount) ? idxUseCount : 1,
                owned = true
            };
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var indexBlobs = context.indexBlobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var kernelW = layer.GetInt(1, 0);
            var kernelH = layer.GetInt(11, kernelW);
            var strideW = layer.GetInt(2, 1);
            var strideH = layer.GetInt(12, strideW);
            var padLeft = layer.GetInt(3, 0);
            var padRight = layer.GetInt(14, padLeft);
            var padTop = layer.GetInt(13, padLeft);
            var padBottom = layer.GetInt(15, padTop);
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var src, out var srcShape))
                throw new InvalidOperationException("MaxPoolingInd render-texture path requires pack4 input: " + layer.name);

            var outW = Mathf.Max(1, (src.width + padLeft + padRight - kernelW) / Mathf.Max(1, strideW) + 1);
            var outH = Mathf.Max(1, (src.height + padTop + padBottom - kernelH) / Mathf.Max(1, strideH) + 1);
            var outRt = owner.RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
            var idxRt = owner.RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBFloat);
            if (owner.UseTextureMaxPoolingInd)
            {
                owner.Ops.PoolingPack4(src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, 0, outRt);
                owner.Ops.MaxPoolingIndicesFromValuePack4(src.texture, outRt, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, idxRt);
            }
            else
            {
                owner.ApplyMaxPoolingIndCpu(src, srcShape, kernelW, kernelH, strideW, strideH, padLeft, padTop, outW, outH, outRt, idxRt);
            }

            if (owner.DebugCompareMaxPoolingLayers != null
                && (owner.DebugCompareMaxPoolingLayers.Contains(layer.name) || owner.DebugCompareMaxPoolingLayers.Contains("*")))
            {
                owner.CompareMaxPoolingIndPath(layer.name, src, srcShape, outRt, idxRt, kernelW, kernelH, strideW, strideH, padLeft, padTop, outW, outH);
            }

            textureBlobs[layer.topNames[0]] = new AexisGraphSession.TensorRef
            {
                texture = outRt,
                width = outW,
                height = outH,
                packs = src.packs,
                refs = 1,
                owned = true
            };
            textureShapes[layer.topNames[0]] = new AexisGraphSession.BufferShape(3, outW, outH, 1, srcShape.c);
            indexBlobs[layer.topNames[1]] = new AexisGraphSession.IndexRef
            {
                texture = idxRt,
                width = outW,
                height = outH,
                packs = src.packs,
                sourceWidth = src.width,
                sourceHeight = src.height,
                kernelW = kernelW,
                kernelH = kernelH,
                strideW = strideW,
                strideH = strideH,
                padLeft = padLeft,
                padTop = padTop,
                refs = owner._blobUseCount.TryGetValue(layer.topNames[1], out var idxUseCount) ? idxUseCount : 1,
                owned = true
            };
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var kernelW = layer.GetInt(1, 0);
            var kernelH = layer.GetInt(11, kernelW);
            var strideW = layer.GetInt(2, 1);
            var strideH = layer.GetInt(12, strideW);
            var padLeft = layer.GetInt(3, 0);
            var padRight = layer.GetInt(14, padLeft);
            var padTop = layer.GetInt(13, padLeft);
            var padBottom = layer.GetInt(15, padTop);
            var outW = Mathf.Max(1, (srcShape.w + padLeft + padRight - kernelW) / Mathf.Max(1, strideW) + 1);
            var outH = Mathf.Max(1, (srcShape.h + padTop + padBottom - kernelH) / Mathf.Max(1, strideH) + 1);
            var outShape = new AexisGraphSession.BufferShape(3, outW, outH, 1, srcShape.c);
            var outArr = owner.RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
            ComputeTexture idxArr = null;
            if (layer.topNames.Length > 1)
            {
                idxArr = owner.RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBFloat);
                owner.Ops.MaxPoolingIndPack4(cmd, src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, outArr, idxArr);
            }
            else
            {
                owner.Ops.PoolingPack4(cmd, src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, 0, outArr);
            }

            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outArr,
                width = outW,
                height = outH,
                packs = src.packs,
                refs = 1,
                owned = true
            };
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            if (layer.topNames.Length > 1)
            {
                blobs[layer.topNames[1]] = new AexisGraphSession.CmdTensorRef
                {
                    texture = idxArr,
                    width = outW,
                    height = outH,
                    packs = src.packs,
                    sourceWidth = srcShape.w,
                    sourceHeight = srcShape.h,
                    kernelW = kernelW,
                    kernelH = kernelH,
                    strideW = strideW,
                    strideH = strideH,
                    padLeft = padLeft,
                    padTop = padTop,
                    refs = 1,
                    owned = true
                };
                if (shapes != null)
                    shapes[layer.topNames[1]] = outShape;
            }
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
