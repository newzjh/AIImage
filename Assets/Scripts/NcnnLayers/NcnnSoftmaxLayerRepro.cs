using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnSoftmaxLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnSoftmaxLayerRepro() : base(NcnnLayerTypes.Softmax, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var softSrcTex,
                    out var softSrcShape)
                && TryResolvePack4SoftmaxWidthAxis(layer, softSrcShape, out _)
                && CanUsePack4Softmax(softSrcTex, softSrcShape))
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

            using var softTensor = owner.GetReadableTensorInput(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var softBuf = softTensor?.buffer;
            if (softBuf != null)
            {
                var axis = layer.GetInt(0, 0);
                if (axis < 0)
                    axis += softTensor.dims;
                var outTensor = owner.RentTempTensorBuffer(
                    softTensor.dims,
                    softTensor.w,
                    softTensor.h,
                    softTensor.d,
                    softTensor.c);

                if (TryResolveContiguousSoftmax2D(softTensor, axis, out var rows, out var cols))
                {
                    owner.Ops.Softmax2D(softBuf, outTensor.buffer, rows, cols);
                }
                else
                {
                    throw new InvalidOperationException("Softmax buffer path unsupported axis for dims=" + softTensor.dims + ": " + layer.name + " axis=" + axis);
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
            }
            else
            {
                var src = NcnnRepro.GetTexture(textureBlobs, layer.bottomNames[0]);
                var outRt = owner.RentTempArray(src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.SoftmaxChannelPack4(src.texture, src.packs, outRt);
                textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
                {
                    texture = outRt,
                    width = src.width,
                    height = src.height,
                    packs = src.packs,
                    refs = 1,
                    owned = true
                };
                var softShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, src.width, src.height, 1, softShape.c);
            }
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                || !CanUsePack4Softmax(srcTex, srcShape))
            {
                throw new InvalidOperationException("Softmax render-texture path requires supported pack4 input: " + layer.name);
            }

            if (!TryResolvePack4SoftmaxWidthAxis(layer, srcShape, out var tensorAxis))
                throw new InvalidOperationException("Softmax render-texture path currently supports softmax over tensor width axis only: " + layer.name);

            if (srcShape.dims == 4 && tensorAxis == 0)
            {
                var outSlices = Mathf.Max(1, srcShape.d) * Mathf.Max(1, Mathf.CeilToInt(srcShape.c / 4f));
                var outRt4 = owner.RentTempArray(srcTex.width, srcTex.height, outSlices, NcnnRepro.ResolveTensorTextureFormat(srcShape.dims));
                owner.Ops.SoftmaxPack4Cdhw(srcTex.texture, srcShape.w, srcShape.h, srcShape.d, srcShape.c, outRt4);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt4, srcShape, srcShape);
            }
            else
            {
                var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.SoftmaxChannelPack4(srcTex.texture, srcTex.packs, outRt);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
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

                        do
                        {
                                                var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
                                                var axis = layer.GetInt(0, 0);
                                                if (axis != 0 || !CanUsePack4Softmax(src, srcShape))
                                                {
                                                    NcnnRepro.ResolveCmdTextureLayout(srcShape, out var width, out var height, out var packs);
                                                    owner.PublishCmdTensorLikeInput(cmd, layer.topNames[0], width, height, packs, blobs, shapes, srcShape);
                                                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                    continue;
                                                }
                                                var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                                                owner.Ops.SoftmaxChannelPack4(cmd, src.texture, src.packs, outArr);
                                                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                                                if (shapes != null)
                                                    shapes[layer.topNames[0]] = srcShape;
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }

        private static bool CanUsePack4Softmax(NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape srcShape)
        {
            return src != null
                && src.texture != null
                && (srcShape.dims == 3 || srcShape.dims == 4)
                && (srcShape.dims != 3 || srcShape.d == 1)
                && srcShape.w == src.width
                && srcShape.h == src.height
                && srcShape.c > 0
                && srcShape.c <= src.packs * 4;
        }

        private static bool CanUsePack4Softmax(NcnnRepro.TensorRef src, NcnnRepro.BufferShape srcShape)
        {
            return src != null
                && src.texture != null
                && (srcShape.dims == 3 || srcShape.dims == 4)
                && (srcShape.dims != 3 || srcShape.d == 1)
                && srcShape.w == src.width
                && srcShape.h == src.height
                && srcShape.c > 0
                && srcShape.c <= src.packs * 4;
        }

        private static bool TryResolvePack4SoftmaxWidthAxis(NcnnParamModel.Layer layer, NcnnRepro.BufferShape srcShape, out int tensorAxis)
        {
            tensorAxis = -1;
            if (layer == null)
                return false;

            var axis = layer.GetInt(0, 0);
            if (axis < 0)
                axis += srcShape.dims;
            if (axis < 0 || axis >= srcShape.dims)
                return false;

            tensorAxis = NcnnRepro.MapNcnnAxisToTensorAxis(srcShape.dims, axis);
            return tensorAxis == 0;
        }

        private static bool TryResolveContiguousSoftmax2D(NcnnTensorBuffer tensor, int axis, out int rows, out int cols)
        {
            rows = 0;
            cols = 0;
            if (tensor == null)
                return false;

            if (axis < 0)
                axis += tensor.dims;

            switch (tensor.dims)
            {
                case 1 when axis == 0:
                    rows = 1;
                    cols = tensor.w;
                    return true;
                case 2 when axis == 1:
                    rows = tensor.h;
                    cols = tensor.w;
                    return true;
                case 3 when axis == 2:
                    rows = tensor.h * tensor.c;
                    cols = tensor.w;
                    return true;
                case 4 when axis == 3:
                    rows = tensor.h * tensor.d * tensor.c;
                    cols = tensor.w;
                    return true;
                default:
                    return false;
            }
        }
    }
}
