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
            if (NcnnRepro.TryGetExistingTexture(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.bottomNames[0],
                    out var softScalarTex,
                    out var softScalarShape)
                && TryResolvePack4SoftmaxWidthAxis(layer, softScalarShape, out _)
                && CanUseScalar2DSoftmax(softScalarTex, softScalarShape))
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

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
            var scalarTextureFormat = NcnnRepro.ResolveTensorTextureFormat(2);

            if (NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var scalarSrcTex, out var scalarSrcShape)
                && CanUseScalar2DSoftmax(scalarSrcTex, scalarSrcShape))
            {
                if (!TryResolvePack4SoftmaxWidthAxis(layer, scalarSrcShape, out var scalarAxis))
                    throw new NotSupportedException("Softmax axis is outside the texture-native rank: " + layer.name);

                if (NcnnRepro.IsStrictLinearMatTexture(scalarSrcTex))
                {
                    var storageShape = NcnnRepro.GetTextureStorageShape(scalarSrcTex, scalarSrcShape);
                    var outScalarRt = owner.RentTempMat(storageShape.w, storageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
                    owner.Ops.SoftmaxLinearMat2D(scalarSrcTex.texture, scalarSrcShape.w, scalarSrcShape.h, outScalarRt, scalarAxis);
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outScalarRt, scalarSrcShape, storageShape);
                }
                else
                {
                    var outScalarRt = owner.RentTempArray(scalarSrcTex.width, scalarSrcTex.height, scalarSrcTex.packs, scalarTextureFormat);
                    owner.Ops.SoftmaxPack4Cdhw(scalarSrcTex.texture, scalarSrcShape.w, scalarSrcShape.h, 1, 1, outScalarRt, scalarAxis);
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outScalarRt, scalarSrcShape);
                }
                owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
                return;
            }

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                || !CanUsePack4Softmax(srcTex, srcShape))
            {
                throw new InvalidOperationException("Softmax render-texture path requires supported pack4 input: " + layer.name);
            }

            if (!TryResolvePack4SoftmaxWidthAxis(layer, srcShape, out var tensorAxis))
                throw new NotSupportedException("Softmax axis is outside the texture-native rank: " + layer.name);

            var logicalDepth = srcShape.dims == 4 ? Mathf.Max(1, srcShape.d) : 1;
            var outSlices = logicalDepth * Mathf.Max(1, Mathf.CeilToInt(srcShape.c / 4f));
            var outRt4 = owner.RentTempArray(srcTex.width, srcTex.height, outSlices, NcnnRepro.ResolveTensorTextureFormat(srcShape.dims));
            owner.Ops.SoftmaxPack4Cdhw(srcTex.texture, srcShape.w, srcShape.h, srcShape.d, srcShape.c, outRt4, tensorAxis);
            NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt4, srcShape, srcShape);
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
            if (CanUseScalar2DSoftmax(src, srcShape))
            {
                if (!TryResolvePack4SoftmaxWidthAxis(layer, srcShape, out var scalarAxis))
                    throw new NotSupportedException("Softmax axis is outside the texture-native rank: " + layer.name);

                if (NcnnRepro.IsStrictLinearMatTexture(src))
                {
                    var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
                    var outRt = owner.RentTempMat(cmd, storageShape.w, storageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
                    owner.Ops.SoftmaxLinearMat2D(cmd, src.texture, srcShape.w, srcShape.h, outRt, scalarAxis);
                    blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(outRt, srcShape, storageShape, owned: true);
                }
                else
                {
                    var outRt = owner.RentTempArray(cmd, src.width, src.height, src.packs, NcnnRepro.ResolveTensorTextureFormat(2));
                    owner.Ops.SoftmaxPack4Cdhw(cmd, src.texture, srcShape.w, srcShape.h, 1, 1, outRt, scalarAxis);
                    blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                    {
                        texture = outRt,
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
                }
                if (shapes != null)
                    shapes[layer.topNames[0]] = srcShape;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            if (!CanUsePack4Softmax(src, srcShape) || !TryResolvePack4SoftmaxWidthAxis(layer, srcShape, out var tensorAxis))
                throw new NotSupportedException("CommandBuffer Softmax requires a Pack4 rank-3/rank-4 texture and a valid axis"
                    + " | layer=" + layer.name
                    + " | rejectedFallback=placeholder-or-buffer-materialization");

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(srcShape.c / 4f));
            var logicalDepth = srcShape.dims == 4 ? Mathf.Max(1, srcShape.d) : 1;
            var outSlices = logicalDepth * outPacks;
            var outPack4 = owner.RentTempArray(cmd, src.width, src.height, outSlices, NcnnRepro.ResolveTensorTextureFormat(srcShape.dims));
            owner.Ops.SoftmaxPack4Cdhw(cmd, src.texture, srcShape.w, srcShape.h, srcShape.d, srcShape.c, outPack4, tensorAxis);
            blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
            {
                texture = outPack4,
                width = src.width,
                height = src.height,
                packs = outPacks,
                refs = 1,
                owned = true,
                hasLogicalShape = true,
                logicalShape = srcShape,
                hasStorageShape = true,
                storageShape = srcShape
            };
            if (shapes != null)
                shapes[layer.topNames[0]] = srcShape;

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool CanUseScalar2DSoftmax(NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape srcShape)
        {
            return src != null
                && src.texture != null
                && srcShape.dims == 2
                && srcShape.w > 0
                && srcShape.h > 0
                && srcShape.w == src.width
                && srcShape.h == src.height
                && src.packs == 1;
        }

        private static bool CanUseScalar2DSoftmax(NcnnRepro.TensorRef src, NcnnRepro.BufferShape srcShape)
        {
            return src != null
                && src.texture != null
                && srcShape.dims == 2
                && srcShape.w > 0
                && srcShape.h > 0
                && srcShape.w == src.width
                && srcShape.h == src.height
                && src.packs == 1;
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
            return tensorAxis >= 0 && tensorAxis < srcShape.dims;
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
