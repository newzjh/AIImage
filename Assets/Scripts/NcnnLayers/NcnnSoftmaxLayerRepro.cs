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
            var axis = layer.GetInt(0, 0);
            if (axis == 0
                && owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var softSrcTex,
                    out var softSrcShape)
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
                var outTensor = owner.RentTempTensorBuffer(
                    softTensor.dims,
                    softTensor.w,
                    softTensor.h,
                    softTensor.d,
                    softTensor.c);
                if (softTensor.dims == 3)
                {
                    var batch = softTensor.c;
                    var rows = softTensor.h;
                    var cols = softTensor.w;
                    var matrixCount = rows * cols;
                    var sliceIn = owner.RentTempBuffer(matrixCount, sizeof(float));
                    var sliceOut = owner.RentTempBuffer(matrixCount, sizeof(float));
                    try
                    {
                        for (var p = 0; p < batch; p++)
                        {
                            var offset = p * matrixCount;
                            owner.Ops.CopyBufPartial(softBuf, offset, sliceIn, matrixCount);
                            owner.Ops.Softmax2D(sliceIn, sliceOut, rows, cols);
                            owner.Ops.CopyBufPartial(sliceOut, 0, outTensor.buffer, matrixCount, offset);
                        }
                    }
                    finally
                    {
                        owner.ReturnTempBuffer(sliceIn);
                        owner.ReturnTempBuffer(sliceOut);
                    }
                }
                else
                {
                    var rows = softTensor.dims == 2 ? softTensor.h : 1;
                    var cols = softTensor.dims == 2 ? softTensor.w : softBuf.count;
                    owner.Ops.Softmax2D(softBuf, outTensor.buffer, rows, cols);
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
            var axis = layer.GetInt(0, 0);
            if (axis != 0)
                throw new InvalidOperationException("Softmax render-texture path currently supports axis == 0 only: " + layer.name);

            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                || !CanUsePack4Softmax(srcTex, srcShape))
            {
                throw new InvalidOperationException("Softmax render-texture path requires supported pack4 input: " + layer.name);
            }

            var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.SoftmaxChannelPack4(srcTex.texture, srcTex.packs, outRt);
            NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
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
                && srcShape.dims == 3
                && srcShape.d == 1
                && srcShape.w == src.width
                && srcShape.h == src.height
                && srcShape.c > 0
                && srcShape.c <= src.packs * 4;
        }

        private static bool CanUsePack4Softmax(NcnnRepro.TensorRef src, NcnnRepro.BufferShape srcShape)
        {
            return src != null
                && src.texture != null
                && srcShape.dims == 3
                && srcShape.d == 1
                && srcShape.w == src.width
                && srcShape.h == src.height
                && srcShape.c > 0
                && srcShape.c <= src.packs * 4;
        }
    }
}
