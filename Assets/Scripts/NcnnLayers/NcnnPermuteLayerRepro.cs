using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnPermuteLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnPermuteLayerRepro() : base(NcnnLayerTypes.Permute, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var orderType = layer.GetInt(0, 0);
            if (owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var srcTex,
                    out var srcShape)
                && CanUsePack4Permute(srcTex, srcShape, orderType, out _, out _))
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

            var orderType = layer.GetInt(0, 0);
            var srcTensor = owner.GetReadableTensorInput(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            if (srcTensor == null || srcTensor.buffer == null)
                throw new InvalidOperationException("Permute source not found: " + layer.bottomNames[0]);

            var dims = Mathf.Clamp(srcTensor.dims, 2, 4);
            var axes = NcnnRepro.ResolvePermuteAxes(dims, orderType, layer.name);
            var outShape = NcnnRepro.ResolvePermuteShape(srcTensor, dims, axes);
            var outTensor = owner.RentTempTensorBuffer(outShape.dims, outShape.w, outShape.h, outShape.d, outShape.c);
            owner.Ops.Permute(srcTensor.buffer, dims, srcTensor.w, srcTensor.h, srcTensor.d, srcTensor.c, orderType, outTensor.buffer);

            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: outShape.dims <= 3,
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
            var orderType = layer.GetInt(0, 0);

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                || !CanUsePack4Permute(srcTex, srcShape, orderType, out var axes, out var outShape))
            {
                throw new InvalidOperationException("Permute render-texture path requires supported pack4 input: " + layer.name);
            }

            if (orderType == 0)
            {
                textureBlobs[layer.topNames[0]] = srcTex;
                textureShapes[layer.topNames[0]] = srcShape;
                srcTex.refs++;
            }
            else
            {
                if (srcShape.dims == 4)
                {
                    var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
                    var outSlices = Mathf.Max(1, outShape.d) * outPacks;
                    var outRt = owner.RentTempArray(outShape.w, outShape.h, outSlices, NcnnRepro.ResolveTensorTextureFormat(outShape.dims));
                    owner.Ops.PermutePack4Cdhw(
                        srcTex.texture,
                        srcShape.w,
                        srcShape.h,
                        srcShape.d,
                        srcShape.c,
                        axes,
                        outShape.w,
                        outShape.h,
                        outShape.d,
                        outShape.c,
                        outRt);
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape);
                }
                else
                {
                    var outChannels = srcShape.dims == 2 ? 1 : outShape.c;
                    var outPacks = Mathf.Max(1, Mathf.CeilToInt(outChannels / 4f));
                    var outRt = owner.RentTempArray(outShape.w, outShape.h, outPacks, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PermutePack4(srcTex.texture, srcShape.w, srcShape.h, srcShape.dims == 2 ? 1 : srcShape.c, axes, outShape.w, outShape.h, outChannels, outRt);
                    if (srcShape.dims == 2)
                    {
                        NcnnRepro.SetTextureBlob(
                            textureBlobs,
                            textureShapes,
                            layer.topNames[0],
                            outRt,
                            outShape,
                            new NcnnRepro.BufferShape(3, outShape.w, outShape.h, 1, 1));
                    }
                    else
                    {
                        NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape);
                    }
                }
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
            var orderType = layer.GetInt(0, 0);
            if (CanUsePack4Permute(src, srcShape, orderType, out var axes, out var outShape))
            {
                if (orderType == 0)
                {
                    blobs[layer.topNames[0]] = src;
                    src.refs++;
                    shapes[layer.topNames[0]] = srcShape;
                }
                else
                {
                    var outChannels = srcShape.dims == 2 ? 1 : outShape.c;
                    var outPacks = Mathf.Max(1, Mathf.CeilToInt(outChannels / 4f));
                    var outDepth = srcShape.dims == 4 ? Mathf.Max(1, outShape.d) * outPacks : outPacks;
                    var outFormat = srcShape.dims == 4 ? NcnnRepro.ResolveTensorTextureFormat(outShape.dims) : RenderTextureFormat.ARGBHalf;
                    var outArr = owner.RentTempArray(cmd, outShape.w, outShape.h, outDepth, outFormat);
                    if (srcShape.dims == 4)
                    {
                        owner.Ops.PermutePack4Cdhw(
                            cmd,
                            src.texture,
                            srcShape.w,
                            srcShape.h,
                            srcShape.d,
                            srcShape.c,
                            axes,
                            outShape.w,
                            outShape.h,
                            outShape.d,
                            outShape.c,
                            outArr);
                    }
                    else
                    {
                        owner.Ops.PermutePack4(cmd, src.texture, srcShape.w, srcShape.h, srcShape.dims == 2 ? 1 : srcShape.c, axes, outShape.w, outShape.h, outChannels, outArr);
                    }
                    var storageShape = srcShape.dims == 2
                        ? new NcnnRepro.BufferShape(3, outShape.w, outShape.h, 1, 1)
                        : outShape;
                    blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                    {
                        texture = outArr,
                        width = outShape.w,
                        height = outShape.h,
                        packs = outPacks,
                        refs = 1,
                        owned = true,
                        hasLogicalShape = true,
                        logicalShape = outShape,
                        hasStorageShape = true,
                        storageShape = storageShape
                    };
                    shapes[layer.topNames[0]] = outShape;
                }
            }
            else
            {
                owner.DebugLog?.Invoke(
                    "[CmdPlaceholder][Permute]"
                    + " | layer=" + layer.name
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | orderType=" + orderType);
                NcnnRepro.ResolveCmdTextureLayout(srcShape, out var width, out var height, out var packs);
                owner.PublishCmdTensorLikeInput(cmd, layer.topNames[0], width, height, packs, blobs, shapes, srcShape);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool CanUsePack4Permute(NcnnRepro.TensorRef srcTex, NcnnRepro.BufferShape srcShape, int orderType, out Vector4Int axes, out NcnnRepro.BufferShape outShape)
        {
            axes = default;
            outShape = default;
            if (srcTex == null || srcTex.texture == null)
                return false;
            return CanUsePack4PermuteCore(srcShape, orderType, out axes, out outShape);
        }

        private static bool CanUsePack4Permute(NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape srcShape, int orderType, out Vector4Int axes, out NcnnRepro.BufferShape outShape)
        {
            axes = default;
            outShape = default;
            if (src == null || src.texture == null)
                return false;
            return CanUsePack4PermuteCore(srcShape, orderType, out axes, out outShape);
        }

        private static bool CanUsePack4PermuteCore(NcnnRepro.BufferShape srcShape, int orderType, out Vector4Int axes, out NcnnRepro.BufferShape outShape)
        {
            axes = default;
            outShape = default;
            if (srcShape.dims == 2)
            {
                if (orderType == 0)
                {
                    axes = new Vector4Int(0, 1, 2, 0);
                    outShape = srcShape;
                    return true;
                }

                if (orderType == 1)
                {
                    axes = new Vector4Int(1, 0, 2, 0);
                    outShape = new NcnnRepro.BufferShape(2, srcShape.h, srcShape.w, 1, 1);
                    return true;
                }

                return false;
            }

            if (srcShape.dims == 3 && srcShape.d == 1)
            {
                axes = NcnnRepro.ResolvePermuteAxes(3, orderType, "PermutePack4");
                outShape = NcnnRepro.ResolvePermuteShape(srcShape, 3, axes);
                return outShape.dims == 3 && outShape.d == 1;
            }

            if (srcShape.dims == 4)
            {
                axes = NcnnRepro.ResolvePermuteAxes(4, orderType, "PermutePack4CDHW");
                outShape = NcnnRepro.ResolvePermuteShape(srcShape, 4, axes);
                return outShape.dims == 4;
            }

            return false;
        }
    }
}
