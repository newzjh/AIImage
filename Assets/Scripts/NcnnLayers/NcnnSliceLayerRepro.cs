using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnSliceLayerRepro : NcnnBaseLayerRepro
    {
        private readonly struct SliceSpec
        {
            public readonly int axis;
            public readonly int begin;
            public readonly NcnnRepro.BufferShape shape;

            public SliceSpec(int axis, int begin, NcnnRepro.BufferShape shape)
            {
                this.axis = axis;
                this.begin = begin;
                this.shape = shape;
            }
        }

        public NcnnSliceLayerRepro() : base(NcnnLayerTypes.Slice, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var srcTex,
                    out var srcShape)
                && CanUsePack4Slice(srcTex, srcShape))
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

            NcnnRepro.BufferShape srcShape;
            if (bufferViews.TryGetValue(layer.bottomNames[0], out var existingView) && existingView != null)
                srcShape = NcnnRepro.GetShapeOf(existingView);
            else if (textureBlobs.TryGetValue(layer.bottomNames[0], out var existingTex) && existingTex != null && existingTex.texture != null)
                srcShape = NcnnRepro.GetTextureShape(textureShapes, existingTex, layer.bottomNames[0]);
            else
            {
                var srcBufTmp = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                var srcViewTmp = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                if (srcBufTmp == null || srcViewTmp == null)
                    throw new InvalidOperationException("Slice source not found: " + layer.name);
                srcShape = NcnnRepro.GetShapeOf(srcViewTmp);
            }

            var specs = ResolveSliceSpecs(layer, srcShape);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Slice source not found: " + layer.name);

            for (var i = 0; i < layer.topNames.Length; i++)
            {
                var spec = specs[i];
                var outTensor = owner.RentTempTensorBuffer(spec.shape.dims, spec.shape.w, spec.shape.h, spec.shape.d, spec.shape.c);
                if (srcView.dims == 1)
                {
                    owner.Ops.CopyBufPartial(srcBuf, spec.begin, outTensor.buffer, spec.shape.w);
                }
                else
                {
                    owner.Ops.Slice(srcBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, spec.axis, spec.begin, spec.shape.w, spec.shape.h, spec.shape.d, spec.shape.c, outTensor.buffer);
                }

                owner.PublishTensorBufferOutput(
                    layer.topNames[i],
                    outTensor,
                    preferTexture: srcView.dims <= 3,
                    textureBlobs,
                    textureShapes,
                    bufferBlobs,
                    bufferRefs,
                    bufferViews,
                    tempOwned);
            }

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var texShape)
                || !CanUsePack4Slice(srcTex, texShape))
            {
                throw new InvalidOperationException("Slice render-texture path requires supported pack4 input: " + layer.name);
            }

            var specs = ResolveSliceSpecs(layer, texShape);
            for (var i = 0; i < layer.topNames.Length; i++)
            {
                var spec = specs[i];
                if (IsIdentitySlice(texShape, spec))
                {
                    textureBlobs[layer.topNames[i]] = srcTex;
                    textureShapes[layer.topNames[i]] = texShape;
                    srcTex.refs++;
                    continue;
                }

                var outPacks = Mathf.Max(1, Mathf.CeilToInt(spec.shape.c / 4f));
                if (texShape.dims == 4)
                {
                    var outSlices = Mathf.Max(1, spec.shape.d) * outPacks;
                    var outRt = owner.RentTempArray(spec.shape.w, spec.shape.h, outSlices, NcnnRepro.ResolveTensorTextureFormat(spec.shape.dims));
                    owner.Ops.SlicePack4Cdhw(
                        srcTex.texture,
                        texShape.w,
                        texShape.h,
                        texShape.d,
                        texShape.c,
                        spec.axis,
                        spec.begin,
                        spec.shape.w,
                        spec.shape.h,
                        spec.shape.d,
                        spec.shape.c,
                        outRt);
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[i], outRt, spec.shape, spec.shape);
                }
                else
                {
                    var outRt = owner.RentTempArray(spec.shape.w, spec.shape.h, outPacks, RenderTextureFormat.ARGBHalf);
                    owner.Ops.SlicePack4(srcTex.texture, texShape.w, texShape.h, texShape.c, spec.axis, spec.begin, spec.shape.w, spec.shape.h, spec.shape.c, outRt);
                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[i], outRt, spec.shape);
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
            var specs = ResolveSliceSpecs(layer, srcShape);
            var canUsePack4 = CanUsePack4Slice(src, srcShape);

            for (var i = 0; i < layer.topNames.Length; i++)
            {
                var spec = specs[i];
                if (canUsePack4)
                {
                    if (IsIdentitySlice(srcShape, spec))
                    {
                        blobs[layer.topNames[i]] = src;
                        src.refs++;
                        shapes[layer.topNames[i]] = srcShape;
                        continue;
                    }

                    var outPacks = Mathf.Max(1, Mathf.CeilToInt(spec.shape.c / 4f));
                    var outDepth = srcShape.dims == 4 ? Mathf.Max(1, spec.shape.d) * outPacks : outPacks;
                    var outFormat = srcShape.dims == 4 ? NcnnRepro.ResolveTensorTextureFormat(spec.shape.dims) : RenderTextureFormat.ARGBHalf;
                    var outArr = owner.RentTempArray(cmd, spec.shape.w, spec.shape.h, outDepth, outFormat);
                    if (srcShape.dims == 4)
                    {
                        owner.Ops.SlicePack4Cdhw(
                            cmd,
                            src.texture,
                            srcShape.w,
                            srcShape.h,
                            srcShape.d,
                            srcShape.c,
                            spec.axis,
                            spec.begin,
                            spec.shape.w,
                            spec.shape.h,
                            spec.shape.d,
                            spec.shape.c,
                            outArr);
                    }
                    else
                    {
                        owner.Ops.SlicePack4(cmd, src.texture, srcShape.w, srcShape.h, srcShape.c, spec.axis, spec.begin, spec.shape.w, spec.shape.h, spec.shape.c, outArr);
                    }
                    blobs[layer.topNames[i]] = new NcnnRepro.CmdTensorRef
                    {
                        texture = outArr,
                        width = spec.shape.w,
                        height = spec.shape.h,
                        packs = outPacks,
                        refs = 1,
                        owned = true,
                        hasLogicalShape = true,
                        logicalShape = spec.shape,
                        hasStorageShape = true,
                        storageShape = spec.shape
                    };
                    shapes[layer.topNames[i]] = spec.shape;
                    continue;
                }

                owner.DebugLog?.Invoke(
                    "[CmdPlaceholder][Slice]"
                    + " | layer=" + layer.name
                    + " | top=" + layer.topNames[i]
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | out=d" + spec.shape.dims + ":" + spec.shape.w + "x" + spec.shape.h + "x" + spec.shape.d + "x" + spec.shape.c
                    + " | axis=" + spec.axis
                    + " | begin=" + spec.begin);
                NcnnRepro.ResolveCmdTextureLayout(spec.shape, out var width, out var height, out var packs);
                owner.PublishCmdTensorLikeInput(cmd, layer.topNames[i], width, height, packs, blobs, shapes, spec.shape);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static SliceSpec[] ResolveSliceSpecs(NcnnParamModel.Layer layer, NcnnRepro.BufferShape srcShape)
        {
            var sliceParams = layer.GetInts(-23300, null);
            var indices = layer.GetInts(-23302, null);
            var ncnnAxis = layer.GetInt(1, 0);
            if (ncnnAxis < 0)
                ncnnAxis += srcShape.dims;
            var axis = NcnnRepro.MapNcnnAxisToTensorAxis(srcShape.dims, ncnnAxis);
            var axisSize = NcnnRepro.GetAxisSize(srcShape.dims, srcShape.w, srcShape.h, srcShape.d, srcShape.c, axis);

            var specs = new SliceSpec[layer.topNames.Length];
            var begin = 0;
            for (var i = 0; i < layer.topNames.Length; i++)
            {
                int sliceSize;
                if (indices != null && indices.Length > 0)
                {
                    if (i == layer.topNames.Length - 1)
                    {
                        sliceSize = axisSize - begin;
                    }
                    else
                    {
                        var indice = indices[Mathf.Min(i, indices.Length - 1)];
                        if (indice < 0)
                            indice += axisSize;
                        sliceSize = indice - begin;
                    }
                }
                else
                {
                    if (sliceParams == null || sliceParams.Length == 0)
                        throw new InvalidOperationException("Slice missing params: " + layer.name);
                    sliceSize = sliceParams[Mathf.Min(i, sliceParams.Length - 1)];
                    if (sliceSize == -233)
                        sliceSize = (axisSize - begin) / Mathf.Max(1, layer.topNames.Length - i);
                }

                if (sliceSize <= 0)
                    throw new InvalidOperationException("Slice produced empty output: " + layer.name + " top=" + i);

                var outW = srcShape.w;
                var outH = srcShape.h;
                var outD = srcShape.d;
                var outC = srcShape.c;
                if (axis == 0) outW = sliceSize;
                else if (axis == 1) outH = sliceSize;
                else if (axis == 2 && srcShape.dims == 4) outD = sliceSize;
                else if (axis == 2 || axis == 3) outC = sliceSize;

                specs[i] = new SliceSpec(axis, begin, new NcnnRepro.BufferShape(srcShape.dims, outW, outH, outD, outC));
                begin += sliceSize;
            }

            return specs;
        }

        private static bool CanUsePack4Slice(NcnnRepro.TensorRef srcTex, NcnnRepro.BufferShape srcShape)
        {
            return srcTex != null
                && srcTex.texture != null
                && srcShape.dims >= 1
                && srcShape.dims <= 4
                && srcShape.w == srcTex.width
                && srcShape.h == srcTex.height
                && (srcShape.dims != 4 || srcShape.d > 0);
        }

        private static bool CanUsePack4Slice(NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape srcShape)
        {
            return src != null
                && src.texture != null
                && srcShape.dims >= 1
                && srcShape.dims <= 4
                && srcShape.w == src.width
                && srcShape.h == src.height
                && (srcShape.dims != 4 || srcShape.d > 0);
        }

        private static bool IsIdentitySlice(NcnnRepro.BufferShape srcShape, SliceSpec spec)
        {
            return spec.begin == 0
                && spec.shape.w == srcShape.w
                && spec.shape.h == srcShape.h
                && spec.shape.d == srcShape.d
                && spec.shape.c == srcShape.c;
        }
    }
}
