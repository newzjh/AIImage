using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisSliceLayer : AexisBaseLayer
    {
        private readonly struct SliceSpec
        {
            public readonly int axis;
            public readonly int begin;
            public readonly AexisGraphSession.BufferShape shape;

            public SliceSpec(int axis, int begin, AexisGraphSession.BufferShape shape)
            {
                this.axis = axis;
                this.begin = begin;
                this.shape = shape;
            }
        }

        public AexisSliceLayer() : base(AexisLayerTypes.Slice, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var existingTex, out var existingShape))
            {
                var specs = ResolveSliceSpecs(layer, existingShape);
                if (CanUseLinearMatSlice(existingTex, existingShape, specs)
                    || CanUsePack4LinearMatSlice(existingTex, existingShape, specs)
                    || CanUsePack4Slice(existingTex, existingShape))
                {
                    ExecuteRenderTexturePath(owner, layer, context);
                    return;
                }
            }
            else if (owner.TryGetPack4Texture(
                         layer.bottomNames[0],
                         context.textureBlobs,
                         context.textureShapes,
                         context.bufferBlobs,
                         context.bufferViews,
                         out var srcTex,
                         out var srcShape)
                     && (CanUseLinearMatSliceSource(srcTex, srcShape) || CanUsePack4Slice(srcTex, srcShape)))
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

            AexisGraphSession.BufferShape srcShape;
            if (bufferViews.TryGetValue(layer.bottomNames[0], out var existingView) && existingView != null)
                srcShape = AexisGraphSession.GetShapeOf(existingView);
            else if (textureBlobs.TryGetValue(layer.bottomNames[0], out var existingTex) && existingTex != null && existingTex.texture != null)
                srcShape = AexisGraphSession.GetTextureShape(textureShapes, existingTex, layer.bottomNames[0]);
            else
            {
                var srcBufTmp = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                var srcViewTmp = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                if (srcBufTmp == null || srcViewTmp == null)
                    throw new InvalidOperationException("Slice source not found: " + layer.name);
                srcShape = AexisGraphSession.GetShapeOf(srcViewTmp);
            }

            var specs = ResolveSliceSpecs(layer, srcShape);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
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

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            if (!AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var srcTex, out var texShape)
                && !owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out srcTex, out texShape))
            {
                throw new InvalidOperationException("Slice render-texture path requires existing texture input: " + layer.name);
            }

            var specs = ResolveSliceSpecs(layer, texShape);
            var canUseLinearMat = CanUseLinearMatSlice(srcTex, texShape, specs);
            var canUsePack4LinearMat = CanUsePack4LinearMatSlice(srcTex, texShape, specs);
            var canUsePack4 = CanUsePack4Slice(srcTex, texShape);
            if (!canUseLinearMat && !canUsePack4LinearMat && !canUsePack4)
                throw new InvalidOperationException("Slice render-texture path requires supported LinearMat or pack4 input: " + layer.name);

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

                if (canUseLinearMat)
                {
                    var storageShape = AexisGraphSession.ResolveLinearMatStorageShape(spec.shape);
                    var outMat = owner.RentTempMat(storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                    owner.Ops.SliceLinearMat2D(srcTex.texture, spec.axis, spec.begin, outMat);
                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[i], outMat, spec.shape, storageShape);
                }
                else if (canUsePack4LinearMat)
                {
                    var srcStorageShape = AexisGraphSession.GetTextureStorageShape(srcTex, texShape);
                    var outStorageShape = AexisGraphSession.ResolvePack4LinearMatStorageShape(spec.shape);
                    var outRt = owner.RentTempArray(outStorageShape.w, outStorageShape.h, 1, srcTex.texture.format);
                    if (!TryCopyPack4LinearMatSlice(srcTex.texture, srcStorageShape, spec, outRt))
                    {
                        var sliceAxis = spec.axis == 0 ? 0 : 1;
                        var sliceBegin = spec.axis == 0 ? spec.begin / 4 : spec.begin;
                        owner.Ops.SlicePack4(srcTex.texture, srcStorageShape.w, srcStorageShape.h, 4, sliceAxis, sliceBegin, outStorageShape.w, outStorageShape.h, 4, outRt);
                    }
                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[i], outRt, spec.shape, outStorageShape);
                }
                else if (texShape.dims == 4)
                {
                    var outPacks = Mathf.Max(1, Mathf.CeilToInt(spec.shape.c / 4f));
                    var outSlices = Mathf.Max(1, spec.shape.d) * outPacks;
                    var outRt = owner.RentTempArray(spec.shape.w, spec.shape.h, outSlices, AexisGraphSession.ResolveTensorTextureFormat(spec.shape.dims));
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
                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[i], outRt, spec.shape, spec.shape);
                }
                else
                {
                    var outPacks = Mathf.Max(1, Mathf.CeilToInt(spec.shape.c / 4f));
                    var outRt = owner.RentTempArray(spec.shape.w, spec.shape.h, outPacks, RenderTextureFormat.ARGBHalf);
                    owner.Ops.SlicePack4(srcTex.texture, texShape.w, texShape.h, texShape.c, spec.axis, spec.begin, spec.shape.w, spec.shape.h, spec.shape.c, outRt);
                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[i], outRt, spec.shape);
                }
            }

            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var sourceContract = AexisGraphSession.GetCmdTensorContract(src);
            var srcShape = sourceContract.LogicalShape;
            var specs = ResolveSliceSpecs(layer, srcShape);
            var canUseLinearMat = CanUseLinearMatSlice(src, srcShape, specs);
            var canUsePack4LinearMat = CanUsePack4LinearMatSlice(src, srcShape, specs);
            var canUsePack4 = CanUsePack4Slice(src, srcShape);

            for (var i = 0; i < layer.topNames.Length; i++)
            {
                var spec = specs[i];
                if (canUseLinearMat || canUsePack4LinearMat || canUsePack4)
                {
                    if (IsIdentitySlice(srcShape, spec))
                    {
                        blobs[layer.topNames[i]] = AexisGraphSession.CreateCmdTensorAlias(src, spec.shape, sourceContract.StorageShape);
                        if (shapes != null)
                            shapes[layer.topNames[i]] = spec.shape;
                        continue;
                    }

                    if (canUseLinearMat)
                    {
                        var storageShape = AexisGraphSession.ResolveLinearMatStorageShape(spec.shape);
                        var outMat = owner.RentTempMat(
                            cmd,
                            storageShape.w,
                            storageShape.h,
                            AexisGraphSession.ResolveLinearMatTextureFormat(),
                            layer.topNames[i]);
                        owner.Ops.SliceLinearMat2D(cmd, src.texture, spec.axis, spec.begin, outMat);
                        blobs[layer.topNames[i]] = AexisGraphSession.CreateCmdTensorRef(outMat, spec.shape, storageShape, owned: true);
                        shapes[layer.topNames[i]] = spec.shape;
                        continue;
                    }

                    if (canUsePack4LinearMat)
                    {
                        var srcStorageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                        var outStorageShape = AexisGraphSession.ResolvePack4LinearMatStorageShape(spec.shape);
                        var pack4LinearOut = owner.RentTempArray(
                            cmd,
                            outStorageShape.w,
                            outStorageShape.h,
                            1,
                            src.texture.format,
                            layer.topNames[i]);
                        if (!TryCopyPack4LinearMatSlice(cmd, src.texture, srcStorageShape, spec, pack4LinearOut))
                        {
                            var sliceAxis = spec.axis == 0 ? 0 : 1;
                            var sliceBegin = spec.axis == 0 ? spec.begin / 4 : spec.begin;
                            owner.Ops.SlicePack4(cmd, src.texture, srcStorageShape.w, srcStorageShape.h, 4, sliceAxis, sliceBegin, outStorageShape.w, outStorageShape.h, 4, pack4LinearOut);
                        }
                        blobs[layer.topNames[i]] = AexisGraphSession.CreateCmdTensorRef(pack4LinearOut, spec.shape, outStorageShape, owned: true);
                        shapes[layer.topNames[i]] = spec.shape;
                        continue;
                    }

                    var outPacks = Mathf.Max(1, Mathf.CeilToInt(spec.shape.c / 4f));
                    var outDepth = srcShape.dims == 4 ? Mathf.Max(1, spec.shape.d) * outPacks : outPacks;
                    var outFormat = srcShape.dims == 4 ? AexisGraphSession.ResolveTensorTextureFormat(spec.shape.dims) : RenderTextureFormat.ARGBHalf;
                    var outArr = owner.RentTempArray(
                        cmd,
                        spec.shape.w,
                        spec.shape.h,
                        outDepth,
                        outFormat,
                        layer.topNames[i]);
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
                    var directStorageShape = spec.shape.dims <= 2
                        ? new AexisGraphSession.BufferShape(3, spec.shape.w, spec.shape.dims == 2 ? spec.shape.h : 1, 1, 1)
                        : spec.shape;
                    blobs[layer.topNames[i]] = AexisGraphSession.CreateCmdTensorRef(outArr, spec.shape, directStorageShape, owned: true, blobName: layer.topNames[i]);
                    if (shapes != null)
                        shapes[layer.topNames[i]] = spec.shape;
                    continue;
                }

                throw new InvalidOperationException(
                    "Slice command-buffer Pack4 profile is unsupported; placeholder publication is prohibited"
                    + " | layer=" + layer.name
                    + " | top=" + layer.topNames[i]
                    + " | logical=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | storage=d" + sourceContract.StorageShape.dims + ":" + sourceContract.StorageShape.w + "x" + sourceContract.StorageShape.h + "x" + sourceContract.StorageShape.d + "x" + sourceContract.StorageShape.c
                    + " | layout=" + sourceContract.LayoutKind
                    + " | axis=" + spec.axis
                    + " | begin=" + spec.begin);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static SliceSpec[] ResolveSliceSpecs(AexisGraphModel.Layer layer, AexisGraphSession.BufferShape srcShape)
        {
            var sliceParams = layer.GetInts(-23300, null);
            var indices = layer.GetInts(-23302, null);
            var ncnnAxis = layer.GetInt(1, 0);
            if (ncnnAxis < 0)
                ncnnAxis += srcShape.dims;
            var axis = AexisGraphSession.MapNcnnAxisToTensorAxis(srcShape.dims, ncnnAxis);
            var axisSize = AexisGraphSession.GetAxisSize(srcShape.dims, srcShape.w, srcShape.h, srcShape.d, srcShape.c, axis);

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

                specs[i] = new SliceSpec(axis, begin, new AexisGraphSession.BufferShape(srcShape.dims, outW, outH, outD, outC));
                begin += sliceSize;
            }

            return specs;
        }

        private static bool CanUsePack4Slice(AexisGraphSession.TensorRef srcTex, AexisGraphSession.BufferShape srcShape)
        {
            return srcTex != null
                && srcTex.texture != null
                && !AexisGraphSession.IsStrictLinearMatTexture(srcTex)
                && srcShape.dims >= 1
                && srcShape.dims <= 4
                && srcShape.w == srcTex.width
                && srcShape.h == srcTex.height
                && (srcShape.dims != 4 || srcShape.d > 0);
        }

        private static bool CanUsePack4Slice(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape srcShape)
        {
            return src != null
                && src.texture != null
                && !AexisGraphSession.IsStrictLinearMatTexture(src)
                && srcShape.dims >= 1
                && srcShape.dims <= 4
                && srcShape.w == src.width
                && srcShape.h == src.height
                && (srcShape.dims != 4 || srcShape.d > 0);
        }

        private static bool CanUseLinearMatSliceSource(AexisGraphSession.TensorRef srcTex, AexisGraphSession.BufferShape srcShape)
        {
            if (srcTex == null || srcTex.texture == null || !AexisGraphSession.IsStrictLinearMatTexture(srcTex))
                return false;

            var storageShape = AexisGraphSession.GetTextureStorageShape(srcTex, srcShape);
            return (srcShape.dims == 1 || srcShape.dims == 2)
                && storageShape.dims == 2
                && storageShape.w == srcTex.width
                && storageShape.h == srcTex.height
                && srcTex.packs == 1;
        }

        private static bool CanUseLinearMatSliceSource(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape srcShape)
        {
            if (src == null || src.texture == null || !AexisGraphSession.IsStrictLinearMatTexture(src))
                return false;

            var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
            return (srcShape.dims == 1 || srcShape.dims == 2)
                && storageShape.dims == 2
                && storageShape.w == src.width
                && storageShape.h == src.height
                && src.packs == 1;
        }

        private static bool CanUseLinearMatSlice(AexisGraphSession.TensorRef srcTex, AexisGraphSession.BufferShape srcShape, SliceSpec[] specs)
        {
            if (!CanUseLinearMatSliceSource(srcTex, srcShape) || specs == null || specs.Length == 0)
                return false;
            for (var i = 0; i < specs.Length; i++)
            {
                var spec = specs[i];
                if (spec.shape.dims != srcShape.dims
                    || spec.axis < 0
                    || spec.axis > 1
                    || (srcShape.dims == 1 && spec.axis != 0))
                    return false;
            }
            return true;
        }

        private static bool CanUseLinearMatSlice(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape srcShape, SliceSpec[] specs)
        {
            if (!CanUseLinearMatSliceSource(src, srcShape) || specs == null || specs.Length == 0)
                return false;
            for (var i = 0; i < specs.Length; i++)
            {
                var spec = specs[i];
                if (spec.shape.dims != srcShape.dims
                    || spec.axis < 0
                    || spec.axis > 1
                    || (srcShape.dims == 1 && spec.axis != 0))
                    return false;
            }
            return true;
        }

        private static bool CanUsePack4LinearMatSlice(AexisGraphSession.TensorRef srcTex, AexisGraphSession.BufferShape srcShape, SliceSpec[] specs)
        {
            if (!AexisGraphSession.IsPack4LinearMatTexture(srcTex, srcShape) || specs == null || specs.Length == 0)
                return false;
            for (var i = 0; i < specs.Length; i++)
            {
                var spec = specs[i];
                if (spec.shape.dims != 2)
                    return false;
                if (spec.axis == 0)
                {
                    if (spec.begin % 4 != 0 || spec.shape.w % 4 != 0)
                        return false;
                }
                else if (spec.axis != 1)
                {
                    return false;
                }
            }
            return true;
        }

        private static bool CanUsePack4LinearMatSlice(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape srcShape, SliceSpec[] specs)
        {
            if (!AexisGraphSession.IsPack4LinearMatTexture(src, srcShape) || specs == null || specs.Length == 0)
                return false;
            for (var i = 0; i < specs.Length; i++)
            {
                var spec = specs[i];
                if (spec.shape.dims != 2)
                    return false;
                if (spec.axis == 0)
                {
                    if (spec.begin % 4 != 0 || spec.shape.w % 4 != 0)
                        return false;
                }
                else if (spec.axis != 1)
                {
                    return false;
                }
            }
            return true;
        }

        private static bool TryCopyPack4LinearMatSlice(RenderTexture input, AexisGraphSession.BufferShape storageShape, SliceSpec spec, RenderTexture output)
        {
            if (input == null || output == null || spec.shape.dims != 2 || (spec.axis != 0 && spec.axis != 1))
                return false;

            var srcX = spec.axis == 0 ? spec.begin / 4 : 0;
            var srcY = spec.axis == 1 ? spec.begin : 0;
            var copyW = output.width;
            var copyH = output.height;
            if (!CanCopyPack4LinearMatSlice(storageShape, srcX, srcY, copyW, copyH))
                return false;

            var depth = Mathf.Max(1, output.volumeDepth);
            for (var slice = 0; slice < depth; slice++)
                Graphics.CopyTexture(input, slice, 0, srcX, srcY, copyW, copyH, output, slice, 0, 0, 0);
            return true;
        }

        private static bool TryCopyPack4LinearMatSlice(CommandBuffer cmd, ComputeTexture input, AexisGraphSession.BufferShape storageShape, SliceSpec spec, ComputeTexture output)
        {
            if (cmd == null || input == null || output == null || spec.shape.dims != 2 || (spec.axis != 0 && spec.axis != 1))
                return false;

            var srcX = spec.axis == 0 ? spec.begin / 4 : 0;
            var srcY = spec.axis == 1 ? spec.begin : 0;
            var copyW = output.width;
            var copyH = output.height;
            if (!CanCopyPack4LinearMatSlice(storageShape, srcX, srcY, copyW, copyH))
                return false;

            var depth = Mathf.Max(1, output.depth);
            for (var slice = 0; slice < depth; slice++)
                cmd.CopyTexture(input.nameID, slice, 0, srcX, srcY, copyW, copyH, output.nameID, slice, 0, 0, 0);
            return true;
        }

        private static bool CanCopyPack4LinearMatSlice(AexisGraphSession.BufferShape storageShape, int srcX, int srcY, int copyW, int copyH)
        {
            return storageShape.dims == 2
                && srcX >= 0
                && srcY >= 0
                && copyW > 0
                && copyH > 0
                && srcX + copyW <= storageShape.w
                && srcY + copyH <= storageShape.h;
        }

        private static bool IsIdentitySlice(AexisGraphSession.BufferShape srcShape, SliceSpec spec)
        {
            return spec.begin == 0
                && spec.shape.w == srcShape.w
                && spec.shape.h == srcShape.h
                && spec.shape.d == srcShape.d
                && spec.shape.c == srcShape.c;
        }
    }
}
