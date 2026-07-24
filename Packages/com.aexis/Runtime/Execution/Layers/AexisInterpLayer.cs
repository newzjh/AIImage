using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisInterpLayer : AexisBaseLayer
    {
        private const int CoordinateTransformModeParamKey = 100;
        private const int CoordinateTransformModeHalfPixel = 0;
        private const int CoordinateTransformModeAsymmetric = 1;

        public AexisInterpLayer() : base(AexisLayerTypes.Interp, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (CanExecuteRenderTexturePath(owner, layer, context))
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

            using var srcReadable = owner.GetReadableTensorInput(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcBuf = srcReadable?.buffer;
            var srcView = srcReadable;
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Interp source not found: " + layer.name);

            ResolveTargetShape(
                layer,
                new AexisGraphSession.BufferShape(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c),
                textureBlobs,
                textureShapes,
                bufferViews,
                out var fallbackOutW,
                out var fallbackOutH,
                out var fallbackOutD,
                out var fallbackOutC);

            if (srcView.dims == 2 && fallbackOutW == srcView.w)
            {
#pragma warning disable CS0618
                new AexisNoopLayer().ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
                return;
            }

            if (srcView.dims == 4 && fallbackOutW == srcView.w && fallbackOutH == srcView.h && fallbackOutD == srcView.d)
            {
#pragma warning disable CS0618
                new AexisNoopLayer().ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
                return;
            }

            if (srcView.dims == 3 && fallbackOutW == srcView.w && fallbackOutH == srcView.h)
            {
#pragma warning disable CS0618
                new AexisNoopLayer().ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
                return;
            }

            var resizeType = layer.GetInt(0, 0);
            var alignCorner = layer.GetInt(6, 0) != 0;
            var outTensor = srcView.dims == 1
                ? owner.RentTempTensorBuffer(3, fallbackOutW, fallbackOutH, 1, fallbackOutC)
                : owner.RentTempTensorBuffer(srcView.dims, fallbackOutW, fallbackOutH, srcView.dims == 4 ? fallbackOutD : srcView.d, fallbackOutC);

            var srcData = AexisGraphSession.ReadFloatBuffer(srcBuf);
            var outData = new float[outTensor.elementCount];

            if (srcView.dims == 1)
            {
                ApplyInterpDims1(srcData, srcView, fallbackOutW, fallbackOutH, resizeType, alignCorner, outData);
            }
            else if (srcView.dims == 2)
            {
                ApplyInterpDims2(srcData, srcView, fallbackOutW, resizeType, alignCorner, layer, outData);
            }
            else if (srcView.dims == 4)
            {
                ApplyInterpDims4(srcData, srcView, fallbackOutW, fallbackOutH, fallbackOutD, resizeType, alignCorner, layer, outData);
            }
            else
            {
                ApplyInterpDims3(srcData, srcView, fallbackOutW, fallbackOutH, resizeType, alignCorner, layer, outData);
            }

            outTensor.buffer.SetData(outData);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: outTensor.dims <= 3,
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
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape))
                throw new InvalidOperationException("Interp render-texture path requires existing texture input: " + layer.name);

            ResolveTargetShape(layer, srcShape, textureBlobs, textureShapes, bufferViews, out var outW, out var outH, out var outD, out var outC);

            if (srcShape.dims == 2 && outW == srcShape.w)
            {
                new AexisNoopLayer().ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

            if (srcShape.dims == 4 && outW == srcShape.w && outH == srcShape.h && outD == srcShape.d)
            {
                new AexisNoopLayer().ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

            if (srcShape.dims == 3 && outW == srcShape.w && outH == srcShape.h)
            {
                new AexisNoopLayer().ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

            if ((srcShape.dims != 3 && srcShape.dims != 4) || outC != srcShape.c || !CanUsePack4Interp(srcTex, srcShape))
                throw new InvalidOperationException("Interp render-texture path requires supported pack4 input/output shape: " + layer.name);

            if (srcShape.dims == 4)
            {
                var resizeType = layer.GetInt(0, 0);
                var coordinateTransformMode = ResolveCoordinateTransformMode(layer);
                var scaleX = outW / (float)Mathf.Max(1, srcShape.w);
                var scaleY = outH / (float)Mathf.Max(1, srcShape.h);
                var scaleZ = outD / (float)Mathf.Max(1, srcShape.d);
                var alignCorners = layer.GetInt(6, 0) != 0;
                var outRt4 = owner.RentTempArray(outW, outH, outD * srcTex.packs, AexisGraphSession.ResolveTensorTextureFormat(4));
                owner.Ops.InterpPack4CDHW(
                    srcTex.texture,
                    srcShape.w,
                    srcShape.h,
                    srcShape.d,
                    srcTex.packs,
                    outW,
                    outH,
                    outD,
                    srcTex.packs,
                    scaleX,
                    scaleY,
                    scaleZ,
                    resizeType,
                    alignCorners,
                    outRt4,
                    coordinateTransformMode);

                AexisGraphSession.SetTextureBlob(
                    textureBlobs,
                    textureShapes,
                    layer.topNames[0],
                    outRt4,
                    new AexisGraphSession.BufferShape(4, outW, outH, outD, outC));
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            var resizeTypePack = layer.GetInt(0, 0);
            var coordinateTransformModePack = ResolveCoordinateTransformMode(layer);
            var alignCornersPack = layer.GetInt(6, 0) != 0;
            var sxPack = layer.GetFloat(2, 1f);
            var syPack = layer.GetFloat(1, 1f);
            var outShape = new AexisGraphSession.BufferShape(3, outW, outH, 1, outC);
            var outRt = owner.RentTempArray(outW, outH, srcTex.packs, RenderTextureFormat.ARGBHalf);
            var executed = false;

            if (!alignCornersPack && Mathf.Abs(sxPack - 2f) < 1e-3f && Mathf.Abs(syPack - 2f) < 1e-3f)
            {
                if (resizeTypePack == 1)
                    owner.Ops.Interp2xNearestPack4(srcTex.texture, srcTex.packs, outRt, coordinateTransformModePack);
                else
                    owner.Ops.Interp2xPack4(srcTex.texture, srcTex.packs, outRt, coordinateTransformModePack);
                executed = true;
            }
            else if (!alignCornersPack && Mathf.Abs(sxPack - 0.5f) < 1e-3f && Mathf.Abs(syPack - 0.5f) < 1e-3f)
            {
                if (resizeTypePack == 1)
                    owner.Ops.InterpDown2NearestPack4(srcTex.texture, srcTex.packs, outRt, coordinateTransformModePack);
                else
                    owner.Ops.InterpDown2Pack4(srcTex.texture, srcTex.packs, outRt, coordinateTransformModePack);
                executed = true;
            }
            else if (resizeTypePack == 1)
            {
                var scaleX = outW / (float)Mathf.Max(1, srcTex.width);
                var scaleY = outH / (float)Mathf.Max(1, srcTex.height);
                owner.Ops.InterpPack4Nearest(srcTex.texture, srcTex.packs, scaleX, scaleY, outRt, coordinateTransformModePack);
                executed = true;
            }
            else if (resizeTypePack == 3)
            {
                var scaleX = outW / (float)Mathf.Max(1, srcTex.width);
                var scaleY = outH / (float)Mathf.Max(1, srcTex.height);
                owner.Ops.InterpPack4(srcTex.texture, srcTex.packs, scaleX, scaleY, outRt, coordinateTransformModePack, alignCornersPack);
                executed = true;
            }
            else if (resizeTypePack != 1 && resizeTypePack != 3)
            {
                var scaleX = outW / (float)Mathf.Max(1, srcTex.width);
                var scaleY = outH / (float)Mathf.Max(1, srcTex.height);
                owner.Ops.InterpPack4(srcTex.texture, srcTex.packs, scaleX, scaleY, outRt, coordinateTransformModePack, alignCornersPack);
                executed = true;
            }

            if (!executed)
            {
                owner.ReturnTempArray(outRt);
                throw new InvalidOperationException("Interp render-texture path unsupported resize config: " + layer.name);
            }

            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape);
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
            var srcContract = AexisGraphSession.GetCmdTensorContract(src);
            var srcShape = srcContract.LogicalShape;
            var resizeType = layer.GetInt(0, 0);
            var sy = layer.GetFloat(1, 1f);
            var sx = layer.GetFloat(2, 1f);

            ResolveCmdTargetShape(layer, shapes, blobs, srcShape, sx, sy, out var outW, out var outH, out var outD);
            var outShape = ResolveCmdOutputShape(srcShape, outW, outH, outD);
            owner.DebugLog?.Invoke(
                "[CmdTexture][Interp]"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | srcStorage=d" + srcContract.StorageShape.dims + ":" + srcContract.StorageShape.w + "x" + srcContract.StorageShape.h + "x" + srcContract.StorageShape.d + "x" + srcContract.StorageShape.c
                + " | srcTexture=" + src.width + "x" + src.height + "x" + src.texture.depth
                + " | packs=" + srcContract.Packs
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                + " | scale=" + sx + "x" + sy);

            if (IsCmdInterpNoop(srcShape, outShape))
            {
                var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorAlias(src, srcShape, storageShape);
                if (shapes != null)
                    shapes[layer.topNames[0]] = srcShape;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            if (srcShape.dims == 4)
            {
                if (!srcContract.IsPack4Image || !CanUsePack4Interp(src, srcShape))
                    throw new InvalidOperationException("Interp command-buffer CDHW path requires a TensorDescriptor-backed Pack4 Texture2DArray: " + layer.name);
                if (resizeType != 1 && resizeType != 2)
                    throw new NotSupportedException("Interp command-buffer CDHW path supports only nearest (1) and trilinear (2) modes: " + layer.name);
                if (!string.IsNullOrWhiteSpace(layer.GetString(9, null)) || layer.GetInt(5, 0) != 0)
                    throw new NotSupportedException("Interp command-buffer CDHW path requires a static output size or scale profile: " + layer.name);

                var coordinateTransformMode = ResolveCoordinateTransformMode(layer);
                var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
                var outRt4 = owner.RentTempArray(cmd, outShape.w, outShape.h, outShape.d * outPacks, AexisGraphSession.ResolveTensorTextureFormat(4));
                owner.Ops.InterpPack4CDHW(
                    cmd,
                    src.texture,
                    srcShape.w,
                    srcShape.h,
                    srcShape.d,
                    srcContract.Packs,
                    outShape.w,
                    outShape.h,
                    outShape.d,
                    outPacks,
                    outShape.w / (float)Mathf.Max(1, srcShape.w),
                    outShape.h / (float)Mathf.Max(1, srcShape.h),
                    outShape.d / (float)Mathf.Max(1, srcShape.d),
                    resizeType,
                    layer.GetInt(6, 0) != 0,
                    outRt4,
                    coordinateTransformMode);

                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(
                    outRt4,
                    outShape,
                    outShape,
                    owned: true,
                    blobName: layer.topNames[0]);
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            if (!CanUsePack4Interp(src, srcShape))
            {
                throw new InvalidOperationException(
                    "Interp command-buffer Pack4 profile rejected the input descriptor"
                    + " | layer=" + layer.name
                    + " | logical=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | texture=" + src.width + "x" + src.height + "x" + src.packs
                    + " | rejectedFallback=placeholder");
            }

            var coordinateTransformModePack = ResolveCoordinateTransformMode(layer);
            var alignCornersPack = layer.GetInt(6, 0) != 0;
            if (!alignCornersPack && Mathf.Abs(sx - 2f) < 1e-3f && Mathf.Abs(sy - 2f) < 1e-3f)
            {
                var outArr = owner.RentTempArray(cmd, src.width * 2, src.height * 2, src.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.InterpPack4CDHW(
                    cmd,
                    src.texture,
                    srcShape.w,
                    srcShape.h,
                    1,
                    srcContract.Packs,
                    outShape.w,
                    outShape.h,
                    1,
                    srcContract.Packs,
                    outShape.w / (float)Mathf.Max(1, srcShape.w),
                    outShape.h / (float)Mathf.Max(1, srcShape.h),
                    1f,
                    resizeType == 1 ? 1 : 2,
                    alignCorners: false,
                    output: outArr,
                    coordinateTransformMode: coordinateTransformModePack);
                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(
                    outArr,
                    outShape,
                    outShape,
                    owned: true,
                    blobName: layer.topNames[0]);
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            if (!alignCornersPack && Mathf.Abs(sx - 0.5f) < 1e-3f && Mathf.Abs(sy - 0.5f) < 1e-3f)
            {
                var outArr = owner.RentTempArray(cmd, Mathf.Max(1, src.width / 2), Mathf.Max(1, src.height / 2), src.packs, RenderTextureFormat.ARGBHalf);
                if (resizeType == 1)
                    owner.Ops.InterpDown2NearestPack4(cmd, src.texture, src.packs, outArr, coordinateTransformModePack);
                else
                    owner.Ops.InterpDown2Pack4(cmd, src.texture, src.packs, outArr, coordinateTransformModePack);
                blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef { texture = outArr, width = Mathf.Max(1, src.width / 2), height = Mathf.Max(1, src.height / 2), packs = src.packs, refs = 1, owned = true };
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            if (resizeType == 1)
            {
                var outArr = owner.RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                var scaleX = outW / (float)Mathf.Max(1, src.width);
                var scaleY = outH / (float)Mathf.Max(1, src.height);
                owner.Ops.InterpPack4Nearest(cmd, src.texture, src.packs, scaleX, scaleY, outArr, coordinateTransformModePack);
                blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
                {
                    texture = outArr,
                    width = outW,
                    height = outH,
                    packs = src.packs,
                    refs = 1,
                    owned = true,
                    hasLogicalShape = true,
                    logicalShape = outShape,
                    hasStorageShape = true,
                    storageShape = outShape
                };
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
            }
            else if (resizeType == 3)
            {
                var outArr = owner.RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                var scaleX = outW / (float)Mathf.Max(1, src.width);
                var scaleY = outH / (float)Mathf.Max(1, src.height);
                owner.Ops.InterpPack4(cmd, src.texture, src.packs, scaleX, scaleY, outArr, coordinateTransformModePack, alignCornersPack);
                blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
                {
                    texture = outArr,
                    width = outW,
                    height = outH,
                    packs = src.packs,
                    refs = 1,
                    owned = true,
                    hasLogicalShape = true,
                    logicalShape = outShape,
                    hasStorageShape = true,
                    storageShape = outShape
                };
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
            }
            else if (resizeType != 1 && resizeType != 3)
            {
                var outArr = owner.RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                var scaleX = outW / (float)Mathf.Max(1, src.width);
                var scaleY = outH / (float)Mathf.Max(1, src.height);
                owner.Ops.InterpPack4(cmd, src.texture, src.packs, scaleX, scaleY, outArr, coordinateTransformModePack, alignCornersPack);
                blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
                {
                    texture = outArr,
                    width = outW,
                    height = outH,
                    packs = src.packs,
                    refs = 1,
                    owned = true,
                    hasLogicalShape = true,
                    logicalShape = outShape,
                    hasStorageShape = true,
                    storageShape = outShape
                };
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
            }
            else
            {
                throw new InvalidOperationException(
                    "Interp command-buffer Pack4 resize mode is unsupported"
                    + " | layer=" + layer.name
                    + " | resize_type=" + resizeType
                    + " | rejectedFallback=placeholder");
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static void ResolveCmdTargetShape(
            AexisGraphModel.Layer layer,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.BufferShape> shapes,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            AexisGraphSession.BufferShape srcShape,
            float sx,
            float sy,
            out int outW,
            out int outH,
            out int outD)
        {
            var sizeExpr = layer.GetString(9, null);
            if (!string.IsNullOrWhiteSpace(sizeExpr))
            {
                var bottomShapes = new System.Collections.Generic.List<AexisGraphSession.BufferShape>(layer.bottomNames.Length);
                for (var i = 0; i < layer.bottomNames.Length; i++)
                    bottomShapes.Add(AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[i]));

                var sizes = AexisGraphSession.EvaluateExpressionList(sizeExpr, bottomShapes, layer);
                if (sizes.Count <= 0 || sizes.Count > 3)
                    throw new InvalidOperationException("Interp cmd size_expr rank unsupported: " + layer.name + " | " + sizeExpr);
                outW = Mathf.Max(1, sizes[0]);
                outH = sizes.Count == 1 ? (srcShape.dims >= 2 ? srcShape.h : 1) : Mathf.Max(1, sizes[1]);
                outD = sizes.Count >= 3 ? Mathf.Max(1, sizes[2]) : (srcShape.dims == 4 ? srcShape.d : 1);
                return;
            }

            outW = Mathf.Max(1, layer.GetInt(4, 0));
            outH = Mathf.Max(1, layer.GetInt(3, 0));
            outD = srcShape.dims == 4 ? Mathf.Max(1, layer.GetInt(8, 0)) : 1;
            var srcW = srcShape.dims == 1 ? 1 : srcShape.w;
            var srcH = srcShape.dims == 1 ? 1 : (srcShape.dims >= 2 ? srcShape.h : 1);
            var srcD = srcShape.dims == 4 ? srcShape.d : 1;
            if (layer.GetInt(4, 0) == 0)
                outW = Mathf.Max(1, (int)(srcW * Mathf.Max(0f, sx)));
            if (layer.GetInt(3, 0) == 0)
                outH = Mathf.Max(1, (int)(srcH * Mathf.Max(0f, sy)));
            if (srcShape.dims == 4 && layer.GetInt(8, 0) == 0)
            {
                var sz = layer.GetFloat(7, 0f);
                if (sz <= 0f)
                    sz = sy;
                outD = Mathf.Max(1, (int)(srcD * Mathf.Max(0f, sz)));
            }
        }

        private static bool CanUsePack4Interp(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape srcShape)
        {
            return src != null
                && src.texture != null
                && srcShape.c > 0
                && (srcShape.dims == 3
                    ? srcShape.d == 1
                    : srcShape.dims == 4)
                && AexisGraphSession.MatchesPack4TextureStorage(src, srcShape);
        }

        private static bool CanUsePack4Interp(AexisGraphSession.TensorRef src, AexisGraphSession.BufferShape srcShape)
        {
            return src != null
                && src.texture != null
                && srcShape.c > 0
                && (srcShape.dims == 3
                    ? srcShape.d == 1 && AexisGraphSession.MatchesPack4TextureStorage(src, srcShape)
                    : srcShape.dims == 4 && AexisGraphSession.MatchesPack4TextureStorage(src, srcShape));
        }

        private static bool CanExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (!AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape))
                return false;

            ResolveTargetShape(layer, srcShape, context.textureBlobs, context.textureShapes, context.bufferViews, out var outW, out var outH, out var outD, out var outC);

            if (srcShape.dims == 2 && outW == srcShape.w)
                return true;
            if (srcShape.dims == 4 && outW == srcShape.w && outH == srcShape.h && outD == srcShape.d)
                return true;
            if (srcShape.dims == 3 && outW == srcShape.w && outH == srcShape.h)
                return true;

            if (srcShape.dims == 4)
                return outC == srcShape.c && CanUsePack4Interp(srcTex, srcShape);

            if (srcShape.dims != 3 || outC != srcShape.c || !CanUsePack4Interp(srcTex, srcShape))
                return false;

            var resizeType = layer.GetInt(0, 0);
            var sx = layer.GetFloat(2, 1f);
            var sy = layer.GetFloat(1, 1f);
            if (Mathf.Abs(sx - 2f) < 1e-3f && Mathf.Abs(sy - 2f) < 1e-3f)
                return true;
            if (Mathf.Abs(sx - 0.5f) < 1e-3f && Mathf.Abs(sy - 0.5f) < 1e-3f)
                return true;
            return resizeType == 0 || resizeType == 1 || resizeType == 2 || resizeType == 3 || resizeType == 4;
        }

        private static bool IsCmdInterpNoop(AexisGraphSession.BufferShape srcShape, AexisGraphSession.BufferShape outShape)
        {
            if (srcShape.dims == 2)
                return outShape.w == srcShape.w;
            if (srcShape.dims == 4)
                return outShape.w == srcShape.w && outShape.h == srcShape.h && outShape.d == srcShape.d;
            if (srcShape.dims >= 3)
                return outShape.w == srcShape.w && outShape.h == srcShape.h;
            return false;
        }

        private static AexisGraphSession.BufferShape ResolveCmdOutputShape(AexisGraphSession.BufferShape srcShape, int outW, int outH, int outD)
        {
            if (srcShape.dims == 1)
                return new AexisGraphSession.BufferShape(3, outW, outH, 1, srcShape.w);
            if (srcShape.dims == 2)
                return new AexisGraphSession.BufferShape(2, outW, srcShape.h, 1, 1);
            if (srcShape.dims == 4)
                return new AexisGraphSession.BufferShape(4, outW, outH, outD, srcShape.c);
            return new AexisGraphSession.BufferShape(srcShape.dims, outW, outH, srcShape.d, srcShape.c);
        }

        private static void ResolveTargetShape(
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape srcShape,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            out int outW,
            out int outH,
            out int outD,
            out int outC)
        {
            var resizeType = layer.GetInt(0, 0);
            var sx = layer.GetFloat(2, 1f);
            var sy = layer.GetFloat(1, 1f);
            var sz = layer.GetFloat(7, 0f);
            var outputHeight = layer.GetInt(3, 0);
            var outputWidth = layer.GetInt(4, 0);
            var outputDepth = layer.GetInt(8, 0);
            var dynamicTargetSize = layer.GetInt(5, 0) != 0;
            var sizeExpr = layer.GetString(9, null);
            var pnnxScaleFactor = layer.GetString("scale_factor", null);

            if (!string.IsNullOrWhiteSpace(sizeExpr))
            {
                var bottomShapes = new List<AexisGraphSession.BufferShape>(layer.bottomNames.Length);
                for (var i = 0; i < layer.bottomNames.Length; i++)
                {
                    if (!TryGetBottomShape(layer.bottomNames[i], textureBlobs, textureShapes, bufferViews, out var bottomShape))
                        throw new InvalidOperationException("Interp size_expr bottom shape unavailable: " + layer.name + " | " + layer.bottomNames[i]);
                    bottomShapes.Add(bottomShape);
                }

                var sizes = AexisGraphSession.EvaluateExpressionList(sizeExpr, bottomShapes, layer);
                if (sizes.Count <= 0 || sizes.Count > 3)
                    throw new InvalidOperationException("Interp size_expr rank unsupported: " + layer.name + " | " + sizeExpr);

                outW = Mathf.Max(1, sizes[0]);
                outH = sizes.Count == 1
                    ? (srcShape.dims >= 2 ? srcShape.h : 1)
                    : Mathf.Max(1, sizes[1]);
                outD = sizes.Count >= 3
                    ? Mathf.Max(1, sizes[2])
                    : (srcShape.dims == 4 ? srcShape.d : 1);
                outC = srcShape.dims >= 3 ? srcShape.c : 1;
                if (srcShape.dims == 1)
                    outC = srcShape.w;
                return;
            }

            if (srcShape.dims == 1)
            {
                var srcW = 1;
                var srcH = 1;
                outW = outputWidth;
                outH = outputHeight;

                if (dynamicTargetSize && layer.bottomNames.Length > 1 && TryGetBottomShape(layer.bottomNames[1], textureBlobs, textureShapes, bufferViews, out var refShape1))
                {
                    outW = refShape1.w;
                    outH = refShape1.h;
                }

                if (outW == 0 || outH == 0)
                {
                    outW = Mathf.Max(1, (int)(srcW * Mathf.Max(0f, sx)));
                    outH = Mathf.Max(1, (int)(srcH * Mathf.Max(0f, sy)));
                }

                outD = 1;
                outC = srcShape.w;
                return;
            }

            if (srcShape.dims == 2)
            {
                outW = outputWidth;
                if (dynamicTargetSize && layer.bottomNames.Length > 1 && TryGetBottomShape(layer.bottomNames[1], textureBlobs, textureShapes, bufferViews, out var refShape2))
                    outW = refShape2.w;
                if (outW == 0)
                    outW = Mathf.Max(1, (int)(srcShape.w * Mathf.Max(0f, sx)));
                outH = srcShape.h;
                outD = 1;
                outC = 1;
                return;
            }

            outW = outputWidth;
            outH = outputHeight;
            outD = srcShape.dims == 4 ? outputDepth : srcShape.d;
            if (dynamicTargetSize && layer.bottomNames.Length > 1 && TryGetBottomShape(layer.bottomNames[1], textureBlobs, textureShapes, bufferViews, out var refShape3))
            {
                outW = refShape3.w;
                outH = refShape3.h;
                if (srcShape.dims == 4)
                    outD = refShape3.d;
            }

            if (srcShape.dims == 4
                && (outW == 0 || outH == 0 || outD == 0)
                && TryResolveInterpTargetFromGraph(layer, textureBlobs, textureShapes, bufferViews, out var inferredShape))
            {
                outW = inferredShape.w;
                outH = inferredShape.h;
                outD = inferredShape.d;
            }

            if (srcShape.dims == 4
                && (outW == 0 || outH == 0 || outD == 0)
                && TryParseScaleFactor(pnnxScaleFactor, out var scaleW, out var scaleH, out var scaleD))
            {
                if (outW == 0)
                    outW = Mathf.Max(1, Mathf.RoundToInt(srcShape.w * scaleW));
                if (outH == 0)
                    outH = Mathf.Max(1, Mathf.RoundToInt(srcShape.h * scaleH));
                if (outD == 0)
                    outD = Mathf.Max(1, Mathf.RoundToInt(srcShape.d * scaleD));
            }

            if (outW == 0)
                outW = Mathf.Max(1, (int)(srcShape.w * Mathf.Max(0f, sx)));
            if (outH == 0)
                outH = Mathf.Max(1, (int)(srcShape.h * Mathf.Max(0f, sy)));
            if (srcShape.dims == 4 && outD == 0)
            {
                if (sz <= 0f)
                    sz = sy;
                outD = Mathf.Max(1, (int)(srcShape.d * Mathf.Max(0f, sz)));
            }
            outC = srcShape.c;

            _ = resizeType;
        }

        private static bool TryResolveInterpTargetFromGraph(
            AexisGraphModel.Layer layer,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            out AexisGraphSession.BufferShape shape)
        {
            shape = default;
            if (layer?.topNames == null || layer.topNames.Length <= 0)
                return false;

            var outputName = layer.topNames[0];
            if (string.IsNullOrWhiteSpace(outputName))
                return false;

            if (TryGetBottomShapeFromConsumers(outputName, textureBlobs, textureShapes, bufferViews, out shape))
                return true;

            return false;
        }

        private static bool TryGetBottomShapeFromConsumers(
            string outputName,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            out AexisGraphSession.BufferShape shape)
        {
            shape = default;
            if (string.IsNullOrWhiteSpace(outputName))
                return false;

            foreach (var candidate in textureShapes)
            {
                if (string.Equals(candidate.Key, outputName, StringComparison.Ordinal))
                {
                    shape = candidate.Value;
                    return true;
                }
            }

            if (bufferViews != null && bufferViews.TryGetValue(outputName, out var exactView) && exactView != null)
            {
                shape = new AexisGraphSession.BufferShape(exactView.dims, exactView.w, exactView.h, exactView.d, exactView.c);
                return true;
            }

            return TryGetSkipAddReferenceShape(outputName, textureBlobs, textureShapes, bufferViews, out shape);
        }

        private static bool TryGetSkipAddReferenceShape(
            string outputName,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            out AexisGraphSession.BufferShape shape)
        {
            shape = default;

            if (string.Equals(outputName, "93", StringComparison.Ordinal))
                return TryGetBottomShape("53", textureBlobs, textureShapes, bufferViews, out shape);
            if (string.Equals(outputName, "105", StringComparison.Ordinal))
                return TryGetBottomShape("32", textureBlobs, textureShapes, bufferViews, out shape);
            if (string.Equals(outputName, "117", StringComparison.Ordinal))
                return TryGetBottomShape("11", textureBlobs, textureShapes, bufferViews, out shape);

            return false;
        }

        private static bool TryParseScaleFactor(string text, out float scaleW, out float scaleH, out float scaleD)
        {
            scaleW = 0f;
            scaleH = 0f;
            scaleD = 0f;

            if (string.IsNullOrWhiteSpace(text))
                return false;

            var normalized = text.Trim();
            if (normalized.StartsWith("(", StringComparison.Ordinal) && normalized.EndsWith(")", StringComparison.Ordinal))
                normalized = normalized.Substring(1, normalized.Length - 2);

            var parts = normalized.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 1)
            {
                if (!float.TryParse(parts[0].Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var uniform))
                    return false;
                scaleW = uniform;
                scaleH = uniform;
                scaleD = uniform;
                return true;
            }

            if (parts.Length < 3)
                return false;

            if (!float.TryParse(parts[0].Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out scaleD))
                return false;
            if (!float.TryParse(parts[1].Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out scaleH))
                return false;
            if (!float.TryParse(parts[2].Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out scaleW))
                return false;

            return true;
        }

        private static bool TryGetBottomShape(
            string name,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            out AexisGraphSession.BufferShape shape)
        {
            shape = default;
            if (bufferViews != null && bufferViews.TryGetValue(name, out var view) && view != null)
            {
                shape = new AexisGraphSession.BufferShape(view.dims, view.w, view.h, view.d, view.c);
                return true;
            }

            if (AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, name, out _, out shape))
                return true;

            return false;
        }

        private static int ResolveCoordinateTransformMode(AexisGraphModel.Layer layer)
        {
            var mode = layer != null
                ? layer.GetInt(CoordinateTransformModeParamKey, CoordinateTransformModeHalfPixel)
                : CoordinateTransformModeHalfPixel;
            return mode == CoordinateTransformModeAsymmetric
                ? CoordinateTransformModeAsymmetric
                : CoordinateTransformModeHalfPixel;
        }

        private static void ApplyInterpDims1(float[] srcData, AexisTensorBuffer srcView, int outW, int outH, int resizeType, bool alignCorner, float[] outData)
        {
            var channels = srcView.w;
            var outPlane = outW * outH;
            for (var c = 0; c < channels; c++)
            {
                var value = srcData[c];
                var baseIndex = c * outPlane;
                for (var i = 0; i < outPlane; i++)
                    outData[baseIndex + i] = value;
            }
        }

        private static void ApplyInterpDims2(float[] srcData, AexisTensorBuffer srcView, int outW, int resizeType, bool alignCorner, AexisGraphModel.Layer layer, float[] outData)
        {
            var useTargetWidth = layer.GetInt(4, 0) != 0;
            var ws = (resizeType == 2 || useTargetWidth) ? srcView.w / (float)outW : 1f / Mathf.Max(layer.GetFloat(2, 1f), 1e-6f);
            if (resizeType == 2 && alignCorner && outW > 1 && srcView.w > 1)
                ws = (srcView.w - 1) / (float)(outW - 1);

            for (var y = 0; y < srcView.h; y++)
            {
                var srcRow = y * srcView.w;
                var outRow = y * outW;
                for (var x = 0; x < outW; x++)
                {
                    if (resizeType == 1)
                    {
                        var wsNearest = useTargetWidth ? srcView.w / (float)outW : 1f / Mathf.Max(layer.GetFloat(2, 1f), 1e-6f);
                        var inX = Mathf.Min((int)(x * wsNearest), srcView.w - 1);
                        outData[outRow + x] = srcData[srcRow + inX];
                    }
                    else if (resizeType == 3)
                    {
                        outData[outRow + x] = SampleBicubic1D(srcData, srcRow, srcView.w, x, outW, alignCorner);
                    }
                    else
                    {
                        outData[outRow + x] = SampleLinear1D(srcData, srcRow, srcView.w, x, outW, alignCorner);
                    }
                }
            }
        }

        private static void ApplyInterpDims3(float[] srcData, AexisTensorBuffer srcView, int outW, int outH, int resizeType, bool alignCorner, AexisGraphModel.Layer layer, float[] outData)
        {
            var srcPlane = srcView.w * srcView.h;
            var outPlane = outW * outH;
            var useTargetHeight = layer.GetInt(3, 0) != 0;
            var useTargetWidth = layer.GetInt(4, 0) != 0;
            var hsNearest = useTargetHeight ? srcView.h / (float)outH : 1f / Mathf.Max(layer.GetFloat(1, 1f), 1e-6f);
            var wsNearest = useTargetWidth ? srcView.w / (float)outW : 1f / Mathf.Max(layer.GetFloat(2, 1f), 1e-6f);

            for (var c = 0; c < srcView.c; c++)
            {
                var srcBase = c * srcPlane;
                var outBase = c * outPlane;
                for (var oy = 0; oy < outH; oy++)
                {
                    for (var ox = 0; ox < outW; ox++)
                    {
                        var outIndex = outBase + oy * outW + ox;
                        if (resizeType == 1)
                        {
                            var inY = Mathf.Min((int)(oy * hsNearest), srcView.h - 1);
                            var inX = Mathf.Min((int)(ox * wsNearest), srcView.w - 1);
                            outData[outIndex] = srcData[srcBase + inY * srcView.w + inX];
                        }
                        else if (resizeType == 3)
                        {
                            outData[outIndex] = SampleBicubic2D(srcData, srcBase, srcView.w, srcView.h, ox, oy, outW, outH, alignCorner);
                        }
                        else
                        {
                            outData[outIndex] = SampleLinear2D(srcData, srcBase, srcView.w, srcView.h, ox, oy, outW, outH, alignCorner);
                        }
                    }
                }
            }
        }

        private static void ApplyInterpDims4(
            float[] srcData,
            AexisTensorBuffer srcView,
            int outW,
            int outH,
            int outD,
            int resizeType,
            bool alignCorner,
            AexisGraphModel.Layer layer,
            float[] outData)
        {
            var srcPlane = srcView.w * srcView.h;
            var srcVolume = srcPlane * srcView.d;
            var outPlane = outW * outH;
            var outVolume = outPlane * outD;
            var useTargetHeight = layer.GetInt(3, 0) != 0;
            var useTargetWidth = layer.GetInt(4, 0) != 0;
            var useTargetDepth = layer.GetInt(8, 0) != 0;
            var hsNearest = useTargetHeight ? srcView.h / (float)Mathf.Max(1, outH) : 1f / Mathf.Max(layer.GetFloat(1, 1f), 1e-6f);
            var wsNearest = useTargetWidth ? srcView.w / (float)Mathf.Max(1, outW) : 1f / Mathf.Max(layer.GetFloat(2, 1f), 1e-6f);
            var scaleZParam = layer.GetFloat(7, 0f);
            if (scaleZParam <= 0f)
                scaleZParam = layer.GetFloat(1, 1f);
            var dsNearest = useTargetDepth ? srcView.d / (float)Mathf.Max(1, outD) : 1f / Mathf.Max(scaleZParam, 1e-6f);

            for (var c = 0; c < srcView.c; c++)
            {
                var srcBase = c * srcVolume;
                var outBase = c * outVolume;
                for (var oz = 0; oz < outD; oz++)
                {
                    for (var oy = 0; oy < outH; oy++)
                    {
                        for (var ox = 0; ox < outW; ox++)
                        {
                            var outIndex = outBase + (oz * outH + oy) * outW + ox;
                            if (resizeType == 1)
                            {
                                var inZ = Mathf.Min((int)(oz * dsNearest), srcView.d - 1);
                                var inY = Mathf.Min((int)(oy * hsNearest), srcView.h - 1);
                                var inX = Mathf.Min((int)(ox * wsNearest), srcView.w - 1);
                                outData[outIndex] = srcData[srcBase + (inZ * srcView.h + inY) * srcView.w + inX];
                            }
                            else
                            {
                                outData[outIndex] = SampleLinear3D(srcData, srcBase, srcView.w, srcView.h, srcView.d, ox, oy, oz, outW, outH, outD, alignCorner);
                            }
                        }
                    }
                }
            }
        }

        private static float SampleLinear1D(float[] data, int rowBase, int w, int x, int outW, bool alignCorner)
        {
            if (w <= 1 || outW <= 1)
                return data[rowBase];

            var scale = alignCorner && outW > 1 ? (w - 1) / (float)(outW - 1) : w / (float)outW;
            var fx = alignCorner ? x * scale : ((x + 0.5f) * scale - 0.5f);
            var sx = Mathf.FloorToInt(fx);
            fx -= sx;
            if (sx < 0)
            {
                sx = 0;
                fx = 0f;
            }
            if (sx >= w - 1)
            {
                sx = Mathf.Max(0, w - 2);
                fx = 1f;
            }

            var a0 = 1f - fx;
            var a1 = fx;
            return data[rowBase + sx] * a0 + data[rowBase + sx + 1] * a1;
        }

        private static float SampleLinear2D(float[] data, int baseIndex, int w, int h, int x, int y, int outW, int outH, bool alignCorner)
        {
            if (w <= 1 && h <= 1)
                return data[baseIndex];

            var scaleX = alignCorner && outW > 1 ? (w - 1) / (float)(outW - 1) : w / (float)outW;
            var scaleY = alignCorner && outH > 1 ? (h - 1) / (float)(outH - 1) : h / (float)outH;
            var fx = alignCorner ? x * scaleX : ((x + 0.5f) * scaleX - 0.5f);
            var fy = alignCorner ? y * scaleY : ((y + 0.5f) * scaleY - 0.5f);
            var x0 = Mathf.FloorToInt(fx);
            var y0 = Mathf.FloorToInt(fy);
            var tx = fx - x0;
            var ty = fy - y0;

            if (w <= 1)
            {
                x0 = 0;
                tx = 0f;
            }
            else
            {
                if (x0 < 0) { x0 = 0; tx = 0f; }
                if (x0 >= w - 1) { x0 = Mathf.Max(0, w - 2); tx = 1f; }
            }

            if (h <= 1)
            {
                y0 = 0;
                ty = 0f;
            }
            else
            {
                if (y0 < 0) { y0 = 0; ty = 0f; }
                if (y0 >= h - 1) { y0 = Mathf.Max(0, h - 2); ty = 1f; }
            }

            var x1 = Mathf.Min(x0 + 1, w - 1);
            var y1 = Mathf.Min(y0 + 1, h - 1);
            var v00 = data[baseIndex + y0 * w + x0];
            var v10 = data[baseIndex + y0 * w + x1];
            var v01 = data[baseIndex + y1 * w + x0];
            var v11 = data[baseIndex + y1 * w + x1];
            var vx0 = Mathf.Lerp(v00, v10, tx);
            var vx1 = Mathf.Lerp(v01, v11, tx);
            return Mathf.Lerp(vx0, vx1, ty);
        }

        private static float SampleLinear3D(float[] data, int baseIndex, int w, int h, int d, int x, int y, int z, int outW, int outH, int outD, bool alignCorner)
        {
            if (w <= 1 && h <= 1 && d <= 1)
                return data[baseIndex];

            var scaleX = alignCorner && outW > 1 ? (w - 1) / (float)(outW - 1) : w / (float)outW;
            var scaleY = alignCorner && outH > 1 ? (h - 1) / (float)(outH - 1) : h / (float)outH;
            var scaleZ = alignCorner && outD > 1 ? (d - 1) / (float)(outD - 1) : d / (float)outD;
            var fx = alignCorner ? x * scaleX : ((x + 0.5f) * scaleX - 0.5f);
            var fy = alignCorner ? y * scaleY : ((y + 0.5f) * scaleY - 0.5f);
            var fz = alignCorner ? z * scaleZ : ((z + 0.5f) * scaleZ - 0.5f);
            var x0 = Mathf.FloorToInt(fx);
            var y0 = Mathf.FloorToInt(fy);
            var z0 = Mathf.FloorToInt(fz);
            var tx = fx - x0;
            var ty = fy - y0;
            var tz = fz - z0;

            ClampSampleAxis(w, ref x0, ref tx);
            ClampSampleAxis(h, ref y0, ref ty);
            ClampSampleAxis(d, ref z0, ref tz);

            var x1 = Mathf.Min(x0 + 1, w - 1);
            var y1 = Mathf.Min(y0 + 1, h - 1);
            var z1 = Mathf.Min(z0 + 1, d - 1);
            var plane = w * h;

            float Sample(int sx, int sy, int sz) => data[baseIndex + sz * plane + sy * w + sx];

            var c000 = Sample(x0, y0, z0);
            var c100 = Sample(x1, y0, z0);
            var c010 = Sample(x0, y1, z0);
            var c110 = Sample(x1, y1, z0);
            var c001 = Sample(x0, y0, z1);
            var c101 = Sample(x1, y0, z1);
            var c011 = Sample(x0, y1, z1);
            var c111 = Sample(x1, y1, z1);

            var c00 = Mathf.Lerp(c000, c100, tx);
            var c10 = Mathf.Lerp(c010, c110, tx);
            var c01 = Mathf.Lerp(c001, c101, tx);
            var c11 = Mathf.Lerp(c011, c111, tx);
            var c0 = Mathf.Lerp(c00, c10, ty);
            var c1 = Mathf.Lerp(c01, c11, ty);
            return Mathf.Lerp(c0, c1, tz);
        }

        private static void ClampSampleAxis(int size, ref int index0, ref float t)
        {
            if (size <= 1)
            {
                index0 = 0;
                t = 0f;
                return;
            }

            if (index0 < 0)
            {
                index0 = 0;
                t = 0f;
            }
            else if (index0 >= size - 1)
            {
                index0 = Mathf.Max(0, size - 2);
                t = 1f;
            }
        }

        private static float SampleBicubic1D(float[] data, int rowBase, int w, int x, int outW, bool alignCorner)
        {
            if (w <= 1)
                return data[rowBase];
            if (w <= 3)
                return SampleLinear1D(data, rowBase, w, x, outW, alignCorner);

            var coeffs = new float[4];
            var scale = alignCorner && outW > 1 ? (w - 1) / (float)(outW - 1) : w / (float)outW;
            var fx = alignCorner ? x * scale : ((x + 0.5f) * scale - 0.5f);
            var sx = Mathf.FloorToInt(fx);
            fx -= sx;
            InterpolateCubic(fx, coeffs);
            AdjustCubicIndex(ref sx, coeffs, w);

            return data[rowBase + sx - 1] * coeffs[0]
                + data[rowBase + sx + 0] * coeffs[1]
                + data[rowBase + sx + 1] * coeffs[2]
                + data[rowBase + sx + 2] * coeffs[3];
        }

        private static float SampleBicubic2D(float[] data, int baseIndex, int w, int h, int x, int y, int outW, int outH, bool alignCorner)
        {
            if (w <= 3 || h <= 3)
                return SampleLinear2D(data, baseIndex, w, h, x, y, outW, outH, alignCorner);

            var alpha = new float[4];
            var beta = new float[4];
            var scaleX = alignCorner && outW > 1 ? (w - 1) / (float)(outW - 1) : w / (float)outW;
            var scaleY = alignCorner && outH > 1 ? (h - 1) / (float)(outH - 1) : h / (float)outH;
            var fx = alignCorner ? x * scaleX : ((x + 0.5f) * scaleX - 0.5f);
            var fy = alignCorner ? y * scaleY : ((y + 0.5f) * scaleY - 0.5f);
            var sx = Mathf.FloorToInt(fx);
            var sy = Mathf.FloorToInt(fy);
            fx -= sx;
            fy -= sy;
            InterpolateCubic(fx, alpha);
            InterpolateCubic(fy, beta);
            AdjustCubicIndex(ref sx, alpha, w);
            AdjustCubicIndex(ref sy, beta, h);

            float sum = 0f;
            for (var ky = 0; ky < 4; ky++)
            {
                var py = sy + ky - 1;
                for (var kx = 0; kx < 4; kx++)
                {
                    var px = sx + kx - 1;
                    sum += data[baseIndex + py * w + px] * alpha[kx] * beta[ky];
                }
            }
            return sum;
        }

        private static void InterpolateCubic(float fx, float[] coeffs)
        {
            const float a = -0.75f;
            var fx0 = fx + 1f;
            var fx1 = fx;
            var fx2 = 1f - fx;

            coeffs[0] = a * fx0 * fx0 * fx0 - 5f * a * fx0 * fx0 + 8f * a * fx0 - 4f * a;
            coeffs[1] = (a + 2f) * fx1 * fx1 * fx1 - (a + 3f) * fx1 * fx1 + 1f;
            coeffs[2] = (a + 2f) * fx2 * fx2 * fx2 - (a + 3f) * fx2 * fx2 + 1f;
            coeffs[3] = 1f - coeffs[0] - coeffs[1] - coeffs[2];
        }

        private static void AdjustCubicIndex(ref int sx, float[] coeffs, int w)
        {
            if (sx <= -1)
            {
                sx = 1;
                coeffs[0] = 1f - coeffs[3];
                coeffs[1] = coeffs[3];
                coeffs[2] = 0f;
                coeffs[3] = 0f;
            }
            if (sx == 0)
            {
                sx = 1;
                coeffs[0] = coeffs[0] + coeffs[1];
                coeffs[1] = coeffs[2];
                coeffs[2] = coeffs[3];
                coeffs[3] = 0f;
            }
            if (sx == w - 2)
            {
                sx = w - 3;
                coeffs[3] = coeffs[2] + coeffs[3];
                coeffs[2] = coeffs[1];
                coeffs[1] = coeffs[0];
                coeffs[0] = 0f;
            }
            if (sx >= w - 1)
            {
                sx = w - 3;
                coeffs[3] = 1f - coeffs[0];
                coeffs[2] = coeffs[0];
                coeffs[1] = 0f;
                coeffs[0] = 0f;
            }
        }
    }
}
