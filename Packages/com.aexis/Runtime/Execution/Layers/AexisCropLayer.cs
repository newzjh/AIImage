using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisCropLayer : AexisBaseLayer
    {
        public AexisCropLayer() : base(AexisLayerTypes.Crop, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!TryGetBottomShape(owner, layer.bottomNames[0], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, context.tempOwned, out var srcShape))
                throw new InvalidOperationException("Crop source shape not found: " + layer.name);

            var roi = ResolveCropRoi(owner, layer, srcShape, context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, context.tempOwned);
            if (owner.TryGetPack4Texture(layer.bottomNames[0], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out var srcTex, out var texShape)
                && CanUsePack4Crop(srcTex, texShape, roi))
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

            if (!TryGetBottomShape(owner, layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, tempOwned, out var srcShape))
                throw new InvalidOperationException("Crop source shape not found: " + layer.name);

            var roi = ResolveCropRoi(owner, layer, srcShape, textureBlobs, textureShapes, bufferBlobs, bufferViews, tempOwned);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Crop source not found: " + layer.name);

            var cropResult = owner.ApplyCrop(srcBuf, srcView, roi, tempOwned);
            var isAlias = ReferenceEquals(cropResult.buffer, srcBuf)
                && cropResult.dims == srcView.dims
                && cropResult.w == srcView.w
                && cropResult.h == srcView.h
                && cropResult.d == srcView.d
                && cropResult.c == srcView.c;

            if (isAlias && TryAliasExistingBuffer(layer.bottomNames[0], layer.topNames[0], bufferBlobs, bufferRefs, bufferViews, cropResult, out var aliased) && aliased)
            {
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (isAlias)
            {
                var outBuf = owner.RentTempBuffer(srcView.elementCount, sizeof(float));
                owner.Ops.CopyBufPartial(srcBuf, 0, outBuf, srcView.elementCount);
                bufferBlobs[layer.topNames[0]] = outBuf;
                bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], outBuf);
                bufferViews[layer.topNames[0]] = new AexisTensorBuffer(outBuf, cropResult.dims, cropResult.w, cropResult.h, cropResult.d, cropResult.c, false);
                tempOwned.Add(outBuf);
            }
            else
            {
                owner.PublishTensorBufferOutput(
                    layer.topNames[0],
                    cropResult,
                    preferTexture: cropResult.dims <= 3,
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
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!TryGetBottomShape(owner, layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, context.bufferViews, context.tempOwned, out var srcShape))
                throw new InvalidOperationException("Crop source shape not found: " + layer.name);
            var roi = ResolveCropRoi(owner, layer, srcShape, textureBlobs, textureShapes, bufferBlobs, context.bufferViews, context.tempOwned);
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, context.bufferViews, out var srcTex, out var texShape)
                || !CanUsePack4Crop(srcTex, texShape, roi))
            {
                throw new InvalidOperationException("Crop render-texture path requires supported pack4 input: " + layer.name);
            }

            if (IsIdentityCrop(texShape, roi))
            {
                var storageShape = AexisGraphSession.GetTextureStorageShape(srcTex, texShape);
                textureBlobs[layer.topNames[0]] = AexisGraphSession.CreateTextureAlias(srcTex, texShape, storageShape);
                textureShapes[layer.topNames[0]] = texShape;
            }
            else
            {
                var outPacks = Mathf.Max(1, Mathf.CeilToInt(roi.outc / 4f));
                var outRt = owner.RentTempArray(roi.outw, roi.outh, outPacks, RenderTextureFormat.ARGBHalf);
                owner.Ops.CropPack4(srcTex.texture, texShape.w, texShape.h, texShape.c, roi.woffset, roi.hoffset, roi.coffset, roi.outw, roi.outh, roi.outc, outRt);
                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, new AexisGraphSession.BufferShape(3, roi.outw, roi.outh, 1, roi.outc));
            }

            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, remaining, layer.bottomNames, pinnedNames);
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
            if (!TryResolveCmdCropRoi(layer, shapes, blobs, srcShape, out var roi))
            {
                owner.CopyCmdTensor(cmd, src, layer.topNames[0], blobs, src.width, src.height, src.packs, shapes, srcShape);
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            if (CanUsePack4Crop(src, srcShape, roi))
            {
                if (IsIdentityCrop(srcShape, roi))
                {
                    var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                    blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorAlias(src, srcShape, storageShape);
                    shapes[layer.topNames[0]] = srcShape;
                }
                else
                {
                    var outPacks = Mathf.Max(1, Mathf.CeilToInt(roi.outc / 4f));
                    var outArr = owner.RentTempArray(cmd, roi.outw, roi.outh, outPacks, RenderTextureFormat.ARGBHalf);
                    owner.Ops.CropPack4(cmd, src.texture, srcShape.w, srcShape.h, srcShape.c, roi.woffset, roi.hoffset, roi.coffset, roi.outw, roi.outh, roi.outc, outArr);
                    blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
                    {
                        texture = outArr,
                        width = roi.outw,
                        height = roi.outh,
                        packs = outPacks,
                        refs = 1,
                        owned = true
                    };
                    shapes[layer.topNames[0]] = new AexisGraphSession.BufferShape(3, roi.outw, roi.outh, 1, roi.outc);
                }
            }
            else
            {
                var outPacks = Mathf.Max(1, Mathf.CeilToInt(roi.outc / 4f));
                owner.CopyCmdTensor(
                    cmd,
                    src,
                    layer.topNames[0],
                    blobs,
                    roi.outw,
                    roi.outh,
                    outPacks,
                    shapes,
                    new AexisGraphSession.BufferShape(3, roi.outw, roi.outh, 1, roi.outc));
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static AexisGraphSession.CropRoi ResolveCropRoi(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape srcShape,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            List<IDisposable> tempOwned)
        {
            var hasSliceArrays = (layer.GetInts(-23309, null)?.Length ?? 0) > 0 && (layer.GetInts(-23310, null)?.Length ?? 0) > 0;
            var hasSliceExpr = !string.IsNullOrWhiteSpace(layer.GetString(19, null)) && !string.IsNullOrWhiteSpace(layer.GetString(20, null));

            if (hasSliceArrays || hasSliceExpr)
            {
                var bottomShapes = new List<AexisGraphSession.BufferShape>(layer.bottomNames.Length) { srcShape };
                if (layer.bottomNames.Length > 1 && TryGetBottomShape(owner, layer.bottomNames[1], textureBlobs, textureShapes, bufferBlobs, bufferViews, tempOwned, out var secondaryShape))
                    bottomShapes.Add(secondaryShape);
                return AexisGraphSession.ResolveCropRoi(srcShape, layer, bottomShapes);
            }

            if (layer.bottomNames.Length > 1)
            {
                if (layer.GetInt(0, 0) == -233)
                {
                    var refBuf = owner.GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    if (refBuf == null)
                        throw new InvalidOperationException("Crop param_data source not found: " + layer.name);
                    var refData = AexisGraphSession.ReadFloatBuffer(refBuf);
                    var paramData = new int[refData.Length];
                    for (var i = 0; i < refData.Length; i++)
                        paramData[i] = Mathf.RoundToInt(refData[i]);
                    return AexisGraphSession.ResolveCropRoi(srcShape, paramData, layer);
                }

                if (!TryGetBottomShape(owner, layer.bottomNames[1], textureBlobs, textureShapes, bufferBlobs, bufferViews, tempOwned, out var referenceShape))
                    throw new InvalidOperationException("Crop reference shape missing: " + layer.name);
                return AexisGraphSession.ResolveCropRoi(srcShape, referenceShape, layer);
            }

            var singleBottomShapes = new List<AexisGraphSession.BufferShape>(1) { srcShape };
            return AexisGraphSession.ResolveCropRoi(srcShape, layer, singleBottomShapes);
        }

        private static bool TryResolveCmdCropRoi(
            AexisGraphModel.Layer layer,
            Dictionary<string, AexisGraphSession.BufferShape> shapes,
            Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            AexisGraphSession.BufferShape srcShape,
            out AexisGraphSession.CropRoi roi)
        {
            roi = default;
            var hasSliceArrays = (layer.GetInts(-23309, null)?.Length ?? 0) > 0 && (layer.GetInts(-23310, null)?.Length ?? 0) > 0;
            var hasSliceExpr = !string.IsNullOrWhiteSpace(layer.GetString(19, null)) && !string.IsNullOrWhiteSpace(layer.GetString(20, null));
            if (hasSliceArrays || hasSliceExpr)
            {
                var bottomShapes = new List<AexisGraphSession.BufferShape>(layer.bottomNames.Length) { srcShape };
                for (var i = 1; i < layer.bottomNames.Length; i++)
                {
                    if (AexisGraphSession.TryGetCmdShape(shapes, blobs, layer.bottomNames[i], out var secondaryShape))
                        bottomShapes.Add(secondaryShape);
                }
                roi = AexisGraphSession.ResolveCropRoi(srcShape, layer, bottomShapes);
                return true;
            }

            if (layer.bottomNames.Length > 1)
            {
                if (layer.GetInt(0, 0) == -233)
                    return false;

                if (!AexisGraphSession.TryGetCmdShape(shapes, blobs, layer.bottomNames[1], out var referenceShape))
                    return false;
                roi = AexisGraphSession.ResolveCropRoi(srcShape, referenceShape, layer);
                return true;
            }

            roi = AexisGraphSession.ResolveCropRoi(srcShape, layer, new[] { srcShape });
            return true;
        }

        private static bool CanUsePack4Crop(AexisGraphSession.TensorRef srcTex, AexisGraphSession.BufferShape srcShape, AexisGraphSession.CropRoi roi)
        {
            return srcTex != null
                && srcTex.texture != null
                && srcShape.dims == 3
                && srcShape.w == srcTex.width
                && srcShape.h == srcTex.height
                && srcShape.d == 1
                && roi.outd == 1;
        }

        private static bool CanUsePack4Crop(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape srcShape, AexisGraphSession.CropRoi roi)
        {
            return src != null
                && src.texture != null
                && srcShape.dims == 3
                && srcShape.w == src.width
                && srcShape.h == src.height
                && srcShape.d == 1
                && roi.outd == 1;
        }

        private static bool IsIdentityCrop(AexisGraphSession.BufferShape srcShape, AexisGraphSession.CropRoi roi)
        {
            return roi.woffset == 0
                && roi.hoffset == 0
                && roi.doffset == 0
                && roi.coffset == 0
                && roi.outw == srcShape.w
                && roi.outh == srcShape.h
                && roi.outd == srcShape.d
                && roi.outc == srcShape.c;
        }

        private static bool TryGetBottomShape(
            AexisGraphSession owner,
            string name,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            List<IDisposable> tempOwned,
            out AexisGraphSession.BufferShape shape)
        {
            if (bufferViews.TryGetValue(name, out var view) && view != null)
            {
                shape = AexisGraphSession.GetShapeOf(view);
                return true;
            }

            if (textureBlobs.TryGetValue(name, out var tr) && tr != null && tr.texture != null)
            {
                shape = AexisGraphSession.GetTextureShape(textureShapes, tr, name);
                return true;
            }

            owner.GetOrConvertToBuffer(name, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            if (bufferViews.TryGetValue(name, out view) && view != null)
            {
                shape = AexisGraphSession.GetShapeOf(view);
                return true;
            }

            shape = default;
            return false;
        }

        private static bool TryAliasExistingBuffer(
            string bottomName,
            string topName,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, AexisGraphSession.BufferRef> bufferRefs,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            AexisTensorBuffer outView,
            out bool aliased)
        {
            aliased = false;
            if (!bufferBlobs.TryGetValue(bottomName, out var existing) || existing == null)
                return false;
            if (!bufferRefs.TryGetValue(bottomName, out var existingRef) || existingRef == null || !existingRef.owned)
                return false;

            bufferBlobs[topName] = existing;
            bufferRefs[topName] = existingRef;
            existingRef.refs++;
            bufferViews[topName] = outView;
            aliased = true;
            return true;
        }
    }
}
