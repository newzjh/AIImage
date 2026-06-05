using System;
using System.Collections.Generic;
using UnityEngine;

namespace NcnnCompute
{
    public sealed class NcnnCropLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnCropLayerRepro() : base(NcnnLayerTypes.Crop, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Crop source not found: " + layer.name);

            var srcShape = NcnnRepro.GetShapeOf(srcView);
            NcnnRepro.CropRoi roi;
            var hasSliceArrays = (layer.GetInts(-23309, null)?.Length ?? 0) > 0 && (layer.GetInts(-23310, null)?.Length ?? 0) > 0;
            var hasSliceExpr = !string.IsNullOrWhiteSpace(layer.GetString(19, null)) && !string.IsNullOrWhiteSpace(layer.GetString(20, null));

            if (hasSliceArrays || hasSliceExpr)
            {
                var bottomShapes = new List<NcnnRepro.BufferShape>(layer.bottomNames.Length) { srcShape };
                if (layer.bottomNames.Length > 1 && TryGetBottomShape(owner, layer.bottomNames[1], textureBlobs, textureShapes, bufferBlobs, bufferViews, tempOwned, out var secondaryShape))
                    bottomShapes.Add(secondaryShape);
                roi = NcnnRepro.ResolveCropRoi(srcShape, layer, bottomShapes);
            }
            else if (layer.bottomNames.Length > 1)
            {
                if (layer.GetInt(0, 0) == -233)
                {
                    var refBuf = owner.GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    if (refBuf == null)
                        throw new InvalidOperationException("Crop param_data source not found: " + layer.name);
                    var refData = NcnnRepro.ReadFloatBuffer(refBuf);
                    var paramData = new int[refData.Length];
                    for (var i = 0; i < refData.Length; i++)
                        paramData[i] = Mathf.RoundToInt(refData[i]);
                    roi = NcnnRepro.ResolveCropRoi(srcShape, paramData, layer);
                }
                else
                {
                    if (!TryGetBottomShape(owner, layer.bottomNames[1], textureBlobs, textureShapes, bufferBlobs, bufferViews, tempOwned, out var referenceShape))
                        throw new InvalidOperationException("Crop reference shape missing: " + layer.name);
                    roi = NcnnRepro.ResolveCropRoi(srcShape, referenceShape, layer);
                }
            }
            else
            {
                var bottomShapes = new List<NcnnRepro.BufferShape>(1) { srcShape };
                roi = NcnnRepro.ResolveCropRoi(srcShape, layer, bottomShapes);
            }

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
                bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, cropResult.dims, cropResult.w, cropResult.h, cropResult.d, cropResult.c, false);
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

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            owner.CopyCmdTensor(cmd, src, layer.topNames[0], blobs, src.width, src.height, src.packs);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
        }

        private static bool TryGetBottomShape(
            NcnnRepro owner,
            string name,
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            List<IDisposable> tempOwned,
            out NcnnRepro.BufferShape shape)
        {
            if (bufferViews.TryGetValue(name, out var view) && view != null)
            {
                shape = NcnnRepro.GetShapeOf(view);
                return true;
            }

            if (textureBlobs.TryGetValue(name, out var tr) && tr != null && tr.texture != null)
            {
                shape = NcnnRepro.GetTextureShape(textureShapes, tr, name);
                return true;
            }

            owner.GetOrConvertToBuffer(name, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            if (bufferViews.TryGetValue(name, out view) && view != null)
            {
                shape = NcnnRepro.GetShapeOf(view);
                return true;
            }

            shape = default;
            return false;
        }

        private static bool TryAliasExistingBuffer(
            string bottomName,
            string topName,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnRepro.BufferRef> bufferRefs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            NcnnTensorBuffer outView,
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
