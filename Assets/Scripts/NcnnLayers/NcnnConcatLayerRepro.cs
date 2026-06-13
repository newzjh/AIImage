using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnConcatLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnConcatLayerRepro() : base(NcnnLayerTypes.Concat, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (CanExecuteTextureConcat(owner, layer, context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews))
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

            var partBuffers = new ComputeBuffer[layer.bottomNames.Length];
            var partViews = new NcnnTensorBuffer[layer.bottomNames.Length];
            for (var i = 0; i < layer.bottomNames.Length; i++)
            {
                partBuffers[i] = owner.GetOrConvertToBuffer(layer.bottomNames[i], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                partViews[i] = NcnnRepro.TryGetBufferView(layer.bottomNames[i], bufferBlobs, bufferViews);
                if (partBuffers[i] == null || partViews[i] == null)
                    throw new InvalidOperationException("Concat source not found: " + layer.name + " | " + layer.bottomNames[i]);
            }

            var firstView = partViews[0];
            var positiveAxis = layer.GetInt(0, 0);
            if (positiveAxis < 0)
                positiveAxis += firstView.dims;
            if (positiveAxis < 0 || positiveAxis >= firstView.dims)
                throw new InvalidOperationException("Concat axis out of range: " + layer.name);

            var tensorAxis = NcnnRepro.MapNcnnAxisToTensorAxis(firstView.dims, positiveAxis);
            var outW = firstView.w;
            var outH = firstView.h;
            var outD = firstView.d;
            var outC = firstView.c;

            for (var i = 0; i < partViews.Length; i++)
            {
                var v = partViews[i];
                if (v.dims != firstView.dims)
                    throw new InvalidOperationException("Concat dims mismatch: " + layer.name);

                if (tensorAxis != 0 && v.w != firstView.w)
                    throw new InvalidOperationException("Concat width mismatch: " + layer.name);
                if (tensorAxis != 1 && v.h != firstView.h)
                    throw new InvalidOperationException("Concat height mismatch: " + layer.name);
                if (firstView.dims == 4 && tensorAxis != 2 && v.d != firstView.d)
                    throw new InvalidOperationException("Concat depth mismatch: " + layer.name);
                var channelAxis = firstView.dims == 4 ? 3 : 2;
                if (tensorAxis != channelAxis && v.c != firstView.c)
                    throw new InvalidOperationException("Concat channel mismatch: " + layer.name);

                if (i == 0)
                    continue;

                if (tensorAxis == 0) outW += v.w;
                else if (tensorAxis == 1) outH += v.h;
                else if (tensorAxis == 2 && firstView.dims == 4) outD += v.d;
                else outC += v.c;
            }

            var concatChannelAxis = firstView.dims == 4 ? 3 : 2;
            if (tensorAxis == concatChannelAxis)
            {
                var fastOutTensor = owner.RentTempTensorBuffer(firstView.dims, outW, outH, outD, outC);
                var dstElementOffset = 0;
                for (var i = 0; i < partViews.Length; i++)
                {
                    owner.Ops.CopyBufPartial(partBuffers[i], 0, fastOutTensor.buffer, partViews[i].elementCount, dstElementOffset);
                    dstElementOffset += partViews[i].elementCount;
                }

                owner.PublishTensorBufferOutput(
                    layer.topNames[0],
                    fastOutTensor,
                    preferTexture: firstView.dims <= 3,
                    textureBlobs,
                    textureShapes,
                    bufferBlobs,
                    bufferRefs,
                    bufferViews,
                    tempOwned);
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            var outCount = outW * outH * outD * outC;
            var outData = new float[outCount];
            var dstAxisOffset = 0;

            for (var i = 0; i < partViews.Length; i++)
            {
                var v = partViews[i];
                var srcData = NcnnRepro.ReadFloatBuffer(partBuffers[i]);

                for (var c = 0; c < v.c; c++)
                {
                    var dstC = tensorAxis == (firstView.dims == 4 ? 3 : 2) ? dstAxisOffset + c : c;
                    for (var z = 0; z < v.d; z++)
                    {
                        var dstDLocal = tensorAxis == 2 && firstView.dims == 4 ? dstAxisOffset + z : z;
                        for (var y = 0; y < v.h; y++)
                        {
                            var dstYLocal = tensorAxis == 1 ? dstAxisOffset + y : y;
                            for (var x = 0; x < v.w; x++)
                            {
                                var dstXLocal = tensorAxis == 0 ? dstAxisOffset + x : x;
                                var srcIndex = (((c * v.d) + z) * v.h + y) * v.w + x;
                                var dstIndex = (((dstC * outD) + dstDLocal) * outH + dstYLocal) * outW + dstXLocal;
                                outData[dstIndex] = srcData[srcIndex];
                            }
                        }
                    }
                }

                if (tensorAxis == 0) dstAxisOffset += v.w;
                else if (tensorAxis == 1) dstAxisOffset += v.h;
                else if (tensorAxis == 2 && firstView.dims == 4) dstAxisOffset += v.d;
                else dstAxisOffset += v.c;
            }

            var outBuf = owner.RentTempBuffer(outCount, sizeof(float));
            outBuf.SetData(outData);
            var outTensor = new NcnnTensorBuffer(outBuf, firstView.dims, outW, outH, outD, outC, true, owner.ReturnTempBuffer);

            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: firstView.dims <= 3 && tensorAxis == (firstView.dims == 3 ? 2 : -1),
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

            if (!TryExecuteTextureConcat(owner, layer, textureBlobs, textureShapes, bufferBlobs, context.bufferViews))
                throw new InvalidOperationException("Concat render-texture path requires exact pack4 concat support: " + layer.name);

            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        private static bool CanExecuteTextureConcat(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            System.Collections.Generic.Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            System.Collections.Generic.Dictionary<string, ComputeBuffer> bufferBlobs,
            System.Collections.Generic.Dictionary<string, NcnnTensorBuffer> bufferViews)
        {
            if (owner == null || layer == null || owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length == 0)
                return false;

            if (!TryResolvePack4ConcatSource(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var firstShape, out var firstWidth, out var firstHeight, out var firstPacks))
                return false;

            var positiveAxis = layer.GetInt(0, 0);
            if (positiveAxis < 0)
                positiveAxis += firstShape.dims;
            if (positiveAxis < 0 || positiveAxis >= firstShape.dims)
                throw new InvalidOperationException("Concat axis out of range: " + layer.name);

            var tensorAxis = NcnnRepro.MapNcnnAxisToTensorAxis(firstShape.dims, positiveAxis);
            var concatChannelAxis = firstShape.dims == 4 ? 3 : 2;
            if ((firstShape.dims != 3 && firstShape.dims != 4) || tensorAxis != concatChannelAxis)
                return false;

            for (var i = 0; i < layer.bottomNames.Length; i++)
            {
                if (!TryResolvePack4ConcatSource(layer.bottomNames[i], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var shape, out var width, out var height, out var packs))
                    return false;

                if (shape.dims != firstShape.dims
                    || shape.w != firstShape.w
                    || shape.h != firstShape.h
                    || shape.d != firstShape.d
                    || width != firstWidth
                    || height != firstHeight
                    || shape.c <= 0
                    || shape.c > packs * 4)
                {
                    return false;
                }

                if (i < layer.bottomNames.Length - 1 && (shape.c & 3) != 0)
                    return false;
            }

            return true;
        }

        private static bool TryResolvePack4ConcatSource(
            string name,
            System.Collections.Generic.Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            System.Collections.Generic.Dictionary<string, ComputeBuffer> bufferBlobs,
            System.Collections.Generic.Dictionary<string, NcnnTensorBuffer> bufferViews,
            out NcnnRepro.BufferShape shape,
            out int width,
            out int height,
            out int packs)
        {
            shape = default;
            width = 0;
            height = 0;
            packs = 0;

            if (NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, name, out var existingTexture, out shape))
            {
                width = existingTexture.width;
                height = existingTexture.height;
                packs = existingTexture.packs;
                return shape.dims == 3 || shape.dims == 4;
            }

            if (bufferBlobs != null
                && bufferViews != null
                && bufferBlobs.TryGetValue(name, out var buffer)
                && buffer != null
                && bufferViews.TryGetValue(name, out var view)
                && view != null)
            {
                shape = new NcnnRepro.BufferShape(view.dims, view.w, view.h, view.d, view.c);
                width = view.w;
                height = view.h;
                packs = Mathf.Max(1, Mathf.CeilToInt(view.c / 4f));
                return (view.dims == 3 || view.dims == 4) && view.c > 0;
            }

            return false;
        }

        private static bool TryExecuteTextureConcat(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            System.Collections.Generic.Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            System.Collections.Generic.Dictionary<string, ComputeBuffer> bufferBlobs,
            System.Collections.Generic.Dictionary<string, NcnnTensorBuffer> bufferViews)
        {
            if (layer.bottomNames == null || layer.bottomNames.Length == 0)
                return false;

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var firstTex, out var firstShape))
                return false;

            var positiveAxis = layer.GetInt(0, 0);
            if (positiveAxis < 0)
                positiveAxis += firstShape.dims;
            if (positiveAxis < 0 || positiveAxis >= firstShape.dims)
                throw new InvalidOperationException("Concat axis out of range: " + layer.name);

            var tensorAxis = NcnnRepro.MapNcnnAxisToTensorAxis(firstShape.dims, positiveAxis);
            var concatChannelAxis = firstShape.dims == 4 ? 3 : 2;
            if ((firstShape.dims != 3 && firstShape.dims != 4) || tensorAxis != concatChannelAxis)
                return false;

            var parts = new NcnnRepro.TensorRef[layer.bottomNames.Length];
            var shapes = new NcnnRepro.BufferShape[layer.bottomNames.Length];
            parts[0] = firstTex;
            shapes[0] = firstShape;

            var outC = firstShape.c;
            for (var i = 0; i < layer.bottomNames.Length; i++)
            {
                if (i > 0)
                {
                    if (!owner.TryGetPack4Texture(layer.bottomNames[i], textureBlobs, textureShapes, bufferBlobs, bufferViews, out parts[i], out shapes[i]))
                        return false;
                    outC += shapes[i].c;
                }

                var shape = shapes[i];
                var tex = parts[i];
                if (shape.dims != firstShape.dims
                    || shape.w != firstShape.w
                    || shape.h != firstShape.h
                    || shape.d != firstShape.d
                    || shape.c <= 0
                    || !NcnnRepro.MatchesPack4TextureStorage(tex, shape))
                {
                    return false;
                }

                if (i < layer.bottomNames.Length - 1 && (shape.c & 3) != 0)
                    return false;
            }

            if (parts.Length == 1)
            {
                textureBlobs[layer.topNames[0]] = firstTex;
                textureShapes[layer.topNames[0]] = firstShape;
                firstTex.refs++;
                return true;
            }

            if (firstShape.dims == 3)
            {
                var outShape = new NcnnRepro.BufferShape(3, firstShape.w, firstShape.h, 1, outC);
                var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
                var outRt = owner.RentTempArray(outShape.w, outShape.h, outPacks, RenderTextureFormat.ARGBHalf);
                var packOffset = 0;
                for (var i = 0; i < parts.Length; i++)
                {
                    owner.Ops.CopyPack4(parts[i].texture, 0, outRt, packOffset, parts[i].packs);
                    packOffset += parts[i].packs;
                }

                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape);
                return true;
            }

            var currentTexture = parts[0].texture;
            var currentShape = shapes[0];
            var currentOwned = false;

            for (var i = 1; i < parts.Length; i++)
            {
                var nextShape = shapes[i];
                var combinedShape = new NcnnRepro.BufferShape(4, currentShape.w, currentShape.h, currentShape.d, currentShape.c + nextShape.c);
                var outSlices = currentShape.d * Mathf.Max(1, Mathf.CeilToInt(combinedShape.c / 4f));
                var outRt = owner.RentTempArray(combinedShape.w, combinedShape.h, outSlices, NcnnRepro.ResolveTensorTextureFormat(4));
                owner.Ops.ConcatPack4Cdhw(
                    currentTexture,
                    parts[i].texture,
                    combinedShape.w,
                    combinedShape.h,
                    combinedShape.d,
                    currentShape.c,
                    nextShape.c,
                    combinedShape.c,
                    outRt);

                if (currentOwned && currentTexture != null)
                    owner.ReturnTempArray(currentTexture);

                currentTexture = outRt;
                currentShape = combinedShape;
                currentOwned = true;
            }

            NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], currentTexture, currentShape);
            return true;
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var parts = new NcnnRepro.CmdTensorRef[layer.bottomNames.Length];
            var partShapes = new NcnnRepro.BufferShape[layer.bottomNames.Length];
            for (var i = 0; i < layer.bottomNames.Length; i++)
            {
                parts[i] = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[i]);
                partShapes[i] = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[i]);
            }

            var firstShape = partShapes[0];
            var positiveAxis = layer.GetInt(0, 0);
            if (positiveAxis < 0)
                positiveAxis += firstShape.dims;
            if (positiveAxis < 0 || positiveAxis >= firstShape.dims)
                throw new InvalidOperationException("Concat axis out of range: " + layer.name);

            var tensorAxis = NcnnRepro.MapNcnnAxisToTensorAxis(firstShape.dims, positiveAxis);
            var outW = firstShape.w;
            var outH = firstShape.h;
            var outD = firstShape.d;
            var outC = firstShape.c;
            var concatChannelAxis = firstShape.dims == 4 ? 3 : 2;
            var canUseExactPack4 = (firstShape.dims == 3 || firstShape.dims == 4) && tensorAxis == concatChannelAxis;

            for (var i = 0; i < partShapes.Length; i++)
            {
                var shape = partShapes[i];
                if (shape.dims != firstShape.dims)
                    throw new InvalidOperationException("Concat dims mismatch: " + layer.name);

                if (tensorAxis != 0 && shape.w != firstShape.w)
                    throw new InvalidOperationException("Concat width mismatch: " + layer.name);
                if (tensorAxis != 1 && shape.h != firstShape.h)
                    throw new InvalidOperationException("Concat height mismatch: " + layer.name);
                if (firstShape.dims == 4 && tensorAxis != 2 && shape.d != firstShape.d)
                    throw new InvalidOperationException("Concat depth mismatch: " + layer.name);
                var channelAxis = firstShape.dims == 4 ? 3 : 2;
                if (tensorAxis != channelAxis && shape.c != firstShape.c)
                    throw new InvalidOperationException("Concat channel mismatch: " + layer.name);

                if (i > 0)
                {
                    if (tensorAxis == 0) outW += shape.w;
                    else if (tensorAxis == 1) outH += shape.h;
                    else if (tensorAxis == 2 && firstShape.dims == 4) outD += shape.d;
                    else outC += shape.c;
                }

                var supportsPack4Storage =
                    shape.dims == 3
                        ? shape.d == 1
                        : shape.dims == 4;
                if (!supportsPack4Storage
                    || !NcnnRepro.MatchesPack4TextureStorage(parts[i], shape)
                    || (i < partShapes.Length - 1 && (shape.c & 3) != 0))
                {
                    canUseExactPack4 = false;
                }
            }

            var outShape = new NcnnRepro.BufferShape(firstShape.dims, outW, outH, outD, outC);
            if (layer.bottomNames.Length == 1)
            {
                blobs[layer.topNames[0]] = parts[0];
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
                parts[0].refs++;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            if (canUseExactPack4)
            {
                var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
                if (outShape.dims == 3)
                {
                    var outArr = owner.RentTempArray(cmd, outShape.w, outShape.h, outPacks, RenderTextureFormat.ARGBHalf);
                    var packOffset = 0;
                    for (var i = 0; i < parts.Length; i++)
                    {
                        owner.Ops.CopyPack4(cmd, parts[i].texture, 0, outArr, packOffset, parts[i].packs);
                        packOffset += parts[i].packs;
                    }

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
                        storageShape = outShape
                    };
                }
                else
                {
                    var current = parts[0];
                    var currentShape = partShapes[0];
                    var currentOwned = false;
                    for (var i = 1; i < parts.Length; i++)
                    {
                        var next = parts[i];
                        var nextShape = partShapes[i];
                        var combinedChannels = currentShape.c + nextShape.c;
                        var outArr = owner.RentTempArray(cmd, outShape.w, outShape.h, outShape.d * Mathf.Max(1, Mathf.CeilToInt(combinedChannels / 4f)), RenderTextureFormat.ARGBHalf);
                        owner.Ops.ConcatPack4Cdhw(
                            cmd,
                            current.texture,
                            next.texture,
                            outShape.w,
                            outShape.h,
                            outShape.d,
                            currentShape.c,
                            nextShape.c,
                            combinedChannels,
                            outArr);

                        if (currentOwned && current.texture != null)
                            owner.ReturnTempArray(cmd, current.texture);

                        current = new NcnnRepro.CmdTensorRef
                        {
                            texture = outArr,
                            width = outShape.w,
                            height = outShape.h,
                            packs = Mathf.Max(1, Mathf.CeilToInt(combinedChannels / 4f)),
                            refs = 1,
                            owned = true
                        };
                        currentShape = new NcnnRepro.BufferShape(4, outShape.w, outShape.h, outShape.d, combinedChannels);
                        currentOwned = true;
                    }

                    current.hasLogicalShape = true;
                    current.logicalShape = outShape;
                    current.hasStorageShape = true;
                    current.storageShape = outShape;
                    blobs[layer.topNames[0]] = current;
                }
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
            }
            else
            {
                owner.DebugLog?.Invoke(
                    "[CmdPlaceholder][Concat]"
                    + " | layer=" + layer.name
                    + " | axis=" + positiveAxis.ToString()
                    + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                    + " | parts=" + partShapes.Length.ToString());
                if (owner.DisallowBufferAccess || owner.DisallowBufferOutputs || owner.DisallowBufferToTextureMaterialization)
                {
                    throw new InvalidOperationException(
                        "pack4-only guard: command-buffer Concat placeholder disallowed"
                        + " | layer=" + layer.name
                        + " | axis=" + positiveAxis
                        + " | outShape=" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
                }
                owner.PublishCmdPlaceholder(cmd, layer.topNames[0], outShape, blobs, shapes);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
