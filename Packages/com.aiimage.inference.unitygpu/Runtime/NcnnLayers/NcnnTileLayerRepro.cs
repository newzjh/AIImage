using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnTileLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnTileLayerRepro() : base(NcnnLayerTypes.Tile, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (TryResolveSourceShape(owner, layer, context, out var srcShape)
                && TryResolveTileSpec(layer, srcShape, out var spec)
                && (spec.isPassthrough || owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out _,
                    out _)))
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

            var axis = layer.GetInt(0, 0);
            var tiles = layer.GetInt(1, 1);
            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Tile source not found: " + layer.name);

            if (axis < 0)
                axis += srcView.dims;
            if (axis < 0 || axis >= srcView.dims)
                throw new InvalidOperationException("Tile axis out of range: " + layer.name);

            var tensorAxis = NcnnRepro.MapNcnnAxisToTensorAxis(srcView.dims, axis);
            var outW = srcView.w;
            var outH = srcView.h;
            var outD = srcView.d;
            var outC = srcView.c;
            if (tensorAxis == 0) outW *= tiles;
            else if (tensorAxis == 1) outH *= tiles;
            else if (tensorAxis == 2 && srcView.dims == 4) outD *= tiles;
            else if (tensorAxis == 2 || tensorAxis == 3) outC *= tiles;

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, outW, outH, outD, outC);
            owner.Ops.Tile(srcBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, tensorAxis, tiles, outW, outH, outD, outC, outTensor.buffer);

            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: srcView.dims <= 3,
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
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!TryResolveSourceShape(owner, layer, context, out var srcShape))
                throw new InvalidOperationException("Tile source shape not found: " + layer.name);
            if (!TryResolveTileSpec(layer, srcShape, out var spec))
                throw new InvalidOperationException("Tile parameters are unsupported: " + layer.name);

            if (spec.isPassthrough)
            {
                if (textureBlobs.TryGetValue(layer.bottomNames[0], out var passthroughTex) && passthroughTex != null && passthroughTex.texture != null)
                {
                    var storageShape = NcnnRepro.GetTextureStorageShape(passthroughTex, srcShape);
                    textureBlobs[layer.topNames[0]] = NcnnRepro.CreateTextureAlias(passthroughTex, spec.outShape, storageShape);
                    textureShapes[layer.topNames[0]] = spec.outShape;
                }

                if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var passthroughBuf) && passthroughBuf != null)
                {
                    bufferBlobs[layer.topNames[0]] = passthroughBuf;
                    if (bufferRefs.TryGetValue(layer.bottomNames[0], out var passthroughRef) && passthroughRef != null)
                    {
                        bufferRefs[layer.topNames[0]] = passthroughRef;
                        passthroughRef.refs++;
                    }
                    else
                    {
                        bufferRefs[layer.topNames[0]] = owner.NewBufferRef(passthroughBuf, owned: false);
                    }

                    if (bufferViews.TryGetValue(layer.bottomNames[0], out var passthroughView) && passthroughView != null)
                        bufferViews[layer.topNames[0]] = passthroughView.Reshape(spec.outShape.dims, spec.outShape.w, spec.outShape.h, spec.outShape.d, spec.outShape.c);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var src, out srcShape))
                throw new InvalidOperationException("Tile render-texture path requires texture input: " + layer.name);

            var srcStorageShape = NcnnRepro.GetTextureStorageShape(src, srcShape);
            if (NcnnRepro.IsStrictLinearMatTexture(src))
            {
                var outStorageShape = NcnnRepro.ResolveLinearMatStorageShape(spec.outShape);
                var output = owner.RentTempMat(outStorageShape.w, outStorageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
                owner.Ops.TileLinearMat(src.texture, srcShape, srcStorageShape, spec.outShape, outStorageShape, spec.repeats, output);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], output, spec.outShape, outStorageShape);
            }
            else
            {
                if (src.texture.dimension != TextureDimension.Tex2DArray || !NcnnRepro.BufferShapeEquals(srcShape, srcStorageShape))
                    throw new InvalidOperationException(
                        "Tile render-texture path requires direct pack4 storage"
                        + " | layer=" + layer.name
                        + " | logical=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                        + " | storage=d" + srcStorageShape.dims + ":" + srcStorageShape.w + "x" + srcStorageShape.h + "x" + srcStorageShape.d + "x" + srcStorageShape.c);

                var output = owner.RentTempArray(
                    spec.outShape.w,
                    spec.outShape.dims >= 2 ? spec.outShape.h : 1,
                    ResolveTileArrayDepth(spec.outShape),
                    NcnnRepro.ResolveTensorTextureFormat(spec.outShape.dims));
                owner.Ops.TilePack4(src.texture, srcShape, spec.outShape, spec.repeats, output);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], output, spec.outShape, spec.outShape);
            }

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
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
            if (!TryResolveTileSpec(layer, srcShape, out var spec))
                throw new InvalidOperationException("Tile parameters are unsupported: " + layer.name);
            if (spec.isPassthrough)
            {
                var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
                blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorAlias(src, spec.outShape, storageShape);
                if (shapes != null)
                    shapes[layer.topNames[0]] = spec.outShape;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            var srcStorageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
            if (NcnnRepro.IsStrictLinearMatTexture(src))
            {
                var outStorageShape = NcnnRepro.ResolveLinearMatStorageShape(spec.outShape);
                var output = owner.RentTempMat(cmd, outStorageShape.w, outStorageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
                owner.Ops.TileLinearMat(cmd, src.texture, srcShape, srcStorageShape, spec.outShape, outStorageShape, spec.repeats, output);
                blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(output, spec.outShape, outStorageShape, owned: true);
            }
            else
            {
                if (src.texture.dimension != TextureDimension.Tex2DArray || !NcnnRepro.BufferShapeEquals(srcShape, srcStorageShape))
                    throw new InvalidOperationException(
                        "Tile command-buffer path requires direct pack4 storage"
                        + " | layer=" + layer.name
                        + " | logical=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                        + " | storage=d" + srcStorageShape.dims + ":" + srcStorageShape.w + "x" + srcStorageShape.h + "x" + srcStorageShape.d + "x" + srcStorageShape.c);

                var output = owner.RentTempArray(
                    cmd,
                    spec.outShape.w,
                    spec.outShape.dims >= 2 ? spec.outShape.h : 1,
                    ResolveTileArrayDepth(spec.outShape),
                    NcnnRepro.ResolveTensorTextureFormat(spec.outShape.dims));
                owner.Ops.TilePack4(cmd, src.texture, srcShape, spec.outShape, spec.repeats, output);
                blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(output, spec.outShape, spec.outShape, owned: true);
            }

            if (shapes != null)
                shapes[layer.topNames[0]] = spec.outShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private readonly struct TileSpec
        {
            public readonly NcnnRepro.BufferShape outShape;
            public readonly Vector4Int repeats;
            public readonly bool isPassthrough;

            public TileSpec(NcnnRepro.BufferShape outShape, Vector4Int repeats, bool isPassthrough)
            {
                this.outShape = outShape;
                this.repeats = repeats;
                this.isPassthrough = isPassthrough;
            }
        }

        private static bool TryResolveSourceShape(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnLayerBufferContext context,
            out NcnnRepro.BufferShape shape)
        {
            shape = default;
            if (context.textureBlobs != null
                && context.textureBlobs.TryGetValue(layer.bottomNames[0], out var tex)
                && tex != null
                && tex.texture != null)
            {
                shape = NcnnRepro.GetTextureShape(context.textureShapes, tex, layer.bottomNames[0]);
                return shape.dims >= 1 && shape.dims <= 4;
            }

            var view = NcnnRepro.TryGetBufferView(layer.bottomNames[0], context.bufferBlobs, context.bufferViews);
            if (view != null)
            {
                shape = new NcnnRepro.BufferShape(view.dims, view.w, view.h, view.d, view.c);
                return shape.dims >= 1 && shape.dims <= 4;
            }

            return false;
        }

        private static bool TryResolveTileSpec(NcnnParamModel.Layer layer, NcnnRepro.BufferShape srcShape, out TileSpec spec)
        {
            spec = default;
            if (srcShape.dims < 1 || srcShape.dims > 4)
                return false;

            var repeatW = 1;
            var repeatH = 1;
            var repeatD = 1;
            var repeatC = 1;
            var repeats = ResolveRepeats(layer);
            var repeatsNum = repeats?.Length ?? 0;

            if (repeatsNum == 0)
            {
                var axis = layer.GetInt(0, 0);
                var tiles = Mathf.Max(1, layer.GetInt(1, 1));
                if (axis < 0)
                    axis += srcShape.dims;
                if (axis < 0 || axis >= srcShape.dims)
                    return false;

                if (srcShape.dims == 1)
                    repeatW = tiles;
                else if (srcShape.dims == 2)
                {
                    if (axis == 0) repeatH = tiles;
                    else repeatW = tiles;
                }
                else if (srcShape.dims == 3)
                {
                    if (axis == 0) repeatC = tiles;
                    else if (axis == 1) repeatH = tiles;
                    else repeatW = tiles;
                }
                else
                {
                    if (axis == 0) repeatC = tiles;
                    else if (axis == 1) repeatD = tiles;
                    else if (axis == 2) repeatH = tiles;
                    else repeatW = tiles;
                }
            }
            else
            {
                if (repeatsNum == 1)
                {
                    repeatW = Mathf.Max(1, repeats[0]);
                }
                else if (repeatsNum == 2)
                {
                    repeatH = Mathf.Max(1, repeats[0]);
                    repeatW = Mathf.Max(1, repeats[1]);
                }
                else if (repeatsNum == 3)
                {
                    if (srcShape.dims == 4)
                    {
                        repeatD = Mathf.Max(1, repeats[0]);
                        repeatH = Mathf.Max(1, repeats[1]);
                        repeatW = Mathf.Max(1, repeats[2]);
                    }
                    else
                    {
                        repeatC = Mathf.Max(1, repeats[0]);
                        repeatH = Mathf.Max(1, repeats[1]);
                        repeatW = Mathf.Max(1, repeats[2]);
                    }
                }
                else if (repeatsNum == 4)
                {
                    repeatC = Mathf.Max(1, repeats[0]);
                    repeatD = Mathf.Max(1, repeats[1]);
                    repeatH = Mathf.Max(1, repeats[2]);
                    repeatW = Mathf.Max(1, repeats[3]);
                }
                else
                {
                    return false;
                }
            }

            var outDims = Mathf.Max(srcShape.dims, repeatsNum);
            var outW = Mathf.Max(1, srcShape.w) * repeatW;
            var outH = Mathf.Max(1, srcShape.h) * repeatH;
            var outD = Mathf.Max(1, srcShape.d) * repeatD;
            var outC = Mathf.Max(1, srcShape.c) * repeatC;
            NcnnRepro.BufferShape outShape;
            if (outDims == 1)
                outShape = new NcnnRepro.BufferShape(1, outW, 1, 1, 1);
            else if (outDims == 2)
                outShape = new NcnnRepro.BufferShape(2, outW, outH, 1, 1);
            else if (outDims == 3)
                outShape = new NcnnRepro.BufferShape(3, outW, outH, 1, outC);
            else
                outShape = new NcnnRepro.BufferShape(4, outW, outH, outD, outC);

            var sameShape = NcnnRepro.BufferShapeEquals(srcShape, outShape);
            var allRepeatsOne = repeatW == 1 && repeatH == 1 && repeatD == 1 && repeatC == 1;
            var canAlias = allRepeatsOne && (repeatsNum == 0 || sameShape);
            spec = new TileSpec(outShape, new Vector4Int(repeatW, repeatH, repeatD, repeatC), canAlias);
            return true;
        }

        private static int[] ResolveRepeats(NcnnParamModel.Layer layer)
        {
            var repeats = layer.GetInts(-23302, null);
            if (repeats == null || repeats.Length == 0)
                repeats = layer.GetInts(2, null);
            if (repeats == null || repeats.Length == 0)
                repeats = layer.GetInts(-23330, null);
            if (repeats == null || repeats.Length == 0)
                repeats = layer.GetInts(30, null);
            return repeats;
        }

        private static int ResolveTileArrayDepth(NcnnRepro.BufferShape shape)
        {
            var packs = Mathf.Max(1, Mathf.CeilToInt(Mathf.Max(1, shape.c) / 4f));
            return shape.dims == 4 ? Mathf.Max(1, shape.d) * packs : packs;
        }
    }
}
