using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    internal static class AexisShapeIndexLayerUtil
    {
        public readonly struct RenderLinearInput
        {
            public readonly RenderTexture texture;
            public readonly AexisGraphSession.BufferShape logicalShape;
            public readonly AexisGraphSession.BufferShape storageShape;

            public RenderLinearInput(RenderTexture texture, AexisGraphSession.BufferShape logicalShape, AexisGraphSession.BufferShape storageShape)
            {
                this.texture = texture;
                this.logicalShape = logicalShape;
                this.storageShape = storageShape;
            }
        }

        public readonly struct CmdLinearInput
        {
            public readonly ComputeTexture texture;
            public readonly AexisGraphSession.BufferShape logicalShape;
            public readonly AexisGraphSession.BufferShape storageShape;

            public CmdLinearInput(ComputeTexture texture, AexisGraphSession.BufferShape logicalShape, AexisGraphSession.BufferShape storageShape)
            {
                this.texture = texture;
                this.logicalShape = logicalShape;
                this.storageShape = storageShape;
            }
        }

        public static int ElementCount(AexisGraphSession.BufferShape shape)
        {
            return Mathf.Max(1, shape.w) * Mathf.Max(1, shape.h) * Mathf.Max(1, shape.d) * Mathf.Max(1, shape.c);
        }

        public static int[] ToAxisSizes(AexisGraphSession.BufferShape shape)
        {
            return shape.dims switch
            {
                1 => new[] { Mathf.Max(1, shape.w) },
                2 => new[] { Mathf.Max(1, shape.h), Mathf.Max(1, shape.w) },
                3 => new[] { Mathf.Max(1, shape.c), Mathf.Max(1, shape.h), Mathf.Max(1, shape.w) },
                4 => new[] { Mathf.Max(1, shape.c), Mathf.Max(1, shape.d), Mathf.Max(1, shape.h), Mathf.Max(1, shape.w) },
                _ => throw new InvalidOperationException("Unsupported tensor rank: " + shape.dims)
            };
        }

        public static AexisGraphSession.BufferShape FromAxisSizes(int[] sizes)
        {
            if (sizes == null || sizes.Length == 0)
                return new AexisGraphSession.BufferShape(1, 1, 1, 1, 1);

            if (sizes.Length > 4)
                throw new InvalidOperationException("Texture-backed Sentis ops currently support rank <= 4, got rank=" + sizes.Length);
            for (var i = 0; i < sizes.Length; i++)
            {
                if (sizes[i] <= 0)
                    throw new InvalidOperationException("Texture-backed Sentis ops do not support zero/negative axis sizes: axis=" + i + " size=" + sizes[i]);
            }

            return sizes.Length switch
            {
                1 => new AexisGraphSession.BufferShape(1, sizes[0], 1, 1, 1),
                2 => new AexisGraphSession.BufferShape(2, sizes[1], sizes[0], 1, 1),
                3 => new AexisGraphSession.BufferShape(3, sizes[2], sizes[1], 1, sizes[0]),
                4 => new AexisGraphSession.BufferShape(4, sizes[3], sizes[2], sizes[1], sizes[0]),
                _ => new AexisGraphSession.BufferShape(1, 1, 1, 1, 1)
            };
        }

        public static int NormalizeAxis(int axis, int rank, string layerName)
        {
            if (axis < 0)
                axis += rank;
            if (axis < 0 || axis >= rank)
                throw new InvalidOperationException("Axis out of range"
                    + " | layer=" + layerName
                    + " | axis=" + axis
                    + " | rank=" + rank);
            return axis;
        }

        public static void RequireBottoms(AexisGraphModel.Layer layer, int minCount)
        {
            if (layer?.bottomNames == null || layer.bottomNames.Length < minCount)
                throw new InvalidOperationException("Sentis texture path input count mismatch"
                    + " | layer=" + (layer?.name ?? string.Empty)
                    + " | type=" + (layer?.typeName ?? string.Empty)
                    + " | expected>=" + minCount
                    + " | actual=" + (layer?.bottomNames == null ? 0 : layer.bottomNames.Length));
        }

        public static void RequireTops(AexisGraphModel.Layer layer, int minCount)
        {
            if (layer?.topNames == null || layer.topNames.Length < minCount)
                throw new InvalidOperationException("Sentis texture path output count mismatch"
                    + " | layer=" + (layer?.name ?? string.Empty)
                    + " | type=" + (layer?.typeName ?? string.Empty)
                    + " | expected>=" + minCount
                    + " | actual=" + (layer?.topNames == null ? 0 : layer.topNames.Length));
        }

        public static bool TryGetInt(AexisGraphModel.Layer layer, string name, out int value)
        {
            value = 0;
            if (layer?.stringParams == null || string.IsNullOrEmpty(name) || !layer.stringParams.TryGetValue(name, out var raw))
                return false;
            raw = CleanToken(raw);
            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
        }

        public static bool TryGetInt(AexisGraphModel.Layer layer, int key, out int value)
        {
            value = 0;
            if (layer?.intParams == null || !layer.intParams.TryGetValue(key, out var raw))
                return false;
            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
        }

        public static int GetInt(AexisGraphModel.Layer layer, int key, string name, int defaultValue)
        {
            if (TryGetInt(layer, name, out var named))
                return named;
            return TryGetInt(layer, key, out var keyed) ? keyed : defaultValue;
        }

        public static bool TryGetFloat(AexisGraphModel.Layer layer, string name, out float value)
        {
            value = 0f;
            if (layer?.stringParams == null || string.IsNullOrEmpty(name) || !layer.stringParams.TryGetValue(name, out var raw))
                return false;
            raw = CleanToken(raw);
            return float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
        }

        public static bool TryGetFloat(AexisGraphModel.Layer layer, int key, out float value)
        {
            value = 0f;
            if (layer?.intParams == null || !layer.intParams.TryGetValue(key, out var raw))
                return false;
            return float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
        }

        public static float GetFloat(AexisGraphModel.Layer layer, int key, string name, float defaultValue)
        {
            if (TryGetFloat(layer, name, out var named))
                return named;
            return TryGetFloat(layer, key, out var keyed) ? keyed : defaultValue;
        }

        public static bool TryGetShapeParam(AexisGraphModel.Layer layer, out int[] shape)
        {
            shape = null;
            if (layer == null)
                return false;

            var names = new[] { "shape", "out_shape", "output_shape", "dims", "sizes" };
            for (var i = 0; i < names.Length; i++)
            {
                if (layer.stringParams != null
                    && layer.stringParams.TryGetValue(names[i], out var raw)
                    && TryParseIntArray(raw, out shape)
                    && shape.Length > 0)
                {
                    return true;
                }
            }

            var keys = new[] { -23300, -23301, -23302, -23303, -23330, 30, 2, 1, 0 };
            for (var i = 0; i < keys.Length; i++)
            {
                if (layer.intParams != null
                    && layer.intParams.TryGetValue(keys[i], out var raw)
                    && raw.IndexOf(',', StringComparison.Ordinal) >= 0
                    && TryParseIntArray(raw, out shape)
                    && shape.Length > 0)
                {
                    return true;
                }
            }

            return false;
        }

        public static bool TryParseIntArray(string raw, out int[] values)
        {
            values = null;
            if (string.IsNullOrWhiteSpace(raw))
                return false;

            raw = raw.Trim().Trim('"', '\'', '[', ']', '(', ')');
            var parts = raw.Split(new[] { ',', ';', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0)
                return false;

            var start = 0;
            if (parts.Length >= 2
                && int.TryParse(CleanToken(parts[0]), NumberStyles.Integer, CultureInfo.InvariantCulture, out var count)
                && count == parts.Length - 1)
            {
                start = 1;
            }

            var parsed = new int[parts.Length - start];
            for (var i = start; i < parts.Length; i++)
            {
                if (!int.TryParse(CleanToken(parts[i]), NumberStyles.Integer, CultureInfo.InvariantCulture, out parsed[i - start]))
                    return false;
            }

            values = parsed;
            return true;
        }

        public static AexisGraphSession.BufferShape GetInputShape(
            AexisGraphModel.Layer layer,
            AexisLayerBufferContext context,
            string bottomName)
        {
            if (AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, bottomName, out var tex, out var shape))
                return shape;

            var view = AexisGraphSession.TryGetBufferView(bottomName, context.bufferBlobs, context.bufferViews);
            if (view != null)
                return new AexisGraphSession.BufferShape(view.dims, view.w, view.h, view.d, view.c);

            throw new InvalidOperationException("Sentis layer input shape not found"
                + " | layer=" + layer.name
                + " | bottom=" + bottomName);
        }

        public static RenderLinearInput GetRenderLinearInput(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisLayerBufferContext context,
            string bottomName,
            List<RenderTexture> temps)
        {
            if (!AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, bottomName, out var src, out var shape)
                || src == null
                || src.texture == null)
            {
                throw new InvalidOperationException("Sentis texture path requires existing texture input"
                    + " | layer=" + layer.name
                    + " | bottom=" + bottomName);
            }

            var storage = AexisGraphSession.GetTextureStorageShape(src, shape);
            if (AexisGraphSession.IsStrictLinearMatTexture(src))
                return new RenderLinearInput(src.texture, shape, storage);

            if (src.texture.dimension != TextureDimension.Tex2DArray || !AexisGraphSession.MatchesPack4TextureStorage(src, shape))
            {
                throw new InvalidOperationException("Sentis texture path requires LinearMat or direct pack4 input"
                    + " | layer=" + layer.name
                    + " | bottom=" + bottomName
                    + " | logical=d" + shape.dims + ":" + shape.w + "x" + shape.h + "x" + shape.d + "x" + shape.c
                    + " | storage=d" + storage.dims + ":" + storage.w + "x" + storage.h + "x" + storage.d + "x" + storage.c);
            }

            var linearStorage = AexisGraphSession.ResolveLinearMatStorageShape(shape);
            var linear = owner.RentTempMat(linearStorage.w, linearStorage.h, AexisGraphSession.ResolveLinearMatTextureFormat());
            owner.Ops.ReshapePack4ToLinearMat(src.texture, shape.w, shape.h, shape.d, shape.c, shape.dims, linear);
            temps?.Add(linear);
            return new RenderLinearInput(linear, shape, linearStorage);
        }

        public static CmdLinearInput GetCmdLinearInput(
            AexisGraphSession owner,
            CommandBuffer cmd,
            AexisGraphModel.Layer layer,
            AexisLayerCommandBufferContext context,
            string bottomName,
            List<ComputeTexture> temps)
        {
            var src = AexisGraphSession.GetCmdTensor(context.blobs, bottomName);
            var shape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, bottomName);
            var storage = AexisGraphSession.GetCmdStorageShape(src, shape);
            if (AexisGraphSession.IsStrictLinearMatTexture(src))
                return new CmdLinearInput(src.texture, shape, storage);

            if (src.texture.dimension != TextureDimension.Tex2DArray || !AexisGraphSession.MatchesPack4TextureStorage(src, shape))
            {
                throw new InvalidOperationException("Sentis command-buffer path requires LinearMat or direct pack4 input"
                    + " | layer=" + layer.name
                    + " | bottom=" + bottomName
                    + " | logical=d" + shape.dims + ":" + shape.w + "x" + shape.h + "x" + shape.d + "x" + shape.c
                    + " | storage=d" + storage.dims + ":" + storage.w + "x" + storage.h + "x" + storage.d + "x" + storage.c);
            }

            var linearStorage = AexisGraphSession.ResolveLinearMatStorageShape(shape);
            var linear = owner.RentTempMat(cmd, linearStorage.w, linearStorage.h, AexisGraphSession.ResolveLinearMatTextureFormat());
            owner.Ops.ReshapePack4ToLinearMat(cmd, src.texture, shape.w, shape.h, shape.d, shape.c, shape.dims, linear);
            temps?.Add(linear);
            return new CmdLinearInput(linear, shape, linearStorage);
        }

        public static void PublishRenderLinear(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisLayerBufferContext context,
            AexisGraphSession.BufferShape logicalShape,
            Action<RenderTexture, AexisGraphSession.BufferShape> fill)
        {
            RequireTops(layer, 1);
            var storage = AexisGraphSession.ResolveLinearMatStorageShape(logicalShape);
            var output = owner.RentTempMat(storage.w, storage.h, AexisGraphSession.ResolveLinearMatTextureFormat());
            fill(output, storage);
            AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, logicalShape, storage);
        }

        public static void PublishCmdLinear(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisLayerCommandBufferContext context,
            AexisGraphSession.BufferShape logicalShape,
            Action<ComputeTexture, AexisGraphSession.BufferShape> fill)
        {
            RequireTops(layer, 1);
            var storage = AexisGraphSession.ResolveLinearMatStorageShape(logicalShape);
            var output = owner.RentTempMat(context.commandBuffer, storage.w, storage.h, AexisGraphSession.ResolveLinearMatTextureFormat());
            fill(output, storage);
            context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, logicalShape, storage, owned: true);
            if (context.shapes != null)
                context.shapes[layer.topNames[0]] = logicalShape;
        }

        public static AexisGraphSession.BufferShape BroadcastShapes(params AexisGraphSession.BufferShape[] shapes)
        {
            var rank = 1;
            for (var i = 0; i < shapes.Length; i++)
                rank = Mathf.Max(rank, shapes[i].dims);

            var result = new int[rank];
            for (var i = 0; i < rank; i++)
                result[i] = 1;

            for (var si = 0; si < shapes.Length; si++)
            {
                var axes = ToAxisSizes(shapes[si]);
                var offset = rank - axes.Length;
                for (var ai = 0; ai < axes.Length; ai++)
                {
                    var dst = offset + ai;
                    var value = Mathf.Max(1, axes[ai]);
                    if (result[dst] == 1)
                        result[dst] = value;
                    else if (value != 1 && result[dst] != value)
                        throw new InvalidOperationException("Sentis broadcast shape mismatch.");
                }
            }

            return FromAxisSizes(result);
        }

        public static AexisGraphSession.BufferShape ResolveExpandShape(AexisGraphSession.BufferShape inputShape, int[] requested)
        {
            if (requested == null || requested.Length == 0)
                throw new InvalidOperationException("Expand static shape is missing.");
            if (requested.Length > 4)
                throw new InvalidOperationException("Expand rank > 4 is not supported on texture path.");

            var inAxes = ToAxisSizes(inputShape);
            var rank = Mathf.Max(inAxes.Length, requested.Length);
            var outAxes = new int[rank];
            for (var axis = 0; axis < rank; axis++)
            {
                var inIndex = axis - (rank - inAxes.Length);
                var reqIndex = axis - (rank - requested.Length);
                var inSize = inIndex >= 0 ? Mathf.Max(1, inAxes[inIndex]) : 1;
                var reqSize = reqIndex >= 0 ? requested[reqIndex] : 1;
                if (reqSize == 0)
                    throw new InvalidOperationException("Expand requested shape contains zero, which is not supported on the texture path.");
                if (reqSize < 0)
                    reqSize = inSize;
                if (inSize != 1 && reqSize != inSize)
                    throw new InvalidOperationException("Expand requested shape is not broadcast-compatible.");
                outAxes[axis] = Mathf.Max(inSize, reqSize);
            }

            return FromAxisSizes(outAxes);
        }

        public static AexisGraphSession.BufferShape ResolveGatherShape(AexisGraphSession.BufferShape dataShape, AexisGraphSession.BufferShape indicesShape, int axis)
        {
            var dataAxes = ToAxisSizes(dataShape);
            var indexAxes = ToAxisSizes(indicesShape);
            axis = NormalizeAxis(axis, dataAxes.Length, string.Empty);
            var outRank = dataAxes.Length + indexAxes.Length - 1;
            if (outRank > 4)
                throw new InvalidOperationException("Gather output rank > 4 is not supported on texture path.");

            var outAxes = new int[outRank];
            var cursor = 0;
            for (var i = 0; i < axis; i++)
                outAxes[cursor++] = dataAxes[i];
            for (var i = 0; i < indexAxes.Length; i++)
                outAxes[cursor++] = indexAxes[i];
            for (var i = axis + 1; i < dataAxes.Length; i++)
                outAxes[cursor++] = dataAxes[i];
            return FromAxisSizes(outAxes);
        }

        public static void ValidateGatherElementsShape(
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape dataShape,
            AexisGraphSession.BufferShape indicesShape,
            int axis)
        {
            var dataAxes = ToAxisSizes(dataShape);
            var indicesAxes = ToAxisSizes(indicesShape);
            axis = NormalizeAxis(axis, dataAxes.Length, layer?.name ?? string.Empty);
            if (dataAxes.Length != indicesAxes.Length)
                throw new InvalidOperationException("GatherElements requires data and indices to have the same rank"
                    + " | layer=" + (layer?.name ?? string.Empty)
                    + " | dataRank=" + dataAxes.Length
                    + " | indicesRank=" + indicesAxes.Length);

            for (var i = 0; i < dataAxes.Length; i++)
            {
                if (i == axis)
                    continue;
                if (indicesAxes[i] > dataAxes[i])
                    throw new InvalidOperationException("GatherElements indices shape exceeds data shape on a non-axis dimension"
                        + " | layer=" + (layer?.name ?? string.Empty)
                        + " | axis=" + axis
                        + " | dim=" + i
                        + " | data=" + dataAxes[i]
                        + " | indices=" + indicesAxes[i]);
            }
        }

        public static AexisGraphSession.BufferShape ResolveArgReduceShape(AexisGraphSession.BufferShape inputShape, int axis, bool keepDims)
        {
            var axes = ToAxisSizes(inputShape);
            axis = NormalizeAxis(axis, axes.Length, string.Empty);
            if (keepDims)
            {
                axes[axis] = 1;
                return FromAxisSizes(axes);
            }

            if (axes.Length == 1)
                return new AexisGraphSession.BufferShape(1, 1, 1, 1, 1);
            var outAxes = new int[axes.Length - 1];
            var cursor = 0;
            for (var i = 0; i < axes.Length; i++)
            {
                if (i != axis)
                    outAxes[cursor++] = axes[i];
            }
            return FromAxisSizes(outAxes);
        }

        public static AexisGraphSession.BufferShape ResolveTopKShape(AexisGraphSession.BufferShape inputShape, int axis, int k)
        {
            var axes = ToAxisSizes(inputShape);
            axis = NormalizeAxis(axis, axes.Length, string.Empty);
            if (k <= 0 || k > axes[axis])
                throw new InvalidOperationException("TopK k is out of range.");
            axes[axis] = k;
            return FromAxisSizes(axes);
        }

        public static AexisGraphSession.BufferShape ResolveOneHotShape(AexisGraphSession.BufferShape indicesShape, int axis, int depth, out int normalizedAxis)
        {
            if (depth <= 0)
                throw new InvalidOperationException("OneHot depth must be positive on the texture path.");
            var indexAxes = ToAxisSizes(indicesShape);
            var outRank = indexAxes.Length + 1;
            if (outRank > 4)
                throw new InvalidOperationException("OneHot output rank > 4 is not supported on texture path.");
            normalizedAxis = axis < 0 ? axis + outRank : axis;
            if (normalizedAxis < 0 || normalizedAxis > indexAxes.Length)
                throw new InvalidOperationException("OneHot axis out of range.");

            var outAxes = new int[outRank];
            var cursor = 0;
            for (var i = 0; i < outRank; i++)
            {
                if (i == normalizedAxis)
                    outAxes[i] = depth;
                else
                    outAxes[i] = indexAxes[cursor++];
            }
            return FromAxisSizes(outAxes);
        }

        public static void ReturnTemps(AexisGraphSession owner, List<RenderTexture> temps)
        {
            if (temps == null)
                return;
            for (var i = temps.Count - 1; i >= 0; i--)
            {
                if (temps[i] != null)
                    owner.ReturnTempArray(temps[i]);
            }
        }

        public static void ReturnTemps(AexisGraphSession owner, CommandBuffer cmd, List<ComputeTexture> temps)
        {
            if (temps == null)
                return;
            for (var i = temps.Count - 1; i >= 0; i--)
            {
                if (temps[i] != null)
                    owner.ReturnTempArray(cmd, temps[i]);
            }
        }

        private static string CleanToken(string raw)
        {
            return (raw ?? string.Empty).Trim().Trim('"', '\'', '[', ']', '(', ')');
        }
    }

    public sealed class AexisUnsupportedSentisLayer : AexisBaseLayer
    {
        private readonly string _reason;

        public AexisUnsupportedSentisLayer(AexisLayerTypeKey typeKey, string reason)
            : base(typeKey, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
            _reason = reason ?? "Unsupported on texture-backed path.";
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            throw CreateException(layer, "RenderTexture");
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            throw CreateException(layer, "CommandBuffer");
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            throw CreateException(layer, "ComputeBuffer");
        }

        private NotSupportedException CreateException(AexisGraphModel.Layer layer, string path)
        {
            return new NotSupportedException(TypeKey
                + " is not implemented for the " + path + " path"
                + " | layer=" + (layer?.name ?? string.Empty)
                + " | reason=" + _reason);
        }
    }

    public sealed class AexisShapeLayer : AexisBaseLayer
    {
        public AexisShapeLayer() : base(AexisLayerTypes.Shape, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            var shape = AexisShapeIndexLayerUtil.GetInputShape(layer, context, layer.bottomNames[0]);
            var values = ResolveShapeValues(layer, shape);
            var logical = new AexisGraphSession.BufferShape(1, Mathf.Max(1, values.Length), 1, 1, 1);
            AexisShapeIndexLayerUtil.PublishRenderLinear(owner, layer, context, logical, (output, _) => owner.Ops.FillScalarTexture(values, output));
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            var shape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            var values = ResolveShapeValues(layer, shape);
            var logical = new AexisGraphSession.BufferShape(1, Mathf.Max(1, values.Length), 1, 1, 1);
            AexisShapeIndexLayerUtil.PublishCmdLinear(owner, layer, context, logical, (output, _) => owner.Ops.FillScalarTexture(context.commandBuffer, values, output));
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static float[] ResolveShapeValues(AexisGraphModel.Layer layer, AexisGraphSession.BufferShape shape)
        {
            var axes = AexisShapeIndexLayerUtil.ToAxisSizes(shape);
            var start = AexisShapeIndexLayerUtil.GetInt(layer, 0, "start", 0);
            var end = AexisShapeIndexLayerUtil.GetInt(layer, 1, "end", axes.Length);
            if (start < 0) start += axes.Length;
            if (end < 0) end += axes.Length;
            start = Mathf.Clamp(start, 0, axes.Length);
            end = Mathf.Clamp(end, start, axes.Length);
            if (end <= start)
                throw new InvalidOperationException("Shape texture path cannot publish an empty shape tensor: " + layer.name);
            var values = new float[end - start];
            for (var i = start; i < end; i++)
                values[i - start] = axes[i];
            return values;
        }
    }

    public sealed class AexisSizeLayer : AexisBaseLayer
    {
        public AexisSizeLayer() : base(AexisLayerTypes.Size, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            var shape = AexisShapeIndexLayerUtil.GetInputShape(layer, context, layer.bottomNames[0]);
            var logical = new AexisGraphSession.BufferShape(1, 1, 1, 1, 1);
            var value = new[] { (float)AexisShapeIndexLayerUtil.ElementCount(shape) };
            AexisShapeIndexLayerUtil.PublishRenderLinear(owner, layer, context, logical, (output, _) => owner.Ops.FillScalarTexture(value, output));
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            var shape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            var logical = new AexisGraphSession.BufferShape(1, 1, 1, 1, 1);
            var value = new[] { (float)AexisShapeIndexLayerUtil.ElementCount(shape) };
            AexisShapeIndexLayerUtil.PublishCmdLinear(owner, layer, context, logical, (output, _) => owner.Ops.FillScalarTexture(context.commandBuffer, value, output));
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class AexisRangeLayer : AexisBaseLayer
    {
        public AexisRangeLayer() : base(AexisLayerTypes.Range, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var spec = ResolveStaticRange(layer);
            var logical = new AexisGraphSession.BufferShape(1, spec.count, 1, 1, 1);
            AexisShapeIndexLayerUtil.PublishRenderLinear(owner, layer, context, logical, (output, _) => owner.Ops.SentisRangeLinearMat(spec.start, spec.delta, spec.count, output));
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var spec = ResolveStaticRange(layer);
            var logical = new AexisGraphSession.BufferShape(1, spec.count, 1, 1, 1);
            AexisShapeIndexLayerUtil.PublishCmdLinear(owner, layer, context, logical, (output, _) => owner.Ops.SentisRangeLinearMat(context.commandBuffer, spec.start, spec.delta, spec.count, output));
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static (float start, float limit, float delta, int count) ResolveStaticRange(AexisGraphModel.Layer layer)
        {
            if (!AexisShapeIndexLayerUtil.TryGetFloat(layer, "start", out var start) && !AexisShapeIndexLayerUtil.TryGetFloat(layer, 0, out start))
                throw new InvalidOperationException("Range texture path requires static start param: " + layer.name);
            if (!AexisShapeIndexLayerUtil.TryGetFloat(layer, "limit", out var limit) && !AexisShapeIndexLayerUtil.TryGetFloat(layer, 1, out limit))
                throw new InvalidOperationException("Range texture path requires static limit param: " + layer.name);
            var delta = AexisShapeIndexLayerUtil.GetFloat(layer, 2, "delta", 1f);
            if (Mathf.Abs(delta) < 1e-12f)
                throw new InvalidOperationException("Range delta cannot be zero: " + layer.name);
            var span = (limit - start) / delta;
            if (span <= 0f)
                throw new InvalidOperationException("Range texture path cannot publish an empty output: " + layer.name);
            var count = Mathf.CeilToInt(span);
            return (start, limit, delta, count);
        }
    }

    public sealed class AexisConstantOfShapeLayer : AexisBaseLayer
    {
        public AexisConstantOfShapeLayer() : base(AexisLayerTypes.ConstantOfShape, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireTops(layer, 1);
            if (!AexisShapeIndexLayerUtil.TryGetShapeParam(layer, out var shapeValues))
                throw new InvalidOperationException("ConstantOfShape texture path requires a static shape param: " + layer.name);
            var logical = AexisShapeIndexLayerUtil.FromAxisSizes(shapeValues);
            var value = AexisShapeIndexLayerUtil.GetFloat(layer, 1, "value", AexisShapeIndexLayerUtil.GetFloat(layer, 0, "fill", 0f));
            AexisShapeIndexLayerUtil.PublishRenderLinear(owner, layer, context, logical, (output, _) => owner.Ops.SentisConstantLinearMat(value, AexisShapeIndexLayerUtil.ElementCount(logical), output));
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireTops(layer, 1);
            if (!AexisShapeIndexLayerUtil.TryGetShapeParam(layer, out var shapeValues))
                throw new InvalidOperationException("ConstantOfShape command-buffer path requires a static shape param: " + layer.name);
            var logical = AexisShapeIndexLayerUtil.FromAxisSizes(shapeValues);
            var value = AexisShapeIndexLayerUtil.GetFloat(layer, 1, "value", AexisShapeIndexLayerUtil.GetFloat(layer, 0, "fill", 0f));
            AexisShapeIndexLayerUtil.PublishCmdLinear(owner, layer, context, logical, (output, _) => owner.Ops.SentisConstantLinearMat(context.commandBuffer, value, AexisShapeIndexLayerUtil.ElementCount(logical), output));
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class AexisExpandLayer : AexisBaseLayer
    {
        public AexisExpandLayer() : base(AexisLayerTypes.Expand, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<RenderTexture>();
            try
            {
                var input = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                if (!AexisShapeIndexLayerUtil.TryGetShapeParam(layer, out var requested))
                    throw new InvalidOperationException("Expand texture path requires a static shape param: " + layer.name);
                var outShape = AexisShapeIndexLayerUtil.ResolveExpandShape(input.logicalShape, requested);
                AexisShapeIndexLayerUtil.PublishRenderLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisExpandLinearMat(input.texture, input.logicalShape, input.storageShape, outShape, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<ComputeTexture>();
            try
            {
                var input = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                if (!AexisShapeIndexLayerUtil.TryGetShapeParam(layer, out var requested))
                    throw new InvalidOperationException("Expand command-buffer path requires a static shape param: " + layer.name);
                var outShape = AexisShapeIndexLayerUtil.ResolveExpandShape(input.logicalShape, requested);
                AexisShapeIndexLayerUtil.PublishCmdLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisExpandLinearMat(context.commandBuffer, input.texture, input.logicalShape, input.storageShape, outShape, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class AexisWhereLayer : AexisBaseLayer
    {
        public AexisWhereLayer() : base(AexisLayerTypes.Where, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 3);
            var temps = new List<RenderTexture>();
            try
            {
                var cond = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var a = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[1], temps);
                var b = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[2], temps);
                var outShape = AexisShapeIndexLayerUtil.BroadcastShapes(cond.logicalShape, a.logicalShape, b.logicalShape);
                AexisShapeIndexLayerUtil.PublishRenderLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisWhereLinearMat(cond.texture, cond.logicalShape, cond.storageShape, a.texture, a.logicalShape, a.storageShape, b.texture, b.logicalShape, b.storageShape, outShape, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 3);
            var temps = new List<ComputeTexture>();
            try
            {
                var cond = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var a = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[1], temps);
                var b = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[2], temps);
                var outShape = AexisShapeIndexLayerUtil.BroadcastShapes(cond.logicalShape, a.logicalShape, b.logicalShape);
                AexisShapeIndexLayerUtil.PublishCmdLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisWhereLinearMat(context.commandBuffer, cond.texture, cond.logicalShape, cond.storageShape, a.texture, a.logicalShape, a.storageShape, b.texture, b.logicalShape, b.storageShape, outShape, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class AexisGatherLayer : AexisBaseLayer
    {
        public AexisGatherLayer() : base(AexisLayerTypes.Gather, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 2);
            var temps = new List<RenderTexture>();
            try
            {
                var data = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var indices = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[1], temps);
                var axis = AexisShapeIndexLayerUtil.GetInt(layer, 0, "axis", 0);
                var outShape = AexisShapeIndexLayerUtil.ResolveGatherShape(data.logicalShape, indices.logicalShape, axis);
                AexisShapeIndexLayerUtil.PublishRenderLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisGatherLinearMat(data.texture, data.logicalShape, data.storageShape, indices.texture, indices.logicalShape, indices.storageShape, axis, outShape, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 2);
            var temps = new List<ComputeTexture>();
            try
            {
                var data = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var indices = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[1], temps);
                var axis = AexisShapeIndexLayerUtil.GetInt(layer, 0, "axis", 0);
                var outShape = AexisShapeIndexLayerUtil.ResolveGatherShape(data.logicalShape, indices.logicalShape, axis);
                AexisShapeIndexLayerUtil.PublishCmdLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisGatherLinearMat(context.commandBuffer, data.texture, data.logicalShape, data.storageShape, indices.texture, indices.logicalShape, indices.storageShape, axis, outShape, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class AexisGatherElementsLayer : AexisBaseLayer
    {
        public AexisGatherElementsLayer() : base(AexisLayerTypes.GatherElements, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 2);
            var temps = new List<RenderTexture>();
            try
            {
                var data = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var indices = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[1], temps);
                var axis = AexisShapeIndexLayerUtil.GetInt(layer, 0, "axis", 0);
                AexisShapeIndexLayerUtil.ValidateGatherElementsShape(layer, data.logicalShape, indices.logicalShape, axis);
                var outShape = indices.logicalShape;
                AexisShapeIndexLayerUtil.PublishRenderLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisGatherElementsLinearMat(data.texture, data.logicalShape, data.storageShape, indices.texture, indices.logicalShape, indices.storageShape, axis, outShape, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 2);
            var temps = new List<ComputeTexture>();
            try
            {
                var data = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var indices = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[1], temps);
                var axis = AexisShapeIndexLayerUtil.GetInt(layer, 0, "axis", 0);
                AexisShapeIndexLayerUtil.ValidateGatherElementsShape(layer, data.logicalShape, indices.logicalShape, axis);
                var outShape = indices.logicalShape;
                AexisShapeIndexLayerUtil.PublishCmdLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisGatherElementsLinearMat(context.commandBuffer, data.texture, data.logicalShape, data.storageShape, indices.texture, indices.logicalShape, indices.storageShape, axis, outShape, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class AexisArgReduceLayer : AexisBaseLayer
    {
        private readonly bool _reduceMax;

        public AexisArgReduceLayer(AexisLayerTypeKey typeKey, bool reduceMax)
            : base(typeKey, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
            _reduceMax = reduceMax;
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<RenderTexture>();
            try
            {
                var input = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var axis = AexisShapeIndexLayerUtil.GetInt(layer, 0, "axis", 0);
                var keepDims = AexisShapeIndexLayerUtil.GetInt(layer, 1, "keepdims", AexisShapeIndexLayerUtil.GetInt(layer, 1, "keepDims", 1)) != 0;
                var selectLast = AexisShapeIndexLayerUtil.GetInt(layer, 2, "selectLastIndex", 0) != 0;
                var outShape = AexisShapeIndexLayerUtil.ResolveArgReduceShape(input.logicalShape, axis, keepDims);
                AexisShapeIndexLayerUtil.PublishRenderLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisArgReduceLinearMat(input.texture, input.logicalShape, input.storageShape, axis, keepDims, selectLast, _reduceMax, outShape, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<ComputeTexture>();
            try
            {
                var input = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var axis = AexisShapeIndexLayerUtil.GetInt(layer, 0, "axis", 0);
                var keepDims = AexisShapeIndexLayerUtil.GetInt(layer, 1, "keepdims", AexisShapeIndexLayerUtil.GetInt(layer, 1, "keepDims", 1)) != 0;
                var selectLast = AexisShapeIndexLayerUtil.GetInt(layer, 2, "selectLastIndex", 0) != 0;
                var outShape = AexisShapeIndexLayerUtil.ResolveArgReduceShape(input.logicalShape, axis, keepDims);
                AexisShapeIndexLayerUtil.PublishCmdLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisArgReduceLinearMat(context.commandBuffer, input.texture, input.logicalShape, input.storageShape, axis, keepDims, selectLast, _reduceMax, outShape, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class AexisTopKLayer : AexisBaseLayer
    {
        public AexisTopKLayer() : base(AexisLayerTypes.TopK, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            AexisShapeIndexLayerUtil.RequireTops(layer, layer.tops > 1 ? 2 : 1);
            var temps = new List<RenderTexture>();
            try
            {
                var input = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var spec = ResolveSpec(layer, input.logicalShape);
                var storage = AexisGraphSession.ResolveLinearMatStorageShape(spec.outShape);
                var values = owner.RentTempMat(storage.w, storage.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                var indices = layer.tops > 1 ? owner.RentTempMat(storage.w, storage.h, AexisGraphSession.ResolveLinearMatTextureFormat()) : null;
                owner.Ops.SentisTopKLinearMat(input.texture, input.logicalShape, input.storageShape, spec.axis, spec.k, spec.largest, spec.outShape, storage, values, indices);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], values, spec.outShape, storage);
                if (indices != null)
                    AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[1], indices, spec.outShape, storage);
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            AexisShapeIndexLayerUtil.RequireTops(layer, layer.tops > 1 ? 2 : 1);
            var temps = new List<ComputeTexture>();
            try
            {
                var input = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var spec = ResolveSpec(layer, input.logicalShape);
                var storage = AexisGraphSession.ResolveLinearMatStorageShape(spec.outShape);
                var values = owner.RentTempMat(context.commandBuffer, storage.w, storage.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                var indices = layer.tops > 1 ? owner.RentTempMat(context.commandBuffer, storage.w, storage.h, AexisGraphSession.ResolveLinearMatTextureFormat()) : null;
                owner.Ops.SentisTopKLinearMat(context.commandBuffer, input.texture, input.logicalShape, input.storageShape, spec.axis, spec.k, spec.largest, spec.outShape, storage, values, indices);
                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(values, spec.outShape, storage, owned: true);
                if (context.shapes != null)
                    context.shapes[layer.topNames[0]] = spec.outShape;
                if (indices != null)
                {
                    context.blobs[layer.topNames[1]] = AexisGraphSession.CreateCmdTensorRef(indices, spec.outShape, storage, owned: true);
                    if (context.shapes != null)
                        context.shapes[layer.topNames[1]] = spec.outShape;
                }
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static (int axis, int k, bool largest, AexisGraphSession.BufferShape outShape) ResolveSpec(AexisGraphModel.Layer layer, AexisGraphSession.BufferShape inputShape)
        {
            var axis = AexisShapeIndexLayerUtil.GetInt(layer, 0, "axis", -1);
            if (!AexisShapeIndexLayerUtil.TryGetInt(layer, "k", out var k)
                && !AexisShapeIndexLayerUtil.TryGetInt(layer, 1, out k))
            {
                throw new InvalidOperationException("TopK texture path requires static k param: " + layer.name);
            }
            var largest = AexisShapeIndexLayerUtil.GetInt(layer, 2, "largest", 1) != 0;
            var outShape = AexisShapeIndexLayerUtil.ResolveTopKShape(inputShape, axis, k);
            axis = AexisShapeIndexLayerUtil.NormalizeAxis(axis, inputShape.dims, layer.name);
            return (axis, k, largest, outShape);
        }
    }

    public sealed class AexisOneHotLayer : AexisBaseLayer
    {
        public AexisOneHotLayer() : base(AexisLayerTypes.OneHot, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<RenderTexture>();
            try
            {
                var indices = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var spec = ResolveSpec(layer, indices.logicalShape);
                AexisShapeIndexLayerUtil.PublishRenderLinear(owner, layer, context, spec.outShape, (output, storage) =>
                    owner.Ops.SentisOneHotLinearMat(indices.texture, indices.logicalShape, indices.storageShape, spec.axis, spec.depth, spec.offValue, spec.onValue, spec.outShape, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<ComputeTexture>();
            try
            {
                var indices = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var spec = ResolveSpec(layer, indices.logicalShape);
                AexisShapeIndexLayerUtil.PublishCmdLinear(owner, layer, context, spec.outShape, (output, storage) =>
                    owner.Ops.SentisOneHotLinearMat(context.commandBuffer, indices.texture, indices.logicalShape, indices.storageShape, spec.axis, spec.depth, spec.offValue, spec.onValue, spec.outShape, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static (int axis, int depth, float offValue, float onValue, AexisGraphSession.BufferShape outShape) ResolveSpec(AexisGraphModel.Layer layer, AexisGraphSession.BufferShape indicesShape)
        {
            if (!AexisShapeIndexLayerUtil.TryGetInt(layer, "depth", out var depth)
                && !AexisShapeIndexLayerUtil.TryGetInt(layer, 1, out depth)
                && !AexisShapeIndexLayerUtil.TryGetInt(layer, 0, out depth))
            {
                throw new InvalidOperationException("OneHot texture path requires static depth param: " + layer.name);
            }
            var axis = AexisShapeIndexLayerUtil.GetInt(layer, 2, "axis", -1);
            var onValue = AexisShapeIndexLayerUtil.GetFloat(layer, 3, "on_value", 1f);
            var offValue = AexisShapeIndexLayerUtil.GetFloat(layer, 4, "off_value", 0f);
            var outShape = AexisShapeIndexLayerUtil.ResolveOneHotShape(indicesShape, axis, depth, out var normalizedAxis);
            return (normalizedAxis, depth, offValue, onValue, outShape);
        }
    }

    public sealed class AexisCumSumLayer : AexisBaseLayer
    {
        public AexisCumSumLayer() : base(AexisLayerTypes.CumSum, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<RenderTexture>();
            try
            {
                var input = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var spec = ResolveSpec(layer, input.logicalShape);
                AexisShapeIndexLayerUtil.PublishRenderLinear(owner, layer, context, input.logicalShape, (output, storage) =>
                    owner.Ops.SentisCumSumLinearMat(input.texture, input.logicalShape, input.storageShape, spec.axis, spec.exclusive, spec.reverse, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<ComputeTexture>();
            try
            {
                var input = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var spec = ResolveSpec(layer, input.logicalShape);
                AexisShapeIndexLayerUtil.PublishCmdLinear(owner, layer, context, input.logicalShape, (output, storage) =>
                    owner.Ops.SentisCumSumLinearMat(context.commandBuffer, input.texture, input.logicalShape, input.storageShape, spec.axis, spec.exclusive, spec.reverse, storage, output));
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static (int axis, bool exclusive, bool reverse) ResolveSpec(AexisGraphModel.Layer layer, AexisGraphSession.BufferShape inputShape)
        {
            if (!AexisShapeIndexLayerUtil.TryGetInt(layer, "axis", out var axis)
                && !AexisShapeIndexLayerUtil.TryGetInt(layer, 0, out axis))
            {
                throw new InvalidOperationException("CumSum texture path requires static axis param: " + layer.name);
            }
            axis = AexisShapeIndexLayerUtil.NormalizeAxis(axis, inputShape.dims, layer.name);
            var exclusive = AexisShapeIndexLayerUtil.GetInt(layer, 1, "exclusive", 0) != 0;
            var reverse = AexisShapeIndexLayerUtil.GetInt(layer, 2, "reverse", 0) != 0;
            return (axis, exclusive, reverse);
        }
    }
}
