using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    internal static class NcnnSentisLayerUtil
    {
        public readonly struct RenderLinearInput
        {
            public readonly RenderTexture texture;
            public readonly NcnnRepro.BufferShape logicalShape;
            public readonly NcnnRepro.BufferShape storageShape;

            public RenderLinearInput(RenderTexture texture, NcnnRepro.BufferShape logicalShape, NcnnRepro.BufferShape storageShape)
            {
                this.texture = texture;
                this.logicalShape = logicalShape;
                this.storageShape = storageShape;
            }
        }

        public readonly struct CmdLinearInput
        {
            public readonly ComputeTexture texture;
            public readonly NcnnRepro.BufferShape logicalShape;
            public readonly NcnnRepro.BufferShape storageShape;

            public CmdLinearInput(ComputeTexture texture, NcnnRepro.BufferShape logicalShape, NcnnRepro.BufferShape storageShape)
            {
                this.texture = texture;
                this.logicalShape = logicalShape;
                this.storageShape = storageShape;
            }
        }

        public static int ElementCount(NcnnRepro.BufferShape shape)
        {
            return Mathf.Max(1, shape.w) * Mathf.Max(1, shape.h) * Mathf.Max(1, shape.d) * Mathf.Max(1, shape.c);
        }

        public static int[] ToAxisSizes(NcnnRepro.BufferShape shape)
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

        public static NcnnRepro.BufferShape FromAxisSizes(int[] sizes)
        {
            if (sizes == null || sizes.Length == 0)
                return new NcnnRepro.BufferShape(1, 1, 1, 1, 1);

            if (sizes.Length > 4)
                throw new InvalidOperationException("Texture-backed Sentis ops currently support rank <= 4, got rank=" + sizes.Length);
            for (var i = 0; i < sizes.Length; i++)
            {
                if (sizes[i] <= 0)
                    throw new InvalidOperationException("Texture-backed Sentis ops do not support zero/negative axis sizes: axis=" + i + " size=" + sizes[i]);
            }

            return sizes.Length switch
            {
                1 => new NcnnRepro.BufferShape(1, sizes[0], 1, 1, 1),
                2 => new NcnnRepro.BufferShape(2, sizes[1], sizes[0], 1, 1),
                3 => new NcnnRepro.BufferShape(3, sizes[2], sizes[1], 1, sizes[0]),
                4 => new NcnnRepro.BufferShape(4, sizes[3], sizes[2], sizes[1], sizes[0]),
                _ => new NcnnRepro.BufferShape(1, 1, 1, 1, 1)
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

        public static void RequireBottoms(NcnnParamModel.Layer layer, int minCount)
        {
            if (layer?.bottomNames == null || layer.bottomNames.Length < minCount)
                throw new InvalidOperationException("Sentis texture path input count mismatch"
                    + " | layer=" + (layer?.name ?? string.Empty)
                    + " | type=" + (layer?.typeName ?? string.Empty)
                    + " | expected>=" + minCount
                    + " | actual=" + (layer?.bottomNames == null ? 0 : layer.bottomNames.Length));
        }

        public static void RequireTops(NcnnParamModel.Layer layer, int minCount)
        {
            if (layer?.topNames == null || layer.topNames.Length < minCount)
                throw new InvalidOperationException("Sentis texture path output count mismatch"
                    + " | layer=" + (layer?.name ?? string.Empty)
                    + " | type=" + (layer?.typeName ?? string.Empty)
                    + " | expected>=" + minCount
                    + " | actual=" + (layer?.topNames == null ? 0 : layer.topNames.Length));
        }

        public static bool TryGetInt(NcnnParamModel.Layer layer, string name, out int value)
        {
            value = 0;
            if (layer?.stringParams == null || string.IsNullOrEmpty(name) || !layer.stringParams.TryGetValue(name, out var raw))
                return false;
            raw = CleanToken(raw);
            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
        }

        public static bool TryGetInt(NcnnParamModel.Layer layer, int key, out int value)
        {
            value = 0;
            if (layer?.intParams == null || !layer.intParams.TryGetValue(key, out var raw))
                return false;
            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
        }

        public static int GetInt(NcnnParamModel.Layer layer, int key, string name, int defaultValue)
        {
            if (TryGetInt(layer, name, out var named))
                return named;
            return TryGetInt(layer, key, out var keyed) ? keyed : defaultValue;
        }

        public static bool TryGetFloat(NcnnParamModel.Layer layer, string name, out float value)
        {
            value = 0f;
            if (layer?.stringParams == null || string.IsNullOrEmpty(name) || !layer.stringParams.TryGetValue(name, out var raw))
                return false;
            raw = CleanToken(raw);
            return float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
        }

        public static bool TryGetFloat(NcnnParamModel.Layer layer, int key, out float value)
        {
            value = 0f;
            if (layer?.intParams == null || !layer.intParams.TryGetValue(key, out var raw))
                return false;
            return float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
        }

        public static float GetFloat(NcnnParamModel.Layer layer, int key, string name, float defaultValue)
        {
            if (TryGetFloat(layer, name, out var named))
                return named;
            return TryGetFloat(layer, key, out var keyed) ? keyed : defaultValue;
        }

        public static bool TryGetShapeParam(NcnnParamModel.Layer layer, out int[] shape)
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

        public static NcnnRepro.BufferShape GetInputShape(
            NcnnParamModel.Layer layer,
            NcnnLayerBufferContext context,
            string bottomName)
        {
            if (NcnnRepro.TryGetExistingTexture(context.textureBlobs, context.textureShapes, bottomName, out var tex, out var shape))
                return shape;

            var view = NcnnRepro.TryGetBufferView(bottomName, context.bufferBlobs, context.bufferViews);
            if (view != null)
                return new NcnnRepro.BufferShape(view.dims, view.w, view.h, view.d, view.c);

            throw new InvalidOperationException("Sentis layer input shape not found"
                + " | layer=" + layer.name
                + " | bottom=" + bottomName);
        }

        public static RenderLinearInput GetRenderLinearInput(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnLayerBufferContext context,
            string bottomName,
            List<RenderTexture> temps)
        {
            if (!NcnnRepro.TryGetExistingTexture(context.textureBlobs, context.textureShapes, bottomName, out var src, out var shape)
                || src == null
                || src.texture == null)
            {
                throw new InvalidOperationException("Sentis texture path requires existing texture input"
                    + " | layer=" + layer.name
                    + " | bottom=" + bottomName);
            }

            var storage = NcnnRepro.GetTextureStorageShape(src, shape);
            if (NcnnRepro.IsStrictLinearMatTexture(src))
                return new RenderLinearInput(src.texture, shape, storage);

            if (src.texture.dimension != TextureDimension.Tex2DArray || !NcnnRepro.MatchesPack4TextureStorage(src, shape))
            {
                throw new InvalidOperationException("Sentis texture path requires LinearMat or direct pack4 input"
                    + " | layer=" + layer.name
                    + " | bottom=" + bottomName
                    + " | logical=d" + shape.dims + ":" + shape.w + "x" + shape.h + "x" + shape.d + "x" + shape.c
                    + " | storage=d" + storage.dims + ":" + storage.w + "x" + storage.h + "x" + storage.d + "x" + storage.c);
            }

            var linearStorage = NcnnRepro.ResolveLinearMatStorageShape(shape);
            var linear = owner.RentTempMat(linearStorage.w, linearStorage.h, NcnnRepro.ResolveLinearMatTextureFormat());
            owner.Ops.ReshapePack4ToLinearMat(src.texture, shape.w, shape.h, shape.d, shape.c, shape.dims, linear);
            temps?.Add(linear);
            return new RenderLinearInput(linear, shape, linearStorage);
        }

        public static CmdLinearInput GetCmdLinearInput(
            NcnnRepro owner,
            CommandBuffer cmd,
            NcnnParamModel.Layer layer,
            NcnnLayerCommandBufferContext context,
            string bottomName,
            List<ComputeTexture> temps)
        {
            var src = NcnnRepro.GetCmdTensor(context.blobs, bottomName);
            var shape = NcnnRepro.GetCmdShape(context.shapes, context.blobs, bottomName);
            var storage = NcnnRepro.GetCmdStorageShape(src, shape);
            if (NcnnRepro.IsStrictLinearMatTexture(src))
                return new CmdLinearInput(src.texture, shape, storage);

            if (src.texture.dimension != TextureDimension.Tex2DArray || !NcnnRepro.MatchesPack4TextureStorage(src, shape))
            {
                throw new InvalidOperationException("Sentis command-buffer path requires LinearMat or direct pack4 input"
                    + " | layer=" + layer.name
                    + " | bottom=" + bottomName
                    + " | logical=d" + shape.dims + ":" + shape.w + "x" + shape.h + "x" + shape.d + "x" + shape.c
                    + " | storage=d" + storage.dims + ":" + storage.w + "x" + storage.h + "x" + storage.d + "x" + storage.c);
            }

            var linearStorage = NcnnRepro.ResolveLinearMatStorageShape(shape);
            var linear = owner.RentTempMat(cmd, linearStorage.w, linearStorage.h, NcnnRepro.ResolveLinearMatTextureFormat());
            owner.Ops.ReshapePack4ToLinearMat(cmd, src.texture, shape.w, shape.h, shape.d, shape.c, shape.dims, linear);
            temps?.Add(linear);
            return new CmdLinearInput(linear, shape, linearStorage);
        }

        public static void PublishRenderLinear(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnLayerBufferContext context,
            NcnnRepro.BufferShape logicalShape,
            Action<RenderTexture, NcnnRepro.BufferShape> fill)
        {
            RequireTops(layer, 1);
            var storage = NcnnRepro.ResolveLinearMatStorageShape(logicalShape);
            var output = owner.RentTempMat(storage.w, storage.h, NcnnRepro.ResolveLinearMatTextureFormat());
            fill(output, storage);
            NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, logicalShape, storage);
        }

        public static void PublishCmdLinear(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnLayerCommandBufferContext context,
            NcnnRepro.BufferShape logicalShape,
            Action<ComputeTexture, NcnnRepro.BufferShape> fill)
        {
            RequireTops(layer, 1);
            var storage = NcnnRepro.ResolveLinearMatStorageShape(logicalShape);
            var output = owner.RentTempMat(context.commandBuffer, storage.w, storage.h, NcnnRepro.ResolveLinearMatTextureFormat());
            fill(output, storage);
            context.blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(output, logicalShape, storage, owned: true);
            if (context.shapes != null)
                context.shapes[layer.topNames[0]] = logicalShape;
        }

        public static NcnnRepro.BufferShape BroadcastShapes(params NcnnRepro.BufferShape[] shapes)
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

        public static NcnnRepro.BufferShape ResolveExpandShape(NcnnRepro.BufferShape inputShape, int[] requested)
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

        public static NcnnRepro.BufferShape ResolveGatherShape(NcnnRepro.BufferShape dataShape, NcnnRepro.BufferShape indicesShape, int axis)
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
            NcnnParamModel.Layer layer,
            NcnnRepro.BufferShape dataShape,
            NcnnRepro.BufferShape indicesShape,
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

        public static NcnnRepro.BufferShape ResolveArgReduceShape(NcnnRepro.BufferShape inputShape, int axis, bool keepDims)
        {
            var axes = ToAxisSizes(inputShape);
            axis = NormalizeAxis(axis, axes.Length, string.Empty);
            if (keepDims)
            {
                axes[axis] = 1;
                return FromAxisSizes(axes);
            }

            if (axes.Length == 1)
                return new NcnnRepro.BufferShape(1, 1, 1, 1, 1);
            var outAxes = new int[axes.Length - 1];
            var cursor = 0;
            for (var i = 0; i < axes.Length; i++)
            {
                if (i != axis)
                    outAxes[cursor++] = axes[i];
            }
            return FromAxisSizes(outAxes);
        }

        public static NcnnRepro.BufferShape ResolveTopKShape(NcnnRepro.BufferShape inputShape, int axis, int k)
        {
            var axes = ToAxisSizes(inputShape);
            axis = NormalizeAxis(axis, axes.Length, string.Empty);
            if (k <= 0 || k > axes[axis])
                throw new InvalidOperationException("TopK k is out of range.");
            axes[axis] = k;
            return FromAxisSizes(axes);
        }

        public static NcnnRepro.BufferShape ResolveOneHotShape(NcnnRepro.BufferShape indicesShape, int axis, int depth, out int normalizedAxis)
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

        public static void ReturnTemps(NcnnRepro owner, List<RenderTexture> temps)
        {
            if (temps == null)
                return;
            for (var i = temps.Count - 1; i >= 0; i--)
            {
                if (temps[i] != null)
                    owner.ReturnTempArray(temps[i]);
            }
        }

        public static void ReturnTemps(NcnnRepro owner, CommandBuffer cmd, List<ComputeTexture> temps)
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

    public sealed class NcnnUnsupportedSentisLayerRepro : NcnnBaseLayerRepro
    {
        private readonly string _reason;

        public NcnnUnsupportedSentisLayerRepro(NcnnLayerTypeKey typeKey, string reason)
            : base(typeKey, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
            _reason = reason ?? "Unsupported on texture-backed path.";
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            throw CreateException(layer, "RenderTexture");
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            throw CreateException(layer, "CommandBuffer");
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            throw CreateException(layer, "ComputeBuffer");
        }

        private NotSupportedException CreateException(NcnnParamModel.Layer layer, string path)
        {
            return new NotSupportedException(TypeKey
                + " is not implemented for the " + path + " path"
                + " | layer=" + (layer?.name ?? string.Empty)
                + " | reason=" + _reason);
        }
    }

    public sealed class NcnnShapeLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnShapeLayerRepro() : base(NcnnLayerTypes.Shape, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            var shape = NcnnSentisLayerUtil.GetInputShape(layer, context, layer.bottomNames[0]);
            var values = ResolveShapeValues(layer, shape);
            var logical = new NcnnRepro.BufferShape(1, Mathf.Max(1, values.Length), 1, 1, 1);
            NcnnSentisLayerUtil.PublishRenderLinear(owner, layer, context, logical, (output, _) => owner.Ops.FillScalarTexture(values, output));
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            var shape = NcnnRepro.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            var values = ResolveShapeValues(layer, shape);
            var logical = new NcnnRepro.BufferShape(1, Mathf.Max(1, values.Length), 1, 1, 1);
            NcnnSentisLayerUtil.PublishCmdLinear(owner, layer, context, logical, (output, _) => owner.Ops.FillScalarTexture(context.commandBuffer, values, output));
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static float[] ResolveShapeValues(NcnnParamModel.Layer layer, NcnnRepro.BufferShape shape)
        {
            var axes = NcnnSentisLayerUtil.ToAxisSizes(shape);
            var start = NcnnSentisLayerUtil.GetInt(layer, 0, "start", 0);
            var end = NcnnSentisLayerUtil.GetInt(layer, 1, "end", axes.Length);
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

    public sealed class NcnnSizeLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnSizeLayerRepro() : base(NcnnLayerTypes.Size, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            var shape = NcnnSentisLayerUtil.GetInputShape(layer, context, layer.bottomNames[0]);
            var logical = new NcnnRepro.BufferShape(1, 1, 1, 1, 1);
            var value = new[] { (float)NcnnSentisLayerUtil.ElementCount(shape) };
            NcnnSentisLayerUtil.PublishRenderLinear(owner, layer, context, logical, (output, _) => owner.Ops.FillScalarTexture(value, output));
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            var shape = NcnnRepro.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            var logical = new NcnnRepro.BufferShape(1, 1, 1, 1, 1);
            var value = new[] { (float)NcnnSentisLayerUtil.ElementCount(shape) };
            NcnnSentisLayerUtil.PublishCmdLinear(owner, layer, context, logical, (output, _) => owner.Ops.FillScalarTexture(context.commandBuffer, value, output));
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class NcnnRangeLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnRangeLayerRepro() : base(NcnnLayerTypes.Range, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var spec = ResolveStaticRange(layer);
            var logical = new NcnnRepro.BufferShape(1, spec.count, 1, 1, 1);
            NcnnSentisLayerUtil.PublishRenderLinear(owner, layer, context, logical, (output, _) => owner.Ops.SentisRangeLinearMat(spec.start, spec.delta, spec.count, output));
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var spec = ResolveStaticRange(layer);
            var logical = new NcnnRepro.BufferShape(1, spec.count, 1, 1, 1);
            NcnnSentisLayerUtil.PublishCmdLinear(owner, layer, context, logical, (output, _) => owner.Ops.SentisRangeLinearMat(context.commandBuffer, spec.start, spec.delta, spec.count, output));
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static (float start, float limit, float delta, int count) ResolveStaticRange(NcnnParamModel.Layer layer)
        {
            if (!NcnnSentisLayerUtil.TryGetFloat(layer, "start", out var start) && !NcnnSentisLayerUtil.TryGetFloat(layer, 0, out start))
                throw new InvalidOperationException("Range texture path requires static start param: " + layer.name);
            if (!NcnnSentisLayerUtil.TryGetFloat(layer, "limit", out var limit) && !NcnnSentisLayerUtil.TryGetFloat(layer, 1, out limit))
                throw new InvalidOperationException("Range texture path requires static limit param: " + layer.name);
            var delta = NcnnSentisLayerUtil.GetFloat(layer, 2, "delta", 1f);
            if (Mathf.Abs(delta) < 1e-12f)
                throw new InvalidOperationException("Range delta cannot be zero: " + layer.name);
            var span = (limit - start) / delta;
            if (span <= 0f)
                throw new InvalidOperationException("Range texture path cannot publish an empty output: " + layer.name);
            var count = Mathf.CeilToInt(span);
            return (start, limit, delta, count);
        }
    }

    public sealed class NcnnConstantOfShapeLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnConstantOfShapeLayerRepro() : base(NcnnLayerTypes.ConstantOfShape, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            NcnnSentisLayerUtil.RequireTops(layer, 1);
            if (!NcnnSentisLayerUtil.TryGetShapeParam(layer, out var shapeValues))
                throw new InvalidOperationException("ConstantOfShape texture path requires a static shape param: " + layer.name);
            var logical = NcnnSentisLayerUtil.FromAxisSizes(shapeValues);
            var value = NcnnSentisLayerUtil.GetFloat(layer, 1, "value", NcnnSentisLayerUtil.GetFloat(layer, 0, "fill", 0f));
            NcnnSentisLayerUtil.PublishRenderLinear(owner, layer, context, logical, (output, _) => owner.Ops.SentisConstantLinearMat(value, NcnnSentisLayerUtil.ElementCount(logical), output));
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            NcnnSentisLayerUtil.RequireTops(layer, 1);
            if (!NcnnSentisLayerUtil.TryGetShapeParam(layer, out var shapeValues))
                throw new InvalidOperationException("ConstantOfShape command-buffer path requires a static shape param: " + layer.name);
            var logical = NcnnSentisLayerUtil.FromAxisSizes(shapeValues);
            var value = NcnnSentisLayerUtil.GetFloat(layer, 1, "value", NcnnSentisLayerUtil.GetFloat(layer, 0, "fill", 0f));
            NcnnSentisLayerUtil.PublishCmdLinear(owner, layer, context, logical, (output, _) => owner.Ops.SentisConstantLinearMat(context.commandBuffer, value, NcnnSentisLayerUtil.ElementCount(logical), output));
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class NcnnExpandLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnExpandLayerRepro() : base(NcnnLayerTypes.Expand, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<RenderTexture>();
            try
            {
                var input = NcnnSentisLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                if (!NcnnSentisLayerUtil.TryGetShapeParam(layer, out var requested))
                    throw new InvalidOperationException("Expand texture path requires a static shape param: " + layer.name);
                var outShape = NcnnSentisLayerUtil.ResolveExpandShape(input.logicalShape, requested);
                NcnnSentisLayerUtil.PublishRenderLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisExpandLinearMat(input.texture, input.logicalShape, input.storageShape, outShape, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<ComputeTexture>();
            try
            {
                var input = NcnnSentisLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                if (!NcnnSentisLayerUtil.TryGetShapeParam(layer, out var requested))
                    throw new InvalidOperationException("Expand command-buffer path requires a static shape param: " + layer.name);
                var outShape = NcnnSentisLayerUtil.ResolveExpandShape(input.logicalShape, requested);
                NcnnSentisLayerUtil.PublishCmdLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisExpandLinearMat(context.commandBuffer, input.texture, input.logicalShape, input.storageShape, outShape, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class NcnnWhereLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnWhereLayerRepro() : base(NcnnLayerTypes.Where, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 3);
            var temps = new List<RenderTexture>();
            try
            {
                var cond = NcnnSentisLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var a = NcnnSentisLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[1], temps);
                var b = NcnnSentisLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[2], temps);
                var outShape = NcnnSentisLayerUtil.BroadcastShapes(cond.logicalShape, a.logicalShape, b.logicalShape);
                NcnnSentisLayerUtil.PublishRenderLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisWhereLinearMat(cond.texture, cond.logicalShape, cond.storageShape, a.texture, a.logicalShape, a.storageShape, b.texture, b.logicalShape, b.storageShape, outShape, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 3);
            var temps = new List<ComputeTexture>();
            try
            {
                var cond = NcnnSentisLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var a = NcnnSentisLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[1], temps);
                var b = NcnnSentisLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[2], temps);
                var outShape = NcnnSentisLayerUtil.BroadcastShapes(cond.logicalShape, a.logicalShape, b.logicalShape);
                NcnnSentisLayerUtil.PublishCmdLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisWhereLinearMat(context.commandBuffer, cond.texture, cond.logicalShape, cond.storageShape, a.texture, a.logicalShape, a.storageShape, b.texture, b.logicalShape, b.storageShape, outShape, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class NcnnGatherLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnGatherLayerRepro() : base(NcnnLayerTypes.Gather, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 2);
            var temps = new List<RenderTexture>();
            try
            {
                var data = NcnnSentisLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var indices = NcnnSentisLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[1], temps);
                var axis = NcnnSentisLayerUtil.GetInt(layer, 0, "axis", 0);
                var outShape = NcnnSentisLayerUtil.ResolveGatherShape(data.logicalShape, indices.logicalShape, axis);
                NcnnSentisLayerUtil.PublishRenderLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisGatherLinearMat(data.texture, data.logicalShape, data.storageShape, indices.texture, indices.logicalShape, indices.storageShape, axis, outShape, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 2);
            var temps = new List<ComputeTexture>();
            try
            {
                var data = NcnnSentisLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var indices = NcnnSentisLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[1], temps);
                var axis = NcnnSentisLayerUtil.GetInt(layer, 0, "axis", 0);
                var outShape = NcnnSentisLayerUtil.ResolveGatherShape(data.logicalShape, indices.logicalShape, axis);
                NcnnSentisLayerUtil.PublishCmdLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisGatherLinearMat(context.commandBuffer, data.texture, data.logicalShape, data.storageShape, indices.texture, indices.logicalShape, indices.storageShape, axis, outShape, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class NcnnGatherElementsLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnGatherElementsLayerRepro() : base(NcnnLayerTypes.GatherElements, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 2);
            var temps = new List<RenderTexture>();
            try
            {
                var data = NcnnSentisLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var indices = NcnnSentisLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[1], temps);
                var axis = NcnnSentisLayerUtil.GetInt(layer, 0, "axis", 0);
                NcnnSentisLayerUtil.ValidateGatherElementsShape(layer, data.logicalShape, indices.logicalShape, axis);
                var outShape = indices.logicalShape;
                NcnnSentisLayerUtil.PublishRenderLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisGatherElementsLinearMat(data.texture, data.logicalShape, data.storageShape, indices.texture, indices.logicalShape, indices.storageShape, axis, outShape, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 2);
            var temps = new List<ComputeTexture>();
            try
            {
                var data = NcnnSentisLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var indices = NcnnSentisLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[1], temps);
                var axis = NcnnSentisLayerUtil.GetInt(layer, 0, "axis", 0);
                NcnnSentisLayerUtil.ValidateGatherElementsShape(layer, data.logicalShape, indices.logicalShape, axis);
                var outShape = indices.logicalShape;
                NcnnSentisLayerUtil.PublishCmdLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisGatherElementsLinearMat(context.commandBuffer, data.texture, data.logicalShape, data.storageShape, indices.texture, indices.logicalShape, indices.storageShape, axis, outShape, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class NcnnArgReduceLayerRepro : NcnnBaseLayerRepro
    {
        private readonly bool _reduceMax;

        public NcnnArgReduceLayerRepro(NcnnLayerTypeKey typeKey, bool reduceMax)
            : base(typeKey, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
            _reduceMax = reduceMax;
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<RenderTexture>();
            try
            {
                var input = NcnnSentisLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var axis = NcnnSentisLayerUtil.GetInt(layer, 0, "axis", 0);
                var keepDims = NcnnSentisLayerUtil.GetInt(layer, 1, "keepdims", NcnnSentisLayerUtil.GetInt(layer, 1, "keepDims", 1)) != 0;
                var selectLast = NcnnSentisLayerUtil.GetInt(layer, 2, "selectLastIndex", 0) != 0;
                var outShape = NcnnSentisLayerUtil.ResolveArgReduceShape(input.logicalShape, axis, keepDims);
                NcnnSentisLayerUtil.PublishRenderLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisArgReduceLinearMat(input.texture, input.logicalShape, input.storageShape, axis, keepDims, selectLast, _reduceMax, outShape, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<ComputeTexture>();
            try
            {
                var input = NcnnSentisLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var axis = NcnnSentisLayerUtil.GetInt(layer, 0, "axis", 0);
                var keepDims = NcnnSentisLayerUtil.GetInt(layer, 1, "keepdims", NcnnSentisLayerUtil.GetInt(layer, 1, "keepDims", 1)) != 0;
                var selectLast = NcnnSentisLayerUtil.GetInt(layer, 2, "selectLastIndex", 0) != 0;
                var outShape = NcnnSentisLayerUtil.ResolveArgReduceShape(input.logicalShape, axis, keepDims);
                NcnnSentisLayerUtil.PublishCmdLinear(owner, layer, context, outShape, (output, storage) =>
                    owner.Ops.SentisArgReduceLinearMat(context.commandBuffer, input.texture, input.logicalShape, input.storageShape, axis, keepDims, selectLast, _reduceMax, outShape, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }

    public sealed class NcnnTopKLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnTopKLayerRepro() : base(NcnnLayerTypes.TopK, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            NcnnSentisLayerUtil.RequireTops(layer, layer.tops > 1 ? 2 : 1);
            var temps = new List<RenderTexture>();
            try
            {
                var input = NcnnSentisLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var spec = ResolveSpec(layer, input.logicalShape);
                var storage = NcnnRepro.ResolveLinearMatStorageShape(spec.outShape);
                var values = owner.RentTempMat(storage.w, storage.h, NcnnRepro.ResolveLinearMatTextureFormat());
                var indices = layer.tops > 1 ? owner.RentTempMat(storage.w, storage.h, NcnnRepro.ResolveLinearMatTextureFormat()) : null;
                owner.Ops.SentisTopKLinearMat(input.texture, input.logicalShape, input.storageShape, spec.axis, spec.k, spec.largest, spec.outShape, storage, values, indices);
                NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], values, spec.outShape, storage);
                if (indices != null)
                    NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[1], indices, spec.outShape, storage);
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            NcnnSentisLayerUtil.RequireTops(layer, layer.tops > 1 ? 2 : 1);
            var temps = new List<ComputeTexture>();
            try
            {
                var input = NcnnSentisLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var spec = ResolveSpec(layer, input.logicalShape);
                var storage = NcnnRepro.ResolveLinearMatStorageShape(spec.outShape);
                var values = owner.RentTempMat(context.commandBuffer, storage.w, storage.h, NcnnRepro.ResolveLinearMatTextureFormat());
                var indices = layer.tops > 1 ? owner.RentTempMat(context.commandBuffer, storage.w, storage.h, NcnnRepro.ResolveLinearMatTextureFormat()) : null;
                owner.Ops.SentisTopKLinearMat(context.commandBuffer, input.texture, input.logicalShape, input.storageShape, spec.axis, spec.k, spec.largest, spec.outShape, storage, values, indices);
                context.blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(values, spec.outShape, storage, owned: true);
                if (context.shapes != null)
                    context.shapes[layer.topNames[0]] = spec.outShape;
                if (indices != null)
                {
                    context.blobs[layer.topNames[1]] = NcnnRepro.CreateCmdTensorRef(indices, spec.outShape, storage, owned: true);
                    if (context.shapes != null)
                        context.shapes[layer.topNames[1]] = spec.outShape;
                }
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static (int axis, int k, bool largest, NcnnRepro.BufferShape outShape) ResolveSpec(NcnnParamModel.Layer layer, NcnnRepro.BufferShape inputShape)
        {
            var axis = NcnnSentisLayerUtil.GetInt(layer, 0, "axis", -1);
            if (!NcnnSentisLayerUtil.TryGetInt(layer, "k", out var k)
                && !NcnnSentisLayerUtil.TryGetInt(layer, 1, out k))
            {
                throw new InvalidOperationException("TopK texture path requires static k param: " + layer.name);
            }
            var largest = NcnnSentisLayerUtil.GetInt(layer, 2, "largest", 1) != 0;
            var outShape = NcnnSentisLayerUtil.ResolveTopKShape(inputShape, axis, k);
            axis = NcnnSentisLayerUtil.NormalizeAxis(axis, inputShape.dims, layer.name);
            return (axis, k, largest, outShape);
        }
    }

    public sealed class NcnnOneHotLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnOneHotLayerRepro() : base(NcnnLayerTypes.OneHot, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<RenderTexture>();
            try
            {
                var indices = NcnnSentisLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var spec = ResolveSpec(layer, indices.logicalShape);
                NcnnSentisLayerUtil.PublishRenderLinear(owner, layer, context, spec.outShape, (output, storage) =>
                    owner.Ops.SentisOneHotLinearMat(indices.texture, indices.logicalShape, indices.storageShape, spec.axis, spec.depth, spec.offValue, spec.onValue, spec.outShape, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<ComputeTexture>();
            try
            {
                var indices = NcnnSentisLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var spec = ResolveSpec(layer, indices.logicalShape);
                NcnnSentisLayerUtil.PublishCmdLinear(owner, layer, context, spec.outShape, (output, storage) =>
                    owner.Ops.SentisOneHotLinearMat(context.commandBuffer, indices.texture, indices.logicalShape, indices.storageShape, spec.axis, spec.depth, spec.offValue, spec.onValue, spec.outShape, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static (int axis, int depth, float offValue, float onValue, NcnnRepro.BufferShape outShape) ResolveSpec(NcnnParamModel.Layer layer, NcnnRepro.BufferShape indicesShape)
        {
            if (!NcnnSentisLayerUtil.TryGetInt(layer, "depth", out var depth)
                && !NcnnSentisLayerUtil.TryGetInt(layer, 1, out depth)
                && !NcnnSentisLayerUtil.TryGetInt(layer, 0, out depth))
            {
                throw new InvalidOperationException("OneHot texture path requires static depth param: " + layer.name);
            }
            var axis = NcnnSentisLayerUtil.GetInt(layer, 2, "axis", -1);
            var onValue = NcnnSentisLayerUtil.GetFloat(layer, 3, "on_value", 1f);
            var offValue = NcnnSentisLayerUtil.GetFloat(layer, 4, "off_value", 0f);
            var outShape = NcnnSentisLayerUtil.ResolveOneHotShape(indicesShape, axis, depth, out var normalizedAxis);
            return (normalizedAxis, depth, offValue, onValue, outShape);
        }
    }

    public sealed class NcnnCumSumLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnCumSumLayerRepro() : base(NcnnLayerTypes.CumSum, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<RenderTexture>();
            try
            {
                var input = NcnnSentisLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var spec = ResolveSpec(layer, input.logicalShape);
                NcnnSentisLayerUtil.PublishRenderLinear(owner, layer, context, input.logicalShape, (output, storage) =>
                    owner.Ops.SentisCumSumLinearMat(input.texture, input.logicalShape, input.storageShape, spec.axis, spec.exclusive, spec.reverse, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, temps);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            NcnnSentisLayerUtil.RequireBottoms(layer, 1);
            var temps = new List<ComputeTexture>();
            try
            {
                var input = NcnnSentisLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var spec = ResolveSpec(layer, input.logicalShape);
                NcnnSentisLayerUtil.PublishCmdLinear(owner, layer, context, input.logicalShape, (output, storage) =>
                    owner.Ops.SentisCumSumLinearMat(context.commandBuffer, input.texture, input.logicalShape, input.storageShape, spec.axis, spec.exclusive, spec.reverse, storage, output));
            }
            finally
            {
                NcnnSentisLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static (int axis, bool exclusive, bool reverse) ResolveSpec(NcnnParamModel.Layer layer, NcnnRepro.BufferShape inputShape)
        {
            if (!NcnnSentisLayerUtil.TryGetInt(layer, "axis", out var axis)
                && !NcnnSentisLayerUtil.TryGetInt(layer, 0, out axis))
            {
                throw new InvalidOperationException("CumSum texture path requires static axis param: " + layer.name);
            }
            axis = NcnnSentisLayerUtil.NormalizeAxis(axis, inputShape.dims, layer.name);
            var exclusive = NcnnSentisLayerUtil.GetInt(layer, 1, "exclusive", 0) != 0;
            var reverse = NcnnSentisLayerUtil.GetInt(layer, 2, "reverse", 0) != 0;
            return (axis, exclusive, reverse);
        }
    }
}
