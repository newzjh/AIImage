using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Fixed-capacity data-dependent operations. The second output is a GPU-resident
    // count texture, so consumers never need to materialize a texture into a buffer.
    public sealed class AexisNonZeroLayer : AexisBaseLayer
    {
        public AexisNonZeroLayer() : base(AexisLayerTypes.NonZero, false, true) { }
        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context) => ExecuteRender(owner, layer, context, false);
        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context) => ExecuteCmd(owner, layer, context, false);
        internal static void ExecuteRender(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context, bool compress)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, compress ? 2 : 1); AexisShapeIndexLayerUtil.RequireTops(layer, 2);
            var temps = new List<RenderTexture>();
            RenderTexture output = null;
            RenderTexture count = null;
            try
            {
                var values = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                RequireRankOne(values.logicalShape, compress ? "Compress" : "NonZero", layer);
                var capacity = Capacity(layer, values.logicalShape);
                AexisShapeIndexLayerUtil.RenderLinearInput condition = default;
                if (compress)
                {
                    condition = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[1], temps);
                    RequireRankOne(condition.logicalShape, "Compress condition", layer);
                    if (AexisShapeIndexLayerUtil.ElementCount(condition.logicalShape) != AexisShapeIndexLayerUtil.ElementCount(values.logicalShape))
                        throw new InvalidOperationException("Compress texture path requires data and condition lengths to match | layer=" + layer.name);
                }

                var outShape = compress
                    ? new AexisGraphSession.BufferShape(1, capacity, 1, 1, 1)
                    : new AexisGraphSession.BufferShape(2, capacity, 1, 1, 1);
                var storage = AexisGraphSession.ResolveLinearMatStorageShape(outShape);
                var countShape = new AexisGraphSession.BufferShape(1, 1, 1, 1, 1);
                output = owner.RentTempMat(storage.w, storage.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                count = owner.RentTempMat(1, 1, AexisGraphSession.ResolveLinearMatTextureFormat());
                if (compress)
                    owner.Ops.SentisCompressLinearMat(values.texture, values.logicalShape, values.storageShape, condition.texture, condition.logicalShape, condition.storageShape, capacity, output, count);
                else
                    owner.Ops.SentisNonZeroLinearMat(values.texture, values.logicalShape, values.storageShape, capacity, output, count);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, outShape, storage);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[1], count, countShape, countShape);
                output = null;
                count = null;
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, temps);
                if (output != null) owner.ReturnTempArray(output);
                if (count != null) owner.ReturnTempArray(count);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }
        internal static void ExecuteCmd(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context, bool compress)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, compress ? 2 : 1); AexisShapeIndexLayerUtil.RequireTops(layer, 2);
            var temps = new List<ComputeTexture>();
            ComputeTexture output = null;
            ComputeTexture count = null;
            try
            {
                var values = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                RequireRankOne(values.logicalShape, compress ? "Compress" : "NonZero", layer);
                var capacity = Capacity(layer, values.logicalShape);
                AexisShapeIndexLayerUtil.CmdLinearInput condition = default;
                if (compress)
                {
                    condition = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[1], temps);
                    RequireRankOne(condition.logicalShape, "Compress condition", layer);
                    if (AexisShapeIndexLayerUtil.ElementCount(condition.logicalShape) != AexisShapeIndexLayerUtil.ElementCount(values.logicalShape))
                        throw new InvalidOperationException("Compress texture path requires data and condition lengths to match | layer=" + layer.name);
                }

                var outShape = compress
                    ? new AexisGraphSession.BufferShape(1, capacity, 1, 1, 1)
                    : new AexisGraphSession.BufferShape(2, capacity, 1, 1, 1);
                var storage = AexisGraphSession.ResolveLinearMatStorageShape(outShape);
                var countShape = new AexisGraphSession.BufferShape(1, 1, 1, 1, 1);
                output = owner.RentTempMat(context.commandBuffer, storage.w, storage.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                count = owner.RentTempMat(context.commandBuffer, 1, 1, AexisGraphSession.ResolveLinearMatTextureFormat());
                if (compress)
                    owner.Ops.SentisCompressLinearMat(context.commandBuffer, values.texture, values.logicalShape, values.storageShape, condition.texture, condition.logicalShape, condition.storageShape, capacity, output, count);
                else
                    owner.Ops.SentisNonZeroLinearMat(context.commandBuffer, values.texture, values.logicalShape, values.storageShape, capacity, output, count);
                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, outShape, storage, true);
                context.blobs[layer.topNames[1]] = AexisGraphSession.CreateCmdTensorRef(count, countShape, countShape, true);
                context.shapes[layer.topNames[0]] = outShape;
                context.shapes[layer.topNames[1]] = countShape;
                output = null;
                count = null;
            }
            finally
            {
                AexisShapeIndexLayerUtil.ReturnTemps(owner, context.commandBuffer, temps);
                if (output != null) owner.ReturnTempArray(context.commandBuffer, output);
                if (count != null) owner.ReturnTempArray(context.commandBuffer, count);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
        internal static int Capacity(AexisGraphModel.Layer layer, AexisGraphSession.BufferShape inputShape) { var capacity = AexisShapeIndexLayerUtil.GetInt(layer, 30, "capacity", 0); var required = AexisShapeIndexLayerUtil.ElementCount(inputShape); if (capacity <= 0) throw new InvalidOperationException("Data-dependent texture operation requires positive capacity | layer=" + layer.name); if (capacity < required) throw new InvalidOperationException("Data-dependent texture operation capacity cannot truncate the static result | layer=" + layer.name + " | capacity=" + capacity + " | required=" + required); return capacity; }
        private static void RequireRankOne(AexisGraphSession.BufferShape shape, string op, AexisGraphModel.Layer layer) { if (shape.dims != 1) throw new InvalidOperationException(op + " texture path requires rank-1 LinearMat data | layer=" + layer.name); }
    }

    public sealed class AexisCompressLayer : AexisBaseLayer
    {
        public AexisCompressLayer() : base(AexisLayerTypes.Compress, false, true) { }
        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context) => AexisNonZeroLayer.ExecuteRender(owner, layer, context, true);
        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context) => AexisNonZeroLayer.ExecuteCmd(owner, layer, context, true);
    }

    public sealed class AexisGatherNDLayer : AexisBaseLayer
    {
        public AexisGatherNDLayer() : base(AexisLayerTypes.GatherND, false, true) { }
        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context) { AexisShapeIndexLayerUtil.RequireBottoms(layer, 2); var temps = new List<RenderTexture>(); try { var data = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps); var indices = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[1], temps); var outputShape = RequireLinearIndices(layer, data.logicalShape, indices.logicalShape); AexisShapeIndexLayerUtil.PublishRenderLinear(owner, layer, context, outputShape, (output, storage) => owner.Ops.SentisGatherNdLinearMat(data.texture, data.logicalShape, data.storageShape, indices.texture, indices.logicalShape, indices.storageShape, outputShape, storage, output)); } finally { AexisShapeIndexLayerUtil.ReturnTemps(owner, temps); } owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames); }
        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context) { AexisShapeIndexLayerUtil.RequireBottoms(layer, 2); var temps = new List<ComputeTexture>(); try { var data = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps); var indices = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[1], temps); var outputShape = RequireLinearIndices(layer, data.logicalShape, indices.logicalShape); AexisShapeIndexLayerUtil.PublishCmdLinear(owner, layer, context, outputShape, (output, storage) => owner.Ops.SentisGatherNdLinearMat(context.commandBuffer, data.texture, data.logicalShape, data.storageShape, indices.texture, indices.logicalShape, indices.storageShape, outputShape, storage, output)); } finally { AexisShapeIndexLayerUtil.ReturnTemps(owner, context.commandBuffer, temps); } owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes); }

        private static AexisGraphSession.BufferShape RequireLinearIndices(AexisGraphModel.Layer layer, AexisGraphSession.BufferShape data, AexisGraphSession.BufferShape indices)
        {
            if (data.dims != 1 || indices.dims != 2 || indices.w != 1
                || AexisShapeIndexLayerUtil.GetInt(layer, 1, "index_depth", 1) != 1
                || AexisShapeIndexLayerUtil.GetInt(layer, 0, "batch_dims", 0) != 0
                || AexisShapeIndexLayerUtil.GetInt(layer, int.MinValue, "indices_in_range", 0) != 1
                || !string.Equals(layer.GetString("index_dtype", null), "Int32", StringComparison.Ordinal))
                throw new InvalidOperationException("GatherND texture path requires rank-1 data, batch_dims=0, rank-2 [N,1] Int32 indices, index_depth=1, and indices_in_range=1 exporter proof | layer=" + layer.name);
            return new AexisGraphSession.BufferShape(1, indices.h, 1, 1, 1);
        }
    }

    public sealed class AexisScatterLayer : AexisBaseLayer
    {
        public AexisScatterLayer(AexisLayerTypeKey typeKey) : base(typeKey, false, true) { }
        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context) { AexisShapeIndexLayerUtil.RequireBottoms(layer, 3); var temps = new List<RenderTexture>(); try { var data = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps); var indices = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[1], temps); var updates = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[2], temps); var updateCount = RequireUniqueLinear(layer, data.logicalShape, indices.logicalShape, updates.logicalShape); AexisShapeIndexLayerUtil.PublishRenderLinear(owner, layer, context, data.logicalShape, (output, _) => owner.Ops.SentisScatterLinearMat(data.texture, data.logicalShape, data.storageShape, indices.texture, indices.logicalShape, indices.storageShape, updates.texture, updates.logicalShape, updates.storageShape, updateCount, output)); } finally { AexisShapeIndexLayerUtil.ReturnTemps(owner, temps); } owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames); }
        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context) { AexisShapeIndexLayerUtil.RequireBottoms(layer, 3); var temps = new List<ComputeTexture>(); try { var data = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps); var indices = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[1], temps); var updates = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[2], temps); var updateCount = RequireUniqueLinear(layer, data.logicalShape, indices.logicalShape, updates.logicalShape); AexisShapeIndexLayerUtil.PublishCmdLinear(owner, layer, context, data.logicalShape, (output, _) => owner.Ops.SentisScatterLinearMat(context.commandBuffer, data.texture, data.logicalShape, data.storageShape, indices.texture, indices.logicalShape, indices.storageShape, updates.texture, updates.logicalShape, updates.storageShape, updateCount, output)); } finally { AexisShapeIndexLayerUtil.ReturnTemps(owner, context.commandBuffer, temps); } owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes); }

        private static int RequireUniqueLinear(AexisGraphModel.Layer layer, AexisGraphSession.BufferShape data, AexisGraphSession.BufferShape indices, AexisGraphSession.BufferShape updates)
        {
            var operatorName = string.IsNullOrWhiteSpace(layer.typeName) ? layer.type.ToString() : layer.typeName;
            var scatterNd = string.Equals(operatorName, "ScatterND", StringComparison.Ordinal);
            var axis = AexisShapeIndexLayerUtil.GetInt(layer, 0, "axis", 0);
            var validShapes = data.dims == 1 && updates.dims == 1
                && (scatterNd
                    ? indices.dims == 2 && indices.w == 1 && indices.h == updates.w
                    : indices.dims == 1 && indices.w == updates.w && axis == 0);
            var reduction = layer.GetString("reduction", null);
            if (!validShapes
                || AexisShapeIndexLayerUtil.GetInt(layer, -1, "unique_indices", 0) != 1
                || AexisShapeIndexLayerUtil.GetInt(layer, int.MinValue, "indices_in_range", 0) != 1
                || !string.Equals(layer.GetString("index_dtype", null), "Int32", StringComparison.Ordinal)
                || !string.Equals(reduction, "none", StringComparison.Ordinal))
                throw new InvalidOperationException((scatterNd
                    ? "ScatterND texture path requires rank-1 data/updates and in-range unique rank-2 [N,1] Int32 indices"
                    : "Scatter/ScatterElements texture path requires rank-1 data/updates, axis=0, and in-range unique rank-1 Int32 indices")
                    + " with reduction=none | layer=" + layer.name);
            return updates.w;
        }
    }
}
