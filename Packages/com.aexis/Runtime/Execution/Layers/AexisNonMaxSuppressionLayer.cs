using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // ONNX NonMaxSuppression with an explicit fixed output capacity. The primary
    // output is padded with [-1,-1,-1] rows and the second output is a texture-only
    // count scalar. Consumers must be count-aware; normal ONNX variable-length
    // semantics are never faked through CPU readback.
    public sealed class AexisNonMaxSuppressionLayer : AexisBaseLayer
    {
        public AexisNonMaxSuppressionLayer() : base(AexisLayerTypes.Nms, false, true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 2);
            AexisShapeIndexLayerUtil.RequireTops(layer, 2);
            var temps = new List<RenderTexture>();
            RenderTexture output = null;
            RenderTexture count = null;
            try
            {
                var boxes = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[0], temps);
                var scores = AexisShapeIndexLayerUtil.GetRenderLinearInput(owner, layer, context, layer.bottomNames[1], temps);
                var profile = ResolveProfile(layer, boxes.logicalShape, scores.logicalShape);
                var outputShape = new AexisGraphSession.BufferShape(2, 3, profile.capacity, 1, 1);
                var countShape = new AexisGraphSession.BufferShape(1, 1, 1, 1, 1);
                output = owner.RentTempMat(3, profile.capacity, AexisGraphSession.ResolveLinearMatTextureFormat());
                count = owner.RentTempMat(1, 1, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.NonMaxSuppressionPack4(boxes.texture, scores.texture, profile.numBoxes, profile.numClasses,
                    profile.capacity, profile.maxOutputPerClass, profile.iouThreshold, profile.scoreThreshold,
                    profile.centerPointBox, output, count);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, outputShape, outputShape);
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

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            AexisShapeIndexLayerUtil.RequireBottoms(layer, 2);
            AexisShapeIndexLayerUtil.RequireTops(layer, 2);
            var temps = new List<ComputeTexture>();
            ComputeTexture output = null;
            ComputeTexture count = null;
            try
            {
                var boxes = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[0], temps);
                var scores = AexisShapeIndexLayerUtil.GetCmdLinearInput(owner, context.commandBuffer, layer, context, layer.bottomNames[1], temps);
                var profile = ResolveProfile(layer, boxes.logicalShape, scores.logicalShape);
                var outputShape = new AexisGraphSession.BufferShape(2, 3, profile.capacity, 1, 1);
                var countShape = new AexisGraphSession.BufferShape(1, 1, 1, 1, 1);
                output = owner.RentTempMat(context.commandBuffer, 3, profile.capacity, AexisGraphSession.ResolveLinearMatTextureFormat());
                count = owner.RentTempMat(context.commandBuffer, 1, 1, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.NonMaxSuppressionPack4(context.commandBuffer, boxes.texture, scores.texture,
                    profile.numBoxes, profile.numClasses, profile.capacity, profile.maxOutputPerClass,
                    profile.iouThreshold, profile.scoreThreshold, profile.centerPointBox, output, count);
                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, outputShape, outputShape, true);
                context.blobs[layer.topNames[1]] = AexisGraphSession.CreateCmdTensorRef(count, countShape, countShape, true);
                context.shapes[layer.topNames[0]] = outputShape;
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

        internal static NmsProfile ResolveProfile(AexisGraphModel.Layer layer, AexisGraphSession.BufferShape boxes, AexisGraphSession.BufferShape scores)
        {
            if (boxes.dims != 2 || boxes.w != 4 || boxes.h <= 0
                || scores.dims != 2 || scores.w != boxes.h || scores.h <= 0)
            {
                throw new InvalidOperationException("NonMaxSuppression requires LinearMat boxes[num_boxes,4] and scores[num_classes,num_boxes] after static batch=1 removal | layer=" + layer.name);
            }
            var capacity = AexisShapeIndexLayerUtil.GetInt(layer, 0, "capacity", 0);
            var maxOutputPerClass = AexisShapeIndexLayerUtil.GetInt(layer, 1, "max_output_boxes_per_class", 0);
            var centerPointBox = AexisShapeIndexLayerUtil.GetInt(layer, 2, "center_point_box", 0);
            var iouThreshold = layer.GetFloat(3, 0f);
            var scoreThreshold = layer.GetFloat(4, float.NegativeInfinity);
            if (capacity <= 0 || maxOutputPerClass < 0 || maxOutputPerClass > capacity
                || centerPointBox < 0 || centerPointBox > 1 || iouThreshold < 0f || iouThreshold > 1f)
            {
                throw new InvalidOperationException("NonMaxSuppression requires capacity>0, 0<=max_output_boxes_per_class<=capacity, center_point_box=0|1, and 0<=iou_threshold<=1 | layer=" + layer.name);
            }
            return new NmsProfile
            {
                numBoxes = boxes.h,
                numClasses = scores.h,
                capacity = capacity,
                maxOutputPerClass = maxOutputPerClass,
                iouThreshold = iouThreshold,
                scoreThreshold = scoreThreshold,
                centerPointBox = centerPointBox != 0
            };
        }

        internal struct NmsProfile
        {
            public int numBoxes;
            public int numClasses;
            public int capacity;
            public int maxOutputPerClass;
            public float iouThreshold;
            public float scoreThreshold;
            public bool centerPointBox;
        }
    }
}
