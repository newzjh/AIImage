using System;

namespace Aexis.Execution
{
    // Schema validation is deliberately separated from the shader implementation.
    // Strict production execution rejects an operator until a matching Pack4 kernel
    // profile is registered; it never reads activations back into a ComputeBuffer.
    public sealed class AexisP1VisionLayer : AexisBaseLayer
    {
        public AexisP1VisionLayer(AexisLayerTypeKey typeKey)
            : base(typeKey, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            AexisP1VisionSchema.Validate(layer);
            return default;
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            AexisP1VisionSchema.Validate(layer);
            ResolveKernel(owner, layer).ExecuteRenderTexture(owner, layer, context);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            AexisP1VisionSchema.Validate(layer);
            ResolveKernel(owner, layer).ExecuteCommandBuffer(owner, layer, context);
        }

        private static IAexisShaderKernelExtension ResolveKernel(AexisGraphSession owner, AexisGraphModel.Layer layer)
        {
            var kernelId = layer?.GetString("aexis.kernel", string.Empty);
            if (string.IsNullOrWhiteSpace(kernelId) && owner?.Model?.extensionDeclarations != null)
            {
                foreach (var declaration in owner.Model.extensionDeclarations)
                {
                    if (declaration != null
                        && string.Equals(declaration.typeName, layer?.typeName, StringComparison.Ordinal)
                        && !string.IsNullOrWhiteSpace(declaration.kernelId))
                    {
                        kernelId = declaration.kernelId;
                        break;
                    }
                }
            }
            if (string.IsNullOrWhiteSpace(kernelId))
                kernelId = layer?.typeName ?? string.Empty;
            if (AexisShaderKernelRegistry.TryGet(kernelId, out var extension))
                return extension;
            throw MissingKernel(layer, "Pack4");
        }

        private static NotSupportedException MissingKernel(AexisGraphModel.Layer layer, string path)
        {
            return new NotSupportedException(
                "P1 visual operator requires a registered texture-native shader kernel"
                + " | layer=" + (layer?.name ?? string.Empty)
                + " | type=" + (layer?.typeName ?? string.Empty)
                + " | path=" + path
                + " | rejected_fallback=ComputeBuffer");
        }
    }

    public static class AexisP1VisionSchema
    {
        public static void Validate(AexisGraphModel.Layer layer)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));
            var type = layer.typeName ?? string.Empty;
            var inputs = layer.bottomNames?.Length ?? 0;
            var outputs = layer.topNames?.Length ?? 0;
            if (outputs != 1)
                throw Invalid(layer, "requires exactly one output.");

            switch (type)
            {
                case "GridSample":
                    RequireInputs(layer, inputs, 2, 2);
                    RequireRange(layer, 0, 1, 3, "sample_type");
                    RequireRange(layer, 1, 1, 3, "padding_mode");
                    RequireRange(layer, 2, 0, 1, "align_corner");
                    break;
                case "DeformableConv2D":
                    RequireInputs(layer, inputs, 2, 3);
                    RequirePositive(layer, 0, "num_output");
                    RequirePositive(layer, 1, "kernel_w");
                    RequirePositive(layer, 6, "weight_data_size");
                    RequirePositiveOrDefault(layer, 11, layer.GetInt(1), "kernel_h");
                    RequirePositiveOrDefault(layer, 2, 1, "dilation_w");
                    RequirePositiveOrDefault(layer, 3, 1, "stride_w");
                    break;
                case "Fold":
                    RequireInputs(layer, inputs, 1, 1);
                    RequirePositive(layer, 1, "kernel_w");
                    RequirePositiveOrDefault(layer, 11, layer.GetInt(1), "kernel_h");
                    RequirePositive(layer, 20, "output_w");
                    RequirePositiveOrDefault(layer, 21, layer.GetInt(20), "output_h");
                    break;
                case "Flip":
                    RequireInputs(layer, inputs, 1, 1);
                    break;
                case "GLU":
                case "Diag":
                    RequireInputs(layer, inputs, 1, 1);
                    break;
                case "Einsum":
                    RequireInputs(layer, inputs, 2, 3);
                    var equation = layer.GetString("equation", layer.GetString("onnx.equation", string.Empty));
                    if (string.IsNullOrWhiteSpace(equation) || equation.IndexOf("->", StringComparison.Ordinal) < 0)
                        throw Invalid(layer, "requires an explicit equation parameter.");
                    break;
                case "SPP":
                    RequireInputs(layer, inputs, 1, 1);
                    RequireRange(layer, 0, 0, 1, "pooling_type");
                    RequirePositive(layer, 1, "pooling_kernel");
                    break;
                case "ROIAlign":
                    RequireInputs(layer, inputs, 2, 2);
                    RequirePositive(layer, 0, "pooled_width");
                    RequirePositive(layer, 1, "pooled_height");
                    RequirePositiveFloatOrDefault(layer, 2, 1f, "spatial_scale");
                    RequireRange(layer, 4, 0, 1, "aligned");
                    break;
                case "ROIPooling":
                case "PSROIPooling":
                    RequireInputs(layer, inputs, 2, 2);
                    RequirePositive(layer, 0, "pooled_width");
                    RequirePositive(layer, 1, "pooled_height");
                    RequirePositiveFloatOrDefault(layer, 2, 1f, "spatial_scale");
                    break;
                case "Proposal":
                    RequireInputs(layer, inputs, 3, 3);
                    RequirePositive(layer, 0, "feat_stride");
                    RequirePositive(layer, 1, "base_size");
                    RequirePositive(layer, 2, "pre_nms_topN");
                    RequirePositive(layer, 3, "post_nms_topN");
                    break;
                case "DetectionOutput":
                    RequireInputs(layer, inputs, 3, 3);
                    RequirePositive(layer, 0, "num_class");
                    RequirePositive(layer, 3, "keep_top_k");
                    break;
                case "YoloDetectionOutput":
                case "Yolov3DetectionOutput":
                case "YoloDetectOut":
                case "Yolov3DetectOut":
                    RequireInputs(layer, inputs, 1, int.MaxValue);
                    RequirePositive(layer, 0, "num_class");
                    RequirePositive(layer, 1, "num_box");
                    break;
                default:
                    throw Invalid(layer, "has no P1 visual schema.");
            }
        }

        private static void RequireInputs(AexisGraphModel.Layer layer, int actual, int minimum, int maximum)
        {
            if (actual < minimum || actual > maximum)
                throw Invalid(layer, "requires " + minimum + (minimum == maximum ? string.Empty : ".." + maximum) + " inputs.");
        }

        private static void RequirePositive(AexisGraphModel.Layer layer, int key, string name)
        {
            if (layer.intParams == null || !layer.intParams.ContainsKey(key) || layer.GetInt(key) <= 0)
                throw Invalid(layer, name + " must be a positive integer parameter.");
        }

        private static void RequirePositiveOrDefault(AexisGraphModel.Layer layer, int key, int defaultValue, string name)
        {
            if (layer.GetInt(key, defaultValue) <= 0)
                throw Invalid(layer, name + " must be positive.");
        }

        private static void RequirePositiveFloatOrDefault(AexisGraphModel.Layer layer, int key, float defaultValue, string name)
        {
            if (layer.GetFloat(key, defaultValue) <= 0f)
                throw Invalid(layer, name + " must be positive.");
        }

        private static void RequireRange(AexisGraphModel.Layer layer, int key, int minimum, int maximum, string name)
        {
            var value = layer.GetInt(key, minimum);
            if (value < minimum || value > maximum)
                throw Invalid(layer, name + " must be in [" + minimum + ", " + maximum + "].");
        }

        private static InvalidOperationException Invalid(AexisGraphModel.Layer layer, string message)
        {
            return new InvalidOperationException(
                "Invalid P1 visual layer"
                + " | type=" + (layer?.typeName ?? string.Empty)
                + " | layer=" + (layer?.name ?? string.Empty)
                + " | " + message);
        }
    }
}
