using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Runtime ncnn params in this project only keep constant pnnx.Expression nodes.
    // They mainly feed aten::to dtype/non_blocking inputs, so we materialize a tiny
    // scalar/list buffer here and keep the implementation intentionally lightweight.
    public sealed class AexisPnnxExpressionLayer : AexisBaseLayer
    {
        public AexisPnnxExpressionLayer()
            : base(AexisLayerTypes.PnnxExpression, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));
            if (context == null)
                throw new ArgumentNullException(nameof(context));

            if (layer.bottomNames != null && layer.bottomNames.Length > 0)
                throw new InvalidOperationException("Dynamic pnnx.Expression is not supported in runtime param path: " + layer.name);

            var values = ResolveConstantValues(layer);
            var logicalShape = ResolveOutputShape(values);
            var storageShape = ResolveStorageShape(values);
            var outTexture = owner.RentTempArray(storageShape.w, storageShape.h, 1, RenderTextureFormat.ARGBHalf);
            owner.Ops.FillScalarTexture(values, outTexture);
            AexisGraphSession.SetTextureBlob(
                context.textureBlobs,
                context.textureShapes,
                layer.topNames[0],
                outTexture,
                logicalShape,
                storageShape);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));
            if (context == null)
                throw new ArgumentNullException(nameof(context));

            if (layer.bottomNames != null && layer.bottomNames.Length > 0)
                throw new InvalidOperationException("Dynamic pnnx.Expression is not supported in command-buffer path: " + layer.name);

            var values = ResolveConstantValues(layer);
            var logicalShape = ResolveOutputShape(values);
            var storageShape = ResolveStorageShape(values);
            var outTexture = owner.RentTempArray(context.commandBuffer, storageShape.w, storageShape.h, 1, RenderTextureFormat.ARGBHalf);
            owner.Ops.FillScalarTexture(context.commandBuffer, values, outTexture);
            context.blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outTexture,
                width = outTexture.width,
                height = outTexture.height,
                packs = 1,
                refs = 1,
                owned = true,
                hasLogicalShape = true,
                logicalShape = logicalShape,
                hasStorageShape = true,
                storageShape = storageShape
            };
            if (context.shapes != null)
                context.shapes[layer.topNames[0]] = logicalShape;
        }

        private static AexisGraphSession.BufferShape ResolveOutputShape(float[] values)
        {
            return new AexisGraphSession.BufferShape(1, Mathf.Max(1, values?.Length ?? 0), 1, 1, 1);
        }

        private static AexisGraphSession.BufferShape ResolveStorageShape(float[] values)
        {
            return new AexisGraphSession.BufferShape(3, Mathf.Max(1, values?.Length ?? 0), 1, 1, 1);
        }

        // The strict CommandBuffer planner must prove the same narrow constant contract that
        // this layer dispatches.  It intentionally does not accept pnnx expressions with
        // runtime inputs, nor lists wider than the FillScalarTexture kernel's four lanes.
        internal static bool TryResolveConstantValueCount(AexisGraphModel.Layer layer, out int count, out string reason)
        {
            count = 0;
            reason = null;
            if (layer == null)
            {
                reason = "The pnnx.Expression layer is missing.";
                return false;
            }
            if (layer.bottomNames != null && layer.bottomNames.Length > 0)
            {
                reason = "Only a pnnx.Expression without runtime input blobs has a CommandBuffer scalar-fill path.";
                return false;
            }

            try
            {
                var values = ResolveConstantValues(layer);
                count = Mathf.Max(1, values?.Length ?? 0);
                if (count > 4)
                {
                    reason = "The CommandBuffer scalar-fill kernel supports at most four constant values.";
                    return false;
                }
                return true;
            }
            catch (Exception exception)
            {
                reason = "The pnnx.Expression constant cannot be parsed: " + exception.Message;
                return false;
            }
        }

        private static float[] ResolveConstantValues(AexisGraphModel.Layer layer)
        {
            var expr = layer?.GetString("expr", null);
            if (string.IsNullOrWhiteSpace(expr))
                return new[] { 0f };

            expr = expr.Trim();
            if (expr.Length >= 2 && expr[0] == '"' && expr[expr.Length - 1] == '"')
                expr = expr.Substring(1, expr.Length - 2);

            if (expr.Length >= 2 && expr[0] == '[' && expr[expr.Length - 1] == ']')
            {
                var body = expr.Substring(1, expr.Length - 2);
                var parts = body.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 0)
                    return new[] { 0f };

                var values = new float[parts.Length];
                for (var i = 0; i < parts.Length; i++)
                    values[i] = ParseConstant(parts[i].Trim(), layer);
                return values;
            }

            return new[] { ParseConstant(expr, layer) };
        }

        private static float ParseConstant(string token, AexisGraphModel.Layer layer)
        {
            if (string.Equals(token, "False", StringComparison.OrdinalIgnoreCase))
                return 0f;
            if (string.Equals(token, "True", StringComparison.OrdinalIgnoreCase))
                return 1f;

            if (int.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out var i))
                return i;
            if (float.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out var f))
                return f;

            throw new InvalidOperationException("Unsupported constant pnnx.Expression token: " + token + " | " + layer?.name);
        }
    }
}
