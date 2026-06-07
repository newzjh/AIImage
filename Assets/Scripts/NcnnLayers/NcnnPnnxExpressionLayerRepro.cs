using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

namespace NcnnCompute
{
    // Runtime ncnn params in this project only keep constant pnnx.Expression nodes.
    // They mainly feed aten::to dtype/non_blocking inputs, so we materialize a tiny
    // scalar/list buffer here and keep the implementation intentionally lightweight.
    public sealed class NcnnPnnxExpressionLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnPnnxExpressionLayerRepro()
            : base(NcnnLayerTypes.PnnxExpression, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
            var outTensor = owner.RentTempTensorBuffer(1, Mathf.Max(1, values.Length));
            outTensor.buffer.SetData(values);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: false,
                context.textureBlobs,
                context.textureShapes,
                context.bufferBlobs,
                context.bufferRefs,
                context.bufferViews,
                context.tempOwned);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
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
            owner.PublishCmdPlaceholder(
                context.commandBuffer,
                layer.topNames[0],
                new NcnnRepro.BufferShape(1, Mathf.Max(1, values.Length), 1, 1, 1),
                context.blobs,
                context.shapes);
        }

        private static float[] ResolveConstantValues(NcnnParamModel.Layer layer)
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

        private static float ParseConstant(string token, NcnnParamModel.Layer layer)
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
