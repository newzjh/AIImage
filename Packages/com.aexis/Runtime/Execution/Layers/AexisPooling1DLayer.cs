using System;
using UnityEngine;

namespace Aexis.Execution
{
    public sealed class AexisPooling1DLayer : AexisBaseLayer
    {
        internal readonly struct Pooling1DSpec
        {
            public readonly int type;
            public readonly int kernel;
            public readonly int stride;
            public readonly int padLeft;
            public readonly bool includePad;
            public readonly bool adaptive;
            public readonly bool global;
            public readonly int outputWidth;

            public Pooling1DSpec(int type, int kernel, int stride, int padLeft, bool includePad, bool adaptive, bool global, int outputWidth)
            {
                this.type = type;
                this.kernel = kernel;
                this.stride = stride;
                this.padLeft = padLeft;
                this.includePad = includePad;
                this.adaptive = adaptive;
                this.global = global;
                this.outputWidth = outputWidth;
            }
        }

        public AexisPooling1DLayer() : base(AexisLayerTypes.Pooling1D, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out var src, out var shape)
                || !AexisGraphSession.MatchesPack4TextureStorage(src, shape))
                throw new InvalidOperationException("Pooling1D requires a descriptor-valid Pack4 texture input: " + layer.name);
            if (!TryResolveSpec(layer, shape, out var spec, out var reason))
                throw new InvalidOperationException("Pooling1D profile rejected layer " + layer.name + ": " + reason);

            var output = owner.RentTempArray(spec.outputWidth, 1, src.packs, src.texture.format);
            owner.Ops.Pooling1DPack4(src.texture, shape.w, src.packs, spec.type, spec.kernel, spec.stride, spec.padLeft, spec.includePad, spec.adaptive || spec.global, output);
            var outputShape = new AexisGraphSession.BufferShape(3, spec.outputWidth, 1, 1, shape.c);
            AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, outputShape);
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var src = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var shape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (!AexisGraphSession.MatchesPack4TextureStorage(src, shape))
                throw new InvalidOperationException("Pooling1D requires descriptor-valid Pack4 command-buffer storage: " + layer.name);
            if (!TryResolveSpec(layer, shape, out var spec, out var reason))
                throw new InvalidOperationException("Pooling1D profile rejected layer " + layer.name + ": " + reason);

            var output = owner.RentTempArray(context.commandBuffer, spec.outputWidth, 1, src.packs, src.texture.format);
            owner.Ops.Pooling1DPack4(context.commandBuffer, src.texture, shape.w, src.packs, spec.type, spec.kernel, spec.stride, spec.padLeft, spec.includePad, spec.adaptive || spec.global, output);
            var outputShape = new AexisGraphSession.BufferShape(3, spec.outputWidth, 1, 1, shape.c);
            context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, outputShape, outputShape, owned: true, blobName: layer.topNames[0]);
            if (context.shapes != null) context.shapes[layer.topNames[0]] = outputShape;
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        internal static bool TryResolveSpec(
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape shape,
            out Pooling1DSpec spec,
            out string reason)
        {
            spec = default;
            reason = null;
            if (shape.dims != 3 || shape.d != 1 || shape.h != 1 || shape.w <= 0 || shape.c <= 0)
            {
                reason = "input must be a static dims=3 Pack4 activation with logical height=1";
                return false;
            }

            var type = layer.GetInt(0, 0);
            var kernel = layer.GetInt(1, 0);
            var stride = layer.GetInt(2, 1);
            var left = layer.GetInt(3, 0);
            var right = layer.GetInt(14, left);
            var global = layer.GetInt(4, 0) != 0;
            var padMode = layer.GetInt(5, 0);
            var includePadValue = layer.GetInt(6, 0);
            var adaptive = layer.GetInt(7, 0) != 0;
            var adaptiveOut = layer.GetInt(8, 0);
            if ((type != 0 && type != 1) || padMode < 0 || padMode > 3 || (includePadValue != 0 && includePadValue != 1))
            {
                reason = "pooling_type must be max/average, pad_mode must be 0..3, and include_pad must be 0/1";
                return false;
            }

            if (global)
            {
                spec = new Pooling1DSpec(type, shape.w, 1, 0, false, false, true, 1);
                return true;
            }
            if (adaptive)
            {
                if (adaptiveOut <= 0)
                {
                    reason = "adaptive pooling requires a positive out_w parameter 8";
                    return false;
                }
                spec = new Pooling1DSpec(type, 1, 1, 0, false, true, false, adaptiveOut);
                return true;
            }
            if (kernel <= 0 || stride <= 0 || left < 0 || right < 0)
            {
                reason = "kernel/stride must be positive and explicit padding must be non-negative";
                return false;
            }

            int effectiveLeft;
            int paddedWidth;
            if (padMode == 0)
            {
                effectiveLeft = left;
                paddedWidth = shape.w + left + right;
                if (paddedWidth < kernel)
                {
                    reason = "kernel exceeds the explicitly padded input";
                    return false;
                }
                var tail = (paddedWidth - kernel) % stride;
                if (tail != 0) paddedWidth += stride - tail;
            }
            else if (padMode == 1)
            {
                effectiveLeft = left;
                paddedWidth = shape.w + left + right;
            }
            else
            {
                var total = kernel + ((shape.w - 1) / stride) * stride - shape.w;
                if (total < 0) total = 0;
                effectiveLeft = padMode == 2 ? total / 2 : total - total / 2;
                paddedWidth = shape.w + total;
            }

            if (paddedWidth < kernel)
            {
                reason = "pooling resolves a non-positive output width";
                return false;
            }
            var outputWidth = (paddedWidth - kernel) / stride + 1;
            spec = new Pooling1DSpec(type, kernel, stride, effectiveLeft, includePadValue != 0, false, false, outputWidth);
            return true;
        }
    }
}
