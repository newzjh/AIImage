using System;
using UnityEngine;

namespace Aexis.Execution
{
    // Bounded temporal statistics pooling. The public NCNN-style operator uses a
    // rank-2 [frames, channels] LinearMat input and emits [1, channels] mean or
    // [1, 2 * channels] mean/std rows without materializing an activation buffer.
    public sealed class AexisStatisticsPoolingLayer : AexisBaseLayer
    {
        public AexisStatisticsPoolingLayer()
            : base(AexisLayerTypes.StatsPooling, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var source = GetRenderInput(context, layer);
            var shape = AexisGraphSession.GetTextureShape(context.textureShapes, source, layer.bottomNames[0]);
            var profile = Validate(source, shape, layer);
            RenderTexture output = null;
            try
            {
                output = owner.RentTempMat(1, profile.outputRows, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.StatisticsPoolingPack4(source.texture, shape.w, shape.h, profile.includeStd, profile.epsilon, output);
                var outputShape = new AexisGraphSession.BufferShape(2, 1, profile.outputRows, 1, 1);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, outputShape, outputShape);
                output = null;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(output);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var source = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var shape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            var profile = Validate(source, shape, layer);
            var output = owner.RentTempMat(context.commandBuffer, 1, profile.outputRows, AexisGraphSession.ResolveLinearMatTextureFormat());
            owner.Ops.StatisticsPoolingPack4(context.commandBuffer, source.texture, shape.w, shape.h, profile.includeStd, profile.epsilon, output);
            var outputShape = new AexisGraphSession.BufferShape(2, 1, profile.outputRows, 1, 1);
            context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, outputShape, outputShape, owned: true, blobName: layer.topNames[0]);
            if (context.shapes != null)
                context.shapes[layer.topNames[0]] = outputShape;
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static AexisGraphSession.TensorRef GetRenderInput(AexisLayerBufferContext context, AexisGraphModel.Layer layer)
        {
            if (layer?.bottomNames == null || layer.bottomNames.Length != 1
                || !context.textureBlobs.TryGetValue(layer.bottomNames[0], out var source)
                || source?.texture == null)
            {
                throw new InvalidOperationException("StatisticsPooling requires one texture-backed input: " + (layer?.name ?? string.Empty));
            }
            return source;
        }

        private static Profile Validate(AexisGraphSession.TensorRef source, AexisGraphSession.BufferShape shape, AexisGraphModel.Layer layer)
        {
            if (source?.texture == null || shape.dims != 2 || shape.w <= 0 || shape.h <= 0 || shape.d != 1 || shape.c != 1
                || !AexisGraphSession.IsStrictLinearMatTexture(source))
            {
                throw new InvalidOperationException("StatisticsPooling requires exact FP32 LinearMat [frames,channels] storage: " + (layer?.name ?? string.Empty));
            }
            return ResolveProfile(shape, layer);
        }

        private static Profile Validate(AexisGraphSession.CmdTensorRef source, AexisGraphSession.BufferShape shape, AexisGraphModel.Layer layer)
        {
            if (source?.texture == null || shape.dims != 2 || shape.w <= 0 || shape.h <= 0 || shape.d != 1 || shape.c != 1
                || !AexisGraphSession.IsStrictLinearMatTexture(source))
            {
                throw new InvalidOperationException("StatisticsPooling requires exact FP32 LinearMat [frames,channels] storage: " + (layer?.name ?? string.Empty));
            }
            return ResolveProfile(shape, layer);
        }

        private static Profile ResolveProfile(AexisGraphSession.BufferShape shape, AexisGraphModel.Layer layer)
        {
            var includeStd = layer.GetInt(0, 1);
            var epsilon = layer.GetFloat(1, 0f);
            if ((includeStd != 0 && includeStd != 1) || float.IsNaN(epsilon) || float.IsInfinity(epsilon) || epsilon < 0f)
                throw new InvalidOperationException("StatisticsPooling requires include_std=0|1 and finite non-negative epsilon: " + (layer?.name ?? string.Empty));
            return new Profile
            {
                includeStd = includeStd != 0,
                epsilon = epsilon,
                outputRows = includeStd != 0 ? checked(shape.h * 2) : shape.h
            };
        }

        private struct Profile
        {
            public bool includeStd;
            public float epsilon;
            public int outputRows;
        }
    }
}
