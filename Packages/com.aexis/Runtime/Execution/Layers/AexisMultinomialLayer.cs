using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Bounded ONNX Multinomial. Categorical logits and emitted Int32 indices use
    // exact Pack4 Texture2DArray storage throughout execution. The physical FP32
    // lanes are exact for the admitted category range and never leave the GPU.
    public sealed class AexisMultinomialLayer : AexisBaseLayer
    {
        internal const int MaxBatch = 256;
        internal const int MaxClasses = 4096;
        internal const int MaxSamples = 256;

        public AexisMultinomialLayer()
            : base(AexisLayerTypes.Multinomial, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            if (context == null) throw new ArgumentNullException(nameof(context));
            ValidateLayer(layer);
            if (!AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var source, out var input)
                || source?.texture == null)
                throw new InvalidOperationException("Multinomial requires a texture-backed Pack4 logits input: " + layer.name);
            var profile = ValidateInput(source, input, layer.name);
            profile.samples = layer.GetInt(0);
            RenderTexture output = null;
            try
            {
                output = owner.RentTempArray(1, profile.batch, PackCount(profile.samples), owner.ResolveActivationTextureFormat(layer, 3));
                owner.Ops.MultinomialPack4(source.texture, layer.GetInt(1), profile.batch, profile.classes, profile.samples, output);
                var outputShape = OutputShape(profile);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, outputShape, outputShape);
                output = null;
            }
            finally
            {
                if (output != null) owner.ReturnTempArray(output);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            if (context == null) throw new ArgumentNullException(nameof(context));
            ValidateLayer(layer);
            var source = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var input = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            var profile = ValidateInput(source, input, layer.name);
            profile.samples = layer.GetInt(0);
            ComputeTexture output = null;
            try
            {
                output = owner.RentTempArray(context.commandBuffer, 1, profile.batch, PackCount(profile.samples), owner.ResolveActivationTextureFormat(layer, 3));
                owner.Ops.MultinomialPack4(context.commandBuffer, source.texture, layer.GetInt(1), profile.batch, profile.classes, profile.samples, output);
                var outputShape = OutputShape(profile);
                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, outputShape, outputShape, owned: true, blobName: layer.topNames[0]);
                context.shapes[layer.topNames[0]] = outputShape;
                output = null;
            }
            finally
            {
                if (output != null) owner.ReturnTempArray(context.commandBuffer, output);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        internal static void ValidateLayer(AexisGraphModel.Layer layer)
        {
            if (layer == null) throw new ArgumentNullException(nameof(layer));
            if (layer.bottomNames == null || layer.bottomNames.Length != 1 || layer.topNames == null || layer.topNames.Length != 1)
                throw new InvalidOperationException("Bounded Multinomial requires exactly one logits input and one output: " + layer.name);
            var samples = layer.GetInt(0, 0);
            if (samples <= 0 || samples > MaxSamples)
                throw new InvalidOperationException("Bounded Multinomial sample_size must be in [1," + MaxSamples + "]: " + layer.name);
            if (layer.intParams == null || !layer.intParams.ContainsKey(1))
                throw new InvalidOperationException("Bounded Multinomial requires an explicit immutable integer seed parameter 1: " + layer.name);
        }

        internal static Profile ValidateInput(AexisGraphSession.TensorRef source, AexisGraphSession.BufferShape input, string layerName)
        {
            if (source?.texture == null || !AexisGraphSession.MatchesPack4TextureStorage(source, input))
                throw new InvalidOperationException("Bounded Multinomial requires exact Pack4 Texture2DArray logits storage: " + layerName);
            return ValidateShape(input, layerName);
        }

        internal static Profile ValidateInput(AexisGraphSession.CmdTensorRef source, AexisGraphSession.BufferShape input, string layerName)
        {
            if (source?.texture == null || !AexisGraphSession.MatchesPack4TextureStorage(source, input))
                throw new InvalidOperationException("Bounded Multinomial requires exact Pack4 Texture2DArray logits storage: " + layerName);
            return ValidateShape(input, layerName);
        }

        internal static Profile ValidateShape(AexisGraphSession.BufferShape input, string layerName)
        {
            // ONNX [batch,classes] is represented as [w=1,h=batch,c=classes] so
            // class values occupy Pack4 lanes and each output sample pack is owned
            // by exactly one compute invocation.
            if (input.dims != 3 || input.w != 1 || input.d != 1
                || input.h <= 0 || input.h > MaxBatch || input.c <= 1 || input.c > MaxClasses)
            {
                throw new InvalidOperationException(
                    "Bounded Multinomial requires static Pack4 logits [w=1,batch=1.." + MaxBatch
                    + ",d=1,classes=2.." + MaxClasses + "]: " + layerName);
            }
            return new Profile { batch = input.h, classes = input.c };
        }

        internal static AexisGraphSession.BufferShape OutputShape(Profile profile)
        {
            return new AexisGraphSession.BufferShape(3, 1, profile.batch, 1, profile.samples);
        }

        internal static int PackCount(int channels) => Mathf.Max(1, (channels + 3) / 4);

        internal struct Profile
        {
            public int batch;
            public int classes;
            public int samples;
        }
    }
}
