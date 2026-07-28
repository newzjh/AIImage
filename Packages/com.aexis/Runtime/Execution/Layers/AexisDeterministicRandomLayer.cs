using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // P2 deterministic RNG profile. It deliberately has no CPU state and never
    // materializes an activation into a ComputeBuffer: the coordinate/seed hash is
    // evaluated directly into Pack4 Texture2DArray storage.
    public sealed class AexisDeterministicRandomLayer : AexisBaseLayer
    {
        private const int UniformLike = 0;
        private const int NormalLike = 1;
        private const int Bernoulli = 2;

        public AexisDeterministicRandomLayer()
            : base(AexisLayerTypes.RandomLike, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            ValidateLayer(layer);
            if (IsStaticRandom(layer))
            {
                var profile = ResolveProfile(layer);
                var staticShape = ResolveStaticOutputShape(layer);
                RenderTexture staticOutput = null;
                try
                {
                    var packs = PackCount(staticShape);
                    staticOutput = owner.RentTempArray(staticShape.w, staticShape.h, SliceCount(staticShape, packs), owner.ResolveActivationTextureFormat(layer, staticShape.dims));
                    owner.Ops.StaticRandomPack4(staticOutput, profile.mode, profile.seed, profile.parameter0, profile.parameter1, packs, staticShape.c);
                    AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], staticOutput, staticShape, staticShape);
                    staticOutput = null;
                }
                finally
                {
                    if (staticOutput != null) owner.ReturnTempArray(staticOutput);
                }
                return;
            }
            if (!AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var source, out var shape)
                || source?.texture == null)
                throw new InvalidOperationException("Deterministic RNG requires a texture-backed Pack4 input: " + layer.name);
            if (!AexisGraphSession.MatchesPack4TextureStorage(source, shape))
                throw new InvalidOperationException("Deterministic RNG rejects non-Pack4 activation storage: " + layer.name);

            RenderTexture output = null;
            try
            {
                var packs = PackCount(shape);
                output = owner.RentTempArray(shape.w, shape.h, SliceCount(shape, packs), owner.ResolveActivationTextureFormat(shape.dims));
                var profile = ResolveProfile(layer);
                owner.Ops.DeterministicRandomPack4(source.texture, profile.mode, profile.seed, profile.parameter0, profile.parameter1, packs, shape.c, output);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, shape, shape);
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
            ValidateLayer(layer);
            if (IsStaticRandom(layer))
            {
                var profile = ResolveProfile(layer);
                var staticShape = ResolveStaticOutputShape(layer);
                ComputeTexture staticOutput = null;
                try
                {
                    var packs = PackCount(staticShape);
                    staticOutput = owner.RentTempArray(context.commandBuffer, staticShape.w, staticShape.h, SliceCount(staticShape, packs), owner.ResolveActivationTextureFormat(layer, staticShape.dims));
                    owner.Ops.StaticRandomPack4(context.commandBuffer, staticOutput, profile.mode, profile.seed, profile.parameter0, profile.parameter1, packs, staticShape.c);
                    context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(staticOutput, staticShape, staticShape, owned: true, blobName: layer.topNames[0]);
                    context.shapes[layer.topNames[0]] = staticShape;
                    staticOutput = null;
                }
                finally
                {
                    if (staticOutput != null) owner.ReturnTempArray(context.commandBuffer, staticOutput);
                }
                return;
            }
            var source = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var shape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (source?.texture == null || !AexisGraphSession.MatchesPack4TextureStorage(source, shape))
                throw new InvalidOperationException("Deterministic RNG CommandBuffer path requires exact Pack4 Texture2DArray storage: " + layer.name);

            ComputeTexture output = null;
            try
            {
                var packs = PackCount(shape);
                output = owner.RentTempArray(context.commandBuffer, shape.w, shape.h, SliceCount(shape, packs), owner.ResolveActivationTextureFormat(shape.dims));
                var profile = ResolveProfile(layer);
                owner.Ops.DeterministicRandomPack4(context.commandBuffer, source.texture, profile.mode, profile.seed, profile.parameter0, profile.parameter1, packs, shape.c, output);
                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, shape, shape, owned: true);
                context.shapes[layer.topNames[0]] = shape;
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
            var type = ResolveOperatorType(layer);
            var staticRandom = type == "RandomUniform" || type == "RandomNormal";
            if (type != "RandomUniformLike" && type != "RandomNormalLike" && type != "Bernoulli" && !staticRandom)
                throw new NotSupportedException("Unsupported deterministic RNG operator: " + type);
            if (layer.topNames == null || layer.topNames.Length != 1
                || (staticRandom ? layer.bottomNames != null && layer.bottomNames.Length != 0 : layer.bottomNames == null || layer.bottomNames.Length != 1))
            {
                throw new InvalidOperationException(staticRandom
                    ? "Static deterministic RNG requires no activation input and one output: " + layer.name
                    : "Deterministic RNG profile requires exactly one texture input and one output: " + layer.name);
            }
            if (layer.intParams == null || !layer.intParams.ContainsKey(0))
                throw new InvalidOperationException("Deterministic RNG requires explicit immutable seed parameter 0: " + layer.name);
            var profile = ResolveProfile(layer);
            if (profile.mode == UniformLike && !(profile.parameter1 > profile.parameter0))
                throw new InvalidOperationException("RandomUniformLike requires high > low: " + layer.name);
            if (profile.mode == NormalLike && !(profile.parameter1 > 0f))
                throw new InvalidOperationException("RandomNormalLike requires scale > 0: " + layer.name);
            if (staticRandom)
                ResolveStaticOutputShape(layer);
        }

        private static RandomProfile ResolveProfile(AexisGraphModel.Layer layer)
        {
            var type = ResolveOperatorType(layer);
            if (type == "RandomUniformLike" || type == "RandomUniform")
                return new RandomProfile { mode = UniformLike, seed = layer.GetInt(0), parameter0 = layer.GetFloat(1, 0f), parameter1 = layer.GetFloat(2, 1f) };
            if (type == "RandomNormalLike" || type == "RandomNormal")
                return new RandomProfile { mode = NormalLike, seed = layer.GetInt(0), parameter0 = layer.GetFloat(1, 0f), parameter1 = layer.GetFloat(2, 1f) };
            return new RandomProfile { mode = Bernoulli, seed = layer.GetInt(0), parameter0 = 0f, parameter1 = 1f };
        }

        private static string ResolveOperatorType(AexisGraphModel.Layer layer)
        {
            if (layer?.stringParams != null
                && layer.stringParams.TryGetValue("aexis.random.operator", out var importedOperator)
                && !string.IsNullOrWhiteSpace(importedOperator))
                return importedOperator;
            return layer?.typeName ?? string.Empty;
        }

        internal static bool IsStaticRandom(AexisGraphModel.Layer layer)
        {
            var type = ResolveOperatorType(layer);
            return type == "RandomUniform" || type == "RandomNormal";
        }

        internal static AexisGraphSession.BufferShape ResolveStaticOutputShape(AexisGraphModel.Layer layer)
        {
            var dims = layer.GetInt(10, 0);
            var w = layer.GetInt(11, 0);
            var h = layer.GetInt(12, 0);
            var d = layer.GetInt(13, 1);
            var c = layer.GetInt(14, 0);
            if ((dims != 3 && dims != 4) || w <= 0 || h <= 0 || d <= 0 || c <= 0)
                throw new InvalidOperationException("Static RandomUniform/RandomNormal requires immutable rank-3/4 Pack4 shape parameters 10:dims,11:w,12:h,13:d,14:c: " + layer.name);
            if (dims == 3 && d != 1)
                throw new InvalidOperationException("Static rank-3 RandomUniform/RandomNormal requires d=1: " + layer.name);
            return new AexisGraphSession.BufferShape(dims, w, h, d, c);
        }

        private static int PackCount(AexisGraphSession.BufferShape shape) => Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f));
        private static int SliceCount(AexisGraphSession.BufferShape shape, int packs) => shape.dims == 4 ? Mathf.Max(1, shape.d) * packs : packs;

        private struct RandomProfile
        {
            public int mode;
            public int seed;
            public float parameter0;
            public float parameter1;
        }
    }
}
