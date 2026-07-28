using System;
using UnityEngine;

namespace Aexis.Execution
{
    // A bounded, real-valued DFT/iDFT profile. Spectrogram emits one-sided complex
    // bins into LinearMat texture storage; inverse reconstructs and overlap-averages
    // matching frames. Activations never leave GPU textures.
    public sealed class AexisSpectrogramLayer : AexisBaseLayer
    {
        private readonly bool _inverse;

        public AexisSpectrogramLayer(bool inverse)
            : base(inverse ? AexisLayerTypes.InvSpectrogram : AexisLayerTypes.Spectrogram, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
            _inverse = inverse;
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var source = GetRenderInput(context, layer);
            var shape = AexisGraphSession.GetTextureShape(context.textureShapes, source, layer.bottomNames[0]);
            var profile = ResolveProfile(source, shape, layer, _inverse);
            RenderTexture output = null;
            try
            {
                output = owner.RentTempMat(profile.output.w, profile.output.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                if (_inverse)
                    owner.Ops.InverseSpectrogramPack4(source.texture, profile.nfft, profile.hop, profile.channels, profile.frames, output);
                else
                    owner.Ops.SpectrogramPack4(source.texture, profile.nfft, profile.hop, profile.channels, profile.frames, output);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, profile.output, profile.output);
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
            var profile = ResolveProfile(source, shape, layer, _inverse);
            var output = owner.RentTempMat(context.commandBuffer, profile.output.w, profile.output.h, AexisGraphSession.ResolveLinearMatTextureFormat());
            if (_inverse)
                owner.Ops.InverseSpectrogramPack4(context.commandBuffer, source.texture, profile.nfft, profile.hop, profile.channels, profile.frames, output);
            else
                owner.Ops.SpectrogramPack4(context.commandBuffer, source.texture, profile.nfft, profile.hop, profile.channels, profile.frames, output);
            context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, profile.output, profile.output, owned: true, blobName: layer.topNames[0]);
            if (context.shapes != null)
                context.shapes[layer.topNames[0]] = profile.output;
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static AexisGraphSession.TensorRef GetRenderInput(AexisLayerBufferContext context, AexisGraphModel.Layer layer)
        {
            if (layer?.bottomNames == null || layer.bottomNames.Length != 1
                || !context.textureBlobs.TryGetValue(layer.bottomNames[0], out var source)
                || source?.texture == null)
            {
                throw new InvalidOperationException("Spectrogram requires one texture-backed input: " + (layer?.name ?? string.Empty));
            }
            return source;
        }

        private static Profile ResolveProfile(AexisGraphSession.TensorRef source, AexisGraphSession.BufferShape input, AexisGraphModel.Layer layer, bool inverse)
        {
            if (source?.texture == null || !AexisGraphSession.IsStrictLinearMatTexture(source))
                throw new InvalidOperationException("Spectrogram requires exact FP32 LinearMat texture storage: " + (layer?.name ?? string.Empty));
            return ResolveProfile(input, layer, inverse);
        }

        private static Profile ResolveProfile(AexisGraphSession.CmdTensorRef source, AexisGraphSession.BufferShape input, AexisGraphModel.Layer layer, bool inverse)
        {
            if (source?.texture == null || !AexisGraphSession.IsStrictLinearMatTexture(source))
                throw new InvalidOperationException("Spectrogram requires exact FP32 LinearMat texture storage: " + (layer?.name ?? string.Empty));
            return ResolveProfile(input, layer, inverse);
        }

        private static Profile ResolveProfile(AexisGraphSession.BufferShape input, AexisGraphModel.Layer layer, bool inverse)
        {
            if (input.dims != 2 || input.w <= 0 || input.h <= 0 || input.d != 1 || input.c != 1)
                throw new InvalidOperationException("Spectrogram requires rank-2 [width,height] logical storage: " + (layer?.name ?? string.Empty));

            var nfft = layer.GetInt(0, 0);
            var hop = layer.GetInt(1, 0);
            var channels = layer.GetInt(2, 0);
            if (nfft < 2 || nfft > 256 || (nfft & 1) != 0 || hop <= 0 || hop > nfft || channels <= 0)
                throw new InvalidOperationException("Spectrogram requires even n_fft in [2,256], hop in [1,n_fft], and explicit positive channels: " + (layer?.name ?? string.Empty));

            if (!inverse)
            {
                if (input.h != channels || input.w < nfft || (input.w - nfft) % hop != 0)
                    throw new InvalidOperationException("Spectrogram requires complete static [samples,channels] frames: " + (layer?.name ?? string.Empty));
                var frames = 1 + (input.w - nfft) / hop;
                return new Profile
                {
                    nfft = nfft,
                    hop = hop,
                    channels = channels,
                    frames = frames,
                    output = new AexisGraphSession.BufferShape(2, 2 * (nfft / 2 + 1), checked(frames * channels), 1, 1)
                };
            }

            var complexWidth = 2 * (nfft / 2 + 1);
            if (input.w != complexWidth || input.h % channels != 0)
                throw new InvalidOperationException("InverseSpectrogram requires one-sided complex [2*(n_fft/2+1),frames*channels] input: " + (layer?.name ?? string.Empty));
            var inverseFrames = input.h / channels;
            if (inverseFrames <= 0)
                throw new InvalidOperationException("InverseSpectrogram requires at least one static frame: " + (layer?.name ?? string.Empty));
            return new Profile
            {
                nfft = nfft,
                hop = hop,
                channels = channels,
                frames = inverseFrames,
                output = new AexisGraphSession.BufferShape(2, checked(nfft + hop * (inverseFrames - 1)), channels, 1, 1)
            };
        }

        private struct Profile
        {
            public int nfft;
            public int hop;
            public int channels;
            public int frames;
            public AexisGraphSession.BufferShape output;
        }
    }
}
