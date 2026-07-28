using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    public enum AexisRecurrentKind
    {
        Rnn = 0,
        Gru = 1,
        Lstm = 2
    }

    // Bounded P2 recurrent profile. Gate weights are immutable uploads while
    // sequence values and all observable state are Pack4 Texture2DArrays.
    public sealed class AexisRecurrentLayer : AexisBaseLayer
    {
        private const int MaxProfileExtent = 256;
        private readonly AexisRecurrentKind _kind;

        public AexisRecurrentLayer(AexisRecurrentKind kind)
            : base(TypeFor(kind), supportsBufferPath: false, supportsCommandBufferPath: true)
        {
            _kind = kind;
        }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader reader)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            if (layer == null) throw new ArgumentNullException(nameof(layer));
            if (reader == null) throw new ArgumentNullException(nameof(reader));

            var bytesStart = reader.Position;
            ValidateLayerContract(layer, _kind);
            var inputSize = layer.GetInt(0, 0);
            var hiddenSize = layer.GetInt(1, 0);
            var gates = GateCount(_kind);
            var inputCount = checked(gates * hiddenSize * inputSize);
            var stateCount = checked(gates * hiddenSize * hiddenSize);
            var biasCount = checked(gates * hiddenSize);
            var inputWeights = reader.ReadFloat32Array(inputCount);
            var stateWeights = reader.ReadFloat32Array(stateCount);
            var bias = reader.ReadFloat32Array(biasCount);

            var pack = new AexisGraphSession.RecurrentPack
            {
                kind = (int)_kind,
                inputSize = inputSize,
                hiddenSize = hiddenSize,
                inputWeights = NewImmutableBuffer(inputWeights, "AexisGraphSession.RecurrentInputWeights:" + layer.name),
                recurrentWeights = NewImmutableBuffer(stateWeights, "AexisGraphSession.RecurrentStateWeights:" + layer.name),
                bias = NewImmutableBuffer(bias, "AexisGraphSession.RecurrentBias:" + layer.name)
            };
            if (owner._extraPacks.TryGetValue(layer.name, out var existing))
                existing?.Dispose();
            owner._extraPacks[layer.name] = pack;
            return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, reader.Position - bytesStart), 0, 0, 0);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var pack = RequirePack(owner, layer, _kind);
            if (!AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var source, out var input))
                throw new InvalidOperationException("Recurrent profile input is not a texture: " + layer.name);
            ValidateInput(source, input, pack, layer.name);
            var outputShape = OutputShape(input, pack);
            RenderTexture output = null;
            try
            {
                output = owner.RentTempArray(outputShape.w, outputShape.h, PackCount(outputShape.c), owner.ResolveActivationTextureFormat(layer, outputShape.dims));
                owner.Ops.RecurrentPack4(source.texture, pack.inputWeights, pack.recurrentWeights, pack.bias,
                    pack.kind, input.w, pack.inputSize, pack.hiddenSize, output);
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
            var pack = RequirePack(owner, layer, _kind);
            var source = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var input = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            ValidateInput(source, input, pack, layer.name);
            var outputShape = OutputShape(input, pack);
            ComputeTexture output = null;
            try
            {
                output = owner.RentTempArray(context.commandBuffer, outputShape.w, outputShape.h, PackCount(outputShape.c), owner.ResolveActivationTextureFormat(layer, outputShape.dims));
                owner.Ops.RecurrentPack4(context.commandBuffer, source.texture, pack.inputWeights, pack.recurrentWeights, pack.bias,
                    pack.kind, input.w, pack.inputSize, pack.hiddenSize, output);
                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, outputShape, outputShape, owned: true, blobName: layer.topNames[0]);
                context.shapes[layer.topNames[0]] = outputShape;
                output = null;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(context.commandBuffer, output);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        internal static void ValidateLayerContract(AexisGraphModel.Layer layer, AexisRecurrentKind kind)
        {
            if (layer.bottomNames == null || layer.bottomNames.Length != 1 || layer.topNames == null || layer.topNames.Length != 1)
                throw new NotSupportedException("The bounded " + kind + " profile requires exactly one sequence input and one sequence output: " + layer.name);
            var inputSize = layer.GetInt(0, 0);
            var hiddenSize = layer.GetInt(1, 0);
            if (inputSize <= 0 || inputSize > MaxProfileExtent || hiddenSize <= 0 || hiddenSize > MaxProfileExtent)
                throw new NotSupportedException("The bounded " + kind + " profile requires input_size and hidden_size in [1,256]: " + layer.name);
            if (layer.GetInt(2, 0) != 0 || layer.GetInt(3, 0) != 0 || layer.GetInt(4, 0) != 0)
                throw new NotSupportedException("The bounded " + kind + " profile supports only forward direction, zero initial state, and no optional state outputs: " + layer.name);
        }

        internal static int GateCount(AexisRecurrentKind kind) => kind == AexisRecurrentKind.Gru ? 3 : kind == AexisRecurrentKind.Lstm ? 4 : 1;

        private static AexisLayerTypeKey TypeFor(AexisRecurrentKind kind)
        {
            return kind == AexisRecurrentKind.Gru ? AexisLayerTypes.GRU
                : kind == AexisRecurrentKind.Lstm ? AexisLayerTypes.LSTM
                : AexisLayerTypes.RNN;
        }

        private static ComputeBuffer NewImmutableBuffer(float[] values, string label)
        {
            var buffer = new ComputeBuffer(values.Length, sizeof(float), ComputeBufferType.Structured);
            AexisGpuResourceTracker.RegisterBuffer(buffer, values.Length, sizeof(float), label);
            buffer.SetData(values);
            return buffer;
        }

        private static AexisGraphSession.RecurrentPack RequirePack(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisRecurrentKind kind)
        {
            ValidateLayerContract(layer, kind);
            if (!owner._extraPacks.TryGetValue(layer.name, out var loaded) || loaded is not AexisGraphSession.RecurrentPack pack
                || pack.kind != (int)kind || pack.inputWeights == null || pack.recurrentWeights == null || pack.bias == null)
            {
                throw new InvalidOperationException("The bounded " + kind + " profile has no loaded immutable GPU gate weights: " + layer.name);
            }
            return pack;
        }

        private static AexisGraphSession.BufferShape OutputShape(AexisGraphSession.BufferShape input, AexisGraphSession.RecurrentPack pack)
        {
            return new AexisGraphSession.BufferShape(3, input.w, 1, 1, pack.hiddenSize);
        }

        private static int PackCount(int channels) => Mathf.Max(1, Mathf.CeilToInt(channels / 4f));

        private static void ValidateInput(AexisGraphSession.TensorRef source, AexisGraphSession.BufferShape input, AexisGraphSession.RecurrentPack pack, string layerName)
        {
            if (source?.texture == null || input.dims != 3 || input.w <= 0 || input.w > MaxProfileExtent || input.h != 1 || input.d != 1
                || input.c != pack.inputSize || source.packs != PackCount(input.c) || !AexisGraphSession.MatchesPack4TextureStorage(source, input))
            {
                throw new InvalidOperationException("The bounded recurrent profile requires exact Pack4 [sequence<=256,1,input_size] storage: " + layerName);
            }
        }

        private static void ValidateInput(AexisGraphSession.CmdTensorRef source, AexisGraphSession.BufferShape input, AexisGraphSession.RecurrentPack pack, string layerName)
        {
            if (source?.texture == null || input.dims != 3 || input.w <= 0 || input.w > MaxProfileExtent || input.h != 1 || input.d != 1
                || input.c != pack.inputSize || source.packs != PackCount(input.c) || !AexisGraphSession.MatchesPack4TextureStorage(source, input))
            {
                throw new InvalidOperationException("The bounded recurrent profile requires exact Pack4 [sequence<=256,1,input_size] storage: " + layerName);
            }
        }
    }
}
