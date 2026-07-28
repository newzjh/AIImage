using System;
using UnityEngine;

namespace Aexis.Execution
{
    // P2 CDHW one-to-one depthwise transposed convolution. Activations never
    // leave Pack4 Texture2DArray storage; structured buffers are immutable
    // vectorized weights and bias uploaded at model-load time only.
    public sealed class AexisDeconvolutionDepthWise3DLayer : AexisBaseLayer
    {
        public AexisDeconvolutionDepthWise3DLayer()
            : base(AexisLayerTypes.DeconvDw3D, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader reader)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            if (layer == null) throw new ArgumentNullException(nameof(layer));
            if (reader == null) throw new ArgumentNullException(nameof(reader));

            var start = reader.Position;
            var pack = new AexisGraphSession.DeconvPack
            {
                outC = layer.GetInt(0, 0),
                group = layer.GetInt(7, 1),
                kernelW = layer.GetInt(1, 0),
                kernelH = layer.GetInt(11, layer.GetInt(1, 0)),
                kernelD = layer.GetInt(21, layer.GetInt(1, 0)),
                dilationW = layer.GetInt(2, 1),
                dilationH = layer.GetInt(12, layer.GetInt(2, 1)),
                dilationD = layer.GetInt(22, layer.GetInt(2, 1)),
                strideW = layer.GetInt(3, 1),
                strideH = layer.GetInt(13, layer.GetInt(3, 1)),
                strideD = layer.GetInt(23, layer.GetInt(3, 1)),
                padLeft = layer.GetInt(4, 0),
                padRight = layer.GetInt(15, layer.GetInt(4, 0)),
                padTop = layer.GetInt(14, layer.GetInt(4, 0)),
                padBottom = layer.GetInt(16, layer.GetInt(14, layer.GetInt(4, 0))),
                padFront = layer.GetInt(24, layer.GetInt(4, 0)),
                padBehind = layer.GetInt(17, layer.GetInt(24, layer.GetInt(4, 0))),
                outputPadRight = layer.GetInt(18, 0),
                outputPadBottom = layer.GetInt(19, layer.GetInt(18, 0)),
                outputPadBehind = layer.GetInt(20, layer.GetInt(18, 0)),
                biasTerm = layer.GetInt(5, 0),
                weightSize = layer.GetInt(6, 0),
                activationType = layer.GetInt(9, 0),
                activationSlope = AexisGraphSession.ParseLeakySlope(layer)
            };
            var kernelVolume = checked(pack.kernelW * pack.kernelH * pack.kernelD);
            if (pack.outC <= 0 || pack.group != pack.outC || kernelVolume <= 0
                || pack.strideW <= 0 || pack.strideH <= 0 || pack.strideD <= 0
                || pack.dilationW <= 0 || pack.dilationH <= 0 || pack.dilationD <= 0
                || pack.padLeft < 0 || pack.padRight < 0 || pack.padTop < 0 || pack.padBottom < 0 || pack.padFront < 0 || pack.padBehind < 0
                || pack.outputPadRight < 0 || pack.outputPadBottom < 0 || pack.outputPadBehind < 0
                || pack.weightSize != checked(pack.outC * kernelVolume))
            {
                throw new NotSupportedException("DeconvolutionDepthWise3D requires immutable one-to-one group=inC=outC CDHW weights: " + layer.name);
            }

            pack.inC = pack.outC;
            pack.inPacks = (pack.inC + 3) / 4;
            pack.outPacks = pack.inPacks;
            var weights = AexisGraphSession.ReadPackedOrRawWeightArray(reader, pack.weightSize, layer.name);
            var bias = pack.biasTerm != 0 ? reader.ReadFloat32Array(pack.outC) : new float[pack.outC];
            var packedWeights = AexisGraphSession.PackDepthWiseWeightsToP4KdKhKw(weights, pack.outC, pack.kernelW, pack.kernelH, pack.kernelD, pack.outPacks);
            var packedBias = AexisGraphSession.PackBiasToO4(bias, pack.outC, pack.outPacks);
            pack.packedWeight4 = new ComputeBuffer(packedWeights.Length, sizeof(float) * 4, ComputeBufferType.Structured);
            AexisGpuResourceTracker.RegisterBuffer(pack.packedWeight4, packedWeights.Length, sizeof(float) * 4, "AexisGraphSession.DeconvDw3DPackedWeight4:" + layer.name);
            pack.packedBias4 = new ComputeBuffer(packedBias.Length, sizeof(float) * 4, ComputeBufferType.Structured);
            AexisGpuResourceTracker.RegisterBuffer(pack.packedBias4, packedBias.Length, sizeof(float) * 4, "AexisGraphSession.DeconvDw3DPackedBias4:" + layer.name);
            pack.packedWeight4.SetData(packedWeights);
            pack.packedBias4.SetData(packedBias);
            owner._deconv[layer.name] = pack;
            return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, reader.Position - start), 0, 0, 0);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner._deconv.TryGetValue(layer.name, out var deconv) || deconv == null)
                throw new InvalidOperationException("DeconvolutionDepthWise3D immutable weights were not loaded: " + layer.name);
            if (!context.textureBlobs.TryGetValue(layer.bottomNames[0], out var source) || source?.texture == null)
                throw new InvalidOperationException("DeconvolutionDepthWise3D rejects non-texture activation input: " + layer.name);
            var inputShape = AexisGraphSession.GetTextureShape(context.textureShapes, source, layer.bottomNames[0]);
            Validate(source, inputShape, deconv, layer.name);
            var outputShape = ResolveOutput(inputShape, deconv);
            var output = owner.RentTempArray(outputShape.w, outputShape.h, outputShape.d * deconv.outPacks, AexisGraphSession.ResolveTensorTextureFormat(4));
            owner.Ops.DeconvolutionDepthWise3dPack4CDHW(source.texture, inputShape.w, inputShape.h, inputShape.d, deconv.inC,
                deconv.packedWeight4, deconv.packedBias4, outputShape.w, outputShape.h, outputShape.d,
                deconv.kernelW, deconv.kernelH, deconv.kernelD, deconv.strideW, deconv.strideH, deconv.strideD,
                deconv.padLeft, deconv.padTop, deconv.padFront, deconv.dilationW, deconv.dilationH, deconv.dilationD,
                deconv.activationType, deconv.activationSlope, output);
            AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, outputShape, outputShape);
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (!owner._deconv.TryGetValue(layer.name, out var deconv) || deconv == null)
                throw new InvalidOperationException("DeconvolutionDepthWise3D immutable weights were not loaded: " + layer.name);
            var source = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var inputShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            Validate(source, inputShape, deconv, layer.name);
            var outputShape = ResolveOutput(inputShape, deconv);
            var output = owner.RentTempArray(context.commandBuffer, outputShape.w, outputShape.h, outputShape.d * deconv.outPacks, AexisGraphSession.ResolveTensorTextureFormat(4));
            owner.Ops.DeconvolutionDepthWise3dPack4CDHW(context.commandBuffer, source.texture, inputShape.w, inputShape.h, inputShape.d, deconv.inC,
                deconv.packedWeight4, deconv.packedBias4, outputShape.w, outputShape.h, outputShape.d,
                deconv.kernelW, deconv.kernelH, deconv.kernelD, deconv.strideW, deconv.strideH, deconv.strideD,
                deconv.padLeft, deconv.padTop, deconv.padFront, deconv.dilationW, deconv.dilationH, deconv.dilationD,
                deconv.activationType, deconv.activationSlope, output);
            context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, outputShape, outputShape, owned: true, blobName: layer.topNames[0]);
            if (context.shapes != null) context.shapes[layer.topNames[0]] = outputShape;
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static AexisGraphSession.BufferShape ResolveOutput(AexisGraphSession.BufferShape input, AexisGraphSession.DeconvPack deconv)
        {
            var output = new AexisGraphSession.BufferShape(4,
                AexisGraphSession.ComputeDeconvOut(input.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight),
                AexisGraphSession.ComputeDeconvOut(input.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom),
                AexisGraphSession.ComputeDeconvOut(input.d, deconv.kernelD, deconv.dilationD, deconv.strideD, deconv.padFront, deconv.padBehind, deconv.outputPadBehind),
                deconv.outC);
            if (output.w <= 0 || output.h <= 0 || output.d <= 0)
                throw new InvalidOperationException("DeconvolutionDepthWise3D resolves a non-positive output shape.");
            return output;
        }

        private static void Validate(AexisGraphSession.TensorRef source, AexisGraphSession.BufferShape shape, AexisGraphSession.DeconvPack deconv, string layer)
        {
            if (source?.texture == null || shape.dims != 4 || shape.c != deconv.inC || source.packs != deconv.inPacks || !AexisGraphSession.MatchesPack4TextureStorage(source, shape))
                throw new InvalidOperationException("DeconvolutionDepthWise3D requires CDHW Pack4 Texture2DArray input: " + layer);
        }

        private static void Validate(AexisGraphSession.CmdTensorRef source, AexisGraphSession.BufferShape shape, AexisGraphSession.DeconvPack deconv, string layer)
        {
            if (source?.texture == null || shape.dims != 4 || shape.c != deconv.inC || source.packs != deconv.inPacks || !AexisGraphSession.MatchesPack4TextureStorage(source, shape))
                throw new InvalidOperationException("DeconvolutionDepthWise3D requires CDHW Pack4 Texture2DArray input: " + layer);
        }
    }
}
