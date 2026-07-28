using System;
using UnityEngine;

namespace Aexis.Execution
{
    // P2 one-to-one depthwise 3D profile. Activations are always CDHW Pack4
    // Texture2DArrays; the two ComputeBuffers below are immutable packed weights.
    public sealed class AexisConvolutionDepthWise3DLayer : AexisBaseLayer
    {
        public AexisConvolutionDepthWise3DLayer()
            : base(AexisLayerTypes.ConvDw3D, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader reader)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            if (layer == null) throw new ArgumentNullException(nameof(layer));
            if (reader == null) throw new ArgumentNullException(nameof(reader));

            var bytesStart = reader.Position;
            var pack = new AexisGraphSession.ConvPack
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
                biasTerm = layer.GetInt(5, 0),
                weightSize = layer.GetInt(6, 0),
                activationType = layer.GetInt(9, 0),
                activationSlope = AexisGraphSession.ParseLeakySlope(layer),
                isDepthWise = true,
                useBufferPath = false
            };

            var kernelVolume = checked(pack.kernelW * pack.kernelH * pack.kernelD);
            if (pack.outC <= 0 || pack.group != pack.outC || kernelVolume <= 0
                || pack.strideW <= 0 || pack.strideH <= 0 || pack.strideD <= 0
                || pack.dilationW <= 0 || pack.dilationH <= 0 || pack.dilationD <= 0
                || pack.padLeft < 0 || pack.padRight < 0 || pack.padTop < 0 || pack.padBottom < 0 || pack.padFront < 0 || pack.padBehind < 0
                || pack.weightSize != checked(pack.outC * kernelVolume))
            {
                throw new NotSupportedException("ConvolutionDepthWise3D requires the one-to-one static profile group=inC=outC and weight_data_size=outC*kernel_w*kernel_h*kernel_d: " + layer.name);
            }
            pack.inC = pack.outC;
            pack.inPacks = (pack.inC + 3) / 4;
            pack.outPacks = pack.inPacks;

            var weights = AexisGraphSession.ReadPackedOrRawWeightArray(reader, pack.weightSize, layer.name);
            var bias = pack.biasTerm != 0 ? reader.ReadFloat32Array(pack.outC) : new float[pack.outC];
            var packedWeights = AexisGraphSession.PackDepthWiseWeightsToP4KdKhKw(
                weights, pack.outC, pack.kernelW, pack.kernelH, pack.kernelD, pack.outPacks);
            var packedBias = AexisGraphSession.PackBiasToO4(bias, pack.outC, pack.outPacks);
            pack.packedDepthWiseWeight4 = new ComputeBuffer(packedWeights.Length, sizeof(float) * 4, ComputeBufferType.Structured);
            AexisGpuResourceTracker.RegisterBuffer(pack.packedDepthWiseWeight4, packedWeights.Length, sizeof(float) * 4, "AexisGraphSession.ConvDw3DPackedWeight4:" + layer.name);
            pack.packedBias4 = new ComputeBuffer(packedBias.Length, sizeof(float) * 4, ComputeBufferType.Structured);
            AexisGpuResourceTracker.RegisterBuffer(pack.packedBias4, packedBias.Length, sizeof(float) * 4, "AexisGraphSession.ConvDw3DPackedBias4:" + layer.name);
            pack.packedDepthWiseWeight4.SetData(packedWeights);
            pack.packedBias4.SetData(packedBias);
            owner._conv[layer.name] = pack;
            return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, reader.Position - bytesStart), 0, 0, 0);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner._conv.TryGetValue(layer.name, out var conv) || conv == null)
                throw new InvalidOperationException("ConvolutionDepthWise3D immutable GPU weights were not loaded: " + layer.name);
            if (!context.textureBlobs.TryGetValue(layer.bottomNames[0], out var source) || source?.texture == null)
                throw new InvalidOperationException("ConvolutionDepthWise3D rejects non-texture activation input: " + layer.name);
            var sourceShape = AexisGraphSession.GetTextureShape(context.textureShapes, source, layer.bottomNames[0]);
            ValidateInput(source, sourceShape, conv, layer.name);

            var outputShape = ResolveOutputShape(sourceShape, conv);
            var output = owner.RentTempArray(outputShape.w, outputShape.h, outputShape.d * conv.outPacks, AexisGraphSession.ResolveTensorTextureFormat(4));
            owner.Ops.ConvDepthWise3dPack4CDHW(
                source.texture, sourceShape.w, sourceShape.h, sourceShape.d, conv.inC,
                conv.packedDepthWiseWeight4, conv.packedBias4,
                outputShape.w, outputShape.h, outputShape.d,
                conv.kernelW, conv.kernelH, conv.kernelD,
                conv.strideW, conv.strideH, conv.strideD,
                conv.padLeft, conv.padRight, conv.padTop, conv.padBottom, conv.padFront, conv.padBehind,
                conv.dilationW, conv.dilationH, conv.dilationD,
                conv.activationType, conv.activationSlope, output);
            AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, outputShape, outputShape);
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (!owner._conv.TryGetValue(layer.name, out var conv) || conv == null)
                throw new InvalidOperationException("ConvolutionDepthWise3D immutable GPU weights were not loaded: " + layer.name);
            var source = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var sourceShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            ValidateInput(source, sourceShape, conv, layer.name);

            var outputShape = ResolveOutputShape(sourceShape, conv);
            var output = owner.RentTempArray(context.commandBuffer, outputShape.w, outputShape.h, outputShape.d * conv.outPacks, AexisGraphSession.ResolveTensorTextureFormat(4));
            owner.Ops.ConvDepthWise3dPack4CDHW(
                context.commandBuffer, source.texture, sourceShape.w, sourceShape.h, sourceShape.d, conv.inC,
                conv.packedDepthWiseWeight4, conv.packedBias4,
                outputShape.w, outputShape.h, outputShape.d,
                conv.kernelW, conv.kernelH, conv.kernelD,
                conv.strideW, conv.strideH, conv.strideD,
                conv.padLeft, conv.padRight, conv.padTop, conv.padBottom, conv.padFront, conv.padBehind,
                conv.dilationW, conv.dilationH, conv.dilationD,
                conv.activationType, conv.activationSlope, output);
            context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, outputShape, outputShape, owned: true, blobName: layer.topNames[0]);
            if (context.shapes != null)
                context.shapes[layer.topNames[0]] = outputShape;
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static AexisGraphSession.BufferShape ResolveOutputShape(AexisGraphSession.BufferShape input, AexisGraphSession.ConvPack conv)
        {
            var output = new AexisGraphSession.BufferShape(
                4,
                AexisGraphSession.ComputeConvOut(input.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight),
                AexisGraphSession.ComputeConvOut(input.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom),
                AexisGraphSession.ComputeConvOut(input.d, conv.kernelD, conv.dilationD, conv.strideD, conv.padFront, conv.padBehind),
                conv.outC);
            if (output.w <= 0 || output.h <= 0 || output.d <= 0)
                throw new InvalidOperationException("ConvolutionDepthWise3D resolves a non-positive output shape.");
            return output;
        }

        private static void ValidateInput(AexisGraphSession.TensorRef source, AexisGraphSession.BufferShape shape, AexisGraphSession.ConvPack conv, string layerName)
        {
            if (source?.texture == null || shape.dims != 4 || shape.c != conv.inC || source.packs != conv.inPacks
                || !AexisGraphSession.MatchesPack4TextureStorage(source, shape))
                throw new InvalidOperationException("ConvolutionDepthWise3D requires a CDHW Pack4 Texture2DArray matching immutable one-to-one depthwise weights: " + layerName);
        }

        private static void ValidateInput(AexisGraphSession.CmdTensorRef source, AexisGraphSession.BufferShape shape, AexisGraphSession.ConvPack conv, string layerName)
        {
            if (source?.texture == null || shape.dims != 4 || shape.c != conv.inC || source.packs != conv.inPacks
                || !AexisGraphSession.MatchesPack4TextureStorage(source, shape))
                throw new InvalidOperationException("ConvolutionDepthWise3D requires a CDHW Pack4 Texture2DArray matching immutable one-to-one depthwise weights: " + layerName);
        }
    }
}
