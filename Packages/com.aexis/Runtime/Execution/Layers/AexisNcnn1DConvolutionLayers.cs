using System;
using System.Collections.Generic;
using UnityEngine;

namespace Aexis.Execution
{
    internal static class AexisNcnn1DLayerAdapter
    {
        public static AexisGraphModel.Layer Normalize(AexisGraphModel.Layer source, AexisLayerTypeKey targetType, string targetName)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            var layer = new AexisGraphModel.Layer
            {
                type = targetType,
                typeName = targetName,
                name = source.name,
                bottoms = source.bottoms,
                tops = source.tops,
                bottomNames = source.bottomNames,
                topNames = source.topNames,
                intParams = source.intParams != null ? new Dictionary<int, string>(source.intParams) : new Dictionary<int, string>(),
                stringParams = source.stringParams != null ? new Dictionary<string, string>(source.stringParams, StringComparer.Ordinal) : new Dictionary<string, string>(StringComparer.Ordinal)
            };
            layer.intParams[11] = "1";
            layer.intParams[12] = "1";
            layer.intParams[13] = "1";
            layer.intParams[14] = "0";
            layer.intParams[16] = "0";
            layer.intParams[19] = "0";
            return layer;
        }
    }

    public sealed class AexisConvolutionDepthWise1DLayer : AexisBaseLayer
    {
        private readonly AexisConvolutionDepthWiseLayer _inner = new AexisConvolutionDepthWiseLayer();
        public AexisConvolutionDepthWise1DLayer() : base(AexisLayerTypes.Convolution1D, false, true) { }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            if (layer.GetInt(19, 0) != 0)
                throw new NotSupportedException("ConvolutionDepthWise1D dynamic_weight=1 has no immutable Pack4 weight profile: " + layer.name);
            var padValue = layer.GetFloat(18, 0f);
            if (float.IsNaN(padValue) || float.IsInfinity(padValue) || Math.Abs(padValue) > 0f)
                throw new NotSupportedException("ConvolutionDepthWise1D non-zero pad_value is not implemented by the Pack4 convolution kernel: " + layer.name);
            var previous = owner.EnableDepthWiseTextureConvolution;
            try
            {
                owner.EnableDepthWiseTextureConvolution = true;
                return _inner.LoadLayer(owner, AexisNcnn1DLayerAdapter.Normalize(layer, AexisLayerTypes.ConvolutionDepthWise, "ConvolutionDepthWise"), br);
            }
            finally
            {
                owner.EnableDepthWiseTextureConvolution = previous;
            }
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner._conv.TryGetValue(layer.name, out var conv) || conv == null)
                throw new InvalidOperationException("ConvolutionDepthWise1D weights were not loaded: " + layer.name);
            if (!context.textureBlobs.TryGetValue(layer.bottomNames[0], out var src) || src?.texture == null)
                throw new InvalidOperationException("ConvolutionDepthWise1D requires a texture-backed rank-2 input: " + layer.name);
            var shape = AexisGraphSession.GetTextureShape(context.textureShapes, src, layer.bottomNames[0]);
            ValidateLinearInput(src, shape, conv, layer.name);

            var outW = AexisGraphSession.ComputeConvOut(shape.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
            var outputShape = new AexisGraphSession.BufferShape(2, outW, conv.outC, 1, 1);
            var sourceStorage = AexisGraphSession.GetTextureStorageShape(src, shape);
            RenderTexture packedInput = null;
            RenderTexture packedOutput = null;
            RenderTexture output = null;
            try
            {
                packedInput = owner.RentTempArray(shape.w, 1, conv.inPacks, owner.TensorTextureFormat);
                if (AexisGraphSession.IsStrictLinearMatTexture(src))
                    owner.Ops.ReshapeLinearMatToPack4(src.texture, sourceStorage.w, sourceStorage.h, shape.w, 1, 1, conv.inC, 3, packedInput);
                else
                    owner.Ops.ReshapeScalar2DToPack4(src.texture, sourceStorage.w, sourceStorage.h, shape.w, 1, 1, conv.inC, 3, packedInput);

                packedOutput = owner.RentTempArray(outW, 1, conv.outPacks, owner.TensorTextureFormat);
                owner.Ops.SetFp16DepthWiseWeights(owner.UsesFp16WeightStorage ? conv.packedDepthWiseWeight4Fp16 : null);
                owner.Ops.ConvDepthWisePack4(
                    packedInput, conv.packedDepthWiseWeight4, conv.packedBias4,
                    conv.inC, conv.outC, conv.group, conv.outPacks,
                    conv.kernelW, 1, conv.strideW, 1, conv.padLeft, 0,
                    conv.dilationW, 1, conv.activationType, conv.activationSlope, packedOutput);

                output = owner.RentTempMat(outputShape.w, outputShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.ReshapePack4ToLinearMat(packedOutput, outW, 1, 1, conv.outC, 3, output);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, outputShape, outputShape);
                output = null;
            }
            finally
            {
                if (packedInput != null) owner.ReturnTempArray(packedInput);
                if (packedOutput != null) owner.ReturnTempArray(packedOutput);
                if (output != null) owner.ReturnTempArray(output);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var shape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            var src = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            if (!owner._conv.TryGetValue(layer.name, out var conv) || conv == null)
                throw new InvalidOperationException("ConvolutionDepthWise1D weights were not loaded: " + layer.name);
            ValidateLinearInput(src, shape, conv, layer.name);
            var outW = AexisGraphSession.ComputeConvOut(shape.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
            var outputShape = new AexisGraphSession.BufferShape(2, outW, conv.outC, 1, 1);
            var sourceStorage = AexisGraphSession.GetCmdStorageShape(src, shape);
            ComputeTexture packedInput = null;
            ComputeTexture packedOutput = null;
            ComputeTexture output = null;
            try
            {
                packedInput = owner.RentTempArray(context.commandBuffer, shape.w, 1, conv.inPacks, owner.TensorTextureFormat);
                if (AexisGraphSession.IsStrictLinearMatTexture(src))
                    owner.Ops.ReshapeLinearMatToPack4(context.commandBuffer, src.texture, sourceStorage.w, sourceStorage.h, shape.w, 1, 1, conv.inC, 3, packedInput);
                else
                    owner.Ops.ReshapeScalar2DToPack4(context.commandBuffer, src.texture, sourceStorage.w, sourceStorage.h, shape.w, 1, 1, conv.inC, 3, packedInput);

                packedOutput = owner.RentTempArray(context.commandBuffer, outW, 1, conv.outPacks, owner.TensorTextureFormat);
                owner.Ops.SetFp16DepthWiseWeights(owner.UsesFp16WeightStorage ? conv.packedDepthWiseWeight4Fp16 : null);
                owner.Ops.ConvDepthWisePack4(
                    context.commandBuffer, packedInput, conv.packedDepthWiseWeight4, conv.packedBias4,
                    conv.inC, conv.outC, conv.group, conv.outPacks,
                    conv.kernelW, 1, conv.strideW, 1, conv.padLeft, 0,
                    conv.dilationW, 1, conv.activationType, conv.activationSlope, packedOutput);

                output = owner.RentTempMat(context.commandBuffer, outputShape.w, outputShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.ReshapePack4ToLinearMat(context.commandBuffer, packedOutput, outW, 1, 1, conv.outC, 3, output);
                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, outputShape, outputShape, owned: true, blobName: layer.topNames[0]);
                if (context.shapes != null) context.shapes[layer.topNames[0]] = outputShape;
                output = null;
            }
            finally
            {
                if (packedInput != null) owner.ReturnTempArray(context.commandBuffer, packedInput);
                if (packedOutput != null) owner.ReturnTempArray(context.commandBuffer, packedOutput);
                if (output != null) owner.ReturnTempArray(context.commandBuffer, output);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static void ValidateLinearInput(AexisGraphSession.TensorRef src, AexisGraphSession.BufferShape shape, AexisGraphSession.ConvPack conv, string layerName)
        {
            var storage = src != null ? AexisGraphSession.GetTextureStorageShape(src, shape) : default;
            if (src?.texture == null || shape.dims != 2 || shape.w <= 0 || shape.h != conv.inC
                || conv.packedDepthWiseWeight4 == null || conv.packedBias4 == null
                || (!AexisGraphSession.IsStrictLinearMatTexture(src)
                    && (storage.dims != 2 || storage.w != shape.w || storage.h != shape.h || src.packs != 1)))
                throw new InvalidOperationException("ConvolutionDepthWise1D requires native dims=2 [width,channels] LinearMat/scalar texture storage: " + layerName);
        }

        private static void ValidateLinearInput(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape shape, AexisGraphSession.ConvPack conv, string layerName)
        {
            var storage = src != null ? AexisGraphSession.GetCmdStorageShape(src, shape) : default;
            if (src?.texture == null || shape.dims != 2 || shape.w <= 0 || shape.h != conv.inC
                || conv.packedDepthWiseWeight4 == null || conv.packedBias4 == null
                || (!AexisGraphSession.IsStrictLinearMatTexture(src)
                    && (storage.dims != 2 || storage.w != shape.w || storage.h != shape.h || src.packs != 1)))
                throw new InvalidOperationException("ConvolutionDepthWise1D requires native dims=2 [width,channels] LinearMat/scalar texture storage: " + layerName);
        }
    }

    public sealed class AexisDeconvolution1DLayer : AexisBaseLayer
    {
        private readonly AexisDeconvolutionLayer _inner = new AexisDeconvolutionLayer();
        public AexisDeconvolution1DLayer() : base(AexisLayerTypes.Deconvolution, false, true) { }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            if (layer.GetInt(28, 0) != 0)
                throw new NotSupportedException("Deconvolution1D dynamic_weight=1 has no immutable Pack4 weight profile: " + layer.name);
            if (layer.GetInt(20, 0) != 0)
                throw new NotSupportedException("Deconvolution1D output_w cropping is not implemented by the Pack4 adapter: " + layer.name);
            var previous = owner.EnableGeneralTextureConvolution;
            try
            {
                owner.EnableGeneralTextureConvolution = true;
                return _inner.LoadLayer(owner, AexisNcnn1DLayerAdapter.Normalize(layer, AexisLayerTypes.Deconvolution, "Deconvolution"), br);
            }
            finally
            {
                owner.EnableGeneralTextureConvolution = previous;
            }
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner._deconv.TryGetValue(layer.name, out var deconv) || deconv == null)
                throw new InvalidOperationException("Deconvolution1D weights were not loaded: " + layer.name);
            if (!context.textureBlobs.TryGetValue(layer.bottomNames[0], out var src) || src?.texture == null)
                throw new InvalidOperationException("Deconvolution1D requires a texture-backed rank-2 input: " + layer.name);
            var shape = AexisGraphSession.GetTextureShape(context.textureShapes, src, layer.bottomNames[0]);
            ValidateLinearInput(src, shape, deconv, layer.name);
            var outW = AexisGraphSession.ComputeDeconvOut(shape.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outputShape = new AexisGraphSession.BufferShape(2, outW, deconv.outC, 1, 1);
            var sourceStorage = AexisGraphSession.GetTextureStorageShape(src, shape);
            RenderTexture packedInput = null;
            RenderTexture packedOutput = null;
            RenderTexture output = null;
            try
            {
                packedInput = owner.RentTempArray(shape.w, 1, deconv.inPacks, owner.TensorTextureFormat);
                if (AexisGraphSession.IsStrictLinearMatTexture(src))
                    owner.Ops.ReshapeLinearMatToPack4(src.texture, sourceStorage.w, sourceStorage.h, shape.w, 1, 1, deconv.inC, 3, packedInput);
                else
                    owner.Ops.ReshapeScalar2DToPack4(src.texture, sourceStorage.w, sourceStorage.h, shape.w, 1, 1, deconv.inC, 3, packedInput);
                packedOutput = owner.RentTempArray(outW, 1, deconv.outPacks, owner.TensorTextureFormat);
                owner.Ops.SetFp16ConvWeights(owner.UsesFp16WeightStorage ? deconv.packedWeight4Fp16 : null);
                owner.Ops.DeconvolutionPack4General(
                    packedInput, deconv.inPacks, deconv.packedWeight4, deconv.packedBias4, deconv.outPacks,
                    deconv.kernelW, 1, deconv.strideW, 1, deconv.padLeft, 0, deconv.dilationW, 1,
                    deconv.activationType, deconv.activationSlope, packedOutput);
                output = owner.RentTempMat(outputShape.w, outputShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.ReshapePack4ToLinearMat(packedOutput, outW, 1, 1, deconv.outC, 3, output);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, outputShape, outputShape);
                output = null;
            }
            finally
            {
                if (packedInput != null) owner.ReturnTempArray(packedInput);
                if (packedOutput != null) owner.ReturnTempArray(packedOutput);
                if (output != null) owner.ReturnTempArray(output);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var shape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            var src = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            if (!owner._deconv.TryGetValue(layer.name, out var deconv) || deconv == null)
                throw new InvalidOperationException("Deconvolution1D weights were not loaded: " + layer.name);
            ValidateLinearInput(src, shape, deconv, layer.name);
            var outW = AexisGraphSession.ComputeDeconvOut(shape.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outputShape = new AexisGraphSession.BufferShape(2, outW, deconv.outC, 1, 1);
            var sourceStorage = AexisGraphSession.GetCmdStorageShape(src, shape);
            ComputeTexture packedInput = null;
            ComputeTexture packedOutput = null;
            ComputeTexture output = null;
            try
            {
                packedInput = owner.RentTempArray(context.commandBuffer, shape.w, 1, deconv.inPacks, owner.TensorTextureFormat);
                if (AexisGraphSession.IsStrictLinearMatTexture(src))
                    owner.Ops.ReshapeLinearMatToPack4(context.commandBuffer, src.texture, sourceStorage.w, sourceStorage.h, shape.w, 1, 1, deconv.inC, 3, packedInput);
                else
                    owner.Ops.ReshapeScalar2DToPack4(context.commandBuffer, src.texture, sourceStorage.w, sourceStorage.h, shape.w, 1, 1, deconv.inC, 3, packedInput);
                packedOutput = owner.RentTempArray(context.commandBuffer, outW, 1, deconv.outPacks, owner.TensorTextureFormat);
                owner.Ops.SetFp16ConvWeights(owner.UsesFp16WeightStorage ? deconv.packedWeight4Fp16 : null);
                owner.Ops.DeconvolutionPack4General(
                    context.commandBuffer, packedInput, deconv.inPacks, deconv.packedWeight4, deconv.packedBias4, deconv.outPacks,
                    deconv.kernelW, 1, deconv.strideW, 1, deconv.padLeft, 0, deconv.dilationW, 1,
                    deconv.activationType, deconv.activationSlope, packedOutput);
                output = owner.RentTempMat(context.commandBuffer, outputShape.w, outputShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.ReshapePack4ToLinearMat(context.commandBuffer, packedOutput, outW, 1, 1, deconv.outC, 3, output);
                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, outputShape, outputShape, owned: true, blobName: layer.topNames[0]);
                if (context.shapes != null) context.shapes[layer.topNames[0]] = outputShape;
                output = null;
            }
            finally
            {
                if (packedInput != null) owner.ReturnTempArray(context.commandBuffer, packedInput);
                if (packedOutput != null) owner.ReturnTempArray(context.commandBuffer, packedOutput);
                if (output != null) owner.ReturnTempArray(context.commandBuffer, output);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static void ValidateLinearInput(AexisGraphSession.TensorRef src, AexisGraphSession.BufferShape shape, AexisGraphSession.DeconvPack deconv, string layerName)
        {
            var storage = src != null ? AexisGraphSession.GetTextureStorageShape(src, shape) : default;
            if (src?.texture == null || shape.dims != 2 || shape.w <= 0 || shape.h != deconv.inC
                || deconv.group != 1 || deconv.packedWeight4 == null || deconv.packedBias4 == null
                || (!AexisGraphSession.IsStrictLinearMatTexture(src)
                    && (storage.dims != 2 || storage.w != shape.w || storage.h != shape.h || src.packs != 1)))
                throw new InvalidOperationException("Deconvolution1D requires native dims=2 [width,channels] LinearMat/scalar texture storage: " + layerName);
        }

        private static void ValidateLinearInput(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape shape, AexisGraphSession.DeconvPack deconv, string layerName)
        {
            var storage = src != null ? AexisGraphSession.GetCmdStorageShape(src, shape) : default;
            if (src?.texture == null || shape.dims != 2 || shape.w <= 0 || shape.h != deconv.inC
                || deconv.group != 1 || deconv.packedWeight4 == null || deconv.packedBias4 == null
                || (!AexisGraphSession.IsStrictLinearMatTexture(src)
                    && (storage.dims != 2 || storage.w != shape.w || storage.h != shape.h || src.packs != 1)))
                throw new InvalidOperationException("Deconvolution1D requires native dims=2 [width,channels] LinearMat/scalar texture storage: " + layerName);
        }
    }
}
