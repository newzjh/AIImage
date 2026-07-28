using System;
using UnityEngine;

namespace Aexis.Execution
{
    // NCNN's rank-2 [width,channels] depthwise transpose profile. The rank-2
    // boundary is a texture; conversion, transpose convolution, and result
    // all remain on Pack4 RenderTextures/CommandBuffer textures.
    public sealed class AexisDeconvolutionDepthWise1DLayer : AexisBaseLayer
    {
        private readonly AexisDeconvolutionDepthWiseLayer _loader = new AexisDeconvolutionDepthWiseLayer();

        public AexisDeconvolutionDepthWise1DLayer()
            : base(AexisLayerTypes.DeconvDw1D, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader reader)
        {
            if (layer.GetInt(28, 0) != 0)
                throw new NotSupportedException("DeconvolutionDepthWise1D dynamic_weight=1 has no immutable Pack4 profile: " + layer.name);
            if (layer.GetInt(20, 0) != 0)
                throw new NotSupportedException("DeconvolutionDepthWise1D output_w cropping is not implemented by the Pack4 profile: " + layer.name);
            return _loader.LoadLayer(owner, AexisNcnn1DLayerAdapter.Normalize(layer, AexisLayerTypes.DeconvolutionDepthWise, "DeconvolutionDepthWise"), reader);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner._deconv.TryGetValue(layer.name, out var deconv) || deconv == null)
                throw new InvalidOperationException("DeconvolutionDepthWise1D immutable weights were not loaded: " + layer.name);
            if (!context.textureBlobs.TryGetValue(layer.bottomNames[0], out var source) || source?.texture == null)
                throw new InvalidOperationException("DeconvolutionDepthWise1D rejects non-texture activation input: " + layer.name);
            var shape = AexisGraphSession.GetTextureShape(context.textureShapes, source, layer.bottomNames[0]);
            Validate(source, shape, deconv, layer.name);
            var outputShape = OutputShape(shape, deconv);
            var storage = AexisGraphSession.GetTextureStorageShape(source, shape);
            RenderTexture packedInput = null;
            RenderTexture packedOutput = null;
            RenderTexture output = null;
            try
            {
                packedInput = owner.RentTempArray(shape.w, 1, deconv.inPacks, owner.TensorTextureFormat);
                if (AexisGraphSession.IsStrictLinearMatTexture(source))
                    owner.Ops.ReshapeLinearMatToPack4(source.texture, storage.w, storage.h, shape.w, 1, 1, deconv.inC, 3, packedInput);
                else
                    owner.Ops.ReshapeScalar2DToPack4(source.texture, storage.w, storage.h, shape.w, 1, 1, deconv.inC, 3, packedInput);
                packedOutput = owner.RentTempArray(outputShape.w, 1, deconv.outPacks, owner.TensorTextureFormat);
                if (deconv.packedWeight4 == null || deconv.packedBias4 == null)
                    throw new InvalidOperationException("DeconvolutionDepthWise1D immutable Pack4 weights are unavailable: " + layer.name);
                owner.Ops.DeconvolutionDepthWisePack4(packedInput, deconv.packedWeight4, deconv.packedBias4, deconv.inC, deconv.outC, deconv.group,
                    deconv.outPacks, deconv.kernelW, 1, deconv.strideW, 1, deconv.padLeft, 0, deconv.dilationW, 1, deconv.activationType, deconv.activationSlope, packedOutput);
                output = owner.RentTempMat(outputShape.w, outputShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.ReshapePack4ToLinearMat(packedOutput, outputShape.w, 1, 1, deconv.outC, 3, output);
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
            if (!owner._deconv.TryGetValue(layer.name, out var deconv) || deconv == null)
                throw new InvalidOperationException("DeconvolutionDepthWise1D immutable weights were not loaded: " + layer.name);
            var source = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var shape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            Validate(source, shape, deconv, layer.name);
            var outputShape = OutputShape(shape, deconv);
            var storage = AexisGraphSession.GetCmdStorageShape(source, shape);
            ComputeTexture packedInput = null;
            ComputeTexture packedOutput = null;
            ComputeTexture output = null;
            try
            {
                packedInput = owner.RentTempArray(context.commandBuffer, shape.w, 1, deconv.inPacks, owner.TensorTextureFormat);
                if (AexisGraphSession.IsStrictLinearMatTexture(source))
                    owner.Ops.ReshapeLinearMatToPack4(context.commandBuffer, source.texture, storage.w, storage.h, shape.w, 1, 1, deconv.inC, 3, packedInput);
                else
                    owner.Ops.ReshapeScalar2DToPack4(context.commandBuffer, source.texture, storage.w, storage.h, shape.w, 1, 1, deconv.inC, 3, packedInput);
                packedOutput = owner.RentTempArray(context.commandBuffer, outputShape.w, 1, deconv.outPacks, owner.TensorTextureFormat);
                owner.Ops.Deconvolution2dGroupPack4(context.commandBuffer, packedInput, deconv.rawWeight, deconv.rawBias, deconv.inC, deconv.outC, deconv.group,
                    deconv.kernelW, 1, deconv.strideW, 1, deconv.padLeft, 0, deconv.dilationW, 1, deconv.activationType, deconv.activationSlope, packedOutput);
                output = owner.RentTempMat(context.commandBuffer, outputShape.w, outputShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.ReshapePack4ToLinearMat(context.commandBuffer, packedOutput, outputShape.w, 1, 1, deconv.outC, 3, output);
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

        private static AexisGraphSession.BufferShape OutputShape(AexisGraphSession.BufferShape input, AexisGraphSession.DeconvPack deconv)
        {
            var width = AexisGraphSession.ComputeDeconvOut(input.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            if (width <= 0) throw new InvalidOperationException("DeconvolutionDepthWise1D resolves a non-positive output width.");
            return new AexisGraphSession.BufferShape(2, width, deconv.outC, 1, 1);
        }

        private static bool IsProfile(AexisGraphSession.DeconvPack deconv)
        {
            return deconv != null && deconv.inC > 0 && deconv.outC == deconv.inC && deconv.group == deconv.inC
                && deconv.rawWeight != null && deconv.rawBias != null && deconv.kernelW > 0 && deconv.kernelH == 1
                && deconv.strideW > 0 && deconv.dilationW > 0 && deconv.padLeft >= 0 && deconv.padRight >= 0
                && deconv.weightSize == deconv.outC * deconv.kernelW;
        }

        private static void Validate(AexisGraphSession.TensorRef source, AexisGraphSession.BufferShape shape, AexisGraphSession.DeconvPack deconv, string layer)
        {
            var storage = source != null ? AexisGraphSession.GetTextureStorageShape(source, shape) : default;
            if (source?.texture == null || shape.dims != 2 || shape.w <= 0 || shape.h != deconv.inC || !IsProfile(deconv)
                || (!AexisGraphSession.IsStrictLinearMatTexture(source) && (storage.dims != 2 || storage.w != shape.w || storage.h != shape.h || source.packs != 1)))
                throw new InvalidOperationException("DeconvolutionDepthWise1D requires immutable one-to-one Pack4 transpose profile and rank-2 texture storage: " + layer);
        }

        private static void Validate(AexisGraphSession.CmdTensorRef source, AexisGraphSession.BufferShape shape, AexisGraphSession.DeconvPack deconv, string layer)
        {
            var storage = source != null ? AexisGraphSession.GetCmdStorageShape(source, shape) : default;
            if (source?.texture == null || shape.dims != 2 || shape.w <= 0 || shape.h != deconv.inC || !IsProfile(deconv)
                || (!AexisGraphSession.IsStrictLinearMatTexture(source) && (storage.dims != 2 || storage.w != shape.w || storage.h != shape.h || source.packs != 1)))
                throw new InvalidOperationException("DeconvolutionDepthWise1D requires immutable one-to-one Pack4 transpose profile and rank-2 texture storage: " + layer);
        }
    }
}
