using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // ONNX Trilu masks the final two logical axes. Aexis stores those axes as
    // texture X (column) and Y (row), so no tensor materialization is required.
    public sealed class AexisTriluLayer : AexisBaseLayer
    {
        public AexisTriluLayer()
            : base(AexisLayerTypes.Trilu, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteRenderTexturePath(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisLayerBufferContext context)
        {
            if (!AexisGraphSession.TryGetExistingTexture(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.bottomNames[0],
                    out var source,
                    out var logicalShape))
            {
                throw new InvalidOperationException("Trilu requires an existing texture-native input: " + layer.name);
            }

            ValidateContract(layer, source, logicalShape);
            var storageShape = AexisGraphSession.GetTextureStorageShape(source, logicalShape);
            var upper = layer.GetInt(0, 1);
            var diagonal = layer.GetInt(1, 0);

            if (AexisGraphSession.IsStrictLinearMatTexture(source))
            {
                var output = owner.RentTempMat(
                    storageShape.w,
                    storageShape.h,
                    AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.TriluLinearMat(source.texture, diagonal, upper, output);
                AexisGraphSession.SetTextureBlob(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.topNames[0],
                    output,
                    logicalShape,
                    storageShape);
            }
            else
            {
                var outputDepth = logicalShape.dims == 4
                    ? Mathf.Max(1, logicalShape.d) * Mathf.Max(1, source.packs)
                    : Mathf.Max(1, source.packs);
                var output = owner.RentTempArray(
                    source.width,
                    source.height,
                    outputDepth,
                    source.texture.format);
                owner.Ops.TriluPack4(source.texture, outputDepth, diagonal, upper, output);
                AexisGraphSession.SetTextureBlob(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.topNames[0],
                    output,
                    logicalShape,
                    storageShape);
            }

            owner.Consume(
                context.textureBlobs,
                context.bufferBlobs,
                context.bufferRefs,
                context.bufferViews,
                context.remaining,
                layer.bottomNames,
                context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisLayerCommandBufferContext context)
        {
            var source = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var logicalShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            ValidateContract(layer, source, logicalShape);

            var storageShape = AexisGraphSession.GetCmdStorageShape(source, logicalShape);
            var upper = layer.GetInt(0, 1);
            var diagonal = layer.GetInt(1, 0);
            ComputeTexture output;

            if (AexisGraphSession.IsStrictLinearMatTexture(source))
            {
                output = owner.RentTempMat(
                    context.commandBuffer,
                    storageShape.w,
                    storageShape.h,
                    AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.TriluLinearMat(context.commandBuffer, source.texture, diagonal, upper, output);
            }
            else
            {
                var outputDepth = logicalShape.dims == 4
                    ? Mathf.Max(1, logicalShape.d) * Mathf.Max(1, source.packs)
                    : Mathf.Max(1, source.packs);
                output = owner.RentTempArray(
                    context.commandBuffer,
                    source.width,
                    source.height,
                    outputDepth,
                    source.texture.format);
                owner.Ops.TriluPack4(context.commandBuffer, source.texture, outputDepth, diagonal, upper, output);
            }

            context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(
                output,
                logicalShape,
                storageShape,
                owned: true,
                blobName: layer.topNames[0]);
            if (context.shapes != null)
                context.shapes[layer.topNames[0]] = logicalShape;
            owner.ConsumeCmd(
                context.commandBuffer,
                context.blobs,
                context.remaining,
                layer.bottomNames,
                context.pinnedNames,
                context.shapes);
        }

        private static void ValidateContract(
            AexisGraphModel.Layer layer,
            AexisGraphSession.TensorRef source,
            AexisGraphSession.BufferShape logicalShape)
        {
            if (source == null || source.texture == null)
                throw new InvalidOperationException("Trilu source texture is missing: " + layer.name);
            ValidateShapeAndParameters(layer, logicalShape);
            if (AexisGraphSession.IsStrictLinearMatTexture(source))
            {
                if (logicalShape.dims != 2 || source.width != logicalShape.w || source.height != logicalShape.h)
                    throw new InvalidOperationException("Trilu LinearMat path requires exact rank-2 [rows,columns] storage: " + layer.name);
                return;
            }
            if (source.texture.dimension != TextureDimension.Tex2DArray
                || source.width != logicalShape.w
                || source.height != logicalShape.h)
            {
                throw new InvalidOperationException("Trilu Pack4 path requires exact Texture2DArray X/Y storage for the final two axes: " + layer.name);
            }
        }

        private static void ValidateContract(
            AexisGraphModel.Layer layer,
            AexisGraphSession.CmdTensorRef source,
            AexisGraphSession.BufferShape logicalShape)
        {
            if (source == null || source.texture == null)
                throw new InvalidOperationException("Trilu source command-buffer texture is missing: " + layer.name);
            ValidateShapeAndParameters(layer, logicalShape);
            if (AexisGraphSession.IsStrictLinearMatTexture(source))
            {
                if (logicalShape.dims != 2 || source.width != logicalShape.w || source.height != logicalShape.h)
                    throw new InvalidOperationException("Trilu LinearMat command-buffer path requires exact rank-2 storage: " + layer.name);
                return;
            }
            if (source.texture.dimension != TextureDimension.Tex2DArray
                || source.width != logicalShape.w
                || source.height != logicalShape.h)
            {
                throw new InvalidOperationException("Trilu Pack4 command-buffer path requires exact Texture2DArray X/Y storage: " + layer.name);
            }
        }

        private static void ValidateShapeAndParameters(
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape logicalShape)
        {
            if (logicalShape.dims < 2 || logicalShape.dims > 4)
                throw new InvalidOperationException("Trilu supports logical rank 2 through 4: " + layer.name);
            var upper = layer.GetInt(0, 1);
            if (upper != 0 && upper != 1)
                throw new InvalidOperationException("Trilu upper must be 0 or 1: " + layer.name);
        }
    }
}
