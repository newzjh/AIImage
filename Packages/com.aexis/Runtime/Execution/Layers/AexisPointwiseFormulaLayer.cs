using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Formula aliases are production Pack4 texture operators. No implicit buffer/texture
    // conversion is allowed: strict preflight and runtime require an existing texture input.
    public sealed class AexisPointwiseFormulaLayer : AexisBaseLayer
    {
        public AexisPointwiseFormulaLayer(AexisLayerTypeKey typeKey)
            : base(typeKey, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            ResolveFormula(TypeKey, layer, out var type, out var a, out var b, out var c);
            if (!AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape)
                || AexisGraphSession.IsStrictLinearMatTexture(srcTex)
                || !AexisGraphSession.MatchesPack4TextureStorage(srcTex, srcShape))
                throw new InvalidOperationException(TypeKey + " render-texture path requires an existing descriptor-valid Pack4 texture input: " + layer.name);

            var storageShape = AexisGraphSession.GetTextureStorageShape(srcTex, srcShape);
            var outputDepth = srcShape.dims == 4 ? srcShape.d * srcTex.packs : srcTex.packs;
            var outRt = owner.RentTempArray(srcTex.width, srcTex.height, outputDepth, srcTex.texture.format);
            owner.Ops.PointwisePack4(
                srcTex.texture,
                outputDepth,
                type,
                a,
                b,
                outRt,
                c,
                srcShape.dims >= 3 ? srcShape.c : 0);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape, storageShape);
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            ResolveFormula(TypeKey, layer, out var type, out var a, out var b, out var c);

            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
            var outputDepth = srcShape.dims == 4 ? srcShape.d * src.packs : src.packs;
            var outArr = owner.RentTempArray(cmd, src.width, src.height, outputDepth, src.texture.format);
            owner.Ops.PointwisePack4(
                cmd,
                src.texture,
                outputDepth,
                type,
                a,
                b,
                outArr,
                c,
                srcShape.dims >= 3 ? srcShape.c : 0);
            blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outArr, srcShape, storageShape, owned: true, blobName: layer.topNames[0]);
            if (shapes != null)
                shapes[layer.topNames[0]] = srcShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static void ResolveFormula(
            AexisLayerTypeKey typeKey,
            AexisGraphModel.Layer layer,
            out AexisOps.PointwiseType pointwiseType,
            out float a,
            out float b,
            out float c)
        {
            a = 0f;
            b = 0f;
            c = 0f;

            if (typeKey == AexisLayerTypes.ELU)
            {
                pointwiseType = AexisOps.PointwiseType.Elu;
                a = layer.GetFloat(0, 0.1f);
                return;
            }

            if (typeKey == AexisLayerTypes.Erf)
            {
                pointwiseType = AexisOps.PointwiseType.Erf;
                return;
            }

            if (typeKey == AexisLayerTypes.HardSigmoid)
            {
                pointwiseType = AexisOps.PointwiseType.HardSigmoid;
                a = layer.GetFloat(0, 0.2f);
                b = layer.GetFloat(1, 0.5f);
                return;
            }

            if (typeKey == AexisLayerTypes.HardSwish)
            {
                pointwiseType = AexisOps.PointwiseType.HardSwish;
                a = layer.GetFloat(0, 0.2f);
                b = layer.GetFloat(1, 0.5f);
                return;
            }

            if (typeKey == AexisLayerTypes.Mish)
            {
                pointwiseType = AexisOps.PointwiseType.Mish;
                return;
            }

            if (typeKey == AexisLayerTypes.SELU)
            {
                pointwiseType = AexisOps.PointwiseType.Selu;
                a = layer.GetFloat(0, 1.67326324f);
                b = layer.GetFloat(1, 1.050700987f);
                return;
            }

            if (typeKey == AexisLayerTypes.Shrink)
            {
                pointwiseType = AexisOps.PointwiseType.Shrink;
                a = layer.GetFloat(0, 0f);
                b = layer.GetFloat(1, 0.5f);
                return;
            }

            if (typeKey == AexisLayerTypes.Softplus)
            {
                pointwiseType = AexisOps.PointwiseType.Softplus;
                return;
            }

            if (typeKey == AexisLayerTypes.BNLL)
            {
                pointwiseType = AexisOps.PointwiseType.Softplus;
                return;
            }

            if (typeKey == AexisLayerTypes.Exp)
            {
                pointwiseType = AexisOps.PointwiseType.Exp;
                a = layer.GetFloat(0, -1f);
                b = layer.GetFloat(1, 1f);
                c = layer.GetFloat(2, 0f);
                return;
            }

            if (typeKey == AexisLayerTypes.Log)
            {
                pointwiseType = AexisOps.PointwiseType.Log;
                a = layer.GetFloat(0, -1f);
                b = layer.GetFloat(1, 1f);
                c = layer.GetFloat(2, 0f);
                return;
            }

            if (typeKey == AexisLayerTypes.Softsign)
            {
                pointwiseType = AexisOps.PointwiseType.Softsign;
                return;
            }

            if (typeKey == AexisLayerTypes.IsInf)
            {
                pointwiseType = AexisOps.PointwiseType.IsInf;
                a = layer.GetInt(0, 1);
                b = layer.GetInt(1, 1);
                return;
            }

            if (typeKey == AexisLayerTypes.IsNaN)
            {
                pointwiseType = AexisOps.PointwiseType.IsNaN;
                return;
            }

            if (typeKey == AexisLayerTypes.CELU)
            {
                pointwiseType = AexisOps.PointwiseType.Celu;
                a = layer.GetFloat(0, 1f);
                return;
            }

            if (typeKey == AexisLayerTypes.Power)
            {
                pointwiseType = AexisOps.PointwiseType.Power;
                a = layer.GetFloat(0, 1f);
                b = layer.GetFloat(1, 1f);
                c = layer.GetFloat(2, 0f);
                return;
            }

            if (typeKey == AexisLayerTypes.Threshold)
            {
                pointwiseType = AexisOps.PointwiseType.Threshold;
                a = layer.GetFloat(0, 0f);
                return;
            }

            if (typeKey == AexisLayerTypes.ThresholdedRelu)
            {
                pointwiseType = AexisOps.PointwiseType.ThresholdedRelu;
                a = layer.GetFloat(0, 1f);
                return;
            }

            throw new InvalidOperationException("Unsupported pointwise formula layer: " + typeKey);
        }
    }
}
