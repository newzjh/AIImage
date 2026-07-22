using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Named unary aliases are production texture operators. They deliberately expose no
    // ComputeBuffer path so a missing texture descriptor cannot trigger materialization.
    public sealed class AexisUnaryOpAliasLayer : AexisBaseLayer
    {
        private readonly int _opType;

        public AexisUnaryOpAliasLayer(AexisLayerTypeKey typeKey, int opType)
            : base(typeKey, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
            _opType = opType;
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;

            if (!AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape))
                throw new InvalidOperationException(TypeKey + " render-texture path requires supported texture input: " + layer.name);

            if (AexisGraphSession.IsStrictLinearMatTexture(srcTex))
            {
                var storageShape = AexisGraphSession.GetTextureStorageShape(srcTex, srcShape);
                var outRt = owner.RentTempMat(storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.UnaryOpLinearMat(srcTex.texture, _opType, outRt);
                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape, storageShape);
            }
            else
            {
                var storageShape = AexisGraphSession.GetTextureStorageShape(srcTex, srcShape);
                var outputDepth = srcShape.dims == 4 ? srcShape.d * srcTex.packs : srcTex.packs;
                var outRt = owner.RentTempArray(srcTex.width, srcTex.height, outputDepth, AexisGraphSession.ResolveTensorTextureFormat(srcShape.dims));
                owner.Ops.UnaryOpPack4(srcTex.texture, outputDepth, _opType, outRt, srcShape.dims >= 3 ? srcShape.c : 0);
                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape, storageShape);
            }
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            if (AexisGraphSession.IsStrictLinearMatTexture(src) && srcShape.dims <= 2)
            {
                var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                var outMat = owner.RentTempMat(cmd, storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.UnaryOpLinearMat(cmd, src.texture, _opType, outMat);
                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outMat, srcShape, storageShape, owned: true);
            }
            else
            {
                var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                var outputDepth = srcShape.dims == 4 ? srcShape.d * src.packs : src.packs;
                var outArr = owner.RentTempArray(cmd, src.width, src.height, outputDepth, AexisGraphSession.ResolveTensorTextureFormat(srcShape.dims));
                owner.Ops.UnaryOpPack4(cmd, src.texture, outputDepth, _opType, outArr, srcShape.dims >= 3 ? srcShape.c : 0);
                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outArr, srcShape, storageShape, owned: true, blobName: layer.topNames[0]);
            }
            if (shapes != null)
                shapes[layer.topNames[0]] = srcShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
