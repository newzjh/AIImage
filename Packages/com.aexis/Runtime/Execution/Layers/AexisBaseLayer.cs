using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    public enum AexisLayerPathPreference
    {
        Auto,
        Pack4Rt,
        Buffer
    }

    public sealed class AexisLayerBufferContext
    {
        public Dictionary<string, AexisGraphSession.TensorRef> textureBlobs;
        public Dictionary<string, AexisGraphSession.BufferShape> textureShapes;
        public Dictionary<string, ComputeBuffer> bufferBlobs;
        public Dictionary<string, AexisGraphSession.BufferRef> bufferRefs;
        public Dictionary<string, AexisTensorBuffer> bufferViews;
        public Dictionary<string, AexisGraphSession.IndexRef> indexBlobs;
        public Dictionary<string, int> remaining;
        public ICollection<string> pinnedNames;
        public List<IDisposable> tempOwned;
    }

    public sealed class AexisLayerCommandBufferContext
    {
        public CommandBuffer commandBuffer;
        public Dictionary<string, AexisGraphSession.CmdTensorRef> blobs;
        public Dictionary<string, AexisGraphSession.BufferShape> shapes;
        public Dictionary<string, int> remaining;
        public ICollection<string> pinnedNames;
    }

    // Production execution is texture-native. Buffer implementations are explicit
    // debug oracles only; a missing texture path must fail instead of publishing an
    // uninitialized tensor or materializing an activation through ComputeBuffer.
    public abstract class AexisBaseLayer
    {
        internal const string ComputeBufferPathObsoleteMessage = "ComputeBuffer path is only kept for temporary real-time debugging. Please migrate the final implementation to ExecuteRenderTexturePath and ExecuteCommandBuffer.";

        protected AexisBaseLayer(
            AexisLayerTypeKey typeKey,
            bool supportsBufferPath,
            bool supportsCommandBufferPath)
        {
            TypeKey = typeKey;
            SupportsBufferPath = supportsBufferPath;
            SupportsCommandBufferPath = supportsCommandBufferPath;
        }

        public AexisLayerTypeKey TypeKey { get; }
        public bool SupportsBufferPath { get; }
        public bool SupportsCommandBufferPath { get; }
        public AexisLayerPathPreference PreferredPath { get; set; } = AexisLayerPathPreference.Auto;

        public virtual AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            return default;
        }

        public virtual void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            ExecuteRenderTexturePath(owner, layer, context);
        }

        // Most layers are a single GPU submission. Layers with a verified internal
        // dependency chain can override this to expose safe frame-yield boundaries
        // without changing the texture-only execution contract.
        public virtual IEnumerable<bool> ExecuteBufferIncremental(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            ExecuteBuffer(owner, layer, context);
            yield return true;
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public virtual void ExecuteComputeBufferPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            throw new NotSupportedException("ComputeBuffer path is not implemented for layer type: " + TypeKey);
        }

        public virtual void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public virtual void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));
            if (context == null)
                throw new ArgumentNullException(nameof(context));

            throw new NotSupportedException(
                "CommandBuffer Pack4 path is not implemented"
                + " | layer=" + (layer.name ?? string.Empty)
                + " | type=" + (layer.typeName ?? TypeKey.ToString())
                + " | bottoms=" + string.Join(",", layer.bottomNames ?? Array.Empty<string>())
                + " | tops=" + string.Join(",", layer.topNames ?? Array.Empty<string>())
                + " | rejected_fallback=placeholder");
        }
    }

    internal static class AexisPack4LayerHelpers
    {
        public static void ExecuteShapePreservingRenderTexture(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisLayerBufferContext context,
            string opName,
            Action<RenderTexture, AexisGraphSession.BufferShape, RenderTexture> execute)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (execute == null)
                throw new ArgumentNullException(nameof(execute));

            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var src, out var srcShape))
                throw new InvalidOperationException(opName + " render-texture path requires pack4 texture input: " + layer.name);

            var storageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
            var strictLinear = AexisGraphSession.IsStrictLinearMatTexture(src);
            if (!strictLinear && !AexisGraphSession.BufferShapeEquals(srcShape, storageShape))
                throw new InvalidOperationException(
                    opName + " render-texture path requires matching logical/storage shape"
                    + " | layer=" + layer.name
                    + " | logical=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | storage=d" + storageShape.dims + ":" + storageShape.w + "x" + storageShape.h + "x" + storageShape.d + "x" + storageShape.c);

            RenderTexture materialized = null;
            RenderTexture outRt = null;
            try
            {
                var input = src.texture;
                if (strictLinear)
                {
                    materialized = owner.RentTempArray(
                        Mathf.Max(1, srcShape.w),
                        Mathf.Max(1, srcShape.h),
                        ResolvePack4SliceCount(srcShape),
                        AexisGraphSession.ResolveTensorTextureFormat(srcShape.dims));
                    owner.Ops.ReshapeLinearMatToPack4(
                        src.texture,
                        storageShape.w,
                        storageShape.h,
                        srcShape.w,
                        srcShape.h,
                        srcShape.d,
                        srcShape.c,
                        srcShape.dims,
                        materialized);
                    input = materialized;
                }

                var outputDepth = Mathf.Max(1, input.volumeDepth > 0 ? input.volumeDepth : 1);
                outRt = owner.RentTempArray(input.width, input.height, outputDepth, AexisGraphSession.ResolveTensorTextureFormat(srcShape.dims));
                execute(input, srcShape, outRt);

                if (strictLinear)
                {
                    var outMat = owner.RentTempMat(storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                    owner.Ops.ReshapePack4ToLinearMat(outRt, srcShape.w, srcShape.h, srcShape.d, srcShape.c, srcShape.dims, outMat);
                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outMat, srcShape, storageShape);
                }
                else
                {
                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape, storageShape);
                    outRt = null;
                }
            }
            finally
            {
                if (materialized != null)
                    owner.ReturnTempArray(materialized);
                if (outRt != null)
                    owner.ReturnTempArray(outRt);
            }

            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public static void ExecuteShapePreservingCommandBuffer(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisLayerCommandBufferContext context,
            string opName,
            Action<CommandBuffer, ComputeTexture, AexisGraphSession.BufferShape, ComputeTexture> execute)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (execute == null)
                throw new ArgumentNullException(nameof(execute));

            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
            var strictLinear = AexisGraphSession.IsStrictLinearMatTexture(src);
            if (!strictLinear && !AexisGraphSession.BufferShapeEquals(srcShape, storageShape))
                throw new InvalidOperationException(
                    opName + " command-buffer path requires matching logical/storage shape"
                    + " | layer=" + layer.name
                    + " | logical=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | storage=d" + storageShape.dims + ":" + storageShape.w + "x" + storageShape.h + "x" + storageShape.d + "x" + storageShape.c);

            ComputeTexture materialized = null;
            ComputeTexture outArr = null;
            try
            {
                var input = src.texture;
                if (strictLinear)
                {
                    materialized = owner.RentTempArray(
                        cmd,
                        Mathf.Max(1, srcShape.w),
                        Mathf.Max(1, srcShape.h),
                        ResolvePack4SliceCount(srcShape),
                        AexisGraphSession.ResolveTensorTextureFormat(srcShape.dims));
                    owner.Ops.ReshapeLinearMatToPack4(
                        cmd,
                        src.texture,
                        storageShape.w,
                        storageShape.h,
                        srcShape.w,
                        srcShape.h,
                        srcShape.d,
                        srcShape.c,
                        srcShape.dims,
                        materialized);
                    input = materialized;
                }

                var outputDepth = Mathf.Max(1, input.depth);
                outArr = owner.RentTempArray(cmd, input.width, input.height, outputDepth, AexisGraphSession.ResolveTensorTextureFormat(srcShape.dims));
                execute(cmd, input, srcShape, outArr);

                if (strictLinear)
                {
                    var outMat = owner.RentTempMat(cmd, storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                    owner.Ops.ReshapePack4ToLinearMat(cmd, outArr, srcShape.w, srcShape.h, srcShape.d, srcShape.c, srcShape.dims, outMat);
                    blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outMat, srcShape, storageShape, owned: true);
                }
                else
                {
                    blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outArr, srcShape, storageShape, owned: true);
                    outArr = null;
                }

                if (shapes != null)
                    shapes[layer.topNames[0]] = srcShape;
            }
            finally
            {
                if (materialized != null)
                    owner.ReturnTempArray(cmd, materialized);
                if (outArr != null)
                    owner.ReturnTempArray(cmd, outArr);
            }

            owner.ConsumeCmd(cmd, blobs, context.remaining, layer.bottomNames, context.pinnedNames, shapes);
        }

        private static int ResolvePack4SliceCount(AexisGraphSession.BufferShape shape)
        {
            var channels = shape.dims >= 3 ? Mathf.Max(1, shape.c) : 1;
            var packs = Mathf.Max(1, Mathf.CeilToInt(channels / 4f));
            return shape.dims == 4 ? Mathf.Max(1, shape.d) * packs : packs;
        }
    }
}
