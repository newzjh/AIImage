using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    public enum NcnnLayerPathPreference
    {
        Auto,
        Pack4Rt,
        Buffer
    }

    public sealed class NcnnLayerBufferContext
    {
        public Dictionary<string, NcnnGraphSession.TensorRef> textureBlobs;
        public Dictionary<string, NcnnGraphSession.BufferShape> textureShapes;
        public Dictionary<string, ComputeBuffer> bufferBlobs;
        public Dictionary<string, NcnnGraphSession.BufferRef> bufferRefs;
        public Dictionary<string, NcnnTensorBuffer> bufferViews;
        public Dictionary<string, NcnnGraphSession.IndexRef> indexBlobs;
        public Dictionary<string, int> remaining;
        public ICollection<string> pinnedNames;
        public List<IDisposable> tempOwned;
    }

    public sealed class NcnnLayerCommandBufferContext
    {
        public CommandBuffer commandBuffer;
        public Dictionary<string, NcnnGraphSession.CmdTensorRef> blobs;
        public Dictionary<string, NcnnGraphSession.BufferShape> shapes;
        public Dictionary<string, int> remaining;
        public ICollection<string> pinnedNames;
    }

    // Migration guidance for all LayerRepro implementations:
    // 1. Keep the compute-buffer path only as a compatibility / truth-path fallback and avoid expanding new buffer-only branches.
    // 2. Prefer migrating execution to the pack4 RenderTexture path first, because it is the near-term primary runtime path.
    // 3. Long term, migrate toward the ComputeTexture-based ExecuteCommandBuffer pack4 RT path so async compute and command-buffer temporary RT allocation are both supported.
    public abstract class NcnnBaseLayerRepro
    {
        internal const string ComputeBufferPathObsoleteMessage = "ComputeBuffer path is only kept for temporary real-time debugging. Please migrate the final implementation to ExecuteRenderTexturePath and ExecuteCommandBuffer.";

        protected NcnnBaseLayerRepro(
            NcnnLayerTypeKey typeKey,
            bool supportsBufferPath,
            bool supportsCommandBufferPath)
        {
            TypeKey = typeKey;
            SupportsBufferPath = supportsBufferPath;
            SupportsCommandBufferPath = supportsCommandBufferPath;
        }

        public NcnnLayerTypeKey TypeKey { get; }
        public bool SupportsBufferPath { get; }
        public bool SupportsCommandBufferPath { get; }
        public NcnnLayerPathPreference PreferredPath { get; set; } = NcnnLayerPathPreference.Auto;

        public virtual NcnnGraphSession.LayerLoadMetrics LoadLayer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            return default;
        }

        public virtual void ExecuteBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            ExecuteRenderTexturePath(owner, layer, context);
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public virtual void ExecuteComputeBufferPath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            throw new NotSupportedException("ComputeBuffer path is not implemented for layer type: " + TypeKey);
        }

        public virtual void ExecuteRenderTexturePath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public virtual void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));
            if (context == null)
                throw new ArgumentNullException(nameof(context));

            if (layer.bottomNames != null && layer.bottomNames.Length > 0)
            {
                new NcnnNoopLayerRepro().ExecuteCommandBuffer(owner, layer, context);
                return;
            }

            if (layer.topNames != null && layer.topNames.Length > 0)
                owner.PublishCmdTensorLikeInput(context.commandBuffer, layer.topNames[0], 1, 1, 1, context.blobs, context.shapes);
        }
    }

    internal static class NcnnPack4LayerHelpers
    {
        public static void ExecuteShapePreservingRenderTexture(
            NcnnGraphSession owner,
            NcnnParamModel.Layer layer,
            NcnnLayerBufferContext context,
            string opName,
            Action<RenderTexture, NcnnGraphSession.BufferShape, RenderTexture> execute)
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

            var storageShape = NcnnGraphSession.GetTextureStorageShape(src, srcShape);
            var strictLinear = NcnnGraphSession.IsStrictLinearMatTexture(src);
            if (!strictLinear && !NcnnGraphSession.BufferShapeEquals(srcShape, storageShape))
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
                        NcnnGraphSession.ResolveTensorTextureFormat(srcShape.dims));
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
                outRt = owner.RentTempArray(input.width, input.height, outputDepth, NcnnGraphSession.ResolveTensorTextureFormat(srcShape.dims));
                execute(input, srcShape, outRt);

                if (strictLinear)
                {
                    var outMat = owner.RentTempMat(storageShape.w, storageShape.h, NcnnGraphSession.ResolveLinearMatTextureFormat());
                    owner.Ops.ReshapePack4ToLinearMat(outRt, srcShape.w, srcShape.h, srcShape.d, srcShape.c, srcShape.dims, outMat);
                    NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outMat, srcShape, storageShape);
                }
                else
                {
                    NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape, storageShape);
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
            NcnnGraphSession owner,
            NcnnParamModel.Layer layer,
            NcnnLayerCommandBufferContext context,
            string opName,
            Action<CommandBuffer, ComputeTexture, NcnnGraphSession.BufferShape, ComputeTexture> execute)
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
            var src = NcnnGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = NcnnGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var storageShape = NcnnGraphSession.GetCmdStorageShape(src, srcShape);
            var strictLinear = NcnnGraphSession.IsStrictLinearMatTexture(src);
            if (!strictLinear && !NcnnGraphSession.BufferShapeEquals(srcShape, storageShape))
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
                        NcnnGraphSession.ResolveTensorTextureFormat(srcShape.dims));
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
                outArr = owner.RentTempArray(cmd, input.width, input.height, outputDepth, NcnnGraphSession.ResolveTensorTextureFormat(srcShape.dims));
                execute(cmd, input, srcShape, outArr);

                if (strictLinear)
                {
                    var outMat = owner.RentTempMat(cmd, storageShape.w, storageShape.h, NcnnGraphSession.ResolveLinearMatTextureFormat());
                    owner.Ops.ReshapePack4ToLinearMat(cmd, outArr, srcShape.w, srcShape.h, srcShape.d, srcShape.c, srcShape.dims, outMat);
                    blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(outMat, srcShape, storageShape, owned: true);
                }
                else
                {
                    blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(outArr, srcShape, storageShape, owned: true);
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

        private static int ResolvePack4SliceCount(NcnnGraphSession.BufferShape shape)
        {
            var channels = shape.dims >= 3 ? Mathf.Max(1, shape.c) : 1;
            var packs = Mathf.Max(1, Mathf.CeilToInt(channels / 4f));
            return shape.dims == 4 ? Mathf.Max(1, shape.d) * packs : packs;
        }
    }
}
