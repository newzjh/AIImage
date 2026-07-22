using System;
using UnityEngine;

namespace Aexis.Execution
{
    // ncnn CopyTo clones the first input and overwrites an offset ROI with the
    // second input. Each shader invocation owns one destination pack, so an
    // unaligned channel offset cannot cause competing read-modify-write stores.
    public sealed class AexisCopyToLayer : AexisBaseLayer
    {
        internal readonly struct CopyToOffsets
        {
            public readonly int w;
            public readonly int h;
            public readonly int d;
            public readonly int c;

            public CopyToOffsets(int w, int h, int d, int c)
            {
                this.w = w;
                this.h = h;
                this.d = d;
                this.c = c;
            }
        }

        public AexisCopyToLayer()
            : base(AexisLayerTypes.CopyTo, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (layer?.bottomNames == null || layer.bottomNames.Length != 2 || layer.topNames == null || layer.topNames.Length != 1)
                throw new InvalidOperationException("CopyTo requires exactly two inputs and one output: " + layer?.name);
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out var self, out var selfShape)
                || !owner.TryGetPack4Texture(layer.bottomNames[1], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out var src, out var srcShape))
            {
                throw new InvalidOperationException("CopyTo requires two Pack4 texture inputs and does not permit buffer materialization: " + layer.name);
            }
            if (!CanExecute(self, selfShape, src, srcShape, layer, out var offsets, out var reason))
                throw new InvalidOperationException("CopyTo texture profile rejected layer " + layer.name + ": " + reason);

            var storageShape = AexisGraphSession.GetTextureStorageShape(self, selfShape);
            var outputDepth = Mathf.Max(1, self.texture.volumeDepth);
            var output = owner.RentTempArray(self.width, self.height, outputDepth, self.texture.format);
            owner.Ops.CopyPack4(self.texture, 0, output, 0, outputDepth);
            owner.Ops.CopyToPack4(
                src.texture,
                srcShape.w,
                srcShape.h,
                srcShape.dims == 4 ? srcShape.d : 1,
                srcShape.c,
                selfShape.c,
                offsets.w,
                offsets.h,
                offsets.d,
                offsets.c,
                output);
            AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, selfShape, storageShape);
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (layer?.bottomNames == null || layer.bottomNames.Length != 2 || layer.topNames == null || layer.topNames.Length != 1)
                throw new InvalidOperationException("CopyTo requires exactly two inputs and one output: " + layer?.name);

            var self = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var src = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[1]);
            var selfShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[1]);
            if (!CanExecute(self, selfShape, src, srcShape, layer, out var offsets, out var reason))
                throw new InvalidOperationException("CopyTo command-buffer profile rejected layer " + layer.name + ": " + reason);

            var storageShape = AexisGraphSession.GetCmdStorageShape(self, selfShape);
            var outputDepth = Mathf.Max(1, self.texture.depth);
            var output = owner.RentTempArray(context.commandBuffer, self.width, self.height, outputDepth, self.texture.format);
            owner.Ops.CopyPack4(context.commandBuffer, self.texture, 0, output, 0, outputDepth);
            owner.Ops.CopyToPack4(
                context.commandBuffer,
                src.texture,
                srcShape.w,
                srcShape.h,
                srcShape.dims == 4 ? srcShape.d : 1,
                srcShape.c,
                selfShape.c,
                offsets.w,
                offsets.h,
                offsets.d,
                offsets.c,
                output);
            context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, selfShape, storageShape, owned: true, blobName: layer.topNames[0]);
            if (context.shapes != null)
                context.shapes[layer.topNames[0]] = selfShape;
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        internal static bool TryResolveOffsets(
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape self,
            AexisGraphSession.BufferShape src,
            out CopyToOffsets offsets,
            out string reason)
        {
            offsets = default;
            reason = null;
            if (self.dims != src.dims || (self.dims != 3 && self.dims != 4))
            {
                reason = "the verified Pack4 profile requires equal rank-3 or rank-4 inputs";
                return false;
            }
            if (self.w <= 0 || self.h <= 0 || self.d <= 0 || self.c <= 0
                || src.w <= 0 || src.h <= 0 || src.d <= 0 || src.c <= 0)
            {
                reason = "logical shapes must be fully static and positive";
                return false;
            }

            // ncnn returns src directly for an exact-shape copy before resolving offsets.
            if (ShapesEqual(self, src))
            {
                offsets = new CopyToOffsets(0, 0, 0, 0);
                return true;
            }

            var w = layer.GetInt(0, 0);
            var h = layer.GetInt(1, 0);
            var d = layer.GetInt(13, 0);
            var c = layer.GetInt(2, 0);
            var starts = layer.GetInts(-23309, null);
            if (starts != null && starts.Length > 0)
            {
                var axes = layer.GetInts(-23311, null);
                if (axes == null || axes.Length == 0)
                {
                    if (starts.Length != self.dims)
                    {
                        reason = "numpy-style starts without axes must contain one entry per dimension";
                        return false;
                    }
                    axes = new int[self.dims];
                    for (var i = 0; i < axes.Length; i++) axes[i] = i;
                }
                else if (starts.Length != axes.Length)
                {
                    reason = "numpy-style starts and axes lengths differ";
                    return false;
                }

                w = h = d = c = 0;
                for (var i = 0; i < axes.Length; i++)
                {
                    var axis = axes[i] < 0 ? axes[i] + self.dims : axes[i];
                    if (axis < 0 || axis >= self.dims)
                    {
                        reason = "numpy-style axis " + axes[i] + " is outside rank " + self.dims;
                        return false;
                    }
                    var axisSize = GetNcnnAxisSize(self, axis);
                    var start = starts[i] == -233 ? 0 : starts[i];
                    if (start < 0) start += axisSize;
                    if (self.dims == 3)
                    {
                        if (axis == 0) c = start;
                        else if (axis == 1) h = start;
                        else w = start;
                    }
                    else
                    {
                        if (axis == 0) c = start;
                        else if (axis == 1) d = start;
                        else if (axis == 2) h = start;
                        else w = start;
                    }
                }
            }

            offsets = new CopyToOffsets(w, h, d, c);
            var srcDepth = src.dims == 4 ? src.d : 1;
            var selfDepth = self.dims == 4 ? self.d : 1;
            if (w < 0 || h < 0 || d < 0 || c < 0
                || w + src.w > self.w
                || h + src.h > self.h
                || d + srcDepth > selfDepth
                || c + src.c > self.c)
            {
                reason = "resolved ROI is outside the destination logical shape";
                return false;
            }
            return true;
        }

        private static bool CanExecute(
            AexisGraphSession.TensorRef self,
            AexisGraphSession.BufferShape selfShape,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            AexisGraphModel.Layer layer,
            out CopyToOffsets offsets,
            out string reason)
        {
            if (!AexisGraphSession.MatchesPack4TextureStorage(self, selfShape)
                || !AexisGraphSession.MatchesPack4TextureStorage(src, srcShape))
            {
                offsets = default;
                reason = "logical/storage descriptors do not match Pack4 texture storage";
                return false;
            }
            if (self.texture.format != src.texture.format)
            {
                offsets = default;
                reason = "input texture dtypes differ";
                return false;
            }
            return TryResolveOffsets(layer, selfShape, srcShape, out offsets, out reason);
        }

        private static bool CanExecute(
            AexisGraphSession.CmdTensorRef self,
            AexisGraphSession.BufferShape selfShape,
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape srcShape,
            AexisGraphModel.Layer layer,
            out CopyToOffsets offsets,
            out string reason)
        {
            if (!AexisGraphSession.MatchesPack4TextureStorage(self, selfShape)
                || !AexisGraphSession.MatchesPack4TextureStorage(src, srcShape))
            {
                offsets = default;
                reason = "logical/storage descriptors do not match Pack4 texture storage";
                return false;
            }
            if (self.texture.format != src.texture.format)
            {
                offsets = default;
                reason = "input texture dtypes differ";
                return false;
            }
            return TryResolveOffsets(layer, selfShape, srcShape, out offsets, out reason);
        }

        private static int GetNcnnAxisSize(AexisGraphSession.BufferShape shape, int axis)
        {
            if (shape.dims == 3)
                return axis == 0 ? shape.c : axis == 1 ? shape.h : shape.w;
            return axis == 0 ? shape.c : axis == 1 ? shape.d : axis == 2 ? shape.h : shape.w;
        }

        private static bool ShapesEqual(AexisGraphSession.BufferShape left, AexisGraphSession.BufferShape right)
        {
            return left.dims == right.dims && left.w == right.w && left.h == right.h
                && left.d == right.d && left.c == right.c;
        }
    }
}
