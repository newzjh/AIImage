using System;
using UnityEngine;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnFlattenLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnFlattenLayerRepro()
            : base(NcnnLayerTypes.Flatten, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Flatten source not found: " + layer.name);

            if (TryAliasExistingBuffer(layer.bottomNames[0], layer.topNames[0], bufferBlobs, bufferRefs, bufferViews, srcView, out var aliased))
            {
                if (aliased)
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }
            }

            var outTensor = owner.RentTempTensorBuffer(1, srcView.elementCount);
            owner.Ops.CopyBufPartial(srcBuf, 0, outTensor.buffer, srcView.elementCount);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: true,
                textureBlobs,
                textureShapes,
                bufferBlobs,
                bufferRefs,
                bufferViews,
                tempOwned);

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            blobs[layer.topNames[0]] = src;
            if (shapes != null)
                shapes[layer.topNames[0]] = new NcnnRepro.BufferShape(1, srcShape.w * srcShape.h * srcShape.d * srcShape.c, 1, 1, 1);
            src.refs++;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool TryAliasExistingBuffer(
            string bottomName,
            string topName,
            System.Collections.Generic.Dictionary<string, ComputeBuffer> bufferBlobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferRef> bufferRefs,
            System.Collections.Generic.Dictionary<string, NcnnTensorBuffer> bufferViews,
            NcnnTensorBuffer srcView,
            out bool aliased)
        {
            aliased = false;
            if (!bufferBlobs.TryGetValue(bottomName, out var existing) || existing == null)
                return false;
            if (!bufferRefs.TryGetValue(bottomName, out var existingRef) || existingRef == null || !existingRef.owned)
                return false;

            bufferBlobs[topName] = existing;
            bufferRefs[topName] = existingRef;
            existingRef.refs++;
            bufferViews[topName] = srcView.Reshape(1, srcView.elementCount);
            aliased = true;
            return true;
        }
    }
}
