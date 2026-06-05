using System;
using UnityEngine;

namespace NcnnCompute
{
    public sealed class NcnnFlattenLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnFlattenLayerRepro()
            : base(NcnnLayerTypes.Flatten, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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

            var outBuf = owner.RentTempBuffer(srcView.elementCount, sizeof(float));
            owner.Ops.CopyBufPartial(srcBuf, 0, outBuf, srcView.elementCount);
            bufferBlobs[layer.topNames[0]] = outBuf;
            bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], outBuf);
            bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, 1, srcView.elementCount, 1, 1, 1, false);
            tempOwned.Add(outBuf);

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            blobs[layer.topNames[0]] = src;
            src.refs++;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
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
