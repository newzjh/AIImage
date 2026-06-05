using System;
using UnityEngine;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnSqueezeLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnSqueezeLayerRepro() : base(NcnnLayerTypes.Squeeze, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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
            var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcTensor == null)
                throw new InvalidOperationException("Squeeze source not found: " + layer.name);

            var squeezed = NcnnRepro.ResolveSqueezeView(srcTensor, layer);
            if (TryAliasExistingBuffer(layer.bottomNames[0], layer.topNames[0], bufferBlobs, bufferRefs, bufferViews, squeezed, out var aliased))
            {
                if (aliased)
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }
            }

            var outTensor = owner.RentTempTensorBuffer(squeezed.dims, squeezed.w, squeezed.h, squeezed.d, squeezed.c);
            owner.Ops.CopyBufPartial(srcBuf, 0, outTensor.buffer, srcTensor.elementCount);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: squeezed.dims <= 3,
                textureBlobs,
                textureShapes,
                bufferBlobs,
                bufferRefs,
                bufferViews,
                tempOwned);

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
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
                shapes[layer.topNames[0]] = NcnnRepro.ResolveSqueezeShape(srcShape, layer);
            src.refs++;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool TryAliasExistingBuffer(
            string bottomName,
            string topName,
            System.Collections.Generic.Dictionary<string, ComputeBuffer> bufferBlobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferRef> bufferRefs,
            System.Collections.Generic.Dictionary<string, NcnnTensorBuffer> bufferViews,
            NcnnTensorBuffer outView,
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
            bufferViews[topName] = outView;
            aliased = true;
            return true;
        }
    }
}
