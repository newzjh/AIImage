using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnSplitLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnSplitLayerRepro() : base(NcnnLayerTypes.Split, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            ExecuteRenderTexturePath(owner, layer, context);
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!bufferBlobs.TryGetValue(layer.bottomNames[0], out var srcBuf) || srcBuf == null)
            {
                srcBuf = owner.GetOrConvertToBuffer(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    bufferBlobs,
                    context.textureShapes,
                    bufferViews,
                    context.tempOwned);
            }
            if (srcBuf == null)
                throw new InvalidOperationException("Split source not found: " + layer.name);

            var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            for (var i = 0; i < layer.topNames.Length; i++)
            {
                bufferBlobs[layer.topNames[i]] = srcBuf;
                if (bufferRefs.TryGetValue(layer.bottomNames[0], out var srcRef) && srcRef != null)
                {
                    bufferRefs[layer.topNames[i]] = srcRef;
                    srcRef.refs++;
                }
                else
                {
                    bufferRefs[layer.topNames[i]] = owner.NewOwnedBufferRef(layer.topNames[i], srcBuf);
                }

                if (srcTensor != null)
                    bufferViews[layer.topNames[i]] = srcTensor;
            }

            owner.Consume(context.textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var hasTexture = textureBlobs.TryGetValue(layer.bottomNames[0], out var srcTex) && srcTex != null && srcTex.texture != null;
            var srcTexShape = hasTexture ? NcnnRepro.GetTextureShape(textureShapes, srcTex, layer.bottomNames[0]) : default;
            if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var srcBuf) && srcBuf != null)
            {
                var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                for (var i = 0; i < layer.topNames.Length; i++)
                {
                    bufferBlobs[layer.topNames[i]] = srcBuf;
                    if (bufferRefs.TryGetValue(layer.bottomNames[0], out var srcRef) && srcRef != null)
                    {
                        bufferRefs[layer.topNames[i]] = srcRef;
                        srcRef.refs++;
                    }
                    if (srcTensor != null)
                        bufferViews[layer.topNames[i]] = srcTensor;

                    if (hasTexture)
                    {
                        var storageShape = NcnnRepro.GetTextureStorageShape(srcTex, srcTexShape);
                        textureBlobs[layer.topNames[i]] = NcnnRepro.CreateTextureAlias(srcTex, srcTexShape, storageShape);
                        textureShapes[layer.topNames[i]] = srcTexShape;
                    }
                }
            }
            else
            {
                var src = hasTexture ? srcTex : owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                var shape = hasTexture ? srcTexShape : NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                var storageShape = NcnnRepro.GetTextureStorageShape(src, shape);
                for (var i = 0; i < layer.topNames.Length; i++)
                {
                    textureBlobs[layer.topNames[i]] = NcnnRepro.CreateTextureAlias(src, shape, storageShape);
                    textureShapes[layer.topNames[i]] = shape;
                }
            }

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
                        var cmd = context.commandBuffer;
                        var blobs = context.blobs;
                        var shapes = context.shapes;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;

                        do
                        {
                                                var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
                                                for (var i = 0; i < layer.topNames.Length; i++)
                                                {
                                                    var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
                                                    blobs[layer.topNames[i]] = NcnnRepro.CreateCmdTensorAlias(src, srcShape, storageShape);
                                                    if (shapes != null)
                                                        shapes[layer.topNames[i]] = srcShape;
                                                }
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }
    }
}
