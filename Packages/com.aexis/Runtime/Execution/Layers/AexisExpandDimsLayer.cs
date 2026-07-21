using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisExpandDimsLayer : AexisBaseLayer
    {
        public AexisExpandDimsLayer() : base(AexisLayerTypes.ExpandDims, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            ExecuteRenderTexturePath(owner, layer, context);
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var axes = ResolveAxes(layer);
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
                throw new InvalidOperationException("ExpandDims source not found: " + layer.name);

            bufferBlobs[layer.topNames[0]] = srcBuf;
            if (bufferRefs.TryGetValue(layer.bottomNames[0], out var expandRef) && expandRef != null)
            {
                bufferRefs[layer.topNames[0]] = expandRef;
                expandRef.refs++;
            }
            else
            {
                bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], srcBuf);
            }

            var srcTensor = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcTensor == null)
                throw new InvalidOperationException("ExpandDims expects buffer input: " + layer.name);
            bufferViews[layer.topNames[0]] = ExpandBufferView(srcTensor, axes);

            owner.Consume(context.textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var axes = ResolveAxes(layer);
            if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var expandBuf) && expandBuf != null)
            {
                bufferBlobs[layer.topNames[0]] = expandBuf;
                if (bufferRefs.TryGetValue(layer.bottomNames[0], out var expandRef) && expandRef != null)
                {
                    bufferRefs[layer.topNames[0]] = expandRef;
                    expandRef.refs++;
                }

                var srcTensor = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                if (srcTensor == null)
                    throw new InvalidOperationException("ExpandDims expects buffer input: " + layer.name);
                bufferViews[layer.topNames[0]] = ExpandBufferView(srcTensor, axes);

                if (textureBlobs.TryGetValue(layer.bottomNames[0], out var expandTex) && expandTex != null && expandTex.texture != null)
                {
                    var sourceContract = AexisGraphSession.GetTextureContract(textureShapes, expandTex, layer.bottomNames[0]);
                    var srcShape = sourceContract.LogicalShape;
                    var outShape = ExpandTextureShape(srcShape, axes);
                    textureBlobs[layer.topNames[0]] = AexisGraphSession.CreateTextureAlias(expandTex, outShape, sourceContract.StorageShape);
                    textureShapes[layer.topNames[0]] = outShape;
                }
            }
            else
            {
                var src = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                var sourceContract = AexisGraphSession.GetTextureContract(textureShapes, src, layer.bottomNames[0]);
                var srcShape = sourceContract.LogicalShape;
                var outShape = ExpandTextureShape(srcShape, axes);
                textureBlobs[layer.topNames[0]] = AexisGraphSession.CreateTextureAlias(src, outShape, sourceContract.StorageShape);
                textureShapes[layer.topNames[0]] = outShape;
            }

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var sourceContract = AexisGraphSession.GetCmdTensorContract(src);
            var srcShape = sourceContract.LogicalShape;
            var axes = layer.GetInts(-23303, null);
            if (axes == null || axes.Length == 0)
                axes = layer.GetInts(3, Array.Empty<int>());
            if (axes == null || axes.Length == 0)
                throw new InvalidOperationException("ExpandDims missing axes: " + layer.name);

            static AexisGraphSession.BufferShape ExpandShape(AexisGraphSession.BufferShape input, int[] expandAxes)
            {
                var dims = input.dims;
                var w = input.w;
                var h = input.h;
                var d = input.d;
                var c = input.c;

                for (var i = 0; i < expandAxes.Length; i++)
                {
                    var outDims = dims + 1;
                    if (outDims > 4)
                        throw new InvalidOperationException("ExpandDims would exceed dims=4");

                    var ncnnAxis = expandAxes[i];
                    if (ncnnAxis < 0)
                        ncnnAxis += outDims;
                    if (ncnnAxis < 0 || ncnnAxis >= outDims)
                        throw new InvalidOperationException("ExpandDims axis out of range");

                    var tensorAxis = AexisGraphSession.MapNcnnAxisToTensorAxis(outDims, ncnnAxis);
                    var sizes = new[] { w, h, dims == 4 ? d : c, dims == 4 ? c : 1 };
                    var expanded = new[] { 1, 1, 1, 1 };
                    for (var axisIndex = 0; axisIndex < outDims; axisIndex++)
                    {
                        if (axisIndex < tensorAxis)
                            expanded[axisIndex] = sizes[axisIndex];
                        else if (axisIndex == tensorAxis)
                            expanded[axisIndex] = 1;
                        else
                            expanded[axisIndex] = sizes[axisIndex - 1];
                    }

                    dims = outDims;
                    w = expanded[0];
                    h = expanded[1];
                    if (dims == 3)
                    {
                        d = 1;
                        c = expanded[2];
                    }
                    else
                    {
                        d = expanded[2];
                        c = expanded[3];
                    }
                }

                return new AexisGraphSession.BufferShape(dims, w, h, d, c);
            }

            var outShape = ExpandShape(srcShape, axes);
            blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorAlias(src, outShape, sourceContract.StorageShape);
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static int[] ResolveAxes(AexisGraphModel.Layer layer)
        {
            var axes = layer.GetInts(-23303, null);
            if (axes == null || axes.Length == 0)
                axes = layer.GetInts(3, Array.Empty<int>());
            if (axes == null || axes.Length == 0)
                throw new InvalidOperationException("ExpandDims missing axes: " + layer.name);
            return axes;
        }

        private static AexisTensorBuffer ExpandBufferView(AexisTensorBuffer input, int[] expandAxes)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            var current = input;
            for (var i = 0; i < expandAxes.Length; i++)
            {
                var outDims = current.dims + 1;
                if (outDims > 4)
                    throw new InvalidOperationException("ExpandDims would exceed dims=4");

                var ncnnAxis = expandAxes[i];
                if (ncnnAxis < 0)
                    ncnnAxis += outDims;
                if (ncnnAxis < 0 || ncnnAxis >= outDims)
                    throw new InvalidOperationException("ExpandDims axis out of range");

                var tensorAxis = AexisGraphSession.MapNcnnAxisToTensorAxis(outDims, ncnnAxis);
                current = current.ExpandDims(tensorAxis);
            }

            return current;
        }

        private static AexisGraphSession.BufferShape ExpandTextureShape(AexisGraphSession.BufferShape input, int[] expandAxes)
        {
            var dims = input.dims;
            var w = input.w;
            var h = input.h;
            var d = input.d;
            var c = input.c;

            for (var i = 0; i < expandAxes.Length; i++)
            {
                var outDims = dims + 1;
                if (outDims > 4)
                    throw new InvalidOperationException("ExpandDims would exceed dims=4");

                var ncnnAxis = expandAxes[i];
                if (ncnnAxis < 0)
                    ncnnAxis += outDims;
                if (ncnnAxis < 0 || ncnnAxis >= outDims)
                    throw new InvalidOperationException("ExpandDims axis out of range");

                var tensorAxis = AexisGraphSession.MapNcnnAxisToTensorAxis(outDims, ncnnAxis);
                var sizes = new[] { w, h, dims == 4 ? d : c, dims == 4 ? c : 1 };
                var expanded = new[] { 1, 1, 1, 1 };
                for (var axisIndex = 0; axisIndex < outDims; axisIndex++)
                {
                    if (axisIndex < tensorAxis)
                        expanded[axisIndex] = sizes[axisIndex];
                    else if (axisIndex == tensorAxis)
                        expanded[axisIndex] = 1;
                    else
                        expanded[axisIndex] = sizes[axisIndex - 1];
                }

                dims = outDims;
                w = expanded[0];
                h = expanded[1];
                if (dims == 3)
                {
                    d = 1;
                    c = expanded[2];
                }
                else
                {
                    d = expanded[2];
                    c = expanded[3];
                }
            }

            return new AexisGraphSession.BufferShape(dims, w, h, d, c);
        }
    }
}
