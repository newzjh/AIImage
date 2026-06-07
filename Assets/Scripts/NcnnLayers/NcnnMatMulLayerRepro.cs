using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnMatMulLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnMatMulLayerRepro() : base(NcnnLayerTypes.MatMul, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
                        var textureBlobs = context.textureBlobs;
                        var textureShapes = context.textureShapes;
                        var bufferBlobs = context.bufferBlobs;
                        var bufferRefs = context.bufferRefs;
                        var bufferViews = context.bufferViews;
                        var indexBlobs = context.indexBlobs;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;
                        var tempOwned = context.tempOwned;

                        do
                        {
                                                var aBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var bBuf = owner.GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var aView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                var bView = NcnnRepro.TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
                                                if (aBuf == null || bBuf == null || aView == null || bView == null)
                                                    throw new InvalidOperationException("MatMul source not found: " + layer.name);

                                                var outTensor = owner.RunMatMulLayer(aBuf, aView, bBuf, bView, layer.GetInt(0, 0) != 0);
                                                owner.PublishTensorBufferOutput(
                                                    layer.topNames[0],
                                                    outTensor,
                                                    preferTexture: outTensor.dims <= 3,
                                                    textureBlobs,
                                                    textureShapes,
                                                    bufferBlobs,
                                                    bufferRefs,
                                                    bufferViews,
                                                    tempOwned);
                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
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

            static void GetMatrixShape(NcnnRepro.BufferShape shape, out int rows, out int cols)
            {
                if (shape.dims == 1)
                {
                    rows = 1;
                    cols = shape.w;
                    return;
                }

                if (shape.dims == 2 || shape.dims == 3)
                {
                    rows = shape.h;
                    cols = shape.w;
                    return;
                }

                throw new InvalidOperationException("MatMul currently supports dims 1/2/3 only");
            }

            var aShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var bShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[1]);
            GetMatrixShape(aShape, out var aRows, out var aCols);
            GetMatrixShape(bShape, out var bRows, out var bCols);

            var transB = layer.GetInt(0, 0) != 0;
            var n = transB ? bRows : bCols;
            var batchA = aShape.dims == 3 ? aShape.c : 1;
            var batchB = bShape.dims == 3 ? bShape.c : 1;
            var batch = Mathf.Max(batchA, batchB);
            var outShape = batch > 1
                ? new NcnnRepro.BufferShape(3, Mathf.Max(1, n), Mathf.Max(1, aRows), 1, batch)
                : new NcnnRepro.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, aRows), 1, 1);
            owner.PublishCmdPlaceholder(cmd, layer.topNames[0], outShape, blobs, shapes);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
