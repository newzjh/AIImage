using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnEmbedLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnEmbedLayerRepro() : base(NcnnLayerTypes.Embed, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var ep = new NcnnRepro.EmbedPack();
                                        ep.numOutput = layer.GetInt(0, 0);
                                        ep.inputDim = layer.GetInt(1, 0);
                                        ep.biasTerm = layer.GetInt(2, 0);
                                        ep.weightSize = layer.GetInt(3, 0);

                                        phaseSw.Restart();
                                        var w = NcnnRepro.ReadClipArrayAsFloat32(br, ep.weightSize, 0);
                                        float[] b = null;
                                        if (ep.biasTerm != 0)
                                            b = br.ReadNcnnMatAsFloat32(ep.numOutput, 0, 0, 0, 1);
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        phaseSw.Restart();
                                        ep.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                                        ep.w.SetData(w);
                                        if (b != null)
                                        {
                                            ep.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                                            ep.b.SetData(b);
                                        }
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;

                                        owner._embed[layer.name] = ep;
                                        return new NcnnRepro.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            // Fixed token/index buffers are legal inference inputs. Embed writes its activation
            // directly to a LinearMat texture, so it must not select the legacy buffer output path.
            ExecuteRenderTexturePath(owner, layer, context);
        }

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
                                                if (!owner._embed.TryGetValue(layer.name, out var ep) || ep.w == null)
                                                    throw new InvalidOperationException("Embed not found: " + layer.name);
                                                if (!bufferBlobs.TryGetValue(layer.bottomNames[0], out var indicesBuf) || indicesBuf == null)
                                                    throw new InvalidOperationException("Embed input buffer not found: " + layer.bottomNames[0]);

                                                var words = indicesBuf.count;
                                                var outTensor = owner.RentTempTensorBuffer(2, ep.numOutput, words);
                                                owner.Ops.Embed(indicesBuf, words, ep.w, ep.b, ep.numOutput, ep.inputDim, ep.biasTerm != 0, outTensor.buffer);
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
                                                continue;
                        } while (false);
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

            if (!owner._embed.TryGetValue(layer.name, out var ep) || ep.w == null)
                throw new InvalidOperationException("Embed not found: " + layer.name);

            var words = ResolveInputElementCount(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
            var logicalShape = new NcnnRepro.BufferShape(2, Mathf.Max(1, ep.numOutput), words, 1, 1);
            var storageShape = NcnnRepro.ResolveLinearMatStorageShape(logicalShape);
            var output = owner.RentTempMat(storageShape.w, storageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());

            if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var indexBuffer) && indexBuffer != null)
            {
                owner.Ops.EmbedTexture(indexBuffer, words, ep.w, ep.b, ep.numOutput, ep.inputDim, ep.biasTerm != 0, output);
            }
            else if (textureBlobs.TryGetValue(layer.bottomNames[0], out var indexTexture) && indexTexture != null && indexTexture.texture != null)
            {
                var indexShape = NcnnRepro.GetTextureShape(textureShapes, indexTexture, layer.bottomNames[0]);
                var indexStorage = NcnnRepro.GetTextureStorageShape(indexTexture, indexShape);
                var isLinear = NcnnRepro.IsStrictLinearMatTexture(indexTexture);
                if (!isLinear && indexTexture.texture.dimension != TextureDimension.Tex2DArray)
                    throw new InvalidOperationException("Embed texture index input requires linear mat or texture array: " + layer.name);
                if (!isLinear && indexShape.dims > 2)
                    throw new InvalidOperationException("Embed pack4 texture index input only supports dims<=2: " + layer.name);
                owner.Ops.EmbedTexture(indexTexture.texture, isLinear, indexStorage.w, indexStorage.h, words, ep.w, ep.b, ep.numOutput, ep.inputDim, ep.biasTerm != 0, output);
            }
            else
            {
                owner.ReturnTempArray(output);
                throw new InvalidOperationException("Embed input not found for texture path: " + layer.bottomNames[0]);
            }

            NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], output, logicalShape, storageShape);
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._embed.TryGetValue(layer.name, out var ep) || ep.w == null)
                throw new InvalidOperationException("Embed not found: " + layer.name);

            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var words = Mathf.Max(1, srcShape.w * srcShape.h * srcShape.d * srcShape.c);
            var logicalShape = new NcnnRepro.BufferShape(2, Mathf.Max(1, ep.numOutput), words, 1, 1);
            var storageShape = NcnnRepro.ResolveLinearMatStorageShape(logicalShape);
            var output = owner.RentTempMat(cmd, storageShape.w, storageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcStorage = NcnnRepro.GetCmdStorageShape(src, srcShape);
            var isLinear = NcnnRepro.IsStrictLinearMatTexture(src);
            if (!isLinear && src.texture.dimension != TextureDimension.Tex2DArray)
                throw new InvalidOperationException("Embed command-buffer index input requires linear mat or texture array: " + layer.name);
            if (!isLinear && srcShape.dims > 2)
                throw new InvalidOperationException("Embed command-buffer pack4 index input only supports dims<=2: " + layer.name);

            owner.Ops.EmbedTexture(cmd, src.texture, isLinear, srcStorage.w, srcStorage.h, words, ep.w, ep.b, ep.numOutput, ep.inputDim, ep.biasTerm != 0, output);
            blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(output, logicalShape, storageShape, owned: true);
            if (shapes != null)
                shapes[layer.topNames[0]] = logicalShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static int ResolveInputElementCount(
            string name,
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews)
        {
            if (textureBlobs != null
                && textureBlobs.TryGetValue(name, out var texture)
                && texture != null
                && texture.texture != null)
            {
                var shape = NcnnRepro.GetTextureShape(textureShapes, texture, name);
                return Mathf.Max(1, shape.w) * Mathf.Max(1, shape.h) * Mathf.Max(1, shape.d) * Mathf.Max(1, shape.c);
            }

            var view = NcnnRepro.TryGetBufferView(name, bufferBlobs, bufferViews);
            if (view != null)
                return Mathf.Max(1, view.elementCount);

            if (bufferBlobs != null && bufferBlobs.TryGetValue(name, out var buffer) && buffer != null)
                return Mathf.Max(1, buffer.count);

            throw new InvalidOperationException("Embed input shape not found: " + name);
        }
    }
}
