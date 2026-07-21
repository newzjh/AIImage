using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisEmbedLayer : AexisBaseLayer
    {
        public AexisEmbedLayer() : base(AexisLayerTypes.Embed, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var ep = new AexisGraphSession.EmbedPack();
                                        ep.numOutput = layer.GetInt(0, 0);
                                        ep.inputDim = layer.GetInt(1, 0);
                                        ep.biasTerm = layer.GetInt(2, 0);
                                        ep.weightSize = layer.GetInt(3, 0);

                                        phaseSw.Restart();
                                        var useSharedWeights = owner.SharedTokenEmbeddingElementCount == ep.weightSize
                                            && (owner.SharedTokenEmbeddingWeights != null || owner.SharedTokenEmbeddingWeightsInt8Packed != null);
                                        float[] w = null;
                                        if (useSharedWeights)
                                            br.SkipTensor(ep.weightSize, 0, 0, 0, 0);
                                        else
                                            w = AexisGraphSession.ReadClipArrayAsFloat32(br, ep.weightSize, 0);
                                        float[] b = null;
                                        if (ep.biasTerm != 0)
                                            b = br.ReadTensorAsFloat32(ep.numOutput, 0, 0, 0, 1);
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        phaseSw.Restart();
                                        if (useSharedWeights)
                                        {
                                            ep.w = owner.SharedTokenEmbeddingWeights;
                                            ep.wInt8Packed = owner.SharedTokenEmbeddingWeightsInt8Packed;
                                            ep.wInt8Scales = owner.SharedTokenEmbeddingWeightsInt8Scales;
                                            ep.ownsW = false;
                                            ep.ownsWInt8 = false;
                                        }
                                        else
                                        {
                                            ep.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                                            ep.w.SetData(w);
                                        }
                                        if (b != null)
                                        {
                                            ep.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                                            ep.b.SetData(b);
                                        }
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;

                                        owner._embed[layer.name] = ep;
                                        return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            // Fixed token/index buffers are legal inference inputs. Embed writes its activation
            // directly to a LinearMat texture, so it must not select the legacy buffer output path.
            ExecuteRenderTexturePath(owner, layer, context);
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
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
                                                if (!owner._embed.TryGetValue(layer.name, out var ep) || ep.WeightBinding == null)
                                                    throw new InvalidOperationException("Embed not found: " + layer.name);
                                                if (!bufferBlobs.TryGetValue(layer.bottomNames[0], out var indicesBuf) || indicesBuf == null)
                                                    throw new InvalidOperationException("Embed input buffer not found: " + layer.bottomNames[0]);

                                                var words = indicesBuf.count;
                                                var outTensor = owner.RentTempTensorBuffer(2, ep.numOutput, words);
                                                owner.Ops.SetInt8EmbedWeights(ep.wInt8Packed, ep.wInt8Scales);
                                                owner.Ops.Embed(indicesBuf, words, ep.WeightBinding, ep.b, ep.numOutput, ep.inputDim, ep.biasTerm != 0, outTensor.buffer);
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

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._embed.TryGetValue(layer.name, out var ep) || ep.WeightBinding == null)
                throw new InvalidOperationException("Embed not found: " + layer.name);
            owner.Ops.SetInt8EmbedWeights(ep.wInt8Packed, ep.wInt8Scales);

            var words = ResolveInputElementCount(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
            var logicalShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, ep.numOutput), words, 1, 1);
            var storageShape = AexisGraphSession.ResolveLinearMatStorageShape(logicalShape);
            var output = owner.RentTempMat(storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());

            if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var indexBuffer) && indexBuffer != null)
            {
                owner.Ops.EmbedTexture(indexBuffer, words, ep.WeightBinding, ep.b, ep.numOutput, ep.inputDim, ep.biasTerm != 0, output);
            }
            else if (textureBlobs.TryGetValue(layer.bottomNames[0], out var indexTexture) && indexTexture != null && indexTexture.texture != null)
            {
                var indexShape = AexisGraphSession.GetTextureShape(textureShapes, indexTexture, layer.bottomNames[0]);
                var indexStorage = AexisGraphSession.GetTextureStorageShape(indexTexture, indexShape);
                var isLinear = AexisGraphSession.IsStrictLinearMatTexture(indexTexture);
                if (!isLinear && indexTexture.texture.dimension != TextureDimension.Tex2DArray)
                    throw new InvalidOperationException("Embed texture index input requires linear mat or texture array: " + layer.name);
                if (!isLinear && indexShape.dims > 2)
                    throw new InvalidOperationException("Embed pack4 texture index input only supports dims<=2: " + layer.name);
                owner.Ops.EmbedTexture(indexTexture.texture, isLinear, indexStorage.w, indexStorage.h, words, ep.WeightBinding, ep.b, ep.numOutput, ep.inputDim, ep.biasTerm != 0, output);
            }
            else
            {
                owner.ReturnTempArray(output);
                throw new InvalidOperationException("Embed input not found for texture path: " + layer.bottomNames[0]);
            }

            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], output, logicalShape, storageShape);
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._embed.TryGetValue(layer.name, out var ep) || ep.WeightBinding == null)
                throw new InvalidOperationException("Embed not found: " + layer.name);
            owner.Ops.SetInt8EmbedWeights(ep.wInt8Packed, ep.wInt8Scales);

            var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var words = Mathf.Max(1, srcShape.w * srcShape.h * srcShape.d * srcShape.c);
            var logicalShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, ep.numOutput), words, 1, 1);
            var storageShape = AexisGraphSession.ResolveLinearMatStorageShape(logicalShape);
            var output = owner.RentTempMat(cmd, storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcStorage = AexisGraphSession.GetCmdStorageShape(src, srcShape);
            var isLinear = AexisGraphSession.IsStrictLinearMatTexture(src);
            if (!isLinear && src.texture.dimension != TextureDimension.Tex2DArray)
                throw new InvalidOperationException("Embed command-buffer index input requires linear mat or texture array: " + layer.name);
            if (!isLinear && srcShape.dims > 2)
                throw new InvalidOperationException("Embed command-buffer pack4 index input only supports dims<=2: " + layer.name);

            owner.Ops.EmbedTexture(cmd, src.texture, isLinear, srcStorage.w, srcStorage.h, words, ep.WeightBinding, ep.b, ep.numOutput, ep.inputDim, ep.biasTerm != 0, output);
            blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, logicalShape, storageShape, owned: true);
            if (shapes != null)
                shapes[layer.topNames[0]] = logicalShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static int ResolveInputElementCount(
            string name,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, AexisTensorBuffer> bufferViews)
        {
            if (textureBlobs != null
                && textureBlobs.TryGetValue(name, out var texture)
                && texture != null
                && texture.texture != null)
            {
                var shape = AexisGraphSession.GetTextureShape(textureShapes, texture, name);
                return Mathf.Max(1, shape.w) * Mathf.Max(1, shape.h) * Mathf.Max(1, shape.d) * Mathf.Max(1, shape.c);
            }

            var view = AexisGraphSession.TryGetBufferView(name, bufferBlobs, bufferViews);
            if (view != null)
                return Mathf.Max(1, view.elementCount);

            if (bufferBlobs != null && bufferBlobs.TryGetValue(name, out var buffer) && buffer != null)
                return Mathf.Max(1, buffer.count);

            throw new InvalidOperationException("Embed input shape not found: " + name);
        }
    }
}
