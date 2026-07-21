using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisMatMulLayer : AexisBaseLayer
    {
        public AexisMatMulLayer() : base(AexisLayerTypes.MatMul, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        private sealed class VistaTailPack4Plan
        {
            public string featureTextureBlobName;
            public string promptBlobName;
            public AexisGraphSession.BufferShape featureShape;
            public AexisGraphSession.BufferShape outputShape;
            public string promptMemoryLayerName;
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
                                                var aBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var bBuf = owner.GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var aView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                var bView = AexisGraphSession.TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
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

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (TryExecutePack4AttentionMatMulPath(owner, layer, context))
                return;

            if (TryExecuteVistaTailPack4Path(owner, layer, context))
                return;

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (TryExecutePack4AttentionMatMulPath(owner, layer, context))
                return;
            var aShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            var bShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[1]);
            throw new NotSupportedException(
                "CommandBuffer Pack4 MatMul requires Pack4 rank-3/rank-4 textures with compatible K and broadcast batch dimensions"
                + " | layer=" + layer.name
                + " | a=d" + aShape.dims + ":" + aShape.w + "x" + aShape.h + "x" + aShape.d + "x" + aShape.c
                + " | b=d" + bShape.dims + ":" + bShape.w + "x" + bShape.h + "x" + bShape.d + "x" + bShape.c
                + " | transB=" + (layer.GetInt(0, 0) != 0 ? "1" : "0")
                + " | rejectedFallback=placeholder-or-buffer-materialization");
        }

        private static bool TryExecutePack4AttentionMatMulPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length != 2 || layer.topNames == null || layer.topNames.Length != 1)
                return false;
            if (!owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var aTex,
                    out var aShape)
                || !owner.TryGetPack4Texture(
                    layer.bottomNames[1],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var bTex,
                    out var bShape))
            {
                return false;
            }

            if (!CanUsePack4AttentionMatMul(aTex, aShape) || !CanUsePack4AttentionMatMul(bTex, bShape))
                return false;

            var transB = layer.GetInt(0, 0) != 0;
            GetMatrixShape(aShape, out var aRows, out var aCols);
            GetMatrixShape(bShape, out var bRows, out var bCols);
            var k = aCols;
            var kFromB = transB ? bCols : bRows;
            var n = transB ? bRows : bCols;
            if (k <= 0 || aRows <= 0 || n <= 0 || k != kFromB)
                return false;

            var outBatchD = Mathf.Max(aShape.d, bShape.d);
            var outBatchC = Mathf.Max(aShape.c, bShape.c);
            if ((aShape.d != 1 && aShape.d != outBatchD)
                || (bShape.d != 1 && bShape.d != outBatchD)
                || (aShape.c != 1 && aShape.c != outBatchC)
                || (bShape.c != 1 && bShape.c != outBatchC))
            {
                return false;
            }

            var maxDims = Mathf.Max(aShape.dims, bShape.dims);
            AexisGraphSession.BufferShape outShape;
            if (outBatchD == 1 && outBatchC == 1)
            {
                outShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, aRows), 1, 1);
            }
            else if (maxDims >= 4 || outBatchD > 1)
            {
                outShape = new AexisGraphSession.BufferShape(4, Mathf.Max(1, n), Mathf.Max(1, aRows), Mathf.Max(1, outBatchD), Mathf.Max(1, outBatchC));
            }
            else
            {
                outShape = new AexisGraphSession.BufferShape(3, Mathf.Max(1, n), Mathf.Max(1, aRows), 1, Mathf.Max(1, outBatchC));
            }
            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = Mathf.Max(1, outShape.d) * outPacks;
            var outRt = owner.RentTempArray(outShape.w, outShape.h, outSlices, AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.MatMulPack4Cdhw(
                aTex.texture,
                aRows,
                aCols,
                aShape.d,
                aShape.c,
                bTex.texture,
                bRows,
                bCols,
                bShape.d,
                bShape.c,
                transB,
                outShape.d,
                outShape.c,
                outRt);
            AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outRt, outShape, outShape);
            owner.DebugLog?.Invoke(
                "[MatMulPack4CDHW] applied"
                + " | layer=" + layer.name
                + " | transB=" + (transB ? "1" : "0")
                + " | a=d" + aShape.dims + ":" + aShape.w + "x" + aShape.h + "x" + aShape.d + "x" + aShape.c
                + " | b=d" + bShape.dims + ":" + bShape.w + "x" + bShape.h + "x" + bShape.d + "x" + bShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            owner.Consume(
                context.textureBlobs,
                context.bufferBlobs,
                context.bufferRefs,
                context.bufferViews,
                context.remaining,
                layer.bottomNames,
                context.pinnedNames);
            return true;
        }

        private static bool TryExecutePack4AttentionMatMulPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length != 2 || layer.topNames == null || layer.topNames.Length != 1)
                return false;

            var aTex = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var bTex = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[1]);
            var aShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            var bShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[1]);
            if (!CanUsePack4AttentionMatMul(aTex, aShape) || !CanUsePack4AttentionMatMul(bTex, bShape))
                return false;

            var transB = layer.GetInt(0, 0) != 0;
            GetMatrixShape(aShape, out var aRows, out var aCols);
            GetMatrixShape(bShape, out var bRows, out var bCols);
            var k = aCols;
            var kFromB = transB ? bCols : bRows;
            var n = transB ? bRows : bCols;
            if (k <= 0 || aRows <= 0 || n <= 0 || k != kFromB)
                return false;

            var outBatchD = Mathf.Max(aShape.d, bShape.d);
            var outBatchC = Mathf.Max(aShape.c, bShape.c);
            if ((aShape.d != 1 && aShape.d != outBatchD)
                || (bShape.d != 1 && bShape.d != outBatchD)
                || (aShape.c != 1 && aShape.c != outBatchC)
                || (bShape.c != 1 && bShape.c != outBatchC))
            {
                return false;
            }

            var maxDims = Mathf.Max(aShape.dims, bShape.dims);
            AexisGraphSession.BufferShape outShape;
            if (outBatchD == 1 && outBatchC == 1)
            {
                outShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, aRows), 1, 1);
            }
            else if (maxDims >= 4 || outBatchD > 1)
            {
                outShape = new AexisGraphSession.BufferShape(4, Mathf.Max(1, n), Mathf.Max(1, aRows), Mathf.Max(1, outBatchD), Mathf.Max(1, outBatchC));
            }
            else
            {
                outShape = new AexisGraphSession.BufferShape(3, Mathf.Max(1, n), Mathf.Max(1, aRows), 1, Mathf.Max(1, outBatchC));
            }

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            var outSlices = Mathf.Max(1, outShape.d) * outPacks;
            var outRt = owner.RentTempArray(context.commandBuffer, outShape.w, outShape.h, outSlices, AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
            owner.Ops.MatMulPack4Cdhw(
                context.commandBuffer,
                aTex.texture,
                aRows,
                aCols,
                aShape.d,
                aShape.c,
                bTex.texture,
                bRows,
                bCols,
                bShape.d,
                bShape.c,
                transB,
                outShape.d,
                outShape.c,
                outRt);
            context.blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outRt,
                width = outShape.w,
                height = outShape.h,
                packs = outPacks,
                refs = 1,
                owned = true,
                hasLogicalShape = true,
                logicalShape = outShape,
                hasStorageShape = true,
                storageShape = outShape
            };
            context.shapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[MatMulPack4CDHW][cmd] applied"
                + " | layer=" + layer.name
                + " | transB=" + (transB ? "1" : "0")
                + " | a=d" + aShape.dims + ":" + aShape.w + "x" + aShape.h + "x" + aShape.d + "x" + aShape.c
                + " | b=d" + bShape.dims + ":" + bShape.w + "x" + bShape.h + "x" + bShape.d + "x" + bShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c);
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
            return true;
        }

        private static bool TryExecuteVistaTailPack4Path(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (!owner.EnableVistaTailPack4Specializations)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (!TryResolveVistaTailPack4Plan(owner, layer, context, out var plan))
                return false;
            if (context.textureBlobs == null
                || !context.textureBlobs.TryGetValue(plan.featureTextureBlobName, out var featureTex)
                || featureTex == null
                || featureTex.texture == null)
            {
                return false;
            }

            var promptView = AexisGraphSession.TryGetBufferView(plan.promptBlobName, context.bufferBlobs, context.bufferViews);
            var promptBuf = promptView?.buffer;
            var featureShape = plan.featureShape;
            var usedPromptRt = false;
            RenderTexture promptRt = null;
            if (promptBuf == null
                && !string.IsNullOrWhiteSpace(plan.promptMemoryLayerName)
                && owner._memoryData.TryGetValue(plan.promptMemoryLayerName, out var memoryDataPack))
            {
                if (AexisMemoryDataLayer.TryGetOrCreateVistaPromptPack4Rt(memoryDataPack, featureShape.c, out promptRt)
                    && promptRt != null)
                {
                    usedPromptRt = true;
                }
                else if (memoryDataPack?.data != null)
                {
                    promptBuf = memoryDataPack.data;
                    promptView = new AexisTensorBuffer(
                        memoryDataPack.data,
                        memoryDataPack.dims,
                        memoryDataPack.w,
                        memoryDataPack.h,
                        memoryDataPack.d,
                        memoryDataPack.c,
                        false);
                }
            }
            if (!usedPromptRt && promptBuf == null)
            {
                promptBuf = owner.GetOrConvertToBuffer(
                    plan.promptBlobName,
                    context.textureBlobs,
                    context.bufferBlobs,
                    context.textureShapes,
                    context.bufferViews,
                    context.tempOwned);
                promptView = AexisGraphSession.TryGetBufferView(plan.promptBlobName, context.bufferBlobs, context.bufferViews);
            }
            if (!usedPromptRt && (promptBuf == null || promptView == null))
                return false;
            if (!usedPromptRt && (promptView.dims != 1 || promptView.w != featureShape.c))
                return false;
            if (featureShape.dims != 4 || featureShape.c <= 0 || featureTex.packs != Mathf.CeilToInt(featureShape.c / 4f))
                return false;

            var outDepth = Mathf.Max(1, plan.outputShape.d);
            var outPacks = Mathf.Max(1, Mathf.CeilToInt(plan.outputShape.c / 4f));
            var outSlices = outDepth * outPacks;
            var outRt = owner.RentTempArray(plan.outputShape.w, plan.outputShape.h, outSlices, RenderTextureFormat.ARGBFloat);
            if (usedPromptRt && promptRt != null)
            {
                owner.Ops.VistaTailPromptDotPack4(
                    featureTex.texture,
                    plan.outputShape.w,
                    plan.outputShape.h,
                    outDepth,
                    featureTex.packs,
                    promptRt,
                    outRt);
            }
            else
            {
                usedPromptRt = false;
                owner.Ops.VistaTailPromptDotPack4(
                    featureTex.texture,
                    plan.outputShape.w,
                    plan.outputShape.h,
                    outDepth,
                    featureTex.packs,
                    promptBuf,
                    outRt);
            }
            AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outRt, plan.outputShape);
            owner.DebugLog?.Invoke(
                "[VistaTailPack4] specialized path"
                + " | layer=" + layer.name
                + " | feature=" + plan.featureTextureBlobName
                + " | prompt=" + plan.promptBlobName
                + " | prompt_mode=" + (usedPromptRt ? "rt" : "buffer")
                + " | output=" + layer.topNames[0]
                + " | featureShape=d" + featureShape.dims + ":" + featureShape.w + "x" + featureShape.h + "x" + featureShape.d + "x" + featureShape.c
                + " | outputShape=d" + plan.outputShape.dims + ":" + plan.outputShape.w + "x" + plan.outputShape.h + "x" + plan.outputShape.d + "x" + plan.outputShape.c);
            owner.Consume(
                context.textureBlobs,
                context.bufferBlobs,
                context.bufferRefs,
                context.bufferViews,
                context.remaining,
                layer.bottomNames,
                context.pinnedNames);
            return true;
        }

        private static bool TryResolveVistaTailPack4Plan(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisLayerBufferContext context,
            out VistaTailPack4Plan plan)
        {
            plan = null;
            if (owner?.Model?.layers == null || layer == null)
                return false;
            if (!owner.EnableVistaTailPack4Specializations)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length != 2 || layer.topNames == null || layer.topNames.Length != 1)
                return false;
            if (layer.GetInt(0, 0) != 0)
                return false;

            var aName = layer.bottomNames[0];
            var bName = layer.bottomNames[1];
            if (!TryGetBlobShape(context, aName, out var aShape) || !TryGetBlobShape(context, bName, out var bShape))
                return false;
            if (aShape.dims != 1 || bShape.dims != 2)
                return false;
            if (aShape.w <= 0 || bShape.h != aShape.w || bShape.w <= 0)
                return false;

            var reshape = FindSingleProducer(owner.Model, bName);
            if (reshape == null || reshape.type != AexisLayerTypes.Reshape || reshape.bottomNames == null || reshape.bottomNames.Length != 1)
                return false;
            if (!string.Equals(reshape.name, "reshape_124", StringComparison.Ordinal))
                return false;

            var featureSourceBlobName = reshape.bottomNames[0];
            if (!TryGetBlobShape(context, featureSourceBlobName, out var featureShape))
                return false;
            if (featureShape.dims != 4)
                return false;
            if (featureShape.w <= 0 || featureShape.h <= 0 || featureShape.d <= 0 || featureShape.c != aShape.w)
                return false;
            if (bShape.w != featureShape.w * featureShape.h * featureShape.d)
                return false;
            if (bShape.h != featureShape.c)
                return false;

            string promptMemoryLayerName = null;
            var promptProducer = FindSingleProducer(owner.Model, aName);
            if (promptProducer != null && promptProducer.type == AexisLayerTypes.MemoryData)
                promptMemoryLayerName = promptProducer.name;

            plan = new VistaTailPack4Plan
            {
                featureTextureBlobName = bName,
                promptBlobName = aName,
                featureShape = featureShape,
                outputShape = new AexisGraphSession.BufferShape(4, featureShape.w, featureShape.h, featureShape.d, 1),
                promptMemoryLayerName = promptMemoryLayerName
            };
            return true;
        }

        private static bool TryGetBlobShape(
            AexisLayerBufferContext context,
            string blobName,
            out AexisGraphSession.BufferShape shape)
        {
            shape = default;
            if (context == null || string.IsNullOrWhiteSpace(blobName))
                return false;

            if (context.textureShapes != null && context.textureShapes.TryGetValue(blobName, out shape))
                return true;

            if (context.textureBlobs != null
                && context.textureBlobs.TryGetValue(blobName, out var textureRef)
                && textureRef != null
                && textureRef.texture != null
                && textureRef.hasLogicalShape)
            {
                shape = textureRef.logicalShape;
                return true;
            }

            if (context.bufferViews != null
                && context.bufferViews.TryGetValue(blobName, out var view)
                && view != null
                && view.buffer != null)
            {
                shape = new AexisGraphSession.BufferShape(view.dims, view.w, view.h, view.d, view.c);
                return true;
            }

            return false;
        }

        private static AexisGraphModel.Layer FindSingleProducer(AexisGraphModel model, string blobName)
        {
            if (model?.layers == null || string.IsNullOrWhiteSpace(blobName))
                return null;

            AexisGraphModel.Layer producer = null;
            for (var i = 0; i < model.layers.Count; i++)
            {
                var candidate = model.layers[i];
                var tops = candidate?.topNames;
                if (tops == null)
                    continue;

                for (var ti = 0; ti < tops.Length; ti++)
                {
                    if (!string.Equals(tops[ti], blobName, StringComparison.Ordinal))
                        continue;
                    if (producer != null)
                        return null;
                    producer = candidate;
                }
            }

            return producer;
        }

        private static void GetMatrixShape(AexisGraphSession.BufferShape shape, out int rows, out int cols)
        {
            if (shape.dims == 1)
            {
                rows = 1;
                cols = shape.w;
                return;
            }

            if (shape.dims == 2 || shape.dims == 3 || shape.dims == 4)
            {
                rows = shape.h;
                cols = shape.w;
                return;
            }

            throw new InvalidOperationException("MatMul currently supports dims 1/2/3/4 only");
        }

        private static bool CanUsePack4AttentionMatMul(AexisGraphSession.TensorRef src, AexisGraphSession.BufferShape shape)
        {
            return src != null
                && src.texture != null
                && (shape.dims == 3 || shape.dims == 4)
                && shape.w == src.width
                && shape.h == src.height
                && shape.d > 0
                && shape.c > 0
                && (shape.dims != 3 || shape.d == 1)
                && src.packs == Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f));
        }

        private static bool CanUsePack4AttentionMatMul(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape shape)
        {
            return src != null
                && src.texture != null
                && (shape.dims == 3 || shape.dims == 4)
                && shape.w == src.width
                && shape.h == src.height
                && shape.d > 0
                && shape.c > 0
                && (shape.dims != 3 || shape.d == 1)
                && AexisGraphSession.MatchesPack4TextureStorage(src, shape);
        }
    }
}
