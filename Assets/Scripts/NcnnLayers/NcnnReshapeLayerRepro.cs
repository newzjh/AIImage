using System;
using System.Collections.Generic;
using UnityEngine;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnReshapeLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnReshapeLayerRepro() : base(NcnnLayerTypes.Reshape, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (context.bufferBlobs.TryGetValue(layer.bottomNames[0], out var existingBuffer) && existingBuffer != null)
            {
#pragma warning disable CS0618
                ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
                return;
            }

            if (CanExecuteRenderTexturePath(owner, layer, context))
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
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

            var shapeExpr = layer.GetString(6, null);
            var bottomShapes = BuildBottomShapes(owner, layer, textureBlobs, textureShapes, bufferBlobs, bufferViews, tempOwned, !string.IsNullOrWhiteSpace(shapeExpr));

            if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var reshapeBuf) && reshapeBuf != null)
            {
                var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                if (srcTensor != null
                    && TryResolveWindowPartitionOutput(owner, layer, srcTensor, out var partitionTensor))
                {
                    owner.PublishTensorBufferOutput(
                        layer.topNames[0],
                        partitionTensor,
                        preferTexture: true,
                        textureBlobs,
                        textureShapes,
                        bufferBlobs,
                        bufferRefs,
                        bufferViews,
                        tempOwned);
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (srcTensor != null
                    && TryResolveWindowUnpartitionOutput(owner, layer, srcTensor, out var unpartitionTensor))
                {
                    owner.PublishTensorBufferOutput(
                        layer.topNames[0],
                        unpartitionTensor,
                        preferTexture: true,
                        textureBlobs,
                        textureShapes,
                        bufferBlobs,
                        bufferRefs,
                        bufferViews,
                        tempOwned);
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (srcTensor != null
                    && TryResolveAttentionContextFlattenOutput(owner, layer, srcTensor, out var attentionFlattenTensor))
                {
                    owner.DebugLog?.Invoke(
                        "TryResolveAttentionContextFlattenOutput applied"
                        + " | layer=" + layer.name
                        + " | src=" + srcTensor.dims + ":" + srcTensor.w + "x" + srcTensor.h + "x" + srcTensor.d + "x" + srcTensor.c
                        + " | dst=" + attentionFlattenTensor.dims + ":" + attentionFlattenTensor.w + "x" + attentionFlattenTensor.h + "x" + attentionFlattenTensor.d + "x" + attentionFlattenTensor.c);
                    owner.PublishTensorBufferOutput(
                        layer.topNames[0],
                        attentionFlattenTensor,
                        preferTexture: true,
                        textureBlobs,
                        textureShapes,
                        bufferBlobs,
                        bufferRefs,
                        bufferViews,
                        tempOwned);
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (srcTensor != null)
                {
                    // Some lowered attention reshapes perform a real data reorder and must publish
                    // the newly produced buffer instead of aliasing the original input buffer.
                    var attentionTensor = TryResolveImplicitAttentionReshape(owner, layer, srcTensor);
                    if (attentionTensor != null)
                    {
                        owner.PublishTensorBufferOutput(
                            layer.topNames[0],
                            attentionTensor,
                            preferTexture: false,
                            textureBlobs,
                            textureShapes,
                            bufferBlobs,
                            bufferRefs,
                            bufferViews,
                            tempOwned);
                        owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                        return;
                    }
                }

                bufferBlobs[layer.topNames[0]] = reshapeBuf;
                if (bufferRefs.TryGetValue(layer.bottomNames[0], out var reshapeRef) && reshapeRef != null)
                {
                    bufferRefs[layer.topNames[0]] = reshapeRef;
                    reshapeRef.refs++;
                }

                if (srcTensor != null)
                {
                    var outView = NcnnRepro.ResolveReshapeTensor(srcTensor, layer, bottomShapes);
                    bufferViews[layer.topNames[0]] = outView;

                    if (textureBlobs.TryGetValue(layer.bottomNames[0], out var reshapeTex) && reshapeTex != null && reshapeTex.texture != null)
                    {
                        var srcShape = NcnnRepro.GetTextureShape(textureShapes, reshapeTex, layer.bottomNames[0]);
                        var outShape = new NcnnRepro.BufferShape(outView.dims, outView.w, outView.h, outView.d, outView.c);
                        var canAliasTexture = CanAliasTextureLayout(srcShape, outShape);
                        if (canAliasTexture)
                        {
                            textureBlobs[layer.topNames[0]] = reshapeTex;
                            textureShapes[layer.topNames[0]] = outShape;
                            reshapeTex.refs++;
                        }
                    }
                }
            }
            else
            {
                var src = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                var srcShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                var outShape = NcnnRepro.ResolveReshapeShape(srcShape, layer, bottomShapes);
                var canAliasTexture = CanAliasTextureLayout(srcShape, outShape);

                if (!canAliasTexture)
                {
                    var scratchTensor = owner.RentScratchTensorFromTexture(src, srcShape);
                    if (TryResolveWindowPartitionOutput(owner, layer, scratchTensor, out var partitionTensor))
                    {
                        owner.PublishTensorBufferOutput(
                            layer.topNames[0],
                            partitionTensor,
                            preferTexture: true,
                            textureBlobs,
                            textureShapes,
                            bufferBlobs,
                            bufferRefs,
                            bufferViews,
                            tempOwned);
                        scratchTensor.Dispose();
                        owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                        return;
                    }

                    if (TryResolveWindowUnpartitionOutput(owner, layer, scratchTensor, out var unpartitionTensor))
                    {
                        owner.PublishTensorBufferOutput(
                            layer.topNames[0],
                            unpartitionTensor,
                            preferTexture: true,
                            textureBlobs,
                            textureShapes,
                            bufferBlobs,
                            bufferRefs,
                            bufferViews,
                            tempOwned);
                        scratchTensor.Dispose();
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (TryResolveAttentionContextFlattenOutput(owner, layer, scratchTensor, out var attentionFlattenTensor))
                {
                    owner.DebugLog?.Invoke(
                        "TryResolveAttentionContextFlattenOutput applied"
                        + " | layer=" + layer.name
                        + " | src=" + scratchTensor.dims + ":" + scratchTensor.w + "x" + scratchTensor.h + "x" + scratchTensor.d + "x" + scratchTensor.c
                        + " | dst=" + attentionFlattenTensor.dims + ":" + attentionFlattenTensor.w + "x" + attentionFlattenTensor.h + "x" + attentionFlattenTensor.d + "x" + attentionFlattenTensor.c);
                    owner.PublishTensorBufferOutput(
                        layer.topNames[0],
                        attentionFlattenTensor,
                        preferTexture: true,
                        textureBlobs,
                        textureShapes,
                        bufferBlobs,
                        bufferRefs,
                        bufferViews,
                        tempOwned);
                    scratchTensor.Dispose();
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                var attentionTensor = TryResolveImplicitAttentionReshape(owner, layer, scratchTensor);
                if (attentionTensor != null)
                {
                        owner.PublishTensorBufferOutput(
                            layer.topNames[0],
                            attentionTensor,
                            preferTexture: false,
                            textureBlobs,
                            textureShapes,
                            bufferBlobs,
                            bufferRefs,
                            bufferViews,
                            tempOwned);
                        scratchTensor.Dispose();
                        owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                        return;
                    }

                    var outView = NcnnRepro.ResolveReshapeTensor(scratchTensor, layer, bottomShapes);
                    var outTensor = new NcnnTensorBuffer(
                        scratchTensor.buffer,
                        outView.dims,
                        outView.w,
                        outView.h,
                        outView.d,
                        outView.c,
                        true,
                        owner.ReturnTempBuffer);
                    owner.PublishTensorBufferOutput(
                        layer.topNames[0],
                        outTensor,
                        preferTexture: outView.dims <= 3,
                        textureBlobs,
                        textureShapes,
                        bufferBlobs,
                        bufferRefs,
                        bufferViews,
                        tempOwned);
                }
                else
                {
                    textureBlobs[layer.topNames[0]] = src;
                    textureShapes[layer.topNames[0]] = outShape;
                    src.refs++;
                }
            }

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
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
            var tempOwned = context.tempOwned;

            var shapeExpr = layer.GetString(6, null);
            var bottomShapes = BuildBottomShapes(owner, layer, textureBlobs, textureShapes, bufferBlobs, bufferViews, tempOwned, !string.IsNullOrWhiteSpace(shapeExpr));
            if (!NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var src, out var srcShape))
                throw new InvalidOperationException("Reshape render-texture path requires existing texture input: " + layer.name);

            var outShape = NcnnRepro.ResolveReshapeShape(srcShape, layer, bottomShapes);

            if (!CanAliasTextureLayout(srcShape, outShape))
                throw new InvalidOperationException("Reshape render-texture path only supports alias-compatible layout: " + layer.name);

            textureBlobs[layer.topNames[0]] = src;
            textureShapes[layer.topNames[0]] = outShape;
            src.refs++;
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
                                                if (TryResolveCmdOutputShape(owner, layer, blobs, shapes, src, out var outShape, out var outW, out var outH, out var outPacks)
                                                    && outW == src.width
                                                    && outH == src.height
                                                    && outPacks == src.packs)
                                                {
                                                    blobs[layer.topNames[0]] = src;
                                                    if (shapes != null)
                                                        shapes[layer.topNames[0]] = outShape;
                                                    src.refs++;
                                                }
                                                else if (TryResolveCmdOutputShape(owner, layer, blobs, shapes, src, out outShape, out outW, out outH, out outPacks))
                                                {
                                                    owner.CopyCmdTensor(cmd, src, layer.topNames[0], blobs, outW, outH, outPacks, shapes, outShape);
                                                }
                                                else
                                                {
                                                    blobs[layer.topNames[0]] = src;
                                                    if (shapes != null)
                                                        shapes[layer.topNames[0]] = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
                                                    src.refs++;
                                                }
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }

        private static bool TryResolveCmdOutputShape(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            System.Collections.Generic.Dictionary<string, NcnnRepro.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> shapes,
            NcnnRepro.CmdTensorRef src,
            out NcnnRepro.BufferShape outShape,
            out int outW,
            out int outH,
            out int outPacks)
        {
            outShape = NcnnRepro.InferCmdShape(src);
            outW = src.width;
            outH = src.height;
            outPacks = src.packs;

            if (src == null || layer == null)
                return false;

            var bottomShapes = BuildCmdBottomShapes(layer, blobs, shapes);
            if (!string.IsNullOrWhiteSpace(layer.GetString(6, null)))
            {
                outShape = NcnnRepro.EvaluateReshapeShapeExpression(layer.GetString(6, null), bottomShapes, layer);
                if (outShape.dims > 3)
                    return false;
                outW = Mathf.Max(1, outShape.w);
                outH = outShape.dims >= 2 ? Mathf.Max(1, outShape.h) : 1;
                outPacks = outShape.dims >= 3 ? Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f)) : 1;
                return true;
            }

            outShape = NcnnRepro.ResolveReshapeShape(bottomShapes[0], layer, bottomShapes);
            if (outShape.dims > 3)
                return false;
            outW = Mathf.Max(1, outShape.w);
            outH = outShape.dims >= 2 ? Mathf.Max(1, outShape.h) : 1;
            outPacks = outShape.dims >= 3 ? Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f)) : 1;
            return true;
        }

        private static System.Collections.Generic.List<NcnnRepro.BufferShape> BuildBottomShapes(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            System.Collections.Generic.Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            System.Collections.Generic.Dictionary<string, ComputeBuffer> bufferBlobs,
            System.Collections.Generic.Dictionary<string, NcnnTensorBuffer> bufferViews,
            System.Collections.Generic.List<IDisposable> tempOwned,
            bool materializeAll)
        {
            var shapes = new System.Collections.Generic.List<NcnnRepro.BufferShape>(layer.bottomNames.Length);
            for (var i = 0; i < layer.bottomNames.Length; i++)
            {
                var name = layer.bottomNames[i];
                if (bufferViews.TryGetValue(name, out var view) && view != null)
                {
                    shapes.Add(new NcnnRepro.BufferShape(view.dims, view.w, view.h, view.d, view.c));
                    continue;
                }

                if (textureBlobs.TryGetValue(name, out var tr) && tr != null && tr.texture != null)
                {
                    shapes.Add(NcnnRepro.GetTextureShape(textureShapes, tr, name));
                    continue;
                }

                if (materializeAll)
                {
                    owner.GetOrConvertToBuffer(name, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    if (bufferViews.TryGetValue(name, out view) && view != null)
                    {
                        shapes.Add(new NcnnRepro.BufferShape(view.dims, view.w, view.h, view.d, view.c));
                        continue;
                    }
                }

            throw new InvalidOperationException("Reshape bottom shape unavailable: " + layer.name + " | " + name);
        }
        return shapes;
    }

        private static bool CanAliasTextureLayout(NcnnRepro.BufferShape srcShape, NcnnRepro.BufferShape outShape)
        {
            if (srcShape.dims > 3 || outShape.dims > 3)
                return false;

            // A 4D tensor flattened into 2D/3D often changes the logical row-major interpretation
            // even if the pack4 texture dimensions happen to match. Keep those cases on the buffer
            // path so downstream matrix-style consumers read the intended linear order.
            if (srcShape.dims != outShape.dims)
                return false;

            var srcCount = srcShape.w * srcShape.h * srcShape.d * srcShape.c;
            var outCount = outShape.w * outShape.h * outShape.d * outShape.c;
            if (srcCount != outCount)
                return false;

            if ((srcShape.dims == 3 && (srcShape.c % 4) != 0) || (outShape.dims == 3 && (outShape.c % 4) != 0))
                return false;

            NcnnRepro.ResolveCmdTextureLayout(srcShape, out var srcW, out var srcH, out var srcPacks);
            NcnnRepro.ResolveCmdTextureLayout(outShape, out var outW, out var outH, out var outPacks);
            return srcW == outW && srcH == outH && srcPacks == outPacks;
        }

        private static bool CanExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;

            var shapeExpr = layer.GetString(6, null);
            var bottomShapes = BuildBottomShapes(owner, layer, context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, context.tempOwned, !string.IsNullOrWhiteSpace(shapeExpr));
            if (!NcnnRepro.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var src, out var srcShape))
                return false;

            if (TryResolveImplicitAttentionReshapeShape(owner, layer, srcShape, out _))
                return false;
            if (TryResolveWindowPartitionPattern(owner, layer, srcShape, out _, out _, out _, out _, out _, out _, out _))
                return false;
            if (TryResolveWindowUnpartitionPattern(owner, layer, srcShape, out _, out _, out _, out _, out _, out _, out _))
                return false;

            var outShape = NcnnRepro.ResolveReshapeShape(srcShape, layer, bottomShapes);
            return CanAliasTextureLayout(srcShape, outShape) && src != null && src.texture != null;
        }

        private static NcnnTensorBuffer TryResolveImplicitAttentionReshape(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnTensorBuffer src)
        {
            if (src == null)
                return null;
            if (!TryResolveImplicitAttentionReshapeShape(owner, layer, new NcnnRepro.BufferShape(src.dims, src.w, src.h, src.d, src.c), out var outShape))
                return null;

            var source = NcnnRepro.ReadFloatBuffer(src.buffer);
            var destination = new float[source.Length];
            var headDim = outShape.w;
            var tokens = outShape.h;
            var windows = outShape.d;
            var qkvHeadChannels = outShape.c;

            for (var window = 0; window < windows; window++)
            {
                for (var token = 0; token < tokens; token++)
                {
                    var srcBase = ((window * src.h) + token) * src.w;
                    for (var qkvHead = 0; qkvHead < qkvHeadChannels; qkvHead++)
                    {
                        var dstBase = (((qkvHead * windows) + window) * tokens + token) * headDim;
                        Array.Copy(
                            source,
                            srcBase + (qkvHead * headDim),
                            destination,
                            dstBase,
                            headDim);
                    }
                }
            }

            var outBuffer = owner.RentTempBuffer(destination.Length, sizeof(float));
            outBuffer.SetData(destination);
            return new NcnnTensorBuffer(
                outBuffer,
                outShape.dims,
                outShape.w,
                outShape.h,
                outShape.d,
                outShape.c,
                true,
                owner.ReturnTempBuffer);
        }

        private static bool TryResolveImplicitAttentionReshapeShape(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnRepro.BufferShape srcShape, out NcnnRepro.BufferShape outShape)
        {
            outShape = default;
            if (owner?.Model?.layers == null || layer == null)
                return false;
            if (!IsParamlessReshape(layer))
                return false;
            if (srcShape.dims != 3 || srcShape.w <= 0 || srcShape.h <= 0 || srcShape.c <= 0)
                return false;
            if (layer.topNames == null || layer.topNames.Length == 0 || string.IsNullOrWhiteSpace(layer.topNames[0]))
                return false;

            var permute = FindSingleConsumer(owner.Model, layer.topNames[0]);
            if (permute == null || permute.type != NcnnLayerTypes.Permute || permute.GetInt(0, -1) != 0)
                return false;
            if (permute.topNames == null || permute.topNames.Length == 0 || string.IsNullOrWhiteSpace(permute.topNames[0]))
                return false;

            var slice = FindSingleConsumer(owner.Model, permute.topNames[0]);
            if (slice == null || slice.type != NcnnLayerTypes.Slice || slice.topNames == null || slice.topNames.Length != 3)
                return false;
            var positiveAxis = slice.GetInt(1, 0);
            if (positiveAxis < 0)
                positiveAxis += srcShape.dims;
            if (positiveAxis != 0)
                return false;

            if (!TryInferAttentionHeadDim(owner.Model, slice, out var headDim))
                return false;
            if (headDim <= 0 || (srcShape.w % headDim) != 0)
                return false;

            var totalChannels = srcShape.w / headDim;
            if (totalChannels <= 0 || (totalChannels % 3) != 0)
                return false;

            outShape = new NcnnRepro.BufferShape(4, headDim, srcShape.h, srcShape.c, totalChannels);
            return true;
        }

        private static bool TryResolveWindowPartitionOutput(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnTensorBuffer src,
            out NcnnTensorBuffer outTensor)
        {
            outTensor = null;
            if (src == null || src.buffer == null)
                return false;
            if (!TryResolveWindowPartitionPattern(
                    owner,
                    layer,
                    new NcnnRepro.BufferShape(src.dims, src.w, src.h, src.d, src.c),
                    out var outShape,
                    out var groupsA,
                    out var groupsB,
                    out var groupsC,
                    out var tokensA,
                    out var tokensB,
                    out var tokensC))
            {
                return false;
            }

            var axisA = src.c;
            var axisB = src.d;
            var axisC = src.h;
            var embedDim = src.w;
            var source = NcnnRepro.ReadFloatBuffer(src.buffer);
            var destination = new float[source.Length];
            var outputGroups = outShape.c;
            var outputTokens = outShape.h;

            for (var groupA = 0; groupA < groupsA; groupA++)
            {
                for (var groupB = 0; groupB < groupsB; groupB++)
                {
                    for (var groupC = 0; groupC < groupsC; groupC++)
                    {
                        var outputGroup = ((groupA * groupsB) + groupB) * groupsC + groupC;
                        for (var tokenA = 0; tokenA < tokensA; tokenA++)
                        {
                            var inputA = groupA * tokensA + tokenA;
                            for (var tokenB = 0; tokenB < tokensB; tokenB++)
                            {
                                var inputB = groupB * tokensB + tokenB;
                                for (var tokenC = 0; tokenC < tokensC; tokenC++)
                                {
                                    var inputC = groupC * tokensC + tokenC;
                                    var outputToken = ((tokenA * tokensB) + tokenB) * tokensC + tokenC;
                                    var sourceBase = (((inputA * axisB) + inputB) * axisC + inputC) * embedDim;
                                    var destinationBase = ((outputGroup * outputTokens) + outputToken) * embedDim;
                                    System.Array.Copy(source, sourceBase, destination, destinationBase, embedDim);
                                }
                            }
                        }
                    }
                }
            }

            var outBuffer = owner.RentTempBuffer(destination.Length, sizeof(float));
            outBuffer.SetData(destination);
            outTensor = new NcnnTensorBuffer(outBuffer, outShape.dims, outShape.w, outShape.h, outShape.d, outShape.c, true, owner.ReturnTempBuffer);
            return true;
        }

        private static bool TryResolveWindowPartitionPattern(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnRepro.BufferShape srcShape,
            out NcnnRepro.BufferShape outShape,
            out int groupsA,
            out int groupsB,
            out int groupsC,
            out int tokensA,
            out int tokensB,
            out int tokensC)
        {
            outShape = default;
            groupsA = 0;
            groupsB = 0;
            groupsC = 0;
            tokensA = 0;
            tokensB = 0;
            tokensC = 0;

            if (owner?.Model?.layers == null || layer == null)
                return false;
            if (srcShape.dims != 4 || srcShape.w <= 0 || srcShape.h <= 0 || srcShape.d <= 0 || srcShape.c <= 0)
                return false;
            if (srcShape.h != srcShape.d || srcShape.d != srcShape.c)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length == 0 || string.IsNullOrWhiteSpace(layer.bottomNames[0]))
                return false;

            outShape = NcnnRepro.ResolveReshapeShape(srcShape, layer);
            if (outShape.dims != 3 || outShape.w != srcShape.w || outShape.h <= 0 || outShape.c <= 0)
                return false;
            if ((srcShape.c * srcShape.d * srcShape.h) != (outShape.c * outShape.h))
                return false;

            if (!TryMatchWindowPartitionProducerChain(owner.Model, layer))
                return false;
            if (!TryResolvePerfectCube(outShape.c, out var groupsEdge))
                return false;
            if (!TryResolvePerfectCube(outShape.h, out var tokensEdge))
                return false;
            if ((groupsEdge * tokensEdge) != srcShape.h)
                return false;

            groupsA = groupsB = groupsC = groupsEdge;
            tokensA = tokensB = tokensC = tokensEdge;
            return true;
        }

        private static bool TryResolveWindowUnpartitionOutput(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnTensorBuffer src,
            out NcnnTensorBuffer outTensor)
        {
            outTensor = null;
            if (src == null || src.buffer == null)
                return false;
            if (!TryResolveWindowUnpartitionPattern(
                    owner,
                    layer,
                    new NcnnRepro.BufferShape(src.dims, src.w, src.h, src.d, src.c),
                    out var outShape,
                    out var groupsA,
                    out var groupsB,
                    out var groupsC,
                    out var tokensA,
                    out var tokensB,
                    out var tokensC))
            {
                return false;
            }

            var axisA = outShape.c;
            var axisB = outShape.d;
            var axisC = outShape.h;
            var embedDim = src.w;
            var source = NcnnRepro.ReadFloatBuffer(src.buffer);
            var destination = new float[source.Length];
            var inputTokens = src.h;

            for (var groupA = 0; groupA < groupsA; groupA++)
            {
                for (var groupB = 0; groupB < groupsB; groupB++)
                {
                    for (var groupC = 0; groupC < groupsC; groupC++)
                    {
                        var inputGroup = ((groupA * groupsB) + groupB) * groupsC + groupC;
                        for (var tokenA = 0; tokenA < tokensA; tokenA++)
                        {
                            var outputA = groupA * tokensA + tokenA;
                            for (var tokenB = 0; tokenB < tokensB; tokenB++)
                            {
                                var outputB = groupB * tokensB + tokenB;
                                for (var tokenC = 0; tokenC < tokensC; tokenC++)
                                {
                                    var outputC = groupC * tokensC + tokenC;
                                    var inputToken = ((tokenA * tokensB) + tokenB) * tokensC + tokenC;
                                    var sourceBase = ((inputGroup * inputTokens) + inputToken) * embedDim;
                                    var destinationBase = (((outputA * axisB) + outputB) * axisC + outputC) * embedDim;
                                    System.Array.Copy(source, sourceBase, destination, destinationBase, embedDim);
                                }
                            }
                        }
                    }
                }
            }

            var outBuffer = owner.RentTempBuffer(destination.Length, sizeof(float));
            outBuffer.SetData(destination);
            outTensor = new NcnnTensorBuffer(outBuffer, outShape.dims, outShape.w, outShape.h, outShape.d, outShape.c, true, owner.ReturnTempBuffer);
            return true;
        }

        private static bool TryResolveWindowUnpartitionPattern(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnRepro.BufferShape srcShape,
            out NcnnRepro.BufferShape outShape,
            out int groupsA,
            out int groupsB,
            out int groupsC,
            out int tokensA,
            out int tokensB,
            out int tokensC)
        {
            outShape = default;
            groupsA = 0;
            groupsB = 0;
            groupsC = 0;
            tokensA = 0;
            tokensB = 0;
            tokensC = 0;

            if (owner?.Model?.layers == null || layer == null)
                return false;
            if (srcShape.dims != 3 || srcShape.w <= 0 || srcShape.h <= 0 || srcShape.c <= 0)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length == 0 || string.IsNullOrWhiteSpace(layer.bottomNames[0]))
                return false;

            outShape = NcnnRepro.ResolveReshapeShape(srcShape, layer);
            if (outShape.dims != 4 || outShape.w != srcShape.w || outShape.h <= 0 || outShape.d <= 0 || outShape.c <= 0)
                return false;
            if (outShape.h != outShape.d || outShape.d != outShape.c)
                return false;
            if ((srcShape.c * srcShape.h) != (outShape.c * outShape.d * outShape.h))
                return false;
            if (!TryMatchWindowUnpartitionProducerChain(owner.Model, layer))
                return false;
            if (!TryResolvePerfectCube(srcShape.c, out var groupsEdge))
                return false;
            if (!TryResolvePerfectCube(srcShape.h, out var tokensEdge))
                return false;
            if (outShape.h != (groupsEdge * tokensEdge))
                return false;

            groupsA = groupsB = groupsC = groupsEdge;
            tokensA = tokensB = tokensC = tokensEdge;
            return true;
        }

        private static bool TryResolveAttentionContextFlattenOutput(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnTensorBuffer src,
            out NcnnTensorBuffer outTensor)
        {
            outTensor = null;
            if (src == null || src.buffer == null)
                return false;
            if (!TryResolveAttentionContextFlattenShape(owner, layer, new NcnnRepro.BufferShape(src.dims, src.w, src.h, src.d, src.c), out var outShape))
                return false;

            var source = NcnnRepro.ReadFloatBuffer(src.buffer);
            var destination = new float[source.Length];
            var headDim = src.w;
            var tokens = src.h;
            var windows = src.d;
            var heads = src.c;

            for (var window = 0; window < windows; window++)
            {
                for (var token = 0; token < tokens; token++)
                {
                    var dstBase = ((window * tokens) + token) * outShape.w;
                    for (var dim = 0; dim < headDim; dim++)
                    {
                        for (var head = 0; head < heads; head++)
                        {
                            var srcIndex = ((((head * windows) + window) * tokens) + token) * headDim + dim;
                            var dstIndex = dstBase + dim * heads + head;
                            destination[dstIndex] = source[srcIndex];
                        }
                    }
                }
            }

            var outBuffer = owner.RentTempBuffer(destination.Length, sizeof(float));
            outBuffer.SetData(destination);
            outTensor = new NcnnTensorBuffer(
                outBuffer,
                outShape.dims,
                outShape.w,
                outShape.h,
                outShape.d,
                outShape.c,
                true,
                owner.ReturnTempBuffer);
            return true;
        }

        private static bool TryResolveAttentionContextFlattenShape(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnRepro.BufferShape srcShape,
            out NcnnRepro.BufferShape outShape)
        {
            outShape = default;
            if (owner?.Model?.layers == null || layer == null)
                return false;
            if (srcShape.dims != 4 || srcShape.w <= 0 || srcShape.h <= 0 || srcShape.d <= 0 || srcShape.c <= 0)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length == 0 || string.IsNullOrWhiteSpace(layer.bottomNames[0]))
                return false;

            var producer = FindSingleProducer(owner.Model, layer.bottomNames[0]);
            if (producer == null || producer.type != NcnnLayerTypes.MatMul)
                return false;

            outShape = NcnnRepro.ResolveReshapeShape(srcShape, layer);
            if (outShape.dims != 2 && outShape.dims != 3)
                return false;
            if (outShape.w != srcShape.w * srcShape.c)
                return false;

            var collapsesWindowsIntoTokens =
                outShape.h == srcShape.h * srcShape.d
                && (outShape.dims != 3 || outShape.c == 1);
            if (collapsesWindowsIntoTokens)
                return true;

            var preservesWindowAxis =
                outShape.dims == 3
                && outShape.h == srcShape.h
                && outShape.c == srcShape.d;
            return preservesWindowAxis;
        }

        private static bool TryMatchWindowPartitionProducerChain(NcnnParamModel model, NcnnParamModel.Layer layer)
        {
            if (model?.layers == null || layer?.bottomNames == null || layer.bottomNames.Length == 0)
                return false;

            var permute0 = FindSingleProducer(model, layer.bottomNames[0]);
            if (permute0 == null || permute0.type != NcnnLayerTypes.Permute || permute0.GetInt(0, -1) != 0)
                return false;
            if (permute0.bottomNames == null || permute0.bottomNames.Length == 0 || string.IsNullOrWhiteSpace(permute0.bottomNames[0]))
                return false;

            var reshapeMarker = FindSingleProducer(model, permute0.bottomNames[0]);
            if (reshapeMarker == null || reshapeMarker.type != NcnnLayerTypes.Reshape || !IsParamlessReshape(reshapeMarker))
                return false;
            if (reshapeMarker.bottomNames == null || reshapeMarker.bottomNames.Length == 0 || string.IsNullOrWhiteSpace(reshapeMarker.bottomNames[0]))
                return false;

            var permute9 = FindSingleProducer(model, reshapeMarker.bottomNames[0]);
            return permute9 != null && permute9.type == NcnnLayerTypes.Permute && permute9.GetInt(0, -1) == 9;
        }

        private static bool TryMatchWindowUnpartitionProducerChain(NcnnParamModel model, NcnnParamModel.Layer layer)
        {
            if (model?.layers == null || layer?.bottomNames == null || layer.bottomNames.Length == 0)
                return false;

            var permute0 = FindSingleProducer(model, layer.bottomNames[0]);
            if (permute0 == null || permute0.type != NcnnLayerTypes.Permute || permute0.GetInt(0, -1) != 0)
                return false;
            if (permute0.bottomNames == null || permute0.bottomNames.Length == 0 || string.IsNullOrWhiteSpace(permute0.bottomNames[0]))
                return false;

            var reshapeMarker = FindSingleProducer(model, permute0.bottomNames[0]);
            return reshapeMarker != null
                && reshapeMarker.type == NcnnLayerTypes.Reshape
                && IsParamlessReshape(reshapeMarker);
        }

        private static bool TryResolvePerfectCube(int value, out int edge)
        {
            edge = 0;
            if (value <= 0)
                return false;

            var candidate = Mathf.RoundToInt(Mathf.Pow(value, 1f / 3f));
            for (var delta = -1; delta <= 1; delta++)
            {
                var test = candidate + delta;
                if (test <= 0)
                    continue;
                if ((test * test * test) != value)
                    continue;
                edge = test;
                return true;
            }

            return false;
        }

        private static bool TryInferAttentionHeadDim(NcnnParamModel model, NcnnParamModel.Layer slice, out int headDim)
        {
            headDim = 0;
            if (model?.layers == null || slice?.topNames == null)
                return false;

            for (var i = 0; i < slice.topNames.Length; i++)
            {
                var branchTop = slice.topNames[i];
                if (string.IsNullOrWhiteSpace(branchTop))
                    continue;

                var reshape = FindSingleConsumer(model, branchTop);
                if (reshape == null || reshape.type != NcnnLayerTypes.Reshape || !IsParamlessReshape(reshape))
                    continue;
                if (reshape.topNames == null || reshape.topNames.Length == 0 || string.IsNullOrWhiteSpace(reshape.topNames[0]))
                    continue;

                var matmul = FindSingleConsumer(model, reshape.topNames[0]);
                if (matmul == null || matmul.type != NcnnLayerTypes.MatMul)
                    continue;
                if (matmul.topNames == null || matmul.topNames.Length == 0 || string.IsNullOrWhiteSpace(matmul.topNames[0]))
                    continue;

                var mul = FindSingleConsumer(model, matmul.topNames[0]);
                if (mul == null || mul.type != NcnnLayerTypes.BinaryOp || mul.GetInt(0, -1) != 2)
                    continue;

                var scale = mul.GetFloat(2, 0f);
                if (!(scale > 0f))
                    continue;

                var inferred = Mathf.RoundToInt(1f / (scale * scale));
                if (inferred <= 0)
                    continue;

                var reconstructed = 1f / Mathf.Sqrt(inferred);
                if (Mathf.Abs(reconstructed - scale) > 1e-3f)
                    continue;

                headDim = inferred;
                return true;
            }

            return false;
        }

        private static bool IsParamlessReshape(NcnnParamModel.Layer layer)
        {
            if (layer == null)
                return false;
            if (!string.IsNullOrWhiteSpace(layer.GetString(6, null)))
                return false;
            return layer.GetInt(0, -233) == -233
                && layer.GetInt(1, -233) == -233
                && layer.GetInt(11, -233) == -233
                && layer.GetInt(2, -233) == -233;
        }

        private static NcnnParamModel.Layer FindSingleConsumer(NcnnParamModel model, string blobName)
        {
            if (model?.layers == null || string.IsNullOrWhiteSpace(blobName))
                return null;

            NcnnParamModel.Layer found = null;
            for (var i = 0; i < model.layers.Count; i++)
            {
                var candidate = model.layers[i];
                if (candidate?.bottomNames == null)
                    continue;
                for (var j = 0; j < candidate.bottomNames.Length; j++)
                {
                    if (!string.Equals(candidate.bottomNames[j], blobName, StringComparison.Ordinal))
                        continue;
                    if (found != null)
                        return null;
                    found = candidate;
                    break;
                }
            }

            return found;
        }

        private static NcnnParamModel.Layer FindSingleProducer(NcnnParamModel model, string blobName)
        {
            if (model?.layers == null || string.IsNullOrWhiteSpace(blobName))
                return null;

            for (var i = 0; i < model.layers.Count; i++)
            {
                var candidate = model.layers[i];
                if (candidate?.topNames == null)
                    continue;
                for (var j = 0; j < candidate.topNames.Length; j++)
                {
                    if (string.Equals(candidate.topNames[j], blobName, StringComparison.Ordinal))
                        return candidate;
                }
            }

            return null;
        }

        private static System.Collections.Generic.List<NcnnRepro.BufferShape> BuildCmdBottomShapes(
            NcnnParamModel.Layer layer,
            System.Collections.Generic.Dictionary<string, NcnnRepro.CmdTensorRef> blobs,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> cmdShapes)
        {
            var bottomShapes = new System.Collections.Generic.List<NcnnRepro.BufferShape>(layer.bottomNames.Length);
            for (var i = 0; i < layer.bottomNames.Length; i++)
            {
                bottomShapes.Add(NcnnRepro.GetCmdShape(cmdShapes, blobs, layer.bottomNames[i]));
            }
            return bottomShapes;
        }
    }
}
