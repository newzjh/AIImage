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
                        var textureBlobs = context.textureBlobs;
                        var textureShapes = context.textureShapes;
                        var bufferBlobs = context.bufferBlobs;
                        var bufferRefs = context.bufferRefs;
                        var bufferViews = context.bufferViews;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;
                        var tempOwned = context.tempOwned;

                        do
                        {
                                                var shapeExpr = layer.GetString(6, null);
                                                var bottomShapes = BuildBottomShapes(owner, layer, textureBlobs, textureShapes, bufferBlobs, bufferViews, tempOwned, !string.IsNullOrWhiteSpace(shapeExpr));

                                                if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var reshapeBuf) && reshapeBuf != null)
                                                {
                                                    bufferBlobs[layer.topNames[0]] = reshapeBuf;
                                                    if (bufferRefs.TryGetValue(layer.bottomNames[0], out var reshapeRef) && reshapeRef != null)
                                                    {
                                                        bufferRefs[layer.topNames[0]] = reshapeRef;
                                                        reshapeRef.refs++;
                                                    }
                                                    var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
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

                                                    // If logical channels do not fill whole pack4 lanes, keeping the texture view
                                                    // would preserve padded channels and break later buffer consumers such as Permute.
                                                    var canAliasTexture = CanAliasTextureLayout(srcShape, outShape);

                                                    if (!canAliasTexture)
                                                    {
                                                        var scratchTensor = owner.RentScratchTensorFromTexture(src, srcShape);
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
                                                continue;
                        } while (false);
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
