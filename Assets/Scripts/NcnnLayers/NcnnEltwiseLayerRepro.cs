using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnEltwiseLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnEltwiseLayerRepro() : base(NcnnLayerTypes.Eltwise, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        private static float[] ParseEltwiseCoeffs(NcnnParamModel.Layer layer)
        {
            return layer?.GetFloats(-23301, Array.Empty<float>()) ?? Array.Empty<float>();
        }

        private static float ResolveEltwiseCoeff(float[] coeffs, int index)
        {
            if (coeffs == null || coeffs.Length == 0)
                return 1f;
            if (index < 0)
                index = 0;
            if (index >= coeffs.Length)
                index = coeffs.Length - 1;
            return coeffs[index];
        }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                                var opType = layer.GetInt(0, 1);
                                                var coeffs = ParseEltwiseCoeffs(layer);
                                                var isTargetSftResidualLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftResidualLayer)
                                                    ? NcnnRepro.CodeFormerSftResidualLayers.Contains(layer.name)
                                                    : string.Equals(layer.name, owner.CodeFormerTargetSftResidualLayer, StringComparison.Ordinal);
                                                var canUseTexture = layer.bottomNames.Length >= 2;
                                                NcnnRepro.TensorRef currentTex = null;
                                                NcnnRepro.BufferShape currentShape = default;
                                                if (canUseTexture)
                                                {
                                                    canUseTexture = owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out currentTex, out currentShape);
                                                }

                                                if (canUseTexture)
                                                {
                                                    for (var i = 1; i < layer.bottomNames.Length; i++)
                                                    {
                                                        if (!owner.TryGetPack4Texture(layer.bottomNames[i], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var otherTex, out var otherShape)
                                                            || currentTex.width != otherTex.width
                                                            || currentTex.height != otherTex.height
                                                            || currentTex.packs != otherTex.packs
                                                            || currentShape.c != otherShape.c)
                                                        {
                                                            canUseTexture = false;
                                                            break;
                                                        }
                                                    }
                                                }

                                                if (canUseTexture)
                                                {
                                                    RenderTexture accum = null;
                                                    try
                                                    {
                                                        var a = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                        accum = owner.RentTempArray(a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                                                        if (opType == 1)
                                                        {
                                                            owner.Ops.ScalePack4(a.texture, ResolveEltwiseCoeff(coeffs, 0), a.packs, accum);
                                                        }
                                                        else
                                                        {
                                                            owner.Ops.CopyPack4(a.texture, 0, accum, 0, a.packs);
                                                        }

                                                        for (var i = 1; i < layer.bottomNames.Length; i++)
                                                        {
                                                            var b = owner.GetOrMaterializeTexture(layer.bottomNames[i], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                            var next = owner.RentTempArray(a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                                                            if (opType == 0)
                                                            {
                                                                owner.Ops.BinaryOpPack4(accum, b.texture, a.packs, 2, next);
                                                            }
                                                            else if (opType == 2)
                                                            {
                                                                owner.Ops.BinaryOpPack4(accum, b.texture, a.packs, 4, next);
                                                            }
                                                            else
                                                            {
                                                                var coeffB = ResolveEltwiseCoeff(coeffs, i);
                                                                if (owner.CodeFormerSftMulScale != 1f && isTargetSftResidualLayer && i == 1)
                                                                    coeffB *= owner.CodeFormerSftMulScale;
                                                                owner.Ops.AddPack4(accum, b.texture, 1f, coeffB, a.packs, next);
                                                            }

                                                            owner.ReturnTempArray(accum);
                                                            accum = next;
                                                        }

                                                        textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
                                                        {
                                                            texture = accum,
                                                            width = a.width,
                                                            height = a.height,
                                                            packs = a.packs,
                                                            refs = 1,
                                                            owned = true
                                                        };
                                                        textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, a.width, a.height, 1, currentShape.c);
                                                        accum = null;
                                                    }
                                                    finally
                                                    {
                                                        if (accum != null)
                                                            owner.ReturnTempArray(accum);
                                                    }
                                                }
                                                else
                                                {
                                                    var firstBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                    var firstView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                    if (firstBuf == null || firstView == null)
                                                        throw new InvalidOperationException("Eltwise source not found: " + layer.name);

                                                    var accumBuf = owner.RentTempBuffer(firstBuf.count, sizeof(float));
                                                    owner.Ops.CopyBuf(firstBuf, accumBuf, firstBuf.count);
                                                    var coeff0 = ResolveEltwiseCoeff(coeffs, 0);
                                                    if (opType == 1 && Mathf.Abs(coeff0 - 1f) > 1e-6f)
                                                        owner.Ops.MulScalarInplace(accumBuf, coeff0, accumBuf.count);

                                                    for (var i = 1; i < layer.bottomNames.Length; i++)
                                                    {
                                                        var nextBuf = owner.GetOrConvertToBuffer(layer.bottomNames[i], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                        if (nextBuf == null)
                                                            throw new InvalidOperationException("Eltwise source not found: " + layer.name + " | " + layer.bottomNames[i]);

                                                        if (opType == 0)
                                                        {
                                                            owner.Ops.BinaryOpBuf(accumBuf, nextBuf, accumBuf.count, 2, accumBuf);
                                                        }
                                                        else if (opType == 2)
                                                        {
                                                            owner.Ops.BinaryOpBuf(accumBuf, nextBuf, accumBuf.count, 4, accumBuf);
                                                        }
                                                        else
                                                        {
                                                            var coeffB = ResolveEltwiseCoeff(coeffs, i);
                                                            if (owner.CodeFormerSftMulScale != 1f && isTargetSftResidualLayer && i == 1)
                                                                coeffB *= owner.CodeFormerSftMulScale;

                                                            if (Mathf.Abs(coeffB - 1f) < 1e-6f)
                                                            {
                                                                owner.Ops.BinaryOpBuf(accumBuf, nextBuf, accumBuf.count, 0, accumBuf);
                                                            }
                                                            else
                                                            {
                                                                var scaled = owner.RentTempBuffer(nextBuf.count, sizeof(float));
                                                                owner.Ops.CopyBuf(nextBuf, scaled, nextBuf.count);
                                                                owner.Ops.MulScalarInplace(scaled, coeffB, scaled.count);
                                                                owner.Ops.BinaryOpBuf(accumBuf, scaled, accumBuf.count, 0, accumBuf);
                                                                tempOwned.Add(scaled);
                                                            }
                                                        }
                                                    }

                                                    bufferBlobs[layer.topNames[0]] = accumBuf;
                                                    bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], accumBuf);
                                                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(accumBuf, firstView.dims, firstView.w, firstView.h, firstView.d, firstView.c, false);
                                                    tempOwned.Add(accumBuf);
                                                }

                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
                        var cmd = context.commandBuffer;
                        var blobs = context.blobs;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;

                        do
                        {
                                                var opType = layer.GetInt(0, 1);
                                                var coeffs = ParseEltwiseCoeffs(layer);
                                                var a = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var accum = owner.RentTempArray(cmd, a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                                                if (opType == 1)
                                                {
                                                    owner.Ops.BinaryOpScalarPack4(cmd, a.texture, ResolveEltwiseCoeff(coeffs, 0), a.packs, 2, accum);
                                                }
                                                else
                                                {
                                                    owner.Ops.CopyPack4(cmd, a.texture, 0, accum, 0, a.packs);
                                                }

                                                for (var i = 1; i < layer.bottomNames.Length; i++)
                                                {
                                                    var b = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[i]);
                                                    var next = owner.RentTempArray(cmd, a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                                                    if (opType == 0)
                                                    {
                                                        owner.Ops.BinaryOpPack4(cmd, accum, b.texture, a.packs, 2, next);
                                                    }
                                                    else if (opType == 2)
                                                    {
                                                        owner.Ops.BinaryOpPack4(cmd, accum, b.texture, a.packs, 4, next);
                                                    }
                                                    else
                                                    {
                                                        owner.Ops.AddPack4(cmd, accum, b.texture, 1f, ResolveEltwiseCoeff(coeffs, i), a.packs, next);
                                                    }

                                                    owner.ReturnTempArray(cmd, accum);
                                                    accum = next;
                                                }

                                                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = accum, width = a.width, height = a.height, packs = a.packs, refs = 1, owned = true };
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
