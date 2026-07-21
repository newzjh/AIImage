using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisEltwiseLayer : AexisBaseLayer
    {
        public AexisEltwiseLayer() : base(AexisLayerTypes.Eltwise, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        private static float[] ParseEltwiseCoeffs(AexisGraphModel.Layer layer)
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

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
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
        public override void ExecuteComputeBufferPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            var opType = layer.GetInt(0, 1);
            var coeffs = ParseEltwiseCoeffs(layer);
            var isTargetSftResidualLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftResidualLayer)
                ? AexisGraphSession.CodeFormerSftResidualLayers.Contains(layer.name)
                : string.Equals(layer.name, owner.CodeFormerTargetSftResidualLayer, StringComparison.Ordinal);

            var firstBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var firstView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
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
            bufferViews[layer.topNames[0]] = new AexisTensorBuffer(accumBuf, firstView.dims, firstView.w, firstView.h, firstView.d, firstView.c, false);
            tempOwned.Add(accumBuf);
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
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

            var opType = layer.GetInt(0, 1);
            var coeffs = ParseEltwiseCoeffs(layer);
            var isTargetSftResidualLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftResidualLayer)
                ? AexisGraphSession.CodeFormerSftResidualLayers.Contains(layer.name)
                : string.Equals(layer.name, owner.CodeFormerTargetSftResidualLayer, StringComparison.Ordinal);

            var a = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
            if (!AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out _, out var currentShape))
                throw new InvalidOperationException("Eltwise render-texture path requires existing texture shape: " + layer.name);

            RenderTexture accum = null;
            try
            {
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

                textureBlobs[layer.topNames[0]] = new AexisGraphSession.TensorRef
                {
                    texture = accum,
                    width = a.width,
                    height = a.height,
                    packs = a.packs,
                    refs = 1,
                    owned = true
                };
                textureShapes[layer.topNames[0]] = new AexisGraphSession.BufferShape(3, a.width, a.height, 1, currentShape.c);
                accum = null;
            }
            finally
            {
                if (accum != null)
                    owner.ReturnTempArray(accum);
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

                        do
                        {
                                                var opType = layer.GetInt(0, 1);
                                                var coeffs = ParseEltwiseCoeffs(layer);
                                                var a = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var aShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
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
                                                    var b = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[i]);
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

                                                blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef { texture = accum, width = a.width, height = a.height, packs = a.packs, refs = 1, owned = true };
                                                if (shapes != null)
                                                    shapes[layer.topNames[0]] = aShape;
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }

        private static bool CanExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (layer.bottomNames == null || layer.bottomNames.Length < 2)
                return false;

            AexisGraphSession.TensorRef currentTex = null;
            AexisGraphSession.BufferShape currentShape = default;
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out currentTex, out currentShape))
                return false;

            for (var i = 1; i < layer.bottomNames.Length; i++)
            {
                if (!owner.TryGetPack4Texture(layer.bottomNames[i], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out var otherTex, out var otherShape)
                    || currentTex.width != otherTex.width
                    || currentTex.height != otherTex.height
                    || currentTex.packs != otherTex.packs
                    || currentShape.c != otherShape.c
                    || !AexisGraphSession.MatchesPack4TextureStorage(currentTex, currentShape)
                    || !AexisGraphSession.MatchesPack4TextureStorage(otherTex, otherShape))
                {
                    return false;
                }
            }

            return true;
        }
    }
}
