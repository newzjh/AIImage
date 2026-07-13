using System;
using System.Diagnostics;
using UnityEngine;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnConvolution1DLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnConvolution1DLayerRepro()
            : base(NcnnLayerTypes.Convolution1D, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var pack = new NcnnRepro.ConvPack
            {
                outC = layer.GetInt(0, 0),
                kernelW = layer.GetInt(1, 0),
                kernelH = 1,
                dilationW = layer.GetInt(2, 1),
                dilationH = 1,
                strideW = layer.GetInt(3, 1),
                strideH = 1,
                padLeft = layer.GetInt(4, 0),
                padRight = layer.GetInt(15, layer.GetInt(4, 0)),
                padTop = 0,
                padBottom = 0,
                biasTerm = layer.GetInt(5, 0),
                weightSize = layer.GetInt(6, 0),
                activationType = layer.GetInt(9, 0),
                group = 1,
                useBufferPath = true
            };

            var actParams = NcnnRepro.ParseActivationParams(layer);
            pack.activationSlope = actParams.Length > 0 ? actParams[0] : NcnnRepro.ParseLeakySlope(layer);
            pack.inC = Mathf.Max(1, pack.weightSize / Mathf.Max(1, pack.outC * pack.kernelW));
            pack.inPacks = Mathf.Max(1, (pack.inC + 3) / 4);
            pack.outPacks = Mathf.Max(1, (pack.outC + 3) / 4);

            phaseSw.Restart();
            var w = NcnnRepro.ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
            var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
            phaseSw.Stop();
            readMs += phaseSw.ElapsedMilliseconds;

            phaseSw.Restart();
            pack.rawWeight = NcnnRepro.NewBuffer(w);
            pack.rawBias = NcnnRepro.NewBuffer(b);
            phaseSw.Stop();
            uploadMs += phaseSw.ElapsedMilliseconds;

            if (!owner.ForceBufferConvolutionAll && owner.EnableGeneralTextureConvolution && pack.kernelW > 0)
            {
                phaseSw.Restart();
                var w4 = PackWeightsToO4I4K1D(w, pack.outC, pack.inC, pack.kernelW, pack.outPacks, pack.inPacks);
                var b4 = NcnnRepro.PackBiasToO4(b, pack.outC, pack.outPacks);
                pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                pack.packedWeight4.SetData(w4);
                pack.packedBias4.SetData(b4);
                phaseSw.Stop();
                packMs += phaseSw.ElapsedMilliseconds;
            }

            owner._extraPacks[layer.name] = pack;
            return new NcnnRepro.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.ConvPack conv)
                throw new InvalidOperationException("Convolution1D pack not found: " + layer.name);

            if (!owner.ShouldForceCurrentLayerBufferPath()
                && owner.EnableGeneralTextureConvolution
                && CanUseTexturePath(layer, context, conv))
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.ConvPack conv)
                throw new InvalidOperationException("Convolution1D pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || srcView.dims != 2)
                throw new InvalidOperationException("Convolution1D expects dims=2 tensor input: " + layer.name);

            var inputW = srcView.w;
            var inputC = srcView.h;
            var outW = NcnnRepro.ComputeConvOut(inputW, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
            var outTensor = owner.RentTempTensorBuffer(2, outW, conv.outC);
            var actParams = NcnnRepro.ParseActivationParams(layer);
            var param0 = actParams.Length > 0 ? actParams[0] : conv.activationSlope;
            owner.Ops.Conv1dBuf(
                srcBuf,
                conv.rawWeight,
                conv.rawBias,
                inputW,
                inputC,
                outW,
                conv.outC,
                conv.kernelW,
                conv.strideW,
                conv.dilationW,
                conv.padLeft,
                conv.activationType,
                param0,
                outTensor.buffer);
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
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.ConvPack conv)
                throw new InvalidOperationException("Convolution1D pack not found: " + layer.name);
            if (!textureBlobs.TryGetValue(layer.bottomNames[0], out var src) || src == null || src.texture == null)
                throw new InvalidOperationException("Convolution1D render-texture path requires texture input: " + layer.name);

            var srcShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
            if (!CanUseTexturePath(src, srcShape, conv))
                throw new InvalidOperationException("Convolution1D render-texture path requires dims=2 supported input: " + layer.name);

            var outW = Mathf.Max(1, NcnnRepro.ComputeConvOut(srcShape.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight));
            var outShape = new NcnnRepro.BufferShape(2, outW, Mathf.Max(1, conv.outC), 1, 1);
            var storageShape = NcnnRepro.GetTextureStorageShape(src, srcShape);
            RenderTexture packInput = null;
            RenderTexture packOutput = null;
            RenderTexture scalarOutput = null;
            try
            {
                packInput = owner.RentTempArray(srcShape.w, 1, conv.inPacks, RenderTextureFormat.ARGBHalf);
                if (NcnnRepro.IsStrictLinearMatTexture(src))
                {
                    owner.Ops.ReshapeLinearMatToPack4(
                        src.texture,
                        storageShape.w,
                        storageShape.h,
                        srcShape.w,
                        1,
                        1,
                        conv.inC,
                        3,
                        packInput);
                }
                else
                {
                    owner.Ops.ReshapeScalar2DToPack4(
                        src.texture,
                        storageShape.w,
                        storageShape.h,
                        srcShape.w,
                        1,
                        1,
                        conv.inC,
                        3,
                        packInput);
                }

                packOutput = owner.RentTempArray(outW, 1, conv.outPacks, RenderTextureFormat.ARGBHalf);
                owner.Ops.ConvPack4General(
                    packInput,
                    conv.inPacks,
                    conv.packedWeight4,
                    conv.packedBias4,
                    conv.outPacks,
                    conv.kernelW,
                    1,
                    conv.strideW,
                    1,
                    conv.padLeft,
                    0,
                    conv.dilationW,
                    1,
                    conv.activationType,
                    conv.activationSlope,
                    packOutput);

                scalarOutput = owner.RentTempArray(outShape.w, outShape.h, 1, NcnnRepro.ResolveTensorTextureFormat(outShape.dims));
                owner.Ops.ReshapePack4ToScalar2D(packOutput, outW, 1, 1, conv.outC, 3, scalarOutput);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], scalarOutput, outShape);
                scalarOutput = null;
            }
            finally
            {
                if (packInput != null)
                    owner.ReturnTempArray(packInput);
                if (packOutput != null)
                    owner.ReturnTempArray(packOutput);
                if (scalarOutput != null)
                    owner.ReturnTempArray(scalarOutput);
            }

            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.ConvPack conv)
                throw new InvalidOperationException("Convolution1D pack not found: " + layer.name);

            if (!CanUseTexturePath(src, srcShape, conv))
                throw new InvalidOperationException("Convolution1D command-buffer path requires dims=2 supported input: " + layer.name);

            var outW = Mathf.Max(1, NcnnRepro.ComputeConvOut(srcShape.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight));
            var outShape = new NcnnRepro.BufferShape(2, outW, Mathf.Max(1, conv.outC), 1, 1);
            var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
            ComputeTexture packInput = null;
            ComputeTexture packOutput = null;
            ComputeTexture scalarOutput = null;
            try
            {
                packInput = owner.RentTempArray(cmd, srcShape.w, 1, conv.inPacks, RenderTextureFormat.ARGBHalf);
                if (NcnnRepro.IsStrictLinearMatTexture(src))
                {
                    owner.Ops.ReshapeLinearMatToPack4(
                        cmd,
                        src.texture,
                        storageShape.w,
                        storageShape.h,
                        srcShape.w,
                        1,
                        1,
                        conv.inC,
                        3,
                        packInput);
                }
                else
                {
                    owner.Ops.ReshapeScalar2DToPack4(
                        cmd,
                        src.texture,
                        storageShape.w,
                        storageShape.h,
                        srcShape.w,
                        1,
                        1,
                        conv.inC,
                        3,
                        packInput);
                }

                packOutput = owner.RentTempArray(cmd, outW, 1, conv.outPacks, RenderTextureFormat.ARGBHalf);
                owner.Ops.ConvPack4General(
                    cmd,
                    packInput,
                    conv.inPacks,
                    conv.packedWeight4,
                    conv.packedBias4,
                    conv.outPacks,
                    conv.kernelW,
                    1,
                    conv.strideW,
                    1,
                    conv.padLeft,
                    0,
                    conv.dilationW,
                    1,
                    conv.activationType,
                    conv.activationSlope,
                    packOutput);

                scalarOutput = owner.RentTempArray(cmd, outShape.w, outShape.h, 1, NcnnRepro.ResolveTensorTextureFormat(outShape.dims));
                owner.Ops.ReshapePack4ToScalar2D(cmd, packOutput, outW, 1, 1, conv.outC, 3, scalarOutput);
                blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(scalarOutput, outShape, outShape, owned: true);
                scalarOutput = null;
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
            }
            finally
            {
                if (packInput != null)
                    owner.ReturnTempArray(cmd, packInput);
                if (packOutput != null)
                    owner.ReturnTempArray(cmd, packOutput);
                if (scalarOutput != null)
                    owner.ReturnTempArray(cmd, scalarOutput);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool CanUseTexturePath(NcnnParamModel.Layer layer, NcnnLayerBufferContext context, NcnnRepro.ConvPack conv)
        {
            if (context?.textureBlobs == null || layer?.bottomNames == null || layer.bottomNames.Length == 0)
                return false;
            if (!context.textureBlobs.TryGetValue(layer.bottomNames[0], out var src) || src == null || src.texture == null)
                return false;
            var shape = NcnnRepro.GetTextureShape(context.textureShapes, src, layer.bottomNames[0]);
            return CanUseTexturePath(src, shape, conv);
        }

        private static bool CanUseTexturePath(NcnnRepro.TensorRef src, NcnnRepro.BufferShape srcShape, NcnnRepro.ConvPack conv)
        {
            if (src == null || src.texture == null || conv == null)
                return false;
            if (srcShape.dims != 2 || srcShape.w <= 0 || srcShape.h <= 0)
                return false;
            if (srcShape.h != conv.inC)
                return false;
            if (conv.group != 1 || conv.kernelW <= 0 || conv.packedWeight4 == null || conv.packedBias4 == null)
                return false;

            var storageShape = NcnnRepro.GetTextureStorageShape(src, srcShape);
            if (NcnnRepro.IsStrictLinearMatTexture(src))
                return storageShape.w > 0 && storageShape.h > 0;

            return storageShape.dims == 2
                && storageShape.w == src.width
                && storageShape.h == src.height
                && src.packs == 1;
        }

        private static bool CanUseTexturePath(NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape srcShape, NcnnRepro.ConvPack conv)
        {
            if (src == null || src.texture == null || conv == null)
                return false;
            if (srcShape.dims != 2 || srcShape.w <= 0 || srcShape.h <= 0)
                return false;
            if (srcShape.h != conv.inC)
                return false;
            if (conv.group != 1 || conv.kernelW <= 0 || conv.packedWeight4 == null || conv.packedBias4 == null)
                return false;

            var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
            if (NcnnRepro.IsStrictLinearMatTexture(src))
                return storageShape.w > 0 && storageShape.h > 0;

            return storageShape.dims == 2
                && storageShape.w == src.width
                && storageShape.h == src.height
                && src.packs == 1;
        }

        private static Vector4[] PackWeightsToO4I4K1D(float[] w, int outC, int inC, int kernelW, int outPacks, int inPacks)
        {
            var packed = new Vector4[Mathf.Max(1, outPacks) * Mathf.Max(1, inPacks) * Mathf.Max(1, kernelW) * 4];
            if (w == null || outC <= 0 || inC <= 0 || kernelW <= 0)
                return packed;

            for (var op = 0; op < outPacks; op++)
            {
                for (var ip = 0; ip < inPacks; ip++)
                {
                    for (var kx = 0; kx < kernelW; kx++)
                    {
                        var dstBase = ((op * inPacks + ip) * kernelW + kx) * 4;
                        for (var ocLane = 0; ocLane < 4; ocLane++)
                        {
                            var oc = op * 4 + ocLane;
                            var value = Vector4.zero;
                            for (var icLane = 0; icLane < 4; icLane++)
                            {
                                var ic = ip * 4 + icLane;
                                if (oc < outC && ic < inC)
                                    value[icLane] = w[(oc * inC + ic) * kernelW + kx];
                            }
                            packed[dstBase + ocLane] = value;
                        }
                    }
                }
            }

            return packed;
        }
    }
}
