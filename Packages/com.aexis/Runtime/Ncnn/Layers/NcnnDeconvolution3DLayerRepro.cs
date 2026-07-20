using System;
using System.Diagnostics;
using UnityEngine;

namespace Aexis.Ncnn
{
    public sealed class NcnnDeconvolution3DLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnDeconvolution3DLayerRepro()
            : base(NcnnLayerTypes.Deconvolution3D, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (!owner.ShouldForceCurrentLayerBufferPath()
                && owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out _,
                    out var srcShape)
                && srcShape.dims == 4)
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override NcnnGraphSession.LayerLoadMetrics LoadLayer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var pack = new NcnnGraphSession.DeconvPack
            {
                outC = layer.GetInt(0, 0),
                group = 1,
                kernelW = layer.GetInt(1, 0),
                kernelH = layer.GetInt(11, layer.GetInt(1, 0)),
                kernelD = layer.GetInt(21, layer.GetInt(1, 0)),
                dilationW = layer.GetInt(2, 1),
                dilationH = layer.GetInt(12, layer.GetInt(2, 1)),
                dilationD = layer.GetInt(22, layer.GetInt(2, 1)),
                strideW = layer.GetInt(3, 1),
                strideH = layer.GetInt(13, layer.GetInt(3, 1)),
                strideD = layer.GetInt(23, layer.GetInt(3, 1)),
                padLeft = layer.GetInt(4, 0),
                padRight = layer.GetInt(15, layer.GetInt(4, 0)),
                padTop = layer.GetInt(14, layer.GetInt(4, 0)),
                padBottom = layer.GetInt(16, layer.GetInt(14, layer.GetInt(4, 0))),
                padFront = layer.GetInt(24, layer.GetInt(4, 0)),
                padBehind = layer.GetInt(17, layer.GetInt(24, layer.GetInt(4, 0))),
                outputPadRight = layer.GetInt(18, 0),
                outputPadBottom = layer.GetInt(19, layer.GetInt(18, 0)),
                outputPadBehind = layer.GetInt(20, layer.GetInt(18, 0)),
                biasTerm = layer.GetInt(5, 0),
                weightSize = layer.GetInt(6, 0),
                activationType = layer.GetInt(9, 0),
            };

            var actParams = NcnnGraphSession.ParseActivationParams(layer);
            pack.activationSlope = actParams.Length > 0 ? actParams[0] : NcnnGraphSession.ParseLeakySlope(layer);

            var kernelVolume = Mathf.Max(1, pack.kernelW * pack.kernelH * pack.kernelD);
            pack.inC = Mathf.Max(1, pack.weightSize / Mathf.Max(1, pack.outC * kernelVolume));
            pack.inPacks = (pack.inC + 3) / 4;
            pack.outPacks = (pack.outC + 3) / 4;

            phaseSw.Restart();
            var w = NcnnGraphSession.ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
            var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
            phaseSw.Stop();
            readMs += phaseSw.ElapsedMilliseconds;

            if (owner.KeepRawConvWeightsForTexturePath || owner.ForceBufferConvolution || owner.ForceBufferConvolutionAll)
            {
                phaseSw.Restart();
                pack.rawWeight = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                NcnnGpuResourceTracker.RegisterBuffer(pack.rawWeight, w.Length, sizeof(float), "NcnnGraphSession.Deconv3DRawWeight:" + layer.name);
                pack.rawBias = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                NcnnGpuResourceTracker.RegisterBuffer(pack.rawBias, b.Length, sizeof(float), "NcnnGraphSession.Deconv3DRawBias:" + layer.name);
                pack.rawWeight.SetData(w);
                pack.rawBias.SetData(b);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;
            }

            phaseSw.Restart();
            var w4 = NcnnGraphSession.PackWeightsToO4I4K3D(
                w,
                pack.outC,
                pack.inC,
                pack.kernelW,
                pack.kernelH,
                pack.kernelD,
                pack.outPacks,
                pack.inPacks);
            var b4 = NcnnGraphSession.PackBiasToO4(b, pack.outC, pack.outPacks);
            pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
            NcnnGpuResourceTracker.RegisterBuffer(pack.packedWeight4, w4.Length, sizeof(float) * 4, "NcnnGraphSession.Deconv3DPackedWeight4:" + layer.name);
            pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
            NcnnGpuResourceTracker.RegisterBuffer(pack.packedBias4, b4.Length, sizeof(float) * 4, "NcnnGraphSession.Deconv3DPackedBias4:" + layer.name);
            pack.packedWeight4.SetData(w4);
            pack.packedBias4.SetData(b4);
            phaseSw.Stop();
            packMs += phaseSw.ElapsedMilliseconds;

            owner._deconv[layer.name] = pack;
            return new NcnnGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("Deconvolution3D not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || srcView.dims != 4)
                throw new InvalidOperationException("Deconvolution3D expects dims=4 tensor input: " + layer.name);
            if (deconv.rawWeight == null || deconv.rawBias == null)
                throw new InvalidOperationException("Deconvolution3D raw weights unavailable: " + layer.name);

            var outW = NcnnGraphSession.ComputeDeconvOut(srcView.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outH = NcnnGraphSession.ComputeDeconvOut(srcView.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
            var outD = NcnnGraphSession.ComputeDeconvOut(srcView.d, deconv.kernelD, deconv.dilationD, deconv.strideD, deconv.padFront, deconv.padBehind, deconv.outputPadBehind);
            var outTensor = owner.RentTempTensorBuffer(4, outW, outH, outD, deconv.outC);

            owner.Ops.Deconvolution3D(
                srcView,
                deconv.rawWeight,
                deconv.rawBias,
                deconv.outC,
                deconv.kernelW,
                deconv.kernelH,
                deconv.kernelD,
                deconv.strideW,
                deconv.strideH,
                deconv.strideD,
                deconv.padLeft,
                deconv.padRight,
                deconv.padTop,
                deconv.padBottom,
                deconv.padFront,
                deconv.padBehind,
                deconv.outputPadRight,
                deconv.outputPadBottom,
                deconv.outputPadBehind,
                deconv.dilationW,
                deconv.dilationH,
                deconv.dilationD,
                deconv.activationType,
                deconv.activationSlope,
                outTensor);

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

        public override void ExecuteRenderTexturePath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("Deconvolution3D not found: " + layer.name);

            if (!owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    textureBlobs,
                    textureShapes,
                    bufferBlobs,
                    bufferViews,
                    out var srcTex,
                    out var srcShape))
            {
#pragma warning disable CS0618
                ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
                return;
            }

            if (srcShape.dims != 4)
            {
#pragma warning disable CS0618
                ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
                return;
            }
            if (deconv.packedWeight4 == null || deconv.packedBias4 == null)
                throw new InvalidOperationException("Deconvolution3D packed weights unavailable: " + layer.name);
            if (deconv.group != 1)
                throw new NotSupportedException("Deconvolution3D render-texture path currently supports group=1 only: " + layer.name);

            var outW = NcnnGraphSession.ComputeDeconvOut(srcShape.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outH = NcnnGraphSession.ComputeDeconvOut(srcShape.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
            var outD = NcnnGraphSession.ComputeDeconvOut(srcShape.d, deconv.kernelD, deconv.dilationD, deconv.strideD, deconv.padFront, deconv.padBehind, deconv.outputPadBehind);
            var outRt = owner.RentTempArray(outW, outH, outD * deconv.outPacks, NcnnGraphSession.ResolveTensorTextureFormat(4));

            owner.Ops.Deconvolution3dPack4CDHW(
                srcTex.texture,
                srcShape.w,
                srcShape.h,
                srcShape.d,
                deconv.inPacks,
                deconv.packedWeight4,
                deconv.packedBias4,
                outW,
                outH,
                outD,
                deconv.outPacks,
                deconv.kernelW,
                deconv.kernelH,
                deconv.kernelD,
                deconv.strideW,
                deconv.strideH,
                deconv.strideD,
                deconv.padLeft,
                deconv.padRight,
                deconv.padTop,
                deconv.padBottom,
                deconv.padFront,
                deconv.padBehind,
                deconv.dilationW,
                deconv.dilationH,
                deconv.dilationD,
                deconv.activationType,
                deconv.activationSlope,
                outRt);

            NcnnGraphSession.SetTextureBlob(
                textureBlobs,
                textureShapes,
                layer.topNames[0],
                outRt,
                new NcnnGraphSession.BufferShape(4, outW, outH, outD, deconv.outC));
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("Deconvolution3D not found: " + layer.name);

            var src = NcnnGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcContract = NcnnGraphSession.GetCmdTensorContract(src);
            var srcShape = srcContract.LogicalShape;
            if (srcShape.dims != 4)
                throw new InvalidOperationException("Deconvolution3D command-buffer path expects dims=4 input: " + layer.name);
            if (!srcContract.IsPack4Image || !NcnnGraphSession.MatchesPack4TextureStorage(src, srcShape))
                throw new InvalidOperationException("Deconvolution3D command-buffer path requires a TensorDescriptor-backed CDHW Pack4 Texture2DArray: " + layer.name);
            if (deconv.group != 1)
                throw new NotSupportedException("Deconvolution3D command-buffer path currently supports group=1 only: " + layer.name);
            if (deconv.packedWeight4 == null || deconv.packedBias4 == null)
                throw new InvalidOperationException("Deconvolution3D packed weights unavailable: " + layer.name);
            if (src.packs != deconv.inPacks)
                throw new InvalidOperationException("Deconvolution3D command-buffer input pack mismatch: " + layer.name);

            var outW = NcnnGraphSession.ComputeDeconvOut(srcShape.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outH = NcnnGraphSession.ComputeDeconvOut(srcShape.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
            var outD = NcnnGraphSession.ComputeDeconvOut(srcShape.d, deconv.kernelD, deconv.dilationD, deconv.strideD, deconv.padFront, deconv.padBehind, deconv.outputPadBehind);
            var outShape = new NcnnGraphSession.BufferShape(4, outW, outH, outD, deconv.outC);
            var outRt = owner.RentTempArray(cmd, outW, outH, outD * deconv.outPacks, NcnnGraphSession.ResolveTensorTextureFormat(4));
            owner.Ops.Deconvolution3dPack4CDHW(
                cmd,
                src.texture,
                srcShape.w,
                srcShape.h,
                srcShape.d,
                deconv.inPacks,
                deconv.packedWeight4,
                deconv.packedBias4,
                outW,
                outH,
                outD,
                deconv.outPacks,
                deconv.kernelW,
                deconv.kernelH,
                deconv.kernelD,
                deconv.strideW,
                deconv.strideH,
                deconv.strideD,
                deconv.padLeft,
                deconv.padRight,
                deconv.padTop,
                deconv.padBottom,
                deconv.padFront,
                deconv.padBehind,
                deconv.dilationW,
                deconv.dilationH,
                deconv.dilationD,
                deconv.activationType,
                deconv.activationSlope,
                outRt);

            blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(
                outRt,
                outShape,
                outShape,
                owned: true,
                blobName: layer.topNames[0]);
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
