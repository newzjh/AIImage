using System;
using System.Diagnostics;
using UnityEngine;

namespace Aexis.Execution
{
    public sealed class AexisConvolution3DLayer : AexisBaseLayer
    {
        public AexisConvolution3DLayer()
            : base(AexisLayerTypes.Convolution3D, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
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

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            owner.DebugLog?.Invoke(
                "[Convolution3D.LoadLayer] start"
                + " | name=" + (layer?.name ?? string.Empty)
                + " | weightSize=" + layer.GetInt(6, 0).ToString()
                + " | outC=" + layer.GetInt(0, 0).ToString());

            var pack = new AexisGraphSession.ConvPack
            {
                outC = layer.GetInt(0, 0),
                group = Mathf.Max(1, layer.GetInt(7, 1)),
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
                biasTerm = layer.GetInt(5, 0),
                weightSize = layer.GetInt(6, 0),
                activationType = layer.GetInt(9, 0),
                useBufferPath = true,
                isDepthWise = false
            };

            var actParams = AexisGraphSession.ParseActivationParams(layer);
            pack.activationSlope = actParams.Length > 0 ? actParams[0] : AexisGraphSession.ParseLeakySlope(layer);

            var kernelVolume = Mathf.Max(1, pack.kernelW * pack.kernelH * pack.kernelD);
            var expectedDivisor = Mathf.Max(1, pack.outC * kernelVolume);
            pack.inC = Mathf.Max(1, pack.weightSize / expectedDivisor);
            pack.inPacks = (pack.inC + 3) / 4;
            pack.outPacks = (pack.outC + 3) / 4;

            phaseSw.Restart();
            var w = AexisGraphSession.ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
            var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
            phaseSw.Stop();
            readMs += phaseSw.ElapsedMilliseconds;

            if (owner.KeepRawConvWeightsForTexturePath || owner.ForceBufferConvolution || owner.ForceBufferConvolutionAll)
            {
                phaseSw.Restart();
                AexisGraphSession.UploadRawConvWeights(pack, w, b);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;
            }

            phaseSw.Restart();
            var w4 = AexisGraphSession.PackWeightsToO4I4K3D(
                w,
                pack.outC,
                pack.inC,
                pack.kernelW,
                pack.kernelH,
                pack.kernelD,
                pack.outPacks,
                pack.inPacks);
            var b4 = AexisGraphSession.PackBiasToO4(b, pack.outC, pack.outPacks);
            pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
            AexisGpuResourceTracker.RegisterBuffer(pack.packedWeight4, w4.Length, sizeof(float) * 4, "AexisGraphSession.Conv3DPackedWeight4:" + layer.name);
            pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
            AexisGpuResourceTracker.RegisterBuffer(pack.packedBias4, b4.Length, sizeof(float) * 4, "AexisGraphSession.Conv3DPackedBias4:" + layer.name);
            pack.packedWeight4.SetData(w4);
            pack.packedBias4.SetData(b4);
            phaseSw.Stop();
            packMs += phaseSw.ElapsedMilliseconds;

            owner._conv[layer.name] = pack;
            owner.DebugLog?.Invoke(
                "[Convolution3D.LoadLayer] done"
                + " | name=" + layer.name
                + " | inC=" + pack.inC.ToString()
                + " | outC=" + pack.outC.ToString()
                + " | kernel=" + pack.kernelW.ToString() + "x" + pack.kernelH.ToString() + "x" + pack.kernelD.ToString()
                + " | bytesRead=" + Math.Max(0, br.Position - bytesStart).ToString()
                + " | readMs=" + readMs.ToString()
                + " | uploadMs=" + uploadMs.ToString());
            return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
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

            if (!owner._conv.TryGetValue(layer.name, out var conv))
                throw new InvalidOperationException("Convolution3D not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || srcView.dims != 4)
                throw new InvalidOperationException("Convolution3D expects dims=4 tensor input: " + layer.name);
            if (conv.rawWeight == null || conv.rawBias == null)
                throw new InvalidOperationException("Convolution3D raw weights unavailable: " + layer.name);

            var outW = AexisGraphSession.ComputeConvOut(srcView.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
            var outH = AexisGraphSession.ComputeConvOut(srcView.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
            var outD = AexisGraphSession.ComputeConvOut(srcView.d, conv.kernelD, conv.dilationD, conv.strideD, conv.padFront, conv.padBehind);
            var outTensor = owner.RentTempTensorBuffer(4, outW, outH, outD, conv.outC);

            owner.Ops.Conv3dBuf(
                srcView,
                conv.rawWeight,
                conv.rawBias,
                conv.outC,
                conv.kernelW,
                conv.kernelH,
                conv.kernelD,
                conv.strideW,
                conv.strideH,
                conv.strideD,
                conv.padLeft,
                conv.padRight,
                conv.padTop,
                conv.padBottom,
                conv.padFront,
                conv.padBehind,
                conv.dilationW,
                conv.dilationH,
                conv.dilationD,
                conv.activationType,
                conv.activationSlope,
                outTensor);

            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: false,
                textureBlobs,
                textureShapes,
                bufferBlobs,
                bufferRefs,
                bufferViews,
                tempOwned);
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
            var tempOwned = context.tempOwned;

            if (!owner._conv.TryGetValue(layer.name, out var conv))
                throw new InvalidOperationException("Convolution3D not found: " + layer.name);

            if (!owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    textureBlobs,
                    textureShapes,
                    bufferBlobs,
                    bufferViews,
                    out var srcTex,
                    out var srcShape))
            {
                throw new InvalidOperationException("Convolution3D render-texture path requires pack4 texture input: " + layer.name);
            }

            if (srcShape.dims != 4)
            {
#pragma warning disable CS0618
                ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
                return;
            }
            if (conv.packedWeight4 == null || conv.packedBias4 == null)
                throw new InvalidOperationException("Convolution3D packed weights unavailable: " + layer.name);

            var outW = AexisGraphSession.ComputeConvOut(srcShape.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
            var outH = AexisGraphSession.ComputeConvOut(srcShape.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
            var outD = AexisGraphSession.ComputeConvOut(srcShape.d, conv.kernelD, conv.dilationD, conv.strideD, conv.padFront, conv.padBehind);
            var outRt = owner.RentTempArray(outW, outH, outD * conv.outPacks, AexisGraphSession.ResolveTensorTextureFormat(4));

            owner.Ops.Conv3dPack4CDHW(
                srcTex.texture,
                srcShape.w,
                srcShape.h,
                srcShape.d,
                conv.inPacks,
                conv.packedWeight4,
                conv.packedBias4,
                outW,
                outH,
                outD,
                conv.outPacks,
                conv.kernelW,
                conv.kernelH,
                conv.kernelD,
                conv.strideW,
                conv.strideH,
                conv.strideD,
                conv.padLeft,
                conv.padRight,
                conv.padTop,
                conv.padBottom,
                conv.padFront,
                conv.padBehind,
                conv.dilationW,
                conv.dilationH,
                conv.dilationD,
                conv.activationType,
                conv.activationSlope,
                outRt);

            AexisGraphSession.SetTextureBlob(
                textureBlobs,
                textureShapes,
                layer.topNames[0],
                outRt,
                new AexisGraphSession.BufferShape(4, outW, outH, outD, conv.outC));
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._conv.TryGetValue(layer.name, out var conv))
                throw new InvalidOperationException("Convolution3D not found: " + layer.name);

            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcContract = AexisGraphSession.GetCmdTensorContract(src);
            var srcShape = srcContract.LogicalShape;
            if (srcShape.dims != 4)
                throw new InvalidOperationException("Convolution3D command-buffer path expects dims=4 input: " + layer.name);
            if (!srcContract.IsPack4Image || !AexisGraphSession.MatchesPack4TextureStorage(src, srcShape))
                throw new InvalidOperationException("Convolution3D command-buffer path requires a TensorDescriptor-backed CDHW Pack4 Texture2DArray: " + layer.name);
            if (conv.group != 1)
                throw new NotSupportedException("Convolution3D command-buffer path currently supports group=1 only: " + layer.name);
            if (conv.packedWeight4 == null || conv.packedBias4 == null)
                throw new InvalidOperationException("Convolution3D packed weights unavailable: " + layer.name);
            if (src.packs != conv.inPacks)
                throw new InvalidOperationException("Convolution3D command-buffer input pack mismatch: " + layer.name);

            var outW = AexisGraphSession.ComputeConvOut(srcShape.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
            var outH = AexisGraphSession.ComputeConvOut(srcShape.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
            var outD = AexisGraphSession.ComputeConvOut(srcShape.d, conv.kernelD, conv.dilationD, conv.strideD, conv.padFront, conv.padBehind);
            var outShape = new AexisGraphSession.BufferShape(4, outW, outH, outD, conv.outC);
            var outRt = owner.RentTempArray(cmd, outW, outH, outD * conv.outPacks, AexisGraphSession.ResolveTensorTextureFormat(4));
            owner.Ops.Conv3dPack4CDHW(
                cmd,
                src.texture,
                srcShape.w,
                srcShape.h,
                srcShape.d,
                conv.inPacks,
                conv.packedWeight4,
                conv.packedBias4,
                outW,
                outH,
                outD,
                conv.outPacks,
                conv.kernelW,
                conv.kernelH,
                conv.kernelD,
                conv.strideW,
                conv.strideH,
                conv.strideD,
                conv.padLeft,
                conv.padRight,
                conv.padTop,
                conv.padBottom,
                conv.padFront,
                conv.padBehind,
                conv.dilationW,
                conv.dilationH,
                conv.dilationD,
                conv.activationType,
                conv.activationSlope,
                outRt);

            blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(
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
