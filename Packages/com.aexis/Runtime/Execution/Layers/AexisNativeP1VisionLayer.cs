using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Native ncnn P1 visual layers. This class deliberately accepts only
    // texture-backed tensors and materializes LinearMat inputs through existing
    // texture transforms; it never obtains an activation through a ComputeBuffer.
    public sealed class AexisNativeP1VisionLayer : AexisBaseLayer
    {
        private sealed class DeformableConvPack : IDisposable
        {
            public int inChannels;
            public int outChannels;
            public int kernelW;
            public int kernelH;
            public ComputeBuffer weights;
            public ComputeBuffer bias;

            public void Dispose()
            {
                try { AexisGpuResourceTracker.ReleaseBuffer(weights, "AexisNativeP1VisionLayer.DeformableConvPack.weights"); weights?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(bias, "AexisNativeP1VisionLayer.DeformableConvPack.bias"); bias?.Dispose(); } catch { }
            }
        }

        private sealed class DetectionBiasPack : IDisposable
        {
            public ComputeBuffer biases;
            public int count;

            public void Dispose()
            {
                try { AexisGpuResourceTracker.ReleaseBuffer(biases, "AexisNativeP1VisionLayer.DetectionBiasPack.biases"); biases?.Dispose(); } catch { }
            }
        }

        private sealed class RenderInput
        {
            public RenderTexture texture;
            public AexisGraphSession.BufferShape shape;
            public RenderTexture temporary;
        }

        private sealed class CmdInput
        {
            public ComputeTexture texture;
            public AexisGraphSession.BufferShape shape;
            public ComputeTexture temporary;
        }

        public AexisNativeP1VisionLayer(AexisLayerTypeKey typeKey)
            : base(typeKey, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            AexisP1VisionSchema.Validate(layer);
            if (!string.Equals(layer.typeName, "DeformableConv2D", StringComparison.Ordinal)
                && !IsYolo(layer.typeName))
                return default;

            if (IsYolo(layer.typeName))
            {
                var biases = layer.GetFloats(4, Array.Empty<float>());
                var boxes = layer.GetInt(1, 0);
                if (boxes <= 0 || biases == null || biases.Length < boxes * 2 || (biases.Length & 1) != 0)
                    throw new InvalidOperationException("YOLO detection output requires an even bias list with at least 2*num_box values: " + layer.name);
                var biasPack = new DetectionBiasPack
                {
                    count = biases.Length,
                    biases = new ComputeBuffer(biases.Length, sizeof(float), ComputeBufferType.Structured)
                };
                biasPack.biases.SetData(biases);
                AexisGpuResourceTracker.RegisterBuffer(biasPack.biases, biases.Length, sizeof(float), "AexisNativeP1VisionLayer.Yolo.biases:" + layer.name);
                owner._extraPacks[layer.name] = biasPack;
                return default;
            }

            var outChannels = layer.GetInt(0);
            var kernelW = layer.GetInt(1);
            var kernelH = layer.GetInt(11, kernelW);
            var weightCount = layer.GetInt(6);
            var kernelArea = checked(kernelW * kernelH);
            var denominator = checked(outChannels * kernelArea);
            if (denominator <= 0 || weightCount <= 0 || weightCount % denominator != 0)
                throw new InvalidOperationException("DeformableConv2D weight_data_size is not OIHW: " + layer.name);
            var inChannels = weightCount / denominator;
            var bytesStart = br.Position;
            var weights = AexisGraphSession.ReadPackedOrRawWeightArray(br, weightCount, layer.name);
            var biasData = layer.GetInt(5, 0) != 0 ? br.ReadFloat32Array(outChannels) : new float[outChannels];
            var pack = new DeformableConvPack
            {
                inChannels = inChannels,
                outChannels = outChannels,
                kernelW = kernelW,
                kernelH = kernelH,
                weights = new ComputeBuffer(weightCount, sizeof(float), ComputeBufferType.Structured),
                bias = new ComputeBuffer(outChannels, sizeof(float), ComputeBufferType.Structured)
            };
            pack.weights.SetData(weights);
            pack.bias.SetData(biasData);
            AexisGpuResourceTracker.RegisterBuffer(pack.weights, weightCount, sizeof(float), "AexisNativeP1VisionLayer.DeformableConv2D.weights:" + layer.name);
            AexisGpuResourceTracker.RegisterBuffer(pack.bias, outChannels, sizeof(float), "AexisNativeP1VisionLayer.DeformableConv2D.bias:" + layer.name);
            owner._extraPacks[layer.name] = pack;
            return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), 0, 0, 0);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            if (layer == null) throw new ArgumentNullException(nameof(layer));
            if (context == null) throw new ArgumentNullException(nameof(context));
            AexisP1VisionSchema.Validate(layer);

            RenderInput input0 = null;
            RenderInput input1 = null;
            RenderInput input2 = null;
            RenderTexture output = null;
            try
            {
                input0 = PrepareRenderInput(owner, context, layer.bottomNames[0]);
                var needsInput1 = NeedsSecondInput(layer.typeName);
                var hasInput1 = needsInput1 && layer.bottomNames != null && layer.bottomNames.Length > 1;
                input1 = hasInput1 ? PrepareRenderInput(owner, context, layer.bottomNames[1]) : input0;
                var hasInput2 = layer.bottomNames != null && layer.bottomNames.Length > 2;
                input2 = hasInput2 ? PrepareRenderInput(owner, context, layer.bottomNames[2]) : null;
                var dispatch = DescribeDispatch(owner, layer, input0.shape, hasInput1 ? input1.shape : EmptyShape(), input2?.shape ?? EmptyShape());
                output = owner.RentTempArray(dispatch.output.w, dispatch.output.h, SliceCount(dispatch.output), owner.ResolveActivationTextureFormat(dispatch.output.dims));
                owner.Ops.P1VisionPack4(dispatch, input0.texture, input1.texture, input2?.texture, output);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, dispatch.output);
                output = null;
            }
            finally
            {
                ReleaseRenderInput(owner, input2);
                if (!ReferenceEquals(input1, input0)) ReleaseRenderInput(owner, input1);
                ReleaseRenderInput(owner, input0);
                if (output != null) owner.ReturnTempArray(output);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            if (layer == null) throw new ArgumentNullException(nameof(layer));
            if (context == null) throw new ArgumentNullException(nameof(context));
            AexisP1VisionSchema.Validate(layer);

            CmdInput input0 = null;
            CmdInput input1 = null;
            CmdInput input2 = null;
            ComputeTexture output = null;
            try
            {
                input0 = PrepareCmdInput(owner, context, layer.bottomNames[0]);
                var needsInput1 = NeedsSecondInput(layer.typeName);
                var hasInput1 = needsInput1 && layer.bottomNames != null && layer.bottomNames.Length > 1;
                input1 = hasInput1 ? PrepareCmdInput(owner, context, layer.bottomNames[1]) : input0;
                var hasInput2 = layer.bottomNames != null && layer.bottomNames.Length > 2;
                input2 = hasInput2 ? PrepareCmdInput(owner, context, layer.bottomNames[2]) : null;
                var dispatch = DescribeDispatch(owner, layer, input0.shape, hasInput1 ? input1.shape : EmptyShape(), input2?.shape ?? EmptyShape());
                output = owner.RentTempArray(context.commandBuffer, dispatch.output.w, dispatch.output.h, SliceCount(dispatch.output), owner.ResolveActivationTextureFormat(dispatch.output.dims));
                owner.Ops.P1VisionPack4(context.commandBuffer, dispatch, input0.texture, input1.texture, input2?.texture, output);
                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, dispatch.output, dispatch.output, owned: true);
                output = null;
                if (context.shapes != null) context.shapes[layer.topNames[0]] = dispatch.output;
            }
            finally
            {
                ReleaseCmdInput(owner, context.commandBuffer, input2);
                if (!ReferenceEquals(input1, input0)) ReleaseCmdInput(owner, context.commandBuffer, input1);
                ReleaseCmdInput(owner, context.commandBuffer, input0);
                if (output != null) owner.ReturnTempArray(context.commandBuffer, output);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        // Used by strict planning as well as dispatch. Keeping this proof next to the
        // actual ABI prevents the capability catalog from accepting a shape which the
        // Pack4 kernel would subsequently reject at execution time.
        internal static AexisP1VisionDispatch DescribeDispatch(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape input0,
            AexisGraphSession.BufferShape input1,
            AexisGraphSession.BufferShape input2)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            AexisP1VisionSchema.Validate(layer);
            var type = layer.typeName ?? string.Empty;
            var dispatch = new AexisP1VisionDispatch
            {
                input0 = Normalize(input0),
                input1 = Normalize(input1),
                input2 = Normalize(input2),
                dilationW = 1,
                dilationH = 1,
                strideW = 1,
                strideH = 1,
                spatialScale = 1f
            };

            switch (type)
            {
                case "GridSample":
                    dispatch.kernel = AexisP1VisionKernel.GridSample;
                    dispatch.mode = layer.GetInt(0, 1);
                    dispatch.paddingMode = layer.GetInt(1, 1);
                    dispatch.alignCorners = layer.GetInt(2, 0);
                    dispatch.gridPermute = layer.GetInt(3, 0);
                    dispatch.output = ResolveGridOutput(dispatch.input0, dispatch.input1, dispatch.gridPermute, layer.name);
                    break;
                case "GLU":
                    dispatch.kernel = AexisP1VisionKernel.Glu;
                    dispatch.axis = NormalizeAxis(layer.GetInt(0, 0), dispatch.input0.dims, layer.name);
                    dispatch.output = ResolveGluOutput(dispatch.input0, dispatch.axis, layer.name);
                    break;
                case "Einsum":
                    ConfigureEinsum(ref dispatch, layer);
                    break;
                case "Diag":
                    dispatch.kernel = AexisP1VisionKernel.Diag;
                    dispatch.diagonal = layer.GetInt(0, 0);
                    dispatch.output = ResolveDiagOutput(dispatch.input0, dispatch.diagonal, layer.name);
                    break;
                case "Fold":
                    dispatch.kernel = AexisP1VisionKernel.Fold;
                    dispatch.kernelW = layer.GetInt(1);
                    dispatch.kernelH = layer.GetInt(11, dispatch.kernelW);
                    dispatch.dilationW = layer.GetInt(2, 1);
                    dispatch.dilationH = layer.GetInt(12, dispatch.dilationW);
                    dispatch.strideW = layer.GetInt(3, 1);
                    dispatch.strideH = layer.GetInt(13, dispatch.strideW);
                    dispatch.padLeft = layer.GetInt(4, 0);
                    dispatch.padRight = layer.GetInt(15, dispatch.padLeft);
                    dispatch.padTop = layer.GetInt(14, dispatch.padLeft);
                    dispatch.padBottom = layer.GetInt(16, dispatch.padTop);
                    dispatch.output = ResolveFoldOutput(dispatch.input0, dispatch, layer);
                    break;
                case "SPP":
                    dispatch.kernel = AexisP1VisionKernel.Spp;
                    dispatch.poolingType = layer.GetInt(0, 0);
                    dispatch.pyramidHeight = layer.GetInt(1, 1);
                    dispatch.output = new AexisGraphSession.BufferShape(2, SppBinCount(dispatch.pyramidHeight), dispatch.input0.c, 1, 1);
                    break;
                case "ROIAlign":
                    dispatch.kernel = AexisP1VisionKernel.RoiAlign;
                    SetRoiParameters(ref dispatch, layer);
                    dispatch.samplingRatio = layer.GetInt(3, 0);
                    dispatch.aligned = layer.GetInt(4, 0);
                    dispatch.roiVersion = layer.GetInt(5, 0);
                    dispatch.output = RoiOutput(dispatch, dispatch.input0, dispatch.input1, layer.name);
                    break;
                case "ROIPooling":
                    dispatch.kernel = AexisP1VisionKernel.RoiPooling;
                    SetRoiParameters(ref dispatch, layer);
                    dispatch.output = RoiOutput(dispatch, dispatch.input0, dispatch.input1, layer.name);
                    break;
                case "PSROIPooling":
                    dispatch.kernel = AexisP1VisionKernel.PsRoiPooling;
                    SetRoiParameters(ref dispatch, layer);
                    dispatch.psOutputDim = layer.GetInt(3, 0);
                    if (dispatch.psOutputDim <= 0 || dispatch.input0.c != dispatch.psOutputDim * dispatch.pooledW * dispatch.pooledH)
                        throw new InvalidOperationException("PSROIPooling requires C=output_dim*pooled_width*pooled_height: " + layer.name);
                    dispatch.output = new AexisGraphSession.BufferShape(3, dispatch.pooledW, dispatch.pooledH, 1, dispatch.psOutputDim);
                    break;
                case "DeformableConv2D":
                    if (!owner._extraPacks.TryGetValue(layer.name, out var rawPack) || rawPack is not DeformableConvPack pack)
                        throw new InvalidOperationException("DeformableConv2D immutable weights are unavailable: " + layer.name);
                    if (dispatch.input0.dims != 3 || dispatch.input0.c != pack.inChannels)
                        throw new InvalidOperationException("DeformableConv2D requires a rank-3 Pack4 feature tensor with its loaded input channels: " + layer.name);
                    dispatch.kernel = AexisP1VisionKernel.DeformableConv2D;
                    dispatch.kernelW = pack.kernelW;
                    dispatch.kernelH = pack.kernelH;
                    dispatch.dilationW = layer.GetInt(2, 1);
                    dispatch.dilationH = layer.GetInt(12, dispatch.dilationW);
                    dispatch.strideW = layer.GetInt(3, 1);
                    dispatch.strideH = layer.GetInt(13, dispatch.strideW);
                    dispatch.padLeft = layer.GetInt(4, 0);
                    dispatch.padRight = layer.GetInt(15, dispatch.padLeft);
                    dispatch.padTop = layer.GetInt(14, dispatch.padLeft);
                    dispatch.padBottom = layer.GetInt(16, dispatch.padTop);
                    dispatch.biasTerm = layer.GetInt(5, 0);
                    dispatch.activationType = layer.GetInt(9, 0);
                    dispatch.activationSlope = AexisGraphSession.ParseLeakySlope(layer);
                    dispatch.weights = pack.weights;
                    dispatch.bias = pack.bias;
                    dispatch.output = new AexisGraphSession.BufferShape(
                        3,
                        AexisGraphSession.ComputeConvOut(dispatch.input0.w, dispatch.kernelW, dispatch.dilationW, dispatch.strideW, dispatch.padLeft, dispatch.padRight),
                        AexisGraphSession.ComputeConvOut(dispatch.input0.h, dispatch.kernelH, dispatch.dilationH, dispatch.strideH, dispatch.padTop, dispatch.padBottom),
                        1,
                        pack.outChannels);
                    var expectedOffsetChannels = dispatch.kernelW * dispatch.kernelH * 2;
                    if (dispatch.input1.dims != 3 || dispatch.input1.w != dispatch.output.w || dispatch.input1.h != dispatch.output.h || dispatch.input1.c != expectedOffsetChannels)
                        throw new InvalidOperationException("DeformableConv2D offset tensor must be [2*kH*kW,outH,outW]: " + layer.name);
                    if (input2.c != 0 && (dispatch.input2.dims != 3 || dispatch.input2.w != dispatch.output.w || dispatch.input2.h != dispatch.output.h || dispatch.input2.c != dispatch.kernelW * dispatch.kernelH))
                        throw new InvalidOperationException("DeformableConv2D mask tensor must be [kH*kW,outH,outW]: " + layer.name);
                    break;
                case "Proposal":
                    dispatch.kernel = AexisP1VisionKernel.Proposal;
                    dispatch.featStride = layer.GetInt(0, 16);
                    dispatch.baseSize = layer.GetInt(1, 16);
                    dispatch.preNmsTopK = layer.GetInt(2, 6000);
                    dispatch.detectionCapacity = layer.GetInt(3, 300);
                    dispatch.nmsThreshold = layer.GetFloat(4, 0.7f);
                    dispatch.minSize = layer.GetFloat(5, 16f);
                    if (dispatch.input0.dims != 3 || dispatch.input1.dims != 3 || dispatch.input2.c == 0 || dispatch.featStride <= 0 || dispatch.baseSize <= 0 || dispatch.detectionCapacity <= 0)
                        throw new InvalidOperationException("Proposal requires rank-3 score/bbox textures, image-info texture, and positive static limits: " + layer.name);
                    if (dispatch.input0.c != 18 || dispatch.input1.c != 36 || dispatch.input0.w != dispatch.input1.w || dispatch.input0.h != dispatch.input1.h)
                        throw new InvalidOperationException("Proposal native profile uses the ncnn 9-anchor score[18]/bbox[36] layout: " + layer.name);
                    dispatch.output = new AexisGraphSession.BufferShape(2, 4, dispatch.detectionCapacity, 1, 1);
                    break;
                case "DetectionOutput":
                    dispatch.kernel = AexisP1VisionKernel.DetectionOutput;
                    dispatch.numClasses = layer.GetInt(0, 0);
                    dispatch.nmsThreshold = layer.GetFloat(1, 0.05f);
                    dispatch.preNmsTopK = layer.GetInt(2, 300);
                    dispatch.detectionCapacity = layer.GetInt(3, 100);
                    dispatch.confidenceThreshold = layer.GetFloat(4, 0.5f);
                    dispatch.variance0 = layer.GetFloat(5, 0.1f);
                    dispatch.variance1 = layer.GetFloat(6, 0.1f);
                    dispatch.variance2 = layer.GetFloat(7, 0.2f);
                    dispatch.variance3 = layer.GetFloat(8, 0.2f);
                    if (dispatch.numClasses <= 1 || dispatch.detectionCapacity <= 0 || dispatch.input2.c == 0)
                        throw new InvalidOperationException("DetectionOutput requires location, confidence, priorbox, num_class>1, and keep_top_k>0: " + layer.name);
                    if (TotalElements(dispatch.input0) % 4 != 0 || TotalElements(dispatch.input1) != (TotalElements(dispatch.input0) / 4) * dispatch.numClasses)
                        throw new InvalidOperationException("DetectionOutput requires [num_prior,4] locations and [num_prior,num_class] confidence storage: " + layer.name);
                    dispatch.output = new AexisGraphSession.BufferShape(2, 6, dispatch.detectionCapacity, 1, 1);
                    break;
                case "YoloDetectionOutput":
                case "Yolov3DetectionOutput":
                case "YoloDetectOut":
                case "Yolov3DetectOut":
                    if (!owner._extraPacks.TryGetValue(layer.name, out var detectionRaw) || detectionRaw is not DetectionBiasPack detectionPack)
                        throw new InvalidOperationException("YOLO detection output immutable bias constants are unavailable: " + layer.name);
                    dispatch.kernel = AexisP1VisionKernel.YoloDetectionOutput;
                    dispatch.numClasses = layer.GetInt(0, 0);
                    dispatch.numBoxes = layer.GetInt(1, 0);
                    dispatch.confidenceThreshold = layer.GetFloat(2, 0.01f);
                    dispatch.nmsThreshold = layer.GetFloat(3, 0.45f);
                    dispatch.yoloV3 = type.IndexOf("v3", StringComparison.OrdinalIgnoreCase) >= 0 ? 1 : 0;
                    dispatch.detectionBiases = detectionPack.biases;
                    dispatch.detectionCapacity = ResolveDetectionCapacity(layer, dispatch.input0, dispatch.input1, dispatch.input2);
                    ValidateYoloInput(dispatch.input0, dispatch.numBoxes, dispatch.numClasses, layer.name);
                    if (input1.c > 0) ValidateYoloInput(dispatch.input1, dispatch.numBoxes, dispatch.numClasses, layer.name);
                    if (input2.c > 0) ValidateYoloInput(dispatch.input2, dispatch.numBoxes, dispatch.numClasses, layer.name);
                    dispatch.output = new AexisGraphSession.BufferShape(2, 6, dispatch.detectionCapacity, 1, 1);
                    break;
                default:
                    throw new NotSupportedException("No native P1 vision dispatch is registered for " + type + ".");
            }
            return dispatch;
        }

        private static void SetRoiParameters(ref AexisP1VisionDispatch dispatch, AexisGraphModel.Layer layer)
        {
            dispatch.pooledW = layer.GetInt(0);
            dispatch.pooledH = layer.GetInt(1);
            dispatch.spatialScale = layer.GetFloat(2, 1f);
        }

        private static AexisGraphSession.BufferShape RoiOutput(
            AexisP1VisionDispatch dispatch,
            AexisGraphSession.BufferShape feature,
            AexisGraphSession.BufferShape rois,
            string layerName)
        {
            if (feature.dims != 3)
                throw new InvalidOperationException("ROI P1 kernels require rank-3 Pack4 feature input: " + layerName);
            var roiElements = checked(rois.w * rois.h * rois.d * rois.c);
            if (roiElements <= 0 || roiElements % 4 != 0)
                throw new InvalidOperationException("ROI P1 kernels require a statically bounded linear [num_rois,4] ROI tensor: " + layerName);
            var roiCount = roiElements / 4;
            // Each ROI is an independent Texture2DArray depth slice. This keeps the
            // standard ONNX ROI axis on GPU and never requires a CPU loop/readback.
            return new AexisGraphSession.BufferShape(4, dispatch.pooledW, dispatch.pooledH, roiCount, feature.c);
        }

        private static AexisGraphSession.BufferShape ResolveGridOutput(AexisGraphSession.BufferShape input, AexisGraphSession.BufferShape grid, int permute, string layerName)
        {
            if (input.dims != 3 && input.dims != 4)
                throw new InvalidOperationException("GridSample requires rank-3 or rank-4 feature input: " + layerName);
            if (permute != 0)
            {
                if (grid.dims != input.dims || grid.c < (input.dims == 4 ? 3 : 2))
                    throw new InvalidOperationException("GridSample permuted grid must match feature rank and expose coordinate channels: " + layerName);
                return new AexisGraphSession.BufferShape(input.dims, grid.w, grid.h, input.dims == 4 ? grid.d : 1, input.c);
            }
            if (input.dims == 3)
            {
                if (grid.dims != 3 || grid.h <= 0 || grid.c <= 0)
                    throw new InvalidOperationException("GridSample ncnn grid must have rank 3: " + layerName);
                return new AexisGraphSession.BufferShape(3, grid.h, grid.c, 1, input.c);
            }
            if (grid.dims != 4 || grid.h <= 0 || grid.d <= 0 || grid.c <= 0)
                throw new InvalidOperationException("GridSample ncnn 3D grid must have rank 4: " + layerName);
            return new AexisGraphSession.BufferShape(4, grid.h, grid.d, grid.c, input.c);
        }

        private static AexisGraphSession.BufferShape ResolveGluOutput(AexisGraphSession.BufferShape source, int axis, string layerName)
        {
            var w = source.w;
            var h = source.h;
            var d = source.d;
            var c = source.c;
            if (source.dims <= 1) w = RequireEven(w, layerName);
            else if (source.dims == 2) { if (axis == 0) h = RequireEven(h, layerName); else w = RequireEven(w, layerName); }
            else if (source.dims == 3) { if (axis == 0) c = RequireEven(c, layerName); else if (axis == 1) h = RequireEven(h, layerName); else w = RequireEven(w, layerName); }
            else { if (axis == 0) c = RequireEven(c, layerName); else if (axis == 1) d = RequireEven(d, layerName); else if (axis == 2) h = RequireEven(h, layerName); else w = RequireEven(w, layerName); }
            return new AexisGraphSession.BufferShape(source.dims, w, h, d, c);
        }

        private static AexisGraphSession.BufferShape ResolveDiagOutput(AexisGraphSession.BufferShape source, int diagonal, string layerName)
        {
            if (source.dims <= 1)
            {
                var side = source.w + Math.Abs(diagonal);
                return new AexisGraphSession.BufferShape(2, side, side, 1, 1);
            }
            if (source.dims == 2)
            {
                var minimum = Math.Min(source.w - source.h, 0);
                var maximum = Math.Max(source.w - source.h, 0);
                var length = diagonal >= minimum && diagonal <= maximum
                    ? Math.Min(source.w, source.h)
                    : diagonal > -source.h && diagonal < minimum
                        ? diagonal + source.h
                        : diagonal > maximum && diagonal < source.w ? -diagonal + source.w : 0;
                return new AexisGraphSession.BufferShape(1, Math.Max(1, length), 1, 1, 1);
            }
            throw new InvalidOperationException("Diag supports rank-1 and rank-2 inputs only: " + layerName);
        }

        private static AexisGraphSession.BufferShape ResolveFoldOutput(AexisGraphSession.BufferShape source, AexisP1VisionDispatch dispatch, AexisGraphModel.Layer layer)
        {
            if (source.dims != 2)
                throw new InvalidOperationException("Fold requires rank-2 unfolded matrix storage: " + layer.name);
            var kernelArea = checked(dispatch.kernelW * dispatch.kernelH);
            if (source.h % kernelArea != 0)
                throw new InvalidOperationException("Fold source rows must divide kernel area: " + layer.name);
            var outW = layer.GetInt(20);
            var outH = layer.GetInt(21, outW);
            return new AexisGraphSession.BufferShape(3, outW, outH, 1, source.h / kernelArea);
        }

        private static int SppBinCount(int height)
        {
            var total = 0;
            var bins = 1;
            for (var level = 0; level < height; level++) { total += bins * bins; bins <<= 1; }
            return total;
        }

        private static void ConfigureEinsum(ref AexisP1VisionDispatch dispatch, AexisGraphModel.Layer layer)
        {
            var equation = (layer.GetString("equation", layer.GetString("onnx.equation", string.Empty)) ?? string.Empty).Replace(" ", string.Empty);
            var arrow = equation.IndexOf("->", StringComparison.Ordinal);
            if (arrow <= 0 || equation.IndexOf("...", StringComparison.Ordinal) >= 0)
                throw new InvalidOperationException("Einsum requires an explicit static equation without ellipsis: " + layer.name);
            var operands = equation.Substring(0, arrow).Split(',');
            var output = equation.Substring(arrow + 2);
            var expectedInputs = dispatch.input2.c > 0 ? 3 : 2;
            if ((operands.Length != 2 && operands.Length != 3) || operands.Length != expectedInputs || output.Length == 0 || output.Length > 4)
                throw new InvalidOperationException("Einsum native Pack4 profile accepts two or three operands and rank<=4 explicit output: " + layer.name);
            var shapes = operands.Length == 2
                ? new[] { dispatch.input0, dispatch.input1 }
                : new[] { dispatch.input0, dispatch.input1, dispatch.input2 };
            var labels = new Dictionary<char, int>();
            foreach (var operand in operands)
            {
                if (operand.Length == 0 || operand.Length > 4 || !AllDistinct(operand))
                    throw new InvalidOperationException("Einsum operands must have unique rank<=4 labels: " + layer.name);
                foreach (var label in operand)
                    if (!labels.ContainsKey(label)) labels.Add(label, labels.Count);
            }
            if (!AllDistinct(output))
                throw new InvalidOperationException("Einsum output labels must be unique: " + layer.name);
            foreach (var label in output)
                if (!labels.ContainsKey(label))
                    throw new InvalidOperationException("Einsum output label is absent from operands: " + layer.name);
            if (labels.Count > 8)
                throw new InvalidOperationException("Einsum native profile supports at most eight static labels: " + layer.name);

            var extents = new int[8];
            var mappings = new int[3][];
            for (var operand = 0; operand < operands.Length; operand++)
            {
                if (operands[operand].Length != shapes[operand].dims)
                    throw new InvalidOperationException("Einsum operand rank does not match its Pack4 tensor rank: " + layer.name);
                mappings[operand] = PhysicalLabelMap(operands[operand], shapes[operand].dims, labels);
                var physicalExtents = PhysicalExtents(shapes[operand]);
                for (var axis = 0; axis < 4; axis++)
                {
                    var label = mappings[operand][axis];
                    if (label < 0) continue;
                    var extent = physicalExtents[axis];
                    if (extents[label] == 0) extents[label] = extent;
                    else if (extents[label] != extent && extents[label] != 1 && extent != 1)
                        throw new InvalidOperationException("Einsum non-broadcast label extents disagree: " + layer.name);
                    else extents[label] = Math.Max(extents[label], extent);
                }
            }
            var outputPhysical = PhysicalLabelMap(output, output.Length, labels);
            var outputShape = OutputShape(output, extents, labels);
            var reduction = new List<int>();
            for (var label = 0; label < labels.Count; label++)
                if (output.IndexOf(LabelAt(labels, label)) < 0) reduction.Add(label);
            if (reduction.Count > 4)
                throw new InvalidOperationException("Einsum native profile supports at most four reduced labels: " + layer.name);

            dispatch.kernel = AexisP1VisionKernel.Einsum;
            dispatch.output = outputShape;
            dispatch.einsumOperandCount = operands.Length;
            dispatch.einsumLabelCount = labels.Count;
            dispatch.einsumReductionCount = reduction.Count;
            dispatch.einsumDims = extents;
            dispatch.einsumA = mappings[0];
            dispatch.einsumB = mappings[1];
            dispatch.einsumC = operands.Length == 3 ? mappings[2] : new[] { -1, -1, -1, -1 };
            dispatch.einsumOutput = outputPhysical;
            dispatch.einsumReduction = reduction.ToArray();
        }

        private static bool AllDistinct(string value)
        {
            var seen = new HashSet<char>();
            foreach (var character in value) if (!seen.Add(character)) return false;
            return true;
        }

        private static char LabelAt(Dictionary<char, int> labels, int id)
        {
            foreach (var pair in labels) if (pair.Value == id) return pair.Key;
            throw new ArgumentOutOfRangeException(nameof(id));
        }

        // Maps canonical equation axes to Pack4 storage axes [C,D,H,W].
        private static int[] PhysicalLabelMap(string equationAxes, int dims, Dictionary<char, int> labels)
        {
            var result = new[] { -1, -1, -1, -1 };
            if (dims == 1) result[3] = labels[equationAxes[0]];
            else if (dims == 2) { result[2] = labels[equationAxes[0]]; result[3] = labels[equationAxes[1]]; }
            else if (dims == 3) { result[0] = labels[equationAxes[0]]; result[2] = labels[equationAxes[1]]; result[3] = labels[equationAxes[2]]; }
            else if (dims == 4) { result[0] = labels[equationAxes[0]]; result[1] = labels[equationAxes[1]]; result[2] = labels[equationAxes[2]]; result[3] = labels[equationAxes[3]]; }
            else throw new InvalidOperationException("Einsum Pack4 rank must be in [1,4].");
            return result;
        }

        private static int[] PhysicalExtents(AexisGraphSession.BufferShape shape)
        {
            return new[] { shape.dims >= 3 ? shape.c : 1, shape.dims == 4 ? shape.d : 1, shape.dims >= 2 ? shape.h : 1, shape.w };
        }

        private static AexisGraphSession.BufferShape OutputShape(string labels, int[] extents, Dictionary<char, int> labelIds)
        {
            if (labels.Length == 1) return new AexisGraphSession.BufferShape(1, extents[labelIds[labels[0]]], 1, 1, 1);
            if (labels.Length == 2) return new AexisGraphSession.BufferShape(2, extents[labelIds[labels[1]]], extents[labelIds[labels[0]]], 1, 1);
            if (labels.Length == 3) return new AexisGraphSession.BufferShape(3, extents[labelIds[labels[2]]], extents[labelIds[labels[1]]], 1, extents[labelIds[labels[0]]]);
            return new AexisGraphSession.BufferShape(4, extents[labelIds[labels[3]]], extents[labelIds[labels[2]]], extents[labelIds[labels[1]]], extents[labelIds[labels[0]]]);
        }

        private static int NormalizeAxis(int axis, int dims, string layerName)
        {
            if (dims <= 1) return 0;
            var positive = axis < 0 ? axis + dims : axis;
            if (positive < 0 || positive >= dims)
                throw new InvalidOperationException("P1 axis is outside tensor rank: " + layerName);
            return positive;
        }

        private static int RequireEven(int value, string layerName)
        {
            if (value <= 0 || (value & 1) != 0)
                throw new InvalidOperationException("GLU split dimension must be a positive even size: " + layerName);
            return value / 2;
        }

        private static bool NeedsSecondInput(string type)
        {
            return type == "GridSample" || type == "ROIAlign" || type == "ROIPooling" || type == "PSROIPooling"
                || type == "DeformableConv2D" || type == "Proposal" || type == "DetectionOutput" || type == "Einsum" || IsYolo(type);
        }

        private static bool IsYolo(string type)
        {
            return type == "YoloDetectionOutput" || type == "Yolov3DetectionOutput" || type == "YoloDetectOut" || type == "Yolov3DetectOut";
        }

        private static int TotalElements(AexisGraphSession.BufferShape shape)
        {
            return checked(Math.Max(1, shape.w) * Math.Max(1, shape.h) * Math.Max(1, shape.d) * Math.Max(1, shape.c));
        }

        private static int ResolveDetectionCapacity(AexisGraphModel.Layer layer, AexisGraphSession.BufferShape input0, AexisGraphSession.BufferShape input1, AexisGraphSession.BufferShape input2)
        {
            var declared = int.TryParse(layer.GetString("aexis.max_detections", string.Empty), out var parsed) ? parsed : 0;
            if (declared > 0) return declared;
            var candidates = checked(input0.w * input0.h * Math.Max(1, layer.GetInt(1, 0))
                + (input1.c > 0 ? input1.w * input1.h * Math.Max(1, layer.GetInt(1, 0)) : 0)
                + (input2.c > 0 ? input2.w * input2.h * Math.Max(1, layer.GetInt(1, 0)) : 0));
            return Math.Min(candidates, 300);
        }

        private static void ValidateYoloInput(AexisGraphSession.BufferShape input, int boxes, int classes, string layerName)
        {
            if (input.dims != 3 || boxes <= 0 || classes <= 0 || input.c != boxes * (5 + classes))
                throw new InvalidOperationException("YOLO detection input channels must equal num_box*(5+num_class): " + layerName);
        }

        private static AexisGraphSession.BufferShape Normalize(AexisGraphSession.BufferShape shape)
        {
            return new AexisGraphSession.BufferShape(Math.Max(1, shape.dims), Math.Max(1, shape.w), Math.Max(1, shape.h), Math.Max(1, shape.d), Math.Max(0, shape.c));
        }

        private static AexisGraphSession.BufferShape EmptyShape() => new AexisGraphSession.BufferShape(3, 1, 1, 1, 0);

        private static int SliceCount(AexisGraphSession.BufferShape shape)
        {
            return Math.Max(1, shape.d) * Math.Max(1, Mathf.CeilToInt(Math.Max(1, shape.c) / 4f));
        }

        private static RenderInput PrepareRenderInput(AexisGraphSession owner, AexisLayerBufferContext context, string name)
        {
            if (!AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, name, out var source, out var shape))
                throw new InvalidOperationException("P1 texture-native path requires texture input: " + name);
            var result = new RenderInput { texture = source.texture, shape = Normalize(shape) };
            if (!AexisGraphSession.IsStrictLinearMatTexture(source))
            {
                if (!AexisGraphSession.MatchesPack4TextureStorage(source, shape))
                    throw new InvalidOperationException("P1 requires exact Pack4 or LinearMat texture storage: " + name);
                return result;
            }
            var storage = AexisGraphSession.GetTextureStorageShape(source, shape);
            result.temporary = owner.RentTempArray(result.shape.w, result.shape.h, SliceCount(result.shape), owner.ResolveActivationTextureFormat(result.shape.dims));
            owner.Ops.ReshapeLinearMatToPack4(source.texture, storage.w, storage.h, result.shape.w, result.shape.h, result.shape.d, result.shape.c, result.shape.dims, result.temporary);
            result.texture = result.temporary;
            return result;
        }

        private static CmdInput PrepareCmdInput(AexisGraphSession owner, AexisLayerCommandBufferContext context, string name)
        {
            var source = AexisGraphSession.GetCmdTensor(context.blobs, name);
            var shape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, name);
            var result = new CmdInput { texture = source.texture, shape = Normalize(shape) };
            if (!AexisGraphSession.IsStrictLinearMatTexture(source))
            {
                if (!AexisGraphSession.MatchesPack4TextureStorage(source, shape))
                    throw new InvalidOperationException("P1 requires exact Pack4 or LinearMat command texture storage: " + name);
                return result;
            }
            var storage = AexisGraphSession.GetCmdStorageShape(source, shape);
            result.temporary = owner.RentTempArray(context.commandBuffer, result.shape.w, result.shape.h, SliceCount(result.shape), owner.ResolveActivationTextureFormat(result.shape.dims));
            owner.Ops.ReshapeLinearMatToPack4(context.commandBuffer, source.texture, storage.w, storage.h, result.shape.w, result.shape.h, result.shape.d, result.shape.c, result.shape.dims, result.temporary);
            result.texture = result.temporary;
            return result;
        }

        private static void ReleaseRenderInput(AexisGraphSession owner, RenderInput input)
        {
            if (input?.temporary != null) owner.ReturnTempArray(input.temporary);
        }

        private static void ReleaseCmdInput(AexisGraphSession owner, CommandBuffer commandBuffer, CmdInput input)
        {
            if (input?.temporary != null) owner.ReturnTempArray(commandBuffer, input.temporary);
        }
    }
}
