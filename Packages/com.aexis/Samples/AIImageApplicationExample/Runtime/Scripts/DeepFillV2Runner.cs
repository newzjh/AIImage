using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading;
using Aexis.Samples.Async;
using Aexis.Ncnn;
using UnityEngine;
using UnityEngine.Rendering;
using Aexis.Execution;

public enum DeepFillV2Backend
{
    OnnxDirect = 0,
    NcnnBin = 1
}

public struct DeepFillV2Result
{
    public Texture2D texture;
    public string error;
    public long elapsedMs;
    public long loadElapsedMs;
    public long inferenceElapsedMs;
    public DeepFillV2Backend backend;
    public string dumpDir;
    public string modelReport;
}

public sealed class DeepFillV2Runner : MonoBehaviour
{
    private const string MaskInputName = "mask";
    private const string ImageInputName = "image";
    private const string OutputName = "out0";
    private const int InputWidth = 400;
    private const int InputHeight = 512;

    public DeepFillV2Backend backend = DeepFillV2Backend.NcnnBin;
    public string sourceOnnxRelativePath = "DeepFileV2/deepfillv2_case1.source.onnx";
    public string ncnnParamRelativePath = "DeepFileV2/deepfillv2_case1.ncnn.param";
    public string ncnnBinRelativePath = "DeepFileV2/deepfillv2_case1.ncnn.bin";
    public AexisPrecisionMode precisionMode = AexisPrecisionMode.Auto;
    public bool flipYInput = true;
    public bool flipYOutput = true;
    public bool preserveUnmaskedPixels = true;
    public bool enableDebugDump = true;
    public bool useArgbFloatTensor = true;
    public bool enableGeneralTextureConvolution = true;
    public bool enableDepthWiseTextureConvolution = true;
    public bool enableConv1x1TextureConvolution = true;
    // The default executes the strict texture graph in small frame-paced pieces.
    // This keeps transient Pack4 RT lifetime visible to tile-based drivers instead
    // of submitting the whole 223-layer dependency chain as one opaque command.
    // Keep the long command-buffer route as an explicit diagnostic only.
    public bool useLongCommandBufferExecution;
    public bool useAsyncComputeCommandBuffer;
    public bool enableLayerPathDebugLog = false;
    public string debugTensorBlobName = string.Empty;

    public event Action<float, string> ProgressChanged;

    private AexisOps _ops;
    private AexisGraphSession _repro;
    private DeepFillV2Backend _loadedBackend;
    private DeepFillV2Backend _effectiveBackend = DeepFillV2Backend.NcnnBin;
    private string _loadedSignature = string.Empty;
    private bool _hasLoaded;
    private bool _hasAppliedPrecisionMode;
    private AexisPrecisionMode _appliedPrecisionMode;
    private DeepFillV2OnnxNcnnImportReport _lastOnnxReport;
    private string _lastDumpDir;

    public async UniTask<DeepFillV2Result> ProcessAsync(Texture sourceImage, Texture maskImage, CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();
        var loadMs = 0L;
        var inferMs = 0L;
        Texture2D readableSource = null;
        Texture2D readableMask = null;
        Texture2D modelSource = null;
        Texture2D modelMask = null;
        RenderTexture source512 = null;
        RenderTexture mask512 = null;
        RenderTexture sourcePack4 = null;
        RenderTexture maskPack4 = null;
        RenderTexture commandOutputPack4 = null;
        RenderTexture rgb512 = null;
        Texture2D generated512 = null;
        Texture2D generatedFull = null;

        DeepFillV2Result Finish(DeepFillV2Result result)
        {
            sw.Stop();
            result.elapsedMs = sw.ElapsedMilliseconds;
            result.loadElapsedMs = loadMs;
            result.inferenceElapsedMs = inferMs;
            result.backend = _effectiveBackend;
            result.dumpDir = _lastDumpDir;
            if (string.IsNullOrEmpty(result.modelReport))
                result.modelReport = BuildModelReport();
            return result;
        }

        try
        {
            if (sourceImage == null || maskImage == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 source image or mask image is null." });

            _lastDumpDir = enableDebugDump ? CreateDumpDir() : null;
            EnsureRuntimeObjects();
            ApplyReproOptions();
            ReportProgress(0.08f, "Load model");

            var loadSw = Stopwatch.StartNew();
            await EnsureLoadedAsync(ct);
            loadSw.Stop();
            loadMs = loadSw.ElapsedMilliseconds;
            ct.ThrowIfCancellationRequested();
            ReportProgress(0.15f, "Read inputs");

            readableSource = EnsureReadable(sourceImage);
            readableMask = EnsureReadable(maskImage);
            if (readableSource == null || readableMask == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 failed to read source or mask texture." });

            // Use raw encoded values for model sampling, independent of the
            // importing project's Gamma/Linear setting.
            modelSource = CopyTexture(readableSource, true);
            modelMask = CopyTexture(readableMask, true);
            if (modelSource == null || modelMask == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 failed to copy model input textures." });

            ReportProgress(0.22f, "Resize source");
            source512 = PrepareModelInput(modelSource);
            ReportProgress(0.30f, "Resize mask");
            mask512 = PrepareModelInput(modelMask);
            if (source512 == null || mask512 == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 failed to prepare the 400x512 source or mask." });

            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                TryWriteTexturePng(source512, Path.Combine(_lastDumpDir, "00_source_400x512.png"));
                TryWriteTexturePng(mask512, Path.Combine(_lastDumpDir, "01_mask_400x512.png"));
            }
            var textureShapes = new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
            {
                [MaskInputName] = new AexisGraphSession.BufferShape(3, InputWidth, InputHeight, 1, 1),
                [ImageInputName] = new AexisGraphSession.BufferShape(3, InputWidth, InputHeight, 1, 3)
            };
            var pinned = new HashSet<string>(StringComparer.Ordinal) { OutputName };
            var debugTensorBlobs = ParseDebugTensorBlobNames(debugTensorBlobName);
            for (var i = 0; i < debugTensorBlobs.Count; i++)
                pinned.Add(debugTensorBlobs[i]);

            var inferSw = Stopwatch.StartNew();
            ReportProgress(0.34f, "Prepare DeepFillV2");
            var inputFormat = useArgbFloatTensor ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf;
            if (!useLongCommandBufferExecution)
            {
                // Keep the texture-only path responsive without turning every
                // lightweight graph step into a full display frame. Incremental
                // layers that need GPU retirement still yield until their fence
                // passes, so this does not weaken the TBDR dependency guard.
                const int layersPerFrame = 12;
                UnityEngine.Debug.Log("[DeepFillV2] execution=paced-immediate | layers-per-frame=" + layersPerFrame);
                ReportProgress(0.38f, "Pack source");
                sourcePack4 = _repro.RentTempArray(InputWidth, InputHeight, 1, inputFormat);
                _ops.PackRgbToPack4(source512, 0, 0, 1f, 1f, sourcePack4, flipYInput, 1f);

                ReportProgress(0.46f, "Pack mask");
                maskPack4 = _repro.RentTempArray(InputWidth, InputHeight, 1, inputFormat);
                _ops.PackRgbToPack4(mask512, 0, 0, 1f, 1f, maskPack4, flipYInput, 1f);
                ReportProgress(0.52f, "Run DeepFillV2");

                var textureInputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
                {
                    [MaskInputName] = maskPack4,
                    [ImageInputName] = sourcePack4
                };
                using (var infer = await _repro.InferWithMultiInputsAsync(
                           textureInputs,
                           null,
                           pinned,
                           textureShapes,
                           cancellationToken: ct,
                           yieldEveryLayers: layersPerFrame,
                           progress: inferenceProgress => ReportProgress(
                               0.52f + 0.32f * inferenceProgress.progress01,
                               inferenceProgress.layerType == "DeepFillV2ContextualAttention"
                                   ? "DeepFillV2 context"
                                   : "Run DeepFillV2")))
                {
                    if (debugTensorBlobs.Count > 0 && enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                    {
                        for (var i = 0; i < debugTensorBlobs.Count; i++)
                            TryWriteDebugTensorBlob(infer, debugTensorBlobs[i], _lastDumpDir);
                    }

                    var outPack4 = infer.GetTexture(OutputName);
                    rgb512 = NewRenderTexture(InputWidth, InputHeight, RenderTextureFormat.ARGB32, true, "DeepFillV2.Rgb400x512");
                    ReportProgress(0.86f, "Unpack output");
                    _ops.Pack4ToRgbScaled(outPack4, rgb512, 0.5f, 0.5f, flipYOutput);
                    ReportProgress(0.92f, "Read output");
                    generated512 = await ReadRenderTexturePacedAsync(rgb512, TextureFormat.RGBA32, true, ct);
                }
            }
            else
            {
                using (var commandBuffer = new CommandBuffer { name = "DeepFillV2.Pack4" })
                {
                    var useAsyncCompute = useAsyncComputeCommandBuffer && SystemInfo.supportsAsyncCompute;
                    if (useAsyncCompute)
                        commandBuffer.SetExecutionFlags(CommandBufferExecutionFlags.AsyncCompute);
                    UnityEngine.Debug.Log("[DeepFillV2] execution=long-command-buffer | queue=" + (useAsyncCompute ? "async-compute" : "graphics"));

                    ComputeTexture sourceInput = null;
                    ComputeTexture maskInput = null;
                    try
                    {
                        ReportProgress(0.38f, "Pack source");
                        sourceInput = _repro.RentTempArray(commandBuffer, InputWidth, InputHeight, 1, inputFormat);
                        _ops.PackRgbToPack4(commandBuffer, source512, 0, 0, 1f, 1f, sourceInput, flipYInput, 1f);
                        ReportProgress(0.46f, "Pack mask");
                        maskInput = _repro.RentTempArray(commandBuffer, InputWidth, InputHeight, 1, inputFormat);
                        _ops.PackRgbToPack4(commandBuffer, mask512, 0, 0, 1f, 1f, maskInput, flipYInput, 1f);

                        var commandInputs = new Dictionary<string, ComputeTexture>(StringComparer.Ordinal)
                        {
                            [MaskInputName] = maskInput,
                            [ImageInputName] = sourceInput
                        };
                        using (var infer = _repro.ForwardPack4Outputs(
                            commandBuffer,
                            commandInputs,
                            textureShapes,
                            new[] { OutputName },
                            pinned))
                        {
                            var output = infer.GetTexture(OutputName);
                            commandOutputPack4 = CreateCommandBufferReadbackTexture(output);
                            CopyTextureArrayAllSlices(commandBuffer, output, commandOutputPack4);
                        }

                        _repro.ReturnTempArray(commandBuffer, sourceInput);
                        sourceInput = null;
                        _repro.ReturnTempArray(commandBuffer, maskInput);
                        maskInput = null;

                        if (useAsyncCompute)
                        {
                            var completionFence = commandBuffer.CreateGraphicsFence(
                                GraphicsFenceType.AsyncQueueSynchronisation,
                                SynchronisationStageFlags.AllGPUOperations);
                            Graphics.ExecuteCommandBufferAsync(commandBuffer, ComputeQueueType.Default);
                            Graphics.WaitOnAsyncGraphicsFence(completionFence);
                        }
                        else
                        {
                            Graphics.ExecuteCommandBuffer(commandBuffer);
                        }

                        var outPack4 = commandOutputPack4;
                        rgb512 = NewRenderTexture(InputWidth, InputHeight, RenderTextureFormat.ARGB32, true, "DeepFillV2.Rgb400x512");
                        ReportProgress(0.86f, "Unpack output");
                        _ops.Pack4ToRgbScaled(outPack4, rgb512, 0.5f, 0.5f, flipYOutput);
                        ReportProgress(0.92f, "Read output");
                        generated512 = await ReadRenderTexturePacedAsync(rgb512, TextureFormat.RGBA32, true, ct);
                    }
                    finally
                    {
                        if (sourceInput != null)
                            _repro.ReturnTempArray(commandBuffer, sourceInput);
                        if (maskInput != null)
                            _repro.ReturnTempArray(commandBuffer, maskInput);
                    }

                }
            }
            inferSw.Stop();
            inferMs = inferSw.ElapsedMilliseconds;

            if (generated512 == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 output readback failed." });

            ReportProgress(0.96f, "Restore image size");
            generatedFull = await RestoreModelOutputPacedAsync(generated512, readableSource, ct);
            if (generatedFull == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 failed to resize output to source size." });

            ReportProgress(0.98f, "Composite output");
            var finalTexture = preserveUnmaskedPixels
                ? await CompositeMaskedPacedAsync(readableSource, generatedFull, readableMask, ct)
                : generatedFull;
            if (preserveUnmaskedPixels)
                DestroyRuntimeObject(generatedFull);
            if (finalTexture == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 final composite failed." });

            var displayTexture = CopyTexture(finalTexture, false);
            DestroyRuntimeObject(finalTexture);
            if (ReferenceEquals(finalTexture, generatedFull))
                generatedFull = null;
            if (displayTexture == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 failed to create display output." });

            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                TryWriteTexturePng(generated512, Path.Combine(_lastDumpDir, "02_deepfillv2_400x512.png"));
                TryWriteTexturePng(displayTexture, Path.Combine(_lastDumpDir, "03_deepfillv2_final.png"));
                WriteSummary(Path.Combine(_lastDumpDir, "summary.txt"), readableSource, readableMask, displayTexture, loadMs, inferMs);
            }

            ReportProgress(1f, "Done");
            return Finish(new DeepFillV2Result { texture = displayTexture });
        }
        catch (OperationCanceledException)
        {
            return Finish(new DeepFillV2Result { error = "Cancelled" });
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("[DeepFillV2] ProcessAsync failed: " + e);
            return Finish(new DeepFillV2Result { error = e.ToString() });
        }
        finally
        {
            if (sourcePack4 != null) _repro?.ReturnTempArray(sourcePack4);
            if (maskPack4 != null) _repro?.ReturnTempArray(maskPack4);
            if (commandOutputPack4 != null) RenderTexture.ReleaseTemporary(commandOutputPack4);
            ReleaseRenderTexture(source512);
            ReleaseRenderTexture(mask512);
            ReleaseRenderTexture(rgb512);
            if (generated512 != null) DestroyRuntimeObject(generated512);
            if (modelSource != null) DestroyRuntimeObject(modelSource);
            if (modelMask != null) DestroyRuntimeObject(modelMask);
            if (readableSource != null && !ReferenceEquals(readableSource, sourceImage)) DestroyRuntimeObject(readableSource);
            if (readableMask != null && !ReferenceEquals(readableMask, maskImage)) DestroyRuntimeObject(readableMask);
        }
    }

    private void EnsureRuntimeObjects()
    {
        if (_repro != null && _hasAppliedPrecisionMode && _appliedPrecisionMode != precisionMode)
            Release();
        _ops ??= new AexisOps();
        if (_repro == null)
        {
            _repro = NcnnInferenceSessionFactory.Create(_ops, "deepfillv2", precisionMode);
            _appliedPrecisionMode = precisionMode;
            _hasAppliedPrecisionMode = true;
        }
    }

    private void ApplyReproOptions()
    {
        if (_repro == null)
            return;
        _repro.EnableGeneralTextureConvolution = enableGeneralTextureConvolution;
        _repro.EnableDepthWiseTextureConvolution = enableDepthWiseTextureConvolution;
        _repro.EnableConv1x1TextureConvolution = enableConv1x1TextureConvolution;
        _repro.TensorTextureFormat = useArgbFloatTensor ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf;
        _repro.DisallowBufferAccess = true;
        _repro.DisallowBufferOutputs = true;
        _repro.DisallowBufferToTextureMaterialization = true;
        _repro.DisallowInferenceTempComputeBuffers = true;
        _repro.DebugLogAllLayerHeartbeats = enableLayerPathDebugLog;
        _repro.DebugLog = enableLayerPathDebugLog ? line => UnityEngine.Debug.Log("[DeepFillV2][Layer] " + line) : null;
    }

    private async UniTask EnsureLoadedAsync(CancellationToken ct)
    {
        var effectiveBackend = ResolveAvailableBackend(out var paramPath, out var binPath, out var onnxPath);
        _effectiveBackend = effectiveBackend;
        var signature = BuildLoadSignature(paramPath);
        if (effectiveBackend == DeepFillV2Backend.OnnxDirect)
            signature += "|" + BuildLoadSignature(onnxPath);
        else
            signature += "|" + BuildLoadSignature(binPath);

        if (_hasLoaded && _loadedBackend == effectiveBackend && string.Equals(_loadedSignature, signature, StringComparison.Ordinal))
            return;

        if (_repro == null)
            EnsureRuntimeObjects();
        _repro.Release();
        _lastOnnxReport = null;

        if (effectiveBackend == DeepFillV2Backend.OnnxDirect)
        {
            var imported = DeepFillV2OnnxNcnnImporter.Import(onnxPath, paramPath);
            _lastOnnxReport = imported.report;
            using (var ms = new MemoryStream(imported.ncnnBinBytes, false))
            using (var br = new NcnnBinReader(ms))
                await _repro.LoadModelAsync(imported.paramText, br, progress => LogLoadProgress("onnx-direct", progress), ct);
        }
        else
        {
            var paramText = File.ReadAllText(paramPath);
            using (var fs = File.OpenRead(binPath))
            using (var br = new NcnnBinReader(fs))
                await _repro.LoadModelAsync(paramText, br, progress => LogLoadProgress("ncnn-bin", progress), ct);
        }

        _loadedBackend = effectiveBackend;
        _loadedSignature = signature;
        _hasLoaded = true;
    }

    // A packaged model may provide either the NCNN payload or its source ONNX.
    // Prefer the selected backend when it exists, then fall back to the available
    // representation so old serialized scenes keep working after payload trimming.
    private DeepFillV2Backend ResolveAvailableBackend(out string paramPath, out string binPath, out string onnxPath)
    {
        paramPath = ResolveStreamingAssetPath(ncnnParamRelativePath);
        binPath = ResolveStreamingAssetPath(ncnnBinRelativePath);
        onnxPath = ResolveStreamingAssetPath(sourceOnnxRelativePath);

        var hasParam = File.Exists(paramPath);
        var hasNcnn = hasParam && File.Exists(binPath);
        var hasOnnx = hasParam && File.Exists(onnxPath);
        if (backend == DeepFillV2Backend.NcnnBin && hasNcnn)
            return DeepFillV2Backend.NcnnBin;
        if (backend == DeepFillV2Backend.OnnxDirect && hasOnnx)
            return DeepFillV2Backend.OnnxDirect;
        if (hasNcnn)
            return DeepFillV2Backend.NcnnBin;
        if (hasOnnx)
            return DeepFillV2Backend.OnnxDirect;

        throw new FileNotFoundException(
            "DeepFillV2 requires either NCNN .param + .bin or source .onnx + .param."
            + " param=" + paramPath
            + " bin=" + binPath
            + " onnx=" + onnxPath);
    }

    private static string ResolveStreamingAssetPath(string relativePath)
    {
        if (string.IsNullOrWhiteSpace(relativePath))
            return Aexis.Samples.AexisSampleStreamingAssets.ResolveDirectoryPath();
        if (Path.IsPathRooted(relativePath))
            return relativePath;
        var rel = relativePath.Replace('\\', '/').TrimStart('/');
        return Aexis.Samples.AexisSampleStreamingAssets.TryResolveFilePath(rel, out var path)
            ? path
            : null;
    }

    private static string BuildLoadSignature(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return path ?? string.Empty;
        var info = new FileInfo(path);
        return info.FullName + "|" + info.Length.ToString(CultureInfo.InvariantCulture) + "|" + info.LastWriteTimeUtc.Ticks.ToString(CultureInfo.InvariantCulture);
    }

    private static void TryWriteDebugTensorBlob(AexisGraphSession.InferResult infer, string blobName, string dumpDir)
    {
        if (infer == null || string.IsNullOrWhiteSpace(blobName) || string.IsNullOrWhiteSpace(dumpDir))
            return;
        try
        {
            var data = infer.ReadTextureDataForOutput(blobName);
            if (data == null)
                return;

            infer.TryGetLogicalShape(blobName, out var dims, out var w, out var h, out var d, out var c);
            var safeName = MakeSafeFileName(blobName);
            var tensorPath = Path.Combine(dumpDir, "tensor_" + safeName + ".f32");
            using (var fs = File.Create(tensorPath))
            using (var bw = new BinaryWriter(fs))
            {
                for (var i = 0; i < data.Length; i++)
                    bw.Write(data[i]);
            }

            var shapePath = Path.Combine(dumpDir, "tensor_" + safeName + ".shape.txt");
            File.WriteAllText(
                shapePath,
                "blob=" + blobName + Environment.NewLine
                + "dims=" + dims.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                + "w=" + w.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                + "h=" + h.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                + "d=" + d.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                + "c=" + c.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                + "count=" + data.Length.ToString(CultureInfo.InvariantCulture) + Environment.NewLine,
                System.Text.Encoding.UTF8);
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogWarning("[DeepFillV2] Failed to dump debug tensor blob " + blobName + ": " + e.Message);
        }
    }

    private static List<string> ParseDebugTensorBlobNames(string value)
    {
        var result = new List<string>();
        if (string.IsNullOrWhiteSpace(value))
            return result;
        var tokens = value.Split(new[] { ',', ';', ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        for (var i = 0; i < tokens.Length; i++)
        {
            var token = tokens[i].Trim();
            if (token.Length == 0 || result.Contains(token))
                continue;
            result.Add(token);
        }
        return result;
    }

    private static string MakeSafeFileName(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "blob";
        var chars = value.ToCharArray();
        var invalid = Path.GetInvalidFileNameChars();
        for (var i = 0; i < chars.Length; i++)
        {
            if (Array.IndexOf(invalid, chars[i]) >= 0 || chars[i] == ':' || chars[i] == '/' || chars[i] == '\\')
                chars[i] = '_';
        }
        return new string(chars);
    }

    private void LogLoadProgress(string label, AexisGraphSession.LoadProgress progress)
    {
        if (progress.stage == "complete")
            UnityEngine.Debug.Log("[DeepFillV2] LoadModel complete | backend=" + label + " | layers=" + progress.layerCount.ToString(CultureInfo.InvariantCulture));
    }

    private void ReportProgress(float progress, string stage)
    {
        ProgressChanged?.Invoke(Mathf.Clamp01(progress), stage ?? string.Empty);
    }

    private static RenderTexture CreateCommandBufferReadbackTexture(ComputeTexture source)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));

        var descriptor = new RenderTextureDescriptor(source.width, source.height, source.format, 0)
        {
            dimension = source.dimension,
            volumeDepth = source.dimension == TextureDimension.Tex2D ? 1 : Mathf.Max(1, source.depth),
            msaaSamples = 1,
            enableRandomWrite = false
        };
        return RenderTexture.GetTemporary(descriptor);
    }

    private static void CopyTextureArrayAllSlices(CommandBuffer commandBuffer, ComputeTexture source, RenderTexture target)
    {
        if (commandBuffer == null)
            throw new ArgumentNullException(nameof(commandBuffer));
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (target == null)
            throw new ArgumentNullException(nameof(target));

        var depth = source.dimension == TextureDimension.Tex2D ? 1 : Mathf.Max(1, source.depth);
        for (var slice = 0; slice < depth; slice++)
            commandBuffer.CopyTexture(source.nameID, slice, 0, target, slice, 0);
    }

    private string BuildModelReport()
    {
        if (_effectiveBackend != DeepFillV2Backend.OnnxDirect || _lastOnnxReport == null)
            return "backend=" + _effectiveBackend;
        return "backend=OnnxDirect"
               + " onnxBytes=" + _lastOnnxReport.onnxBytes.ToString(CultureInfo.InvariantCulture)
               + " generatedBinBytes=" + _lastOnnxReport.generatedBinBytes.ToString(CultureInfo.InvariantCulture)
               + " conv=" + _lastOnnxReport.convNodeCount.ToString(CultureInfo.InvariantCulture)
               + " extractImagePatches=" + _lastOnnxReport.extractImagePatchesNodeCount.ToString(CultureInfo.InvariantCulture)
               + " sha256=" + _lastOnnxReport.generatedBinSha256;
    }

    private static Texture2D EnsureReadable(Texture texture)
    {
        if (texture == null)
            return null;
        if (texture is Texture2D tex2d && tex2d.isReadable)
            return tex2d;
        var rt = RenderTexture.GetTemporary(Mathf.Max(1, texture.width), Mathf.Max(1, texture.height), 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
        try
        {
            Graphics.Blit(texture, rt);
            return ReadRenderTexture(rt, TextureFormat.RGBA32, true);
        }
        finally
        {
            if (RenderTexture.active == rt)
                RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);
        }
    }

    private static RenderTexture PrepareModelInput(Texture source)
    {
        if (source == null)
            return null;
        return ResizeTexture(source, InputWidth, InputHeight, RenderTextureFormat.ARGB32);
    }

    private static Texture2D RestoreModelOutput(Texture2D modelOutput, Texture2D original)
    {
        if (modelOutput == null || original == null)
            return null;
        return ResizeTextureToTexture2D(modelOutput, original.width, original.height);
    }

    private static async UniTask<Texture2D> RestoreModelOutputPacedAsync(Texture2D modelOutput, Texture2D original, CancellationToken ct)
    {
        if (modelOutput == null || original == null)
            return null;

        var rt = ResizeTexture(modelOutput, original.width, original.height, RenderTextureFormat.ARGB32);
        if (rt == null)
            return null;
        try
        {
            return await ReadRenderTexturePacedAsync(rt, TextureFormat.RGBA32, true, ct);
        }
        finally
        {
            ReleaseRenderTexture(rt);
        }
    }

    private static RenderTexture ResizeTexture(Texture source, int width, int height, RenderTextureFormat format)
    {
        if (source == null)
            return null;
        var rt = NewRenderTexture(width, height, format, false, "DeepFillV2.Resize");
        Graphics.Blit(source, rt);
        return rt;
    }

    private static Texture2D ResizeTextureToTexture2D(Texture source, int width, int height)
    {
        var rt = ResizeTexture(source, width, height, RenderTextureFormat.ARGB32);
        if (rt == null)
            return null;
        try
        {
            return ReadRenderTexture(rt, TextureFormat.RGBA32, false);
        }
        finally
        {
            ReleaseRenderTexture(rt);
        }
    }

    private static RenderTexture NewRenderTexture(int width, int height, RenderTextureFormat format, bool randomWrite, string name)
    {
        var rt = new RenderTexture(Mathf.Max(1, width), Mathf.Max(1, height), 0, format, RenderTextureReadWrite.Linear)
        {
            enableRandomWrite = randomWrite,
            useMipMap = false,
            autoGenerateMips = false,
            name = name ?? "DeepFillV2.RT"
        };
        rt.Create();
        return rt;
    }

    private static Texture2D ReadRenderTexture(RenderTexture rt, TextureFormat format, bool linear)
    {
        if (rt == null)
            return null;
        var prev = RenderTexture.active;
        try
        {
            RenderTexture.active = rt;
            var tex = new Texture2D(rt.width, rt.height, format, false, linear);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0, false);
            tex.Apply(false, false);
            return tex;
        }
        finally
        {
            RenderTexture.active = prev;
        }
    }

    private static Texture2D CopyTexture(Texture2D source, bool linear)
    {
        if (source == null)
            return null;

        var copy = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false, linear);
        copy.SetPixels32(source.GetPixels32());
        copy.Apply(false, false);
        copy.wrapMode = TextureWrapMode.Clamp;
        copy.filterMode = FilterMode.Bilinear;
        return copy;
    }

    private static UniTask<Texture2D> ReadRenderTexturePacedAsync(RenderTexture rt, TextureFormat format, bool linear, CancellationToken ct)
    {
        if (rt == null)
            return UniTask.FromResult<Texture2D>(null);

        ct.ThrowIfCancellationRequested();
        var result = ReadRenderTexture(rt, format, linear);
        return UniTask.FromResult(result);
    }

    private static async UniTask<Texture2D> CompositeMaskedPacedAsync(
        Texture2D original,
        Texture2D candidate,
        Texture2D mask,
        CancellationToken ct)
    {
        if (original == null || candidate == null || mask == null)
            return null;
        Texture2D originalSizedMask = mask;
        if (mask.width != original.width || mask.height != original.height)
        {
            var resizedMask = ResizeTexture(mask, original.width, original.height, RenderTextureFormat.ARGB32);
            if (resizedMask == null)
                return null;
            try
            {
                // The DeepFill graph has just completed a long sequence of Pack4
                // dispatches. Do not synchronously read a fresh Blit on the same
                // graphics queue; tile renderers can otherwise submit a circular
                // wait. This is the same platform-neutral pacing used by the model
                // input and output conversions.
                originalSizedMask = await ReadRenderTexturePacedAsync(
                    resizedMask,
                    TextureFormat.RGBA32,
                    false,
                    ct);
            }
            finally
            {
                ReleaseRenderTexture(resizedMask);
            }
            if (originalSizedMask == null)
                return null;
        }
        try
        {
            var src = original.GetPixels32();
            var gen = candidate.GetPixels32();
            var maskPixels = originalSizedMask.GetPixels32();
            var count = Mathf.Min(src.Length, Mathf.Min(gen.Length, maskPixels.Length));
            var dst = new Color32[src.Length];
            for (var i = 0; i < count; i++)
            {
                var m = Mathf.Max(maskPixels[i].r, Mathf.Max(maskPixels[i].g, maskPixels[i].b)) / 255f;
                dst[i] = Lerp(src[i], gen[i], m);
            }
            for (var i = count; i < dst.Length; i++)
                dst[i] = src[i];
            var tex = new Texture2D(original.width, original.height, TextureFormat.RGBA32, false);
            tex.SetPixels32(dst);
            tex.Apply(false, false);
            return tex;
        }
        finally
        {
            if (originalSizedMask != null && !ReferenceEquals(originalSizedMask, mask))
                DestroyRuntimeObject(originalSizedMask);
        }
    }

    private static Color32 Lerp(Color32 a, Color32 b, float t)
    {
        t = Mathf.Clamp01(t);
        return new Color32(
            (byte)Mathf.RoundToInt(a.r + (b.r - a.r) * t),
            (byte)Mathf.RoundToInt(a.g + (b.g - a.g) * t),
            (byte)Mathf.RoundToInt(a.b + (b.b - a.b) * t),
            255);
    }

    private static string CreateDumpDir()
    {
        var root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "YanQi", "AIImage");
        var dir = Path.Combine(root, "AIImage_DeepFillV2_" + DateTime.Now.ToString("yyyyMMdd-HHmmss-fff", CultureInfo.InvariantCulture));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private void WriteSummary(string path, Texture2D source, Texture2D mask, Texture2D result, long loadMs, long inferMs)
    {
        var coverage = ComputeMaskCoverage(mask);
        var diff = ComputeMaskedMeanAbsDiff(source, result, mask, out var maskedPixels);
        File.WriteAllText(
            path,
            "backend=" + backend + Environment.NewLine
            + "load_ms=" + loadMs.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
            + "inference_ms=" + inferMs.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
            + "mask_coverage=" + coverage.ToString("0.000000", CultureInfo.InvariantCulture) + Environment.NewLine
            + "masked_pixels=" + maskedPixels.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
            + "masked_mean_abs_diff_rgb=" + diff.ToString("0.000000", CultureInfo.InvariantCulture) + Environment.NewLine
            + "model_report=" + BuildModelReport() + Environment.NewLine);
    }

    public static float ComputeMaskedMeanAbsDiff(Texture2D source, Texture2D candidate, Texture2D mask, out int maskedPixels)
    {
        maskedPixels = 0;
        if (source == null || candidate == null || mask == null)
            return 0f;
        var s = source.GetPixels32();
        var c = candidate.GetPixels32();
        var m = mask.width == source.width && mask.height == source.height
            ? mask.GetPixels32()
            : ResizeTextureToTexture2D(mask, source.width, source.height)?.GetPixels32();
        if (m == null)
            return 0f;
        var count = Mathf.Min(s.Length, Mathf.Min(c.Length, m.Length));
        double sum = 0;
        for (var i = 0; i < count; i++)
        {
            var include = Mathf.Max(m[i].r, Mathf.Max(m[i].g, m[i].b)) >= 128;
            if (!include)
                continue;
            sum += (Math.Abs(s[i].r - c[i].r) + Math.Abs(s[i].g - c[i].g) + Math.Abs(s[i].b - c[i].b)) / (3.0 * 255.0);
            maskedPixels++;
        }
        return maskedPixels > 0 ? (float)(sum / maskedPixels) : 0f;
    }

    public static float ComputeMaskCoverage(Texture2D mask)
    {
        if (mask == null)
            return 0f;
        var pixels = mask.GetPixels32();
        if (pixels == null || pixels.Length == 0)
            return 0f;
        var count = 0;
        for (var i = 0; i < pixels.Length; i++)
        {
            if (Mathf.Max(pixels[i].r, Mathf.Max(pixels[i].g, pixels[i].b)) >= 128)
                count++;
        }
        return count / (float)pixels.Length;
    }

    private static void TryWriteTexturePng(Texture texture, string path)
    {
        if (texture == null || string.IsNullOrWhiteSpace(path))
            return;
        Texture2D tex = null;
        try
        {
            tex = texture as Texture2D ?? ResizeTextureToTexture2D(texture, texture.width, texture.height);
            if (tex == null)
                return;
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
            File.WriteAllBytes(path, tex.EncodeToPNG());
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogWarning("[DeepFillV2] Failed to write debug PNG: " + e.Message);
        }
        finally
        {
            if (tex != null && !ReferenceEquals(tex, texture))
                DestroyRuntimeObject(tex);
        }
    }

    private static void ReleaseRenderTexture(RenderTexture rt)
    {
        if (rt == null)
            return;
        if (RenderTexture.active == rt)
            RenderTexture.active = null;
        try { rt.Release(); } catch { }
        DestroyRuntimeObject(rt);
    }

    private static void DestroyRuntimeObject(UnityEngine.Object obj)
    {
        if (obj == null)
            return;
        if (Application.isPlaying)
            Destroy(obj);
        else
            DestroyImmediate(obj);
    }

    private void OnDestroy()
    {
        Release();
    }

    public void Release()
    {
        try { _repro?.Release(); } catch { }
        _repro = null;
        _hasLoaded = false;
    }
}
