using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using Cysharp.Threading.Tasks;
using ICSharpCode.SharpZipLib.GZip;
using NcnnCompute;
using Newtonsoft.Json.Linq;
using UnityEngine;

public enum MonaiInputSourceKind
{
    BaselineTensorDump,
    MedicalVolumeFiles
}

public enum MonaiChannelFillMode
{
    DuplicateFirst,
    DuplicateLast,
    Zero
}

public enum MonaiPostprocessKind
{
    None,
    BratsTumorSubregions
}

public sealed class MonaiRunRequest
{
    public MonaiInputSourceKind inputSource = MonaiInputSourceKind.BaselineTensorDump;
    public string baselineManifestPath;
    public string[] inputVolumePaths;
    public string caseName;
    public string outputDir;
    public string modelParamPath;
    public string modelBinPath;
    public string pnnxParamPath;
    public string bundleManifestPath;
    public string inputBlobName = "in0";
    public string outputBlobName = "out0";
    public int inputChannels;
    public int inputDepth;
    public int inputHeight;
    public int inputWidth;
    public float threshold = 0.5f;
    public bool normalizeNonZero = true;
    public MonaiChannelFillMode channelFillMode = MonaiChannelFillMode.DuplicateFirst;
    public MonaiPostprocessKind postprocessKind = MonaiPostprocessKind.BratsTumorSubregions;
    public bool compareWithBaseline = true;
}

public struct MonaiSegmentationResult
{
    public string caseName;
    public string outputDir;
    public string summaryText;
    public string error;
    public long elapsedMs;
}

public sealed class MONAINcnnReproRunner : MonoBehaviour
{
    private const string DefaultModelParamRelativePath = @"Tools\MonaiToNCNN\outputs\brats_mri_segmentation\brats_mri_segmentation.param";
    private const string DefaultModelBinRelativePath = @"Tools\MonaiToNCNN\outputs\brats_mri_segmentation\brats_mri_segmentation.bin";
    private const string DefaultPnnxParamRelativePath = @"Tools\MonaiToNCNN\outputs\brats_mri_segmentation\brats_mri_segmentation.sim.pnnx.param";
    private const string DefaultBundleManifestRelativePath = @"Tools\MonaiToNCNN\outputs\brats_mri_segmentation\manifest.json";
    private const string DefaultBaselineManifestRelativePath = @"Tools\MonaiToNCNN\manual_test\brats_mri_segmentation_baseline\RegLib_C01_1\baseline_manifest.json";
    private const string DefaultVolumeInputPath = @"E:\Projects\CTData\sliceexampledata2\MRBrainTumor1\RegLib_C01_1.nrrd";
    private const string DefaultInputBlobName = "in0";
    private const string DefaultOutputBlobName = "out0";

    [Serializable]
    private sealed class VolumeInfoRecord
    {
        public string path;
        public int dim0;
        public int dim1;
        public int dim2;
        public string format;
        public float[] spacing;
    }

    private sealed class MonaiVolumeData
    {
        public string path;
        public int dim0;
        public int dim1;
        public int dim2;
        public float[] data;
        public float[] spacing;
        public string sourceFormat;
    }

    private sealed class PreparedInput
    {
        public string caseName;
        public float[] tensorNcdhw;
        public string[] sourcePaths;
        public List<VolumeInfoRecord> volumes;
        public string baselineCaseDir;
        public JObject baselineManifest;
        public string preparationNote;
    }

    private sealed class ResolvedRequest
    {
        public MonaiRunRequest source;
        public string modelParamPath;
        public string modelBinPath;
        public string pnnxParamPath;
        public string bundleManifestPath;
        public string baselineManifestPath;
        public string[] inputVolumePaths;
        public string outputDir;
        public string inputBlobName;
        public string outputBlobName;
        public string caseName;
        public int inputChannels;
        public int inputDepth;
        public int inputHeight;
        public int inputWidth;
        public float threshold;
        public bool normalizeNonZero;
        public MonaiChannelFillMode channelFillMode;
        public MonaiPostprocessKind postprocessKind;
        public bool compareWithBaseline;
        public JObject bundleManifest;
        public JObject baselineManifest;
    }

    public string modelParamRelativePath = DefaultModelParamRelativePath;
    public string modelBinRelativePath = DefaultModelBinRelativePath;
    public string pnnxParamRelativePath = DefaultPnnxParamRelativePath;
    public string bundleManifestRelativePath = DefaultBundleManifestRelativePath;
    public string defaultBaselineManifestRelativePath = DefaultBaselineManifestRelativePath;
    public string defaultInputVolumePaths = DefaultVolumeInputPath;
    public string inputBlobName = DefaultInputBlobName;
    public string outputBlobName = DefaultOutputBlobName;
    public MonaiPostprocessKind defaultPostprocessKind = MonaiPostprocessKind.BratsTumorSubregions;
    public MonaiChannelFillMode defaultChannelFillMode = MonaiChannelFillMode.DuplicateFirst;
    public float defaultThreshold = 0.5f;
    public bool defaultNormalizeNonZero = true;
    public bool enableDebugDump = true;
    public bool enableBaselineCompare = true;
    public bool enableTempPool = true;
    public int maxPooledPerShape = 2;
    public bool forceBufferConvolution = false;
    public bool keepRawConvWeightsForTexturePath = true;
    public bool logAllLayerHeartbeats = false;
    public bool logAllLayerOutputs = false;
    public bool logAllBufferMaterialize = false;
    public bool enableLayerRuntimeProfile = false;
    public bool syncLayerRuntimeProfile = false;

    public event Action<float, string> ProgressChanged;

    private NcnnOps _ops;
    private NcnnRepro _repro;
    private string _loadedModelKey;
    private string _lastDumpDir;
    private string _lastSummaryText;
    private readonly List<string> _debugLines = new List<string>();

    public string LastDumpDir => _lastDumpDir;
    public string LastSummaryText => _lastSummaryText;

    private void Awake()
    {
        EnsureRuntimeObjects();
    }

    private void OnDestroy()
    {
        Release();
    }

    public async UniTask<MonaiSegmentationResult> ProcessDefaultBaselineAsync(CancellationToken ct)
    {
        return await ProcessAsync(new MonaiRunRequest
        {
            inputSource = MonaiInputSourceKind.BaselineTensorDump,
            baselineManifestPath = defaultBaselineManifestRelativePath,
            compareWithBaseline = enableBaselineCompare,
            postprocessKind = defaultPostprocessKind,
            channelFillMode = defaultChannelFillMode,
            threshold = defaultThreshold,
            normalizeNonZero = defaultNormalizeNonZero
        }, ct);
    }

    public async UniTask<MonaiSegmentationResult> ProcessDefaultMedicalAsync(CancellationToken ct)
    {
        return await ProcessAsync(new MonaiRunRequest
        {
            inputSource = MonaiInputSourceKind.MedicalVolumeFiles,
            baselineManifestPath = defaultBaselineManifestRelativePath,
            inputVolumePaths = SplitInputPaths(defaultInputVolumePaths),
            compareWithBaseline = enableBaselineCompare,
            postprocessKind = defaultPostprocessKind,
            channelFillMode = defaultChannelFillMode,
            threshold = defaultThreshold,
            normalizeNonZero = defaultNormalizeNonZero
        }, ct);
    }

    public async UniTask<MonaiSegmentationResult> ProcessAsync(MonaiRunRequest request, CancellationToken ct)
    {
        if (request == null)
            return default;

        var totalSw = Stopwatch.StartNew();
        _lastDumpDir = null;
        _lastSummaryText = null;
        _debugLines.Clear();

        MonaiSegmentationResult Finish(MonaiSegmentationResult result)
        {
            result.elapsedMs = totalSw.ElapsedMilliseconds;
            return result;
        }

        try
        {
            EnsureRuntimeObjects();
            ApplyReproOptions();
            var resolved = ResolveRequest(request);
            ct.ThrowIfCancellationRequested();

            _lastDumpDir = resolved.outputDir;
            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                Directory.CreateDirectory(_lastDumpDir);

            ReportProgress(0.06f, "Load MONAI ncnn model");
            await EnsureLoadedAsync(resolved, ct);
            ct.ThrowIfCancellationRequested();

            ReportProgress(0.18f, "Prepare input tensor");
            var prepared = await PrepareInputAsync(resolved, ct);
            if (prepared == null || prepared.tensorNcdhw == null || prepared.tensorNcdhw.Length == 0)
                return Finish(new MonaiSegmentationResult { error = "MONAI input tensor is empty" });

            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                WriteFloatArray(Path.Combine(_lastDumpDir, "input_tensor_ncdhw_f32.bin"), prepared.tensorNcdhw);
                WriteText(Path.Combine(_lastDumpDir, "input_preparation.txt"), BuildInputPreparationText(resolved, prepared));
            }

            ct.ThrowIfCancellationRequested();
            ReportProgress(0.42f, "Run MONAI inference");

            float[] logits;
            int outW;
            int outH;
            int outD;
            int outC;
            using (var inputTensor = new NcnnTensorBuffer(resolved.inputWidth, resolved.inputHeight, resolved.inputDepth, resolved.inputChannels))
            {
                inputTensor.buffer.SetData(prepared.tensorNcdhw);
                using var infer = _repro.InferWithMultiInputs(
                    null,
                    new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal) { { resolved.inputBlobName, inputTensor } },
                    null,
                    null);

                var outputView = infer.GetBufferView(resolved.outputBlobName);
                if (outputView == null || outputView.buffer == null)
                    return Finish(new MonaiSegmentationResult { error = "MONAI output blob missing: " + resolved.outputBlobName });
                if (outputView.dims != 4)
                    return Finish(new MonaiSegmentationResult { error = "MONAI output dims expected 4 but got " + outputView.dims });

                logits = infer.GetBufferData(resolved.outputBlobName);
                outW = outputView.w;
                outH = outputView.h;
                outD = outputView.d;
                outC = outputView.c;
            }

            if (logits == null || logits.Length == 0)
                return Finish(new MonaiSegmentationResult { error = "MONAI output logits are empty" });

            ct.ThrowIfCancellationRequested();
            ReportProgress(0.70f, "Postprocess output");

            var probs = new float[logits.Length];
            var masks = new byte[logits.Length];
            var voxelCount = checked(outW * outH * outD);
            for (var i = 0; i < logits.Length; i++)
            {
                var p = Sigmoid(logits[i]);
                probs[i] = p;
                masks[i] = p >= resolved.threshold ? (byte)1 : (byte)0;
            }

            byte[] labelMap = null;
            if (resolved.postprocessKind == MonaiPostprocessKind.BratsTumorSubregions)
            {
                if (outC != 3)
                    return Finish(new MonaiSegmentationResult { error = "BraTS postprocess expects 3 output channels but got " + outC });
                labelMap = BuildBratsLabelMap(masks, outW, outH, outD);
            }

            ReportProgress(0.82f, "Write dumps and compare");
            var comparison = resolved.compareWithBaseline && prepared.baselineManifest != null
                ? BuildBaselineComparison(prepared.baselineManifest, prepared.baselineCaseDir, logits, probs, masks, labelMap)
                : null;

            var summary = BuildSummaryText(resolved, prepared, outW, outH, outD, outC, comparison, labelMap);
            _lastSummaryText = summary;

            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                Directory.CreateDirectory(_lastDumpDir);
                WriteFloatArray(Path.Combine(_lastDumpDir, "logits_ncdhw_f32.bin"), logits);
                WriteFloatArray(Path.Combine(_lastDumpDir, "probs_ncdhw_f32.bin"), probs);
                WriteByteArray(Path.Combine(_lastDumpDir, "masks_ncdhw_u8.bin"), masks);
                if (labelMap != null)
                    WriteByteArray(Path.Combine(_lastDumpDir, "labelmap_dhw_u8.bin"), labelMap);
                WriteText(Path.Combine(_lastDumpDir, "summary.txt"), summary);
                WriteText(Path.Combine(_lastDumpDir, "run_manifest.json"), BuildRunManifestJson(resolved, prepared, outW, outH, outD, outC, logits, probs, masks, labelMap, comparison).ToString());
                if (comparison != null)
                    WriteText(Path.Combine(_lastDumpDir, "baseline_compare.json"), comparison.ToString());
                if (_debugLines.Count > 0)
                    WriteText(Path.Combine(_lastDumpDir, "runtime_debug.log"), string.Join(Environment.NewLine, _debugLines));
            }

            ReportProgress(1f, string.Empty);
            return Finish(new MonaiSegmentationResult
            {
                caseName = prepared.caseName,
                outputDir = _lastDumpDir,
                summaryText = summary
            });
        }
        catch (OperationCanceledException)
        {
            FlushDebugLog();
            return Finish(new MonaiSegmentationResult { error = "Cancelled", outputDir = _lastDumpDir, summaryText = _lastSummaryText });
        }
        catch (Exception e)
        {
            AppendDebugLine("ProcessAsync failed | " + e);
            FlushDebugLog();
            return Finish(new MonaiSegmentationResult
            {
                error = e.Message,
                outputDir = _lastDumpDir,
                summaryText = _lastSummaryText
            });
        }
        finally
        {
            if (_repro != null)
            {
                _repro.DebugLog = null;
                _repro.ClearTempPool();
            }
            FlushDebugLog();
            ReportProgress(1f, string.Empty);
        }
    }

    private void EnsureRuntimeObjects()
    {
        _ops ??= new NcnnOps();
        _repro ??= new NcnnRepro(_ops);
    }

    private void ApplyReproOptions()
    {
        if (_repro == null)
            return;

        _repro.EnableTempPool = enableTempPool;
        _repro.MaxPooledPerShape = maxPooledPerShape;
        _repro.ForceBufferConvolution = forceBufferConvolution;
        _repro.KeepRawConvWeightsForTexturePath = keepRawConvWeightsForTexturePath;
        _repro.DebugLogAllLayerHeartbeats = logAllLayerHeartbeats;
        _repro.DebugLogAllLayerOutputs = logAllLayerOutputs;
        _repro.DebugLogAllBufferMaterialize = logAllBufferMaterialize;
        _repro.LayerRuntimeProfileEnabled = enableLayerRuntimeProfile;
        _repro.LayerRuntimeProfileSyncGpu = syncLayerRuntimeProfile;
        _repro.DebugLog = AppendDebugLine;
    }

    private async UniTask EnsureLoadedAsync(ResolvedRequest request, CancellationToken ct)
    {
        if (request == null)
            throw new ArgumentNullException(nameof(request));

        var modelKey = string.Join("|", request.modelParamPath ?? string.Empty, request.modelBinPath ?? string.Empty, request.pnnxParamPath ?? string.Empty);
        if (string.Equals(modelKey, _loadedModelKey, StringComparison.Ordinal) && _repro?.Model != null)
        {
            AppendDebugLine("EnsureLoadedAsync reuse cached model");
            return;
        }

        if (!File.Exists(request.modelParamPath))
            throw new FileNotFoundException("MONAI ncnn param not found", request.modelParamPath);
        if (!File.Exists(request.modelBinPath))
            throw new FileNotFoundException("MONAI ncnn bin not found", request.modelBinPath);

        AppendDebugLine(
            "EnsureLoadedAsync begin"
            + " | param=" + request.modelParamPath
            + " | bin=" + request.modelBinPath
            + " | pnnx=" + (string.IsNullOrWhiteSpace(request.pnnxParamPath) ? "<none>" : request.pnnxParamPath));

        try
        {
            AppendDebugLine("EnsureLoadedAsync read param start");
            ct.ThrowIfCancellationRequested();
            var paramText = File.ReadAllText(request.modelParamPath, Encoding.UTF8);
            AppendDebugLine("EnsureLoadedAsync read param done | chars=" + paramText.Length.ToString(CultureInfo.InvariantCulture));

            string pnnxParamText = null;
            if (!string.IsNullOrWhiteSpace(request.pnnxParamPath) && File.Exists(request.pnnxParamPath))
            {
                AppendDebugLine("EnsureLoadedAsync read pnnx param start");
                ct.ThrowIfCancellationRequested();
                pnnxParamText = File.ReadAllText(request.pnnxParamPath, Encoding.UTF8);
                AppendDebugLine("EnsureLoadedAsync read pnnx param done | chars=" + pnnxParamText.Length.ToString(CultureInfo.InvariantCulture));
            }
            else
            {
                AppendDebugLine("EnsureLoadedAsync skip pnnx param");
            }

            AppendDebugLine("EnsureLoadedAsync read model bin start");
            ct.ThrowIfCancellationRequested();
            var binBytes = File.ReadAllBytes(request.modelBinPath);
            AppendDebugLine("EnsureLoadedAsync read model bin done | bytes=" + binBytes.Length.ToString(CultureInfo.InvariantCulture));

            using var ms = new MemoryStream(binBytes, false);
            using var br = new NcnnBinReader(ms);

            AppendDebugLine("EnsureLoadedAsync LoadModelAsync start");
            await _repro.LoadModelAsync(paramText, br, progress =>
            {
                if (progress.stage == null)
                    return;

                var shouldLog =
                    !string.Equals(progress.stage, "layer", StringComparison.Ordinal)
                    || progress.layerIndex <= 2
                    || progress.layerIndex >= progress.layerCount
                    || (progress.layerIndex % 4) == 0
                    || string.Equals(progress.layerType, "Convolution3D", StringComparison.Ordinal);
                if (!shouldLog)
                    return;

                AppendDebugLine(
                    "LoadModelAsync progress"
                    + " | stage=" + progress.stage
                    + " | layer=" + progress.layerIndex.ToString(CultureInfo.InvariantCulture)
                    + "/" + progress.layerCount.ToString(CultureInfo.InvariantCulture)
                    + " | name=" + (progress.layerName ?? string.Empty)
                    + " | type=" + (progress.layerType ?? string.Empty)
                    + " | progress01=" + progress.progress01.ToString("0.000", CultureInfo.InvariantCulture));
            }, ct, 4);

            var profile = _repro.LastLoadProfile;
            AppendDebugLine(
                "EnsureLoadedAsync LoadModelAsync done"
                + " | layers=" + (profile?.layerCount ?? 0).ToString(CultureInfo.InvariantCulture)
                + " | totalMs=" + (profile?.totalMs ?? 0).ToString(CultureInfo.InvariantCulture)
                + " | totalBytesRead=" + (profile?.totalBytesRead ?? 0).ToString(CultureInfo.InvariantCulture));

            if (!string.IsNullOrWhiteSpace(pnnxParamText))
            {
                AppendDebugLine("EnsureLoadedAsync MergePnnxStringParams start");
                var merged = _repro.MergePnnxStringParams(pnnxParamText, false);
                AppendDebugLine("EnsureLoadedAsync MergePnnxStringParams done | merged=" + merged.ToString(CultureInfo.InvariantCulture));
            }

            _loadedModelKey = modelKey;
            AppendDebugLine("EnsureLoadedAsync complete");
        }
        catch (Exception e)
        {
            AppendDebugLine("EnsureLoadedAsync failed | " + e);
            FlushDebugLog();
            throw;
        }
    }

    private ResolvedRequest ResolveRequest(MonaiRunRequest request)
    {
        var resolved = new ResolvedRequest
        {
            source = request,
            modelParamPath = ResolveProjectPath(string.IsNullOrWhiteSpace(request.modelParamPath) ? modelParamRelativePath : request.modelParamPath),
            modelBinPath = ResolveProjectPath(string.IsNullOrWhiteSpace(request.modelBinPath) ? modelBinRelativePath : request.modelBinPath),
            pnnxParamPath = ResolveProjectPath(string.IsNullOrWhiteSpace(request.pnnxParamPath) ? pnnxParamRelativePath : request.pnnxParamPath),
            bundleManifestPath = ResolveProjectPath(string.IsNullOrWhiteSpace(request.bundleManifestPath) ? bundleManifestRelativePath : request.bundleManifestPath),
            baselineManifestPath = ResolveProjectPath(string.IsNullOrWhiteSpace(request.baselineManifestPath) ? defaultBaselineManifestRelativePath : request.baselineManifestPath),
            inputVolumePaths = request.inputVolumePaths != null && request.inputVolumePaths.Length > 0 ? CloneArray(request.inputVolumePaths) : SplitInputPaths(defaultInputVolumePaths),
            inputBlobName = string.IsNullOrWhiteSpace(request.inputBlobName) ? inputBlobName : request.inputBlobName,
            outputBlobName = string.IsNullOrWhiteSpace(request.outputBlobName) ? outputBlobName : request.outputBlobName,
            threshold = request.threshold > 0f ? request.threshold : defaultThreshold,
            normalizeNonZero = request.normalizeNonZero,
            channelFillMode = request.channelFillMode,
            postprocessKind = request.postprocessKind,
            compareWithBaseline = request.compareWithBaseline
        };

        if (request.threshold <= 0f)
            resolved.threshold = defaultThreshold;
        if (request.channelFillMode == 0 && defaultChannelFillMode != 0 && string.IsNullOrWhiteSpace(request.baselineManifestPath))
            resolved.channelFillMode = defaultChannelFillMode;
        if (request.postprocessKind == 0 && defaultPostprocessKind != 0 && string.IsNullOrWhiteSpace(request.baselineManifestPath))
            resolved.postprocessKind = defaultPostprocessKind;

        if (!string.IsNullOrWhiteSpace(resolved.bundleManifestPath) && File.Exists(resolved.bundleManifestPath))
            resolved.bundleManifest = JObject.Parse(File.ReadAllText(resolved.bundleManifestPath, Encoding.UTF8));
        if (!string.IsNullOrWhiteSpace(resolved.baselineManifestPath) && File.Exists(resolved.baselineManifestPath))
            resolved.baselineManifest = JObject.Parse(File.ReadAllText(resolved.baselineManifestPath, Encoding.UTF8));

        resolved.inputChannels = request.inputChannels;
        resolved.inputDepth = request.inputDepth;
        resolved.inputHeight = request.inputHeight;
        resolved.inputWidth = request.inputWidth;

        if (resolved.baselineManifest != null)
        {
            if (resolved.inputChannels <= 0 || resolved.inputDepth <= 0 || resolved.inputHeight <= 0 || resolved.inputWidth <= 0)
            {
                var shape = ReadIntArray(resolved.baselineManifest["model_input_shape_ncdhw"]);
                if (shape != null && shape.Length >= 5)
                {
                    resolved.inputChannels = shape[1];
                    resolved.inputDepth = shape[2];
                    resolved.inputHeight = shape[3];
                    resolved.inputWidth = shape[4];
                }
            }

            var thresholdToken = resolved.baselineManifest["threshold"];
            if (thresholdToken != null && request.threshold <= 0f)
                resolved.threshold = thresholdToken.Value<float>();
        }

        if ((resolved.inputChannels <= 0 || resolved.inputDepth <= 0 || resolved.inputHeight <= 0 || resolved.inputWidth <= 0) && resolved.bundleManifest != null)
        {
            var bundleShape = ReadIntArray(resolved.bundleManifest["input_shape"]);
            if (bundleShape != null && bundleShape.Length >= 5)
            {
                resolved.inputChannels = bundleShape[1];
                resolved.inputDepth = bundleShape[2];
                resolved.inputHeight = bundleShape[3];
                resolved.inputWidth = bundleShape[4];
            }
        }

        if (resolved.inputChannels <= 0 || resolved.inputDepth <= 0 || resolved.inputHeight <= 0 || resolved.inputWidth <= 0)
            throw new InvalidOperationException("MONAI input shape is unresolved. Please provide bundle/baseline manifest or explicit shape.");

        resolved.caseName = !string.IsNullOrWhiteSpace(request.caseName)
            ? request.caseName.Trim()
            : GuessCaseName(request, resolved);
        resolved.outputDir = !string.IsNullOrWhiteSpace(request.outputDir)
            ? ResolveProjectPath(request.outputDir)
            : CreateDefaultDumpDir(resolved.caseName);
        return resolved;
    }

    private async UniTask<PreparedInput> PrepareInputAsync(ResolvedRequest request, CancellationToken ct)
    {
        if (request.source.inputSource == MonaiInputSourceKind.BaselineTensorDump)
            return await PrepareBaselineInputAsync(request, ct);
        return await PrepareMedicalInputAsync(request, ct);
    }

    private async UniTask<PreparedInput> PrepareBaselineInputAsync(ResolvedRequest request, CancellationToken ct)
    {
        if (request.baselineManifest == null)
            throw new FileNotFoundException("MONAI baseline manifest not found", request.baselineManifestPath);

        var caseDir = Path.GetDirectoryName(request.baselineManifestPath);
        if (string.IsNullOrWhiteSpace(caseDir))
            throw new InvalidOperationException("MONAI baseline manifest directory is empty");

        var inputFileName = request.baselineManifest.SelectToken("files.input_tensor_f32_bin")?.Value<string>();
        if (string.IsNullOrWhiteSpace(inputFileName))
            throw new InvalidOperationException("MONAI baseline manifest missing files.input_tensor_f32_bin");

        var inputTensorPath = Path.Combine(caseDir, inputFileName);
        if (!File.Exists(inputTensorPath))
            throw new FileNotFoundException("MONAI baseline input tensor not found", inputTensorPath);

        ct.ThrowIfCancellationRequested();
        var tensor = await ReadFloatArrayAsync(inputTensorPath, ct);
        var expected = checked(request.inputChannels * request.inputDepth * request.inputHeight * request.inputWidth);
        if (tensor.Length != expected)
            throw new InvalidOperationException("MONAI baseline input tensor size mismatch: expected " + expected + " got " + tensor.Length);

        return new PreparedInput
        {
            caseName = request.baselineManifest["case_name"]?.Value<string>() ?? request.caseName,
            tensorNcdhw = tensor,
            sourcePaths = ReadInputPathsFromManifest(request.baselineManifest),
            volumes = BuildVolumeInfoListFromManifest(request.baselineManifest),
            baselineCaseDir = caseDir,
            baselineManifest = request.baselineManifest,
            preparationNote = "baseline_tensor_dump"
        };
    }

    private async UniTask<PreparedInput> PrepareMedicalInputAsync(ResolvedRequest request, CancellationToken ct)
    {
        var inputPaths = request.inputVolumePaths;
        if (inputPaths == null || inputPaths.Length == 0)
            throw new InvalidOperationException("MONAI medical input path is empty");

        var volumes = new List<MonaiVolumeData>(inputPaths.Length);
        for (var i = 0; i < inputPaths.Length; i++)
        {
            ct.ThrowIfCancellationRequested();
            var resolvedPath = ResolveProjectPath(inputPaths[i]);
            if (!File.Exists(resolvedPath))
                throw new FileNotFoundException("MONAI medical input not found", resolvedPath);
            volumes.Add(await UniTask.RunOnThreadPool(() => LoadVolume(resolvedPath), cancellationToken: ct));
        }

        if (volumes.Count == 0)
            throw new InvalidOperationException("MONAI medical input is empty");

        var dim0 = volumes[0].dim0;
        var dim1 = volumes[0].dim1;
        var dim2 = volumes[0].dim2;
        for (var i = 1; i < volumes.Count; i++)
        {
            if (volumes[i].dim0 != dim0 || volumes[i].dim1 != dim1 || volumes[i].dim2 != dim2)
            {
                throw new InvalidOperationException(
                    "All MONAI medical input volumes must share one shape. Expected "
                    + dim0 + "x" + dim1 + "x" + dim2
                    + " but got "
                    + volumes[i].dim0 + "x" + volumes[i].dim1 + "x" + volumes[i].dim2
                    + " from " + volumes[i].path);
            }
        }

        var channels = FillInputChannels(volumes, request.inputChannels, request.channelFillMode);
        var tensor = CropOrPadAndNormalize(channels, request.inputDepth, request.inputHeight, request.inputWidth, request.normalizeNonZero);
        return new PreparedInput
        {
            caseName = request.caseName,
            tensorNcdhw = tensor,
            sourcePaths = ExtractVolumePaths(volumes),
            volumes = BuildVolumeInfoList(volumes),
            baselineCaseDir = request.baselineManifest != null ? Path.GetDirectoryName(request.baselineManifestPath) : null,
            baselineManifest = request.baselineManifest,
            preparationNote = request.normalizeNonZero ? "medical_volume_prepared_nonzero_norm" : "medical_volume_prepared"
        };
    }

    private static string[] CloneArray(string[] values)
    {
        if (values == null || values.Length == 0)
            return Array.Empty<string>();

        var clone = new string[values.Length];
        Array.Copy(values, clone, values.Length);
        return clone;
    }

    private static string[] ExtractVolumePaths(List<MonaiVolumeData> volumes)
    {
        if (volumes == null || volumes.Count == 0)
            return Array.Empty<string>();

        var paths = new string[volumes.Count];
        for (var i = 0; i < volumes.Count; i++)
            paths[i] = volumes[i].path;
        return paths;
    }

    private static List<VolumeInfoRecord> BuildVolumeInfoList(List<MonaiVolumeData> volumes)
    {
        var result = new List<VolumeInfoRecord>();
        if (volumes == null)
            return result;

        for (var i = 0; i < volumes.Count; i++)
        {
            var volume = volumes[i];
            if (volume == null)
                continue;

            result.Add(new VolumeInfoRecord
            {
                path = volume.path,
                dim0 = volume.dim0,
                dim1 = volume.dim1,
                dim2 = volume.dim2,
                format = volume.sourceFormat,
                spacing = volume.spacing
            });
        }
        return result;
    }

    private static List<VolumeInfoRecord> BuildVolumeInfoListFromManifest(JObject manifest)
    {
        var result = new List<VolumeInfoRecord>();
        if (manifest == null)
            return result;

        var inputs = manifest["inputs"] as JArray;
        if (inputs == null)
            return result;

        for (var i = 0; i < inputs.Count; i++)
        {
            var item = inputs[i] as JObject;
            if (item == null)
                continue;
            var shape = ReadIntArray(item["shape"]);
            result.Add(new VolumeInfoRecord
            {
                path = item["path"]?.Value<string>(),
                dim0 = shape != null && shape.Length > 0 ? shape[0] : 0,
                dim1 = shape != null && shape.Length > 1 ? shape[1] : 0,
                dim2 = shape != null && shape.Length > 2 ? shape[2] : 0,
                format = item["format"]?.Value<string>(),
                spacing = ReadFloatArray(item["spacing"])
            });
        }
        return result;
    }

    private static string[] ReadInputPathsFromManifest(JObject manifest)
    {
        var inputs = manifest?["inputs"] as JArray;
        if (inputs == null || inputs.Count == 0)
            return Array.Empty<string>();

        var values = new List<string>(inputs.Count);
        for (var i = 0; i < inputs.Count; i++)
        {
            var path = inputs[i]?["path"]?.Value<string>();
            if (!string.IsNullOrWhiteSpace(path))
                values.Add(path);
        }
        return values.ToArray();
    }

    private static List<MonaiVolumeData> FillInputChannels(List<MonaiVolumeData> volumes, int requiredChannels, MonaiChannelFillMode mode)
    {
        if (volumes == null || volumes.Count == 0)
            throw new InvalidOperationException("MONAI volume list is empty");

        var channels = new List<MonaiVolumeData>(requiredChannels);
        for (var i = 0; i < volumes.Count && channels.Count < requiredChannels; i++)
            channels.Add(volumes[i]);

        while (channels.Count < requiredChannels)
        {
            switch (mode)
            {
                case MonaiChannelFillMode.DuplicateLast:
                    channels.Add(CloneVolume(channels[channels.Count - 1], false));
                    break;
                case MonaiChannelFillMode.Zero:
                    channels.Add(CreateZeroVolume(volumes[0]));
                    break;
                default:
                    channels.Add(CloneVolume(volumes[0], false));
                    break;
            }
        }

        if (channels.Count > requiredChannels)
            channels.RemoveRange(requiredChannels, channels.Count - requiredChannels);
        return channels;
    }

    private static MonaiVolumeData CloneVolume(MonaiVolumeData source, bool cloneData)
    {
        if (source == null)
            return null;

        return new MonaiVolumeData
        {
            path = source.path,
            dim0 = source.dim0,
            dim1 = source.dim1,
            dim2 = source.dim2,
            data = cloneData ? (float[])source.data.Clone() : source.data,
            spacing = source.spacing,
            sourceFormat = source.sourceFormat
        };
    }

    private static MonaiVolumeData CreateZeroVolume(MonaiVolumeData template)
    {
        if (template == null)
            throw new ArgumentNullException(nameof(template));

        return new MonaiVolumeData
        {
            path = template.path + "|zero-fill",
            dim0 = template.dim0,
            dim1 = template.dim1,
            dim2 = template.dim2,
            data = new float[checked(template.dim0 * template.dim1 * template.dim2)],
            spacing = template.spacing,
            sourceFormat = template.sourceFormat
        };
    }

    private static float[] CropOrPadAndNormalize(List<MonaiVolumeData> channels, int targetDepth, int targetHeight, int targetWidth, bool normalizeNonZero)
    {
        if (channels == null || channels.Count == 0)
            throw new InvalidOperationException("MONAI channels are empty");

        var channelCount = channels.Count;
        var outTensor = new float[checked(channelCount * targetDepth * targetHeight * targetWidth)];
        for (var c = 0; c < channelCount; c++)
        {
            var src = channels[c];
            if (src == null || src.data == null)
                throw new InvalidOperationException("MONAI source channel is empty at " + c);

            var srcDepth = src.dim0;
            var srcHeight = src.dim1;
            var srcWidth = src.dim2;
            var copyDepth = Math.Min(srcDepth, targetDepth);
            var copyHeight = Math.Min(srcHeight, targetHeight);
            var copyWidth = Math.Min(srcWidth, targetWidth);
            var srcDepthStart = Math.Max(0, (srcDepth - targetDepth) / 2);
            var srcHeightStart = Math.Max(0, (srcHeight - targetHeight) / 2);
            var srcWidthStart = Math.Max(0, (srcWidth - targetWidth) / 2);
            var dstDepthStart = Math.Max(0, (targetDepth - srcDepth) / 2);
            var dstHeightStart = Math.Max(0, (targetHeight - srcHeight) / 2);
            var dstWidthStart = Math.Max(0, (targetWidth - srcWidth) / 2);

            for (var z = 0; z < copyDepth; z++)
            {
                var srcZ = srcDepthStart + z;
                var dstZ = dstDepthStart + z;
                for (var y = 0; y < copyHeight; y++)
                {
                    var srcY = srcHeightStart + y;
                    var dstY = dstHeightStart + y;
                    var srcOffset = ((srcZ * srcHeight) + srcY) * srcWidth + srcWidthStart;
                    var dstOffset = (((c * targetDepth) + dstZ) * targetHeight + dstY) * targetWidth + dstWidthStart;
                    Array.Copy(src.data, srcOffset, outTensor, dstOffset, copyWidth);
                }
            }
        }

        if (!normalizeNonZero)
            return outTensor;

        var channelVolume = checked(targetDepth * targetHeight * targetWidth);
        for (var c = 0; c < channelCount; c++)
        {
            var channelOffset = c * channelVolume;
            double sum = 0d;
            double sumSq = 0d;
            var count = 0;
            for (var i = 0; i < channelVolume; i++)
            {
                var value = outTensor[channelOffset + i];
                if (Math.Abs(value) <= 1e-12f)
                    continue;
                sum += value;
                sumSq += value * value;
                count++;
            }

            if (count <= 0)
                continue;

            var mean = sum / count;
            var variance = Math.Max(0d, sumSq / count - mean * mean);
            var std = Math.Sqrt(variance);
            if (std < 1e-6d)
                std = 1d;

            for (var i = 0; i < channelVolume; i++)
            {
                var index = channelOffset + i;
                var value = outTensor[index];
                if (Math.Abs(value) <= 1e-12f)
                    continue;
                outTensor[index] = (float)((value - mean) / std);
            }
        }

        return outTensor;
    }

    private static byte[] BuildBratsLabelMap(byte[] masksNcdhw, int width, int height, int depth)
    {
        var volume = checked(width * height * depth);
        var labelMap = new byte[volume];
        for (var i = 0; i < volume; i++)
        {
            var tc = masksNcdhw[i] != 0;
            var wt = masksNcdhw[volume + i] != 0;
            var et = masksNcdhw[(volume * 2) + i] != 0;
            var label = (byte)0;
            if (wt) label = 2;
            if (tc) label = 1;
            if (et) label = 4;
            labelMap[i] = label;
        }
        return labelMap;
    }

    private JObject BuildBaselineComparison(JObject baselineManifest, string baselineCaseDir, float[] logits, float[] probs, byte[] masks, byte[] labelMap)
    {
        if (baselineManifest == null || string.IsNullOrWhiteSpace(baselineCaseDir))
            return null;

        var files = baselineManifest["files"] as JObject;
        if (files == null)
            return null;

        var result = new JObject();
        TryAddFloatArrayDiff(result, "logits", ResolveBaselineFile(files, "logits_f32_bin", baselineCaseDir), logits);
        TryAddFloatArrayDiff(result, "probs", ResolveBaselineFile(files, "probs_f32_bin", baselineCaseDir), probs);
        TryAddByteArrayDiff(result, "masks", ResolveBaselineFile(files, "masks_u8_bin", baselineCaseDir), masks);
        if (labelMap != null)
            TryAddByteArrayDiff(result, "labelmap", ResolveBaselineFile(files, "labelmap_u8_bin", baselineCaseDir), labelMap);
        return result;
    }

    private static void TryAddFloatArrayDiff(JObject dst, string name, string path, float[] current)
    {
        if (dst == null || string.IsNullOrWhiteSpace(name) || string.IsNullOrWhiteSpace(path) || current == null || !File.Exists(path))
            return;

        var baseline = ReadFloatArray(path);
        var obj = new JObject
        {
            ["path"] = path,
            ["current_count"] = current.Length,
            ["baseline_count"] = baseline.Length
        };
        if (baseline.Length != current.Length)
        {
            obj["error"] = "count_mismatch";
            dst[name] = obj;
            return;
        }

        double sumAbs = 0d;
        var maxAbs = 0f;
        var validCount = 0;
        var nanCount = 0;
        var infCount = 0;
        for (var i = 0; i < current.Length; i++)
        {
            var a = current[i];
            var b = baseline[i];
            if (float.IsNaN(a) || float.IsNaN(b))
            {
                nanCount++;
                continue;
            }
            if (float.IsInfinity(a) || float.IsInfinity(b))
            {
                infCount++;
                continue;
            }

            var diff = Mathf.Abs(a - b);
            sumAbs += diff;
            if (diff > maxAbs)
                maxAbs = diff;
            validCount++;
        }

        obj["valid_count"] = validCount;
        obj["nan_or_nan_count"] = nanCount;
        obj["inf_or_inf_count"] = infCount;
        obj["mean_abs"] = validCount > 0 ? sumAbs / validCount : 0d;
        obj["max_abs"] = maxAbs;
        dst[name] = obj;
    }

    private static void TryAddByteArrayDiff(JObject dst, string name, string path, byte[] current)
    {
        if (dst == null || string.IsNullOrWhiteSpace(name) || string.IsNullOrWhiteSpace(path) || current == null || !File.Exists(path))
            return;

        var baseline = File.ReadAllBytes(path);
        var obj = new JObject
        {
            ["path"] = path,
            ["current_count"] = current.Length,
            ["baseline_count"] = baseline.Length
        };
        if (baseline.Length != current.Length)
        {
            obj["error"] = "count_mismatch";
            dst[name] = obj;
            return;
        }

        var mismatch = 0;
        var maxAbs = 0;
        var histCurrent = new Dictionary<byte, int>();
        var histBaseline = new Dictionary<byte, int>();
        for (var i = 0; i < current.Length; i++)
        {
            var a = current[i];
            var b = baseline[i];
            if (a != b)
                mismatch++;
            var diff = Math.Abs(a - b);
            if (diff > maxAbs)
                maxAbs = diff;
            if (histCurrent.ContainsKey(a)) histCurrent[a]++; else histCurrent[a] = 1;
            if (histBaseline.ContainsKey(b)) histBaseline[b]++; else histBaseline[b] = 1;
        }

        obj["mismatch_count"] = mismatch;
        obj["equal_ratio"] = current.Length > 0 ? 1d - mismatch / (double)current.Length : 1d;
        obj["max_abs"] = maxAbs;
        obj["current_histogram"] = BuildHistogramJson(histCurrent);
        obj["baseline_histogram"] = BuildHistogramJson(histBaseline);
        dst[name] = obj;
    }

    private static JObject BuildHistogramJson(Dictionary<byte, int> histogram)
    {
        var obj = new JObject();
        if (histogram == null)
            return obj;

        foreach (var kv in histogram)
            obj[kv.Key.ToString(CultureInfo.InvariantCulture)] = kv.Value;
        return obj;
    }

    private static string ResolveBaselineFile(JObject files, string key, string caseDir)
    {
        var name = files?[key]?.Value<string>();
        if (string.IsNullOrWhiteSpace(name) || string.IsNullOrWhiteSpace(caseDir))
            return null;
        return Path.Combine(caseDir, name);
    }

    private JObject BuildRunManifestJson(
        ResolvedRequest request,
        PreparedInput prepared,
        int outW,
        int outH,
        int outD,
        int outC,
        float[] logits,
        float[] probs,
        byte[] masks,
        byte[] labelMap,
        JObject comparison)
    {
        var root = new JObject
        {
            ["case_name"] = prepared?.caseName ?? request.caseName,
            ["input_source"] = request.source.inputSource.ToString(),
            ["model_param_path"] = request.modelParamPath,
            ["model_bin_path"] = request.modelBinPath,
            ["model_pnnx_param_path"] = request.pnnxParamPath,
            ["bundle_manifest_path"] = request.bundleManifestPath,
            ["baseline_manifest_path"] = request.baselineManifestPath,
            ["input_blob_name"] = request.inputBlobName,
            ["output_blob_name"] = request.outputBlobName,
            ["threshold"] = request.threshold,
            ["channel_fill"] = request.channelFillMode.ToString(),
            ["normalize_nonzero"] = request.normalizeNonZero,
            ["postprocess"] = request.postprocessKind.ToString(),
            ["preparation_note"] = prepared?.preparationNote ?? string.Empty,
            ["model_input_shape_ncdhw"] = new JArray(1, request.inputChannels, request.inputDepth, request.inputHeight, request.inputWidth),
            ["unity_input_shape_whdc"] = new JArray(request.inputWidth, request.inputHeight, request.inputDepth, request.inputChannels),
            ["model_output_shape_ncdhw"] = new JArray(1, outC, outD, outH, outW),
            ["unity_output_shape_whdc"] = new JArray(outW, outH, outD, outC)
        };

        var sources = new JArray();
        if (prepared?.sourcePaths != null)
        {
            for (var i = 0; i < prepared.sourcePaths.Length; i++)
                sources.Add(prepared.sourcePaths[i]);
        }
        root["source_paths"] = sources;

        var volumes = new JArray();
        if (prepared?.volumes != null)
        {
            for (var i = 0; i < prepared.volumes.Count; i++)
            {
                var item = prepared.volumes[i];
                if (item == null)
                    continue;
                volumes.Add(new JObject
                {
                    ["path"] = item.path ?? string.Empty,
                    ["shape"] = new JArray(item.dim0, item.dim1, item.dim2),
                    ["format"] = item.format ?? string.Empty,
                    ["spacing"] = item.spacing != null ? JArray.FromObject(item.spacing) : null
                });
            }
        }
        root["volume_inputs"] = volumes;

        root["stats"] = new JObject
        {
            ["input_tensor"] = BuildFloatStats(prepared?.tensorNcdhw, new[] { request.inputChannels, request.inputDepth, request.inputHeight, request.inputWidth }),
            ["logits"] = BuildFloatStats(logits, new[] { outC, outD, outH, outW }),
            ["probs"] = BuildFloatStats(probs, new[] { outC, outD, outH, outW }),
            ["masks"] = BuildByteStats(masks, new[] { outC, outD, outH, outW }),
            ["labelmap"] = labelMap != null ? BuildByteStats(labelMap, new[] { outD, outH, outW }) : null
        };

        if (comparison != null)
            root["baseline_compare"] = comparison;

        if (request.postprocessKind == MonaiPostprocessKind.BratsTumorSubregions)
        {
            root["label_note"] = "This BraTS bundle predicts tumor subregions (TC/WT/ET). It does not predict skull or ventricles.";
        }

        return root;
    }

    private static JObject BuildFloatStats(float[] values, int[] shape)
    {
        var obj = new JObject
        {
            ["shape"] = shape != null ? JArray.FromObject(shape) : null,
            ["count"] = values != null ? values.Length : 0
        };
        if (values == null || values.Length == 0)
            return obj;

        var finite = 0;
        var nan = 0;
        var inf = 0;
        double sum = 0d;
        float min = float.PositiveInfinity;
        float max = float.NegativeInfinity;
        for (var i = 0; i < values.Length; i++)
        {
            var value = values[i];
            if (float.IsNaN(value))
            {
                nan++;
                continue;
            }
            if (float.IsInfinity(value))
            {
                inf++;
                continue;
            }
            finite++;
            sum += value;
            if (value < min) min = value;
            if (value > max) max = value;
        }

        obj["finite"] = finite;
        obj["nan"] = nan;
        obj["inf"] = inf;
        obj["min"] = finite > 0 ? min : 0f;
        obj["max"] = finite > 0 ? max : 0f;
        obj["mean"] = finite > 0 ? sum / finite : 0d;
        return obj;
    }

    private static JObject BuildByteStats(byte[] values, int[] shape)
    {
        var obj = new JObject
        {
            ["shape"] = shape != null ? JArray.FromObject(shape) : null,
            ["count"] = values != null ? values.Length : 0
        };
        if (values == null || values.Length == 0)
            return obj;

        byte min = byte.MaxValue;
        byte max = byte.MinValue;
        double sum = 0d;
        var histogram = new Dictionary<byte, int>();
        for (var i = 0; i < values.Length; i++)
        {
            var value = values[i];
            if (value < min) min = value;
            if (value > max) max = value;
            sum += value;
            if (histogram.ContainsKey(value)) histogram[value]++; else histogram[value] = 1;
        }

        obj["min"] = min;
        obj["max"] = max;
        obj["mean"] = sum / values.Length;
        obj["histogram"] = BuildHistogramJson(histogram);
        return obj;
    }

    private string BuildInputPreparationText(ResolvedRequest request, PreparedInput prepared)
    {
        var sb = new StringBuilder(1024);
        sb.AppendLine("case=" + (prepared?.caseName ?? request.caseName));
        sb.AppendLine("input_source=" + request.source.inputSource);
        sb.AppendLine("input_blob=" + request.inputBlobName);
        sb.AppendLine("output_blob=" + request.outputBlobName);
        sb.AppendLine("model_input_shape_ncdhw=1," + request.inputChannels + "," + request.inputDepth + "," + request.inputHeight + "," + request.inputWidth);
        sb.AppendLine("unity_input_shape_whdc=" + request.inputWidth + "," + request.inputHeight + "," + request.inputDepth + "," + request.inputChannels);
        sb.AppendLine("channel_fill=" + request.channelFillMode);
        sb.AppendLine("normalize_nonzero=" + request.normalizeNonZero);
        sb.AppendLine("threshold=" + request.threshold.ToString("0.######", CultureInfo.InvariantCulture));
        sb.AppendLine("postprocess=" + request.postprocessKind);
        sb.AppendLine("baseline_manifest=" + (request.baselineManifestPath ?? string.Empty));
        if (prepared?.sourcePaths != null && prepared.sourcePaths.Length > 0)
        {
            for (var i = 0; i < prepared.sourcePaths.Length; i++)
                sb.AppendLine("source[" + i + "]=" + prepared.sourcePaths[i]);
        }
        if (prepared?.volumes != null && prepared.volumes.Count > 0)
        {
            for (var i = 0; i < prepared.volumes.Count; i++)
            {
                var volume = prepared.volumes[i];
                if (volume == null)
                    continue;
                sb.AppendLine(
                    "volume[" + i + "]="
                    + (volume.path ?? string.Empty)
                    + " | shape=" + volume.dim0 + "x" + volume.dim1 + "x" + volume.dim2
                    + " | format=" + (volume.format ?? string.Empty));
            }
        }
        return sb.ToString();
    }

    private string BuildSummaryText(
        ResolvedRequest request,
        PreparedInput prepared,
        int outW,
        int outH,
        int outD,
        int outC,
        JObject comparison,
        byte[] labelMap)
    {
        var sb = new StringBuilder(2048);
        sb.AppendLine("case=" + (prepared?.caseName ?? request.caseName));
        sb.AppendLine("input_source=" + request.source.inputSource);
        sb.AppendLine("model_param=" + (request.modelParamPath ?? string.Empty));
        sb.AppendLine("output_dir=" + (_lastDumpDir ?? string.Empty));
        sb.AppendLine("input_shape_ncdhw=1," + request.inputChannels + "," + request.inputDepth + "," + request.inputHeight + "," + request.inputWidth);
        sb.AppendLine("output_shape_ncdhw=1," + outC + "," + outD + "," + outH + "," + outW);
        sb.AppendLine("channel_fill=" + request.channelFillMode);
        sb.AppendLine("normalize_nonzero=" + request.normalizeNonZero);
        sb.AppendLine("threshold=" + request.threshold.ToString("0.######", CultureInfo.InvariantCulture));
        sb.AppendLine("postprocess=" + request.postprocessKind);
        if (request.postprocessKind == MonaiPostprocessKind.BratsTumorSubregions)
            sb.AppendLine("note=BraTS bundle predicts tumor subregions (TC/WT/ET), not skull or ventricles");

        if (labelMap != null)
        {
            var histogram = new Dictionary<byte, int>();
            for (var i = 0; i < labelMap.Length; i++)
            {
                var label = labelMap[i];
                if (histogram.ContainsKey(label)) histogram[label]++; else histogram[label] = 1;
            }

            sb.Append("label_histogram=");
            var first = true;
            foreach (var kv in histogram)
            {
                if (!first)
                    sb.Append(", ");
                sb.Append(kv.Key.ToString(CultureInfo.InvariantCulture));
                sb.Append(":");
                sb.Append(kv.Value.ToString(CultureInfo.InvariantCulture));
                first = false;
            }
            sb.AppendLine();
        }

        if (comparison != null)
        {
            AppendComparisonLine(sb, comparison, "logits");
            AppendComparisonLine(sb, comparison, "probs");
            AppendComparisonLine(sb, comparison, "masks");
            AppendComparisonLine(sb, comparison, "labelmap");
        }

        return sb.ToString();
    }

    private static void AppendComparisonLine(StringBuilder sb, JObject comparison, string key)
    {
        var token = comparison?[key] as JObject;
        if (token == null)
            return;

        if (token["error"] != null)
        {
            sb.AppendLine("compare_" + key + "=" + token["error"]);
            return;
        }

        if (token["mean_abs"] != null)
        {
            sb.AppendLine(
                "compare_" + key
                + "_mean_abs=" + token["mean_abs"].Value<double>().ToString("G9", CultureInfo.InvariantCulture)
                + " | max_abs=" + token["max_abs"].Value<double>().ToString("G9", CultureInfo.InvariantCulture));
            return;
        }

        if (token["equal_ratio"] != null)
        {
            sb.AppendLine(
                "compare_" + key
                + "_equal_ratio=" + token["equal_ratio"].Value<double>().ToString("G9", CultureInfo.InvariantCulture)
                + " | mismatch_count=" + token["mismatch_count"].Value<int>().ToString(CultureInfo.InvariantCulture));
        }
    }

    private static float Sigmoid(float value)
    {
        return 1f / (1f + Mathf.Exp(-value));
    }

    private static async UniTask<float[]> ReadFloatArrayAsync(string path, CancellationToken ct)
    {
        var bytes = await File.ReadAllBytesAsync(path, ct);
        if (bytes.Length % sizeof(float) != 0)
            throw new InvalidOperationException("Float array file byte length is invalid: " + path);
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static float[] ReadFloatArray(string path)
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length % sizeof(float) != 0)
            throw new InvalidOperationException("Float array file byte length is invalid: " + path);
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static void WriteFloatArray(string path, float[] values)
    {
        if (string.IsNullOrWhiteSpace(path) || values == null)
            return;

        Directory.CreateDirectory(Path.GetDirectoryName(path));
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    private static void WriteByteArray(string path, byte[] values)
    {
        if (string.IsNullOrWhiteSpace(path) || values == null)
            return;

        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllBytes(path, values);
    }

    private static void WriteText(string path, string text)
    {
        if (string.IsNullOrWhiteSpace(path))
            return;

        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, text ?? string.Empty, Encoding.UTF8);
    }

    private void AppendDebugLine(string line)
    {
        if (string.IsNullOrWhiteSpace(line))
            return;

        var stamped = DateTime.Now.ToString("HH:mm:ss.fff", CultureInfo.InvariantCulture) + " | " + line;
        _debugLines.Add(stamped);
        FlushDebugLog();
    }

    private void FlushDebugLog()
    {
        if (!enableDebugDump || string.IsNullOrWhiteSpace(_lastDumpDir) || _debugLines.Count == 0)
            return;

        try
        {
            WriteText(Path.Combine(_lastDumpDir, "runtime_debug.log"), string.Join(Environment.NewLine, _debugLines));
        }
        catch
        {
        }
    }

    private static int[] ReadIntArray(JToken token)
    {
        if (token is not JArray array || array.Count == 0)
            return null;

        var values = new int[array.Count];
        for (var i = 0; i < array.Count; i++)
            values[i] = array[i].Value<int>();
        return values;
    }

    private static float[] ReadFloatArray(JToken token)
    {
        if (token is not JArray array || array.Count == 0)
            return null;

        var values = new float[array.Count];
        for (var i = 0; i < array.Count; i++)
            values[i] = array[i].Value<float>();
        return values;
    }

    private string CreateDefaultDumpDir(string caseName)
    {
        var root = Path.Combine(ProjectRoot, "Logs", "MONAINcnnRepro");
        Directory.CreateDirectory(root);
        var safeCase = string.IsNullOrWhiteSpace(caseName) ? "case" : SanitizeFileName(caseName);
        var dir = Path.Combine(root, safeCase + "_" + DateTime.Now.ToString("yyyyMMdd-HHmmss-fff", CultureInfo.InvariantCulture));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private string ResolveProjectPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return null;
        if (Path.IsPathRooted(path))
            return Path.GetFullPath(path);
        return Path.GetFullPath(Path.Combine(ProjectRoot, path));
    }

    private string GuessCaseName(MonaiRunRequest request, ResolvedRequest resolved)
    {
        if (resolved.baselineManifest != null)
        {
            var name = resolved.baselineManifest["case_name"]?.Value<string>();
            if (!string.IsNullOrWhiteSpace(name))
                return name;
        }

        if (request.inputVolumePaths != null && request.inputVolumePaths.Length > 0)
            return GetCaseNameFromPath(request.inputVolumePaths[0]);
        return "monai_case";
    }

    private static string GetCaseNameFromPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return "monai_case";

        var fileName = Path.GetFileName(path);
        if (fileName.EndsWith(".nii.gz", StringComparison.OrdinalIgnoreCase))
            return fileName.Substring(0, fileName.Length - ".nii.gz".Length);
        return Path.GetFileNameWithoutExtension(fileName);
    }

    private static string SanitizeFileName(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "case";

        var invalid = Path.GetInvalidFileNameChars();
        var sb = new StringBuilder(value.Length);
        for (var i = 0; i < value.Length; i++)
        {
            var c = value[i];
            sb.Append(Array.IndexOf(invalid, c) >= 0 ? '_' : c);
        }
        return sb.ToString();
    }

    private static string[] SplitInputPaths(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Array.Empty<string>();

        var parts = text.Split(new[] { '|', ';' }, StringSplitOptions.RemoveEmptyEntries);
        for (var i = 0; i < parts.Length; i++)
            parts[i] = parts[i].Trim();
        return parts;
    }

    private static MonaiVolumeData LoadVolume(string path)
    {
        var lower = path.ToLowerInvariant();
        if (lower.EndsWith(".nrrd", StringComparison.Ordinal) || lower.EndsWith(".nhdr", StringComparison.Ordinal))
            return LoadNrrd(path);
        if (lower.EndsWith(".nii", StringComparison.Ordinal) || lower.EndsWith(".nii.gz", StringComparison.Ordinal))
            return LoadNifti(path);
        throw new InvalidOperationException("Unsupported MONAI medical input format: " + path);
    }

    private static MonaiVolumeData LoadNrrd(string path)
    {
        var content = File.ReadAllBytes(path);
        var separator = FindNrrdHeaderSeparator(content, out var separatorLength);
        if (separator < 0)
            throw new InvalidOperationException("NRRD header terminator not found: " + path);

        var headerText = Encoding.ASCII.GetString(content, 0, separator);
        var header = ParseNrrdHeader(headerText);
        var sizes = ParseNrrdSizes(header);
        var elementType = ParseNrrdElementType(header);
        var littleEndian = !string.Equals(header.TryGetValue("endian", out var endian) ? endian : "little", "big", StringComparison.OrdinalIgnoreCase);
        var encoding = header.TryGetValue("encoding", out var encodingValue) ? encodingValue.Trim().ToLowerInvariant() : "raw";

        byte[] rawBytes;
        if (path.EndsWith(".nhdr", StringComparison.OrdinalIgnoreCase))
        {
            if (!header.TryGetValue("data file", out var dataFile) || string.IsNullOrWhiteSpace(dataFile))
                throw new InvalidOperationException("Detached NRRD header missing data file field: " + path);
            var rawPath = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(path) ?? string.Empty, dataFile));
            rawBytes = File.ReadAllBytes(rawPath);
        }
        else
        {
            var dataOffset = separator + separatorLength;
            rawBytes = new byte[content.Length - dataOffset];
            Buffer.BlockCopy(content, dataOffset, rawBytes, 0, rawBytes.Length);
        }

        if (encoding == "gzip" || encoding == "gz")
            rawBytes = DecompressGzip(rawBytes);
        else if (encoding != "raw")
            throw new InvalidOperationException("Unsupported NRRD encoding: " + encoding);

        var spacing = ParseNrrdSpacing(header);
        var data = ConvertFirstAxisFastRawToCOrder(rawBytes, sizes[0], sizes[1], sizes[2], elementType, littleEndian, 0);
        return new MonaiVolumeData
        {
            path = path,
            dim0 = sizes[0],
            dim1 = sizes[1],
            dim2 = sizes[2],
            data = data,
            spacing = spacing,
            sourceFormat = "nrrd"
        };
    }

    private static MonaiVolumeData LoadNifti(string path)
    {
        var bytes = File.ReadAllBytes(path);
        if (path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase))
            bytes = DecompressGzip(bytes);
        if (bytes.Length < 348)
            throw new InvalidOperationException("NIfTI header is too short: " + path);

        var littleEndian = true;
        var sizeofHdr = ReadInt32(bytes, 0, littleEndian);
        if (sizeofHdr != 348)
        {
            littleEndian = false;
            sizeofHdr = ReadInt32(bytes, 0, littleEndian);
            if (sizeofHdr != 348)
                throw new InvalidOperationException("Unsupported NIfTI header size: " + path);
        }

        var rank = ReadInt16(bytes, 40, littleEndian);
        if (rank < 3)
            throw new InvalidOperationException("NIfTI rank must be >= 3: " + path);
        var dim0 = ReadInt16(bytes, 42, littleEndian);
        var dim1 = ReadInt16(bytes, 44, littleEndian);
        var dim2 = ReadInt16(bytes, 46, littleEndian);
        var dim3 = rank >= 4 ? Math.Max(1, (int)ReadInt16(bytes, 48, littleEndian)) : 1;
        var datatype = ReadInt16(bytes, 70, littleEndian);
        var voxOffset = Mathf.Max(0, Mathf.RoundToInt(ReadFloat32(bytes, 108, littleEndian)));
        var spacing = new[]
        {
            ReadFloat32(bytes, 80, littleEndian),
            ReadFloat32(bytes, 84, littleEndian),
            ReadFloat32(bytes, 88, littleEndian)
        };

        var elementType = ParseNiftiElementType(datatype);
        var volumeStride = checked(dim0 * dim1 * dim2 * elementType.byteSize);
        if (bytes.Length < voxOffset + volumeStride)
            throw new InvalidOperationException("NIfTI voxel payload is truncated: " + path);

        var data = ConvertFirstAxisFastRawToCOrder(bytes, dim0, dim1, dim2, elementType, littleEndian, voxOffset);
        return new MonaiVolumeData
        {
            path = path,
            dim0 = dim0,
            dim1 = dim1,
            dim2 = dim2,
            data = data,
            spacing = spacing,
            sourceFormat = path.EndsWith(".nii.gz", StringComparison.OrdinalIgnoreCase) ? "nifti-gz" : "nifti"
        };
    }

    private sealed class ScalarType
    {
        public int byteSize;
        public Func<byte[], int, bool, float> read;
    }

    private static ScalarType ParseNrrdElementType(Dictionary<string, string> header)
    {
        if (!header.TryGetValue("type", out var typeName))
            throw new InvalidOperationException("NRRD missing scalar type");
        return ParseScalarType(typeName.Trim().ToLowerInvariant());
    }

    private static ScalarType ParseNiftiElementType(int datatype)
    {
        return datatype switch
        {
            2 => ParseScalarType("uint8"),
            4 => ParseScalarType("int16"),
            8 => ParseScalarType("int32"),
            16 => ParseScalarType("float"),
            64 => ParseScalarType("double"),
            512 => ParseScalarType("uint16"),
            768 => ParseScalarType("uint32"),
            _ => throw new InvalidOperationException("Unsupported NIfTI datatype: " + datatype)
        };
    }

    private static ScalarType ParseScalarType(string normalized)
    {
        switch (normalized)
        {
            case "char":
            case "signed char":
            case "int8":
                return new ScalarType { byteSize = 1, read = static (bytes, offset, _) => unchecked((sbyte)bytes[offset]) };
            case "uchar":
            case "unsigned char":
            case "uint8":
                return new ScalarType { byteSize = 1, read = static (bytes, offset, _) => bytes[offset] };
            case "short":
            case "short int":
            case "signed short":
            case "signed short int":
            case "int16":
                return new ScalarType { byteSize = 2, read = static (bytes, offset, little) => ReadInt16(bytes, offset, little) };
            case "ushort":
            case "unsigned short":
            case "unsigned short int":
            case "uint16":
                return new ScalarType { byteSize = 2, read = static (bytes, offset, little) => ReadUInt16(bytes, offset, little) };
            case "int":
            case "signed int":
            case "int32":
                return new ScalarType { byteSize = 4, read = static (bytes, offset, little) => ReadInt32(bytes, offset, little) };
            case "uint":
            case "unsigned int":
            case "uint32":
                return new ScalarType { byteSize = 4, read = static (bytes, offset, little) => ReadUInt32(bytes, offset, little) };
            case "float":
                return new ScalarType { byteSize = 4, read = static (bytes, offset, little) => ReadFloat32(bytes, offset, little) };
            case "double":
                return new ScalarType { byteSize = 8, read = static (bytes, offset, little) => (float)ReadFloat64(bytes, offset, little) };
            default:
                throw new InvalidOperationException("Unsupported scalar type: " + normalized);
        }
    }

    private static float[] ConvertFirstAxisFastRawToCOrder(byte[] rawBytes, int dim0, int dim1, int dim2, ScalarType elementType, bool littleEndian, int byteOffset)
    {
        var voxelCount = checked(dim0 * dim1 * dim2);
        var expectedBytes = checked(voxelCount * elementType.byteSize);
        if (rawBytes.Length < byteOffset + expectedBytes)
            throw new InvalidOperationException("Voxel payload is shorter than expected");

        var values = new float[voxelCount];
        for (var z = 0; z < dim2; z++)
        {
            for (var y = 0; y < dim1; y++)
            {
                for (var x = 0; x < dim0; x++)
                {
                    var rawIndex = x + dim0 * (y + dim1 * z);
                    var srcOffset = byteOffset + rawIndex * elementType.byteSize;
                    var dstIndex = ((x * dim1) + y) * dim2 + z;
                    values[dstIndex] = elementType.read(rawBytes, srcOffset, littleEndian);
                }
            }
        }
        return values;
    }

    private static int FindNrrdHeaderSeparator(byte[] bytes, out int separatorLength)
    {
        separatorLength = 0;
        for (var i = 0; i <= bytes.Length - 4; i++)
        {
            if (bytes[i] == '\r' && bytes[i + 1] == '\n' && bytes[i + 2] == '\r' && bytes[i + 3] == '\n')
            {
                separatorLength = 4;
                return i;
            }
        }

        for (var i = 0; i <= bytes.Length - 2; i++)
        {
            if (bytes[i] == '\n' && bytes[i + 1] == '\n')
            {
                separatorLength = 2;
                return i;
            }
        }

        return -1;
    }

    private static Dictionary<string, string> ParseNrrdHeader(string headerText)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        using var sr = new StringReader(headerText);
        while (true)
        {
            var line = sr.ReadLine();
            if (line == null)
                break;
            line = line.Trim();
            if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal) || line.StartsWith("NRRD", StringComparison.OrdinalIgnoreCase))
                continue;

            var split = line.IndexOf(":=", StringComparison.Ordinal);
            var sepLen = 2;
            if (split < 0)
            {
                split = line.IndexOf(':');
                sepLen = 1;
            }
            if (split <= 0)
                continue;

            var key = line.Substring(0, split).Trim().ToLowerInvariant();
            var value = line.Substring(split + sepLen).Trim();
            result[key] = value;
        }
        return result;
    }

    private static int[] ParseNrrdSizes(Dictionary<string, string> header)
    {
        if (!header.TryGetValue("sizes", out var sizesText) || string.IsNullOrWhiteSpace(sizesText))
            throw new InvalidOperationException("NRRD missing sizes");

        var parts = sizesText.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length != 3)
            throw new InvalidOperationException("Only 3D scalar NRRD is supported");

        return new[]
        {
            int.Parse(parts[0], CultureInfo.InvariantCulture),
            int.Parse(parts[1], CultureInfo.InvariantCulture),
            int.Parse(parts[2], CultureInfo.InvariantCulture)
        };
    }

    private static float[] ParseNrrdSpacing(Dictionary<string, string> header)
    {
        if (header.TryGetValue("spacings", out var spacingsText) && !string.IsNullOrWhiteSpace(spacingsText))
        {
            var parts = spacingsText.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length >= 3)
            {
                return new[]
                {
                    float.Parse(parts[0], CultureInfo.InvariantCulture),
                    float.Parse(parts[1], CultureInfo.InvariantCulture),
                    float.Parse(parts[2], CultureInfo.InvariantCulture)
                };
            }
        }

        if (!header.TryGetValue("space directions", out var directionsText) || string.IsNullOrWhiteSpace(directionsText))
            return null;

        var values = new List<float>(3);
        var tokens = directionsText.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        for (var i = 0; i < tokens.Length; i++)
        {
            var token = tokens[i].Trim();
            if (string.Equals(token, "none", StringComparison.OrdinalIgnoreCase))
            {
                values.Add(1f);
                continue;
            }
            if (!token.StartsWith("(", StringComparison.Ordinal) || !token.EndsWith(")", StringComparison.Ordinal))
                continue;

            var parts = token.Substring(1, token.Length - 2).Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            double sumSq = 0d;
            for (var j = 0; j < parts.Length; j++)
            {
                var value = double.Parse(parts[j].Trim(), CultureInfo.InvariantCulture);
                sumSq += value * value;
            }
            values.Add((float)Math.Sqrt(sumSq));
        }

        return values.Count >= 3 ? new[] { values[0], values[1], values[2] } : null;
    }

    private static byte[] DecompressGzip(byte[] bytes)
    {
        using var input = new MemoryStream(bytes, false);
        using var gzip = new GZipInputStream(input);
        using var output = new MemoryStream();
        gzip.CopyTo(output);
        return output.ToArray();
    }

    private static short ReadInt16(byte[] bytes, int offset, bool littleEndian)
    {
        if (littleEndian)
            return BitConverter.ToInt16(bytes, offset);
        return (short)((bytes[offset] << 8) | bytes[offset + 1]);
    }

    private static ushort ReadUInt16(byte[] bytes, int offset, bool littleEndian)
    {
        if (littleEndian)
            return BitConverter.ToUInt16(bytes, offset);
        return (ushort)((bytes[offset] << 8) | bytes[offset + 1]);
    }

    private static int ReadInt32(byte[] bytes, int offset, bool littleEndian)
    {
        if (littleEndian)
            return BitConverter.ToInt32(bytes, offset);
        return (bytes[offset] << 24) | (bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3];
    }

    private static uint ReadUInt32(byte[] bytes, int offset, bool littleEndian)
    {
        if (littleEndian)
            return BitConverter.ToUInt32(bytes, offset);
        return ((uint)bytes[offset] << 24) | ((uint)bytes[offset + 1] << 16) | ((uint)bytes[offset + 2] << 8) | bytes[offset + 3];
    }

    private static float ReadFloat32(byte[] bytes, int offset, bool littleEndian)
    {
        if (littleEndian)
            return BitConverter.ToSingle(bytes, offset);

        var tmp = new byte[4];
        tmp[0] = bytes[offset + 3];
        tmp[1] = bytes[offset + 2];
        tmp[2] = bytes[offset + 1];
        tmp[3] = bytes[offset];
        return BitConverter.ToSingle(tmp, 0);
    }

    private static double ReadFloat64(byte[] bytes, int offset, bool littleEndian)
    {
        if (littleEndian)
            return BitConverter.ToDouble(bytes, offset);

        var tmp = new byte[8];
        for (var i = 0; i < 8; i++)
            tmp[i] = bytes[offset + 7 - i];
        return BitConverter.ToDouble(tmp, 0);
    }

    private string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

    private void Release()
    {
        _loadedModelKey = null;
        try { _repro?.Release(); } catch { }
        try { _repro?.Dispose(); } catch { }
        _repro = null;
        _ops = null;
    }

    private void ReportProgress(float progress01, string text)
    {
        try { ProgressChanged?.Invoke(Mathf.Clamp01(progress01), text ?? string.Empty); } catch { }
    }
}
