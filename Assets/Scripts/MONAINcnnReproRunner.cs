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
using UnityEngine.Profiling;

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
    BratsTumorSubregions,
    MulticlassArgmax,
    BinaryLabelPrompt
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
    public bool? normalizeNonZeroOverride;
    public MonaiChannelFillMode channelFillMode = MonaiChannelFillMode.DuplicateFirst;
    public MonaiPostprocessKind postprocessKind = MonaiPostprocessKind.BratsTumorSubregions;
    public bool compareWithBaseline = true;
    public string[] debugPinnedBlobNames;
    public bool probeOnly;
    public int maxSlidingWindowPatches;
    public int probePatchOrdinal;
    public ushort binaryForegroundLabelValue;
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
        public Dictionary<string, string> nrrdHeader;
        public float[] niftiAffine;
    }

    private sealed class VolumeLoadOptions
    {
        public bool includeVoxelData = true;
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
        public MonaiVolumeData referenceVolume;
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
        public int networkInputDepth;
        public int networkInputHeight;
        public int networkInputWidth;
        public int fullInputDepth;
        public int fullInputHeight;
        public int fullInputWidth;
        public int processedInputDepth;
        public int processedInputHeight;
        public int processedInputWidth;
        public float threshold;
        public bool normalizeNonZero;
        public MonaiChannelFillMode channelFillMode;
        public MonaiPostprocessKind postprocessKind;
        public bool compareWithBaseline;
        public string[] debugPinnedBlobNames;
        public JObject bundleManifest;
        public JObject baselineManifest;
        public bool useSlidingWindow;
        public int slidingWindowDepth;
        public int slidingWindowHeight;
        public int slidingWindowWidth;
        public float slidingWindowOverlap;
        public bool probeOnly;
        public int maxSlidingWindowPatches;
        public int probePatchOrdinal;
        public ushort binaryForegroundLabelValue;
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
    public bool dumpLargeTensorFiles = true;
    public bool enableBaselineCompare = true;
    public bool enableTempPool = false;
    public int maxPooledPerShape = 0;
    public bool clearTempPoolAfterEachSlidingWindowPatch = true;
    public int slidingWindowTempPoolClearInterval = 1;
    public int slidingWindowYieldInterval = 1;
    public int slidingWindowManagedCleanupInterval = 1;
    public int slidingWindowResourceSnapshotInterval = 1;
    public float slidingWindowAbortIfPrivateMemoryExceedsMb = 0f;
    public bool forceBufferConvolution = false;
    public bool forceBufferBinaryOp = false;
    public bool forceCpuGemm = false;
    public bool forceBufferAllLayers = false;
    public bool forceBufferOutputsForDims4 = false;
    public ISet<string> forceBufferLayerNames = null;
    public bool useTextureInputForMonaiPatches = false;
    public bool enableAttentionMatMulPack4Specializations = false;
    public bool disallowBufferAccess = false;
    public bool disallowBufferOutputs = false;
    public bool disallowBufferToTextureMaterialization = false;
    public bool keepRawConvWeightsForTexturePath = true;
    public RenderTextureFormat tensorTextureFormat = RenderTextureFormat.ARGBHalf;
    public string debugPinnedBlobNamesCsv = string.Empty;
    public bool logAllLayerHeartbeats = false;
    public bool logAllLayerOutputs = false;
    public bool logAllBufferMaterialize = false;
    public bool enableLayerRuntimeProfile = false;
    public bool syncLayerRuntimeProfile = false;
    public bool enableConv3dTile3x3Pack4FastPath = false;
    public bool enableTimingSplitDiagnostics = false;
    public string timingSplitStopAfterBlobName = string.Empty;
    public string timingSplitSyncAfterTopName = string.Empty;

    public event Action<float, string> ProgressChanged;

    private NcnnOps _ops;
    private NcnnRepro _repro;
    private string _loadedModelKey;
    private string _lastDumpDir;
    private string _lastSummaryText;
    private readonly List<string> _debugLines = new List<string>();
    private readonly List<string> _resourceSnapshotLines = new List<string>();
    private readonly Dictionary<string, long> _timingMs = new Dictionary<string, long>(StringComparer.Ordinal);
    private NcnnRepro.LayerRuntimeProfile _lastLayerRuntimeProfile;
    private string _lastLayerRuntimeProfileText;
    private string _lastPathMode = string.Empty;
    private int _flushedDebugLineCount;
    private bool _loggedPatchInputTextureRoundtrip;
    private bool _timingSplitPatchDiagnosticCaptured;
    private bool _timingSplitReadbackDiagnosticCaptured;
    private readonly Dictionary<string, double> _diagnosticTimingMs = new Dictionary<string, double>(StringComparer.Ordinal);

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
        _resourceSnapshotLines.Clear();
        _timingMs.Clear();
        _lastLayerRuntimeProfile = null;
        _lastLayerRuntimeProfileText = null;
        _lastPathMode = string.Empty;
        _flushedDebugLineCount = 0;
        _loggedPatchInputTextureRoundtrip = false;
        _timingSplitPatchDiagnosticCaptured = false;
        _timingSplitReadbackDiagnosticCaptured = false;
        _diagnosticTimingMs.Clear();

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
            _repro.EnableVistaTailPack4Specializations = resolved.postprocessKind == MonaiPostprocessKind.BinaryLabelPrompt;
            ct.ThrowIfCancellationRequested();

            _lastDumpDir = resolved.outputDir;
            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                Directory.CreateDirectory(_lastDumpDir);
                WriteText(Path.Combine(_lastDumpDir, "runtime_debug.log"), string.Empty);
            }

            ReportProgress(0.06f, "Load MONAI ncnn model");
            var loadSw = Stopwatch.StartNew();
            await EnsureLoadedAsync(resolved, ct);
            loadSw.Stop();
            _timingMs["load_model_ms"] = loadSw.ElapsedMilliseconds;
            ct.ThrowIfCancellationRequested();

            ReportProgress(0.18f, "Prepare input tensor");
            var prepSw = Stopwatch.StartNew();
            var prepared = await PrepareInputAsync(resolved, ct);
            prepSw.Stop();
            _timingMs["prepare_input_ms"] = prepSw.ElapsedMilliseconds;
            if (prepared == null || prepared.tensorNcdhw == null || prepared.tensorNcdhw.Length == 0)
                return Finish(new MonaiSegmentationResult { error = "MONAI input tensor is empty" });

            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                WriteFloatArray(Path.Combine(_lastDumpDir, "input_tensor_ncdhw_f32.bin"), prepared.tensorNcdhw);
                WriteText(Path.Combine(_lastDumpDir, "input_preparation.txt"), BuildInputPreparationText(resolved, prepared));
            }

            ct.ThrowIfCancellationRequested();
            ReportProgress(0.42f, "Run MONAI inference");

            var inferSw = Stopwatch.StartNew();
            var inferResult = resolved.useSlidingWindow
                ? await RunSlidingWindowInferenceAsync(resolved, prepared, ct)
                : RunSinglePassInference(resolved, prepared);
            inferSw.Stop();
            _timingMs["inference_ms"] = inferSw.ElapsedMilliseconds;
            var logits = inferResult.logits;
            var outW = inferResult.width;
            var outH = inferResult.height;
            var outD = inferResult.depth;
            var outC = inferResult.channels;
            var executionNote = inferResult.executionNote;
            var pathMode = string.IsNullOrWhiteSpace(inferResult.pathMode) ? ResolvePathMode(useTextureInputForMonaiPatches) : inferResult.pathMode;
            _lastPathMode = pathMode;

            if ((logits == null || logits.Length == 0)
                && (inferResult.labelMap == null || inferResult.labelMap.Length == 0))
            {
                if (!inferResult.probeOnly)
                    return Finish(new MonaiSegmentationResult { error = "MONAI inference produced neither logits nor labelmap output" });
            }

            ct.ThrowIfCancellationRequested();
            if (inferResult.probeOnly)
            {
                ReportProgress(0.82f, "Write probe dumps");
                var probeSummary = BuildSummaryText(
                    resolved,
                    prepared,
                    outW,
                    outH,
                    outD,
                    outC,
                    null,
                    null,
                    executionNote,
                    false,
                    true,
                    inferResult.executedPatchCount,
                    inferResult.totalPatchCount);
                _lastSummaryText = probeSummary;

                if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                {
                    Directory.CreateDirectory(_lastDumpDir);
                    _timingMs["total_elapsed_ms"] = totalSw.ElapsedMilliseconds;
                    WriteText(Path.Combine(_lastDumpDir, "summary.txt"), probeSummary);
                    WriteMonaiDiagnosticsFiles();
                    WriteText(
                        Path.Combine(_lastDumpDir, "run_manifest.json"),
                        BuildRunManifestJson(
                            resolved,
                            prepared,
                            outW,
                            outH,
                            outD,
                            outC,
                            null,
                            null,
                            null,
                            null,
                            null,
                            executionNote,
                            true,
                            inferResult.executedPatchCount,
                            inferResult.totalPatchCount).ToString());
                    if (_debugLines.Count > 0)
                        WriteText(Path.Combine(_lastDumpDir, "runtime_debug.log"), string.Join(Environment.NewLine, _debugLines));
                }

                ReportProgress(1f, string.Empty);
                return Finish(new MonaiSegmentationResult
                {
                    caseName = prepared.caseName,
                    outputDir = _lastDumpDir,
                    summaryText = probeSummary
                });
            }

            ReportProgress(0.70f, "Postprocess output");

            var postSw = Stopwatch.StartNew();
            float[] probs = null;
            byte[] masks = null;
            ushort[] labelMap16 = inferResult.labelMap;
            var voxelCount = checked(outW * outH * outD);
            if (resolved.postprocessKind == MonaiPostprocessKind.BratsTumorSubregions)
            {
                if (logits == null || logits.Length == 0)
                    return Finish(new MonaiSegmentationResult { error = "BraTS postprocess requires logits output" });

                probs = new float[logits.Length];
                masks = new byte[logits.Length];
                for (var i = 0; i < logits.Length; i++)
                {
                    var p = Sigmoid(logits[i]);
                    probs[i] = p;
                    masks[i] = p >= resolved.threshold ? (byte)1 : (byte)0;
                }

                if (outC != 3)
                    return Finish(new MonaiSegmentationResult { error = "BraTS postprocess expects 3 output channels but got " + outC });
                labelMap16 = ConvertToU16(BuildBratsLabelMap(masks, outW, outH, outD));
            }
            else if (resolved.postprocessKind == MonaiPostprocessKind.BinaryLabelPrompt)
            {
                if (logits == null || logits.Length == 0)
                    return Finish(new MonaiSegmentationResult { error = "Binary label prompt postprocess requires logits output" });

                if (outC != 1)
                    return Finish(new MonaiSegmentationResult { error = "Binary label prompt postprocess expects 1 output channel but got " + outC });

                probs = new float[logits.Length];
                masks = new byte[logits.Length];
                var foregroundLabel = resolved.binaryForegroundLabelValue > 0 ? resolved.binaryForegroundLabelValue : (ushort)1;
                labelMap16 = new ushort[voxelCount];
                for (var i = 0; i < logits.Length; i++)
                {
                    var p = Sigmoid(logits[i]);
                    probs[i] = p;
                    var active = p >= resolved.threshold;
                    masks[i] = active ? (byte)1 : (byte)0;
                    labelMap16[i] = active ? foregroundLabel : (ushort)0;
                }
            }
            else if (resolved.postprocessKind == MonaiPostprocessKind.MulticlassArgmax)
            {
                if (labelMap16 != null && labelMap16.Length != voxelCount)
                {
                    return Finish(new MonaiSegmentationResult
                    {
                        error = "MONAI multiclass labelmap voxel count mismatch: expected " + voxelCount + " got " + labelMap16.Length
                    });
                }

                if (labelMap16 == null)
                {
                    if (logits == null || logits.Length == 0)
                        return Finish(new MonaiSegmentationResult { error = "MONAI multiclass postprocess requires logits or precomputed labelmap" });

                    probs = BuildSoftmaxProbs(logits, outW, outH, outD, outC);
                    labelMap16 = BuildMulticlassLabelMap(probs, voxelCount, outC);
                }
            }
            else
            {
                if (logits == null || logits.Length == 0)
                    return Finish(new MonaiSegmentationResult { error = "MONAI postprocess requires logits output" });
                probs = (float[])logits.Clone();
            }
            postSw.Stop();
            _timingMs["postprocess_ms"] = postSw.ElapsedMilliseconds;

            ReportProgress(0.82f, "Write dumps and compare");
            var compareSw = Stopwatch.StartNew();
            var comparison = resolved.compareWithBaseline && prepared.baselineManifest != null
                ? BuildBaselineComparison(prepared.baselineManifest, prepared.baselineCaseDir, logits, probs, masks, labelMap16)
                : null;
            compareSw.Stop();
            _timingMs["baseline_compare_ms"] = compareSw.ElapsedMilliseconds;

            var summary = BuildSummaryText(
                resolved,
                prepared,
                outW,
                outH,
                outD,
                outC,
                comparison,
                labelMap16,
                executionNote,
                logits != null && logits.Length > 0,
                false,
                inferResult.executedPatchCount,
                inferResult.totalPatchCount);
            _lastSummaryText = summary;

            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                Directory.CreateDirectory(_lastDumpDir);
                if (dumpLargeTensorFiles)
                {
                    WriteFloatArray(Path.Combine(_lastDumpDir, "logits_ncdhw_f32.bin"), logits);
                    WriteFloatArray(Path.Combine(_lastDumpDir, "probs_ncdhw_f32.bin"), probs);
                    if (masks != null)
                        WriteByteArray(Path.Combine(_lastDumpDir, "masks_ncdhw_u8.bin"), masks);
                }
                if (labelMap16 != null)
                {
                    WriteUInt16Array(Path.Combine(_lastDumpDir, "labelmap_dhw_u16.bin"), labelMap16);
                    TryWriteRestoredLabelMap(_lastDumpDir, prepared, labelMap16, outD, outH, outW);
                    var restoredPath = Path.Combine(_lastDumpDir, "labelmap_restored" + (prepared?.referenceVolume?.sourceFormat == "nrrd" ? ".nrrd" : ".nii.gz"));
                    if (comparison != null)
                    {
                        var restoredComparison = BuildRestoredLabelMapComparison(prepared?.baselineManifest, prepared?.baselineCaseDir, restoredPath);
                        if (restoredComparison != null)
                            comparison["restored_labelmap"] = restoredComparison;
                        var subsetComparisons = BuildRestoredLabelSubsetComparisons(
                            prepared?.baselineManifest,
                            prepared?.baselineCaseDir,
                            _lastDumpDir,
                            prepared?.referenceVolume);
                        if (subsetComparisons != null)
                            comparison["label_subsets"] = subsetComparisons;
                    }
                }
                if (comparison != null)
                {
                    summary = BuildSummaryText(
                        resolved,
                        prepared,
                        outW,
                        outH,
                        outD,
                        outC,
                        comparison,
                        labelMap16,
                        executionNote,
                        logits != null && logits.Length > 0,
                        false,
                        inferResult.executedPatchCount,
                        inferResult.totalPatchCount);
                    _lastSummaryText = summary;
                }
                _timingMs["total_elapsed_ms"] = totalSw.ElapsedMilliseconds;
                WriteText(Path.Combine(_lastDumpDir, "summary.txt"), summary);
                WriteMonaiDiagnosticsFiles();
                WriteText(
                    Path.Combine(_lastDumpDir, "run_manifest.json"),
                    BuildRunManifestJson(
                        resolved,
                        prepared,
                        outW,
                        outH,
                        outD,
                        outC,
                        logits,
                        probs,
                        masks,
                        labelMap16,
                        comparison,
                        executionNote,
                        false,
                        inferResult.executedPatchCount,
                        inferResult.totalPatchCount).ToString());
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

        if (_ops != null)
        {
            _ops.EnableConv3dTile3x3Pack4FastPath = enableConv3dTile3x3Pack4FastPath;
        }

        _repro.EnableTempPool = enableTempPool;
        _repro.MaxPooledPerShape = maxPooledPerShape;
        _repro.EnableGroupNormTexturePath = true;
        _repro.UseNcnnStyleGroupNorm = true;
        _repro.ForceBufferConvolution = forceBufferConvolution;
        _repro.ForceBufferBinaryOpAll = forceBufferBinaryOp;
        _repro.ForceCpuGemmAll = forceCpuGemm;
        _repro.ForceBufferLayerTypes = forceBufferAllLayers
            ? new HashSet<string>(StringComparer.Ordinal) { "*" }
            : null;
        _repro.ForceBufferLayerNames = forceBufferLayerNames;
        _repro.ForceBufferOutputsForDims4 = forceBufferOutputsForDims4;
        _repro.DisallowBufferAccess = disallowBufferAccess;
        _repro.DisallowBufferOutputs = disallowBufferOutputs;
        _repro.DisallowBufferToTextureMaterialization = disallowBufferToTextureMaterialization;
        _repro.KeepRawConvWeightsForTexturePath = keepRawConvWeightsForTexturePath;
        _repro.TensorTextureFormat = tensorTextureFormat;
        _repro.DebugLogAllLayerHeartbeats = logAllLayerHeartbeats;
        _repro.DebugLogAllLayerOutputs = logAllLayerOutputs;
        _repro.DebugLogAllBufferMaterialize = logAllBufferMaterialize;
        _repro.LayerRuntimeProfileEnabled = enableLayerRuntimeProfile;
        _repro.LayerRuntimeProfileSyncGpu = syncLayerRuntimeProfile;
        _repro.LayerRuntimeProfilePathKindOverride = ResolvePathMode(useTextureInputForMonaiPatches);
        _repro.TimingSplitSyncAfterTopName = enableTimingSplitDiagnostics ? timingSplitSyncAfterTopName : null;
        _repro.OnTimingSplitSyncPoint = enableTimingSplitDiagnostics ? HandleTimingSplitSyncPoint : null;
        _repro.EnableAttentionMatMulPack4Specializations = enableAttentionMatMulPack4Specializations;
        _repro.EnableVistaTailPack4Specializations = false;
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
            normalizeNonZero = request.normalizeNonZeroOverride ?? request.normalizeNonZero,
            channelFillMode = request.channelFillMode,
            postprocessKind = request.postprocessKind,
            compareWithBaseline = request.compareWithBaseline,
            slidingWindowOverlap = -1f,
            probeOnly = request.probeOnly,
            maxSlidingWindowPatches = Math.Max(0, request.maxSlidingWindowPatches),
            probePatchOrdinal = Math.Max(0, request.probePatchOrdinal),
            binaryForegroundLabelValue = request.binaryForegroundLabelValue
        };

        resolved.debugPinnedBlobNames = request.debugPinnedBlobNames != null && request.debugPinnedBlobNames.Length > 0
            ? CloneArray(request.debugPinnedBlobNames)
            : SplitNameList(debugPinnedBlobNamesCsv);

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
        resolved.networkInputDepth = request.inputDepth;
        resolved.networkInputHeight = request.inputHeight;
        resolved.networkInputWidth = request.inputWidth;

        if (resolved.baselineManifest != null)
        {
            if (resolved.inputChannels <= 0 || resolved.networkInputDepth <= 0 || resolved.networkInputHeight <= 0 || resolved.networkInputWidth <= 0)
            {
                var shape = ReadIntArray(resolved.baselineManifest["model_input_shape_ncdhw"]);
                if (shape != null && shape.Length >= 5)
                {
                    resolved.inputChannels = shape[1];
                    resolved.networkInputDepth = shape[2];
                    resolved.networkInputHeight = shape[3];
                    resolved.networkInputWidth = shape[4];
                }
            }

            var processedShape = ReadIntArray(resolved.baselineManifest["processed_volume_shape_dhw"]);
            if (processedShape != null && processedShape.Length >= 3)
            {
                resolved.processedInputDepth = processedShape[0];
                resolved.processedInputHeight = processedShape[1];
                resolved.processedInputWidth = processedShape[2];
            }

            var thresholdToken = resolved.baselineManifest["threshold"];
            if (thresholdToken != null && request.threshold <= 0f)
                resolved.threshold = thresholdToken.Value<float>();

            var baselineNormalize = resolved.baselineManifest["normalize_nonzero"];
            if (baselineNormalize != null && !request.normalizeNonZeroOverride.HasValue)
                resolved.normalizeNonZero = baselineNormalize.Value<bool>();

            var baselineOriginalShape = ReadIntArray(resolved.baselineManifest["original_volume_shape_dhw"]);
            if (baselineOriginalShape != null && baselineOriginalShape.Length >= 3)
            {
                resolved.fullInputDepth = baselineOriginalShape[0];
                resolved.fullInputHeight = baselineOriginalShape[1];
                resolved.fullInputWidth = baselineOriginalShape[2];
            }

            var baselinePreparedShape = ReadIntArray(resolved.baselineManifest["model_input_shape_ncdhw"]);
            if ((resolved.fullInputDepth <= 0 || resolved.fullInputHeight <= 0 || resolved.fullInputWidth <= 0)
                && baselinePreparedShape != null && baselinePreparedShape.Length >= 5)
            {
                resolved.fullInputDepth = baselinePreparedShape[2];
                resolved.fullInputHeight = baselinePreparedShape[3];
                resolved.fullInputWidth = baselinePreparedShape[4];
            }

            var baselineRoi = ReadIntArray(resolved.baselineManifest["sliding_window_roi_dhw"]);
            if (baselineRoi != null && baselineRoi.Length >= 3)
            {
                resolved.slidingWindowDepth = baselineRoi[0];
                resolved.slidingWindowHeight = baselineRoi[1];
                resolved.slidingWindowWidth = baselineRoi[2];
            }

            var baselineOverlap = resolved.baselineManifest["sliding_window_overlap"];
            if (baselineOverlap != null)
                resolved.slidingWindowOverlap = baselineOverlap.Value<float>();

            var inferMode = resolved.baselineManifest["infer_mode"]?.Value<string>();
            resolved.useSlidingWindow = string.Equals(inferMode, "monai-sliding-window", StringComparison.OrdinalIgnoreCase);

            if (resolved.binaryForegroundLabelValue == 0)
            {
                resolved.binaryForegroundLabelValue = (ushort?)resolved.baselineManifest["prompt"]?["foreground_label_value"]?.Value<int>() ?? (ushort)0;
            }
        }

        if ((resolved.inputChannels <= 0 || resolved.networkInputDepth <= 0 || resolved.networkInputHeight <= 0 || resolved.networkInputWidth <= 0) && resolved.bundleManifest != null)
        {
            var bundleShape = ReadIntArray(resolved.bundleManifest["input_shape"]);
            if (bundleShape != null && bundleShape.Length >= 5)
            {
                resolved.inputChannels = bundleShape[1];
                resolved.networkInputDepth = bundleShape[2];
                resolved.networkInputHeight = bundleShape[3];
                resolved.networkInputWidth = bundleShape[4];
            }
        }

        if (resolved.inputChannels <= 0 || resolved.networkInputDepth <= 0 || resolved.networkInputHeight <= 0 || resolved.networkInputWidth <= 0)
            throw new InvalidOperationException("MONAI input shape is unresolved. Please provide bundle/baseline manifest or explicit shape.");

        if (resolved.processedInputDepth <= 0 || resolved.processedInputHeight <= 0 || resolved.processedInputWidth <= 0)
        {
            resolved.processedInputDepth = resolved.networkInputDepth;
            resolved.processedInputHeight = resolved.networkInputHeight;
            resolved.processedInputWidth = resolved.networkInputWidth;
        }

        if (resolved.fullInputDepth <= 0 || resolved.fullInputHeight <= 0 || resolved.fullInputWidth <= 0)
        {
            resolved.fullInputDepth = resolved.processedInputDepth;
            resolved.fullInputHeight = resolved.processedInputHeight;
            resolved.fullInputWidth = resolved.processedInputWidth;
        }

        if (resolved.slidingWindowDepth <= 0 || resolved.slidingWindowHeight <= 0 || resolved.slidingWindowWidth <= 0)
        {
            resolved.slidingWindowDepth = resolved.networkInputDepth;
            resolved.slidingWindowHeight = resolved.networkInputHeight;
            resolved.slidingWindowWidth = resolved.networkInputWidth;
        }

        if (resolved.slidingWindowOverlap < 0f)
            resolved.slidingWindowOverlap = 0.25f;

        if (resolved.probeOnly)
        {
            resolved.compareWithBaseline = false;
            if (resolved.maxSlidingWindowPatches <= 0)
                resolved.maxSlidingWindowPatches = 1;
        }

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
        AppendDebugLine("PrepareBaselineInputAsync begin | manifest=" + (request?.baselineManifestPath ?? string.Empty));
        if (request.baselineManifest == null)
            throw new FileNotFoundException("MONAI baseline manifest not found", request.baselineManifestPath);

        var caseDir = Path.GetDirectoryName(request.baselineManifestPath);
        if (string.IsNullOrWhiteSpace(caseDir))
            throw new InvalidOperationException("MONAI baseline manifest directory is empty");
        AppendDebugLine("PrepareBaselineInputAsync case dir | dir=" + caseDir);

        var inputFileName = request.baselineManifest.SelectToken("files.input_tensor_f32_bin")?.Value<string>();
        if (string.IsNullOrWhiteSpace(inputFileName))
            throw new InvalidOperationException("MONAI baseline manifest missing files.input_tensor_f32_bin");

        var inputTensorPath = Path.Combine(caseDir, inputFileName);
        if (!File.Exists(inputTensorPath))
            throw new FileNotFoundException("MONAI baseline input tensor not found", inputTensorPath);
        AppendDebugLine("PrepareBaselineInputAsync tensor path | path=" + inputTensorPath);
        AppendDebugLine("PrepareBaselineInputAsync tensor bytes | bytes=" + new FileInfo(inputTensorPath).Length.ToString(CultureInfo.InvariantCulture));

        ct.ThrowIfCancellationRequested();
        AppendDebugLine("PrepareBaselineInputAsync read tensor start");
        var tensor = await UniTask.RunOnThreadPool(() => ReadFloatArray(inputTensorPath), cancellationToken: ct);
        AppendDebugLine("PrepareBaselineInputAsync read tensor done | floats=" + tensor.Length.ToString(CultureInfo.InvariantCulture));
        var expected = checked(request.inputChannels * request.processedInputDepth * request.processedInputHeight * request.processedInputWidth);
        AppendDebugLine("PrepareBaselineInputAsync validate tensor | expected=" + expected.ToString(CultureInfo.InvariantCulture) + " | actual=" + tensor.Length.ToString(CultureInfo.InvariantCulture));
        if (tensor.Length != expected)
            throw new InvalidOperationException("MONAI baseline input tensor size mismatch: expected " + expected + " got " + tensor.Length);

        var manifestReferenceVolume = TryLoadReferenceVolumeFromManifest(request.baselineManifest, new VolumeLoadOptions { includeVoxelData = false });
        MonaiVolumeData referenceVolume = manifestReferenceVolume;
        if (request.inputVolumePaths != null && request.inputVolumePaths.Length > 0)
        {
            var referencePath = ResolveProjectPath(request.inputVolumePaths[0]);
            if (!string.IsNullOrWhiteSpace(referencePath) && File.Exists(referencePath))
            {
                AppendDebugLine("PrepareBaselineInputAsync reference load | path=" + referencePath);
                var requestedReference = await UniTask.RunOnThreadPool(
                    () => LoadVolume(referencePath, new VolumeLoadOptions { includeVoxelData = false }),
                    cancellationToken: ct);
                if (ReferenceVolumeMatchesBaseline(manifestReferenceVolume, requestedReference))
                {
                    referenceVolume = requestedReference;
                }
                else if (requestedReference != null)
                {
                    AppendDebugLine(
                        "PrepareBaselineInputAsync reference shape mismatch"
                        + " | requested=" + requestedReference.dim0 + "x" + requestedReference.dim1 + "x" + requestedReference.dim2
                        + " | baseline=" + (manifestReferenceVolume != null
                            ? manifestReferenceVolume.dim0 + "x" + manifestReferenceVolume.dim1 + "x" + manifestReferenceVolume.dim2
                            : "none")
                        + " | fallback=baseline_manifest");
                }
            }
        }

        referenceVolume ??= manifestReferenceVolume;

        AppendDebugLine("PrepareBaselineInputAsync complete | case=" + (request.baselineManifest["case_name"]?.Value<string>() ?? request.caseName ?? string.Empty));
        return new PreparedInput
        {
            caseName = request.baselineManifest["case_name"]?.Value<string>() ?? request.caseName,
            tensorNcdhw = tensor,
            sourcePaths = ReadInputPathsFromManifest(request.baselineManifest),
            volumes = BuildVolumeInfoListFromManifest(request.baselineManifest),
            baselineCaseDir = caseDir,
            baselineManifest = request.baselineManifest,
            preparationNote = "baseline_tensor_dump",
            referenceVolume = referenceVolume
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
        var tensor = CropOrPadAndNormalize(
            channels,
            request.processedInputDepth,
            request.processedInputHeight,
            request.processedInputWidth,
            request.normalizeNonZero);
        return new PreparedInput
        {
            caseName = request.caseName,
            tensorNcdhw = tensor,
            sourcePaths = ExtractVolumePaths(volumes),
            volumes = BuildVolumeInfoList(volumes),
            baselineCaseDir = request.baselineManifest != null ? Path.GetDirectoryName(request.baselineManifestPath) : null,
            baselineManifest = request.baselineManifest,
            preparationNote = request.normalizeNonZero ? "medical_volume_prepared_nonzero_norm" : "medical_volume_prepared",
            referenceVolume = volumes[0]
        };
    }

    private sealed class InferenceRunResult
    {
        public float[] logits;
        public ushort[] labelMap;
        public int width;
        public int height;
        public int depth;
        public int channels;
        public string executionNote;
        public string pathMode;
        public bool probeOnly;
        public int executedPatchCount;
        public int totalPatchCount;
    }

    private sealed class OutputHeadInfo
    {
        public string layerName;
        public string featureBlobName;
        public int featureChannels;
        public int outputChannels;
        public NcnnRepro.ConvPack conv;
    }

    private readonly struct InferOutputShape
    {
        public readonly int dims;
        public readonly int w;
        public readonly int h;
        public readonly int d;
        public readonly int c;

        public InferOutputShape(int dims, int w, int h, int d, int c)
        {
            this.dims = dims;
            this.w = w;
            this.h = h;
            this.d = d;
            this.c = c;
        }
    }

    private readonly struct PatchInferHandle : IDisposable
    {
        public readonly NcnnRepro.InferResult infer;
        private readonly NcnnRepro _repro;
        private readonly RenderTexture _ownedInputTexture;

        public PatchInferHandle(NcnnRepro.InferResult infer, NcnnRepro repro, RenderTexture ownedInputTexture)
        {
            this.infer = infer;
            _repro = repro;
            _ownedInputTexture = ownedInputTexture;
        }

        public void Dispose()
        {
            try { infer?.Dispose(); } catch { }
            if (_ownedInputTexture != null)
            {
                try { _repro?.ReturnTempArray(_ownedInputTexture); } catch { }
            }
        }
    }

    private static string ResolvePathMode(bool useTextureInputForPatches)
    {
        return useTextureInputForPatches ? "pack4_rt" : "compute_buffer";
    }

    private void CaptureLatestLayerRuntimeProfile(string pathMode)
    {
        _lastPathMode = pathMode ?? string.Empty;
        _lastLayerRuntimeProfile = _repro?.LastRuntimeProfile;
        _lastLayerRuntimeProfileText = _lastLayerRuntimeProfile != null
            ? NcnnRepro.FormatLayerRuntimeProfile(_lastLayerRuntimeProfile, 256)
            : null;
    }

    private InferenceRunResult RunSinglePassInference(ResolvedRequest resolved, PreparedInput prepared)
    {
        using var inputTensor = new NcnnTensorBuffer(
            resolved.networkInputWidth,
            resolved.networkInputHeight,
            resolved.networkInputDepth,
            resolved.inputChannels);
        inputTensor.buffer.SetData(prepared.tensorNcdhw);
        var probeHead = resolved.probeOnly
            && resolved.postprocessKind == MonaiPostprocessKind.MulticlassArgmax
            && TryResolveLinearOutputHead(resolved.outputBlobName, out var singlePassOutputHead)
            ? singlePassOutputHead
            : null;
        var probeBlobName = probeHead != null ? probeHead.featureBlobName : resolved.outputBlobName;
        var pinnedBlobNames = BuildPinnedBlobNames(probeBlobName, resolved.debugPinnedBlobNames);
        using var inferHandle = RunInferenceWithPatchInput(
            resolved,
            inputTensor,
            resolved.networkInputDepth,
            resolved.networkInputHeight,
            resolved.networkInputWidth,
            pinnedBlobNames,
            resolved.probeOnly ? probeBlobName : null);
        var infer = inferHandle.infer;

        var outputView = GetInferOutputShape(infer, probeBlobName, "MONAI output blob missing: ");

        if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            DumpPinnedBlobOutputs(infer, pinnedBlobNames);

        if (resolved.probeOnly)
        {
            CaptureLatestLayerRuntimeProfile(ResolvePathMode(useTextureInputForMonaiPatches));
            return new InferenceRunResult
            {
                width = outputView.w,
                height = outputView.h,
                depth = outputView.d,
                channels = probeHead != null ? probeHead.outputChannels : outputView.c,
                executionNote = "probe_blob:" + probeBlobName,
                pathMode = ResolvePathMode(useTextureInputForMonaiPatches),
                probeOnly = true,
                executedPatchCount = 1,
                totalPatchCount = 1
            };
        }

        if (outputView.dims != 4)
            throw new InvalidOperationException("MONAI output dims expected 4 but got " + outputView.dims);

        CaptureLatestLayerRuntimeProfile(ResolvePathMode(useTextureInputForMonaiPatches));
        return new InferenceRunResult
        {
            logits = ExtractInferOutputData(infer, resolved.outputBlobName, outputView),
            width = outputView.w,
            height = outputView.h,
            depth = outputView.d,
            channels = outputView.c,
            pathMode = ResolvePathMode(useTextureInputForMonaiPatches),
            executedPatchCount = 1,
            totalPatchCount = 1
        };
    }

    private async UniTask<InferenceRunResult> RunSlidingWindowInferenceAsync(ResolvedRequest resolved, PreparedInput prepared, CancellationToken ct)
    {
        if (resolved.probeOnly)
            return await RunSlidingWindowProbeInferenceAsync(resolved, prepared, ct);

        if (resolved.postprocessKind == MonaiPostprocessKind.MulticlassArgmax
            && TryResolveLinearOutputHead(resolved.outputBlobName, out var outputHead))
        {
            return await RunSlidingWindowFeatureAggregationInferenceAsync(resolved, prepared, outputHead, ct);
        }

        var roiD = resolved.slidingWindowDepth;
        var roiH = resolved.slidingWindowHeight;
        var roiW = resolved.slidingWindowWidth;
        var inferD = resolved.processedInputDepth;
        var inferH = resolved.processedInputHeight;
        var inferW = resolved.processedInputWidth;
        var channelCount = resolved.inputChannels;

        if (roiD <= 0 || roiH <= 0 || roiW <= 0)
            throw new InvalidOperationException("Sliding window roi size is invalid.");

        var startsD = BuildSlidingWindowStarts(inferD, roiD, resolved.slidingWindowOverlap);
        var startsH = BuildSlidingWindowStarts(inferH, roiH, resolved.slidingWindowOverlap);
        var startsW = BuildSlidingWindowStarts(inferW, roiW, resolved.slidingWindowOverlap);
        var pinnedBlobNames = BuildPinnedBlobNames(resolved.outputBlobName, resolved.debugPinnedBlobNames);

        float[] accum = null;
        float[] weight = null;
        int outC = 0;
        int patchCount = startsD.Count * startsH.Count * startsW.Count;
        var patchIndex = 0;

        using var inputTensor = new NcnnTensorBuffer(roiW, roiH, roiD, channelCount);
        var patchTensor = new float[checked(channelCount * roiD * roiH * roiW)];
        for (var iz = 0; iz < startsD.Count; iz++)
        {
            var startD = startsD[iz];
            for (var iy = 0; iy < startsH.Count; iy++)
            {
                var startH = startsH[iy];
                for (var ix = 0; ix < startsW.Count; ix++)
                {
                    ct.ThrowIfCancellationRequested();
                    var startW = startsW[ix];
                    patchIndex++;
                    ExtractPatchNcdhw(
                        prepared.tensorNcdhw,
                        channelCount,
                        inferD,
                        inferH,
                        inferW,
                        startD,
                        startH,
                        startW,
                        roiD,
                        roiH,
                        roiW,
                    patchTensor);
                    inputTensor.buffer.SetData(patchTensor);

                    if (ShouldRunTimingSplitDiagnosticsForPatch(patchIndex))
                        RunPatchTimingSplitDiagnostics(resolved, inputTensor, roiD, roiH, roiW);

                    var fullDispatchSw = enableTimingSplitDiagnostics && patchIndex == 1 ? Stopwatch.StartNew() : null;
                    using var inferHandle = RunInferenceWithPatchInput(
                        resolved,
                        inputTensor,
                        roiD,
                        roiH,
                        roiW,
                        pinnedBlobNames,
                        null);
                    fullDispatchSw?.Stop();
                    var infer = inferHandle.infer;

                    if (enableTimingSplitDiagnostics && patchIndex == 1)
                    {
                        var dispatchMs = StopwatchToMilliseconds(fullDispatchSw);
                        RecordDiagnosticTiming("diag_full_dispatch_ms", dispatchMs);
                        var syncSw = Stopwatch.StartNew();
                        _ops?.DebugSyncGpu();
                        syncSw.Stop();
                        var syncMs = StopwatchToMilliseconds(syncSw);
                        RecordDiagnosticTiming("diag_full_sync_after_return_ms", syncMs);
                        RecordDiagnosticTiming("diag_full_total_before_readback_ms", dispatchMs + syncMs);
                        AppendDebugLine(
                            "TimingSplitDiagnostic full_infer"
                            + " | patch=" + patchIndex.ToString(CultureInfo.InvariantCulture)
                            + " | dispatch_ms=" + dispatchMs.ToString("0.###", CultureInfo.InvariantCulture)
                            + " | sync_after_return_ms=" + syncMs.ToString("0.###", CultureInfo.InvariantCulture));
                    }

                    var outputView = GetInferOutputShape(infer, resolved.outputBlobName, "MONAI sliding window output blob missing: ");
                    if (outputView.dims != 4)
                        throw new InvalidOperationException("MONAI sliding window output dims expected 4 but got " + outputView.dims);
                    if (outputView.w != roiW || outputView.h != roiH || outputView.d != roiD)
                    {
                        throw new InvalidOperationException(
                            "Sliding window output spatial shape mismatch: expected "
                            + roiW + "x" + roiH + "x" + roiD
                            + " got " + outputView.w + "x" + outputView.h + "x" + outputView.d);
                    }

                    var patchLogits = ExtractInferOutputData(infer, resolved.outputBlobName, outputView);
                    if (patchLogits == null || patchLogits.Length == 0)
                        throw new InvalidOperationException("MONAI sliding window logits are empty.");

                    if (accum == null)
                    {
                        outC = outputView.c;
                        accum = new float[checked(outC * inferD * inferH * inferW)];
                        weight = new float[checked(inferD * inferH * inferW)];
                    }
                    else if (outputView.c != outC)
                    {
                        throw new InvalidOperationException("Sliding window output channel mismatch across patches.");
                    }

                    AccumulatePatchLogits(
                        accum,
                        weight,
                        patchLogits,
                        inferD,
                        inferH,
                        inferW,
                        roiD,
                        roiH,
                        roiW,
                        outC,
                        startD,
                        startH,
                        startW);

                    if (patchIndex == 1 && enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                        DumpPinnedBlobOutputs(infer, pinnedBlobNames);

                    await HandleSlidingWindowPatchCompletedAsync(
                        patchIndex,
                        patchCount,
                        0.68f,
                        "Run MONAI inference patch ",
                        ct);
                    await MaybeRunSlidingWindowMaintenanceAsync(patchIndex, patchCount, "logits_patch", ct);
                }
            }
        }

        if (accum == null || weight == null || outC <= 0)
            throw new InvalidOperationException("Sliding window inference did not produce any patches.");

        NormalizeAccumulatedLogits(accum, weight, outC, inferD, inferH, inferW);
        CaptureLatestLayerRuntimeProfile(ResolvePathMode(useTextureInputForMonaiPatches));
        return new InferenceRunResult
        {
            logits = accum,
            width = inferW,
            height = inferH,
            depth = inferD,
            channels = outC,
            pathMode = ResolvePathMode(useTextureInputForMonaiPatches),
            executedPatchCount = patchCount,
            totalPatchCount = patchCount
        };
    }

    private bool TryResolveLinearOutputHead(string outputBlobName, out OutputHeadInfo head)
    {
        head = null;
        var model = _repro?.Model;
        if (model?.layers == null || string.IsNullOrWhiteSpace(outputBlobName))
            return false;

        for (var i = model.layers.Count - 1; i >= 0; i--)
        {
            var layer = model.layers[i];
            if (layer?.topNames == null || Array.IndexOf(layer.topNames, outputBlobName) < 0)
                continue;
            if (layer.type != NcnnLayerTypes.Convolution3D)
                return false;
            if (!_repro._conv.TryGetValue(layer.name, out var conv) || conv == null)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length == 0 || string.IsNullOrWhiteSpace(layer.bottomNames[0]))
                return false;

            var isLinear1x1x1 =
                conv.kernelW == 1 && conv.kernelH == 1 && conv.kernelD == 1
                && conv.strideW == 1 && conv.strideH == 1 && conv.strideD == 1
                && conv.dilationW == 1 && conv.dilationH == 1 && conv.dilationD == 1
                && conv.padLeft == 0 && conv.padRight == 0
                && conv.padTop == 0 && conv.padBottom == 0
                && conv.padFront == 0 && conv.padBehind == 0
                && conv.group == 1
                && conv.rawWeight != null
                && conv.rawBias != null;
            if (!isLinear1x1x1)
                return false;

            head = new OutputHeadInfo
            {
                layerName = layer.name,
                featureBlobName = layer.bottomNames[0],
                featureChannels = conv.inC,
                outputChannels = conv.outC,
                conv = conv
            };
            return true;
        }

        return false;
    }

    private async UniTask<InferenceRunResult> RunSlidingWindowFeatureAggregationInferenceAsync(
        ResolvedRequest resolved,
        PreparedInput prepared,
        OutputHeadInfo outputHead,
        CancellationToken ct)
    {
        if (Mathf.Abs(resolved.slidingWindowOverlap) <= 1e-8f)
            return await RunSlidingWindowFeatureOwnershipInferenceAsync(resolved, prepared, outputHead, ct);

        if (outputHead == null || outputHead.conv == null)
            throw new ArgumentNullException(nameof(outputHead));

        var roiD = resolved.slidingWindowDepth;
        var roiH = resolved.slidingWindowHeight;
        var roiW = resolved.slidingWindowWidth;
        var inferD = resolved.processedInputDepth;
        var inferH = resolved.processedInputHeight;
        var inferW = resolved.processedInputWidth;
        var inputChannels = resolved.inputChannels;
        var featureChannels = outputHead.featureChannels;

        if (roiD <= 0 || roiH <= 0 || roiW <= 0)
            throw new InvalidOperationException("Sliding window roi size is invalid.");
        if (featureChannels <= 0 || outputHead.outputChannels <= 0)
            throw new InvalidOperationException("Output head channels are invalid.");

        AppendDebugLine(
            "RunSlidingWindowFeatureAggregationInference begin"
            + " | featureBlob=" + outputHead.featureBlobName
            + " | featureChannels=" + featureChannels.ToString(CultureInfo.InvariantCulture)
            + " | outputChannels=" + outputHead.outputChannels.ToString(CultureInfo.InvariantCulture)
            + " | headLayer=" + (outputHead.layerName ?? string.Empty));

        var startsD = BuildSlidingWindowStarts(inferD, roiD, resolved.slidingWindowOverlap);
        var startsH = BuildSlidingWindowStarts(inferH, roiH, resolved.slidingWindowOverlap);
        var startsW = BuildSlidingWindowStarts(inferW, roiW, resolved.slidingWindowOverlap);
        var pinnedBlobNames = BuildPinnedBlobNames(outputHead.featureBlobName, resolved.debugPinnedBlobNames);
        var patchCount = startsD.Count * startsH.Count * startsW.Count;
        var patchIndex = 0;

        var fullPlane = checked(inferH * inferW);
        var fullVoxelCount = checked(inferD * fullPlane);
        var featurePatchElementCount = checked(featureChannels * roiD * roiH * roiW);
        var bandDepthCapacity = Mathf.Min(roiD, inferD);
        var featureBandElementCount = checked(featureChannels * bandDepthCapacity * fullPlane);
        var weightBandElementCount = checked(bandDepthCapacity * fullPlane);
        AppendDebugLine(
            "RunSlidingWindowFeatureAggregationInference memory_estimate"
            + " | featurePatchMiB=" + FormatMiBFromFloatCount(featurePatchElementCount)
            + " | featureBandMiB=" + FormatMiBFromFloatCount(featureBandElementCount)
            + " | weightBandMiB=" + FormatMiBFromFloatCount(weightBandElementCount)
            + " | labelMapMiB=" + ((fullVoxelCount * (double)sizeof(ushort)) / (1024d * 1024d)).ToString("F3", CultureInfo.InvariantCulture)
            + " | bandDepth=" + bandDepthCapacity.ToString(CultureInfo.InvariantCulture)
            + " | fullShape=" + inferD.ToString(CultureInfo.InvariantCulture) + "x" + inferH.ToString(CultureInfo.InvariantCulture) + "x" + inferW.ToString(CultureInfo.InvariantCulture));
        ThrowIfSlidingWindowMemoryLimitExceeded("before_feature_band_alloc", 0, patchCount);

        var labelMap = new ushort[fullVoxelCount];
        var featureBand = new float[featureBandElementCount];
        var weightBand = new float[weightBandElementCount];
        var featurePatch = new float[featurePatchElementCount];
        var bandBaseDepth = 0;
        var activeBandDepth = 0;

        using var inputTensor = new NcnnTensorBuffer(roiW, roiH, roiD, inputChannels);
        var patchTensor = new float[checked(inputChannels * roiD * roiH * roiW)];
        for (var iz = 0; iz < startsD.Count; iz++)
        {
            var startD = startsD[iz];
            for (var iy = 0; iy < startsH.Count; iy++)
            {
                var startH = startsH[iy];
                for (var ix = 0; ix < startsW.Count; ix++)
                {
                    ct.ThrowIfCancellationRequested();
                    var startW = startsW[ix];
                    patchIndex++;
                    ExtractPatchNcdhw(
                        prepared.tensorNcdhw,
                        inputChannels,
                        inferD,
                        inferH,
                        inferW,
                        startD,
                        startH,
                        startW,
                        roiD,
                        roiH,
                        roiW,
                        patchTensor);
                    inputTensor.buffer.SetData(patchTensor);

                    using var inferHandle = RunInferenceWithPatchInput(
                        resolved,
                        inputTensor,
                        roiD,
                        roiH,
                        roiW,
                        pinnedBlobNames,
                        outputHead.featureBlobName);
                    var infer = inferHandle.infer;

                    var featureView = GetInferOutputShape(infer, outputHead.featureBlobName, "MONAI sliding window feature blob missing: ");
                    if (featureView.dims != 4)
                    {
                        throw new InvalidOperationException(
                            "MONAI sliding window feature dims expected 4 but got " + featureView.dims);
                    }
                    if (featureView.w != roiW || featureView.h != roiH || featureView.d != roiD || featureView.c != featureChannels)
                    {
                        throw new InvalidOperationException(
                            "Sliding window feature shape mismatch: expected "
                            + roiW + "x" + roiH + "x" + roiD + "x" + featureChannels
                            + " got "
                            + featureView.w + "x" + featureView.h + "x" + featureView.d + "x" + featureView.c);
                    }

                    var featureData = ExtractInferOutputData(infer, outputHead.featureBlobName, featureView);
                    Array.Copy(featureData, featurePatch, featurePatch.Length);
                    var localStartDepth = startD - bandBaseDepth;
                    if (localStartDepth < 0)
                    {
                        throw new InvalidOperationException(
                            "Sliding window feature band underflow: bandBaseDepth="
                            + bandBaseDepth.ToString(CultureInfo.InvariantCulture)
                            + " startDepth=" + startD.ToString(CultureInfo.InvariantCulture));
                    }
                    var localEndDepth = checked(localStartDepth + roiD);
                    if (localEndDepth > bandDepthCapacity)
                    {
                        throw new InvalidOperationException(
                            "Sliding window feature band overflow: localEndDepth="
                            + localEndDepth.ToString(CultureInfo.InvariantCulture)
                            + " bandDepthCapacity=" + bandDepthCapacity.ToString(CultureInfo.InvariantCulture)
                            + " | startDepth=" + startD.ToString(CultureInfo.InvariantCulture)
                            + " | bandBaseDepth=" + bandBaseDepth.ToString(CultureInfo.InvariantCulture));
                    }

                    activeBandDepth = Math.Max(activeBandDepth, localEndDepth);
                    AccumulatePatchToDepthBand(
                        featureBand,
                        weightBand,
                        bandDepthCapacity,
                        featurePatch,
                        inferH,
                        inferW,
                        roiD,
                        roiH,
                        roiW,
                        featureChannels,
                        localStartDepth,
                        startH,
                        startW);

                    if (patchIndex == 1 && enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                        DumpPinnedBlobOutputs(infer, pinnedBlobNames);

                    await HandleSlidingWindowPatchCompletedAsync(
                        patchIndex,
                        patchCount,
                        0.64f,
                        "Run MONAI feature patch ",
                        ct);
                    await MaybeRunSlidingWindowMaintenanceAsync(patchIndex, patchCount, "feature_patch", ct);
                }
            }

            var nextDepthStart = (iz + 1) < startsD.Count ? startsD[iz + 1] : inferD;
            var finalizedDepthCount = Mathf.Clamp(nextDepthStart - bandBaseDepth, 0, activeBandDepth);
            if (finalizedDepthCount <= 0)
                continue;

            NormalizeAccumulatedBandRange(featureBand, weightBand, featureChannels, bandDepthCapacity, finalizedDepthCount, fullPlane);
            FillLabelMapFromFeatureBand(
                featureBand,
                bandDepthCapacity,
                0,
                finalizedDepthCount,
                inferW,
                inferH,
                outputHead,
                labelMap,
                bandBaseDepth,
                ct);
            SlideAccumulatedFeatureBand(
                featureBand,
                weightBand,
                featureChannels,
                bandDepthCapacity,
                fullPlane,
                finalizedDepthCount,
                ref activeBandDepth);

            bandBaseDepth += finalizedDepthCount;
            AppendDebugLine(
                "RunSlidingWindowFeatureAggregationInference flush_depth"
                + " | flushedDepth=" + finalizedDepthCount.ToString(CultureInfo.InvariantCulture)
                + " | nextBandBaseDepth=" + bandBaseDepth.ToString(CultureInfo.InvariantCulture)
                + " | remainingActiveDepth=" + activeBandDepth.ToString(CultureInfo.InvariantCulture));
        }

        if (patchCount <= 0)
            throw new InvalidOperationException("Sliding window feature inference did not produce any patches.");
        if (bandBaseDepth != inferD || activeBandDepth != 0)
        {
            throw new InvalidOperationException(
                "Sliding window feature band drain incomplete: bandBaseDepth="
                + bandBaseDepth.ToString(CultureInfo.InvariantCulture)
                + " fullDepth=" + inferD.ToString(CultureInfo.InvariantCulture)
                + " activeBandDepth=" + activeBandDepth.ToString(CultureInfo.InvariantCulture));
        }

        ReportProgress(0.70f, "Classify aggregated MONAI features");

        CaptureLatestLayerRuntimeProfile(ResolvePathMode(useTextureInputForMonaiPatches));
        return new InferenceRunResult
        {
            labelMap = labelMap,
            width = inferW,
            height = inferH,
            depth = inferD,
            channels = outputHead.outputChannels,
            executionNote = "sliding_window_feature_head_band:" + outputHead.featureBlobName,
            pathMode = ResolvePathMode(useTextureInputForMonaiPatches),
            executedPatchCount = patchCount,
            totalPatchCount = patchCount
        };
    }

    private async UniTask<InferenceRunResult> RunSlidingWindowFeatureOwnershipInferenceAsync(
        ResolvedRequest resolved,
        PreparedInput prepared,
        OutputHeadInfo outputHead,
        CancellationToken ct)
    {
        if (outputHead == null || outputHead.conv == null)
            throw new ArgumentNullException(nameof(outputHead));

        var roiD = resolved.slidingWindowDepth;
        var roiH = resolved.slidingWindowHeight;
        var roiW = resolved.slidingWindowWidth;
        var inferD = resolved.processedInputDepth;
        var inferH = resolved.processedInputHeight;
        var inferW = resolved.processedInputWidth;
        var inputChannels = resolved.inputChannels;
        var featureChannels = outputHead.featureChannels;

        if (roiD <= 0 || roiH <= 0 || roiW <= 0)
            throw new InvalidOperationException("Sliding window roi size is invalid.");
        if (featureChannels <= 0 || outputHead.outputChannels <= 0)
            throw new InvalidOperationException("Output head channels are invalid.");

        AppendDebugLine(
            "RunSlidingWindowFeatureOwnershipInference begin"
            + " | featureBlob=" + outputHead.featureBlobName
            + " | featureChannels=" + featureChannels.ToString(CultureInfo.InvariantCulture)
            + " | outputChannels=" + outputHead.outputChannels.ToString(CultureInfo.InvariantCulture)
            + " | headLayer=" + (outputHead.layerName ?? string.Empty));

        var startsD = BuildSlidingWindowStarts(inferD, roiD, resolved.slidingWindowOverlap);
        var startsH = BuildSlidingWindowStarts(inferH, roiH, resolved.slidingWindowOverlap);
        var startsW = BuildSlidingWindowStarts(inferW, roiW, resolved.slidingWindowOverlap);
        var pinnedBlobNames = BuildPinnedBlobNames(outputHead.featureBlobName, resolved.debugPinnedBlobNames);
        var patchCount = startsD.Count * startsH.Count * startsW.Count;
        var patchIndex = 0;
        var fullVoxelCount = checked(inferD * inferH * inferW);
        var labelMap = new ushort[fullVoxelCount];

        ThrowIfSlidingWindowMemoryLimitExceeded("before_feature_ownership_alloc", 0, patchCount);

        using var inputTensor = new NcnnTensorBuffer(roiW, roiH, roiD, inputChannels);
        var patchTensor = new float[checked(inputChannels * roiD * roiH * roiW)];
        for (var iz = 0; iz < startsD.Count; iz++)
        {
            var startD = startsD[iz];
            var ownedD = ComputeOwnedInterval(startsD, iz, roiD, inferD);
            for (var iy = 0; iy < startsH.Count; iy++)
            {
                var startH = startsH[iy];
                var ownedH = ComputeOwnedInterval(startsH, iy, roiH, inferH);
                for (var ix = 0; ix < startsW.Count; ix++)
                {
                    ct.ThrowIfCancellationRequested();
                    var startW = startsW[ix];
                    var ownedW = ComputeOwnedInterval(startsW, ix, roiW, inferW);
                    patchIndex++;

                    ExtractPatchNcdhw(
                        prepared.tensorNcdhw,
                        inputChannels,
                        inferD,
                        inferH,
                        inferW,
                        startD,
                        startH,
                        startW,
                        roiD,
                        roiH,
                        roiW,
                        patchTensor);
                    inputTensor.buffer.SetData(patchTensor);

                    using var inferHandle = RunInferenceWithPatchInput(
                        resolved,
                        inputTensor,
                        roiD,
                        roiH,
                        roiW,
                        pinnedBlobNames,
                        outputHead.featureBlobName);
                    var infer = inferHandle.infer;

                    var featureView = GetInferOutputShape(infer, outputHead.featureBlobName, "MONAI sliding window feature blob missing: ");
                    if (featureView.dims != 4)
                        throw new InvalidOperationException("MONAI sliding window feature dims expected 4 but got " + featureView.dims);
                    if (featureView.w != roiW || featureView.h != roiH || featureView.d != roiD || featureView.c != featureChannels)
                    {
                        throw new InvalidOperationException(
                            "Sliding window feature shape mismatch: expected "
                            + roiW + "x" + roiH + "x" + roiD + "x" + featureChannels
                            + " got "
                            + featureView.w + "x" + featureView.h + "x" + featureView.d + "x" + featureView.c);
                    }

                    var featureData = ExtractInferOutputData(infer, outputHead.featureBlobName, featureView);
                    var patchLabelMap = BuildLabelMapFromAggregatedFeatures(featureData, roiW, roiH, roiD, outputHead, null, ct);
                    CopyOwnedPatchLabelRegion(
                        patchLabelMap,
                        roiD,
                        roiH,
                        roiW,
                        labelMap,
                        inferD,
                        inferH,
                        inferW,
                        startD,
                        startH,
                        startW,
                        ownedD.start,
                        ownedD.end,
                        ownedH.start,
                        ownedH.end,
                        ownedW.start,
                        ownedW.end);

                    if (patchIndex == 1 && enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                        DumpPinnedBlobOutputs(infer, pinnedBlobNames);

                    await HandleSlidingWindowPatchCompletedAsync(
                        patchIndex,
                        patchCount,
                        0.64f,
                        "Run MONAI ownership patch ",
                        ct);
                    await MaybeRunSlidingWindowMaintenanceAsync(patchIndex, patchCount, "feature_ownership_patch", ct);
                }
            }
        }

        CaptureLatestLayerRuntimeProfile(ResolvePathMode(useTextureInputForMonaiPatches));
        return new InferenceRunResult
        {
            labelMap = labelMap,
            width = inferW,
            height = inferH,
            depth = inferD,
            channels = outputHead.outputChannels,
            executionNote = "sliding_window_feature_head_ownership:" + outputHead.featureBlobName,
            pathMode = ResolvePathMode(useTextureInputForMonaiPatches),
            executedPatchCount = patchCount,
            totalPatchCount = patchCount
        };
    }

    private async UniTask<InferenceRunResult> RunSlidingWindowProbeInferenceAsync(ResolvedRequest resolved, PreparedInput prepared, CancellationToken ct)
    {
        var roiD = resolved.slidingWindowDepth;
        var roiH = resolved.slidingWindowHeight;
        var roiW = resolved.slidingWindowWidth;
        var inferD = resolved.processedInputDepth;
        var inferH = resolved.processedInputHeight;
        var inferW = resolved.processedInputWidth;
        var channelCount = resolved.inputChannels;

        if (roiD <= 0 || roiH <= 0 || roiW <= 0)
            throw new InvalidOperationException("Sliding window roi size is invalid.");

        var startsD = BuildSlidingWindowStarts(inferD, roiD, resolved.slidingWindowOverlap);
        var startsH = BuildSlidingWindowStarts(inferH, roiH, resolved.slidingWindowOverlap);
        var startsW = BuildSlidingWindowStarts(inferW, roiW, resolved.slidingWindowOverlap);
        var probeHead = resolved.postprocessKind == MonaiPostprocessKind.MulticlassArgmax
            && TryResolveLinearOutputHead(resolved.outputBlobName, out var slidingProbeOutputHead)
            ? slidingProbeOutputHead
            : null;
        var probeBlobName = probeHead != null ? probeHead.featureBlobName : resolved.outputBlobName;
        var pinnedBlobNames = BuildPinnedBlobNames(probeBlobName, resolved.debugPinnedBlobNames);
        var totalPatchCount = startsD.Count * startsH.Count * startsW.Count;
        var targetPatchOrdinal = Mathf.Clamp(resolved.probePatchOrdinal, 0, totalPatchCount);
        var maxPatchCount = resolved.maxSlidingWindowPatches > 0
            ? Mathf.Min(resolved.maxSlidingWindowPatches, totalPatchCount)
            : 1;
        if (maxPatchCount <= 0)
            maxPatchCount = 1;

        var executedPatchCount = 0;
        var enumeratedPatchOrdinal = 0;
        InferOutputShape? outputView = null;
        var executedStartD = 0;
        var executedStartH = 0;
        var executedStartW = 0;

        using var inputTensor = new NcnnTensorBuffer(roiW, roiH, roiD, channelCount);
        var patchTensor = new float[checked(channelCount * roiD * roiH * roiW)];
        for (var iz = 0; iz < startsD.Count && executedPatchCount < maxPatchCount; iz++)
        {
            var startD = startsD[iz];
            for (var iy = 0; iy < startsH.Count && executedPatchCount < maxPatchCount; iy++)
            {
                var startH = startsH[iy];
                for (var ix = 0; ix < startsW.Count && executedPatchCount < maxPatchCount; ix++)
                {
                    ct.ThrowIfCancellationRequested();
                    var startW = startsW[ix];
                    enumeratedPatchOrdinal++;
                    if (targetPatchOrdinal > 0 && enumeratedPatchOrdinal != targetPatchOrdinal)
                        continue;
                    executedPatchCount++;
                    executedStartD = startD;
                    executedStartH = startH;
                    executedStartW = startW;
                    ExtractPatchNcdhw(
                        prepared.tensorNcdhw,
                        channelCount,
                        inferD,
                        inferH,
                        inferW,
                        startD,
                        startH,
                        startW,
                        roiD,
                        roiH,
                        roiW,
                        patchTensor);
                    inputTensor.buffer.SetData(patchTensor);

                    using var inferHandle = RunInferenceWithPatchInput(
                        resolved,
                        inputTensor,
                        roiD,
                        roiH,
                        roiW,
                        pinnedBlobNames,
                        probeBlobName);
                    var infer = inferHandle.infer;

                    outputView = GetInferOutputShape(infer, probeBlobName, "MONAI sliding window probe blob missing: ");

                    if (executedPatchCount == 1 && enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                        DumpPinnedBlobOutputs(infer, pinnedBlobNames);

                    await HandleSlidingWindowPatchCompletedAsync(
                        executedPatchCount,
                        maxPatchCount,
                        0.62f,
                        "Run MONAI probe patch ",
                        ct);
                    await MaybeRunSlidingWindowMaintenanceAsync(executedPatchCount, maxPatchCount, "probe_patch", ct);
                }
            }
        }

        if (!outputView.HasValue)
            throw new InvalidOperationException("Sliding window probe did not execute any patches.");

        CaptureLatestLayerRuntimeProfile(ResolvePathMode(useTextureInputForMonaiPatches));
        return new InferenceRunResult
        {
            width = outputView.Value.w,
            height = outputView.Value.h,
            depth = outputView.Value.d,
            channels = probeHead != null ? probeHead.outputChannels : outputView.Value.c,
            executionNote =
                "probe_blob:" + probeBlobName
                + " | patch_ordinal=" + (targetPatchOrdinal > 0 ? targetPatchOrdinal : executedPatchCount).ToString(CultureInfo.InvariantCulture)
                + " | patch_start_dhw=" + executedStartD.ToString(CultureInfo.InvariantCulture)
                + "," + executedStartH.ToString(CultureInfo.InvariantCulture)
                + "," + executedStartW.ToString(CultureInfo.InvariantCulture),
            pathMode = ResolvePathMode(useTextureInputForMonaiPatches),
            probeOnly = true,
            executedPatchCount = executedPatchCount,
            totalPatchCount = totalPatchCount
        };
    }

    private PatchInferHandle RunInferenceWithPatchInput(
        ResolvedRequest resolved,
        NcnnTensorBuffer inputTensor,
        int depth,
        int height,
        int width,
        string[] pinnedBlobNames,
        string stopAfterTopName)
    {
        if (_repro == null)
            throw new InvalidOperationException("MONAI ncnn repro is not initialized.");
        if (inputTensor == null || inputTensor.buffer == null)
            throw new ArgumentNullException(nameof(inputTensor));

        if (!useTextureInputForMonaiPatches)
        {
            return new PatchInferHandle(
                _repro.InferWithMultiInputs(
                null,
                new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal) { { resolved.inputBlobName, inputTensor } },
                pinnedBlobNames,
                null,
                stopAfterTopName),
                _repro,
                null);
        }

        var packCount = Mathf.Max(1, Mathf.CeilToInt(resolved.inputChannels / 4f));
        var sliceCount = checked(Mathf.Max(1, depth) * packCount);
        var inputPack4 = _repro.RentTempArray(width, height, sliceCount, NcnnRepro.ResolveTensorTextureFormat(4));
        try
        {
            _ops.FillPack4FromBufferCDHW(inputTensor.buffer, width, height, depth, resolved.inputChannels, inputPack4);
            TryLogPatchInputTextureRoundtrip(resolved, inputTensor, width, height, depth, inputPack4);
            var textureInputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
            {
                { resolved.inputBlobName, inputPack4 }
            };
            var textureInputShapes = new Dictionary<string, NcnnRepro.BufferShape>(StringComparer.Ordinal)
            {
                { resolved.inputBlobName, new NcnnRepro.BufferShape(4, width, height, depth, resolved.inputChannels) }
            };
            var infer = _repro.InferWithMultiInputs(
                textureInputs,
                null,
                pinnedBlobNames,
                textureInputShapes,
                stopAfterTopName);
            return new PatchInferHandle(infer, _repro, inputPack4);
        }
        catch
        {
            _repro.ReturnTempArray(inputPack4);
            throw;
        }
    }

    private void TryLogPatchInputTextureRoundtrip(
        ResolvedRequest resolved,
        NcnnTensorBuffer inputTensor,
        int width,
        int height,
        int depth,
        RenderTexture inputPack4)
    {
        if (_loggedPatchInputTextureRoundtrip
            || !enableDebugDump
            || resolved == null
            || !resolved.probeOnly
            || inputTensor == null
            || inputTensor.buffer == null
            || inputPack4 == null
            || _repro == null
            || _ops == null)
        {
            return;
        }

        _loggedPatchInputTextureRoundtrip = true;

        var count = checked(width * height * depth * Mathf.Max(1, resolved.inputChannels));
        var roundtrip = _repro.RentTempBuffer(count, sizeof(float));
        try
        {
            _ops.DebugSyncGpu();
            _ops.Pack4ToBufferCDHW(inputPack4, width, height, depth, resolved.inputChannels, roundtrip);
            _ops.DebugSyncGpu();

            var src = new float[count];
            var dst = new float[count];
            inputTensor.buffer.GetData(src);
            roundtrip.GetData(dst);
            var cpuSliceDst = ReadPack4TextureToNcdhwCpu(inputPack4, width, height, depth, resolved.inputChannels);

            var srcNz = 0;
            var dstNz = 0;
            var cpuNz = 0;
            double sumAbs = 0d;
            double cpuSumAbs = 0d;
            float maxAbs = 0f;
            float cpuMaxAbs = 0f;
            for (var i = 0; i < count; i++)
            {
                var a = src[i];
                var b = dst[i];
                var c = cpuSliceDst[i];
                if (a != 0f)
                    srcNz++;
                if (b != 0f)
                    dstNz++;
                if (c != 0f)
                    cpuNz++;
                var diff = Mathf.Abs(a - b);
                var cpuDiff = Mathf.Abs(a - c);
                sumAbs += diff;
                cpuSumAbs += cpuDiff;
                if (diff > maxAbs)
                    maxAbs = diff;
                if (cpuDiff > cpuMaxAbs)
                    cpuMaxAbs = cpuDiff;
            }

            AppendDebugLine(
                "PatchInputTextureRoundtrip"
                + " | shape=" + width.ToString(CultureInfo.InvariantCulture)
                + "x" + height.ToString(CultureInfo.InvariantCulture)
                + "x" + depth.ToString(CultureInfo.InvariantCulture)
                + "x" + resolved.inputChannels.ToString(CultureInfo.InvariantCulture)
                + " | tex=" + inputPack4.width.ToString(CultureInfo.InvariantCulture)
                + "x" + inputPack4.height.ToString(CultureInfo.InvariantCulture)
                + "x" + inputPack4.volumeDepth.ToString(CultureInfo.InvariantCulture)
                + " | srcNz=" + srcNz.ToString(CultureInfo.InvariantCulture)
                + "/" + count.ToString(CultureInfo.InvariantCulture)
                + " | dstNz=" + dstNz.ToString(CultureInfo.InvariantCulture)
                + "/" + count.ToString(CultureInfo.InvariantCulture)
                + " | cpuNz=" + cpuNz.ToString(CultureInfo.InvariantCulture)
                + "/" + count.ToString(CultureInfo.InvariantCulture)
                + " | meanAbs=" + (sumAbs / Math.Max(1, count)).ToString("G9", CultureInfo.InvariantCulture)
                + " | maxAbs=" + maxAbs.ToString("G9", CultureInfo.InvariantCulture)
                + " | cpuMeanAbs=" + (cpuSumAbs / Math.Max(1, count)).ToString("G9", CultureInfo.InvariantCulture)
                + " | cpuMaxAbs=" + cpuMaxAbs.ToString("G9", CultureInfo.InvariantCulture)
                + " | src0=" + src[0].ToString("G9", CultureInfo.InvariantCulture)
                + " | dst0=" + dst[0].ToString("G9", CultureInfo.InvariantCulture)
                + " | cpu0=" + cpuSliceDst[0].ToString("G9", CultureInfo.InvariantCulture));
        }
        finally
        {
            _repro.ReturnTempBuffer(roundtrip);
        }
    }

    private static float[] ReadPack4TextureToNcdhwCpu(RenderTexture inputPack4, int width, int height, int depth, int channels)
    {
        if (inputPack4 == null)
            throw new ArgumentNullException(nameof(inputPack4));

        var packCount = Mathf.Max(1, Mathf.CeilToInt(channels / 4f));
        var sliceCount = checked(Mathf.Max(1, depth) * packCount);
        var plane = checked(width * height);
        var data = new float[checked(plane * Math.Max(1, depth) * Math.Max(1, channels))];
        var prevActive = RenderTexture.active;
        var tex2D = new Texture2D(width, height, TextureFormat.RGBAFloat, false, true);
        try
        {
            for (var slice = 0; slice < sliceCount; slice++)
            {
                Graphics.SetRenderTarget(inputPack4, 0, CubemapFace.Unknown, slice);
                tex2D.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
                tex2D.Apply(false, false);
                var pixels = tex2D.GetPixels();

                var z = slice / packCount;
                var pack = slice - z * packCount;
                for (var i = 0; i < pixels.Length; i++)
                {
                    var c0 = pack * 4 + 0;
                    var c1 = pack * 4 + 1;
                    var c2 = pack * 4 + 2;
                    var c3 = pack * 4 + 3;
                    var baseIndex0 = ((c0 * depth + z) * plane) + i;
                    if (c0 < channels)
                        data[baseIndex0] = pixels[i].r;
                    if (c1 < channels)
                        data[((c1 * depth + z) * plane) + i] = pixels[i].g;
                    if (c2 < channels)
                        data[((c2 * depth + z) * plane) + i] = pixels[i].b;
                    if (c3 < channels)
                        data[((c3 * depth + z) * plane) + i] = pixels[i].a;
                }
            }

            return data;
        }
        finally
        {
            RenderTexture.active = prevActive;
            UnityEngine.Object.DestroyImmediate(tex2D);
        }
    }

    private InferOutputShape GetInferOutputShape(NcnnRepro.InferResult infer, string blobName, string missingPrefix)
    {
        if (infer == null)
            throw new ArgumentNullException(nameof(infer));

        if (infer.TryGetExistingTexture(blobName, out var existingTexture) && existingTexture != null
            && infer.TryGetLogicalShape(blobName, out var textureDims, out var textureW, out var textureH, out var textureD, out var textureC))
        {
            return new InferOutputShape(textureDims, textureW, textureH, textureD, textureC);
        }

        if (infer.TryGetExistingBufferView(blobName, out var existingView) && existingView != null && existingView.buffer != null)
            return new InferOutputShape(existingView.dims, existingView.w, existingView.h, existingView.d, existingView.c);

        if (infer.TryGetLogicalShape(blobName, out var dims, out var w, out var h, out var d, out var c))
            return new InferOutputShape(dims, w, h, d, c);

        try
        {
            var view = infer.GetBufferView(blobName);
            if (view != null && view.buffer != null)
                return new InferOutputShape(view.dims, view.w, view.h, view.d, view.c);
        }
        catch (Exception e)
        {
            AppendDebugLine("GetInferOutputShape buffer path unavailable | blob=" + blobName + " | " + e.Message);
        }

        throw new InvalidOperationException((missingPrefix ?? "Infer output missing: ") + blobName);
    }

    private float[] ExtractInferOutputData(NcnnRepro.InferResult infer, string blobName, InferOutputShape outputView)
    {
        if (infer == null)
            throw new ArgumentNullException(nameof(infer));

        if (infer.TryGetExistingBufferView(blobName, out var existingView) && existingView != null && existingView.buffer != null)
            return infer.GetBufferData(blobName);

        var hasExistingTexture = infer.TryGetExistingTexture(blobName, out var texture) && texture != null;
        if (hasExistingTexture)
            return ReadTextureOutputData(
                texture,
                outputView,
                recordTimingDiagnostics: enableTimingSplitDiagnostics && !_timingSplitReadbackDiagnosticCaptured);

        try
        {
            return infer.GetBufferData(blobName);
        }
        catch (Exception e)
        {
            AppendDebugLine("ExtractInferOutputData buffer path unavailable | blob=" + blobName + " | " + e.Message);
        }

        if (!hasExistingTexture)
            texture = infer.ExtractTexture(blobName);
        if (texture == null)
            throw new InvalidOperationException("Infer texture output missing: " + blobName);

        try
        {
            return ReadTextureOutputData(
                texture,
                outputView,
                recordTimingDiagnostics: enableTimingSplitDiagnostics && !_timingSplitReadbackDiagnosticCaptured);
        }
        finally
        {
            if (!hasExistingTexture)
                _repro?.ReturnTempArray(texture);
        }
    }

    private float[] ReadTextureOutputData(RenderTexture texture, InferOutputShape outputView, bool recordTimingDiagnostics = false)
    {
        if (texture == null)
            throw new ArgumentNullException(nameof(texture));

        var count = checked(Mathf.Max(1, outputView.w) * Mathf.Max(1, outputView.h) * Mathf.Max(1, outputView.d) * Mathf.Max(1, outputView.c));
        using var outputBuffer = new ComputeBuffer(count, sizeof(float), ComputeBufferType.Structured);
        var packDispatchSw = recordTimingDiagnostics ? Stopwatch.StartNew() : null;
        if (outputView.dims == 4)
            _ops.Pack4ToBufferCDHW(texture, outputView.w, outputView.h, outputView.d, outputView.c, outputBuffer);
        else
            _ops.Pack4ToBufferCHW(texture, outputView.w, outputView.h, outputView.c, outputBuffer);
        packDispatchSw?.Stop();

        double syncBeforeGetDataMs = 0d;
        if (recordTimingDiagnostics)
        {
            var syncSw = Stopwatch.StartNew();
            _ops?.DebugSyncGpu();
            syncSw.Stop();
            syncBeforeGetDataMs = StopwatchToMilliseconds(syncSw);
        }

        var data = new float[count];
        var getDataSw = recordTimingDiagnostics ? Stopwatch.StartNew() : null;
        outputBuffer.GetData(data);
        getDataSw?.Stop();

        if (recordTimingDiagnostics)
        {
            var packDispatchMs = StopwatchToMilliseconds(packDispatchSw);
            var getDataMs = StopwatchToMilliseconds(getDataSw);
            RecordDiagnosticTiming("diag_output_pack_dispatch_ms", packDispatchMs);
            RecordDiagnosticTiming("diag_output_sync_before_getdata_ms", syncBeforeGetDataMs);
            RecordDiagnosticTiming("diag_output_getdata_ms", getDataMs);
            RecordDiagnosticTiming("diag_output_total_readback_ms", packDispatchMs + syncBeforeGetDataMs + getDataMs);
            _timingSplitReadbackDiagnosticCaptured = true;
            AppendDebugLine(
                "TimingSplitDiagnostic readback"
                + " | dims=" + outputView.dims.ToString(CultureInfo.InvariantCulture)
                + " | shape=" + outputView.w.ToString(CultureInfo.InvariantCulture)
                + "x" + outputView.h.ToString(CultureInfo.InvariantCulture)
                + "x" + outputView.d.ToString(CultureInfo.InvariantCulture)
                + "x" + outputView.c.ToString(CultureInfo.InvariantCulture)
                + " | pack_dispatch_ms=" + packDispatchMs.ToString("0.###", CultureInfo.InvariantCulture)
                + " | sync_before_getdata_ms=" + syncBeforeGetDataMs.ToString("0.###", CultureInfo.InvariantCulture)
                + " | getdata_ms=" + getDataMs.ToString("0.###", CultureInfo.InvariantCulture));
        }
        return data;
    }

    private async UniTask HandleSlidingWindowPatchCompletedAsync(
        int patchIndex,
        int patchCount,
        float progressEnd,
        string progressPrefix,
        CancellationToken ct)
    {
        if (clearTempPoolAfterEachSlidingWindowPatch
            && _repro != null
            && slidingWindowTempPoolClearInterval > 0
            && (patchIndex % slidingWindowTempPoolClearInterval) == 0)
        {
            _repro.ClearTempPool();
        }

        ReportProgress(
            Mathf.Lerp(0.42f, progressEnd, patchCount > 0 ? patchIndex / (float)patchCount : 1f),
            (progressPrefix ?? "Run MONAI patch ")
            + patchIndex.ToString(CultureInfo.InvariantCulture)
            + "/"
            + patchCount.ToString(CultureInfo.InvariantCulture));

        if (slidingWindowYieldInterval > 0
            && (patchIndex % slidingWindowYieldInterval) == 0
            && patchIndex < patchCount)
        {
            RenderTexture.active = null;
            await UniTask.Yield(PlayerLoopTiming.Update, ct);
            RenderTexture.active = null;
        }
    }

    private async UniTask MaybeRunSlidingWindowMaintenanceAsync(
        int patchIndex,
        int patchCount,
        string stage,
        CancellationToken ct)
    {
        if (slidingWindowResourceSnapshotInterval > 0
            && (patchIndex % slidingWindowResourceSnapshotInterval) == 0)
        {
            var snapshot = BuildSlidingWindowResourceSnapshot(stage, patchIndex, patchCount);
            _resourceSnapshotLines.Add(snapshot);
            AppendDebugLine(snapshot);
        }

        ThrowIfSlidingWindowMemoryLimitExceeded(stage, patchIndex, patchCount);

        if (slidingWindowManagedCleanupInterval > 0
            && (patchIndex % slidingWindowManagedCleanupInterval) == 0)
        {
            RenderTexture.active = null;
            await UniTask.Yield(PlayerLoopTiming.Update, ct);
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            RenderTexture.active = null;
            await UniTask.Yield(PlayerLoopTiming.Update, ct);

            if (slidingWindowResourceSnapshotInterval > 0
                && (patchIndex % slidingWindowResourceSnapshotInterval) == 0)
            {
                var snapshot = BuildSlidingWindowResourceSnapshot(stage + "_after_gc", patchIndex, patchCount);
                _resourceSnapshotLines.Add(snapshot);
                AppendDebugLine(snapshot);
            }

            ThrowIfSlidingWindowMemoryLimitExceeded(stage + "_after_gc", patchIndex, patchCount);
        }
    }

    private string BuildSlidingWindowResourceSnapshot(string stage, int patchIndex, int patchCount)
    {
        GetProcessMemorySnapshotMb(out var privateMb, out var workingSetMb, out var managedMb);
        var gfxMb = TryGetGraphicsDriverMemoryMb();
        var rtCount = Resources.FindObjectsOfTypeAll<RenderTexture>().Length;
        var text =
            "SlidingWindowResources"
            + " | stage=" + (stage ?? string.Empty)
            + " | patch=" + patchIndex.ToString(CultureInfo.InvariantCulture)
            + "/" + patchCount.ToString(CultureInfo.InvariantCulture)
            + " | privateMb=" + privateMb.ToString("F3", CultureInfo.InvariantCulture)
            + " | workingSetMb=" + workingSetMb.ToString("F3", CultureInfo.InvariantCulture)
            + " | managedMb=" + managedMb.ToString("F3", CultureInfo.InvariantCulture)
            + " | gfxMb=" + gfxMb.ToString("F3", CultureInfo.InvariantCulture)
            + " | rtCount=" + rtCount.ToString(CultureInfo.InvariantCulture);
        if (_repro != null)
            text += " | gpu=" + NcnnGpuResourceTracker.BuildSummary();
        return text;
    }

    private void ThrowIfSlidingWindowMemoryLimitExceeded(string stage, int patchIndex, int patchCount)
    {
        if (slidingWindowAbortIfPrivateMemoryExceedsMb <= 0f)
            return;

        GetProcessMemorySnapshotMb(out var privateMb, out _, out _);
        if (privateMb <= slidingWindowAbortIfPrivateMemoryExceedsMb)
            return;

        var snapshot = BuildSlidingWindowResourceSnapshot(stage + "_abort", patchIndex, patchCount);
        _resourceSnapshotLines.Add(snapshot);
        AppendDebugLine(snapshot);
        throw new InvalidOperationException(
            "Abort MONAI sliding window due to private memory limit"
            + " | limitMb=" + slidingWindowAbortIfPrivateMemoryExceedsMb.ToString("F3", CultureInfo.InvariantCulture)
            + " | privateMb=" + privateMb.ToString("F3", CultureInfo.InvariantCulture)
            + " | stage=" + (stage ?? string.Empty));
    }

    private static double TryGetGraphicsDriverMemoryMb()
    {
        try
        {
            return Profiler.GetAllocatedMemoryForGraphicsDriver() / (1024d * 1024d);
        }
        catch
        {
            return -1d;
        }
    }

    private JObject BuildResourceStatsJson()
    {
        GetProcessMemorySnapshotMb(out var privateMb, out var workingSetMb, out var managedMb);
        var gfxMb = TryGetGraphicsDriverMemoryMb();
        var rtCount = Resources.FindObjectsOfTypeAll<RenderTexture>().Length;
        var gpu = NcnnGpuResourceTracker.GetStatsSnapshot();

        return new JObject
        {
            ["path_mode"] = _lastPathMode ?? string.Empty,
            ["process_private_mb"] = privateMb,
            ["process_working_set_mb"] = workingSetMb,
            ["managed_heap_mb"] = managedMb,
            ["graphics_driver_mb"] = gfxMb,
            ["unity_rendertexture_object_count"] = rtCount,
            ["gpu_tracker"] = new JObject
            {
                ["current_total_mb"] = gpu.currentTotalMb,
                ["current_buffers_mb"] = gpu.currentBufferMb,
                ["current_rendertextures_mb"] = gpu.currentTextureMb,
                ["peak_total_mb"] = gpu.peakTotalMb,
                ["peak_buffers_mb"] = gpu.peakBufferMb,
                ["peak_rendertextures_mb"] = gpu.peakTextureMb,
                ["live_buffer_count"] = gpu.liveBufferCount,
                ["live_rendertexture_count"] = gpu.liveTextureCount,
                ["peak_buffer_count"] = gpu.peakBufferCount,
                ["peak_rendertexture_count"] = gpu.peakTextureCount,
                ["low_memory_warning_count"] = gpu.lowMemoryWarningCount
            }
        };
    }

    private void WriteMonaiDiagnosticsFiles()
    {
        if (!enableDebugDump || string.IsNullOrWhiteSpace(_lastDumpDir))
            return;

        try
        {
            var timings = new JObject();
            foreach (var kv in _timingMs)
                timings[kv.Key] = kv.Value;
            foreach (var kv in _diagnosticTimingMs)
                timings[kv.Key] = kv.Value;
            timings["total_elapsed_ms"] = _timingMs.ContainsKey("total_elapsed_ms")
                ? _timingMs["total_elapsed_ms"]
                : 0L;
            timings["path_mode"] = _lastPathMode ?? string.Empty;
            if (_lastLayerRuntimeProfile != null)
            {
                timings["layer_profile_inference_index"] = _lastLayerRuntimeProfile.inferenceIndex;
                timings["layer_profile_path_kind"] = _lastLayerRuntimeProfile.pathKind ?? string.Empty;
                timings["layer_profile_total_ms"] = _lastLayerRuntimeProfile.totalMs;
            }
            WriteText(Path.Combine(_lastDumpDir, "timings.json"), timings.ToString());

            if (!string.IsNullOrWhiteSpace(_lastLayerRuntimeProfileText))
                WriteText(Path.Combine(_lastDumpDir, "layer_runtime_profile.tsv"), _lastLayerRuntimeProfileText);

            if (_resourceSnapshotLines.Count > 0)
                WriteText(Path.Combine(_lastDumpDir, "resource_snapshots.txt"), string.Join(Environment.NewLine, _resourceSnapshotLines));

            WriteText(Path.Combine(_lastDumpDir, "resource_stats.json"), BuildResourceStatsJson().ToString());
        }
        catch (Exception e)
        {
            AppendDebugLine("WriteMonaiDiagnosticsFiles failed | " + e.Message);
        }
    }

    private static void GetProcessMemorySnapshotMb(out double privateMb, out double workingSetMb, out double managedMb)
    {
        privateMb = 0d;
        workingSetMb = 0d;
        managedMb = GC.GetTotalMemory(false) / (1024d * 1024d);

        try
        {
            var process = Process.GetCurrentProcess();
            privateMb = process.PrivateMemorySize64 / (1024d * 1024d);
            workingSetMb = process.WorkingSet64 / (1024d * 1024d);
        }
        catch
        {
        }

        try
        {
            var reservedMb = Profiler.GetTotalReservedMemoryLong() / (1024d * 1024d);
            var allocatedMb = Profiler.GetTotalAllocatedMemoryLong() / (1024d * 1024d);
            if (privateMb <= 0d)
                privateMb = Math.Max(allocatedMb, reservedMb);
            if (workingSetMb <= 0d)
                workingSetMb = Math.Max(allocatedMb, reservedMb);
            if (managedMb <= 0d)
                managedMb = allocatedMb;
        }
        catch
        {
        }
    }

    private static string FormatMiBFromFloatCount(int count)
    {
        return ((count * (double)sizeof(float)) / (1024d * 1024d)).ToString("F3", CultureInfo.InvariantCulture);
    }

    private static void AccumulatePatchToDepthBand(
        float[] accum,
        float[] weight,
        int depthCapacity,
        float[] patch,
        int fullHeight,
        int fullWidth,
        int patchDepth,
        int patchHeight,
        int patchWidth,
        int channels,
        int startDepthLocal,
        int startHeight,
        int startWidth)
    {
        AccumulatePatchIntoVolume(
            accum,
            weight,
            depthCapacity,
            fullHeight,
            fullWidth,
            patch,
            patchDepth,
            patchHeight,
            patchWidth,
            channels,
            startDepthLocal,
            startHeight,
            startWidth);
    }

    private static void NormalizeAccumulatedBandRange(
        float[] accum,
        float[] weight,
        int channels,
        int depthCapacity,
        int depthCount,
        int plane)
    {
        var bandVolume = checked(depthCapacity * plane);
        var voxelCount = checked(depthCount * plane);
        for (var voxel = 0; voxel < voxelCount; voxel++)
        {
            var w = weight[voxel];
            if (w <= 1e-12f)
                continue;

            var inv = 1f / w;
            for (var c = 0; c < channels; c++)
                accum[(c * bandVolume) + voxel] *= inv;
        }
    }

    private void FillLabelMapFromFeatureBand(
        float[] featureBand,
        int sourceDepthCapacity,
        int localStartDepth,
        int depthCount,
        int width,
        int height,
        OutputHeadInfo outputHead,
        ushort[] labelMap,
        int labelMapStartDepth,
        CancellationToken ct)
    {
        if (featureBand == null)
            throw new ArgumentNullException(nameof(featureBand));
        if (labelMap == null)
            throw new ArgumentNullException(nameof(labelMap));
        if (outputHead?.conv == null)
            throw new ArgumentNullException(nameof(outputHead));
        if (depthCount <= 0)
            return;

        var plane = checked(width * height);
        var featureChannels = outputHead.featureChannels;
        var outputChannels = outputHead.outputChannels;
        var baseChunkDepth = Math.Max(1, Math.Min(8, depthCount));
        var baseChunkVoxelCount = checked(baseChunkDepth * plane);

        var featureChunk = new float[checked(featureChannels * baseChunkVoxelCount)];
        var logitsChunk = new float[checked(outputChannels * baseChunkVoxelCount)];
        using var featureTensor = new NcnnTensorBuffer(width, height, baseChunkDepth, featureChannels);
        using var logitsTensor = new NcnnTensorBuffer(width, height, baseChunkDepth, outputChannels);

        for (var relativeStartDepth = 0; relativeStartDepth < depthCount; relativeStartDepth += baseChunkDepth)
        {
            ct.ThrowIfCancellationRequested();
            var chunkDepth = Math.Min(baseChunkDepth, depthCount - relativeStartDepth);
            var chunkVoxelCount = checked(chunkDepth * plane);
            var featureCount = checked(featureChannels * chunkVoxelCount);
            var logitsCount = checked(outputChannels * chunkVoxelCount);

            if (chunkDepth == baseChunkDepth)
            {
                for (var c = 0; c < featureChannels; c++)
                {
                    var srcOffset = checked(((c * sourceDepthCapacity) + localStartDepth + relativeStartDepth) * plane);
                    var dstOffset = c * chunkVoxelCount;
                    Array.Copy(featureBand, srcOffset, featureChunk, dstOffset, chunkVoxelCount);
                }

                featureTensor.buffer.SetData(featureChunk);
                _ops.Conv3dBuf(
                    featureTensor,
                    outputHead.conv.rawWeight,
                    outputHead.conv.rawBias,
                    outputChannels,
                    1, 1, 1,
                    1, 1, 1,
                    0, 0, 0, 0, 0, 0,
                    1, 1, 1,
                    outputHead.conv.activationType,
                    outputHead.conv.activationSlope,
                    logitsTensor);
                logitsTensor.buffer.GetData(logitsChunk);
                FillArgmaxLabels(labelMap, labelMapStartDepth + relativeStartDepth, plane, chunkVoxelCount, outputChannels, logitsChunk);
            }
            else
            {
                var featureChunkTail = new float[featureCount];
                var logitsChunkTail = new float[logitsCount];
                using var featureTensorTail = new NcnnTensorBuffer(width, height, chunkDepth, featureChannels);
                using var logitsTensorTail = new NcnnTensorBuffer(width, height, chunkDepth, outputChannels);
                for (var c = 0; c < featureChannels; c++)
                {
                    var srcOffset = checked(((c * sourceDepthCapacity) + localStartDepth + relativeStartDepth) * plane);
                    var dstOffset = c * chunkVoxelCount;
                    Array.Copy(featureBand, srcOffset, featureChunkTail, dstOffset, chunkVoxelCount);
                }

                featureTensorTail.buffer.SetData(featureChunkTail);
                _ops.Conv3dBuf(
                    featureTensorTail,
                    outputHead.conv.rawWeight,
                    outputHead.conv.rawBias,
                    outputChannels,
                    1, 1, 1,
                    1, 1, 1,
                    0, 0, 0, 0, 0, 0,
                    1, 1, 1,
                    outputHead.conv.activationType,
                    outputHead.conv.activationSlope,
                    logitsTensorTail);
                logitsTensorTail.buffer.GetData(logitsChunkTail);
                FillArgmaxLabels(labelMap, labelMapStartDepth + relativeStartDepth, plane, chunkVoxelCount, outputChannels, logitsChunkTail);
            }
        }
    }

    private static void SlideAccumulatedFeatureBand(
        float[] featureBand,
        float[] weightBand,
        int channels,
        int depthCapacity,
        int plane,
        int depthShift,
        ref int activeDepth)
    {
        if (depthShift <= 0)
            return;

        var remainingDepth = Math.Max(0, activeDepth - depthShift);
        var clearDepth = Math.Max(0, depthCapacity - remainingDepth);
        var remainingVoxelCount = checked(remainingDepth * plane);
        var clearVoxelCount = checked(clearDepth * plane);

        if (remainingVoxelCount > 0)
            Array.Copy(weightBand, depthShift * plane, weightBand, 0, remainingVoxelCount);
        if (clearVoxelCount > 0)
            Array.Clear(weightBand, remainingVoxelCount, clearVoxelCount);

        var bandVolume = checked(depthCapacity * plane);
        for (var c = 0; c < channels; c++)
        {
            var channelOffset = c * bandVolume;
            if (remainingVoxelCount > 0)
            {
                Array.Copy(
                    featureBand,
                    channelOffset + depthShift * plane,
                    featureBand,
                    channelOffset,
                    remainingVoxelCount);
            }
            if (clearVoxelCount > 0)
                Array.Clear(featureBand, channelOffset + remainingVoxelCount, clearVoxelCount);
        }

        activeDepth = remainingDepth;
    }

    private ushort[] BuildLabelMapFromAggregatedFeatures(
        float[] featureAccum,
        int width,
        int height,
        int depth,
        OutputHeadInfo outputHead,
        Action<float> reportProgress,
        CancellationToken ct)
    {
        if (featureAccum == null)
            throw new ArgumentNullException(nameof(featureAccum));
        if (outputHead?.conv == null)
            throw new ArgumentNullException(nameof(outputHead));

        var plane = checked(width * height);
        var voxelCount = checked(depth * plane);
        var featureChannels = outputHead.featureChannels;
        var outputChannels = outputHead.outputChannels;
        var baseChunkDepth = Math.Max(1, Math.Min(8, depth));
        var baseChunkVoxelCount = checked(baseChunkDepth * plane);

        var labelMap = new ushort[voxelCount];
        var featureChunk = new float[checked(featureChannels * baseChunkVoxelCount)];
        var logitsChunk = new float[checked(outputChannels * baseChunkVoxelCount)];
        using var featureTensor = new NcnnTensorBuffer(width, height, baseChunkDepth, featureChannels);
        using var logitsTensor = new NcnnTensorBuffer(width, height, baseChunkDepth, outputChannels);

        for (var startZ = 0; startZ < depth; startZ += baseChunkDepth)
        {
            ct.ThrowIfCancellationRequested();
            var chunkDepth = Math.Min(baseChunkDepth, depth - startZ);
            var chunkVoxelCount = checked(chunkDepth * plane);
            var featureCount = checked(featureChannels * chunkVoxelCount);
            var logitsCount = checked(outputChannels * chunkVoxelCount);

            if (chunkDepth == baseChunkDepth)
            {
                for (var c = 0; c < featureChannels; c++)
                {
                    var srcOffset = checked((c * depth + startZ) * plane);
                    var dstOffset = c * chunkVoxelCount;
                    Array.Copy(featureAccum, srcOffset, featureChunk, dstOffset, chunkVoxelCount);
                }

                featureTensor.buffer.SetData(featureChunk);
                _ops.Conv3dBuf(
                    featureTensor,
                    outputHead.conv.rawWeight,
                    outputHead.conv.rawBias,
                    outputChannels,
                    1, 1, 1,
                    1, 1, 1,
                    0, 0, 0, 0, 0, 0,
                    1, 1, 1,
                    outputHead.conv.activationType,
                    outputHead.conv.activationSlope,
                    logitsTensor);
                logitsTensor.buffer.GetData(logitsChunk);
                FillArgmaxLabels(labelMap, startZ, plane, chunkVoxelCount, outputChannels, logitsChunk);
            }
            else
            {
                var featureChunkTail = new float[featureCount];
                var logitsChunkTail = new float[logitsCount];
                using var featureTensorTail = new NcnnTensorBuffer(width, height, chunkDepth, featureChannels);
                using var logitsTensorTail = new NcnnTensorBuffer(width, height, chunkDepth, outputChannels);
                for (var c = 0; c < featureChannels; c++)
                {
                    var srcOffset = checked((c * depth + startZ) * plane);
                    var dstOffset = c * chunkVoxelCount;
                    Array.Copy(featureAccum, srcOffset, featureChunkTail, dstOffset, chunkVoxelCount);
                }

                featureTensorTail.buffer.SetData(featureChunkTail);
                _ops.Conv3dBuf(
                    featureTensorTail,
                    outputHead.conv.rawWeight,
                    outputHead.conv.rawBias,
                    outputChannels,
                    1, 1, 1,
                    1, 1, 1,
                    0, 0, 0, 0, 0, 0,
                    1, 1, 1,
                    outputHead.conv.activationType,
                    outputHead.conv.activationSlope,
                    logitsTensorTail);
                logitsTensorTail.buffer.GetData(logitsChunkTail);
                FillArgmaxLabels(labelMap, startZ, plane, chunkVoxelCount, outputChannels, logitsChunkTail);
            }

            reportProgress?.Invoke(Mathf.Clamp01((startZ + chunkDepth) / (float)Math.Max(1, depth)));
        }

        return labelMap;
    }

    private static void FillArgmaxLabels(
        ushort[] labelMap,
        int startDepth,
        int plane,
        int chunkVoxelCount,
        int outputChannels,
        float[] logitsChunk)
    {
        var dstOffset = startDepth * plane;
        for (var voxel = 0; voxel < chunkVoxelCount; voxel++)
        {
            var bestChannel = 0;
            var bestValue = float.NegativeInfinity;
            for (var c = 0; c < outputChannels; c++)
            {
                var value = logitsChunk[(c * chunkVoxelCount) + voxel];
                if (value > bestValue)
                {
                    bestValue = value;
                    bestChannel = c;
                }
            }

            labelMap[dstOffset + voxel] = (ushort)bestChannel;
        }
    }

    private static (int start, int end) ComputeOwnedInterval(List<int> starts, int axisIndex, int roiSize, int axisSize)
    {
        if (starts == null || starts.Count == 0)
            throw new ArgumentException("Sliding window starts are empty.", nameof(starts));

        var start = starts[axisIndex];
        var ownedStart = axisIndex <= 0 ? 0 : (starts[axisIndex - 1] + start + roiSize) / 2;
        var ownedEnd = axisIndex >= starts.Count - 1
            ? axisSize
            : (start + starts[axisIndex + 1] + roiSize) / 2;
        ownedStart = Mathf.Max(ownedStart, start);
        ownedEnd = Mathf.Min(ownedEnd, start + roiSize, axisSize);
        if (ownedEnd <= ownedStart)
        {
            throw new InvalidOperationException(
                "Invalid ov0 owned interval: axis_size=" + axisSize.ToString(CultureInfo.InvariantCulture)
                + " | roi_size=" + roiSize.ToString(CultureInfo.InvariantCulture)
                + " | axis_index=" + axisIndex.ToString(CultureInfo.InvariantCulture));
        }

        return (ownedStart, ownedEnd);
    }

    private static void CopyOwnedPatchLabelRegion(
        ushort[] patchLabelMap,
        int patchDepth,
        int patchHeight,
        int patchWidth,
        ushort[] dstLabelMap,
        int dstDepth,
        int dstHeight,
        int dstWidth,
        int patchStartDepth,
        int patchStartHeight,
        int patchStartWidth,
        int ownedStartDepth,
        int ownedEndDepth,
        int ownedStartHeight,
        int ownedEndHeight,
        int ownedStartWidth,
        int ownedEndWidth)
    {
        if (patchLabelMap == null || dstLabelMap == null)
            throw new ArgumentNullException(patchLabelMap == null ? nameof(patchLabelMap) : nameof(dstLabelMap));

        var copyDepth = ownedEndDepth - ownedStartDepth;
        var copyHeight = ownedEndHeight - ownedStartHeight;
        var copyWidth = ownedEndWidth - ownedStartWidth;
        if (copyDepth <= 0 || copyHeight <= 0 || copyWidth <= 0)
            return;

        var localStartDepth = ownedStartDepth - patchStartDepth;
        var localStartHeight = ownedStartHeight - patchStartHeight;
        var localStartWidth = ownedStartWidth - patchStartWidth;
        for (var z = 0; z < copyDepth; z++)
        {
            var srcZ = localStartDepth + z;
            var dstZ = ownedStartDepth + z;
            for (var y = 0; y < copyHeight; y++)
            {
                var srcY = localStartHeight + y;
                var dstY = ownedStartHeight + y;
                var srcOffset = ((srcZ * patchHeight) + srcY) * patchWidth + localStartWidth;
                var dstOffset = ((dstZ * dstHeight) + dstY) * dstWidth + ownedStartWidth;
                Array.Copy(patchLabelMap, srcOffset, dstLabelMap, dstOffset, copyWidth);
            }
        }
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

    private static MonaiVolumeData TryLoadReferenceVolumeFromManifest(JObject manifest, VolumeLoadOptions options = null)
    {
        var inputs = manifest?["inputs"] as JArray;
        if (inputs == null || inputs.Count == 0)
            return null;

        var firstPath = inputs[0]?["path"]?.Value<string>();
        if (string.IsNullOrWhiteSpace(firstPath) || !File.Exists(firstPath))
            return null;

        try
        {
            return LoadVolume(firstPath, options);
        }
        catch
        {
            return null;
        }
    }

    private static bool ReferenceVolumeMatchesBaseline(MonaiVolumeData baseline, MonaiVolumeData candidate)
    {
        if (candidate == null)
            return false;
        if (baseline == null)
            return true;
        return baseline.dim0 == candidate.dim0
            && baseline.dim1 == candidate.dim1
            && baseline.dim2 == candidate.dim2;
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

    private static List<int> BuildSlidingWindowStarts(int imageSize, int roiSize, float overlap)
    {
        var starts = new List<int>();
        if (roiSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(roiSize));
        if (imageSize <= roiSize)
        {
            starts.Add(0);
            return starts;
        }

        var interval = Mathf.Max(1, Mathf.FloorToInt(roiSize * (1f - Mathf.Clamp01(overlap))));
        var current = 0;
        while (true)
        {
            starts.Add(current);
            if (current + roiSize >= imageSize)
                break;

            var next = current + interval;
            if (next + roiSize >= imageSize)
                next = imageSize - roiSize;
            if (next <= current)
                break;
            current = next;
        }

        var tail = imageSize - roiSize;
        if (starts.Count == 0 || starts[starts.Count - 1] != tail)
            starts.Add(tail);
        return starts;
    }

    private static void ExtractPatchNcdhw(
        float[] src,
        int channels,
        int srcDepth,
        int srcHeight,
        int srcWidth,
        int startDepth,
        int startHeight,
        int startWidth,
        int patchDepth,
        int patchHeight,
        int patchWidth,
        float[] dst)
    {
        Array.Clear(dst, 0, dst.Length);
        var padDepthBefore = Math.Max(0, (patchDepth - srcDepth) / 2);
        var padHeightBefore = Math.Max(0, (patchHeight - srcHeight) / 2);
        var padWidthBefore = Math.Max(0, (patchWidth - srcWidth) / 2);
        var srcPlane = checked(srcHeight * srcWidth);
        var srcVolume = checked(srcDepth * srcPlane);
        var dstPlane = checked(patchHeight * patchWidth);
        var dstVolume = checked(patchDepth * dstPlane);
        for (var c = 0; c < channels; c++)
        {
            var srcChannel = c * srcVolume;
            var dstChannel = c * dstVolume;
            for (var z = 0; z < patchDepth; z++)
            {
                var srcZ = startDepth - padDepthBefore + z;
                if (srcZ < 0 || srcZ >= srcDepth)
                    continue;

                var srcZOffset = srcChannel + srcZ * srcPlane;
                var dstZOffset = dstChannel + z * dstPlane;
                for (var y = 0; y < patchHeight; y++)
                {
                    var srcY = startHeight - padHeightBefore + y;
                    if (srcY < 0 || srcY >= srcHeight)
                        continue;

                    var srcX = startWidth - padWidthBefore;
                    var dstX = 0;
                    if (srcX < 0)
                    {
                        dstX = -srcX;
                        srcX = 0;
                    }

                    var copyWidth = Math.Min(patchWidth - dstX, srcWidth - srcX);
                    if (copyWidth <= 0)
                        continue;

                    var srcOffset = srcZOffset + srcY * srcWidth + srcX;
                    var dstOffset = dstZOffset + y * patchWidth + dstX;
                    Array.Copy(src, srcOffset, dst, dstOffset, copyWidth);
                }
            }
        }
    }

    private static void AccumulatePatchLogits(
        float[] accum,
        float[] weight,
        float[] patch,
        int fullDepth,
        int fullHeight,
        int fullWidth,
        int patchDepth,
        int patchHeight,
        int patchWidth,
        int channels,
        int startDepth,
        int startHeight,
        int startWidth)
    {
        AccumulatePatchIntoVolume(
            accum,
            weight,
            fullDepth,
            fullHeight,
            fullWidth,
            patch,
            patchDepth,
            patchHeight,
            patchWidth,
            channels,
            startDepth,
            startHeight,
            startWidth);
    }

    private static void AccumulatePatchIntoVolume(
        float[] accum,
        float[] weight,
        int fullDepth,
        int fullHeight,
        int fullWidth,
        float[] patch,
        int patchDepth,
        int patchHeight,
        int patchWidth,
        int channels,
        int startDepth,
        int startHeight,
        int startWidth)
    {
        var padDepthBefore = Math.Max(0, (patchDepth - fullDepth) / 2);
        var padHeightBefore = Math.Max(0, (patchHeight - fullHeight) / 2);
        var padWidthBefore = Math.Max(0, (patchWidth - fullWidth) / 2);
        var fullPlane = checked(fullHeight * fullWidth);
        var fullVolume = checked(fullDepth * fullPlane);
        var patchPlane = checked(patchHeight * patchWidth);
        var patchVolume = checked(patchDepth * patchPlane);
        for (var c = 0; c < channels; c++)
        {
            var accumChannel = c * fullVolume;
            var patchChannel = c * patchVolume;
            for (var z = 0; z < patchDepth; z++)
            {
                var dstZ = startDepth - padDepthBefore + z;
                if (dstZ < 0 || dstZ >= fullDepth)
                    continue;

                var accumZOffset = accumChannel + dstZ * fullPlane;
                var weightZOffset = dstZ * fullPlane;
                var patchZOffset = patchChannel + z * patchPlane;
                for (var y = 0; y < patchHeight; y++)
                {
                    var dstY = startHeight - padHeightBefore + y;
                    if (dstY < 0 || dstY >= fullHeight)
                        continue;

                    var dstX = startWidth - padWidthBefore;
                    var patchX = 0;
                    if (dstX < 0)
                    {
                        patchX = -dstX;
                        dstX = 0;
                    }

                    var copyWidth = Math.Min(patchWidth - patchX, fullWidth - dstX);
                    if (copyWidth <= 0)
                        continue;

                    var accumOffset = accumZOffset + dstY * fullWidth + dstX;
                    var weightOffset = weightZOffset + dstY * fullWidth + dstX;
                    var patchOffset = patchZOffset + y * patchWidth + patchX;
                    for (var x = 0; x < copyWidth; x++)
                    {
                        accum[accumOffset + x] += patch[patchOffset + x];
                        if (c == 0)
                            weight[weightOffset + x] += 1f;
                    }
                }
            }
        }
    }

    private static void NormalizeAccumulatedLogits(float[] accum, float[] weight, int channels, int depth, int height, int width)
    {
        var voxelCount = checked(depth * height * width);
        for (var voxel = 0; voxel < voxelCount; voxel++)
        {
            var w = weight[voxel];
            if (w <= 1e-12f)
                continue;
            var inv = 1f / w;
            for (var c = 0; c < channels; c++)
                accum[c * voxelCount + voxel] *= inv;
        }
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

    private JObject BuildBaselineComparison(JObject baselineManifest, string baselineCaseDir, float[] logits, float[] probs, byte[] masks, ushort[] labelMap)
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
            TryAddLabelMapDiff(result, "labelmap", ResolveBaselineFile(files, "labelmap_u8_bin", baselineCaseDir), labelMap);
        return result;
    }

    private JObject BuildRestoredLabelMapComparison(JObject baselineManifest, string baselineCaseDir, string restoredLabelMapPath)
    {
        if (baselineManifest == null || string.IsNullOrWhiteSpace(restoredLabelMapPath) || !File.Exists(restoredLabelMapPath))
            return null;

        var files = baselineManifest["files"] as JObject;
        if (files == null)
            return null;

        var baselinePath = files["official_restored_labelmap"]?.Value<string>();
        if (string.IsNullOrWhiteSpace(baselinePath))
            baselinePath = ResolveBaselineFile(files, "restored_labelmap", baselineCaseDir);
        if (string.IsNullOrWhiteSpace(baselinePath) || !File.Exists(baselinePath))
            return null;

        return BuildVolumeLabelMapComparison(baselinePath, restoredLabelMapPath);
    }

    private JObject BuildRestoredLabelSubsetComparisons(
        JObject baselineManifest,
        string baselineCaseDir,
        string outputDir,
        MonaiVolumeData reference)
    {
        if (baselineManifest == null
            || string.IsNullOrWhiteSpace(baselineCaseDir)
            || string.IsNullOrWhiteSpace(outputDir)
            || reference == null)
        {
            return null;
        }

        var subsetTokens = EnumerateLabelSubsetTokens(baselineManifest);
        if (subsetTokens == null || subsetTokens.Count == 0)
            return null;

        var result = new JObject();
        for (var i = 0; i < subsetTokens.Count; i++)
        {
            var subsetToken = subsetTokens[i];
            var labelValues = ReadLabelSubsetValues(subsetToken);
            if (labelValues == null || labelValues.Length == 0)
                continue;

            var labelName = SanitizeFileName(subsetToken["label_name"]?.Value<string>() ?? "subset");
            var item = new JObject
            {
                ["label_name"] = labelName
            };

            var baselineMaskPath = ResolveBaselineRelativePath(
                subsetToken["restored_binary_mask"]?.Value<string>(),
                baselineCaseDir);
            var currentMaskPath = ResolveSubsetOutputPath(
                outputDir,
                reference,
                subsetToken["restored_binary_mask"]?.Value<string>(),
                labelName,
                "mask_restored");
            var maskComparison = BuildVolumeLabelMapComparison(baselineMaskPath, currentMaskPath);
            if (maskComparison != null)
                item["restored_binary_mask"] = maskComparison;

            var baselineLabelMapPath = ResolveBaselineRelativePath(
                subsetToken["restored_labelmap"]?.Value<string>(),
                baselineCaseDir);
            var currentLabelMapPath = ResolveSubsetOutputPath(
                outputDir,
                reference,
                subsetToken["restored_labelmap"]?.Value<string>(),
                labelName,
                "labelmap_restored");
            var labelMapComparison = BuildVolumeLabelMapComparison(baselineLabelMapPath, currentLabelMapPath);
            if (labelMapComparison != null)
                item["restored_labelmap"] = labelMapComparison;

            if (item.Count > 1)
                result[labelName] = item;
        }

        return result.Count > 0 ? result : null;
    }

    private static JObject BuildVolumeLabelMapComparison(string baselinePath, string currentPath)
    {
        if (string.IsNullOrWhiteSpace(baselinePath)
            || string.IsNullOrWhiteSpace(currentPath)
            || !File.Exists(baselinePath)
            || !File.Exists(currentPath))
        {
            return null;
        }

        try
        {
            var current = LoadVolume(currentPath);
            var baseline = LoadVolume(baselinePath);
            if (current == null || baseline == null || current.data == null || baseline.data == null)
                return null;

            var currentCount = current.data.Length;
            var baselineCount = baseline.data.Length;
            var obj = new JObject
            {
                ["path"] = baselinePath,
                ["current_count"] = currentCount,
                ["baseline_count"] = baselineCount
            };
            if (currentCount != baselineCount)
            {
                obj["error"] = "count_mismatch";
                return obj;
            }

            var mismatch = 0;
            var maxAbs = 0;
            for (var i = 0; i < currentCount; i++)
            {
                var a = Mathf.RoundToInt(current.data[i]);
                var b = Mathf.RoundToInt(baseline.data[i]);
                if (a != b)
                    mismatch++;
                var diff = Math.Abs(a - b);
                if (diff > maxAbs)
                    maxAbs = diff;
            }

            obj["mismatch_count"] = mismatch;
            obj["equal_ratio"] = currentCount > 0 ? 1d - mismatch / (double)currentCount : 1d;
            obj["max_abs"] = maxAbs;
            return obj;
        }
        catch (Exception e)
        {
            return new JObject
            {
                ["path"] = baselinePath,
                ["error"] = e.Message
            };
        }
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

    private static void TryAddLabelMapDiff(JObject dst, string name, string path, ushort[] current)
    {
        if (dst == null || string.IsNullOrWhiteSpace(name) || string.IsNullOrWhiteSpace(path) || current == null || !File.Exists(path))
            return;

        var bytes = File.ReadAllBytes(path);
        ushort[] baseline;
        if (bytes.Length == current.Length * sizeof(ushort))
        {
            baseline = ReadUInt16Array(bytes);
        }
        else if (bytes.Length == current.Length)
        {
            baseline = new ushort[current.Length];
            for (var i = 0; i < current.Length; i++)
                baseline[i] = bytes[i];
        }
        else
        {
            dst[name] = new JObject
            {
                ["path"] = path,
                ["error"] = "count_mismatch",
                ["current_count"] = current.Length,
                ["baseline_bytes"] = bytes.Length
            };
            return;
        }

        var obj = new JObject
        {
            ["path"] = path,
            ["current_count"] = current.Length,
            ["baseline_count"] = baseline.Length
        };

        var mismatch = 0;
        var maxAbs = 0;
        var histCurrent = new Dictionary<ushort, int>();
        var histBaseline = new Dictionary<ushort, int>();
        for (var i = 0; i < current.Length; i++)
        {
            var a = current[i];
            var b = baseline[i];
            if (a != b)
                mismatch++;
            var diff = Math.Abs((int)a - b);
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

    private static JObject BuildHistogramJson<TKey>(Dictionary<TKey, int> histogram) where TKey : struct, IConvertible
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
        ushort[] labelMap,
        JObject comparison,
        string executionNote,
        bool probeOnly,
        int executedPatchCount,
        int totalPatchCount)
    {
        var labelMapOnlyMode = labelMap != null && (logits == null || logits.Length == 0);
        var root = new JObject
        {
            ["case_name"] = prepared?.caseName ?? request.caseName,
            ["input_source"] = request.source.inputSource.ToString(),
            ["path_mode"] = _lastPathMode ?? string.Empty,
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
            ["probe_only"] = probeOnly,
            ["result_mode"] = labelMapOnlyMode ? "labelmap_only" : "full_logits",
            ["execution_note"] = executionNote ?? string.Empty,
            ["preparation_note"] = prepared?.preparationNote ?? string.Empty,
            ["network_input_shape_ncdhw"] = new JArray(1, request.inputChannels, request.networkInputDepth, request.networkInputHeight, request.networkInputWidth),
            ["processed_input_shape_ncdhw"] = new JArray(1, request.inputChannels, request.processedInputDepth, request.processedInputHeight, request.processedInputWidth),
            ["full_input_shape_ncdhw"] = new JArray(1, request.inputChannels, request.fullInputDepth, request.fullInputHeight, request.fullInputWidth),
            ["unity_input_shape_whdc"] = new JArray(request.processedInputWidth, request.processedInputHeight, request.processedInputDepth, request.inputChannels),
            ["model_output_shape_ncdhw"] = new JArray(1, outC, outD, outH, outW),
            ["unity_output_shape_whdc"] = new JArray(outW, outH, outD, outC)
        };

        root["diagnostics"] = new JObject
        {
            ["timings_json"] = "timings.json",
            ["resource_stats_json"] = "resource_stats.json",
            ["resource_snapshots_txt"] = _resourceSnapshotLines.Count > 0 ? "resource_snapshots.txt" : null,
            ["layer_runtime_profile_tsv"] = !string.IsNullOrWhiteSpace(_lastLayerRuntimeProfileText) ? "layer_runtime_profile.tsv" : null,
            ["gpu_resource_stats_txt"] = "gpu_resource_stats.txt"
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
            ["input_tensor"] = BuildFloatStats(prepared?.tensorNcdhw, new[] { request.inputChannels, request.processedInputDepth, request.processedInputHeight, request.processedInputWidth }),
            ["logits"] = BuildFloatStats(logits, new[] { outC, outD, outH, outW }),
            ["probs"] = BuildFloatStats(probs, new[] { outC, outD, outH, outW }),
            ["masks"] = masks != null ? BuildByteStats(masks, new[] { outC, outD, outH, outW }) : null,
            ["labelmap"] = labelMap != null ? BuildUInt16Stats(labelMap, new[] { outD, outH, outW }) : null
        };

        if (comparison != null)
            root["baseline_compare"] = comparison;

        root["sliding_window"] = new JObject
        {
            ["enabled"] = request.useSlidingWindow,
            ["roi_dhw"] = new JArray(request.slidingWindowDepth, request.slidingWindowHeight, request.slidingWindowWidth),
            ["overlap"] = request.slidingWindowOverlap,
            ["executed_patch_count"] = executedPatchCount,
            ["total_patch_count"] = totalPatchCount
        };

        if (request.postprocessKind == MonaiPostprocessKind.BratsTumorSubregions)
        {
            root["label_note"] = "This BraTS bundle predicts tumor subregions (TC/WT/ET). It does not predict skull or ventricles.";
        }
        else if (request.postprocessKind == MonaiPostprocessKind.MulticlassArgmax)
        {
            root["label_note"] = "This MONAI bundle uses softmax probabilities and multiclass argmax label decoding.";
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

    private static JObject BuildUInt16Stats(ushort[] values, int[] shape)
    {
        var obj = new JObject
        {
            ["shape"] = shape != null ? JArray.FromObject(shape) : null,
            ["count"] = values != null ? values.Length : 0
        };
        if (values == null || values.Length == 0)
            return obj;

        ushort min = ushort.MaxValue;
        ushort max = ushort.MinValue;
        double sum = 0d;
        var histogram = new Dictionary<ushort, int>();
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
        sb.AppendLine("network_input_shape_ncdhw=1," + request.inputChannels + "," + request.networkInputDepth + "," + request.networkInputHeight + "," + request.networkInputWidth);
        sb.AppendLine("processed_input_shape_ncdhw=1," + request.inputChannels + "," + request.processedInputDepth + "," + request.processedInputHeight + "," + request.processedInputWidth);
        sb.AppendLine("full_input_shape_ncdhw=1," + request.inputChannels + "," + request.fullInputDepth + "," + request.fullInputHeight + "," + request.fullInputWidth);
        sb.AppendLine("unity_input_shape_whdc=" + request.processedInputWidth + "," + request.processedInputHeight + "," + request.processedInputDepth + "," + request.inputChannels);
        sb.AppendLine("sliding_window_enabled=" + request.useSlidingWindow);
        sb.AppendLine("sliding_window_roi_dhw=" + request.slidingWindowDepth + "," + request.slidingWindowHeight + "," + request.slidingWindowWidth);
        sb.AppendLine("sliding_window_overlap=" + request.slidingWindowOverlap.ToString("0.######", CultureInfo.InvariantCulture));
        sb.AppendLine("channel_fill=" + request.channelFillMode);
        sb.AppendLine("normalize_nonzero=" + request.normalizeNonZero);
        sb.AppendLine("threshold=" + request.threshold.ToString("0.######", CultureInfo.InvariantCulture));
        sb.AppendLine("postprocess=" + request.postprocessKind);
        if (request.postprocessKind == MonaiPostprocessKind.BinaryLabelPrompt)
            sb.AppendLine("binary_foreground_label=" + request.binaryForegroundLabelValue.ToString(CultureInfo.InvariantCulture));
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
        ushort[] labelMap,
        string executionNote,
        bool hasLogits,
        bool probeOnly,
        int executedPatchCount,
        int totalPatchCount)
    {
        var sb = new StringBuilder(2048);
        var labelMapOnlyMode = labelMap != null && !hasLogits;
        sb.AppendLine("case=" + (prepared?.caseName ?? request.caseName));
        sb.AppendLine("input_source=" + request.source.inputSource);
        sb.AppendLine("model_param=" + (request.modelParamPath ?? string.Empty));
        sb.AppendLine("output_dir=" + (_lastDumpDir ?? string.Empty));
        sb.AppendLine("path_mode=" + (_lastPathMode ?? string.Empty));
        sb.AppendLine("network_input_shape_ncdhw=1," + request.inputChannels + "," + request.networkInputDepth + "," + request.networkInputHeight + "," + request.networkInputWidth);
        sb.AppendLine("processed_input_shape_ncdhw=1," + request.inputChannels + "," + request.processedInputDepth + "," + request.processedInputHeight + "," + request.processedInputWidth);
        sb.AppendLine("full_input_shape_ncdhw=1," + request.inputChannels + "," + request.fullInputDepth + "," + request.fullInputHeight + "," + request.fullInputWidth);
        sb.AppendLine("output_shape_ncdhw=1," + outC + "," + outD + "," + outH + "," + outW);
        sb.AppendLine("sliding_window_enabled=" + request.useSlidingWindow);
        sb.AppendLine("sliding_window_roi_dhw=" + request.slidingWindowDepth + "," + request.slidingWindowHeight + "," + request.slidingWindowWidth);
        sb.AppendLine("sliding_window_overlap=" + request.slidingWindowOverlap.ToString("0.######", CultureInfo.InvariantCulture));
        sb.AppendLine("channel_fill=" + request.channelFillMode);
        sb.AppendLine("normalize_nonzero=" + request.normalizeNonZero);
        sb.AppendLine("threshold=" + request.threshold.ToString("0.######", CultureInfo.InvariantCulture));
        sb.AppendLine("postprocess=" + request.postprocessKind);
        if (request.postprocessKind == MonaiPostprocessKind.BinaryLabelPrompt)
            sb.AppendLine("binary_foreground_label=" + request.binaryForegroundLabelValue.ToString(CultureInfo.InvariantCulture));
        sb.AppendLine("probe_only=" + probeOnly);
        sb.AppendLine("result_mode=" + (labelMapOnlyMode ? "labelmap_only" : "full_logits"));
        sb.AppendLine("has_logits=" + hasLogits);
        if (request.useSlidingWindow)
            sb.AppendLine("sliding_window_patches=" + executedPatchCount.ToString(CultureInfo.InvariantCulture) + "/" + totalPatchCount.ToString(CultureInfo.InvariantCulture));
        if (!string.IsNullOrWhiteSpace(executionNote))
            sb.AppendLine("execution_note=" + executionNote);
        if (_lastLayerRuntimeProfile != null)
            sb.AppendLine("layer_profile_total_ms=" + _lastLayerRuntimeProfile.totalMs.ToString("0.###", CultureInfo.InvariantCulture));
        if (request.postprocessKind == MonaiPostprocessKind.BratsTumorSubregions)
            sb.AppendLine("note=BraTS bundle predicts tumor subregions (TC/WT/ET), not skull or ventricles");
        else if (request.postprocessKind == MonaiPostprocessKind.MulticlassArgmax)
            sb.AppendLine("note=Multiclass bundle uses softmax + argmax label decoding");
        else if (request.postprocessKind == MonaiPostprocessKind.BinaryLabelPrompt)
            sb.AppendLine("note=Fixed-prompt Vista3D bundle uses sigmoid thresholding and writes the configured foreground label value");

        if (labelMap != null)
        {
            var histogram = new Dictionary<ushort, int>();
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
            AppendComparisonLine(sb, comparison, "restored_labelmap");
            AppendSubsetComparisonLines(sb, comparison["label_subsets"] as JObject);
        }

        return sb.ToString();
    }

    private static float[] BuildSoftmaxProbs(float[] logits, int width, int height, int depth, int channels)
    {
        if (logits == null)
            return null;

        var voxelCount = checked(width * height * depth);
        var probs = new float[logits.Length];
        for (var voxel = 0; voxel < voxelCount; voxel++)
        {
            var maxLogit = float.NegativeInfinity;
            for (var c = 0; c < channels; c++)
            {
                var value = logits[(c * voxelCount) + voxel];
                if (value > maxLogit)
                    maxLogit = value;
            }

            double sum = 0d;
            for (var c = 0; c < channels; c++)
            {
                var idx = (c * voxelCount) + voxel;
                var exp = Math.Exp(logits[idx] - maxLogit);
                probs[idx] = (float)exp;
                sum += exp;
            }

            var inv = sum > 0d ? 1d / sum : 0d;
            for (var c = 0; c < channels; c++)
            {
                var idx = (c * voxelCount) + voxel;
                probs[idx] = (float)(probs[idx] * inv);
            }
        }
        return probs;
    }

    private static ushort[] BuildMulticlassLabelMap(float[] probsNcdhw, int voxelCount, int channels)
    {
        var labelMap = new ushort[voxelCount];
        for (var voxel = 0; voxel < voxelCount; voxel++)
        {
            var bestChannel = 0;
            var bestValue = float.NegativeInfinity;
            for (var c = 0; c < channels; c++)
            {
                var value = probsNcdhw[(c * voxelCount) + voxel];
                if (value > bestValue)
                {
                    bestValue = value;
                    bestChannel = c;
                }
            }
            labelMap[voxel] = (ushort)bestChannel;
        }
        return labelMap;
    }

    private static ushort[] ConvertToU16(byte[] values)
    {
        if (values == null)
            return null;
        var result = new ushort[values.Length];
        for (var i = 0; i < values.Length; i++)
            result[i] = values[i];
        return result;
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

    private static void AppendSubsetComparisonLines(StringBuilder sb, JObject subsetComparisons)
    {
        if (sb == null || subsetComparisons == null)
            return;

        foreach (var property in subsetComparisons.Properties())
        {
            if (property?.Value is not JObject item)
                continue;

            AppendNamedComparisonLine(sb, item["restored_binary_mask"] as JObject, "subset_" + property.Name + "_restored_binary_mask");
            AppendNamedComparisonLine(sb, item["restored_labelmap"] as JObject, "subset_" + property.Name + "_restored_labelmap");
        }
    }

    private static void AppendNamedComparisonLine(StringBuilder sb, JObject comparison, string key)
    {
        if (comparison == null || string.IsNullOrWhiteSpace(key))
            return;

        if (comparison["error"] != null)
        {
            sb.AppendLine("compare_" + key + "=" + comparison["error"]);
            return;
        }

        if (comparison["equal_ratio"] != null)
        {
            sb.AppendLine(
                "compare_" + key
                + "_equal_ratio=" + comparison["equal_ratio"].Value<double>().ToString("G9", CultureInfo.InvariantCulture)
                + " | mismatch_count=" + comparison["mismatch_count"].Value<int>().ToString(CultureInfo.InvariantCulture));
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

    private static void WriteUInt16Array(string path, ushort[] values)
    {
        if (string.IsNullOrWhiteSpace(path) || values == null)
            return;

        Directory.CreateDirectory(Path.GetDirectoryName(path));
        var bytes = new byte[values.Length * sizeof(ushort)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    private static void WriteText(string path, string text)
    {
        if (string.IsNullOrWhiteSpace(path))
            return;

        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, text ?? string.Empty, Encoding.UTF8);
    }

    public string RunPack4RoundtripSelfTest(string outputDir, RenderTextureFormat textureFormat)
    {
        EnsureRuntimeObjects();
        ApplyReproOptions();

        _debugLines.Clear();
        _resourceSnapshotLines.Clear();
        _timingMs.Clear();
        _lastDumpDir = outputDir;
        _lastSummaryText = null;
        _flushedDebugLineCount = 0;

        if (string.IsNullOrWhiteSpace(outputDir))
            outputDir = Path.Combine(Path.GetTempPath(), "YanQi", "AIImage", "pack4_roundtrip_selftest");
        Directory.CreateDirectory(outputDir);

        NcnnGpuResourceTracker.Enabled = true;
        NcnnGpuResourceTracker.Reset("MONAI.pack4_roundtrip_selftest");

        var sw = Stopwatch.StartNew();
        var prevActive = RenderTexture.active;
        Texture2D cpuSlice = null;
        RenderTexture rt = null;
        ComputeBuffer roundtrip = null;
        float[] src = null;
        float[] dst = null;
        float[] cpu = null;
        RenderTexture chwRt = null;
        ComputeBuffer chwRoundtrip = null;
        float[] chwSrc = null;
        float[] chwDst = null;
        float[] chwCpu = null;
        Texture2D rgbSrc = null;
        RenderTexture rgbPack4 = null;
        RenderTexture rgbOut = null;
        float[] rgbPixels = null;
        string summary;

        try
        {
            const int width = 3;
            const int height = 2;
            const int depth = 2;
            const int channels = 5;
            const int chwChannels = 5;
            var packCount = Mathf.Max(1, Mathf.CeilToInt(channels / 4f));
            var sliceCount = depth * packCount;
            var elementCount = checked(width * height * depth * channels);
            src = new float[elementCount];
            for (var c = 0; c < channels; c++)
            {
                for (var z = 0; z < depth; z++)
                {
                    for (var y = 0; y < height; y++)
                    {
                        for (var x = 0; x < width; x++)
                        {
                            var index = (((c * depth) + z) * height + y) * width + x;
                            src[index] = (c + 1) * 1000f + z * 100f + y * 10f + x + 0.25f;
                        }
                    }
                }
            }

            var chwPackCount = Mathf.Max(1, Mathf.CeilToInt(chwChannels / 4f));
            var chwElementCount = checked(width * height * chwChannels);
            chwSrc = new float[chwElementCount];
            for (var c = 0; c < chwChannels; c++)
            {
                for (var y = 0; y < height; y++)
                {
                    for (var x = 0; x < width; x++)
                    {
                        var index = ((c * height) + y) * width + x;
                        chwSrc[index] = (c + 1) * 100f + y * 10f + x + 0.5f;
                    }
                }
            }

            AppendDebugLine(
                "Pack4RoundtripSelfTest begin"
                + " | format=" + textureFormat
                + " | shape=" + width + "x" + height + "x" + depth + "x" + channels
                + " | packCount=" + packCount
                + " | slices=" + sliceCount);

            using (var tensor = new NcnnTensorBuffer(width, height, depth, channels))
            {
                tensor.buffer.SetData(src);
                rt = CreateArrayRenderTextureForSelfTest(width, height, sliceCount, textureFormat, "pack4_selftest_cdhw");
                _ops.FillPack4FromBufferCDHW(tensor.buffer, width, height, depth, channels, rt);
                _ops.DebugSyncGpu();

                roundtrip = _repro.RentTempBuffer(elementCount, sizeof(float));
                _ops.Pack4ToBufferCDHW(rt, width, height, depth, channels, roundtrip);
                _ops.DebugSyncGpu();

                dst = new float[elementCount];
                roundtrip.GetData(dst);

                cpu = new float[elementCount];
                cpuSlice = new Texture2D(width, height, TextureFormat.RGBAFloat, false, true);
                for (var slice = 0; slice < sliceCount; slice++)
                {
                    Graphics.SetRenderTarget(rt, 0, CubemapFace.Unknown, slice);
                    cpuSlice.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
                    cpuSlice.Apply(false, false);
                    var pixels = cpuSlice.GetPixels();
                    var z = slice / packCount;
                    var pack = slice - z * packCount;
                    for (var i = 0; i < pixels.Length; i++)
                    {
                        var c0 = pack * 4 + 0;
                        var c1 = pack * 4 + 1;
                        var c2 = pack * 4 + 2;
                        var c3 = pack * 4 + 3;
                        if (c0 < channels) cpu[((c0 * depth + z) * height * width) + i] = pixels[i].r;
                        if (c1 < channels) cpu[((c1 * depth + z) * height * width) + i] = pixels[i].g;
                        if (c2 < channels) cpu[((c2 * depth + z) * height * width) + i] = pixels[i].b;
                        if (c3 < channels) cpu[((c3 * depth + z) * height * width) + i] = pixels[i].a;
                    }
                }
            }

            using (var chwTensor = new NcnnTensorBuffer(width, height, chwChannels))
            {
                chwTensor.buffer.SetData(chwSrc);
                chwRt = CreateArrayRenderTextureForSelfTest(width, height, chwPackCount, textureFormat, "pack4_selftest_chw");
                _ops.FillPack4FromBufferCHW(chwTensor.buffer, width, height, chwChannels, chwRt);
                _ops.DebugSyncGpu();

                chwRoundtrip = _repro.RentTempBuffer(chwElementCount, sizeof(float));
                _ops.Pack4ToBufferCHW(chwRt, width, height, chwChannels, chwRoundtrip);
                _ops.DebugSyncGpu();

                chwDst = new float[chwElementCount];
                chwRoundtrip.GetData(chwDst);

                if (cpuSlice == null)
                    cpuSlice = new Texture2D(width, height, TextureFormat.RGBAFloat, false, true);
                chwCpu = new float[chwElementCount];
                for (var slice = 0; slice < chwPackCount; slice++)
                {
                    Graphics.SetRenderTarget(chwRt, 0, CubemapFace.Unknown, slice);
                    cpuSlice.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
                    cpuSlice.Apply(false, false);
                    var pixels = cpuSlice.GetPixels();
                    var pack = slice;
                    for (var i = 0; i < pixels.Length; i++)
                    {
                        var c0 = pack * 4 + 0;
                        var c1 = pack * 4 + 1;
                        var c2 = pack * 4 + 2;
                        var c3 = pack * 4 + 3;
                        if (c0 < chwChannels) chwCpu[(c0 * height * width) + i] = pixels[i].r;
                        if (c1 < chwChannels) chwCpu[(c1 * height * width) + i] = pixels[i].g;
                        if (c2 < chwChannels) chwCpu[(c2 * height * width) + i] = pixels[i].b;
                        if (c3 < chwChannels) chwCpu[(c3 * height * width) + i] = pixels[i].a;
                    }
                }
            }

            rgbSrc = new Texture2D(width, height, TextureFormat.RGBAFloat, false, true);
            var rgbColors = new Color[width * height];
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    var i = y * width + x;
                    var r = ((x + 1) * 2f - 3f) / 2f;
                    var g = ((y + 1) * 2f - 3f) / 2f;
                    var b = (((x + y) + 1) * 2f - 3f) / 2f;
                    rgbColors[i] = new Color(r, g, b, 1f);
                }
            }
            rgbSrc.SetPixels(rgbColors);
            rgbSrc.Apply(false, false);
            rgbPack4 = CreateArrayRenderTextureForSelfTest(width, height, 1, textureFormat, "pack4_selftest_rgb");
            _ops.PackRgbToPack4(rgbSrc, 0, 0, 1f, 1f, rgbPack4, false);
            _ops.DebugSyncGpu();
            rgbOut = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat)
            {
                enableRandomWrite = true
            };
            rgbOut.Create();
            _ops.Pack4ToRgb01(rgbPack4, rgbOut, false);
            _ops.DebugSyncGpu();
            Graphics.SetRenderTarget(rgbOut);
            if (cpuSlice == null || cpuSlice.width != width || cpuSlice.height != height)
            {
                if (cpuSlice != null)
                    UnityEngine.Object.DestroyImmediate(cpuSlice);
                cpuSlice = new Texture2D(width, height, TextureFormat.RGBAFloat, false, true);
            }
            cpuSlice.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            cpuSlice.Apply(false, false);
            var rgbCpuColors = cpuSlice.GetPixels();
            rgbPixels = new float[rgbCpuColors.Length * 3];
            for (var i = 0; i < rgbCpuColors.Length; i++)
            {
                rgbPixels[i * 3 + 0] = rgbCpuColors[i].r;
                rgbPixels[i * 3 + 1] = rgbCpuColors[i].g;
                rgbPixels[i * 3 + 2] = rgbCpuColors[i].b;
            }

            var srcNz = 0;
            var dstNz = 0;
            var cpuNz = 0;
            double dstSumAbs = 0d;
            double cpuSumAbs = 0d;
            float dstMaxAbs = 0f;
            float cpuMaxAbs = 0f;
            for (var i = 0; i < src.Length; i++)
            {
                if (src[i] != 0f)
                    srcNz++;
                if (dst[i] != 0f)
                    dstNz++;
                if (cpu[i] != 0f)
                    cpuNz++;
                var dstDiff = Mathf.Abs(src[i] - dst[i]);
                var cpuDiff = Mathf.Abs(src[i] - cpu[i]);
                dstSumAbs += dstDiff;
                cpuSumAbs += cpuDiff;
                if (dstDiff > dstMaxAbs)
                    dstMaxAbs = dstDiff;
                if (cpuDiff > cpuMaxAbs)
                    cpuMaxAbs = cpuDiff;
            }

            var chwSrcNz = 0;
            var chwDstNz = 0;
            var chwCpuNz = 0;
            double chwDstSumAbs = 0d;
            double chwCpuSumAbs = 0d;
            float chwDstMaxAbs = 0f;
            float chwCpuMaxAbs = 0f;
            for (var i = 0; i < chwSrc.Length; i++)
            {
                if (chwSrc[i] != 0f)
                    chwSrcNz++;
                if (chwDst[i] != 0f)
                    chwDstNz++;
                if (chwCpu[i] != 0f)
                    chwCpuNz++;
                var chwDstDiff = Mathf.Abs(chwSrc[i] - chwDst[i]);
                var chwCpuDiff = Mathf.Abs(chwSrc[i] - chwCpu[i]);
                chwDstSumAbs += chwDstDiff;
                chwCpuSumAbs += chwCpuDiff;
                if (chwDstDiff > chwDstMaxAbs)
                    chwDstMaxAbs = chwDstDiff;
                if (chwCpuDiff > chwCpuMaxAbs)
                    chwCpuMaxAbs = chwCpuDiff;
            }

            double rgbMeanAbs = 0d;
            float rgbMaxAbs = 0f;
            for (var i = 0; i < rgbColors.Length; i++)
            {
                var expectedR = rgbColors[i].r * 0.5f + 0.5f;
                var expectedG = rgbColors[i].g * 0.5f + 0.5f;
                var expectedB = rgbColors[i].b * 0.5f + 0.5f;
                var dr = Mathf.Abs(expectedR - rgbPixels[i * 3 + 0]);
                var dg = Mathf.Abs(expectedG - rgbPixels[i * 3 + 1]);
                var db = Mathf.Abs(expectedB - rgbPixels[i * 3 + 2]);
                rgbMeanAbs += dr + dg + db;
                rgbMaxAbs = Mathf.Max(rgbMaxAbs, Mathf.Max(dr, Mathf.Max(dg, db)));
            }
            rgbMeanAbs /= Math.Max(1, rgbColors.Length * 3);

            var sb = new StringBuilder(2048);
            sb.AppendLine("Pack4RoundtripSelfTest");
            sb.AppendLine("format=" + textureFormat);
            sb.AppendLine("cdhw_shape=" + width + "x" + height + "x" + depth + "x" + channels);
            sb.AppendLine("cdhw_tex=" + (rt != null ? (rt.width + "x" + rt.height + "x" + rt.volumeDepth + " " + rt.format) : "null"));
            sb.AppendLine("cdhw_src_nz=" + srcNz + "/" + src.Length);
            sb.AppendLine("cdhw_dst_nz=" + dstNz + "/" + src.Length);
            sb.AppendLine("cdhw_cpu_nz=" + cpuNz + "/" + src.Length);
            sb.AppendLine("cdhw_dst_mean_abs=" + (dstSumAbs / Math.Max(1, src.Length)).ToString("G9", CultureInfo.InvariantCulture));
            sb.AppendLine("cdhw_dst_max_abs=" + dstMaxAbs.ToString("G9", CultureInfo.InvariantCulture));
            sb.AppendLine("cdhw_cpu_mean_abs=" + (cpuSumAbs / Math.Max(1, src.Length)).ToString("G9", CultureInfo.InvariantCulture));
            sb.AppendLine("cdhw_cpu_max_abs=" + cpuMaxAbs.ToString("G9", CultureInfo.InvariantCulture));
            sb.AppendLine("cdhw_src_first16=" + string.Join(",", FormatFloatSample(src, 16)));
            sb.AppendLine("cdhw_dst_first16=" + string.Join(",", FormatFloatSample(dst, 16)));
            sb.AppendLine("cdhw_cpu_first16=" + string.Join(",", FormatFloatSample(cpu, 16)));
            sb.AppendLine("chw_shape=" + width + "x" + height + "x" + chwChannels);
            sb.AppendLine("chw_tex=" + (chwRt != null ? (chwRt.width + "x" + chwRt.height + "x" + chwRt.volumeDepth + " " + chwRt.format) : "null"));
            sb.AppendLine("chw_src_nz=" + chwSrcNz + "/" + chwSrc.Length);
            sb.AppendLine("chw_dst_nz=" + chwDstNz + "/" + chwSrc.Length);
            sb.AppendLine("chw_cpu_nz=" + chwCpuNz + "/" + chwSrc.Length);
            sb.AppendLine("chw_dst_mean_abs=" + (chwDstSumAbs / Math.Max(1, chwSrc.Length)).ToString("G9", CultureInfo.InvariantCulture));
            sb.AppendLine("chw_dst_max_abs=" + chwDstMaxAbs.ToString("G9", CultureInfo.InvariantCulture));
            sb.AppendLine("chw_cpu_mean_abs=" + (chwCpuSumAbs / Math.Max(1, chwSrc.Length)).ToString("G9", CultureInfo.InvariantCulture));
            sb.AppendLine("chw_cpu_max_abs=" + chwCpuMaxAbs.ToString("G9", CultureInfo.InvariantCulture));
            sb.AppendLine("chw_src_first16=" + string.Join(",", FormatFloatSample(chwSrc, 16)));
            sb.AppendLine("chw_dst_first16=" + string.Join(",", FormatFloatSample(chwDst, 16)));
            sb.AppendLine("chw_cpu_first16=" + string.Join(",", FormatFloatSample(chwCpu, 16)));
            sb.AppendLine("rgb_pack4_tex=" + (rgbPack4 != null ? (rgbPack4.width + "x" + rgbPack4.height + "x" + rgbPack4.volumeDepth + " " + rgbPack4.format) : "null"));
            sb.AppendLine("rgb_out_mean_abs=" + rgbMeanAbs.ToString("G9", CultureInfo.InvariantCulture));
            sb.AppendLine("rgb_out_max_abs=" + rgbMaxAbs.ToString("G9", CultureInfo.InvariantCulture));
            sb.AppendLine("rgb_out_first12=" + string.Join(",", FormatFloatSample(rgbPixels, 12)));
            sb.AppendLine("gpu_summary=" + NcnnGpuResourceTracker.BuildSummary());
            summary = sb.ToString();

            AppendDebugLine(
                "Pack4RoundtripSelfTest result"
                + " | format=" + textureFormat
                + " | cdhwDstNz=" + dstNz + "/" + src.Length
                + " | cdhwCpuNz=" + cpuNz + "/" + src.Length
                + " | chwDstNz=" + chwDstNz + "/" + chwSrc.Length
                + " | chwCpuNz=" + chwCpuNz + "/" + chwSrc.Length
                + " | rgbMeanAbs=" + rgbMeanAbs.ToString("G9", CultureInfo.InvariantCulture));

            _lastSummaryText = summary;
            WriteFloatArray(Path.Combine(outputDir, "src_ncdhw_f32.bin"), src);
            WriteFloatArray(Path.Combine(outputDir, "dst_roundtrip_f32.bin"), dst);
            WriteFloatArray(Path.Combine(outputDir, "cpu_readback_f32.bin"), cpu);
            WriteFloatArray(Path.Combine(outputDir, "src_chw_f32.bin"), chwSrc);
            WriteFloatArray(Path.Combine(outputDir, "dst_chw_roundtrip_f32.bin"), chwDst);
            WriteFloatArray(Path.Combine(outputDir, "cpu_chw_readback_f32.bin"), chwCpu);
            WriteText(Path.Combine(outputDir, "summary.txt"), summary);
            WriteText(Path.Combine(outputDir, "runtime_debug.log"), string.Join(Environment.NewLine, _debugLines));
            NcnnGpuResourceTracker.WriteReport(outputDir);
            return summary;
        }
        finally
        {
            sw.Stop();
            _timingMs["pack4_selftest_elapsed_ms"] = sw.ElapsedMilliseconds;
            if (!string.IsNullOrWhiteSpace(outputDir))
                WriteText(Path.Combine(outputDir, "timings.json"), new JObject { ["pack4_selftest_elapsed_ms"] = sw.ElapsedMilliseconds }.ToString());
            if (roundtrip != null)
                _repro?.ReturnTempBuffer(roundtrip);
            if (rt != null)
            {
                NcnnGpuResourceTracker.ReleaseTexture(rt, "RunPack4RoundtripSelfTest.manualRt");
                rt.Release();
                UnityEngine.Object.DestroyImmediate(rt);
            }
            if (chwRoundtrip != null)
                _repro?.ReturnTempBuffer(chwRoundtrip);
            if (chwRt != null)
            {
                NcnnGpuResourceTracker.ReleaseTexture(chwRt, "RunPack4RoundtripSelfTest.manualChwRt");
                chwRt.Release();
                UnityEngine.Object.DestroyImmediate(chwRt);
            }
            if (cpuSlice != null)
                UnityEngine.Object.DestroyImmediate(cpuSlice);
            if (rgbSrc != null)
                UnityEngine.Object.DestroyImmediate(rgbSrc);
            if (rgbOut != null)
            {
                rgbOut.Release();
                UnityEngine.Object.DestroyImmediate(rgbOut);
            }
            if (rgbPack4 != null)
            {
                NcnnGpuResourceTracker.ReleaseTexture(rgbPack4, "RunPack4RoundtripSelfTest.manualRgbRt");
                rgbPack4.Release();
                UnityEngine.Object.DestroyImmediate(rgbPack4);
            }
            RenderTexture.active = prevActive;
        }
    }

    private static RenderTexture CreateArrayRenderTextureForSelfTest(int width, int height, int depth, RenderTextureFormat format, string label)
    {
        var desc = new RenderTextureDescriptor(width, height, format, 0)
        {
            dimension = UnityEngine.Rendering.TextureDimension.Tex2DArray,
            volumeDepth = depth,
            enableRandomWrite = true,
            msaaSamples = 1,
            mipCount = 1,
            useMipMap = false,
            autoGenerateMips = false,
            sRGB = false
        };
        var rt = new RenderTexture(desc)
        {
            name = label ?? "pack4_selftest_array",
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        rt.Create();
        NcnnGpuResourceTracker.RegisterTexture(rt, "MONAI." + (label ?? "pack4_selftest_array"));
        return rt;
    }

    private static IEnumerable<string> FormatFloatSample(float[] values, int count)
    {
        if (values == null || values.Length == 0 || count <= 0)
            yield break;

        var limit = Math.Min(count, values.Length);
        for (var i = 0; i < limit; i++)
            yield return values[i].ToString("G9", CultureInfo.InvariantCulture);
    }

    private static ushort[] ReadUInt16Array(byte[] bytes)
    {
        if (bytes == null || bytes.Length == 0)
            return Array.Empty<ushort>();
        if ((bytes.Length % sizeof(ushort)) != 0)
            throw new InvalidOperationException("UInt16 array byte length is invalid.");
        var values = new ushort[bytes.Length / sizeof(ushort)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
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
            if (_flushedDebugLineCount > _debugLines.Count)
                _flushedDebugLineCount = 0;
            if (_flushedDebugLineCount >= _debugLines.Count)
                return;

            var path = Path.Combine(_lastDumpDir, "runtime_debug.log");
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            using var sw = new StreamWriter(path, true, Encoding.UTF8);
            for (var i = _flushedDebugLineCount; i < _debugLines.Count; i++)
                sw.WriteLine(_debugLines[i]);
            _flushedDebugLineCount = _debugLines.Count;
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

    private static string[] SplitNameList(string csv)
    {
        if (string.IsNullOrWhiteSpace(csv))
            return Array.Empty<string>();

        var raw = csv.Split(new[] { ',', ';', '\r', '\n', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        if (raw == null || raw.Length == 0)
            return Array.Empty<string>();

        var names = new List<string>(raw.Length);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (var i = 0; i < raw.Length; i++)
        {
            var item = raw[i]?.Trim();
            if (string.IsNullOrWhiteSpace(item) || !seen.Add(item))
                continue;
            names.Add(item);
        }

        return names.ToArray();
    }

    private static string[] BuildPinnedBlobNames(string outputBlobName, string[] debugPinnedBlobNames)
    {
        var names = new List<string>(8);
        var seen = new HashSet<string>(StringComparer.Ordinal);

        if (!string.IsNullOrWhiteSpace(outputBlobName) && seen.Add(outputBlobName))
            names.Add(outputBlobName);

        if (debugPinnedBlobNames != null)
        {
            for (var i = 0; i < debugPinnedBlobNames.Length; i++)
            {
                var name = debugPinnedBlobNames[i]?.Trim();
                if (string.IsNullOrWhiteSpace(name) || !seen.Add(name))
                    continue;
                names.Add(name);
            }
        }

        return names.ToArray();
    }

    private void DumpPinnedBlobOutputs(NcnnRepro.InferResult infer, string[] pinnedBlobNames)
    {
        if (infer == null || pinnedBlobNames == null || pinnedBlobNames.Length == 0 || string.IsNullOrWhiteSpace(_lastDumpDir))
            return;

        var dumpDir = Path.Combine(_lastDumpDir, "intermediate_blobs");
        Directory.CreateDirectory(dumpDir);

        var manifest = new JObject();
        for (var i = 0; i < pinnedBlobNames.Length; i++)
        {
            var blobName = pinnedBlobNames[i];
            if (string.IsNullOrWhiteSpace(blobName))
                continue;

            try
            {
                InferOutputShape view;
                float[] data;
                if (infer.TryGetExistingTexture(blobName, out var existingTexture) && existingTexture != null
                    && infer.TryGetLogicalShape(blobName, out var dims, out var w, out var h, out var d, out var c))
                {
                    view = new InferOutputShape(dims, w, h, d, c);
                    data = ReadTextureOutputData(existingTexture, view);
                }
                else if (infer.TryGetExistingBufferView(blobName, out var bufferView) && bufferView != null && bufferView.buffer != null)
                {
                    view = new InferOutputShape(bufferView.dims, bufferView.w, bufferView.h, bufferView.d, bufferView.c);
                    data = infer.GetBufferData(blobName);
                }
                else
                {
                    view = GetInferOutputShape(infer, blobName, "Pinned blob missing: ");
                    data = ExtractInferOutputData(infer, blobName, view);
                }

                var safeName = SanitizeFileName(blobName);
                var fileName = safeName + "_d" + view.dims.ToString(CultureInfo.InvariantCulture)
                    + "_" + view.w.ToString(CultureInfo.InvariantCulture)
                    + "x" + view.h.ToString(CultureInfo.InvariantCulture)
                    + "x" + view.d.ToString(CultureInfo.InvariantCulture)
                    + "x" + view.c.ToString(CultureInfo.InvariantCulture)
                    + "_f32.bin";
                WriteFloatArray(Path.Combine(dumpDir, fileName), data);

                manifest[blobName] = new JObject
                {
                    ["dims"] = view.dims,
                    ["w"] = view.w,
                    ["h"] = view.h,
                    ["d"] = view.d,
                    ["c"] = view.c,
                    ["element_count"] = data.Length,
                    ["file"] = fileName
                };
            }
            catch (Exception e)
            {
                manifest[blobName] = new JObject
                {
                    ["error"] = e.Message
                };
            }
        }

        WriteText(Path.Combine(dumpDir, "manifest.json"), manifest.ToString());
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

    private void TryWriteRestoredLabelMap(string outputDir, PreparedInput prepared, ushort[] labelMap, int depth, int height, int width)
    {
        if (string.IsNullOrWhiteSpace(outputDir) || prepared?.referenceVolume == null || labelMap == null)
            return;

        var reference = prepared.referenceVolume;
        try
        {
            var restored = TryRestoreVistaLabelMapFromManifest(prepared.baselineManifest, labelMap, depth, height, width);
            if (restored != null)
            {
                AppendDebugLine(
                    "TryWriteRestoredLabelMap vista_inverse_restore"
                    + " | src=" + depth + "x" + height + "x" + width
                    + " | dst=" + reference.dim0 + "x" + reference.dim1 + "x" + reference.dim2);
            }
            else
            {
                restored = labelMap;
                if (reference.dim0 != depth || reference.dim1 != height || reference.dim2 != width)
                {
                    restored = CenterCropOrPadLabelMap(
                        labelMap,
                        depth,
                        height,
                        width,
                        reference.dim0,
                        reference.dim1,
                        reference.dim2);
                    AppendDebugLine(
                        "TryWriteRestoredLabelMap restore_shape"
                        + " | src=" + depth + "x" + height + "x" + width
                        + " | dst=" + reference.dim0 + "x" + reference.dim1 + "x" + reference.dim2);
                }
            }

            WriteLabelMapForReference(outputDir, "labelmap_restored", restored, reference);
            TryWriteLabelSubsetMasks(outputDir, prepared.baselineManifest, restored, reference);
        }
        catch (Exception e)
        {
            AppendDebugLine("TryWriteRestoredLabelMap failed | " + e.Message);
        }
    }

    private bool ShouldRunTimingSplitDiagnosticsForPatch(int patchIndex)
    {
        return enableTimingSplitDiagnostics
            && !_timingSplitPatchDiagnosticCaptured
            && patchIndex == 1
            && !string.IsNullOrWhiteSpace(timingSplitStopAfterBlobName)
            && _repro != null
            && _ops != null;
    }

    private void RunPatchTimingSplitDiagnostics(ResolvedRequest resolved, NcnnTensorBuffer inputTensor, int depth, int height, int width)
    {
        if (!ShouldRunTimingSplitDiagnosticsForPatch(1))
            return;

        var stopAfterBlobName = timingSplitStopAfterBlobName?.Trim();
        if (string.IsNullOrWhiteSpace(stopAfterBlobName))
            return;

        _timingSplitPatchDiagnosticCaptured = true;
        var pinnedBlobNames = BuildPinnedBlobNames(stopAfterBlobName, resolved?.debugPinnedBlobNames);
        AppendDebugLine(
            "TimingSplitDiagnostic body_begin"
            + " | stop_after=" + stopAfterBlobName
            + " | shape=" + width.ToString(CultureInfo.InvariantCulture)
            + "x" + height.ToString(CultureInfo.InvariantCulture)
            + "x" + depth.ToString(CultureInfo.InvariantCulture));

        var dispatchSw = Stopwatch.StartNew();
        using var inferHandle = RunInferenceWithPatchInput(
            resolved,
            inputTensor,
            depth,
            height,
            width,
            pinnedBlobNames,
            stopAfterBlobName);
        dispatchSw.Stop();
        var outputView = GetInferOutputShape(inferHandle.infer, stopAfterBlobName, "MONAI timing-split stop blob missing: ");

        var syncSw = Stopwatch.StartNew();
        _ops?.DebugSyncGpu();
        syncSw.Stop();

        var dispatchMs = StopwatchToMilliseconds(dispatchSw);
        var syncMs = StopwatchToMilliseconds(syncSw);
        RecordDiagnosticTiming("diag_body_dispatch_ms", dispatchMs);
        RecordDiagnosticTiming("diag_body_sync_ms", syncMs);
        RecordDiagnosticTiming("diag_body_total_ms", dispatchMs + syncMs);
        AppendDebugLine(
            "TimingSplitDiagnostic body_end"
            + " | stop_after=" + stopAfterBlobName
            + " | output_shape=d" + outputView.dims.ToString(CultureInfo.InvariantCulture)
            + ":" + outputView.w.ToString(CultureInfo.InvariantCulture)
            + "x" + outputView.h.ToString(CultureInfo.InvariantCulture)
            + "x" + outputView.d.ToString(CultureInfo.InvariantCulture)
            + "x" + outputView.c.ToString(CultureInfo.InvariantCulture)
            + " | dispatch_ms=" + dispatchMs.ToString("0.###", CultureInfo.InvariantCulture)
            + " | sync_ms=" + syncMs.ToString("0.###", CultureInfo.InvariantCulture));
    }

    private void RecordDiagnosticTiming(string key, double milliseconds)
    {
        if (string.IsNullOrWhiteSpace(key) || double.IsNaN(milliseconds) || double.IsInfinity(milliseconds))
            return;

        _diagnosticTimingMs[key] = milliseconds;
    }

    private void HandleTimingSplitSyncPoint(string topName, double elapsedMs)
    {
        RecordDiagnosticTiming("diag_inline_sync_after_top_ms", elapsedMs);
        AppendDebugLine(
            "TimingSplitDiagnostic inline_sync"
            + " | top=" + (topName ?? string.Empty)
            + " | sync_ms=" + elapsedMs.ToString("0.###", CultureInfo.InvariantCulture));
    }

    private static double StopwatchToMilliseconds(Stopwatch sw)
    {
        if (sw == null)
            return 0d;
        return sw.ElapsedTicks * 1000d / Stopwatch.Frequency;
    }

    private void TryWriteLabelSubsetMasks(string outputDir, JObject baselineManifest, ushort[] restoredLabelMap, MonaiVolumeData reference)
    {
        if (string.IsNullOrWhiteSpace(outputDir)
            || baselineManifest == null
            || restoredLabelMap == null
            || reference == null)
        {
            return;
        }

        var subsetTokens = EnumerateLabelSubsetTokens(baselineManifest);
        if (subsetTokens == null || subsetTokens.Count == 0)
            return;

        try
        {
            var subsetDir = Path.Combine(outputDir, "label_subsets");
            Directory.CreateDirectory(subsetDir);
            var manifestPath = Path.Combine(subsetDir, "manifest.json");
            var manifestItems = new JArray();
            foreach (var subsetToken in subsetTokens)
            {
                var labelValues = ReadLabelSubsetValues(subsetToken);
                if (labelValues == null || labelValues.Length == 0)
                    continue;

                var labelName = SanitizeFileName(subsetToken["label_name"]?.Value<string>() ?? "subset");
                var subsetMask = BuildBinarySubsetMask(restoredLabelMap, labelValues);
                var subsetLabelMap = BuildPreservedLabelSubsetLabelMap(restoredLabelMap, labelValues);
                var subsetPath = ResolveSubsetOutputPath(
                    outputDir,
                    reference,
                    subsetToken["restored_binary_mask"]?.Value<string>(),
                    labelName,
                    "mask_restored");
                var subsetLabelMapPath = ResolveSubsetOutputPath(
                    outputDir,
                    reference,
                    subsetToken["restored_labelmap"]?.Value<string>(),
                    labelName,
                    "labelmap_restored");
                WriteLabelMapFile(subsetPath, subsetMask, reference);
                WriteLabelMapFile(subsetLabelMapPath, subsetLabelMap, reference);

                var item = new JObject
                {
                    ["label_name"] = labelName,
                    ["restored_binary_mask"] = Path.GetFileName(subsetPath),
                    ["restored_labelmap"] = Path.GetFileName(subsetLabelMapPath)
                };
                if (labelValues.Length == 1)
                    item["label_value"] = labelValues[0];
                else
                    item["label_values"] = new JArray(Array.ConvertAll(labelValues, value => (int)value));
                manifestItems.Add(item);
            }

            if (manifestItems.Count == 0)
                return;

            JToken manifest;
            if (manifestItems.Count == 1)
            {
                manifest = (JObject)manifestItems[0];
            }
            else
            {
                manifest = new JObject
                {
                    ["items"] = manifestItems
                };
            }
            WriteText(manifestPath, manifest.ToString());
        }
        catch (Exception e)
        {
            AppendDebugLine("TryWriteLabelSubsetMasks manifest failed | " + e.Message);
        }
    }

    private static List<JToken> EnumerateLabelSubsetTokens(JObject baselineManifest)
    {
        if (baselineManifest == null)
            return null;

        var result = new List<JToken>();
        var subsetArray = baselineManifest["label_subsets"] as JArray;
        if (subsetArray != null)
        {
            for (var i = 0; i < subsetArray.Count; i++)
            {
                var token = subsetArray[i];
                if (token != null && token.Type != JTokenType.Null)
                    result.Add(token);
            }
        }

        var singleSubset = baselineManifest["label_subset"];
        if (singleSubset != null
            && singleSubset.Type != JTokenType.Null
            && result.Count == 0)
        {
            result.Add(singleSubset);
        }

        return result;
    }

    private static ushort[] ReadLabelSubsetValues(JToken subsetToken)
    {
        if (subsetToken == null)
            return null;

        var values = ReadIntArray(subsetToken["label_values"]);
        if (values != null && values.Length > 0)
        {
            var result = new ushort[values.Length];
            for (var i = 0; i < values.Length; i++)
                result[i] = (ushort)Mathf.Max(0, values[i]);
            return result;
        }

        var single = subsetToken["label_value"];
        if (single != null && single.Type != JTokenType.Null)
            return new[] { (ushort)Mathf.Max(0, single.Value<int>()) };

        return null;
    }

    private static ushort[] BuildBinarySubsetMask(ushort[] source, ushort[] labelValues)
    {
        if (source == null || labelValues == null || labelValues.Length == 0)
            return null;

        var wanted = new HashSet<ushort>(labelValues);
        var result = new ushort[source.Length];
        for (var i = 0; i < source.Length; i++)
            result[i] = wanted.Contains(source[i]) ? (ushort)1 : (ushort)0;
        return result;
    }

    private static ushort[] BuildPreservedLabelSubsetLabelMap(ushort[] source, ushort[] labelValues)
    {
        if (source == null || labelValues == null || labelValues.Length == 0)
            return null;

        var wanted = new HashSet<ushort>(labelValues);
        var result = new ushort[source.Length];
        for (var i = 0; i < source.Length; i++)
        {
            var value = source[i];
            result[i] = wanted.Contains(value) ? value : (ushort)0;
        }
        return result;
    }

    private static ushort[] TryRestoreVistaLabelMapFromManifest(JObject baselineManifest, ushort[] source, int srcDepth, int srcHeight, int srcWidth)
    {
        var inverse = baselineManifest?["inverse_restore"] as JObject;
        if (inverse == null)
            return null;
        if (!string.Equals(inverse["kind"]?.Value<string>(), "vista_preprocessing_inverse_v1", StringComparison.Ordinal))
            return null;

        var processedAffine = ReadFloatArray(inverse["processed_affine"]);
        var orientation = inverse["orientation_inverse"] as JObject;
        var crop = inverse["crop_inverse"] as JObject;
        var spacing = inverse["spacing_inverse"] as JObject;
        if (processedAffine == null || processedAffine.Length < 16 || orientation == null || crop == null || spacing == null)
            return null;

        var orientationAffine = ReadFloatArray(orientation["original_affine"]);
        var orientationSize = ReadIntArray(orientation["orig_size_xyz"]);
        var cropSize = ReadIntArray(crop["orig_size_xyz"]);
        var cropped = ReadIntArray(crop["cropped_xyz"]);
        var spacingAffine = ReadFloatArray(spacing["src_affine"]);
        var spacingSize = ReadIntArray(spacing["orig_size_xyz"]);
        if (orientationAffine == null || orientationAffine.Length < 16
            || spacingAffine == null || spacingAffine.Length < 16
            || orientationSize == null || orientationSize.Length < 3
            || cropSize == null || cropSize.Length < 3
            || cropped == null || cropped.Length < 6
            || spacingSize == null || spacingSize.Length < 3)
        {
            return null;
        }

        var orientationRestored = ResampleNearestLabelMap(
            source,
            srcDepth,
            srcHeight,
            srcWidth,
            processedAffine,
            orientationAffine,
            orientationSize[0],
            orientationSize[1],
            orientationSize[2]);
        if (orientationRestored == null)
            return null;

        var cropRestored = new ushort[checked(cropSize[0] * cropSize[1] * cropSize[2])];
        var padBeforeX = Mathf.Max(0, cropped[0]);
        var padAfterX = Mathf.Max(0, cropped[1]);
        var padBeforeY = Mathf.Max(0, cropped[2]);
        var padAfterY = Mathf.Max(0, cropped[3]);
        var padBeforeZ = Mathf.Max(0, cropped[4]);
        var padAfterZ = Mathf.Max(0, cropped[5]);
        var copyX = Math.Min(orientationSize[0], cropSize[0] - padBeforeX - padAfterX);
        var copyY = Math.Min(orientationSize[1], cropSize[1] - padBeforeY - padAfterY);
        var copyZ = Math.Min(orientationSize[2], cropSize[2] - padBeforeZ - padAfterZ);
        if (copyX <= 0 || copyY <= 0 || copyZ <= 0)
            return null;

        for (var x = 0; x < copyX; x++)
        {
            var dstX = padBeforeX + x;
            for (var y = 0; y < copyY; y++)
            {
                var dstY = padBeforeY + y;
                var srcOffset = ((x * orientationSize[1]) + y) * orientationSize[2];
                var dstOffset = ((dstX * cropSize[1]) + dstY) * cropSize[2] + padBeforeZ;
                Array.Copy(orientationRestored, srcOffset, cropRestored, dstOffset, copyZ);
            }
        }

        var cropAffine = (float[])orientationAffine.Clone();
        for (var row = 0; row < 3; row++)
        {
            cropAffine[row * 4 + 3] -= cropAffine[row * 4 + 0] * padBeforeX;
            cropAffine[row * 4 + 3] -= cropAffine[row * 4 + 1] * padBeforeY;
            cropAffine[row * 4 + 3] -= cropAffine[row * 4 + 2] * padBeforeZ;
        }

        return ResampleNearestLabelMap(
            cropRestored,
            cropSize[0],
            cropSize[1],
            cropSize[2],
            cropAffine,
            spacingAffine,
            spacingSize[0],
            spacingSize[1],
            spacingSize[2]);
    }

    private static ushort[] ResampleNearestLabelMap(
        ushort[] source,
        int srcDim0,
        int srcDim1,
        int srcDim2,
        float[] srcAffine,
        float[] dstAffine,
        int dstDim0,
        int dstDim1,
        int dstDim2)
    {
        if (source == null || srcAffine == null || dstAffine == null)
            return null;
        if (srcAffine.Length < 16 || dstAffine.Length < 16)
            return null;
        if (srcDim0 <= 0 || srcDim1 <= 0 || srcDim2 <= 0 || dstDim0 <= 0 || dstDim1 <= 0 || dstDim2 <= 0)
            return null;

        var srcMatrix = BuildMatrix4x4(srcAffine);
        var dstMatrix = BuildMatrix4x4(dstAffine);
        var dstToSrc = srcMatrix.inverse * dstMatrix;
        var result = new ushort[checked(dstDim0 * dstDim1 * dstDim2)];
        for (var x = 0; x < dstDim0; x++)
        {
            for (var y = 0; y < dstDim1; y++)
            {
                for (var z = 0; z < dstDim2; z++)
                {
                    var src = dstToSrc.MultiplyPoint3x4(new Vector3(x, y, z));
                    var sx = Mathf.RoundToInt(src.x);
                    var sy = Mathf.RoundToInt(src.y);
                    var sz = Mathf.RoundToInt(src.z);
                    if (sx < 0 || sy < 0 || sz < 0 || sx >= srcDim0 || sy >= srcDim1 || sz >= srcDim2)
                        continue;
                    result[((x * dstDim1) + y) * dstDim2 + z] = source[((sx * srcDim1) + sy) * srcDim2 + sz];
                }
            }
        }
        return result;
    }

    private static string ResolveBaselineRelativePath(string relativePath, string caseDir)
    {
        var normalized = relativePath?.Trim();
        if (string.IsNullOrWhiteSpace(normalized) || string.IsNullOrWhiteSpace(caseDir))
            return null;

        normalized = normalized.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
        if (Path.IsPathRooted(normalized))
            return Path.GetFullPath(normalized);
        return Path.GetFullPath(Path.Combine(caseDir, normalized));
    }

    private static string ResolveSubsetOutputPath(
        string outputDir,
        MonaiVolumeData reference,
        string relativePath,
        string labelName,
        string defaultSuffix)
    {
        var normalized = relativePath?.Trim();
        if (!string.IsNullOrWhiteSpace(normalized))
        {
            normalized = normalized.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
            if (Path.IsPathRooted(normalized))
                return Path.GetFullPath(normalized);
            if (!normalized.Contains(Path.DirectorySeparatorChar.ToString()))
                return Path.Combine(outputDir, "label_subsets", normalized);
            return Path.Combine(outputDir, normalized);
        }

        var extension = string.Equals(reference?.sourceFormat, "nrrd", StringComparison.OrdinalIgnoreCase)
            ? ".nrrd"
            : ".nii.gz";
        return Path.Combine(outputDir, "label_subsets", labelName + "_" + defaultSuffix + extension);
    }

    private static ushort[] CenterCropOrPadLabelMap(
        ushort[] source,
        int srcDepth,
        int srcHeight,
        int srcWidth,
        int dstDepth,
        int dstHeight,
        int dstWidth)
    {
        if (source == null)
            return null;
        if (srcDepth == dstDepth && srcHeight == dstHeight && srcWidth == dstWidth)
            return source;

        var result = new ushort[checked(dstDepth * dstHeight * dstWidth)];
        var copyDepth = Math.Min(srcDepth, dstDepth);
        var copyHeight = Math.Min(srcHeight, dstHeight);
        var copyWidth = Math.Min(srcWidth, dstWidth);
        var srcDepthStart = Math.Max(0, (srcDepth - copyDepth) / 2);
        var srcHeightStart = Math.Max(0, (srcHeight - copyHeight) / 2);
        var srcWidthStart = Math.Max(0, (srcWidth - copyWidth) / 2);
        var dstDepthStart = Math.Max(0, (dstDepth - copyDepth) / 2);
        var dstHeightStart = Math.Max(0, (dstHeight - copyHeight) / 2);
        var dstWidthStart = Math.Max(0, (dstWidth - copyWidth) / 2);

        for (var z = 0; z < copyDepth; z++)
        {
            var srcZ = srcDepthStart + z;
            var dstZ = dstDepthStart + z;
            for (var y = 0; y < copyHeight; y++)
            {
                var srcY = srcHeightStart + y;
                var dstY = dstHeightStart + y;
                var srcOffset = ((srcZ * srcHeight) + srcY) * srcWidth + srcWidthStart;
                var dstOffset = ((dstZ * dstHeight) + dstY) * dstWidth + dstWidthStart;
                Array.Copy(source, srcOffset, result, dstOffset, copyWidth);
            }
        }

        return result;
    }

    private static string WriteLabelMapForReference(string outputDir, string fileStem, ushort[] values, MonaiVolumeData reference)
    {
        if (string.Equals(reference.sourceFormat, "nrrd", StringComparison.OrdinalIgnoreCase))
        {
            var path = Path.Combine(outputDir, fileStem + ".nrrd");
            WriteNrrdLabelMap(path, values, reference);
            return path;
        }

        var niftiPath = Path.Combine(outputDir, fileStem + ".nii.gz");
        WriteNiftiLabelMap(niftiPath, values, reference);
        return niftiPath;
    }

    private static void WriteLabelMapFile(string path, ushort[] values, MonaiVolumeData reference)
    {
        if (string.Equals(Path.GetExtension(path), ".nrrd", StringComparison.OrdinalIgnoreCase))
        {
            WriteNrrdLabelMap(path, values, reference);
            return;
        }

        WriteNiftiLabelMap(path, values, reference);
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

    private static MonaiVolumeData LoadVolume(string path, VolumeLoadOptions options = null)
    {
        options ??= new VolumeLoadOptions();
        var lower = path.ToLowerInvariant();
        if (lower.EndsWith(".nrrd", StringComparison.Ordinal) || lower.EndsWith(".nhdr", StringComparison.Ordinal))
            return LoadNrrd(path, options);
        if (lower.EndsWith(".nii", StringComparison.Ordinal) || lower.EndsWith(".nii.gz", StringComparison.Ordinal))
            return LoadNifti(path, options);
        throw new InvalidOperationException("Unsupported MONAI medical input format: " + path);
    }

    private static MonaiVolumeData LoadNrrd(string path, VolumeLoadOptions options)
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
        return new MonaiVolumeData
        {
            path = path,
            dim0 = sizes[0],
            dim1 = sizes[1],
            dim2 = sizes[2],
            data = options.includeVoxelData ? ConvertFirstAxisFastRawToCOrder(rawBytes, sizes[0], sizes[1], sizes[2], elementType, littleEndian, 0) : null,
            spacing = spacing,
            sourceFormat = "nrrd",
            nrrdHeader = new Dictionary<string, string>(header, StringComparer.OrdinalIgnoreCase)
        };
    }

    private static MonaiVolumeData LoadNifti(string path, VolumeLoadOptions options)
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

        return new MonaiVolumeData
        {
            path = path,
            dim0 = dim0,
            dim1 = dim1,
            dim2 = dim2,
            data = options.includeVoxelData ? ConvertFirstAxisFastRawToCOrder(bytes, dim0, dim1, dim2, elementType, littleEndian, voxOffset) : null,
            spacing = spacing,
            sourceFormat = path.EndsWith(".nii.gz", StringComparison.OrdinalIgnoreCase) ? "nifti-gz" : "nifti",
            niftiAffine = BuildNiftiAffine(bytes, littleEndian)
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

    private static void WriteNrrdLabelMap(string path, ushort[] values, MonaiVolumeData reference)
    {
        var header = reference?.nrrdHeader;
        var lines = new List<string>
        {
            "NRRD0005",
            "# Generated by MONAINcnnReproRunner",
            "type: uint16",
            "dimension: 3",
            $"sizes: {reference.dim0} {reference.dim1} {reference.dim2}"
        };

        if (header != null && header.TryGetValue("space", out var space) && !string.IsNullOrWhiteSpace(space))
            lines.Add("space: " + space);
        if (header != null && header.TryGetValue("space directions", out var directions) && !string.IsNullOrWhiteSpace(directions))
            lines.Add("space directions: " + directions);
        else if (header != null && header.TryGetValue("spacings", out var spacingsText) && !string.IsNullOrWhiteSpace(spacingsText))
            lines.Add("spacings: " + spacingsText);
        else if (reference.spacing != null && reference.spacing.Length >= 3)
            lines.Add("spacings: " + reference.spacing[0].ToString(CultureInfo.InvariantCulture) + " " + reference.spacing[1].ToString(CultureInfo.InvariantCulture) + " " + reference.spacing[2].ToString(CultureInfo.InvariantCulture));

        if (header != null && header.TryGetValue("kinds", out var kinds) && !string.IsNullOrWhiteSpace(kinds))
            lines.Add("kinds: " + kinds);
        else
            lines.Add("kinds: domain domain domain");

        if (header != null && header.TryGetValue("space origin", out var origin) && !string.IsNullOrWhiteSpace(origin))
            lines.Add("space origin: " + origin);
        if (header != null && header.TryGetValue("space units", out var units) && !string.IsNullOrWhiteSpace(units))
            lines.Add("space units: " + units);

        lines.Add("endian: little");
        lines.Add("encoding: gzip");
        lines.Add(string.Empty);
        lines.Add(string.Empty);

        var raw = ToNrrdPayload(values, reference.dim0, reference.dim1, reference.dim2);
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        using var fs = File.Create(path);
        var headerBytes = Encoding.ASCII.GetBytes(string.Join("\n", lines));
        fs.Write(headerBytes, 0, headerBytes.Length);
        using var gz = new GZipOutputStream(fs);
        gz.Write(raw, 0, raw.Length);
        gz.Finish();
    }

    private static byte[] ToNrrdPayload(ushort[] values, int dim0, int dim1, int dim2)
    {
        var raw = new byte[values.Length * sizeof(ushort)];
        var dst = 0;
        for (var z = 0; z < dim2; z++)
        {
            for (var y = 0; y < dim1; y++)
            {
                for (var x = 0; x < dim0; x++)
                {
                    var src = ((x * dim1) + y) * dim2 + z;
                    var value = values[src];
                    raw[dst++] = (byte)(value & 0xFF);
                    raw[dst++] = (byte)(value >> 8);
                }
            }
        }
        return raw;
    }

    private static byte[] ToNiftiPayload(ushort[] values, int dim0, int dim1, int dim2)
    {
        var raw = new byte[values.Length * sizeof(ushort)];
        var dst = 0;
        for (var z = 0; z < dim2; z++)
        {
            for (var y = 0; y < dim1; y++)
            {
                for (var x = 0; x < dim0; x++)
                {
                    var src = ((x * dim1) + y) * dim2 + z;
                    var value = values[src];
                    raw[dst++] = (byte)(value & 0xFF);
                    raw[dst++] = (byte)(value >> 8);
                }
            }
        }
        return raw;
    }

    private static void WriteNiftiLabelMap(string path, ushort[] values, MonaiVolumeData reference)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        var header = new byte[352];
        WriteInt32ToBytes(header, 0, 348);
        WriteInt16ToBytes(header, 40, 3);
        WriteInt16ToBytes(header, 42, (short)reference.dim0);
        WriteInt16ToBytes(header, 44, (short)reference.dim1);
        WriteInt16ToBytes(header, 46, (short)reference.dim2);
        WriteInt16ToBytes(header, 48, 1);
        WriteInt16ToBytes(header, 70, 512);
        WriteInt16ToBytes(header, 72, 16);
        WriteFloat32ToBytes(header, 76, 0f);
        WriteFloat32ToBytes(header, 80, reference.spacing != null && reference.spacing.Length > 0 ? reference.spacing[0] : 1f);
        WriteFloat32ToBytes(header, 84, reference.spacing != null && reference.spacing.Length > 1 ? reference.spacing[1] : 1f);
        WriteFloat32ToBytes(header, 88, reference.spacing != null && reference.spacing.Length > 2 ? reference.spacing[2] : 1f);
        WriteFloat32ToBytes(header, 92, 1f);
        WriteFloat32ToBytes(header, 108, 352f);
        header[344] = (byte)'n';
        header[345] = (byte)'+';
        header[346] = (byte)'1';
        header[347] = 0;

        if (reference.niftiAffine != null && reference.niftiAffine.Length >= 16)
        {
            WriteInt16ToBytes(header, 254, 1);
            WriteFloat32ToBytes(header, 280, reference.niftiAffine[0]);
            WriteFloat32ToBytes(header, 284, reference.niftiAffine[1]);
            WriteFloat32ToBytes(header, 288, reference.niftiAffine[2]);
            WriteFloat32ToBytes(header, 292, reference.niftiAffine[3]);
            WriteFloat32ToBytes(header, 296, reference.niftiAffine[4]);
            WriteFloat32ToBytes(header, 300, reference.niftiAffine[5]);
            WriteFloat32ToBytes(header, 304, reference.niftiAffine[6]);
            WriteFloat32ToBytes(header, 308, reference.niftiAffine[7]);
            WriteFloat32ToBytes(header, 312, reference.niftiAffine[8]);
            WriteFloat32ToBytes(header, 316, reference.niftiAffine[9]);
            WriteFloat32ToBytes(header, 320, reference.niftiAffine[10]);
            WriteFloat32ToBytes(header, 324, reference.niftiAffine[11]);
        }

        var payload = ToNiftiPayload(values, reference.dim0, reference.dim1, reference.dim2);
        using var fs = File.Create(path);
        using var gz = new GZipOutputStream(fs);
        gz.Write(header, 0, header.Length);
        gz.Write(payload, 0, payload.Length);
        gz.Finish();
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

    private static float[] BuildNiftiAffine(byte[] bytes, bool littleEndian)
    {
        if (bytes == null || bytes.Length < 348)
            return null;

        var sformCode = ReadInt16(bytes, 254, littleEndian);
        if (sformCode > 0 && bytes.Length >= 280 + sizeof(float) * 12)
        {
            var affine = new float[16];
            affine[0] = ReadFloat32(bytes, 280, littleEndian);
            affine[1] = ReadFloat32(bytes, 284, littleEndian);
            affine[2] = ReadFloat32(bytes, 288, littleEndian);
            affine[3] = ReadFloat32(bytes, 292, littleEndian);
            affine[4] = ReadFloat32(bytes, 296, littleEndian);
            affine[5] = ReadFloat32(bytes, 300, littleEndian);
            affine[6] = ReadFloat32(bytes, 304, littleEndian);
            affine[7] = ReadFloat32(bytes, 308, littleEndian);
            affine[8] = ReadFloat32(bytes, 312, littleEndian);
            affine[9] = ReadFloat32(bytes, 316, littleEndian);
            affine[10] = ReadFloat32(bytes, 320, littleEndian);
            affine[11] = ReadFloat32(bytes, 324, littleEndian);
            affine[12] = 0f;
            affine[13] = 0f;
            affine[14] = 0f;
            affine[15] = 1f;
            return affine;
        }

        return null;
    }

    private static Matrix4x4 BuildMatrix4x4(float[] values)
    {
        if (values == null || values.Length < 16)
            throw new ArgumentException("Matrix4x4 requires 16 values.", nameof(values));

        var matrix = new Matrix4x4();
        matrix.m00 = values[0];
        matrix.m01 = values[1];
        matrix.m02 = values[2];
        matrix.m03 = values[3];
        matrix.m10 = values[4];
        matrix.m11 = values[5];
        matrix.m12 = values[6];
        matrix.m13 = values[7];
        matrix.m20 = values[8];
        matrix.m21 = values[9];
        matrix.m22 = values[10];
        matrix.m23 = values[11];
        matrix.m30 = values[12];
        matrix.m31 = values[13];
        matrix.m32 = values[14];
        matrix.m33 = values[15];
        return matrix;
    }

    private static void WriteInt16ToBytes(byte[] bytes, int offset, short value)
    {
        bytes[offset] = (byte)(value & 0xFF);
        bytes[offset + 1] = (byte)((value >> 8) & 0xFF);
    }

    private static void WriteInt32ToBytes(byte[] bytes, int offset, int value)
    {
        bytes[offset] = (byte)(value & 0xFF);
        bytes[offset + 1] = (byte)((value >> 8) & 0xFF);
        bytes[offset + 2] = (byte)((value >> 16) & 0xFF);
        bytes[offset + 3] = (byte)((value >> 24) & 0xFF);
    }

    private static void WriteFloat32ToBytes(byte[] bytes, int offset, float value)
    {
        var src = BitConverter.GetBytes(value);
        bytes[offset] = src[0];
        bytes[offset + 1] = src[1];
        bytes[offset + 2] = src[2];
        bytes[offset + 3] = src[3];
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
