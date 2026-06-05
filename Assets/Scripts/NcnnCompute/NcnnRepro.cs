using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Diagnostics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public partial class NcnnRepro : IDisposable
    {
        internal static readonly HashSet<string> CodeFormerSftMulLayers = new HashSet<string>(StringComparer.Ordinal)
        {
            "Mul_581",
            "Mul_687",
            "Mul_794",
            "Mul_900"
        };

        internal static readonly HashSet<string> CodeFormerSftAddLayers = new HashSet<string>(StringComparer.Ordinal)
        {
            "Add_582",
            "Add_688",
            "Add_795",
            "Add_901"
        };

        internal static readonly HashSet<string> CodeFormerSftResidualLayers = new HashSet<string>(StringComparer.Ordinal)
        {
            "Add_585",
            "Add_691",
            "Add_798",
            "Add_904"
        };

        public readonly struct BufferShape
        {
            public readonly int dims;
            public readonly int w;
            public readonly int h;
            public readonly int d;
            public readonly int c;

            public BufferShape(int dims, int w, int h, int d, int c)
            {
                this.dims = dims;
                this.w = w;
                this.h = h;
                this.d = d;
                this.c = c;
            }
        }

        private readonly struct RtKey : IEquatable<RtKey>
        {
            public readonly int w;
            public readonly int h;
            public readonly int d;
            public readonly RenderTextureFormat format;

            public RtKey(int w, int h, int d, RenderTextureFormat format)
            {
                this.w = w;
                this.h = h;
                this.d = d;
                this.format = format;
            }

            public bool Equals(RtKey other) => w == other.w && h == other.h && d == other.d && format == other.format;
            public override bool Equals(object obj) => obj is RtKey other && Equals(other);
            public override int GetHashCode()
            {
                unchecked
                {
                    var hash = w;
                    hash = (hash * 397) ^ h;
                    hash = (hash * 397) ^ d;
                    hash = (hash * 397) ^ (int)format;
                    return hash;
                }
            }
        }

        public sealed class TensorRef
        {
            public RenderTexture texture;
            public int width;
            public int height;
            public int packs;
            public int refs;
            public bool owned;
        }

        public sealed class IndexRef
        {
            public RenderTexture texture;
            public ComputeBuffer buffer;
            public NcnnTensorBuffer view;
            public int width;
            public int height;
            public int packs;
            public int sourceWidth;
            public int sourceHeight;
            public int refs;
            public bool owned;
        }

        public sealed class CmdTensorRef
        {
            public ComputeTexture texture;
            public int width;
            public int height;
            public int packs;
            public int refs;
            public bool owned;
        }

        public sealed class BufferRef
        {
            public ComputeBuffer buffer;
            public int refs;
            public bool owned;
        }

        public sealed class ConvPack : IDisposable
        {
            public int outC;
            public int inC;
            public int group;
            public int outPacks;
            public int inPacks;
            public int kernelW;
            public int kernelH;
            public int dilationW;
            public int dilationH;
            public int strideW;
            public int strideH;
            public int padLeft;
            public int padRight;
            public int padTop;
            public int padBottom;
            public int biasTerm;
            public int weightSize;
            public int activationType;
            public float activationSlope;
            public bool useBufferPath;
            public bool useWinograd23;
            public bool isDepthWise;
            public ComputeBuffer packedWeight4;
            public ComputeBuffer packedBias4;
            public ComputeBuffer packedWeightTm23;
            public ComputeBuffer packedDepthWiseWeight4;
            public ComputeBuffer rawWeight;
            public ComputeBuffer rawBias;

            public void Dispose()
            {
                try { packedWeight4?.Dispose(); } catch { }
                try { packedBias4?.Dispose(); } catch { }
                try { packedWeightTm23?.Dispose(); } catch { }
                try { packedDepthWiseWeight4?.Dispose(); } catch { }
                try { rawWeight?.Dispose(); } catch { }
                try { rawBias?.Dispose(); } catch { }
            }
        }

        public sealed class InnerProductPack : IDisposable
        {
            public int inFeatures;
            public int outFeatures;
            public int biasTerm;
            public int weightSize;
            public ComputeBuffer w;
            public ComputeBuffer b;

            public void Dispose()
            {
                try { w?.Dispose(); } catch { }
                try { b?.Dispose(); } catch { }
            }
        }

        public sealed class DeconvPack : IDisposable
        {
            public int outC;
            public int inC;
            public int group;
            public int kernelW;
            public int kernelH;
            public int dilationW;
            public int dilationH;
            public int strideW;
            public int strideH;
            public int padLeft;
            public int padRight;
            public int padTop;
            public int padBottom;
            public int outputPadRight;
            public int outputPadBottom;
            public int biasTerm;
            public int weightSize;
            public int activationType;
            public float activationSlope;
            public ComputeBuffer rawWeight;
            public ComputeBuffer rawBias;

            public void Dispose()
            {
                try { rawWeight?.Dispose(); } catch { }
                try { rawBias?.Dispose(); } catch { }
            }
        }

        public sealed class GemmPack : IDisposable
        {
            public float alpha;
            public float beta;
            public bool transA;
            public bool transB;
            public bool constantA;
            public bool constantB;
            public bool constantC;
            public int constantM;
            public int constantN;
            public int constantK;
            public int broadcastTypeC;
            public ComputeBuffer bData;
            public ComputeBuffer cData;
            public float[] bDataCpu;
            public float[] cDataCpu;

            public void Dispose()
            {
                try { bData?.Dispose(); } catch { }
                try { cData?.Dispose(); } catch { }
            }
        }

        public sealed class MemoryDataPack : IDisposable
        {
            public ComputeBuffer data;
            public int dims;
            public int w;
            public int h;
            public int d;
            public int c;

            public void Dispose()
            {
                try { data?.Dispose(); } catch { }
            }
        }

        public sealed class EmbedPack : IDisposable
        {
            public int numOutput;
            public int inputDim;
            public int biasTerm;
            public int weightSize;
            public ComputeBuffer w;
            public ComputeBuffer b;

            public void Dispose()
            {
                try { w?.Dispose(); } catch { }
                try { b?.Dispose(); } catch { }
            }
        }

        public sealed class LayerNormPack : IDisposable
        {
            public int affineSize;
            public float eps;
            public bool affine;
            public ComputeBuffer gamma;
            public ComputeBuffer beta;

            public void Dispose()
            {
                try { gamma?.Dispose(); } catch { }
                try { beta?.Dispose(); } catch { }
            }
        }

        public sealed class GroupNormPack : IDisposable
        {
            public int group;
            public int channels;
            public float eps;
            public bool affine;
            public ComputeBuffer gamma;
            public ComputeBuffer beta;

            public void Dispose()
            {
                try { gamma?.Dispose(); } catch { }
                try { beta?.Dispose(); } catch { }
            }
        }

        public sealed class BatchNormPack : IDisposable
        {
            public int channels;
            public ComputeBuffer biasA;
            public ComputeBuffer scaleB;
            public ComputeBuffer biasA4;
            public ComputeBuffer scaleB4;

            public void Dispose()
            {
                try { biasA?.Dispose(); } catch { }
                try { scaleB?.Dispose(); } catch { }
                try { biasA4?.Dispose(); } catch { }
                try { scaleB4?.Dispose(); } catch { }
            }
        }

        public sealed class MultiHeadAttentionPack : IDisposable
        {
            public int embedDim;
            public int numHeads;
            public int weightDataSize;
            public int kdim;
            public int vdim;
            public int qdim;
            public float scale;
            public ComputeBuffer qW;
            public ComputeBuffer qB;
            public ComputeBuffer kW;
            public ComputeBuffer kB;
            public ComputeBuffer vW;
            public ComputeBuffer vB;
            public ComputeBuffer oW;
            public ComputeBuffer oB;

            public void Dispose()
            {
                try { qW?.Dispose(); } catch { }
                try { qB?.Dispose(); } catch { }
                try { kW?.Dispose(); } catch { }
                try { kB?.Dispose(); } catch { }
                try { vW?.Dispose(); } catch { }
                try { vB?.Dispose(); } catch { }
                try { oW?.Dispose(); } catch { }
                try { oB?.Dispose(); } catch { }
            }
        }

        public sealed class ScalePack : IDisposable
        {
            public int scaleDataSize;
            public bool biasTerm;
            public bool dynamic;
            public ComputeBuffer scale;
            public ComputeBuffer bias;
            public ComputeBuffer packedScale4;
            public ComputeBuffer packedBias4;
            public float[] scaleCpu;
            public float[] biasCpu;

            public void Dispose()
            {
                try { scale?.Dispose(); } catch { }
                try { bias?.Dispose(); } catch { }
                try { packedScale4?.Dispose(); } catch { }
                try { packedBias4?.Dispose(); } catch { }
            }
        }

        public sealed class PReluPack : IDisposable
        {
            public int numSlope;
            public ComputeBuffer slope;
            public float[] slopeCpu;

            public void Dispose()
            {
                try { slope?.Dispose(); } catch { }
            }
        }

        public sealed class QuantizePack : IDisposable
        {
            public int scaleDataSize;
            public ComputeBuffer scale;
            public float[] scaleCpu;

            public void Dispose()
            {
                try { scale?.Dispose(); } catch { }
            }
        }

        public sealed class DequantizePack : IDisposable
        {
            public int scaleDataSize;
            public int biasDataSize;
            public ComputeBuffer scale;
            public ComputeBuffer bias;
            public float[] scaleCpu;
            public float[] biasCpu;

            public void Dispose()
            {
                try { scale?.Dispose(); } catch { }
                try { bias?.Dispose(); } catch { }
            }
        }

        public sealed class RequantizePack : IDisposable
        {
            public int scaleInDataSize;
            public int scaleOutDataSize;
            public int biasDataSize;
            public int activationType;
            public float activationParam0;
            public float activationParam1;
            public ComputeBuffer scaleIn;
            public ComputeBuffer scaleOut;
            public ComputeBuffer bias;
            public float[] scaleInCpu;
            public float[] scaleOutCpu;
            public float[] biasCpu;

            public void Dispose()
            {
                try { scaleIn?.Dispose(); } catch { }
                try { scaleOut?.Dispose(); } catch { }
                try { bias?.Dispose(); } catch { }
            }
        }

        public sealed class NormalizePack : IDisposable
        {
            public bool acrossSpatial;
            public bool acrossChannel;
            public bool channelShared;
            public float eps;
            public int epsMode;
            public int scaleDataSize;
            public ComputeBuffer scale;
            public float[] scaleCpu;

            public void Dispose()
            {
                try { scale?.Dispose(); } catch { }
            }
        }

        public sealed class LrnPack : IDisposable
        {
            public int regionType;
            public int localSize;
            public float alpha;
            public float beta;
            public float bias;

            public void Dispose()
            {
            }
        }

        public sealed class RmsNormPack : IDisposable
        {
            public int affineSize;
            public float eps;
            public bool affine;
            public ComputeBuffer gamma;
            public float[] gammaCpu;

            public void Dispose()
            {
                try { gamma?.Dispose(); } catch { }
            }
        }

        public sealed class RotaryEmbedPack : IDisposable
        {
            public bool interleaved;

            public void Dispose()
            {
            }
        }

        public sealed class SdpaPack : IDisposable
        {
            public bool attnMask;
            public float scale;
            public bool kvCache;
            public bool int8ScaleTerm;

            public void Dispose()
            {
            }
        }

        public sealed class UnfoldPack : IDisposable
        {
            public int kernelW;
            public int kernelH;
            public int dilationW;
            public int dilationH;
            public int strideW;
            public int strideH;
            public int padLeft;
            public int padRight;
            public int padTop;
            public int padBottom;
            public float padValue;

            public void Dispose()
            {
            }
        }

        public sealed class PriorBoxPack : IDisposable
        {
            public float[] minSizes;
            public float[] maxSizes;
            public float[] aspectRatios;
            public float[] variances;
            public bool flip;
            public bool clip;
            public int imageWidth;
            public int imageHeight;
            public float stepWidth;
            public float stepHeight;
            public float offset;
            public bool stepMmdetection;
            public bool centerMmdetection;

            public void Dispose()
            {
            }
        }

        public sealed class InferResult : IDisposable
        {
            private readonly Dictionary<string, TensorRef> _textureBlobs;
            private readonly Dictionary<string, BufferShape> _textureShapes;
            private readonly Dictionary<string, ComputeBuffer> _bufferBlobs;
            private readonly Dictionary<string, BufferRef> _bufferRefs;
            private readonly Dictionary<string, NcnnTensorBuffer> _bufferViews;
            private readonly List<IDisposable> _tempOwned;
            private readonly NcnnRepro _owner;
            private readonly HashSet<TensorRef> _visitedTextures = new HashSet<TensorRef>();
            private readonly HashSet<ComputeBuffer> _visitedBuffers = new HashSet<ComputeBuffer>();

            internal InferResult(
                Dictionary<string, TensorRef> textureBlobs,
                Dictionary<string, BufferShape> textureShapes,
                Dictionary<string, ComputeBuffer> bufferBlobs,
                Dictionary<string, BufferRef> bufferRefs,
                Dictionary<string, NcnnTensorBuffer> bufferViews,
                List<IDisposable> tempOwned,
                NcnnRepro owner)
            {
                _textureBlobs = textureBlobs;
                _textureShapes = textureShapes;
                _bufferBlobs = bufferBlobs;
                _bufferRefs = bufferRefs;
                _bufferViews = bufferViews;
                _tempOwned = tempOwned;
                _owner = owner;
            }

            public RenderTexture GetTexture(string name)
            {
                if (_textureBlobs.TryGetValue(name, out var tr) && tr != null && tr.texture != null)
                    return tr.texture;

                var materialized = _owner.MaterializeTextureFromBuffer(name, _bufferBlobs, _bufferViews);
                if (materialized == null)
                    throw new InvalidOperationException("blob not found: " + name);

                if (_bufferViews.TryGetValue(name, out var view) && view != null)
                    _textureShapes[name] = new BufferShape(view.dims, view.w, view.h, view.d, view.c);

                _textureBlobs[name] = new TensorRef
                {
                    texture = materialized,
                    width = materialized.width,
                    height = materialized.height,
                    packs = materialized.volumeDepth > 0 ? materialized.volumeDepth : 1,
                    refs = 1,
                    owned = true
                };
                return materialized;
            }

            public RenderTexture ExtractTexture(string name)
            {
                if (_textureBlobs.TryGetValue(name, out var tr) && tr != null && tr.texture != null)
                {
                    tr.owned = false;
                    var rt = tr.texture;
                    tr.texture = null;
                    return rt;
                }

                var materialized = _owner.MaterializeTextureFromBuffer(name, _bufferBlobs, _bufferViews);
                if (materialized == null)
                    throw new InvalidOperationException("blob not found: " + name);
                return materialized;
            }

            public ComputeBuffer GetBuffer(string name)
            {
                if (!_bufferBlobs.TryGetValue(name, out var buf) || buf == null)
                    throw new InvalidOperationException("buffer blob not found: " + name);
                return buf;
            }

            public float[] GetBufferData(string name)
            {
                var buf = GetBuffer(name);
                var data = new float[buf.count];
                buf.GetData(data);
                return data;
            }

            public NcnnTensorBuffer GetBufferView(string name)
            {
                if (_bufferViews.TryGetValue(name, out var view) && view != null && view.buffer != null)
                    return view;
                if (_bufferBlobs.TryGetValue(name, out var buf) && buf != null)
                    return new NcnnTensorBuffer(buf, 1, buf.count, 1, 1, 1, false);
                throw new InvalidOperationException("buffer view not found: " + name);
            }

            public bool TryGetLogicalShape(string name, out int dims, out int w, out int h, out int d, out int c)
            {
                if (_bufferViews.TryGetValue(name, out var view) && view != null && view.buffer != null)
                {
                    dims = view.dims;
                    w = view.w;
                    h = view.h;
                    d = view.d;
                    c = view.c;
                    return true;
                }

                if (_textureShapes.TryGetValue(name, out var shape))
                {
                    dims = shape.dims;
                    w = shape.w;
                    h = shape.h;
                    d = shape.d;
                    c = shape.c;
                    return true;
                }

                dims = 0;
                w = 0;
                h = 0;
                d = 0;
                c = 0;
                return false;
            }

            public ComputeBuffer ExtractBuffer(string name)
            {
                if (!_bufferBlobs.TryGetValue(name, out var buf) || buf == null)
                    throw new InvalidOperationException("buffer blob not found: " + name);
                _bufferBlobs.Remove(name);
                _bufferRefs.Remove(name);
                _bufferViews.Remove(name);
                return buf;
            }

            public void Dispose()
            {
                foreach (var kv in _textureBlobs)
                {
                    var tr = kv.Value;
                    if (tr == null || !_visitedTextures.Add(tr))
                        continue;
                    if (tr.owned && tr.texture != null)
                    {
                        try { _owner.ReturnTempArray(tr.texture); } catch { }
                    }
                }

                foreach (var owned in _tempOwned)
                {
                    try
                    {
                        if (owned is ComputeBuffer tempBuffer)
                            _owner.ReturnTempBuffer(tempBuffer);
                        else
                            owned?.Dispose();
                    }
                    catch
                    {
                    }
                }

                foreach (var kv in _bufferRefs)
                {
                    var br = kv.Value;
                    if (br == null || !br.owned || br.buffer == null || !_visitedBuffers.Add(br.buffer))
                        continue;
                    try { _owner.ReturnTempBuffer(br.buffer); } catch { }
                }

                _bufferBlobs.Clear();
                _bufferRefs.Clear();
                _bufferViews.Clear();
                _tempOwned.Clear();
            }
        }

        public NcnnParamModel Model { get; private set; }
        public IReadOnlyList<NcnnBaseLayerRepro> LayerRepros { get; private set; }
        public ModelLoadProfile LastLoadProfile { get; private set; }
        public bool ForceBufferBinaryOpAll { get; set; }
        public bool ForceBufferGeluAll { get; set; }
        // Default false: some runners intentionally use the buffer GELU fallback as a GPU sync point via SetData.
        public bool EnableGpuGeluBufferPath { get; set; }
        public bool EnableDepthWiseTextureConvolution { get; set; } = true;
        public bool EnableConv1x1TextureConvolution { get; set; } = true;
        public bool EnableGroupNormTexturePath { get; set; }
        public bool ForceBufferConvolution { get; set; }
        public bool UseTextureMaxPoolingInd { get; set; }
        public bool UseNcnnStyleGroupNorm { get; set; }

        internal readonly Dictionary<string, ConvPack> _conv = new Dictionary<string, ConvPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, DeconvPack> _deconv = new Dictionary<string, DeconvPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, InnerProductPack> _innerProduct = new Dictionary<string, InnerProductPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, GemmPack> _gemm = new Dictionary<string, GemmPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, MemoryDataPack> _memoryData = new Dictionary<string, MemoryDataPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, EmbedPack> _embed = new Dictionary<string, EmbedPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, LayerNormPack> _layerNorm = new Dictionary<string, LayerNormPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, GroupNormPack> _groupNorm = new Dictionary<string, GroupNormPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, BatchNormPack> _batchNorm = new Dictionary<string, BatchNormPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, MultiHeadAttentionPack> _multiHeadAttention = new Dictionary<string, MultiHeadAttentionPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, IDisposable> _extraPacks = new Dictionary<string, IDisposable>(StringComparer.Ordinal);
        internal Dictionary<string, int> _blobUseCount;
        private readonly Dictionary<RtKey, Stack<RenderTexture>> _rtPool = new Dictionary<RtKey, Stack<RenderTexture>>();
        private readonly NcnnTempComputeBufferPool _bufferPool = new NcnnTempComputeBufferPool();
        private readonly HashSet<ComputeTexture> _cmdSets = new HashSet<ComputeTexture>();
        private readonly float[] _gpuSyncScratch = new float[1];
        private int _runtimeProfileInferenceIndex;

        private readonly NcnnOps _ops;
        private bool _useTempPool = false;
        private int _maxPooledPerShape = 2;

        public bool EnableTempPool
        {
            get => _useTempPool;
            set
            {
                _useTempPool = value;
                _bufferPool.Enabled = value;
            }
        }

        public int MaxPooledPerShape
        {
            get => _maxPooledPerShape;
            set
            {
                _maxPooledPerShape = Mathf.Max(0, value);
                _bufferPool.MaxPooledPerShape = _maxPooledPerShape;
            }
        }

        public const bool EnableWinograd23 = false;
        public bool PreferTexturePathForFaceDetector { get; set; }
        public bool ForceBufferConvolutionAll { get; set; }
        public bool KeepRawConvWeightsForTexturePath { get; set; } = true;
        public bool EnableMhaParallelSoftmax { get; set; }
        public bool EnableMhaQkvFusion { get; set; }
        public RenderTextureFormat TensorTextureFormat { get; set; } = RenderTextureFormat.ARGBHalf;
        public ISet<string> DebugCompareTextureLayers { get; set; }
        public ISet<string> DebugCompareTextureConvLayers { get; set; }
        public ISet<string> DebugCompareMaxPoolingLayers { get; set; }
        public Action<string> DebugLog { get; set; }
        public float CodeFormerSftMulScale { get; set; } = 1f;
        public float CodeFormerSftAddScale { get; set; } = 1f;
        public bool CodeFormerBypassSftMul { get; set; }
        public string CodeFormerTargetSftMulLayer { get; set; }
        public string CodeFormerTargetSftAddLayer { get; set; }
        public string CodeFormerTargetSftResidualLayer { get; set; }
        public NcnnOps Ops => _ops;
        public bool gpuLayerProfileEnabled = false;
        public bool useExperimentalIteratePath = false;
        public bool LayerRuntimeProfileEnabled { get; set; }
        public bool LayerRuntimeProfileSyncGpu { get; set; }
        public LayerRuntimeProfile LastRuntimeProfile { get; private set; }
        public event Action<string, string, int, int, int, int, double> OnConvComplete;

        internal void NotifyConvComplete(string layerName, string mode, int srcW, int srcH, int inPacks, int outPacks, double gpuMs)
        {
            try { OnConvComplete?.Invoke(layerName, mode, srcW, srcH, inPacks, outPacks, gpuMs); } catch { }
        }

        public NcnnRepro(NcnnOps ops)
        {
            _ops = ops ?? throw new ArgumentNullException(nameof(ops));
            _bufferPool.MaxPooledPerShape = _maxPooledPerShape;
        }

        public void LoadModel(string paramText, NcnnBinReader br, Action<LoadProgress> onProgress = null)
        {
            try
            {
                foreach (var progress in EnumerateModelLoad(paramText, br))
                    onProgress?.Invoke(progress);
            }
            catch
            {
                try { Release(); } catch { }
                throw;
            }
        }

        public async UniTask LoadModelAsync(
            string paramText,
            NcnnBinReader br,
            Action<LoadProgress> onProgress = null,
            CancellationToken ct = default,
            int yieldEveryLayers = 6)
        {
            yieldEveryLayers = Mathf.Max(1, yieldEveryLayers);
            var nextYieldLayer = yieldEveryLayers;

            try
            {
                foreach (var progress in EnumerateModelLoad(paramText, br))
                {
                    onProgress?.Invoke(progress);

                    var shouldYield = false;
                    if (!string.Equals(progress.stage, "layer", StringComparison.Ordinal))
                    {
                        shouldYield = true;
                    }
                    else if (progress.layerIndex >= nextYieldLayer || progress.layerIndex >= progress.layerCount)
                    {
                        shouldYield = true;
                        nextYieldLayer = progress.layerIndex + yieldEveryLayers;
                    }

                    ct.ThrowIfCancellationRequested();
                    if (shouldYield)
                        await UniTask.Yield();
                }
            }
            catch
            {
                try { Release(); } catch { }
                throw;
            }
        }

        private IEnumerable<LoadProgress> EnumerateModelLoad(string paramText, NcnnBinReader br)
        {
            if (paramText == null)
                throw new ArgumentNullException(nameof(paramText));
            if (br == null)
                throw new ArgumentNullException(nameof(br));

            var stageSw = Stopwatch.StartNew();
            var profile = new ModelLoadProfile();
            LastLoadProfile = profile;
            long totalLoadMs = 0;

            Release();
            stageSw.Stop();
            profile.releaseMs = stageSw.ElapsedMilliseconds;
            totalLoadMs += profile.releaseMs;
            yield return new LoadProgress("release", 0, 0, null, null, 0.01f);

            stageSw.Restart();
            Model = NcnnParamParser.Parse(paramText);
            LayerRepros = NcnnLayerFactoryRepro.CreateModelLayers(Model?.layers);
            stageSw.Stop();
            profile.parseParamMs = stageSw.ElapsedMilliseconds;
            totalLoadMs += profile.parseParamMs;
            profile.modelMagic = Model?.magic;
            profile.layerCount = Model?.layers?.Count ?? 0;
            yield return new LoadProgress("parse", 0, profile.layerCount, null, null, 0.03f);

            stageSw.Restart();
            _blobUseCount = BuildBlobUseCount(Model);
            stageSw.Stop();
            profile.buildBlobUseCountMs = stageSw.ElapsedMilliseconds;
            totalLoadMs += profile.buildBlobUseCountMs;
            yield return new LoadProgress("build-blobs", 0, profile.layerCount, null, null, 0.05f);

            var totalLayers = Model?.layers?.Count ?? 0;
            for (var i = 0; i < totalLayers; i++)
            {
                var layer = Model.layers[i];
                var layerSw = Stopwatch.StartNew();
                var layerRepro = LayerRepros != null && i < LayerRepros.Count ? LayerRepros[i] : null;
                var metrics = layerRepro != null ? layerRepro.LoadLayer(this, layer, br) : LoadLayer(layer, br);
                layerSw.Stop();
                totalLoadMs += layerSw.ElapsedMilliseconds;

                AccumulateLayerProfile(profile, layer?.typeName, metrics, layerSw.ElapsedMilliseconds);

                var progress01 = totalLayers > 0
                    ? 0.05f + 0.94f * ((float)(i + 1) / totalLayers)
                    : 0.99f;
                yield return new LoadProgress("layer", i + 1, totalLayers, layer?.name, layer?.typeName, progress01);
            }

            profile.totalMs = totalLoadMs;
            yield return new LoadProgress("complete", totalLayers, totalLayers, null, null, 1f);
        }

        private LayerLoadMetrics LoadLayer(NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            if (layer == null)
                return default;

            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            if (layer.type == NcnnLayerTypes.Convolution
                || layer.type == NcnnLayerTypes.ConvolutionDepthWise)
            {
                var pack = new ConvPack();
                pack.outC = layer.GetInt(0, 0);
                pack.group = Mathf.Max(1, layer.GetInt(7, 1));
                pack.kernelW = layer.GetInt(1, 0);
                pack.kernelH = layer.GetInt(11, pack.kernelW);
                pack.dilationW = layer.GetInt(2, 1);
                pack.dilationH = layer.GetInt(12, pack.dilationW);
                pack.strideW = layer.GetInt(3, 1);
                pack.strideH = layer.GetInt(13, pack.strideW);
                pack.padLeft = layer.GetInt(4, 0);
                pack.padRight = layer.GetInt(15, pack.padLeft);
                pack.padTop = layer.GetInt(14, pack.padLeft);
                pack.padBottom = layer.GetInt(16, pack.padTop);
                pack.biasTerm = layer.GetInt(5, 0);
                pack.weightSize = layer.GetInt(6, 0);
                pack.activationType = layer.GetInt(9, 0);
                pack.activationSlope = ParseLeakySlope(layer);
                pack.isDepthWise = layer.type == NcnnLayerTypes.ConvolutionDepthWise;

                var kernelArea = Mathf.Max(1, pack.kernelW * pack.kernelH);
                if (pack.isDepthWise)
                {
                    pack.inC = Mathf.Max(1, (pack.weightSize * pack.group) / Mathf.Max(1, pack.outC * kernelArea));
                    pack.useBufferPath = true;
                }
                else
                {
                    pack.inC = Mathf.Max(1, pack.weightSize / Mathf.Max(1, pack.outC * kernelArea));
                    pack.useBufferPath = pack.strideW != 1
                                         || pack.strideH != 1
                                         || pack.kernelW != 1 && pack.kernelW != 3
                                         || pack.kernelH != pack.kernelW
                                         || pack.dilationW != 1
                                         || pack.dilationH != 1
                                         || pack.padLeft != pack.padRight
                                         || pack.padTop != pack.padBottom
                                         || pack.kernelW != 3 && pack.kernelW != 1;
                }
                pack.inPacks = (pack.inC + 3) / 4;
                pack.outPacks = (pack.outC + 3) / 4;

                phaseSw.Restart();
                var w = ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
                var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                var needGeneralTexturePack = !ForceBufferConvolutionAll
                                             && !pack.useBufferPath
                                             && !pack.isDepthWise
                                             && pack.group == 1
                                             && !(pack.kernelW == 1 && pack.kernelH == 1 && !EnableConv1x1TextureConvolution);
                var needDepthWiseTexturePack = !ForceBufferConvolutionAll
                                               && EnableDepthWiseTextureConvolution
                                               && pack.isDepthWise
                                               && pack.group == pack.inC
                                               && pack.outC == pack.inC
                                               && pack.kernelW == 3
                                               && pack.kernelH == 3
                                               && pack.dilationW == 1
                                               && pack.dilationH == 1
                                               && pack.padLeft == pack.padRight
                                               && pack.padTop == pack.padBottom
                                               && pack.padLeft == pack.padTop;

                if (ShouldKeepRawConvWeightsForTexturePath(layer.name, pack, needGeneralTexturePack, needDepthWiseTexturePack))
                {
                    phaseSw.Restart();
                    UploadRawConvWeights(pack, w, b);
                    phaseSw.Stop();
                    uploadMs += phaseSw.ElapsedMilliseconds;
                }

                if (needGeneralTexturePack)
                {
                    phaseSw.Restart();
                    var w4 = PackWeightsToO4I4K(w, pack.outC, pack.inC, pack.kernelW, pack.outPacks, pack.inPacks);
                    var b4 = PackBiasToO4(b, pack.outC, pack.outPacks);
                    pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    pack.packedWeight4.SetData(w4);
                    pack.packedBias4.SetData(b4);

                    if (EnableWinograd23
                        && pack.kernelW == 3
                        && pack.kernelH == 3
                        && pack.strideW == 1
                        && pack.strideH == 1
                        && pack.padLeft == 1
                        && pack.padRight == 1
                        && pack.padTop == 1
                        && pack.padBottom == 1
                        && NcnnWinograd23.CanUse(pack.kernelW, pack.padLeft, pack.inPacks, pack.outPacks))
                    {
                        pack.useWinograd23 = true;
                        var wTm = NcnnWinograd23.PackWeightTm23(w, pack.outC, pack.inC, pack.outPacks, pack.inPacks);
                        pack.packedWeightTm23 = new ComputeBuffer(wTm.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                        pack.packedWeightTm23.SetData(wTm);
                    }
                    phaseSw.Stop();
                    packMs += phaseSw.ElapsedMilliseconds;
                }
                else if (needDepthWiseTexturePack)
                {
                    phaseSw.Restart();
                    var w4 = PackDepthWiseWeightsToP4K4(w, pack.outC, pack.kernelW, pack.outPacks);
                    var b4 = PackBiasToO4(b, pack.outC, pack.outPacks);
                    pack.packedDepthWiseWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    pack.packedDepthWiseWeight4.SetData(w4);
                    pack.packedBias4.SetData(b4);
                    phaseSw.Stop();
                    packMs += phaseSw.ElapsedMilliseconds;
                }

                _conv[layer.name] = pack;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.Deconvolution)
            {
                var pack = new DeconvPack();
                pack.outC = layer.GetInt(0, 0);
                pack.group = Mathf.Max(1, layer.GetInt(7, 1));
                pack.kernelW = layer.GetInt(1, 0);
                pack.kernelH = layer.GetInt(11, pack.kernelW);
                pack.dilationW = layer.GetInt(2, 1);
                pack.dilationH = layer.GetInt(12, pack.dilationW);
                pack.strideW = layer.GetInt(3, 1);
                pack.strideH = layer.GetInt(13, pack.strideW);
                pack.padLeft = layer.GetInt(4, 0);
                pack.padRight = layer.GetInt(15, pack.padLeft);
                pack.padTop = layer.GetInt(14, pack.padLeft);
                pack.padBottom = layer.GetInt(16, pack.padTop);
                pack.outputPadRight = layer.GetInt(18, 0);
                pack.outputPadBottom = layer.GetInt(19, pack.outputPadRight);
                pack.biasTerm = layer.GetInt(5, 0);
                pack.weightSize = layer.GetInt(6, 0);
                pack.activationType = layer.GetInt(9, 0);
                pack.activationSlope = ParseLeakySlope(layer);

                var kernelArea = Mathf.Max(1, pack.kernelW * pack.kernelH);
                pack.inC = Mathf.Max(1, (pack.weightSize * pack.group) / Mathf.Max(1, pack.outC * kernelArea));

                phaseSw.Restart();
                var w = ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
                var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                pack.rawWeight = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                pack.rawBias = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                pack.rawWeight.SetData(w);
                pack.rawBias.SetData(b);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _deconv[layer.name] = pack;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.InnerProduct)
            {
                var ip = new InnerProductPack();
                ip.outFeatures = layer.GetInt(0, 0);
                ip.biasTerm = layer.GetInt(1, 0);
                ip.weightSize = layer.GetInt(2, 0);
                ip.inFeatures = ip.outFeatures > 0 ? ip.weightSize / ip.outFeatures : 0;

                phaseSw.Restart();
                var w = ReadPackedOrRawWeightArray(br, ip.weightSize, layer.name);
                var b = ip.biasTerm != 0 ? br.ReadFloat32Array(ip.outFeatures) : new float[ip.outFeatures];
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                ip.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                ip.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                ip.w.SetData(w);
                ip.b.SetData(b);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _innerProduct[layer.name] = ip;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.Gemm)
            {
                var gp = new GemmPack
                {
                    alpha = layer.GetFloat(0, 1f),
                    beta = layer.GetFloat(1, 1f),
                    transA = layer.GetInt(2, 0) != 0,
                    transB = layer.GetInt(3, 0) != 0,
                    constantA = layer.GetInt(4, 0) != 0,
                    constantB = layer.GetInt(5, 0) != 0,
                    constantC = layer.GetInt(6, 0) != 0,
                    constantM = layer.GetInt(7, 0),
                    constantN = layer.GetInt(8, 0),
                    constantK = layer.GetInt(9, 0),
                    broadcastTypeC = layer.GetInt(10, 0)
                };

                if (gp.constantA)
                    throw new InvalidOperationException("Gemm constantA is not supported in NcnnRepro: " + layer.name);
                if (!gp.constantB)
                    throw new InvalidOperationException("Gemm currently expects constantB=1: " + layer.name);

                var bw = gp.transB ? gp.constantK : gp.constantN;
                var bh = gp.transB ? gp.constantN : gp.constantK;

                phaseSw.Restart();
                var b = ReadClipMatAsFloat32(br, bw, bh, 0, 0, 0);
                gp.bDataCpu = b;
                if (gp.constantC && gp.broadcastTypeC != -1)
                {
                    int cw;
                    int ch;
                    switch (gp.broadcastTypeC)
                    {
                        case 0: cw = 1; ch = 0; break;
                        case 1: cw = gp.constantM; ch = 0; break;
                        case 2: cw = 1; ch = gp.constantM; break;
                        case 3: cw = gp.constantN; ch = gp.constantM; break;
                        case 4: cw = gp.constantN; ch = 1; break;
                        default:
                            throw new InvalidOperationException("Gemm broadcast_type_C unsupported: " + gp.broadcastTypeC + " | " + layer.name);
                    }

                    var c = ReadClipMatAsFloat32(br, cw, ch, 0, 0, 0);
                    gp.cDataCpu = c;
                }
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                gp.bData = NewBuffer(gp.bDataCpu);
                if (gp.cDataCpu != null)
                    gp.cData = NewBuffer(gp.cDataCpu);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _gemm[layer.name] = gp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.MemoryData)
            {
                var w = layer.GetInt(0, 0);
                var h = layer.GetInt(1, 0);
                var d = layer.GetInt(11, 0);
                var c = layer.GetInt(2, 0);
                var loadType = layer.GetInt(21, 1);

                phaseSw.Restart();
                var a = ReadClipMatAsFloat32(br, w, h, d, c, loadType);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                var buf = new ComputeBuffer(a.Length, sizeof(float), ComputeBufferType.Structured);
                buf.SetData(a);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                var dims = 1;
                if (h > 0) dims = 2;
                if (c > 0) dims = d > 0 ? 4 : 3;
                _memoryData[layer.name] = new MemoryDataPack
                {
                    data = buf,
                    dims = dims,
                    w = Mathf.Max(1, w),
                    h = Mathf.Max(1, h),
                    d = Mathf.Max(1, d),
                    c = Mathf.Max(1, c)
                };
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.Embed)
            {
                var ep = new EmbedPack();
                ep.numOutput = layer.GetInt(0, 0);
                ep.inputDim = layer.GetInt(1, 0);
                ep.biasTerm = layer.GetInt(2, 0);
                ep.weightSize = layer.GetInt(3, 0);

                phaseSw.Restart();
                var w = ReadClipArrayAsFloat32(br, ep.weightSize, 0);
                float[] b = null;
                if (ep.biasTerm != 0)
                    b = br.ReadNcnnMatAsFloat32(ep.numOutput, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                ep.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                ep.w.SetData(w);
                if (b != null)
                {
                    ep.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                    ep.b.SetData(b);
                }
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _embed[layer.name] = ep;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.LayerNorm)
            {
                var lp = new LayerNormPack();
                lp.affineSize = layer.GetInt(0, 0);
                lp.eps = layer.GetFloat(1, 1e-5f);
                lp.affine = layer.GetInt(2, 1) != 0;

                float[] gamma = null;
                float[] beta = null;
                if (lp.affine && lp.affineSize > 0)
                {
                    phaseSw.Restart();
                    gamma = br.ReadNcnnMatAsFloat32(lp.affineSize, 0, 0, 0, 1);
                    beta = br.ReadNcnnMatAsFloat32(lp.affineSize, 0, 0, 0, 1);
                    phaseSw.Stop();
                    readMs += phaseSw.ElapsedMilliseconds;

                    phaseSw.Restart();
                    lp.gamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
                    lp.beta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
                    lp.gamma.SetData(gamma);
                    lp.beta.SetData(beta);
                    phaseSw.Stop();
                    uploadMs += phaseSw.ElapsedMilliseconds;
                }

                _layerNorm[layer.name] = lp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.GroupNorm)
            {
                var gp = new GroupNormPack();
                gp.group = layer.GetInt(0, 1);
                gp.channels = layer.GetInt(1, 0);
                gp.eps = layer.GetFloat(2, 1e-5f);
                gp.affine = layer.GetInt(3, 1) != 0;

                float[] gamma = null;
                float[] beta = null;
                if (gp.affine && gp.channels > 0)
                {
                    phaseSw.Restart();
                    gamma = br.ReadNcnnMatAsFloat32(gp.channels, 0, 0, 0, 1);
                    beta = br.ReadNcnnMatAsFloat32(gp.channels, 0, 0, 0, 1);
                    phaseSw.Stop();
                    readMs += phaseSw.ElapsedMilliseconds;

                    phaseSw.Restart();
                    gp.gamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
                    gp.beta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
                    gp.gamma.SetData(gamma);
                    gp.beta.SetData(beta);
                    phaseSw.Stop();
                    uploadMs += phaseSw.ElapsedMilliseconds;
                }

                _groupNorm[layer.name] = gp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.BatchNorm)
            {
                var bp = new BatchNormPack();
                bp.channels = layer.GetInt(0, 0);

                phaseSw.Restart();
                var slope = br.ReadNcnnMatAsFloat32(bp.channels, 0, 0, 0, 1);
                var mean = br.ReadNcnnMatAsFloat32(bp.channels, 0, 0, 0, 1);
                var variance = br.ReadNcnnMatAsFloat32(bp.channels, 0, 0, 0, 1);
                var bias = br.ReadNcnnMatAsFloat32(bp.channels, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                var eps = layer.GetFloat(1, 0f);
                var a = new float[bp.channels];
                var b = new float[bp.channels];
                for (var i = 0; i < bp.channels; i++)
                {
                    var sqrtVar = Mathf.Sqrt(variance[i] + eps);
                    if (Mathf.Abs(sqrtVar) < 1e-8f)
                        sqrtVar = 1e-4f;
                    b[i] = slope[i] / sqrtVar;
                    a[i] = bias[i] - slope[i] * mean[i] / sqrtVar;
                }

                phaseSw.Restart();
                var packs = (bp.channels + 3) / 4;
                var a4 = PackBiasToO4(a, bp.channels, packs);
                var b4 = PackBiasToO4(b, bp.channels, packs);
                bp.biasA4 = new ComputeBuffer(a4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                bp.scaleB4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                bp.biasA4.SetData(a4);
                bp.scaleB4.SetData(b4);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;
                packMs += phaseSw.ElapsedMilliseconds;

                _batchNorm[layer.name] = bp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.MultiHeadAttention)
            {
                var mp = new MultiHeadAttentionPack();
                mp.embedDim = layer.GetInt(0, 0);
                mp.numHeads = layer.GetInt(1, 1);
                mp.weightDataSize = layer.GetInt(2, 0);
                mp.kdim = layer.GetInt(3, mp.embedDim);
                mp.vdim = layer.GetInt(4, mp.embedDim);
                mp.scale = layer.GetFloat(6, 1f / Mathf.Sqrt(Mathf.Max(1, mp.embedDim / Mathf.Max(1, mp.numHeads))));
                mp.qdim = mp.embedDim > 0 ? mp.weightDataSize / Mathf.Max(1, mp.embedDim) : 0;

                phaseSw.Restart();
                var qW = ReadClipMatAsFloat32(br, mp.embedDim * mp.qdim, 0, 0, 0, 0);
                var qB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                var kW = ReadClipMatAsFloat32(br, mp.embedDim * mp.kdim, 0, 0, 0, 0);
                var kB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                var vW = ReadClipMatAsFloat32(br, mp.embedDim * mp.vdim, 0, 0, 0, 0);
                var vB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                var oW = ReadClipMatAsFloat32(br, mp.qdim * mp.embedDim, 0, 0, 0, 0);
                var oB = br.ReadNcnnMatAsFloat32(mp.qdim, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                mp.qW = NewBuffer(qW);
                mp.qB = NewBuffer(qB);
                mp.kW = NewBuffer(kW);
                mp.kB = NewBuffer(kB);
                mp.vW = NewBuffer(vW);
                mp.vB = NewBuffer(vB);
                mp.oW = NewBuffer(oW);
                mp.oB = NewBuffer(oB);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _multiHeadAttention[layer.name] = mp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        private static void AccumulateLayerProfile(ModelLoadProfile profile, string layerType, LayerLoadMetrics metrics, long totalMs)
        {
            if (profile == null)
                return;

            profile.totalBytesRead += metrics.bytesRead;
            layerType = string.IsNullOrWhiteSpace(layerType) ? "Unknown" : layerType;
            if (!profile.layerTypes.TryGetValue(layerType, out var typeProfile) || typeProfile == null)
            {
                typeProfile = new LayerTypeLoadProfile();
                profile.layerTypes[layerType] = typeProfile;
            }

            typeProfile.count++;
            typeProfile.totalMs += totalMs;
            typeProfile.bytesRead += metrics.bytesRead;
            typeProfile.readMs += metrics.readMs;
            typeProfile.uploadMs += metrics.uploadMs;
            typeProfile.packMs += metrics.packMs;
        }

        public InferResult Infer(RenderTexture inputPack4, int inputPacks, string inputBlobName = "input", ICollection<string> pinnedNames = null)
        {
            if (inputPack4 == null)
                throw new ArgumentNullException(nameof(inputPack4));
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");

            var textureInputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
            {
                { inputBlobName, inputPack4 }
            };
            return InferWithMultiInputs(textureInputs, null, pinnedNames);
        }

        public RenderTexture ForwardPack4(RenderTexture inputPack4, int inputPacks, string inputBlobName = "data", ICollection<string> pinnedNames = null)
        {
            using var infer = Infer(inputPack4, inputPacks, inputBlobName, pinnedNames);
            return infer.ExtractTexture(ResolveDefaultOutputBlobName());
        }

        public InferResult InferFromBuffers(Dictionary<string, ComputeBuffer> inputBuffers, string stopAfterTopName = null)
        {
            if (inputBuffers == null || inputBuffers.Count == 0)
                throw new ArgumentNullException(nameof(inputBuffers));

            var bufferInputs = new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal);
            foreach (var kv in inputBuffers)
            {
                if (kv.Value == null)
                    throw new ArgumentNullException("inputBuffers[\"" + kv.Key + "\"]");
                bufferInputs[kv.Key] = new NcnnTensorBuffer(kv.Value, 1, kv.Value.count, 1, 1, 1, false);
            }

            return InferWithMultiInputs(null, bufferInputs, null);
        }

        public InferResult InferWithMultiInputs(
            Dictionary<string, RenderTexture> textureInputs,
            Dictionary<string, NcnnTensorBuffer> bufferInputs,
            ICollection<string> pinnedNames = null,
            Dictionary<string, BufferShape> textureInputShapes = null)
        {
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");
            if ((textureInputs == null || textureInputs.Count == 0) && (bufferInputs == null || bufferInputs.Count == 0))
                throw new ArgumentNullException(nameof(textureInputs));
            if (LayerRepros != null && LayerRepros.Count == Model.layers.Count)
                return InferWithMultiInputsByLayerRepros(textureInputs, bufferInputs, pinnedNames, textureInputShapes);

            return null;
        }

        private string ResolveDefaultOutputBlobName()
        {
            if (Model != null && Model.layers != null)
            {
                for (var i = Model.layers.Count - 1; i >= 0; i--)
                {
                    var topNames = Model.layers[i]?.topNames;
                    if (topNames != null && topNames.Length > 0 && !string.IsNullOrWhiteSpace(topNames[0]))
                        return topNames[0];
                }
            }

            return "output";
        }

        public ComputeTexture ForwardPack4(CommandBuffer cmd, ComputeTexture inputPack4, int inputPacks, string inputBlobName = "data", ICollection<string> pinnedNames = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (inputPack4 == null)
                throw new ArgumentNullException(nameof(inputPack4));
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");
            if (LayerRepros != null && LayerRepros.Count == Model.layers.Count)
                return ForwardPack4ByLayerRepros(cmd, inputPack4, inputPacks, inputBlobName, pinnedNames);

            var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
            var blobs = new Dictionary<string, CmdTensorRef>(StringComparer.Ordinal)
            {
                [inputBlobName] = new CmdTensorRef
                {
                    texture = inputPack4,
                    width = inputPack4.width,
                    height = inputPack4.height,
                    packs = inputPacks,
                    refs = 1,
                    owned = false
                }
            };

            for (var li = 0; li < Model.layers.Count; li++)
            {
                var l = Model.layers[li];
                if (l.type == NcnnLayerTypes.Input)
                    continue;

                if (l.type == NcnnLayerTypes.Split)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    for (var i = 0; i < l.topNames.Length; i++)
                    {
                        blobs[l.topNames[i]] = src;
                        src.refs++;
                    }
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Concat)
                {
                    var parts = new CmdTensorRef[l.bottomNames.Length];
                    var sumP = 0;
                    var w = 0;
                    var h = 0;
                    for (var i = 0; i < l.bottomNames.Length; i++)
                    {
                        var tr = GetCmdTensor(blobs, l.bottomNames[i]);
                        parts[i] = tr;
                        w = tr.width;
                        h = tr.height;
                        sumP += tr.packs;
                    }

                    var outArr = RentTempArray(cmd, w, h, sumP, RenderTextureFormat.ARGBHalf);
                    var off = 0;
                    for (var i = 0; i < parts.Length; i++)
                    {
                        _ops.CopyPack4(cmd, parts[i].texture, 0, outArr, off, parts[i].packs);
                        off += parts[i].packs;
                    }

                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = w, height = h, packs = sumP, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Reshape)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    blobs[l.topNames[0]] = src;
                    src.refs++;
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Padding)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var top = l.GetInt(0, 0);
                    var bottom = l.GetInt(1, 0);
                    var left = l.GetInt(2, 0);
                    var right = l.GetInt(3, 0);
                    var type = l.GetInt(4, 0);
                    var value = l.GetFloat(5, 0f);

                    var outW = src.width + left + right;
                    var outH = src.height + top + bottom;
                    if (outW <= 0 || outH <= 0)
                        throw new InvalidOperationException("Padding invalid out size: " + outW + "x" + outH);

                    var outArr = RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.PaddingPack4(cmd, src.texture, src.packs, left, right, top, bottom, type, new Vector4(value, value, value, value), outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = outW, height = outH, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Pooling)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var poolingType = l.GetInt(0, 0);
                    var kernelW = l.GetInt(1, 0);
                    var kernelH = l.GetInt(11, kernelW);
                    var strideW = l.GetInt(2, 1);
                    var strideH = l.GetInt(12, strideW);
                    var padLeft = l.GetInt(3, 0);
                    var padTop = l.GetInt(13, padLeft);
                    var globalPooling = l.GetInt(4, 0);
                    var adaptivePooling = l.GetInt(7, 0);
                    if (globalPooling != 0 || adaptivePooling != 0)
                        throw new InvalidOperationException("Pooling(global/adaptive) not supported");

                    var outW = (src.width + padLeft * 2 - kernelW) / strideW + 1;
                    var outH = (src.height + padTop * 2 - kernelH) / strideH + 1;
                    outW = Mathf.Max(1, outW);
                    outH = Mathf.Max(1, outH);
                    var outArr = RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.PoolingPack4(cmd, src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, poolingType, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = outW, height = outH, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Softmax)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var axis = l.GetInt(0, 0);
                    if (axis != 0)
                        throw new InvalidOperationException("Softmax axis not supported: " + axis);
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SoftmaxChannelPack4(cmd, src.texture, src.packs, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Convolution || l.type == NcnnLayerTypes.ConvolutionDepthWise)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    if (!_conv.TryGetValue(l.name, out var conv))
                        throw new InvalidOperationException("Convolution not found: " + l.name);
                    if (src.packs != conv.inPacks)
                        throw new InvalidOperationException("unexpected in packs for " + l.name + ": " + src.packs + " vs " + conv.inPacks);
                    if (conv.isDepthWise || conv.group != 1)
                        throw new InvalidOperationException("CommandBuffer convolution does not support depthwise/group conv: " + l.name);
                    if (conv.strideW != 1 || conv.strideH != 1 || conv.dilationW != 1 || conv.dilationH != 1)
                        throw new InvalidOperationException("CommandBuffer convolution only supports stride=1 dilation=1: " + l.name);

                    var outW = ComputeConvOut(src.width, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
                    var outH = ComputeConvOut(src.height, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
                    var outArr = RentTempArray(cmd, outW, outH, conv.outPacks, RenderTextureFormat.ARGBHalf);

                    if (conv.kernelW == 1 && conv.kernelH == 1)
                    {
                        _ops.Conv1x1Pack4(cmd, src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.activationType, conv.activationSlope, outArr);
                    }
                    else if (conv.kernelW == 3 && conv.kernelH == 3 && conv.padLeft == conv.padRight && conv.padLeft == conv.padTop && conv.padTop == conv.padBottom)
                    {
                        var useWinograd = EnableWinograd23
                            && conv.packedWeightTm23 != null
                            && conv.strideW == 1
                            && conv.strideH == 1
                            && conv.padLeft == 1
                            && conv.padTop == 1;
                        if (useWinograd)
                        {
                            _ops.Conv3x3Pack4Winograd23(cmd, src.texture, conv.inPacks, conv.packedWeightTm23, conv.packedBias4, conv.outPacks, conv.biasTerm, conv.activationType, conv.activationSlope, outArr);
                        }
                        else
                        {
                            _ops.Conv3x3Pack4(cmd, src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.padLeft, conv.activationType, conv.activationSlope, outArr);
                        }
                    }
                    else
                    {
                        throw new InvalidOperationException("CommandBuffer convolution only supports 1x1/3x3 symmetric conv: " + l.name);
                    }

                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = outW, height = outH, packs = conv.outPacks, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Eltwise)
                {
                    var a = GetCmdTensor(blobs, l.bottomNames[0]);
                    var b = GetCmdTensor(blobs, l.bottomNames[1]);
                    var coeff = ParseEltwiseCoeff(l);
                    var outArr = RentTempArray(cmd, a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                    _ops.AddPack4(cmd, a.texture, b.texture, coeff.coeffA, coeff.coeffB, a.packs, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = a.width, height = a.height, packs = a.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.BinaryOp)
                {
                    var opType = l.GetInt(0, 0);
                    var withScalar = l.GetInt(1, 0);
                    var scalarB = l.GetFloat(2, 0f);
                    var a = GetCmdTensor(blobs, l.bottomNames[0]);
                    var outArr = RentTempArray(cmd, a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                    if (withScalar != 0)
                    {
                        _ops.BinaryOpScalarPack4(cmd, a.texture, scalarB, a.packs, opType, outArr);
                    }
                    else
                    {
                        var b = GetCmdTensor(blobs, l.bottomNames[1]);
                        if (a.width != b.width || a.height != b.height || a.packs != b.packs)
                            throw new InvalidOperationException("BinaryOp broadcast not supported: " + l.name);
                        _ops.BinaryOpPack4(cmd, a.texture, b.texture, a.packs, opType, outArr);
                    }
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = a.width, height = a.height, packs = a.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.UnaryOp)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var opType = l.GetInt(0, 0);
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.UnaryOpPack4(cmd, src.texture, src.packs, opType, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Swish)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SwishPack4(cmd, src.texture, src.packs, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Sigmoid)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SigmoidPack4(cmd, src.texture, src.packs, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.GELU)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var fast = l.GetInt(0, 0) != 0;
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.GeluPack4(cmd, src.texture, src.packs, fast, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Interp)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var resizeType = l.GetInt(0, 2);
                    var sx = l.GetFloat(1, 1f);
                    var sy = l.GetFloat(2, 1f);

                    if (Mathf.Abs(sx - 2f) < 1e-3f && Mathf.Abs(sy - 2f) < 1e-3f)
                    {
                        var outArr = RentTempArray(cmd, src.width * 2, src.height * 2, src.packs, RenderTextureFormat.ARGBHalf);
                        if (resizeType == 1)
                            _ops.Interp2xNearestPack4(cmd, src.texture, src.packs, outArr);
                        else
                            _ops.Interp2xPack4(cmd, src.texture, src.packs, outArr);
                        blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width * 2, height = src.height * 2, packs = src.packs, refs = 1, owned = true };
                        ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                        continue;
                    }

                    if (Mathf.Abs(sx - 0.5f) < 1e-3f && Mathf.Abs(sy - 0.5f) < 1e-3f)
                    {
                        var outArr = RentTempArray(cmd, src.width / 2, src.height / 2, src.packs, RenderTextureFormat.ARGBHalf);
                        if (resizeType == 1)
                            _ops.InterpDown2NearestPack4(cmd, src.texture, src.packs, outArr);
                        else
                            _ops.InterpDown2Pack4(cmd, src.texture, src.packs, outArr);
                        blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width / 2, height = src.height / 2, packs = src.packs, refs = 1, owned = true };
                        ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                        continue;
                    }

                    throw new InvalidOperationException("unsupported interp scale: " + sx.ToString("0.###", CultureInfo.InvariantCulture) + "," + sy.ToString("0.###", CultureInfo.InvariantCulture));
                }

                throw new InvalidOperationException("unsupported layer type in CommandBuffer path: " + (l.typeName ?? l.type.ToString()));
            }

            var outBlobName = ResolveDefaultOutputBlobName();
            var outRef = GetCmdTensor(blobs, outBlobName);
            var keep = outRef.texture;
            outRef.texture = null;
            outRef.owned = false;

            var visited = new HashSet<CmdTensorRef>();
            foreach (var kv in blobs)
            {
                var tr = kv.Value;
                if (tr == null || !visited.Add(tr))
                    continue;
                if (tr.owned && tr.texture != null)
                    ReturnTempArray(cmd, tr.texture);
            }

            return keep;
        }

        internal static CmdTensorRef GetCmdTensor(Dictionary<string, CmdTensorRef> blobs, string name)
        {
            if (!blobs.TryGetValue(name, out var tr) || tr == null)
                throw new InvalidOperationException("blob not found: " + name);
            return tr;
        }

        internal static BufferShape InferCmdShape(CmdTensorRef tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            return new BufferShape(3, Mathf.Max(1, tensor.width), Mathf.Max(1, tensor.height), 1, Mathf.Max(1, tensor.packs * 4));
        }

        internal static bool TryGetCmdShape(
            Dictionary<string, BufferShape> shapes,
            Dictionary<string, CmdTensorRef> blobs,
            string name,
            out BufferShape shape)
        {
            if (shapes != null && shapes.TryGetValue(name, out shape))
                return true;

            if (blobs != null && blobs.TryGetValue(name, out var tensor) && tensor != null)
            {
                shape = InferCmdShape(tensor);
                return true;
            }

            shape = default;
            return false;
        }

        internal static BufferShape GetCmdShape(
            Dictionary<string, BufferShape> shapes,
            Dictionary<string, CmdTensorRef> blobs,
            string name)
        {
            if (TryGetCmdShape(shapes, blobs, name, out var shape))
                return shape;
            throw new InvalidOperationException("cmd blob shape not found: " + name);
        }

        internal void ConsumeCmd(
            CommandBuffer cmd,
            Dictionary<string, CmdTensorRef> blobs,
            Dictionary<string, int> remaining,
            string[] bottomNames,
            ICollection<string> pinnedNames,
            Dictionary<string, BufferShape> shapes = null)
        {
            for (var i = 0; i < bottomNames.Length; i++)
            {
                var b = bottomNames[i];
                if (!remaining.TryGetValue(b, out var c))
                    continue;
                c--;
                remaining[b] = c;
                if (c > 0)
                    continue;
                if (pinnedNames != null && pinnedNames.Contains(b))
                    continue;

                if (blobs.TryGetValue(b, out var tr) && tr != null)
                {
                    tr.refs--;
                    if (tr.refs <= 0)
                    {
                        if (tr.owned && tr.texture != null)
                            ReturnTempArray(cmd, tr.texture);
                        tr.texture = null;
                        tr.owned = false;
                    }
                }
                blobs.Remove(b);
                shapes?.Remove(b);
            }
        }

        private static string GetTempBufferLabel(string member, int line)
        {
            return "NcnnRepro.RentTempBuffer(" + (member ?? "?") + ":" + line.ToString(CultureInfo.InvariantCulture) + ")";
        }

        internal ComputeBuffer RentTempBuffer(
            int count,
            int stride,
            ComputeBufferType type = ComputeBufferType.Structured,
            [CallerMemberName] string callerMember = null,
            [CallerLineNumber] int callerLine = 0)
        {
            return _bufferPool.Rent(count, stride, type, GetTempBufferLabel(callerMember, callerLine));
        }

        internal void ReturnTempBuffer(ComputeBuffer buffer)
        {
            _bufferPool.Return(buffer, "NcnnRepro.ReturnTempBuffer");
        }

        internal NcnnTensorBuffer RentTempTensorBuffer(
            int dims,
            int w,
            int h = 1,
            int d = 1,
            int c = 1,
            [CallerMemberName] string callerMember = null,
            [CallerLineNumber] int callerLine = 0)
        {
            var count = checked(w * h * d * c);
            var buffer = RentTempBuffer(count, sizeof(float), ComputeBufferType.Structured, callerMember, callerLine);
            return new NcnnTensorBuffer(buffer, dims, w, h, d, c, true, ReturnTempBuffer);
        }

        public RenderTexture RentTempArray(
            int w,
            int h,
            int depth,
            RenderTextureFormat format,
            [CallerMemberName] string callerMember = null,
            [CallerLineNumber] int callerLine = 0)
        {
            w = Mathf.Max(1, w);
            h = Mathf.Max(1, h);
            depth = Mathf.Max(1, depth);
            if (format == RenderTextureFormat.ARGBHalf)
                format = TensorTextureFormat;

            var allocLabel = "NcnnRepro.RentTempArray(" + (callerMember ?? "?") + ":" + callerLine.ToString(CultureInfo.InvariantCulture) + ")";

            var desc = new RenderTextureDescriptor(w, h, format, 0)
            {
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = depth,
                enableRandomWrite = true,
                msaaSamples = 1,
            };
            var allocated = RenderTexture.GetTemporary(desc);
            NcnnGpuResourceTracker.RegisterTexture(allocated, allocLabel + "|new");
            return allocated;

       
        }

        public void ReturnTempArray(RenderTexture rt)
        {
            if (rt == null)
                return;

            NcnnGpuResourceTracker.ReleaseTexture(rt, "NcnnRepro.ReturnTempArray");
            RenderTexture.ReleaseTemporary(rt);
        }

        public ComputeTexture RentTempArray(CommandBuffer cmd, int w, int h, int depth, RenderTextureFormat format)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));

            w = Mathf.Max(1, w);
            h = Mathf.Max(1, h);
            depth = Mathf.Max(1, depth);
            if (format == RenderTextureFormat.ARGBHalf)
                format = TensorTextureFormat;

            var desc = new RenderTextureDescriptor(w, h, format, 0)
            {
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = depth,
                enableRandomWrite = true,
                msaaSamples = 1,
            };

            var id = Shader.PropertyToID(Guid.NewGuid().ToString());
            cmd.GetTemporaryRT(id, desc);
            var t = new ComputeTexture
            {
                nameID = id,
                width = w,
                height = h,
            };
            _cmdSets.Add(t);
            return t;
        }

        public void ReturnTempArray(CommandBuffer cmd, ComputeTexture t)
        {
            if (cmd == null || t == null)
                return;

            if (_cmdSets.Contains(t))
            {
                cmd.ReleaseTemporaryRT(t.nameID);
                _cmdSets.Remove(t);
            }
        }

        public void ClearTempPool()
        {
            foreach (var kv in _rtPool)
            {
                var pool = kv.Value;
                while (pool.Count > 0)
                {
                    var rt = pool.Pop();
                    if (rt == null)
                        continue;
                    NcnnGpuResourceTracker.ReleaseTexture(rt, "NcnnRepro.ClearTempPool");
                    try { RenderTexture.ReleaseTemporary(rt); } catch { }
                }
            }
            _rtPool.Clear();
            _bufferPool.Clear("NcnnRepro.ClearTempPool");
        }

        public void Release()
        {
            foreach (var kv in _conv) kv.Value?.Dispose();
            foreach (var kv in _deconv) kv.Value?.Dispose();
            foreach (var kv in _innerProduct) kv.Value?.Dispose();
            foreach (var kv in _gemm) kv.Value?.Dispose();
            foreach (var kv in _memoryData) kv.Value?.Dispose();
            foreach (var kv in _embed) kv.Value?.Dispose();
            foreach (var kv in _layerNorm) kv.Value?.Dispose();
            foreach (var kv in _groupNorm) kv.Value?.Dispose();
            foreach (var kv in _batchNorm) kv.Value?.Dispose();
            foreach (var kv in _multiHeadAttention) kv.Value?.Dispose();
            foreach (var kv in _extraPacks) kv.Value?.Dispose();

            _conv.Clear();
            _deconv.Clear();
            _innerProduct.Clear();
            _gemm.Clear();
            _memoryData.Clear();
            _embed.Clear();
            _layerNorm.Clear();
            _groupNorm.Clear();
            _batchNorm.Clear();
            _multiHeadAttention.Clear();
            _extraPacks.Clear();
            Model = null;
            LayerRepros = null;
            _blobUseCount = null;
            ClearTempPool();
        }

        public void Dispose()
        {
            Release();
        }

        internal RenderTexture MaterializeTextureFromBufferView(ComputeBuffer buffer, NcnnTensorBuffer view)
        {
            if (buffer == null || view == null)
                return null;

            int texW;
            int texH;
            int channels;
            if (view.dims == 1)
            {
                texW = view.w;
                texH = 1;
                channels = 1;
            }
            else if (view.dims == 2)
            {
                texW = view.w;
                texH = view.h;
                channels = 1;
            }
            else if (view.dims == 3)
            {
                texW = view.w;
                texH = view.h;
                channels = view.c;
            }
            else
            {
                return null;
            }

            var packs = Mathf.CeilToInt(channels / 4f);
            var rt = RentTempArray(texW, texH, packs, RenderTextureFormat.ARGBHalf);
            _ops.FillPack4FromBufferCHW(buffer, texW, texH, channels, rt);
            return rt;
        }

        internal ComputeTexture MaterializeCmdTextureFromBufferView(CommandBuffer cmd, ComputeBuffer buffer, NcnnTensorBuffer view)
        {
            if (cmd == null || buffer == null || view == null)
                return null;

            int texW;
            int texH;
            int channels;
            if (view.dims == 1)
            {
                texW = view.w;
                texH = 1;
                channels = 1;
            }
            else if (view.dims == 2)
            {
                texW = view.w;
                texH = view.h;
                channels = 1;
            }
            else if (view.dims == 3)
            {
                texW = view.w;
                texH = view.h;
                channels = view.c;
            }
            else
            {
                return null;
            }

            var packs = Mathf.CeilToInt(channels / 4f);
            var rt = RentTempArray(cmd, texW, texH, packs, RenderTextureFormat.ARGBHalf);
            _ops.FillPack4FromBufferCHW(cmd, buffer, texW, texH, channels, rt);
            return rt;
        }

        internal RenderTexture MaterializeTextureFromBuffer(
            string name,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews)
        {
            if (!bufferBlobs.TryGetValue(name, out var buffer) || buffer == null)
                return null;
            if (!bufferViews.TryGetValue(name, out var view) || view == null)
                return null;
            return MaterializeTextureFromBufferView(buffer, view);
        }

        internal TensorRef GetOrMaterializeTexture(
            string name,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews)
        {
            if (textureBlobs.TryGetValue(name, out var tr) && tr != null && tr.texture != null)
                return tr;

            var materialized = MaterializeTextureFromBuffer(name, bufferBlobs, bufferViews);
            if (materialized == null)
                throw new InvalidOperationException("blob not found: " + name);

            var packs = materialized.volumeDepth > 0 ? materialized.volumeDepth : 1;
            var shape = bufferViews.TryGetValue(name, out var view) && view != null
                ? new BufferShape(view.dims, view.w, view.h, view.d, view.c)
                : new BufferShape(3, materialized.width, materialized.height, 1, packs * 4);

            tr = new TensorRef
            {
                texture = materialized,
                width = materialized.width,
                height = materialized.height,
                packs = packs,
                refs = 1,
                owned = true
            };
            textureBlobs[name] = tr;
            textureShapes[name] = shape;
            return tr;
        }

        internal void RegisterTextureInputs(
            Dictionary<string, RenderTexture> textureInputs,
            Dictionary<string, BufferShape> textureInputShapes,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes)
        {
            if (textureInputs == null)
                return;

            foreach (var kv in textureInputs)
            {
                if (kv.Value == null)
                    throw new ArgumentNullException("textureInputs[\"" + kv.Key + "\"]");

                var rt = kv.Value;
                var packs = rt.volumeDepth > 0 ? rt.volumeDepth : 1;
                var useCount = _blobUseCount.TryGetValue(kv.Key, out var c) ? c : 1;
                var logicalShape = textureInputShapes != null && textureInputShapes.TryGetValue(kv.Key, out var suppliedShape)
                    ? suppliedShape
                    : new BufferShape(3, rt.width, rt.height, 1, ResolveInputLogicalChannels(kv.Key, packs * 4));

                var physicalCount = rt.width * rt.height * packs * 4;
                var logicalCount = Mathf.Max(1, logicalShape.w) * Mathf.Max(1, logicalShape.h) * Mathf.Max(1, logicalShape.d) * Mathf.Max(1, logicalShape.c);
                if (logicalCount > physicalCount)
                    throw new InvalidOperationException("texture input logical shape exceeds physical storage: " + kv.Key);

                textureBlobs[kv.Key] = new TensorRef
                {
                    texture = rt,
                    width = rt.width,
                    height = rt.height,
                    packs = packs,
                    refs = useCount,
                    owned = false
                };
                textureShapes[kv.Key] = logicalShape;
            }
        }

        internal bool TryGetPack4Texture(
            string name,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            out TensorRef texture,
            out BufferShape shape)
        {
            texture = null;
            shape = default;

            try
            {
                texture = GetOrMaterializeTexture(name, textureBlobs, textureShapes, bufferBlobs, bufferViews);
                if (texture == null || texture.texture == null)
                    return false;

                shape = GetTextureShape(textureShapes, texture, name);
                return shape.dims == 3;
            }
            catch
            {
                texture = null;
                shape = default;
                return false;
            }
        }

        internal static bool AreAllLayerTopsAlreadyAvailable(
            NcnnParamModel.Layer layer,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, IndexRef> indexBlobs)
        {
            var topNames = layer?.topNames;
            if (topNames == null || topNames.Length == 0)
                return false;

            for (var i = 0; i < topNames.Length; i++)
            {
                var name = topNames[i];
                if (string.IsNullOrEmpty(name))
                    return false;

                var hasTexture = textureBlobs != null
                    && textureBlobs.TryGetValue(name, out var tr)
                    && tr != null
                    && tr.texture != null;
                var hasBuffer = bufferBlobs != null
                    && bufferBlobs.TryGetValue(name, out var buf)
                    && buf != null;
                var hasIndex = indexBlobs != null
                    && indexBlobs.TryGetValue(name, out var ir)
                    && ir != null
                    && (ir.texture != null || ir.buffer != null);

                if (!hasTexture && !hasBuffer && !hasIndex)
                    return false;
            }

            return true;
        }

        internal LayerRuntimeProfile BeginLayerRuntimeProfile(string pathKind)
        {
            if (!LayerRuntimeProfileEnabled)
            {
                LastRuntimeProfile = null;
                return null;
            }

            var profile = new LayerRuntimeProfile
            {
                inferenceIndex = ++_runtimeProfileInferenceIndex,
                pathKind = string.IsNullOrWhiteSpace(pathKind) ? "buffer" : pathKind,
                syncGpu = LayerRuntimeProfileSyncGpu
            };
            LastRuntimeProfile = profile;
            return profile;
        }

        internal static void FinishLayerRuntimeProfile(LayerRuntimeProfile profile)
        {
            if (profile == null)
                return;
            profile.totalMs = TicksToMilliseconds(profile.totalTicks);
        }

        internal static void RecordLayerRuntime(
            LayerRuntimeProfile profile,
            int layerIndex,
            NcnnParamModel.Layer layer,
            string path,
            long elapsedTicks)
        {
            if (profile == null || layer == null)
                return;

            if (elapsedTicks < 0)
                elapsedTicks = 0;

            var typeName = string.IsNullOrWhiteSpace(layer.typeName) ? "Unknown" : layer.typeName;
            var record = new LayerRuntimeRecord
            {
                layerIndex = layerIndex,
                layerName = layer.name ?? string.Empty,
                layerType = typeName,
                path = path ?? string.Empty,
                elapsedTicks = elapsedTicks,
                elapsedMs = TicksToMilliseconds(elapsedTicks)
            };
            profile.layers.Add(record);
            profile.totalTicks += elapsedTicks;

            if (!profile.layerTypes.TryGetValue(typeName, out var typeProfile) || typeProfile == null)
            {
                typeProfile = new LayerRuntimeTypeProfile { layerType = typeName };
                profile.layerTypes[typeName] = typeProfile;
            }

            typeProfile.count++;
            typeProfile.totalTicks += elapsedTicks;
            typeProfile.totalMs = TicksToMilliseconds(typeProfile.totalTicks);
            typeProfile.avgMs = typeProfile.count > 0 ? typeProfile.totalMs / typeProfile.count : 0d;
        }

        internal static string DescribeLayerOutputPath(
            NcnnParamModel.Layer layer,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            Dictionary<string, IndexRef> indexBlobs)
        {
            var topNames = layer?.topNames;
            if (topNames == null || topNames.Length == 0)
                return string.Empty;

            var sb = new StringBuilder();
            for (var i = 0; i < topNames.Length; i++)
            {
                if (i > 0)
                    sb.Append(';');

                var name = topNames[i] ?? string.Empty;
                sb.Append(name);
                sb.Append('=');

                if (textureBlobs != null
                    && textureBlobs.TryGetValue(name, out var tex)
                    && tex != null
                    && tex.texture != null)
                {
                    sb.Append("tex:");
                    sb.Append(tex.width.ToString(CultureInfo.InvariantCulture));
                    sb.Append('x');
                    sb.Append(tex.height.ToString(CultureInfo.InvariantCulture));
                    sb.Append('x');
                    sb.Append(tex.packs.ToString(CultureInfo.InvariantCulture));
                    sb.Append('p');
                    if (textureShapes != null && textureShapes.TryGetValue(name, out var shape))
                    {
                        sb.Append(":c");
                        sb.Append(shape.c.ToString(CultureInfo.InvariantCulture));
                    }
                    continue;
                }

                if (bufferBlobs != null
                    && bufferBlobs.TryGetValue(name, out var buffer)
                    && buffer != null)
                {
                    sb.Append("buf:");
                    if (bufferViews != null && bufferViews.TryGetValue(name, out var view) && view != null)
                    {
                        sb.Append('d');
                        sb.Append(view.dims.ToString(CultureInfo.InvariantCulture));
                        sb.Append(':');
                        sb.Append(view.w.ToString(CultureInfo.InvariantCulture));
                        sb.Append('x');
                        sb.Append(view.h.ToString(CultureInfo.InvariantCulture));
                        sb.Append('x');
                        sb.Append(view.c.ToString(CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        sb.Append(buffer.count.ToString(CultureInfo.InvariantCulture));
                    }
                    continue;
                }

                if (indexBlobs != null
                    && indexBlobs.TryGetValue(name, out var index)
                    && index != null
                    && (index.texture != null || index.buffer != null))
                {
                    sb.Append("idx:");
                    sb.Append(index.width.ToString(CultureInfo.InvariantCulture));
                    sb.Append('x');
                    sb.Append(index.height.ToString(CultureInfo.InvariantCulture));
                    sb.Append('x');
                    if (index.texture != null)
                    {
                        sb.Append(index.packs.ToString(CultureInfo.InvariantCulture));
                        sb.Append('p');
                    }
                    else if (index.view != null)
                    {
                        sb.Append("buf:");
                        sb.Append(index.view.w.ToString(CultureInfo.InvariantCulture));
                        sb.Append('x');
                        sb.Append(index.view.h.ToString(CultureInfo.InvariantCulture));
                        sb.Append('x');
                        sb.Append(index.view.c.ToString(CultureInfo.InvariantCulture));
                    }
                    continue;
                }

                sb.Append("missing");
            }

            return sb.ToString();
        }

        public string FormatLastLayerRuntimeProfile(int topN = 40)
        {
            return FormatLayerRuntimeProfile(LastRuntimeProfile, topN);
        }

        public static string FormatLayerRuntimeProfile(LayerRuntimeProfile profile, int topN = 40)
        {
            if (profile == null)
                return string.Empty;

            topN = Mathf.Max(1, topN);
            var sb = new StringBuilder(4096);
            sb.AppendLine("section\tinference\tpath_kind\tsync_gpu\tlayer_index\tname\ttype\tpath\tcount\tms\tavg_ms");
            sb.Append("summary\t");
            sb.Append(profile.inferenceIndex.ToString(CultureInfo.InvariantCulture));
            sb.Append('\t');
            sb.Append(profile.pathKind ?? string.Empty);
            sb.Append('\t');
            sb.Append(profile.syncGpu ? "1" : "0");
            sb.Append("\t\t\t\t");
            sb.Append(profile.layers.Count.ToString(CultureInfo.InvariantCulture));
            sb.Append('\t');
            sb.Append(profile.totalMs.ToString("0.###", CultureInfo.InvariantCulture));
            sb.AppendLine("\t");

            var typeProfiles = new List<LayerRuntimeTypeProfile>(profile.layerTypes.Values);
            typeProfiles.Sort((a, b) => b.totalTicks.CompareTo(a.totalTicks));
            for (var i = 0; i < typeProfiles.Count; i++)
            {
                var item = typeProfiles[i];
                sb.Append("type\t");
                sb.Append(profile.inferenceIndex.ToString(CultureInfo.InvariantCulture));
                sb.Append('\t');
                sb.Append(profile.pathKind ?? string.Empty);
                sb.Append('\t');
                sb.Append(profile.syncGpu ? "1" : "0");
                sb.Append("\t\t\t");
                sb.Append(item.layerType ?? string.Empty);
                sb.Append("\t\t");
                sb.Append(item.count.ToString(CultureInfo.InvariantCulture));
                sb.Append('\t');
                sb.Append(item.totalMs.ToString("0.###", CultureInfo.InvariantCulture));
                sb.Append('\t');
                sb.AppendLine(item.avgMs.ToString("0.###", CultureInfo.InvariantCulture));
            }

            var records = new List<LayerRuntimeRecord>(profile.layers);
            records.Sort((a, b) => b.elapsedTicks.CompareTo(a.elapsedTicks));
            var count = Mathf.Min(topN, records.Count);
            for (var i = 0; i < count; i++)
            {
                var item = records[i];
                sb.Append("layer\t");
                sb.Append(profile.inferenceIndex.ToString(CultureInfo.InvariantCulture));
                sb.Append('\t');
                sb.Append(profile.pathKind ?? string.Empty);
                sb.Append('\t');
                sb.Append(profile.syncGpu ? "1" : "0");
                sb.Append('\t');
                sb.Append(item.layerIndex.ToString(CultureInfo.InvariantCulture));
                sb.Append('\t');
                sb.Append(item.layerName ?? string.Empty);
                sb.Append('\t');
                sb.Append(item.layerType ?? string.Empty);
                sb.Append('\t');
                sb.Append(item.path ?? string.Empty);
                sb.Append("\t1\t");
                sb.Append(item.elapsedMs.ToString("0.###", CultureInfo.InvariantCulture));
                sb.Append('\t');
                sb.AppendLine(item.elapsedMs.ToString("0.###", CultureInfo.InvariantCulture));
            }

            return sb.ToString();
        }

        private static double TicksToMilliseconds(long ticks)
        {
            return ticks <= 0 ? 0d : (double)ticks * 1000d / Stopwatch.Frequency;
        }

        public sealed class LayerRuntimeRecord
        {
            public int layerIndex;
            public string layerName;
            public string layerType;
            public string path;
            public long elapsedTicks;
            public double elapsedMs;
        }

        public sealed class LayerRuntimeTypeProfile
        {
            public string layerType;
            public int count;
            public long totalTicks;
            public double totalMs;
            public double avgMs;
        }

        public sealed class LayerRuntimeProfile
        {
            public int inferenceIndex;
            public string pathKind;
            public bool syncGpu;
            public long totalTicks;
            public double totalMs;
            public readonly List<LayerRuntimeRecord> layers = new List<LayerRuntimeRecord>();
            public readonly Dictionary<string, LayerRuntimeTypeProfile> layerTypes = new Dictionary<string, LayerRuntimeTypeProfile>(StringComparer.Ordinal);
        }

        public sealed class LayerTypeLoadProfile
        {
            public int count;
            public long totalMs;
            public long bytesRead;
            public long readMs;
            public long uploadMs;
            public long packMs;
        }

        public sealed class ModelLoadProfile
        {
            public string modelMagic;
            public int layerCount;
            public long releaseMs;
            public long parseParamMs;
            public long buildBlobUseCountMs;
            public long totalMs;
            public long totalBytesRead;
            public readonly Dictionary<string, LayerTypeLoadProfile> layerTypes = new Dictionary<string, LayerTypeLoadProfile>(StringComparer.Ordinal);
        }

        public readonly struct LoadProgress
        {
            public readonly string stage;
            public readonly int layerIndex;
            public readonly int layerCount;
            public readonly string layerName;
            public readonly string layerType;
            public readonly float progress01;

            public LoadProgress(string stage, int layerIndex, int layerCount, string layerName, string layerType, float progress01)
            {
                this.stage = stage;
                this.layerIndex = layerIndex;
                this.layerCount = layerCount;
                this.layerName = layerName;
                this.layerType = layerType;
                this.progress01 = progress01;
            }
        }

        public readonly struct LayerLoadMetrics
        {
            public readonly long bytesRead;
            public readonly long readMs;
            public readonly long uploadMs;
            public readonly long packMs;

            public LayerLoadMetrics(long bytesRead, long readMs, long uploadMs, long packMs)
            {
                this.bytesRead = bytesRead;
                this.readMs = readMs;
                this.uploadMs = uploadMs;
                this.packMs = packMs;
            }
        }

        internal static void SetTextureBlob(
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            string name,
            RenderTexture texture,
            BufferShape logicalShape)
        {
            textureBlobs[name] = new TensorRef
            {
                texture = texture,
                width = texture.width,
                height = texture.height,
                packs = texture.volumeDepth > 0 ? texture.volumeDepth : 1,
                refs = 1,
                owned = true
            };
            textureShapes[name] = logicalShape;
        }

        internal void PublishTensorBufferOutput(
            string topName,
            NcnnTensorBuffer tensor,
            bool preferTexture,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, BufferRef> bufferRefs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            List<IDisposable> tempOwned)
        {
            if (string.IsNullOrEmpty(topName))
                throw new ArgumentNullException(nameof(topName));
            if (tensor == null || tensor.buffer == null)
                throw new ArgumentNullException(nameof(tensor));

            var logicalShape = new BufferShape(tensor.dims, tensor.w, tensor.h, tensor.d, tensor.c);
            bufferBlobs[topName] = tensor.buffer;
            bufferRefs[topName] = NewBufferRef(tensor.buffer, tensor.ownsBuffer);
            bufferViews[topName] = new NcnnTensorBuffer(tensor.buffer, tensor.dims, tensor.w, tensor.h, tensor.d, tensor.c, false);

            if (preferTexture && tensor.dims <= 3)
            {
                var rt = MaterializeTextureFromBufferView(tensor.buffer, tensor);
                if (rt != null)
                    SetTextureBlob(textureBlobs, textureShapes, topName, rt, logicalShape);
            }
        }

        internal void PublishCmdTensorBufferOutput(
            CommandBuffer cmd,
            string topName,
            NcnnTensorBuffer tensor,
            bool preferTexture,
            Dictionary<string, CmdTensorRef> blobs,
            Dictionary<string, BufferShape> shapes = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (string.IsNullOrEmpty(topName))
                throw new ArgumentNullException(nameof(topName));
            if (tensor == null || tensor.buffer == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!preferTexture || tensor.dims > 3)
                throw new InvalidOperationException("CommandBuffer outputs currently require dims<=3 materialized texture: " + topName);

            var rt = MaterializeCmdTextureFromBufferView(cmd, tensor.buffer, tensor);
            if (rt == null)
                throw new InvalidOperationException("Failed to materialize CommandBuffer tensor: " + topName);

            blobs[topName] = new CmdTensorRef
            {
                texture = rt,
                width = rt.width,
                height = rt.height,
                packs = Mathf.Max(1, Mathf.CeilToInt(tensor.c / 4f)),
                refs = 1,
                owned = true
            };
            if (shapes != null)
                shapes[topName] = new BufferShape(tensor.dims, tensor.w, tensor.h, tensor.d, tensor.c);
        }

        internal void PublishCmdTensorLikeInput(
            CommandBuffer cmd,
            string topName,
            int width,
            int height,
            int packs,
            Dictionary<string, CmdTensorRef> blobs,
            Dictionary<string, BufferShape> shapes = null,
            BufferShape? logicalShape = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (string.IsNullOrEmpty(topName))
                throw new ArgumentNullException(nameof(topName));
            if (blobs == null)
                throw new ArgumentNullException(nameof(blobs));

            var outArr = RentTempArray(cmd, width, height, packs, RenderTextureFormat.ARGBHalf);
            blobs[topName] = new CmdTensorRef
            {
                texture = outArr,
                width = width,
                height = height,
                packs = packs,
                refs = 1,
                owned = true
            };
            if (shapes != null)
                shapes[topName] = logicalShape ?? new BufferShape(3, Mathf.Max(1, width), Mathf.Max(1, height), 1, Mathf.Max(1, packs * 4));
        }

        internal static void ResolveCmdTextureLayout(NcnnTensorBuffer tensor, out int width, out int height, out int packs)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            width = Mathf.Max(1, tensor.w);
            height = 1;
            packs = 1;

            if (tensor.dims == 2)
            {
                height = Mathf.Max(1, tensor.h);
                return;
            }

            if (tensor.dims == 3)
            {
                height = Mathf.Max(1, tensor.h);
                packs = Mathf.Max(1, Mathf.CeilToInt(tensor.c / 4f));
                return;
            }

            if (tensor.dims >= 4)
            {
                height = Mathf.Max(1, tensor.h * Mathf.Max(1, tensor.d));
                packs = Mathf.Max(1, Mathf.CeilToInt(tensor.c / 4f));
            }
        }

        internal static void ResolveCmdTextureLayout(BufferShape shape, out int width, out int height, out int packs)
        {
            width = Mathf.Max(1, shape.w);
            height = 1;
            packs = 1;

            if (shape.dims == 2)
            {
                height = Mathf.Max(1, shape.h);
                return;
            }

            if (shape.dims == 3)
            {
                height = Mathf.Max(1, shape.h);
                packs = Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f));
                return;
            }

            if (shape.dims >= 4)
            {
                height = Mathf.Max(1, shape.h * Mathf.Max(1, shape.d));
                packs = Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f));
            }
        }

        internal void PublishCmdPlaceholderFromTensorView(
            CommandBuffer cmd,
            string topName,
            NcnnTensorBuffer tensor,
            Dictionary<string, CmdTensorRef> blobs,
            Dictionary<string, BufferShape> shapes = null)
        {
            ResolveCmdTextureLayout(tensor, out var width, out var height, out var packs);
            PublishCmdTensorLikeInput(cmd, topName, width, height, packs, blobs, shapes, new BufferShape(tensor.dims, tensor.w, tensor.h, tensor.d, tensor.c));
        }

        internal void PublishCmdPlaceholder(
            CommandBuffer cmd,
            string topName,
            BufferShape shape,
            Dictionary<string, CmdTensorRef> blobs,
            Dictionary<string, BufferShape> shapes = null)
        {
            ResolveCmdTextureLayout(shape, out var width, out var height, out var packs);
            PublishCmdTensorLikeInput(cmd, topName, width, height, packs, blobs, shapes, shape);
        }

        internal void CopyCmdTensor(
            CommandBuffer cmd,
            CmdTensorRef src,
            string topName,
            Dictionary<string, CmdTensorRef> blobs,
            int width = -1,
            int height = -1,
            int packs = -1,
            Dictionary<string, BufferShape> shapes = null,
            BufferShape? logicalShape = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (blobs == null)
                throw new ArgumentNullException(nameof(blobs));

            var outWidth = width > 0 ? width : src.width;
            var outHeight = height > 0 ? height : src.height;
            var outPacks = packs > 0 ? packs : src.packs;
            var outArr = RentTempArray(cmd, outWidth, outHeight, outPacks, RenderTextureFormat.ARGBHalf);
            if (outWidth == src.width && outHeight == src.height)
            {
                Ops.CopyPack4(cmd, src.texture, 0, outArr, 0, Mathf.Min(src.packs, outPacks));
            }

            blobs[topName] = new CmdTensorRef
            {
                texture = outArr,
                width = outWidth,
                height = outHeight,
                packs = outPacks,
                refs = 1,
                owned = true
            };
            if (shapes != null)
                shapes[topName] = logicalShape ?? new BufferShape(3, Mathf.Max(1, outWidth), Mathf.Max(1, outHeight), 1, Mathf.Max(1, outPacks * 4));
        }

        internal static bool CanUseExactPack4BinaryPath(TensorRef a, BufferShape aShape, TensorRef b, BufferShape bShape)
        {
            return a != null
                && b != null
                && a.texture != null
                && b.texture != null
                && aShape.dims == 3
                && bShape.dims == 3
                && aShape.w == bShape.w
                && aShape.h == bShape.h
                && aShape.c == bShape.c
                && a.width == b.width
                && a.height == b.height
                && a.packs == b.packs;
        }

        internal static bool CanUseGroupNormPack4Path(TensorRef src, BufferShape shape, GroupNormPack gp)
        {
            return src != null
                && src.texture != null
                && gp != null
                && gp.affine
                && gp.gamma != null
                && gp.beta != null
                && shape.dims == 3
                && shape.w == src.width
                && shape.h == src.height
                && shape.c == gp.channels
                && gp.channels > 0
                && gp.group > 0
                && gp.channels % gp.group == 0
                && src.packs == Mathf.CeilToInt(gp.channels / 4f);
        }

        internal static int ComputeConvOut(int inSize, int kernel, int dilation, int stride, int padBefore, int padAfter)
        {
            var kernelExtent = dilation * (kernel - 1) + 1;
            return Mathf.Max(1, (inSize + padBefore + padAfter - kernelExtent) / Mathf.Max(1, stride) + 1);
        }

        internal static int ComputeDeconvOut(int inSize, int kernel, int dilation, int stride, int padBefore, int padAfter, int outputPadAfter)
        {
            var kernelExtent = dilation * (kernel - 1) + 1;
            var bordered = (inSize - 1) * Mathf.Max(1, stride) + kernelExtent + Mathf.Max(0, outputPadAfter);
            return Mathf.Max(1, bordered - Mathf.Max(0, padBefore) - Mathf.Max(0, padAfter));
        }

        internal static ComputeBuffer NewBuffer(float[] data)
        {
            var buf = new ComputeBuffer(data.Length, sizeof(float), ComputeBufferType.Structured);
            buf.SetData(data);
            return buf;
        }

        internal BufferRef NewBufferRef(ComputeBuffer buffer, bool owned)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            return new BufferRef
            {
                buffer = buffer,
                refs = 1,
                owned = owned
            };
        }

        internal BufferRef NewOwnedBufferRef(string name, ComputeBuffer buffer)
        {
            return NewBufferRef(buffer, true);
        }

        internal static TensorRef GetTexture(Dictionary<string, TensorRef> blobs, string name)
        {
            if (!blobs.TryGetValue(name, out var tr) || tr == null || tr.texture == null)
                throw new InvalidOperationException("blob not found: " + name);
            return tr;
        }

        internal static NcnnTensorBuffer TryGetBufferView(
            string name,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews)
        {
            if (bufferViews.TryGetValue(name, out var view) && view != null && view.buffer != null)
                return view;
            if (bufferBlobs.TryGetValue(name, out var buf) && buf != null)
            {
                view = new NcnnTensorBuffer(buf, 1, buf.count, 1, 1, 1, false);
                bufferViews[name] = view;
                return view;
            }
            return null;
        }

        internal ComputeBuffer GetOrConvertToBuffer(
            string name,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            List<IDisposable> tempOwned)
        {
            if (bufferBlobs.TryGetValue(name, out var buf) && buf != null)
                return buf;
            if (!textureBlobs.TryGetValue(name, out var tr) || tr == null || tr.texture == null)
                return null;

            var shape = GetTextureShape(textureShapes, tr, name);
            var physicalChannels = tr.packs * 4;
            var physicalCount = tr.width * tr.height * physicalChannels;
            var logicalCount = shape.w * shape.h * shape.d * shape.c;
            if (physicalCount == logicalCount)
            {
                var convertedExact = RentTempBuffer(logicalCount, sizeof(float));
                _ops.Pack4ToBufferCHW(tr.texture, tr.width, tr.height, physicalChannels, convertedExact);
                bufferBlobs[name] = convertedExact;
                bufferViews[name] = new NcnnTensorBuffer(convertedExact, shape.dims, shape.w, shape.h, shape.d, shape.c, false);
                tempOwned.Add(convertedExact);
                return convertedExact;
            }

            if (logicalCount > 0 && logicalCount < physicalCount)
            {
                var physicalBuffer = RentTempBuffer(physicalCount, sizeof(float));
                _ops.Pack4ToBufferCHW(tr.texture, tr.width, tr.height, physicalChannels, physicalBuffer);
                tempOwned.Add(physicalBuffer);

                var converted = RentTempBuffer(logicalCount, sizeof(float));
                _ops.CopyBufPartial(physicalBuffer, 0, converted, logicalCount);

                bufferBlobs[name] = converted;
                bufferViews[name] = new NcnnTensorBuffer(converted, shape.dims, shape.w, shape.h, shape.d, shape.c, false);
                tempOwned.Add(converted);
                return converted;
            }

            throw new InvalidOperationException("texture logical shape mismatch: " + name + " | physical=" + physicalCount + " logical=" + logicalCount);
        }

        internal static float[] ReadClipArrayAsFloat32(NcnnBinReader br, int count, int loadType)
        {
            if (br == null)
                throw new ArgumentNullException(nameof(br));
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count));
            if (count == 0)
                return Array.Empty<float>();
            return br.ReadNcnnMatAsFloat32(count, 0, 0, 0, loadType);
        }

        internal static float[] ReadClipMatAsFloat32(NcnnBinReader br, int w, int h, int d, int c, int loadType)
        {
            int count;
            if (d != 0) count = checked(w * h * d * c);
            else if (c != 0) count = checked(w * h * c);
            else if (h != 0) count = checked(w * h);
            else if (w != 0) count = w;
            else count = 1;
            return ReadClipArrayAsFloat32(br, count, loadType);
        }

        internal static float[] ReadPackedOrRawWeightArray(NcnnBinReader br, int count, string layerName)
        {
            if (br == null)
                throw new ArgumentNullException(nameof(br));
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count));
            if (count == 0)
                return Array.Empty<float>();

            try
            {
                return ReadClipArrayAsFloat32(br, count, 0);
            }
            catch (Exception e)
            {
                throw new InvalidOperationException("failed to read weights for " + layerName + " at " + br.Position + ": " + e.Message, e);
            }
        }

        internal static float[] ReadFloatBuffer(ComputeBuffer buffer)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            var data = new float[buffer.count];
            buffer.GetData(data);
            return data;
        }

        internal static float[] RunGemmCpu(ComputeBuffer aBuf, NcnnTensorBuffer aView, GemmPack gp)
        {
            if (aBuf == null)
                throw new ArgumentNullException(nameof(aBuf));
            if (aView == null)
                throw new ArgumentNullException(nameof(aView));
            if (gp == null || gp.bDataCpu == null)
                throw new ArgumentNullException(nameof(gp));

            var m = aView.dims == 1 ? 1 : aView.h;
            var k = aView.w;
            var n = gp.constantN;
            var a = ReadFloatBuffer(aBuf);
            var b = gp.bDataCpu;
            var c = gp.cDataCpu;
            var output = new float[m * n];

            for (var row = 0; row < m; row++)
            {
                var aBase = row * k;
                for (var col = 0; col < n; col++)
                {
                    double sum = 0.0;
                    for (var kk = 0; kk < k; kk++)
                    {
                        var aValue = a[aBase + kk];
                        var bValue = gp.transB
                            ? b[col * k + kk]
                            : b[kk * n + col];
                        sum += (double)aValue * bValue;
                    }

                    var value = gp.alpha * (float)sum;
                    if (gp.constantC && c != null && gp.broadcastTypeC != -1)
                    {
                        float cValue;
                        switch (gp.broadcastTypeC)
                        {
                            case 0: cValue = c[0]; break;
                            case 1: cValue = c[row]; break;
                            case 2: cValue = c[row]; break;
                            case 3: cValue = c[row * n + col]; break;
                            case 4: cValue = c[col]; break;
                            default: cValue = 0f; break;
                        }
                        value += gp.beta * cValue;
                    }

                    output[row * n + col] = value;
                }
            }

            return output;
        }

        internal NcnnTensorBuffer RunMatMulLayer(ComputeBuffer aBuf, NcnnTensorBuffer aView, ComputeBuffer bBuf, NcnnTensorBuffer bView, bool transB)
        {
            static void GetMatrixShape(NcnnTensorBuffer view, out int rows, out int cols)
            {
                if (view == null)
                    throw new ArgumentNullException(nameof(view));
                if (view.dims == 1)
                {
                    rows = 1;
                    cols = view.w;
                    return;
                }
                if (view.dims == 2 || view.dims == 3)
                {
                    rows = view.h;
                    cols = view.w;
                    return;
                }
                throw new InvalidOperationException("MatMul currently supports dims 1/2/3 only");
            }

            GetMatrixShape(aView, out var aRows, out var aCols);
            GetMatrixShape(bView, out var bRows, out var bCols);

            var k = aCols;
            var kFromB = transB ? bCols : bRows;
            var n = transB ? bRows : bCols;
            if (k != kFromB)
                throw new InvalidOperationException("MatMul K mismatch: " + k + " vs " + kFromB);

            var batchA = aView.dims == 3 ? aView.c : 1;
            var batchB = bView.dims == 3 ? bView.c : 1;
            var batch = Mathf.Max(batchA, batchB);
            if (batchA != 1 && batchA != batch)
                throw new InvalidOperationException("MatMul batchA mismatch: " + batchA + " vs " + batch);
            if (batchB != 1 && batchB != batch)
                throw new InvalidOperationException("MatMul batchB mismatch: " + batchB + " vs " + batch);

            var aCount = aRows * aCols;
            var bCount = bRows * bCols;
            var outCount = aRows * n;

            if (batch == 1)
            {
                var outTensor2D = RentTempTensorBuffer(2, n, aRows);
                Ops.MatMul2D(aBuf, bBuf, aRows, n, k, transB, outTensor2D.buffer);
                return outTensor2D;
            }

            var outTensor = RentTempTensorBuffer(3, n, aRows, 1, batch);
            var tempA = batchA == 1 ? null : RentTempBuffer(aCount, sizeof(float));
            var tempB = batchB == 1 ? null : RentTempBuffer(bCount, sizeof(float));
            var tempOut = RentTempBuffer(outCount, sizeof(float));
            try
            {
                for (var p = 0; p < batch; p++)
                {
                    var aSrc = aBuf;
                    var bSrc = bBuf;
                    if (batchA != 1)
                    {
                        Ops.CopyBufPartial(aBuf, p * aCount, tempA, aCount);
                        aSrc = tempA;
                    }

                    if (batchB != 1)
                    {
                        Ops.CopyBufPartial(bBuf, p * bCount, tempB, bCount);
                        bSrc = tempB;
                    }

                    Ops.MatMul2D(aSrc, bSrc, aRows, n, k, transB, tempOut);
                    Ops.CopyBufPartial(tempOut, 0, outTensor.buffer, outCount, p * outCount);
                }
            }
            finally
            {
                if (tempA != null)
                    ReturnTempBuffer(tempA);
                if (tempB != null)
                    ReturnTempBuffer(tempB);
                ReturnTempBuffer(tempOut);
            }

            return outTensor;
        }

        internal void Consume(
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, int> remaining,
            string[] bottomNames,
            ICollection<string> pinnedNames)
        {
            for (var i = 0; i < bottomNames.Length; i++)
            {
                var name = bottomNames[i];
                if (!remaining.TryGetValue(name, out var count))
                    continue;

                count--;
                remaining[name] = count;
                if (count > 0)
                    continue;
                if (pinnedNames != null && pinnedNames.Contains(name))
                    continue;

                if (textureBlobs.TryGetValue(name, out var tr) && tr != null)
                {
                    tr.refs--;
                    if (tr.refs <= 0 && tr.owned && tr.texture != null)
                    {
                        try { ReturnTempArray(tr.texture); } catch { }
                        tr.texture = null;
                    }
                }
                textureBlobs.Remove(name);
            }
        }

        internal void Consume(
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, BufferRef> bufferRefs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            Dictionary<string, int> remaining,
            string[] bottomNames,
            ICollection<string> pinnedNames)
        {
            for (var i = 0; i < bottomNames.Length; i++)
            {
                var name = bottomNames[i];
                if (!remaining.TryGetValue(name, out var count))
                    continue;

                count--;
                remaining[name] = count;
                if (count > 0)
                    continue;
                if (pinnedNames != null && pinnedNames.Contains(name))
                    continue;

                if (textureBlobs.TryGetValue(name, out var tr) && tr != null)
                {
                    tr.refs--;
                    if (tr.refs <= 0 && tr.owned && tr.texture != null)
                    {
                        try { ReturnTempArray(tr.texture); } catch { }
                        tr.texture = null;
                    }
                }

                if (bufferRefs != null && bufferRefs.TryGetValue(name, out var br) && br != null)
                {
                    br.refs--;
                    if (br.refs <= 0 && br.owned && br.buffer != null)
                    {
                        try { ReturnTempBuffer(br.buffer); } catch { }
                        br.buffer = null;
                    }
                    bufferRefs.Remove(name);
                }

                textureBlobs.Remove(name);
                bufferViews?.Remove(name);
                bufferBlobs?.Remove(name);
            }
        }

        internal void ConsumeIndex(
            Dictionary<string, IndexRef> indexBlobs,
            Dictionary<string, int> remaining,
            string[] bottomNames,
            ICollection<string> pinnedNames)
        {
            if (indexBlobs == null || bottomNames == null)
                return;

            for (var i = 0; i < bottomNames.Length; i++)
            {
                var name = bottomNames[i];
                if (!remaining.TryGetValue(name, out var count))
                    continue;

                count--;
                remaining[name] = count;
                if (count > 0)
                    continue;
                if (pinnedNames != null && pinnedNames.Contains(name))
                    continue;

                if (indexBlobs.TryGetValue(name, out var ir) && ir != null)
                {
                    ir.refs--;
                    if (ir.refs <= 0 && ir.owned)
                    {
                        if (ir.texture != null)
                        {
                            try { ReturnTempArray(ir.texture); } catch { }
                            ir.texture = null;
                        }
                        if (ir.buffer != null)
                        {
                            try { ReturnTempBuffer(ir.buffer); } catch { }
                            ir.buffer = null;
                        }
                        ir.view = null;
                    }
                }
                indexBlobs.Remove(name);
            }
        }

        internal static NcnnTensorBuffer ResolveReshapeTensor(NcnnTensorBuffer src, NcnnParamModel.Layer layer)
        {
            return ResolveReshapeTensor(src, layer, null);
        }

        internal static NcnnTensorBuffer ResolveReshapeTensor(NcnnTensorBuffer src, NcnnParamModel.Layer layer, IReadOnlyList<BufferShape> bottomShapes)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));

            if (!string.IsNullOrWhiteSpace(layer.GetString(6, null)))
            {
                var exprShape = EvaluateReshapeShapeExpression(layer.GetString(6, null), bottomShapes ?? new[] { new BufferShape(src.dims, src.w, src.h, src.d, src.c) }, layer);
                return src.Reshape(exprShape.dims, exprShape.w, exprShape.h, exprShape.d, exprShape.c);
            }

            var outw = layer.GetInt(0, -233);
            var outh = layer.GetInt(1, -233);
            var outd = layer.GetInt(11, -233);
            var outc = layer.GetInt(2, -233);
            var ndim = 4;
            if (outd == -233) ndim = 3;
            if (outc == -233) ndim = 2;
            if (outh == -233) ndim = 1;

            var total = src.elementCount;

            static int SafeDiv(int a, int b, string reason)
            {
                if (b == 0 || (a % b) != 0)
                    throw new InvalidOperationException(reason + " | " + a + " / " + b);
                return a / b;
            }

            if (ndim == 1)
            {
                if (outw == 0) outw = src.w;
                if (outw == -1) outw = total;
                return src.Reshape(1, outw);
            }

            if (ndim == 2)
            {
                if (outw == 0) outw = src.w;
                if (outh == 0) outh = src.h;
                if (outw == -1) outw = SafeDiv(total, outh, "Reshape outw");
                if (outh == -1) outh = SafeDiv(total, outw, "Reshape outh");
                return src.Reshape(2, outw, outh);
            }

            if (ndim == 3)
            {
                if (outw == 0) outw = src.w;
                if (outh == 0) outh = src.h;
                if (outc == 0) outc = src.c;
                if (outw == -1) outw = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outh), "Reshape outw");
                if (outh == -1) outh = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outw), "Reshape outh");
                if (outc == -1) outc = SafeDiv(total, Mathf.Max(1, outw) * Mathf.Max(1, outh), "Reshape outc");
                return src.Reshape(3, outw, outh, 1, outc);
            }

            if (outw == 0) outw = src.w;
            if (outh == 0) outh = src.h;
            if (outd == 0) outd = src.d;
            if (outc == 0) outc = src.c;
            if (outw == -1) outw = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outd) * Mathf.Max(1, outh), "Reshape outw");
            if (outh == -1) outh = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outd) * Mathf.Max(1, outw), "Reshape outh");
            if (outd == -1) outd = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outw) * Mathf.Max(1, outh), "Reshape outd");
            if (outc == -1) outc = SafeDiv(total, Mathf.Max(1, outd) * Mathf.Max(1, outw) * Mathf.Max(1, outh), "Reshape outc");
            return src.Reshape(4, outw, outh, outd, outc);
        }

        internal static BufferShape ResolveReshapeShape(BufferShape src, NcnnParamModel.Layer layer)
        {
            return ResolveReshapeShape(src, layer, null);
        }

        internal static BufferShape ResolveReshapeShape(BufferShape src, NcnnParamModel.Layer layer, IReadOnlyList<BufferShape> bottomShapes)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));

            if (!string.IsNullOrWhiteSpace(layer.GetString(6, null)))
                return EvaluateReshapeShapeExpression(layer.GetString(6, null), bottomShapes ?? new[] { src }, layer);

            var outw = layer.GetInt(0, -233);
            var outh = layer.GetInt(1, -233);
            var outd = layer.GetInt(11, -233);
            var outc = layer.GetInt(2, -233);
            var ndim = 4;
            if (outd == -233) ndim = 3;
            if (outc == -233) ndim = 2;
            if (outh == -233) ndim = 1;

            var total = src.w * src.h * src.d * src.c;

            static int SafeDiv(int a, int b, string reason)
            {
                if (b == 0 || (a % b) != 0)
                    throw new InvalidOperationException(reason + " | " + a + " / " + b);
                return a / b;
            }

            if (ndim == 1)
            {
                if (outw == 0) outw = src.w;
                if (outw == -1) outw = total;
                return new BufferShape(1, outw, 1, 1, 1);
            }

            if (ndim == 2)
            {
                if (outw == 0) outw = src.w;
                if (outh == 0) outh = src.h;
                if (outw == -1) outw = SafeDiv(total, outh, "Reshape outw");
                if (outh == -1) outh = SafeDiv(total, outw, "Reshape outh");
                return new BufferShape(2, outw, outh, 1, 1);
            }

            if (ndim == 3)
            {
                if (outw == 0) outw = src.w;
                if (outh == 0) outh = src.h;
                if (outc == 0) outc = src.c;
                if (outw == -1) outw = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outh), "Reshape outw");
                if (outh == -1) outh = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outw), "Reshape outh");
                if (outc == -1) outc = SafeDiv(total, Mathf.Max(1, outw) * Mathf.Max(1, outh), "Reshape outc");
                return new BufferShape(3, outw, outh, 1, outc);
            }

            if (outw == 0) outw = src.w;
            if (outh == 0) outh = src.h;
            if (outd == 0) outd = src.d;
            if (outc == 0) outc = src.c;
            if (outw == -1) outw = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outd) * Mathf.Max(1, outh), "Reshape outw");
            if (outh == -1) outh = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outd) * Mathf.Max(1, outw), "Reshape outh");
            if (outd == -1) outd = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outw) * Mathf.Max(1, outh), "Reshape outd");
            if (outc == -1) outc = SafeDiv(total, Mathf.Max(1, outd) * Mathf.Max(1, outw) * Mathf.Max(1, outh), "Reshape outc");
            return new BufferShape(4, outw, outh, outd, outc);
        }

        internal static BufferShape EvaluateReshapeShapeExpression(string expr, NcnnTensorBuffer src, NcnnParamModel.Layer layer)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            return EvaluateReshapeShapeExpression(expr, new[] { new BufferShape(src.dims, src.w, src.h, src.d, src.c) }, layer);
        }

        internal static BufferShape EvaluateReshapeShapeExpression(string expr, BufferShape src, NcnnParamModel.Layer layer)
        {
            return EvaluateReshapeShapeExpression(expr, new[] { src }, layer);
        }

        internal static BufferShape EvaluateReshapeShapeExpression(string expr, IReadOnlyList<BufferShape> bottomShapes, NcnnParamModel.Layer layer)
        {
            if (string.IsNullOrWhiteSpace(expr))
                throw new ArgumentException("shape expr is empty", nameof(expr));
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));
            if (bottomShapes == null || bottomShapes.Count == 0)
                throw new ArgumentException("bottomShapes is empty", nameof(bottomShapes));

            var values = EvaluateExpressionList(expr, bottomShapes, layer);
            if (values.Count <= 0 || values.Count > 4)
                throw new InvalidOperationException("Unsupported reshape shape_expr rank: " + values.Count + " | " + layer.name);

            if (values.Count == 1)
                return new BufferShape(1, values[0], 1, 1, 1);
            if (values.Count == 2)
                return new BufferShape(2, values[0], values[1], 1, 1);
            if (values.Count == 3)
                return new BufferShape(3, values[0], values[1], 1, values[2]);
            return new BufferShape(4, values[0], values[1], values[2], values[3]);
        }

        internal static IReadOnlyList<int> EvaluateExpressionList(string expr, IReadOnlyList<BufferShape> bottomShapes, NcnnParamModel.Layer layer)
        {
            if (string.IsNullOrWhiteSpace(expr))
                throw new ArgumentException("expression is empty", nameof(expr));
            if (bottomShapes == null || bottomShapes.Count == 0)
                throw new ArgumentException("bottomShapes is empty", nameof(bottomShapes));

            var trimmed = expr.Trim();
            if (trimmed.Length >= 2 && trimmed[0] == '"' && trimmed[trimmed.Length - 1] == '"')
                trimmed = trimmed.Substring(1, trimmed.Length - 2);

            var tokens = new List<string>();
            var token = new StringBuilder();
            for (var i = 0; i < trimmed.Length; i++)
            {
                var ch = trimmed[i];
                if (ch == '(' || ch == ')' || ch == ',')
                {
                    if (token.Length > 0)
                    {
                        tokens.Add(token.ToString());
                        token.Clear();
                    }
                }
                else if (!char.IsWhiteSpace(ch))
                {
                    token.Append(ch);
                }
            }
            if (token.Length > 0)
                tokens.Add(token.ToString());

            var stack = new Stack<ExprValue>();
            for (var i = tokens.Count - 1; i >= 0; i--)
            {
                var t = tokens[i];
                if (IsShapeRefToken(t))
                {
                    stack.Push(new ExprValue(GetShapeRefValue(t, bottomShapes, layer)));
                    continue;
                }

                if (t == "+" || t == "-" || t == "*" || t == "//" || t == "max" || t == "min")
                {
                    var a = PopExpr(stack, layer);
                    var b = PopExpr(stack, layer);
                    stack.Push(ApplyBinaryIntPref(t, a, b, layer));
                    continue;
                }

                if (t == "abs" || t == "neg" || t == "sign" || t == "square")
                {
                    var a = PopExpr(stack, layer);
                    stack.Push(ApplyUnarySimple(t, a));
                    continue;
                }

                if (t == "trunc" || t == "ceil" || t == "floor" || t == "round")
                {
                    var a = PopExpr(stack, layer);
                    stack.Push(ApplyUnaryRound(t, a));
                    continue;
                }

                if (t == "acos" || t == "acosh" || t == "asin" || t == "asinh" || t == "atan" || t == "atanh"
                    || t == "cos" || t == "cosh" || t == "erf" || t == "exp" || t == "log" || t == "log10"
                    || t == "reciprocal" || t == "rsqrt" || t == "sin" || t == "sinh" || t == "sqrt" || t == "tan" || t == "tanh")
                {
                    var a = PopExpr(stack, layer);
                    stack.Push(ApplyUnaryFloat(t, a));
                    continue;
                }

                if (t == "/" || t == "atan2" || t == "fmod" || t == "pow" || t == "remainder" || t == "logaddexp")
                {
                    var a = PopExpr(stack, layer);
                    var b = PopExpr(stack, layer);
                    stack.Push(ApplyBinaryFloat(t, a, b));
                    continue;
                }

                if (t == "and" || t == "or" || t == "xor" || t == "lshift" || t == "rshift")
                {
                    var a = PopExpr(stack, layer);
                    var b = PopExpr(stack, layer);
                    stack.Push(ApplyBinaryBitwise(t, a, b));
                    continue;
                }

                if (TryParseExprLiteral(t, out var literal))
                {
                    stack.Push(literal);
                    continue;
                }

                throw new InvalidOperationException("Malformed expression token " + t + " | " + layer?.name);
            }

            var values = new List<int>(stack.Count);
            while (stack.Count > 0)
                values.Add(stack.Pop().ToInt());
            return values;
        }

        internal static int[] EvaluateExpressionListOrNull(string expr, IReadOnlyList<BufferShape> bottomShapes, NcnnParamModel.Layer layer)
        {
            if (string.IsNullOrWhiteSpace(expr))
                return null;
            var values = EvaluateExpressionList(expr, bottomShapes, layer);
            if (values == null || values.Count == 0)
                return Array.Empty<int>();
            var arr = new int[values.Count];
            for (var i = 0; i < values.Count; i++)
                arr[i] = values[i];
            return arr;
        }

        private readonly struct ExprValue
        {
            public readonly bool isFloat;
            public readonly int i;
            public readonly float f;

            public ExprValue(int value)
            {
                isFloat = false;
                i = value;
                f = value;
            }

            public ExprValue(float value)
            {
                isFloat = true;
                i = (int)value;
                f = value;
            }

            public int ToInt() => isFloat ? (int)f : i;
            public float ToFloat() => isFloat ? f : i;
        }

        private static ExprValue PopExpr(Stack<ExprValue> stack, NcnnParamModel.Layer layer)
        {
            if (stack == null || stack.Count == 0)
                throw new InvalidOperationException("Malformed expression stack underflow: " + layer?.name);
            return stack.Pop();
        }

        private static bool IsShapeRefToken(string token)
        {
            return token != null
                && token.Length == 2
                && token[0] >= '0' && token[0] <= '9'
                && (token[1] == 'w' || token[1] == 'h' || token[1] == 'd' || token[1] == 'c');
        }

        private static int GetShapeRefValue(string token, IReadOnlyList<BufferShape> bottomShapes, NcnnParamModel.Layer layer)
        {
            var index = token[0] - '0';
            if (index < 0 || index >= bottomShapes.Count)
                throw new InvalidOperationException("shape expression blob index out of range: " + token + " | " + layer?.name);

            var shape = bottomShapes[index];
            return token[1] switch
            {
                'w' => shape.w,
                'h' => shape.h,
                'd' => shape.d,
                'c' => shape.c,
                _ => throw new InvalidOperationException("invalid shape ref token: " + token)
            };
        }

        private static bool TryParseExprLiteral(string token, out ExprValue value)
        {
            if (int.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out var i)
                && float.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out var fi)
                && Mathf.Approximately(i, fi))
            {
                value = new ExprValue(i);
                return true;
            }

            if (float.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out var f))
            {
                value = new ExprValue(f);
                return true;
            }

            value = default;
            return false;
        }

        private static ExprValue ApplyBinaryIntPref(string op, ExprValue a, ExprValue b, NcnnParamModel.Layer layer)
        {
            if (!a.isFloat && !b.isFloat)
            {
                return op switch
                {
                    "+" => new ExprValue(a.i + b.i),
                    "-" => new ExprValue(a.i - b.i),
                    "*" => new ExprValue(a.i * b.i),
                    "//" => b.i != 0 ? new ExprValue(a.i / b.i) : throw new InvalidOperationException("expr divide by zero | " + layer?.name),
                    "max" => new ExprValue(Mathf.Max(a.i, b.i)),
                    "min" => new ExprValue(Mathf.Min(a.i, b.i)),
                    _ => throw new InvalidOperationException("unsupported int-pref op: " + op)
                };
            }

            var af = a.ToFloat();
            var bf = b.ToFloat();
            return op switch
            {
                "+" => new ExprValue(af + bf),
                "-" => new ExprValue(af - bf),
                "*" => new ExprValue(af * bf),
                "//" => new ExprValue(Mathf.Floor(af / bf)),
                "max" => new ExprValue(Mathf.Max(af, bf)),
                "min" => new ExprValue(Mathf.Min(af, bf)),
                _ => throw new InvalidOperationException("unsupported float-pref op: " + op)
            };
        }

        private static ExprValue ApplyUnarySimple(string op, ExprValue a)
        {
            if (!a.isFloat)
            {
                return op switch
                {
                    "abs" => new ExprValue(Mathf.Abs(a.i)),
                    "neg" => new ExprValue(-a.i),
                    "sign" => new ExprValue(a.i > 0 ? 1 : (a.i == 0 ? 0 : -1)),
                    "square" => new ExprValue(a.i * a.i),
                    _ => throw new InvalidOperationException("unsupported unary op: " + op)
                };
            }

            var af = a.f;
            return op switch
            {
                "abs" => new ExprValue(Mathf.Abs(af)),
                "neg" => new ExprValue(-af),
                "sign" => new ExprValue(af > 0f ? 1f : (af == 0f ? 0f : -1f)),
                "square" => new ExprValue(af * af),
                _ => throw new InvalidOperationException("unsupported unary op: " + op)
            };
        }

        private static ExprValue ApplyUnaryRound(string op, ExprValue a)
        {
            if (!a.isFloat)
                return new ExprValue(a.i);

            return op switch
            {
                "trunc" => new ExprValue((int)a.f),
                "ceil" => new ExprValue((int)Math.Ceiling(a.f)),
                "floor" => new ExprValue((int)Math.Floor(a.f)),
                "round" => new ExprValue((int)Math.Round(a.f, MidpointRounding.AwayFromZero)),
                _ => throw new InvalidOperationException("unsupported round op: " + op)
            };
        }

        private static ExprValue ApplyUnaryFloat(string op, ExprValue a)
        {
            var af = a.ToFloat();
            return op switch
            {
                "acos" => new ExprValue(Mathf.Acos(af)),
                "acosh" => new ExprValue((float)Acosh(af)),
                "asin" => new ExprValue(Mathf.Asin(af)),
                "asinh" => new ExprValue((float)Asinh(af)),
                "atan" => new ExprValue(Mathf.Atan(af)),
                "atanh" => new ExprValue((float)Atanh(af)),
                "cos" => new ExprValue(Mathf.Cos(af)),
                "cosh" => new ExprValue((float)Math.Cosh(af)),
                "erf" => new ExprValue((float)Erf(af)),
                "exp" => new ExprValue(Mathf.Exp(af)),
                "log" => new ExprValue(Mathf.Log(af)),
                "log10" => new ExprValue(Mathf.Log10(af)),
                "reciprocal" => new ExprValue(1f / af),
                "rsqrt" => new ExprValue(1f / Mathf.Sqrt(af)),
                "sin" => new ExprValue(Mathf.Sin(af)),
                "sinh" => new ExprValue((float)Math.Sinh(af)),
                "sqrt" => new ExprValue(Mathf.Sqrt(af)),
                "tan" => new ExprValue(Mathf.Tan(af)),
                "tanh" => new ExprValue((float)Math.Tanh(af)),
                _ => throw new InvalidOperationException("unsupported float unary op: " + op)
            };
        }

        private static ExprValue ApplyBinaryFloat(string op, ExprValue a, ExprValue b)
        {
            var af = a.ToFloat();
            var bf = b.ToFloat();
            return op switch
            {
                "/" => new ExprValue(af / bf),
                "atan2" => new ExprValue(Mathf.Atan2(af, bf)),
                "fmod" => new ExprValue(af % bf),
                "pow" => new ExprValue(Mathf.Pow(af, bf)),
                "remainder" => new ExprValue(RepeatRemainder(af, bf)),
                "logaddexp" => new ExprValue(Mathf.Log(Mathf.Exp(af) + Mathf.Exp(bf))),
                _ => throw new InvalidOperationException("unsupported float binary op: " + op)
            };
        }

        private static ExprValue ApplyBinaryBitwise(string op, ExprValue a, ExprValue b)
        {
            var ai = a.ToInt();
            var bi = b.ToInt();
            return op switch
            {
                "and" => new ExprValue(ai & bi),
                "or" => new ExprValue(ai | bi),
                "xor" => new ExprValue(ai ^ bi),
                "lshift" => new ExprValue(ai << bi),
                "rshift" => new ExprValue(ai >> bi),
                _ => throw new InvalidOperationException("unsupported bitwise op: " + op)
            };
        }

        private static float RepeatRemainder(float a, float b)
        {
            var r = a % b;
            if (a * b < 0f)
                r += b;
            return r;
        }

        private static double Erf(double x)
        {
            var sign = x < 0 ? -1d : 1d;
            x = Math.Abs(x);

            var a1 = 0.254829592d;
            var a2 = -0.284496736d;
            var a3 = 1.421413741d;
            var a4 = -1.453152027d;
            var a5 = 1.061405429d;
            var p = 0.3275911d;
            var t = 1d / (1d + p * x);
            var y = 1d - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
            return sign * y;
        }

        private static double Acosh(double x)
        {
            return Math.Log(x + Math.Sqrt(x * x - 1d));
        }

        private static double Asinh(double x)
        {
            return Math.Log(x + Math.Sqrt(x * x + 1d));
        }

        private static double Atanh(double x)
        {
            return 0.5d * Math.Log((1d + x) / (1d - x));
        }

        internal static BufferShape GetTextureShape(Dictionary<string, BufferShape> textureShapes, TensorRef tr, string name)
        {
            if (textureShapes.TryGetValue(name, out var shape))
                return shape;
            return new BufferShape(3, tr.width, tr.height, 1, tr.packs * 4);
        }

        internal int ResolveInputLogicalChannels(string inputBlobName, int fallbackChannels)
        {
            if (Model?.layers == null || string.IsNullOrWhiteSpace(inputBlobName))
                return fallbackChannels;

            for (var i = 0; i < Model.layers.Count; i++)
            {
                var layer = Model.layers[i];
                if (layer?.bottomNames == null || layer.bottomNames.Length == 0)
                    continue;
                if (!string.Equals(layer.bottomNames[0], inputBlobName, StringComparison.Ordinal))
                    continue;
                if (layer.type != NcnnLayerTypes.Convolution
                    && layer.type != NcnnLayerTypes.ConvolutionDepthWise)
                    continue;
                if (_conv.TryGetValue(layer.name, out var conv) && conv != null && conv.inC > 0)
                    return conv.inC;
            }

            return fallbackChannels;
        }

        internal static Vector4Int ResolvePermuteAxes(int dims, int orderType, string layerName)
        {
            if (dims == 2)
            {
                return orderType switch
                {
                    0 => new Vector4Int(0, 1, 0, 0),
                    1 => new Vector4Int(1, 0, 0, 0),
                    _ => throw new InvalidOperationException("unsupported permute dims=2 orderType: " + orderType + " | " + layerName)
                };
            }

            if (dims == 3)
            {
                return orderType switch
                {
                    0 => new Vector4Int(0, 1, 2, 0),
                    1 => new Vector4Int(1, 0, 2, 0),
                    2 => new Vector4Int(0, 2, 1, 0),
                    3 => new Vector4Int(2, 0, 1, 0),
                    4 => new Vector4Int(1, 2, 0, 0),
                    5 => new Vector4Int(2, 1, 0, 0),
                    _ => throw new InvalidOperationException("unsupported permute dims=3 orderType: " + orderType + " | " + layerName)
                };
            }

            return orderType switch
            {
                0 => new Vector4Int(0, 1, 2, 3),
                1 => new Vector4Int(1, 0, 2, 3),
                2 => new Vector4Int(0, 2, 1, 3),
                3 => new Vector4Int(2, 0, 1, 3),
                4 => new Vector4Int(1, 2, 0, 3),
                5 => new Vector4Int(2, 1, 0, 3),
                6 => new Vector4Int(0, 1, 3, 2),
                7 => new Vector4Int(1, 0, 3, 2),
                8 => new Vector4Int(0, 3, 1, 2),
                9 => new Vector4Int(3, 0, 1, 2),
                10 => new Vector4Int(1, 3, 0, 2),
                11 => new Vector4Int(3, 1, 0, 2),
                12 => new Vector4Int(0, 2, 3, 1),
                13 => new Vector4Int(2, 0, 3, 1),
                14 => new Vector4Int(0, 3, 2, 1),
                15 => new Vector4Int(3, 0, 2, 1),
                16 => new Vector4Int(2, 3, 0, 1),
                17 => new Vector4Int(3, 2, 0, 1),
                18 => new Vector4Int(1, 2, 3, 0),
                19 => new Vector4Int(2, 1, 3, 0),
                20 => new Vector4Int(1, 3, 2, 0),
                21 => new Vector4Int(3, 1, 2, 0),
                22 => new Vector4Int(2, 3, 1, 0),
                23 => new Vector4Int(3, 2, 1, 0),
                _ => throw new InvalidOperationException("unsupported permute dims=4 orderType: " + orderType + " | " + layerName)
            };
        }

        internal static BufferShape ResolvePermuteShape(NcnnTensorBuffer src, int dims, Vector4Int axes)
        {
            int GetAxisSize(int axis)
            {
                if (axis == 0) return src.w;
                if (axis == 1) return src.h;
                if (axis == 2) return dims == 4 ? src.d : src.c;
                if (axis == 3) return src.c;
                throw new InvalidOperationException("invalid axis: " + axis);
            }

            var outW = GetAxisSize(axes.x);
            var outH = GetAxisSize(axes.y);
            var outD = dims == 4 ? GetAxisSize(axes.z) : 1;
            var outC = dims == 2 ? 1 : GetAxisSize(dims == 4 ? axes.w : axes.z);
            return new BufferShape(dims, outW, outH, outD, outC);
        }

        internal static int MapNcnnAxisToTensorAxis(int dims, int axis)
        {
            if (dims == 1)
                return 0;
            if (dims == 2)
                return axis == 0 ? 1 : 0;
            if (dims == 3)
            {
                if (axis == 0) return 2;
                if (axis == 1) return 1;
                return 0;
            }

            if (axis == 0) return 3;
            if (axis == 1) return 2;
            if (axis == 2) return 1;
            return 0;
        }

        internal static int GetAxisSize(int dims, int w, int h, int d, int c, int axis)
        {
            if (axis == 0) return w;
            if (axis == 1) return h;
            if (axis == 2) return dims == 4 ? d : c;
            if (axis == 3) return c;
            throw new ArgumentOutOfRangeException(nameof(axis));
        }

        internal readonly struct CropRoi
        {
            public readonly int woffset;
            public readonly int hoffset;
            public readonly int doffset;
            public readonly int coffset;
            public readonly int outw;
            public readonly int outh;
            public readonly int outd;
            public readonly int outc;

            public CropRoi(int woffset, int hoffset, int doffset, int coffset, int outw, int outh, int outd, int outc)
            {
                this.woffset = woffset;
                this.hoffset = hoffset;
                this.doffset = doffset;
                this.coffset = coffset;
                this.outw = outw;
                this.outh = outh;
                this.outd = outd;
                this.outc = outc;
            }
        }

        internal static BufferShape GetShapeOf(NcnnTensorBuffer view)
        {
            if (view == null)
                throw new ArgumentNullException(nameof(view));
            return new BufferShape(view.dims, view.w, view.h, view.d, view.c);
        }

        internal static BufferShape ResolveSqueezeShape(BufferShape src, NcnnParamModel.Layer layer)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));

            bool squeezeW = false;
            bool squeezeH = false;
            bool squeezeD = false;
            bool squeezeC = false;

            var axes = layer.GetInts(-23303, null);
            if (axes == null || axes.Length == 0)
                axes = layer.GetInts(3, null);

            if (axes == null || axes.Length == 0)
            {
                squeezeW = src.w == 1 && layer.GetInt(0, 0) != 0;
                squeezeH = src.h == 1 && layer.GetInt(1, 0) != 0;
                squeezeD = src.d == 1 && layer.GetInt(11, 0) != 0;
                squeezeC = src.c == 1 && layer.GetInt(2, 0) != 0;
            }
            else
            {
                for (var i = 0; i < axes.Length; i++)
                {
                    var axis = axes[i];
                    if (axis < 0)
                        axis += src.dims;

                    if (src.dims == 1 && axis == 0) squeezeW = src.w == 1;
                    if (src.dims == 2 && axis == 0) squeezeH = src.h == 1;
                    if (src.dims == 2 && axis == 1) squeezeW = src.w == 1;
                    if (src.dims == 3 && axis == 0) squeezeC = src.c == 1;
                    if (src.dims == 3 && axis == 1) squeezeH = src.h == 1;
                    if (src.dims == 3 && axis == 2) squeezeW = src.w == 1;
                    if (src.dims == 4 && axis == 0) squeezeC = src.c == 1;
                    if (src.dims == 4 && axis == 1) squeezeD = src.d == 1;
                    if (src.dims == 4 && axis == 2) squeezeH = src.h == 1;
                    if (src.dims == 4 && axis == 3) squeezeW = src.w == 1;
                }
            }

            if (src.dims == 1)
            {
                if (squeezeW)
                    return new BufferShape(1, 1, 1, 1, 1);
                return src;
            }

            if (src.dims == 2)
            {
                if (squeezeW && squeezeH) return new BufferShape(1, 1, 1, 1, 1);
                if (squeezeW) return new BufferShape(1, src.h, 1, 1, 1);
                if (squeezeH) return new BufferShape(1, src.w, 1, 1, 1);
                return src;
            }

            if (src.dims == 3)
            {
                if (squeezeW && squeezeH && squeezeC) return new BufferShape(1, 1, 1, 1, 1);
                if (squeezeW && squeezeH) return new BufferShape(1, src.c, 1, 1, 1);
                if (squeezeH && squeezeC) return new BufferShape(1, src.w, 1, 1, 1);
                if (squeezeW && squeezeC) return new BufferShape(1, src.h, 1, 1, 1);
                if (squeezeW) return new BufferShape(2, src.h, src.c, 1, 1);
                if (squeezeH) return new BufferShape(2, src.w, src.c, 1, 1);
                if (squeezeC) return new BufferShape(2, src.w, src.h, 1, 1);
                return src;
            }

            if (squeezeW && squeezeH && squeezeD && squeezeC) return new BufferShape(1, 1, 1, 1, 1);
            if (squeezeW && squeezeH && squeezeD) return new BufferShape(1, src.c, 1, 1, 1);
            if (squeezeH && squeezeD && squeezeC) return new BufferShape(1, src.w, 1, 1, 1);
            if (squeezeW && squeezeD && squeezeC) return new BufferShape(1, src.h, 1, 1, 1);
            if (squeezeW && squeezeH && squeezeC) return new BufferShape(1, src.d, 1, 1, 1);
            if (squeezeW && squeezeH) return new BufferShape(2, src.d, src.c, 1, 1);
            if (squeezeW && squeezeD) return new BufferShape(2, src.h, src.c, 1, 1);
            if (squeezeH && squeezeD) return new BufferShape(2, src.w, src.c, 1, 1);
            if (squeezeH && squeezeC) return new BufferShape(2, src.w, src.d, 1, 1);
            if (squeezeW && squeezeC) return new BufferShape(2, src.h, src.d, 1, 1);
            if (squeezeD && squeezeC) return new BufferShape(2, src.w, src.h, 1, 1);
            if (squeezeW) return new BufferShape(3, src.h, src.d, 1, src.c);
            if (squeezeH) return new BufferShape(3, src.w, src.d, 1, src.c);
            if (squeezeD) return new BufferShape(3, src.w, src.h, 1, src.c);
            if (squeezeC) return new BufferShape(3, src.w, src.h, 1, src.d);
            return src;
        }

        internal static NcnnTensorBuffer ResolveSqueezeView(NcnnTensorBuffer src, NcnnParamModel.Layer layer)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            var shape = ResolveSqueezeShape(GetShapeOf(src), layer);
            return src.Reshape(shape.dims, shape.w, shape.h, shape.d, shape.c);
        }

        internal static CropRoi ResolveCropRoi(BufferShape srcShape, NcnnParamModel.Layer layer, IReadOnlyList<BufferShape> bottomShapes)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));

            var startsExpr = layer.GetString(19, null);
            var endsExpr = layer.GetString(20, null);
            var axesExpr = layer.GetString(21, null);
            var starts = layer.GetInts(-23309, null);
            var ends = layer.GetInts(-23310, null);
            var axes = layer.GetInts(-23311, null);

            var hasExprSlice = !string.IsNullOrWhiteSpace(startsExpr) && !string.IsNullOrWhiteSpace(endsExpr);
            if (hasExprSlice)
            {
                starts = EvaluateExpressionListOrNull(startsExpr, bottomShapes, layer);
                ends = EvaluateExpressionListOrNull(endsExpr, bottomShapes, layer);
                axes = EvaluateExpressionListOrNull(axesExpr, bottomShapes, layer);
            }

            if (starts != null && ends != null && starts.Length > 0 && ends.Length > 0)
                return ResolveCropRoiFromSlice(srcShape, starts, ends, axes, layer.name);

            var dims = srcShape.dims;
            var w = srcShape.w;
            var h = srcShape.h;
            var d = srcShape.d;
            var c = srcShape.c;

            var woffset = layer.GetInt(0, 0);
            var hoffset = layer.GetInt(1, 0);
            var doffset = layer.GetInt(13, 0);
            var coffset = layer.GetInt(2, 0);
            var outw = w;
            var outh = h;
            var outd = d;
            var outc = c;
            var woffset2 = layer.GetInt(6, 0);
            var hoffset2 = layer.GetInt(7, 0);
            var doffset2 = layer.GetInt(15, 0);
            var coffset2 = layer.GetInt(8, 0);

            if (dims == 1)
            {
                outw = w - woffset - woffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(3))
                    outw = Math.Min(layer.GetInt(3, outw), outw);
            }
            else if (dims == 2)
            {
                outw = w - woffset - woffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(3))
                    outw = Math.Min(layer.GetInt(3, outw), outw);
                outh = h - hoffset - hoffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(4))
                    outh = Math.Min(layer.GetInt(4, outh), outh);
            }
            else if (dims == 3)
            {
                outw = w - woffset - woffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(3))
                    outw = Math.Min(layer.GetInt(3, outw), outw);
                outh = h - hoffset - hoffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(4))
                    outh = Math.Min(layer.GetInt(4, outh), outh);
                outc = c - coffset - coffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(5))
                    outc = Math.Min(layer.GetInt(5, outc), outc);
            }
            else
            {
                outw = w - woffset - woffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(3))
                    outw = Math.Min(layer.GetInt(3, outw), outw);
                outh = h - hoffset - hoffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(4))
                    outh = Math.Min(layer.GetInt(4, outh), outh);
                outd = d - doffset - doffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(14))
                    outd = Math.Min(layer.GetInt(14, outd), outd);
                outc = c - coffset - coffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(5))
                    outc = Math.Min(layer.GetInt(5, outc), outc);
            }

            return ValidateCropRoi(new CropRoi(woffset, hoffset, doffset, coffset, outw, outh, outd, outc), srcShape, layer.name);
        }

        internal static CropRoi ResolveCropRoi(BufferShape srcShape, BufferShape referenceShape, NcnnParamModel.Layer layer)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));

            var woffset = layer.GetInt(0, 0);
            var hoffset = layer.GetInt(1, 0);
            var doffset = layer.GetInt(13, 0);
            var coffset = layer.GetInt(2, 0);

            if (srcShape.dims == 1)
                return ValidateCropRoi(new CropRoi(woffset, 0, 0, 0, referenceShape.w, 1, 1, 1), srcShape, layer.name);
            if (srcShape.dims == 2)
                return ValidateCropRoi(new CropRoi(woffset, hoffset, 0, 0, referenceShape.w, referenceShape.h, 1, 1), srcShape, layer.name);
            if (srcShape.dims == 3)
                return ValidateCropRoi(new CropRoi(woffset, hoffset, 0, coffset, referenceShape.w, referenceShape.h, 1, referenceShape.dims == 3 ? referenceShape.c : srcShape.c), srcShape, layer.name);
            return ValidateCropRoi(new CropRoi(woffset, hoffset, doffset, coffset, referenceShape.w, referenceShape.h, referenceShape.d, referenceShape.dims == 4 ? referenceShape.c : srcShape.c), srcShape, layer.name);
        }

        internal static CropRoi ResolveCropRoi(BufferShape srcShape, int[] paramData, NcnnParamModel.Layer layer)
        {
            if (paramData == null)
                throw new ArgumentNullException(nameof(paramData));

            CropRoi roi;
            if (srcShape.dims == 1)
                roi = new CropRoi(paramData[0], 0, 0, 0, paramData[3], 1, 1, 1);
            else if (srcShape.dims == 2)
                roi = new CropRoi(paramData[0], paramData[1], 0, 0, paramData[3], paramData[4], 1, 1);
            else if (srcShape.dims == 3)
                roi = new CropRoi(paramData[0], paramData[1], 0, paramData[2], paramData[3], paramData[4], 1, paramData[5]);
            else
                roi = new CropRoi(paramData[0], paramData[1], paramData[2], paramData[3], paramData[4], paramData[5], paramData[6], paramData[7]);

            return ValidateCropRoi(roi, srcShape, layer?.name);
        }

        internal static CropRoi ResolveCropRoiFromSlice(BufferShape srcShape, int[] starts, int[] ends, int[] axes, string layerName)
        {
            if (starts == null || ends == null || starts.Length == 0 || ends.Length == 0)
                throw new InvalidOperationException("Crop slice arrays are empty: " + layerName);

            var dims = srcShape.dims;
            var woffset = 0;
            var hoffset = 0;
            var doffset = 0;
            var coffset = 0;
            var outw = srcShape.w;
            var outh = srcShape.h;
            var outd = srcShape.d;
            var outc = srcShape.c;

            int[] actualAxes;
            if (axes == null || axes.Length == 0)
            {
                actualAxes = new int[Math.Max(starts.Length, dims)];
                for (var i = 0; i < actualAxes.Length; i++)
                    actualAxes[i] = i;
            }
            else
            {
                actualAxes = axes;
            }

            var numAxis = axes == null || axes.Length == 0 ? Math.Min(dims, starts.Length) : Math.Min(starts.Length, axes.Length);
            for (var i = 0; i < numAxis; i++)
            {
                var axis = actualAxes[i];
                if (axis < 0)
                    axis += dims;

                var start = starts[i];
                var end = ends[Math.Min(i, ends.Length - 1)];

                if (dims == 1)
                {
                    ApplySliceBounds(srcShape.w, ref start, ref end, out woffset, out outw);
                    continue;
                }

                if (dims == 2)
                {
                    if (axis == 0) ApplySliceBounds(srcShape.h, ref start, ref end, out hoffset, out outh);
                    else if (axis == 1) ApplySliceBounds(srcShape.w, ref start, ref end, out woffset, out outw);
                    continue;
                }

                if (dims == 3)
                {
                    if (axis == 0) ApplySliceBounds(srcShape.c, ref start, ref end, out coffset, out outc);
                    else if (axis == 1) ApplySliceBounds(srcShape.h, ref start, ref end, out hoffset, out outh);
                    else if (axis == 2) ApplySliceBounds(srcShape.w, ref start, ref end, out woffset, out outw);
                    continue;
                }

                if (axis == 0) ApplySliceBounds(srcShape.c, ref start, ref end, out coffset, out outc);
                else if (axis == 1) ApplySliceBounds(srcShape.d, ref start, ref end, out doffset, out outd);
                else if (axis == 2) ApplySliceBounds(srcShape.h, ref start, ref end, out hoffset, out outh);
                else if (axis == 3) ApplySliceBounds(srcShape.w, ref start, ref end, out woffset, out outw);
            }

            return ValidateCropRoi(new CropRoi(woffset, hoffset, doffset, coffset, outw, outh, outd, outc), srcShape, layerName);
        }

        private static void ApplySliceBounds(int axisSize, ref int start, ref int end, out int offset, out int outSize)
        {
            if (start == -233) start = 0;
            if (end == -233) end = axisSize;
            if (start < 0) start = axisSize + start;
            if (end <= 0) end = axisSize + end;
            if (end == int.MaxValue) end = axisSize;
            offset = Mathf.Clamp(start, 0, axisSize);
            var clampedEnd = Mathf.Clamp(end, offset, axisSize);
            outSize = Mathf.Max(0, clampedEnd - offset);
        }

        internal static CropRoi ValidateCropRoi(CropRoi roi, BufferShape srcShape, string layerName)
        {
            if (roi.outw <= 0 || roi.outh <= 0 || roi.outd <= 0 || roi.outc <= 0)
                throw new InvalidOperationException("Crop produced empty output: " + layerName);

            if (roi.woffset < 0 || roi.hoffset < 0 || roi.doffset < 0 || roi.coffset < 0)
                throw new InvalidOperationException("Crop negative offset: " + layerName);

            if (roi.woffset + roi.outw > srcShape.w
                || roi.hoffset + roi.outh > srcShape.h
                || roi.doffset + roi.outd > srcShape.d
                || roi.coffset + roi.outc > srcShape.c)
            {
                throw new InvalidOperationException("Crop roi out of range: " + layerName);
            }

            return roi;
        }

        internal NcnnTensorBuffer ApplyCrop(
            ComputeBuffer srcBuf,
            NcnnTensorBuffer srcView,
            CropRoi roi,
            List<IDisposable> tempOwned)
        {
            if (srcBuf == null)
                throw new ArgumentNullException(nameof(srcBuf));
            if (srcView == null)
                throw new ArgumentNullException(nameof(srcView));

            var needsCropW = roi.woffset != 0 || roi.outw != srcView.w;
            var needsCropH = roi.hoffset != 0 || roi.outh != srcView.h;
            var needsCropD = roi.doffset != 0 || roi.outd != srcView.d;
            var needsCropC = roi.coffset != 0 || roi.outc != srcView.c;
            if (!needsCropW && !needsCropH && !needsCropD && !needsCropC)
                return srcView;

            var currentBuf = srcBuf;
            var currentView = srcView;

            void ApplyAxisSlice(int tensorAxis, int begin, int outW, int outH, int outD, int outC)
            {
                var outCount = outW * outH * outD * outC;
                var outBuf = RentTempBuffer(outCount, sizeof(float));
                _ops.Slice(currentBuf, currentView.dims, currentView.w, currentView.h, currentView.d, currentView.c, tensorAxis, begin, outW, outH, outD, outC, outBuf);
                var outView = new NcnnTensorBuffer(outBuf, currentView.dims, outW, outH, outD, outC, false);
                tempOwned?.Add(outBuf);
                currentBuf = outBuf;
                currentView = outView;
            }

            if (srcView.dims == 1)
            {
                if (needsCropW)
                    ApplyAxisSlice(0, roi.woffset, roi.outw, 1, 1, 1);
                return currentView;
            }

            if (srcView.dims == 2)
            {
                if (needsCropH)
                    ApplyAxisSlice(1, roi.hoffset, currentView.w, roi.outh, 1, 1);
                if (needsCropW)
                    ApplyAxisSlice(0, roi.woffset, roi.outw, currentView.h, 1, 1);
                return currentView;
            }

            if (srcView.dims == 3)
            {
                if (needsCropC)
                    ApplyAxisSlice(2, roi.coffset, currentView.w, currentView.h, 1, roi.outc);
                if (needsCropH)
                    ApplyAxisSlice(1, roi.hoffset, currentView.w, roi.outh, 1, currentView.c);
                if (needsCropW)
                    ApplyAxisSlice(0, roi.woffset, roi.outw, currentView.h, 1, currentView.c);
                return currentView;
            }

            if (needsCropC)
                ApplyAxisSlice(3, roi.coffset, currentView.w, currentView.h, currentView.d, roi.outc);
            if (needsCropD)
                ApplyAxisSlice(2, roi.doffset, currentView.w, currentView.h, roi.outd, currentView.c);
            if (needsCropH)
                ApplyAxisSlice(1, roi.hoffset, currentView.w, roi.outh, currentView.d, currentView.c);
            if (needsCropW)
                ApplyAxisSlice(0, roi.woffset, roi.outw, currentView.h, currentView.d, currentView.c);

            return currentView;
        }

        internal NcnnTensorBuffer ShuffleChannelCpu(ComputeBuffer srcBuffer, NcnnTensorBuffer srcView, int group, bool reverse)
        {
            if (srcBuffer == null)
                throw new ArgumentNullException(nameof(srcBuffer));
            if (srcView == null)
                throw new ArgumentNullException(nameof(srcView));
            if (srcView.dims < 3)
                throw new InvalidOperationException("ShuffleChannel expects dims>=3");

            var channels = srcView.c;
            if (channels <= 0)
                throw new InvalidOperationException("ShuffleChannel invalid channels");
            if (channels % Mathf.Max(1, group) != 0)
                throw new InvalidOperationException("ShuffleChannel invalid group: " + group + " for c=" + channels);

            var actualGroup = reverse ? channels / Mathf.Max(1, group) : Mathf.Max(1, group);
            var channelsPerGroup = channels / actualGroup;
            var featureSize = srcView.w * srcView.h * srcView.d;

            var srcData = new float[srcBuffer.count];
            srcBuffer.GetData(srcData);
            var dstData = new float[srcData.Length];

            for (var i = 0; i < actualGroup; i++)
            {
                for (var j = 0; j < channelsPerGroup; j++)
                {
                    var srcChannel = channelsPerGroup * i + j;
                    var dstChannel = actualGroup * j + i;
                    Array.Copy(srcData, srcChannel * featureSize, dstData, dstChannel * featureSize, featureSize);
                }
            }

            var outBuffer = RentTempBuffer(dstData.Length, sizeof(float));
            outBuffer.SetData(dstData);
            return new NcnnTensorBuffer(outBuffer, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
        }

        internal static (int mode, int size, int total, NcnnTensorBuffer outputView) ResolveBinaryBroadcast(
            NcnnTensorBuffer aView,
            NcnnTensorBuffer bView,
            int aCount,
            int bCount,
            string layerName)
        {
            if (aCount == bCount)
            {
                return (0, 0, aCount, aView ?? bView);
            }

            if (aView != null && bView != null && aView.dims == 2 && bView.dims == 2)
            {
                // row-wise broadcast: [h,w] op [1,w]
                if (aView.w == bView.w && bView.h == 1 && aView.h > 1)
                    return (2, bCount, aCount, aView);
                if (aView.w == bView.w && aView.h == 1 && bView.h > 1)
                    return (1, aCount, bCount, bView);

                // column-wise broadcast: [h,w] op [h,1]
                if (aView.h == bView.h && bView.w == 1 && aView.w > 1)
                    return (2, bCount, aCount, aView);
                if (aView.h == bView.h && aView.w == 1 && bView.w > 1)
                    return (1, aCount, bCount, bView);
            }

            if (aCount < bCount && bCount % aCount == 0)
                return (1, aCount, bCount, bView);
            if (bCount < aCount && aCount % bCount == 0)
                return (2, bCount, aCount, aView);

            throw new InvalidOperationException("BinaryOp broadcast not supported: " + layerName + " | " + aCount + " vs " + bCount);
        }

        internal bool TryExpand2DBroadcastBuffer(
            ComputeBuffer sourceBuffer,
            NcnnTensorBuffer sourceView,
            NcnnTensorBuffer targetView,
            out ComputeBuffer expandedBuffer,
            out NcnnTensorBuffer expandedView)
        {
            expandedBuffer = null;
            expandedView = null;

            if (sourceBuffer == null || sourceView == null || targetView == null)
                return false;
            if (sourceView.dims != 2 || targetView.dims != 2)
                return false;
            if (sourceView.w == targetView.w && sourceView.h == targetView.h)
                return false;

            bool isRowVector = sourceView.h == 1 && sourceView.w == targetView.w && targetView.h > 1;
            bool isColumnVector = sourceView.w == 1 && sourceView.h == targetView.h && targetView.w > 1;
            if (!isRowVector && !isColumnVector)
                return false;

            var srcData = new float[sourceBuffer.count];
            sourceBuffer.GetData(srcData);

            var expandedData = new float[targetView.w * targetView.h];
            if (isRowVector)
            {
                for (var y = 0; y < targetView.h; y++)
                {
                    var rowBase = y * targetView.w;
                    Array.Copy(srcData, 0, expandedData, rowBase, targetView.w);
                }
            }
            else
            {
                for (var y = 0; y < targetView.h; y++)
                {
                    var value = srcData[y];
                    var rowBase = y * targetView.w;
                    for (var x = 0; x < targetView.w; x++)
                        expandedData[rowBase + x] = value;
                }
            }

            expandedBuffer = RentTempBuffer(expandedData.Length, sizeof(float));
            expandedBuffer.SetData(expandedData);
            expandedView = new NcnnTensorBuffer(expandedBuffer, 2, targetView.w, targetView.h, 1, 1, false);
            return true;
        }

        internal bool TryExpand1DTo2DBroadcastBuffer(
            ComputeBuffer sourceBuffer,
            NcnnTensorBuffer sourceView,
            NcnnTensorBuffer targetView,
            out ComputeBuffer expandedBuffer,
            out NcnnTensorBuffer expandedView)
        {
            expandedBuffer = null;
            expandedView = null;

            if (sourceBuffer == null || sourceView == null || targetView == null)
                return false;
            if (sourceView.dims != 1 || targetView.dims != 2)
                return false;
            if (sourceView.w != targetView.w && sourceView.w != targetView.h)
                return false;

            // Match ncnn binaryop.cpp behavior:
            // if vec length == other.h -> reshape(1, len) => column broadcast
            // else reshape(len, 1) => row broadcast
            bool columnVector = sourceView.w == targetView.h;
            bool rowVector = !columnVector && sourceView.w == targetView.w;
            if (!columnVector && !rowVector)
                return false;

            var srcData = new float[sourceBuffer.count];
            sourceBuffer.GetData(srcData);
            var expandedData = new float[targetView.w * targetView.h];

            if (columnVector)
            {
                for (var y = 0; y < targetView.h; y++)
                {
                    var value = srcData[y];
                    var rowBase = y * targetView.w;
                    for (var x = 0; x < targetView.w; x++)
                        expandedData[rowBase + x] = value;
                }
            }
            else
            {
                for (var y = 0; y < targetView.h; y++)
                {
                    var rowBase = y * targetView.w;
                    Array.Copy(srcData, 0, expandedData, rowBase, targetView.w);
                }
            }

            expandedBuffer = RentTempBuffer(expandedData.Length, sizeof(float));
            expandedBuffer.SetData(expandedData);
            expandedView = new NcnnTensorBuffer(expandedBuffer, 2, targetView.w, targetView.h, 1, 1, false);
            return true;
        }

        internal bool TryExpand3DBroadcastBuffer(
            ComputeBuffer sourceBuffer,
            NcnnTensorBuffer sourceView,
            NcnnTensorBuffer targetView,
            out ComputeBuffer expandedBuffer,
            out NcnnTensorBuffer expandedView)
        {
            expandedBuffer = null;
            expandedView = null;

            if (sourceBuffer == null || sourceView == null || targetView == null)
                return false;
            if (sourceView.dims != 3 || targetView.dims != 3)
                return false;
            if (sourceView.w == targetView.w && sourceView.h == targetView.h && sourceView.c == targetView.c)
                return false;

            // Common CLIP image path broadcast: [1,1,C] op [W,H,C].
            if (!(sourceView.w == 1 && sourceView.h == 1 && sourceView.c == targetView.c))
                return false;

            var srcData = new float[sourceBuffer.count];
            sourceBuffer.GetData(srcData);
            var expandedData = new float[targetView.w * targetView.h * targetView.c];
            var plane = targetView.w * targetView.h;
            for (var c = 0; c < targetView.c; c++)
            {
                var value = srcData[c];
                var dstBase = c * plane;
                for (var i = 0; i < plane; i++)
                    expandedData[dstBase + i] = value;
            }

            expandedBuffer = RentTempBuffer(expandedData.Length, sizeof(float));
            expandedBuffer.SetData(expandedData);
            expandedView = new NcnnTensorBuffer(expandedBuffer, 3, targetView.w, targetView.h, 1, targetView.c, false);
            return true;
        }

        public static Dictionary<string, int> BuildBlobUseCount(NcnnParamModel model)
        {
            var use = new Dictionary<string, int>(StringComparer.Ordinal);
            for (var i = 0; i < model.layers.Count; i++)
            {
                var layer = model.layers[i];
                if (layer.bottomNames == null)
                    continue;
                for (var b = 0; b < layer.bottomNames.Length; b++)
                {
                    var name = layer.bottomNames[b];
                    if (string.IsNullOrEmpty(name))
                        continue;
                    use.TryGetValue(name, out var count);
                    use[name] = count + 1;
                }
            }
            return use;
        }

        public static float ParseLeakySlope(NcnnParamModel.Layer layer)
        {
            if (layer.intParams == null || !layer.intParams.TryGetValue(-23310, out var s) || string.IsNullOrWhiteSpace(s))
                return 0.2f;
            var parts = s.Split(',');
            if (parts.Length >= 2 && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                return v;
            return 0.2f;
        }

        public static float[] ParseActivationParams(NcnnParamModel.Layer layer)
        {
            if (layer == null)
                return Array.Empty<float>();
            return layer.GetFloats(-23310, Array.Empty<float>());
        }

        public static float ApplyActivationScalarCpu(float v, int activationType, float param0 = 0f, float param1 = 0f)
        {
            if (activationType == 1)
                return Mathf.Max(0f, v);
            if (activationType == 2)
                return v < 0f ? v * param0 : v;
            if (activationType == 3)
                return Mathf.Clamp(v, param0, param1);
            if (activationType == 4)
                return 1f / (1f + Mathf.Exp(-v));
            if (activationType == 5)
                return v * (float)Math.Tanh(Mathf.Log(Mathf.Exp(v) + 1f));
            if (activationType == 6)
                return v * Mathf.Clamp(v * param0 + param1, 0f, 1f);
            return v;
        }

        public static (float coeffA, float coeffB) ParseEltwiseCoeff(NcnnParamModel.Layer layer)
        {
            if (layer.intParams == null || !layer.intParams.TryGetValue(-23301, out var s) || string.IsNullOrWhiteSpace(s))
                return (1f, 1f);
            var parts = s.Split(',');
            if (parts.Length >= 3
                && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var a)
                && float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var b))
                return (a, b);
            return (1f, 1f);
        }

        public static Vector4[] PackBiasToO4(float[] b, int outC, int outPacks)
        {
            var packed = new Vector4[outPacks];
            for (var op = 0; op < outPacks; op++)
            {
                var x = op * 4 + 0 < outC ? b[op * 4 + 0] : 0f;
                var y = op * 4 + 1 < outC ? b[op * 4 + 1] : 0f;
                var z = op * 4 + 2 < outC ? b[op * 4 + 2] : 0f;
                var w = op * 4 + 3 < outC ? b[op * 4 + 3] : 0f;
                packed[op] = new Vector4(x, y, z, w);
            }
            return packed;
        }

        public static Vector4[] PackWeightsToO4I4K(float[] w, int outC, int inC, int k, int outPacks, int inPacks)
        {
            var packed = new Vector4[outPacks * inPacks * k * k * 4];
            if (w != null
                && outC > 0
                && inC > 0
                && k > 0
                && (outC & 3) == 0
                && (inC & 3) == 0
                && outPacks * 4 == outC
                && inPacks * 4 == inC)
            {
                PackWeightsToO4I4KAligned(w, packed, inC, k, outPacks, inPacks);
                return packed;
            }

            for (var op = 0; op < outPacks; op++)
            {
                for (var ip = 0; ip < inPacks; ip++)
                {
                    for (var ky = 0; ky < k; ky++)
                    {
                        for (var kx = 0; kx < k; kx++)
                        {
                            var kIndex = ky * k + kx;
                            for (var lane = 0; lane < 4; lane++)
                            {
                                var oc = op * 4 + lane;
                                var baseIndex = ((((op * inPacks + ip) * k + ky) * k + kx) * 4) + lane;
                                var v = Vector4.zero;
                                for (var icLane = 0; icLane < 4; icLane++)
                                {
                                    var ic = ip * 4 + icLane;
                                    if (oc < outC && ic < inC)
                                    {
                                        var srcIndex = (((oc * inC + ic) * k + ky) * k + kx);
                                        v[icLane] = w[srcIndex];
                                    }
                                }
                                packed[baseIndex] = v;
                            }
                        }
                    }
                }
            }
            return packed;
        }

        private static void PackWeightsToO4I4KAligned(float[] src, Vector4[] dst, int inC, int k, int outPacks, int inPacks)
        {
            var kArea = k * k;
            var minParallelWeights = 1 << 20;
            if (src.Length >= minParallelWeights && outPacks >= 4)
            {
                Parallel.For(0, outPacks, op => PackWeightsToO4I4KAlignedOp(src, dst, inC, kArea, inPacks, op));
                return;
            }

            for (var op = 0; op < outPacks; op++)
                PackWeightsToO4I4KAlignedOp(src, dst, inC, kArea, inPacks, op);
        }

        private static void PackWeightsToO4I4KAlignedOp(float[] src, Vector4[] dst, int inC, int kArea, int inPacks, int op)
        {
            var ocBase = op * 4;
            var ocStride = inC * kArea;
            var opDstBase = op * inPacks * kArea * 4;

            for (var ip = 0; ip < inPacks; ip++)
            {
                var icBase = ip * 4;
                var ipDstBase = opDstBase + ip * kArea * 4;
                var srcBase0 = (ocBase * inC + icBase) * kArea;

                for (var ki = 0; ki < kArea; ki++)
                {
                    var dstBase = ipDstBase + ki * 4;
                    var srcBase = srcBase0 + ki;
                    dst[dstBase + 0] = new Vector4(src[srcBase], src[srcBase + kArea], src[srcBase + kArea * 2], src[srcBase + kArea * 3]);
                    srcBase += ocStride;
                    dst[dstBase + 1] = new Vector4(src[srcBase], src[srcBase + kArea], src[srcBase + kArea * 2], src[srcBase + kArea * 3]);
                    srcBase += ocStride;
                    dst[dstBase + 2] = new Vector4(src[srcBase], src[srcBase + kArea], src[srcBase + kArea * 2], src[srcBase + kArea * 3]);
                    srcBase += ocStride;
                    dst[dstBase + 3] = new Vector4(src[srcBase], src[srcBase + kArea], src[srcBase + kArea * 2], src[srcBase + kArea * 3]);
                }
            }
        }

        public static Vector4[] PackDepthWiseWeightsToP4K4(float[] w, int channels, int k, int packs)
        {
            var packed = new Vector4[packs * k * k];
            for (var p = 0; p < packs; p++)
            {
                for (var ky = 0; ky < k; ky++)
                {
                    for (var kx = 0; kx < k; kx++)
                    {
                        var baseIndex = (p * k + ky) * k + kx;
                        var v = Vector4.zero;
                        for (var lane = 0; lane < 4; lane++)
                        {
                            var c = p * 4 + lane;
                            if (c < channels)
                            {
                                var srcIndex = ((c * k + ky) * k + kx);
                                v[lane] = w[srcIndex];
                            }
                        }
                        packed[baseIndex] = v;
                    }
                }
            }
            return packed;
        }

        internal bool ShouldCompareTextureConvLayer(string layerName)
        {
            if (!string.IsNullOrWhiteSpace(layerName))
            {
                if (DebugCompareTextureLayers != null && DebugCompareTextureLayers.Contains(layerName))
                    return true;
                if (DebugCompareTextureConvLayers != null && (DebugCompareTextureConvLayers.Contains(layerName) || DebugCompareTextureConvLayers.Contains("*")))
                    return true;
            }

            return false;
        }

        internal bool ShouldKeepRawConvWeightsForTexturePath(string layerName, ConvPack conv, bool needGeneralTexturePack, bool needDepthWiseTexturePack)
        {
            if (KeepRawConvWeightsForTexturePath)
                return true;
            if (conv == null)
                return true;
            if (ForceBufferConvolutionAll || ForceBufferConvolution)
                return true;
            if (ShouldCompareTextureConvLayer(layerName))
                return true;
            return !needGeneralTexturePack && !needDepthWiseTexturePack;
        }

        internal static void UploadRawConvWeights(ConvPack pack, float[] weights, float[] bias)
        {
            if (pack == null)
                throw new ArgumentNullException(nameof(pack));
            if (weights == null)
                throw new ArgumentNullException(nameof(weights));
            if (bias == null)
                throw new ArgumentNullException(nameof(bias));
            pack.rawWeight = new ComputeBuffer(weights.Length, sizeof(float), ComputeBufferType.Structured);
            pack.rawBias = new ComputeBuffer(bias.Length, sizeof(float), ComputeBufferType.Structured);
            pack.rawWeight.SetData(weights);
            pack.rawBias.SetData(bias);
        }

        internal void CompareTextureConvPath(
            string layerName,
            string bottomName,
            ConvPack conv,
            int outW,
            int outH,
            RenderTexture textureOutput,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            List<IDisposable> tempOwned)
        {
            try
            {
                var srcBuf = GetOrConvertToBuffer(bottomName, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                var srcTensor = TryGetBufferView(bottomName, bufferBlobs, bufferViews);
                if (srcBuf == null || srcTensor == null)
                {
                    DebugLog?.Invoke(layerName + " | compare skipped: source buffer unavailable");
                    return;
                }

                LogBufferStats(layerName, "src", srcBuf, srcTensor.w * srcTensor.h * srcTensor.d * srcTensor.c);
                if (conv.rawWeight == null || conv.rawBias == null)
                {
                    DebugLog?.Invoke(layerName + " | compare skipped: raw convolution weights were not kept");
                    return;
                }
                LogBufferStats(layerName, "weight", conv.rawWeight, conv.weightSize);
                LogBufferStats(layerName, "bias", conv.rawBias, conv.outC);

                using var refTensor = RentTempTensorBuffer(3, outW, outH, 1, conv.outC);
                if (conv.isDepthWise || conv.group > 1 || conv.kernelW != 3 || conv.kernelH != 3 || conv.strideW != conv.strideH || conv.padLeft != conv.padTop)
                {
                    _ops.ConvDepthWise(
                        srcTensor,
                        conv.rawWeight,
                        conv.rawBias,
                        conv.outC,
                        conv.group,
                        conv.kernelW,
                        conv.kernelH,
                        conv.strideW,
                        conv.strideH,
                        conv.padLeft,
                        conv.padTop,
                        conv.dilationW,
                        conv.dilationH,
                        conv.activationType,
                        conv.activationSlope,
                        refTensor);
                }
                else
                {
                    _ops.Conv3x3(srcTensor, conv.rawWeight, conv.rawBias, conv.outC, conv.strideW, conv.padLeft, conv.activationType, conv.activationSlope, refTensor);
                }

                var logicalCount = outW * outH * conv.outC;
                var physicalCount = outW * outH * conv.outPacks * 4;
                var texturePhysical = RentTempBuffer(physicalCount, sizeof(float));
                try
                {
                    _ops.Pack4ToBufferCHW(textureOutput, outW, outH, conv.outPacks * 4, texturePhysical);

                    var refData = new float[logicalCount];
                    refTensor.buffer.GetData(refData);

                    var texPhysicalData = new float[physicalCount];
                    texturePhysical.GetData(texPhysicalData);

                    double sumAbs = 0d;
                    float maxAbs = 0f;
                    var validCount = 0;
                    var refNanCount = 0;
                    var texNanCount = 0;
                    var preview = new List<string>(8);
                    var compareCount = Mathf.Min(logicalCount, texPhysicalData.Length);
                    for (var i = 0; i < compareCount; i++)
                    {
                        var rv = refData[i];
                        var tv = texPhysicalData[i];
                        var refFinite = !float.IsNaN(rv) && !float.IsInfinity(rv);
                        var texFinite = !float.IsNaN(tv) && !float.IsInfinity(tv);
                        if (!refFinite) refNanCount++;
                        if (!texFinite) texNanCount++;
                        if (!refFinite || !texFinite)
                        {
                            if (preview.Count < 8)
                                preview.Add(i + ": ref=" + rv.ToString("G9", CultureInfo.InvariantCulture) + " tex=" + tv.ToString("G9", CultureInfo.InvariantCulture));
                            continue;
                        }

                        var diff = Mathf.Abs(rv - tv);
                        sumAbs += diff;
                        if (diff > maxAbs)
                            maxAbs = diff;
                        validCount++;
                        if (preview.Count < 8)
                            preview.Add(i + ": ref=" + rv.ToString("G9", CultureInfo.InvariantCulture) + " tex=" + tv.ToString("G9", CultureInfo.InvariantCulture));
                    }

                    var meanAbs = validCount > 0 ? (float)(sumAbs / validCount) : float.NaN;
                    DebugLog?.Invoke(layerName
                        + " | texture_vs_buffer mean_abs=" + meanAbs.ToString("G9", CultureInfo.InvariantCulture)
                        + " | max_abs=" + maxAbs.ToString("G9", CultureInfo.InvariantCulture)
                        + " | count=" + compareCount
                        + " | valid=" + validCount
                        + " | ref_nan=" + refNanCount
                        + " | tex_nan=" + texNanCount);
                    for (var i = 0; i < preview.Count; i++)
                        DebugLog?.Invoke(layerName + " | sample[" + i + "] " + preview[i]);
                }
                finally
                {
                    ReturnTempBuffer(texturePhysical);
                }
            }
            catch (Exception e)
            {
                DebugLog?.Invoke(layerName + " | compare failed: " + e.Message);
            }
        }

        internal void SynchronizeGpuBufferUse(ComputeBuffer buffer)
        {
            if (buffer == null || buffer.count <= 0)
                return;

            buffer.GetData(_gpuSyncScratch, 0, 0, 1);
        }

        internal void CompareMaxPoolingIndPath(string layerName, TensorRef src, BufferShape srcShape, RenderTexture textureOutput, RenderTexture textureIndices, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int outW, int outH)
        {
            try
            {
                var srcCount = srcShape.w * srcShape.h * srcShape.c;
                var srcBuffer = RentTempBuffer(srcCount, sizeof(float));
                try
                {
                    _ops.Pack4ToBufferCHW(src.texture, srcShape.w, srcShape.h, srcShape.c, srcBuffer);
                    var srcData = new float[srcCount];
                    srcBuffer.GetData(srcData);

                    var refData = new float[outW * outH * srcShape.c];
                    var refIndexData = new float[outW * outH * srcShape.c];
                    var srcPlane = srcShape.w * srcShape.h;
                    var outPlane = outW * outH;
                    for (var c = 0; c < srcShape.c; c++)
                    {
                        var srcBase = c * srcPlane;
                        var dstBase = c * outPlane;
                        for (var oy = 0; oy < outH; oy++)
                        {
                            var sy0 = oy * strideH - padTop;
                            for (var ox = 0; ox < outW; ox++)
                            {
                                var sx0 = ox * strideW - padLeft;
                                var best = float.NegativeInfinity;
                                var bestIndex = 0;
                                for (var ky = 0; ky < kernelH; ky++)
                                {
                                    var sy = sy0 + ky;
                                    if (sy < 0 || sy >= srcShape.h)
                                        continue;
                                    for (var kx = 0; kx < kernelW; kx++)
                                    {
                                        var sx = sx0 + kx;
                                        if (sx < 0 || sx >= srcShape.w)
                                            continue;
                                        var linear = sy * srcShape.w + sx;
                                        var v = srcData[srcBase + linear];
                                        if (v > best)
                                        {
                                            best = v;
                                            bestIndex = linear;
                                        }
                                    }
                                }

                                var dstIndex = dstBase + oy * outW + ox;
                                refData[dstIndex] = best;
                                refIndexData[dstIndex] = bestIndex;
                            }
                        }
                    }

                    var logicalCount = outW * outH * srcShape.c;
                    var texBuffer = RentTempBuffer(logicalCount, sizeof(float));
                    try
                    {
                        _ops.Pack4ToBufferCHW(textureOutput, outW, outH, srcShape.c, texBuffer);
                        var texData = new float[logicalCount];
                        texBuffer.GetData(texData);

                        float[] texIndexData = null;
                        if (textureIndices != null)
                        {
                            var idxBuffer = RentTempBuffer(logicalCount, sizeof(float));
                            try
                            {
                                _ops.Pack4ToBufferCHW(textureIndices, outW, outH, srcShape.c, idxBuffer);
                                texIndexData = new float[logicalCount];
                                idxBuffer.GetData(texIndexData);
                            }
                            finally
                            {
                                ReturnTempBuffer(idxBuffer);
                            }
                        }

                        double sumAbs = 0d;
                        float maxAbs = 0f;
                        var valid = 0;
                        var refNonFinite = 0;
                        var texNonFinite = 0;
                        double sumIdxAbs = 0d;
                        float maxIdxAbs = 0f;
                        var idxValid = 0;
                        for (var i = 0; i < logicalCount; i++)
                        {
                            var rv = refData[i];
                            var tv = texData[i];
                            var rFinite = !float.IsNaN(rv) && !float.IsInfinity(rv);
                            var tFinite = !float.IsNaN(tv) && !float.IsInfinity(tv);
                            if (!rFinite) refNonFinite++;
                            if (!tFinite) texNonFinite++;
                            if (!rFinite || !tFinite)
                                continue;

                            var diff = Mathf.Abs(rv - tv);
                            sumAbs += diff;
                            if (diff > maxAbs)
                                maxAbs = diff;
                            valid++;

                            if (texIndexData != null)
                            {
                                var idxDiff = Mathf.Abs(refIndexData[i] - texIndexData[i]);
                                sumIdxAbs += idxDiff;
                                if (idxDiff > maxIdxAbs)
                                    maxIdxAbs = idxDiff;
                                idxValid++;
                            }
                        }

                        var meanAbs = valid > 0 ? (float)(sumAbs / valid) : float.NaN;
                        var meanIdxAbs = idxValid > 0 ? (float)(sumIdxAbs / idxValid) : float.NaN;
                        DebugLog?.Invoke(layerName
                            + " | maxpool_compare mean_abs=" + meanAbs.ToString("G9", CultureInfo.InvariantCulture)
                            + " | max_abs=" + maxAbs.ToString("G9", CultureInfo.InvariantCulture)
                            + " | idx_mean_abs=" + meanIdxAbs.ToString("G9", CultureInfo.InvariantCulture)
                            + " | idx_max_abs=" + maxIdxAbs.ToString("G9", CultureInfo.InvariantCulture)
                            + " | valid=" + valid.ToString(CultureInfo.InvariantCulture)
                            + " | ref_nonfinite=" + refNonFinite.ToString(CultureInfo.InvariantCulture)
                            + " | tex_nonfinite=" + texNonFinite.ToString(CultureInfo.InvariantCulture)
                            + " | shape=" + srcShape.w + "x" + srcShape.h + "x" + srcShape.c
                            + " -> " + outW + "x" + outH + "x" + srcShape.c);
                    }
                    finally
                    {
                        ReturnTempBuffer(texBuffer);
                    }
                }
                finally
                {
                    ReturnTempBuffer(srcBuffer);
                }
            }
            catch (Exception e)
            {
                DebugLog?.Invoke(layerName + " | maxpool_compare failed: " + e.Message);
            }
        }

        internal void ApplyMaxPoolingIndCpu(TensorRef src, BufferShape srcShape, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int outW, int outH, RenderTexture outRt, RenderTexture idxRt)
        {
            var srcCount = srcShape.w * srcShape.h * srcShape.c;
            var srcBuffer = RentTempBuffer(srcCount, sizeof(float));
            try
            {
                _ops.Pack4ToBufferCHW(src.texture, srcShape.w, srcShape.h, srcShape.c, srcBuffer);
                var srcData = new float[srcCount];
                srcBuffer.GetData(srcData);

                var outCount = outW * outH * srcShape.c;
                var valueData = new float[outCount];
                var indexData = new float[outCount];
                var srcPlane = srcShape.w * srcShape.h;
                var outPlane = outW * outH;

                for (var c = 0; c < srcShape.c; c++)
                {
                    var srcBase = c * srcPlane;
                    var dstBase = c * outPlane;
                    for (var oy = 0; oy < outH; oy++)
                    {
                        var sy0 = oy * strideH - padTop;
                        for (var ox = 0; ox < outW; ox++)
                        {
                            var sx0 = ox * strideW - padLeft;
                            var best = float.NegativeInfinity;
                            var bestIndex = 0;
                            for (var ky = 0; ky < kernelH; ky++)
                            {
                                var sy = sy0 + ky;
                                if (sy < 0 || sy >= srcShape.h)
                                    continue;
                                for (var kx = 0; kx < kernelW; kx++)
                                {
                                    var sx = sx0 + kx;
                                    if (sx < 0 || sx >= srcShape.w)
                                        continue;
                                    var linear = sy * srcShape.w + sx;
                                    var v = srcData[srcBase + linear];
                                    if (v > best)
                                    {
                                        best = v;
                                        bestIndex = linear;
                                    }
                                }
                            }

                            var dstIndex = dstBase + oy * outW + ox;
                            valueData[dstIndex] = best;
                            indexData[dstIndex] = bestIndex;
                        }
                    }
                }

                var valueBuffer = RentTempBuffer(outCount, sizeof(float));
                var indexBuffer = RentTempBuffer(outCount, sizeof(float));
                try
                {
                    valueBuffer.SetData(valueData);
                    indexBuffer.SetData(indexData);
                    _ops.FillPack4FromBufferCHW(valueBuffer, outW, outH, srcShape.c, outRt);
                    _ops.FillPack4FromBufferCHW(indexBuffer, outW, outH, srcShape.c, idxRt);
                    SynchronizeGpuBufferUse(indexBuffer);
                }
                finally
                {
                    ReturnTempBuffer(valueBuffer);
                    ReturnTempBuffer(indexBuffer);
                }
            }
            finally
            {
                ReturnTempBuffer(srcBuffer);
            }
        }

        internal void ApplyMaxPoolingIndCpu(
            ComputeBuffer srcBuffer,
            NcnnTensorBuffer srcView,
            int kernelW,
            int kernelH,
            int strideW,
            int strideH,
            int padLeft,
            int padTop,
            NcnnTensorBuffer outValue,
            NcnnTensorBuffer outIndex)
        {
            if (srcBuffer == null)
                throw new ArgumentNullException(nameof(srcBuffer));
            if (srcView == null)
                throw new ArgumentNullException(nameof(srcView));
            if (outValue == null)
                throw new ArgumentNullException(nameof(outValue));
            if (outIndex == null)
                throw new ArgumentNullException(nameof(outIndex));

            var srcCount = srcView.elementCount;
            var srcData = new float[srcCount];
            srcBuffer.GetData(srcData);

            var outCount = outValue.elementCount;
            var valueData = new float[outCount];
            var indexData = new float[outCount];
            var srcPlane = srcView.w * srcView.h;
            var outPlane = outValue.w * outValue.h;

            for (var c = 0; c < srcView.c; c++)
            {
                var srcBase = c * srcPlane;
                var dstBase = c * outPlane;
                for (var oy = 0; oy < outValue.h; oy++)
                {
                    var sy0 = oy * strideH - padTop;
                    for (var ox = 0; ox < outValue.w; ox++)
                    {
                        var sx0 = ox * strideW - padLeft;
                        var best = float.NegativeInfinity;
                        var bestIndex = 0;
                        for (var ky = 0; ky < kernelH; ky++)
                        {
                            var sy = sy0 + ky;
                            if (sy < 0 || sy >= srcView.h)
                                continue;
                            for (var kx = 0; kx < kernelW; kx++)
                            {
                                var sx = sx0 + kx;
                                if (sx < 0 || sx >= srcView.w)
                                    continue;
                                var linear = sy * srcView.w + sx;
                                var v = srcData[srcBase + linear];
                                if (v > best)
                                {
                                    best = v;
                                    bestIndex = linear;
                                }
                            }
                        }

                        var dstIndex = dstBase + oy * outValue.w + ox;
                        valueData[dstIndex] = best;
                        indexData[dstIndex] = bestIndex;
                    }
                }
            }

            outValue.buffer.SetData(valueData);
            outIndex.buffer.SetData(indexData);
        }

        internal void ApplyMaxUnPoolingCpu(TensorRef src, BufferShape srcShape, IndexRef idx, int outW, int outH, RenderTexture outRt)
        {
            var pooledCount = src.width * src.height * srcShape.c;
            var pooledBuffer = RentTempBuffer(pooledCount, sizeof(float));
            var indexBuffer = RentTempBuffer(src.width * src.height * srcShape.c, sizeof(float));
            try
            {
                _ops.Pack4ToBufferCHW(src.texture, src.width, src.height, srcShape.c, pooledBuffer);
                var pooledData = new float[pooledCount];
                pooledBuffer.GetData(pooledData);

                _ops.Pack4ToBufferCHW(idx.texture, idx.width, idx.height, srcShape.c, indexBuffer);
                var indexData = new float[indexBuffer.count];
                indexBuffer.GetData(indexData);

                var outPlane = outW * outH;
                var pooledPlane = src.width * src.height;
                var outData = new float[outPlane * srcShape.c];

                for (var c = 0; c < srcShape.c; c++)
                {
                    var pooledBase = c * pooledPlane;
                    var outBase = c * outPlane;
                    for (var i = 0; i < pooledPlane; i++)
                    {
                        var dstIndex = Mathf.RoundToInt(indexData[pooledBase + i]);
                        if (dstIndex < 0 || dstIndex >= outPlane)
                            continue;
                        outData[outBase + dstIndex] = pooledData[pooledBase + i];
                    }
                }

                var outBuffer = RentTempBuffer(outData.Length, sizeof(float));
                try
                {
                    outBuffer.SetData(outData);
                    _ops.FillPack4FromBufferCHW(outBuffer, outW, outH, srcShape.c, outRt);
                    SynchronizeGpuBufferUse(outBuffer);
                }
                finally
                {
                    ReturnTempBuffer(outBuffer);
                }
            }
            finally
            {
                ReturnTempBuffer(pooledBuffer);
                ReturnTempBuffer(indexBuffer);
            }
        }

        internal void ApplyMaxUnPoolingCpu(
            ComputeBuffer pooledBuffer,
            NcnnTensorBuffer pooledView,
            ComputeBuffer indexBuffer,
            NcnnTensorBuffer indexView,
            int outW,
            int outH,
            NcnnTensorBuffer outTensor)
        {
            if (pooledBuffer == null)
                throw new ArgumentNullException(nameof(pooledBuffer));
            if (pooledView == null)
                throw new ArgumentNullException(nameof(pooledView));
            if (indexBuffer == null)
                throw new ArgumentNullException(nameof(indexBuffer));
            if (indexView == null)
                throw new ArgumentNullException(nameof(indexView));
            if (outTensor == null)
                throw new ArgumentNullException(nameof(outTensor));

            var pooledData = new float[pooledView.elementCount];
            pooledBuffer.GetData(pooledData);
            var indexData = new float[indexView.elementCount];
            indexBuffer.GetData(indexData);

            var outPlane = outW * outH;
            var pooledPlane = pooledView.w * pooledView.h;
            var outData = new float[outPlane * pooledView.c];

            for (var c = 0; c < pooledView.c; c++)
            {
                var pooledBase = c * pooledPlane;
                var outBase = c * outPlane;
                for (var i = 0; i < pooledPlane; i++)
                {
                    var dstIndex = Mathf.RoundToInt(indexData[pooledBase + i]);
                    if (dstIndex < 0 || dstIndex >= outPlane)
                        continue;
                    outData[outBase + dstIndex] = pooledData[pooledBase + i];
                }
            }

            outTensor.buffer.SetData(outData);
        }

        internal void LogBufferStats(string layerName, string kind, ComputeBuffer buffer, int logicalCount)
        {
            if (DebugLog == null || buffer == null)
                return;

            var count = logicalCount > 0 ? Mathf.Min(logicalCount, buffer.count) : buffer.count;
            if (count <= 0)
            {
                DebugLog(layerName + " | " + kind + " count=0");
                return;
            }

            var data = new float[count];
            buffer.GetData(data, 0, 0, count);

            var finite = 0;
            var nan = 0;
            var inf = 0;
            var min = float.PositiveInfinity;
            var max = float.NegativeInfinity;
            var preview = new List<string>(4);
            for (var i = 0; i < data.Length; i++)
            {
                var v = data[i];
                if (float.IsNaN(v))
                {
                    nan++;
                }
                else if (float.IsInfinity(v))
                {
                    inf++;
                }
                else
                {
                    finite++;
                    if (v < min) min = v;
                    if (v > max) max = v;
                }

                if (preview.Count < 4)
                    preview.Add(v.ToString("G9", CultureInfo.InvariantCulture));
            }

            DebugLog(layerName
                + " | " + kind
                + " finite=" + finite
                + " nan=" + nan
                + " inf=" + inf
                + " min=" + (finite > 0 ? min.ToString("G9", CultureInfo.InvariantCulture) : "NaN")
                + " max=" + (finite > 0 ? max.ToString("G9", CultureInfo.InvariantCulture) : "NaN")
                + " sample=" + string.Join(",", preview));
        }
    }
}

