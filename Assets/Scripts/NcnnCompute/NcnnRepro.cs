using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnRepro : IDisposable
    {
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

        private readonly struct PooledRt
        {
            public readonly RenderTexture rt;
            public readonly GraphicsFence fence;
            public readonly int frame;

            public PooledRt(RenderTexture rt, GraphicsFence fence, int frame)
            {
                this.rt = rt;
                this.fence = fence;
                this.frame = frame;
            }
        }

        public sealed class ConvPack : IDisposable
        {
            public int outC;
            public int inC;
            public int outPacks;
            public int inPacks;
            public int kernel;
            public int pad;
            public int biasTerm;
            public int weightSize;
            public int activationType;
            public float activationSlope;
            public ComputeBuffer w4;
            public ComputeBuffer wTm23;
            public ComputeBuffer b4;
            public bool useWinograd23;

            public void Dispose()
            {
                try { w4?.Dispose(); } catch { }
                try { wTm23?.Dispose(); } catch { }
                try { b4?.Dispose(); } catch { }
            }
        }

        public sealed class TensorRef
        {
            public ComputeTexture t2;
            public RenderTexture t1;
            public int w;
            public int h;
            public int packs;
            public int refs;
            public bool owned;
        }

        public NcnnParamModel Model { get; private set; }
        public Dictionary<string, ConvPack> Conv => _conv;
        private readonly Dictionary<string, ConvPack> _conv = new Dictionary<string, ConvPack>(StringComparer.Ordinal);
        private Dictionary<string, int> _blobUseCount;

        private NcnnOps _ops;

        private readonly Dictionary<RtKey, Stack<PooledRt>> _rtPool = new Dictionary<RtKey, Stack<PooledRt>>();
        private bool _useTempPool = false;
        private int _maxPooledPerShape = 2;
        private readonly HashSet<ComputeTexture> _cmdSets = new HashSet<ComputeTexture>();

        public bool EnableTempPool
        {
            get => _useTempPool;
            set => _useTempPool = value;
        }

        public int MaxPooledPerShape
        {
            get => _maxPooledPerShape;
            set => _maxPooledPerShape = Mathf.Max(0, value);
        }

        public NcnnOps Ops => _ops;

        public event Func<string, ConvPack, int, int, bool> SelectWinograd23;
        public event Action<string, string, int, int, int, int, double> OnConvComplete;

        private bool ShouldUseWinograd23(ConvPack pack, int srcW, int srcH)
        {
            var handler = SelectWinograd23;
            return handler != null && handler(pack == null ? "" : "", pack, srcW, srcH);
        }

        private void NotifyConvComplete(string layerName, string mode, int srcW, int srcH, int inPacks, int outPacks, double gpuMs)
        {
            try { OnConvComplete?.Invoke(layerName, mode, srcW, srcH, inPacks, outPacks, gpuMs); } catch { }
        }

        public NcnnRepro(NcnnOps ops)
        {
            _ops = ops ?? throw new ArgumentNullException(nameof(ops));
        }

        public void SetOps(NcnnOps ops)
        {
            _ops = ops ?? throw new ArgumentNullException(nameof(ops));
        }

        public void LoadModel(string paramText, NcnnBinReader br)
        {
            Model = NcnnParamParser.Parse(paramText);
            _blobUseCount = BuildBlobUseCount(Model);

            foreach (var layer in Model.layers)
            {
                if (!string.Equals(layer.type, "Convolution", StringComparison.Ordinal))
                    continue;

                var pack = new ConvPack();
                pack.outC = layer.GetInt(0, 0);
                pack.kernel = layer.GetInt(1, 3);
                pack.pad = layer.GetInt(4, 0);
                pack.biasTerm = layer.GetInt(5, 0);
                pack.weightSize = layer.GetInt(6, 0);
                pack.activationType = layer.GetInt(9, 0);
                pack.activationSlope = ParseLeakySlope(layer);
                pack.inC = Mathf.Max(1, pack.weightSize / Mathf.Max(1, pack.outC * pack.kernel * pack.kernel));
                pack.inPacks = (pack.inC + 3) / 4;
                pack.outPacks = (pack.outC + 3) / 4;

                var tag = br.ReadInt32();
                if (tag != 0x01306B47)
                    throw new InvalidOperationException("unexpected weight tag at " + br.Position + ": 0x" + tag.ToString("X8", CultureInfo.InvariantCulture));

                var w = br.ReadFp16ArrayAsFloat32(pack.weightSize);
                var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
                var w4 = PackWeightsToO4I4K(w, pack.outC, pack.inC, pack.kernel, pack.outPacks, pack.inPacks);
                var b4 = PackBiasToO4(b, pack.outC, pack.outPacks);
                pack.w4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                pack.b4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                pack.w4.SetData(w4);
                pack.b4.SetData(b4);

                if (pack.kernel == 3 && pack.pad == 1 && NcnnWinograd23.CanUse(pack.kernel, pack.pad, pack.inPacks, pack.outPacks))
                {
                    pack.useWinograd23 = true;
                    var wTm = NcnnWinograd23.PackWeightTm23(w, pack.outC, pack.inC, pack.outPacks, pack.inPacks);
                    pack.wTm23 = new ComputeBuffer(wTm.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    pack.wTm23.SetData(wTm);
                }

                _conv[layer.name] = pack;
            }
        }

        public RenderTexture ForwardPack4(RenderTexture inputPack4, int inputPacks, string inputBlobName = "data", ICollection<string> pinnedNames = null)
        {
            var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
            var blobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);

            var inputRef = new TensorRef { t1 = inputPack4, w = inputPack4.width, h = inputPack4.height, packs = inputPacks, refs = 1, owned = false };
            blobs[inputBlobName] = inputRef;

            for (var li = 0; li < Model.layers.Count; li++)
            {
                var l = Model.layers[li];
                if (string.Equals(l.type, "Input", StringComparison.Ordinal))
                    continue;

                if (string.Equals(l.type, "Split", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    for (var i = 0; i < l.topNames.Length; i++)
                    {
                        blobs[l.topNames[i]] = src;
                        src.refs++;
                    }
                    Consume(blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Concat", StringComparison.Ordinal))
                {
                    var parts = new TensorRef[l.bottomNames.Length];
                    var sumP = 0;
                    var w = 0;
                    var h = 0;
                    for (var i = 0; i < l.bottomNames.Length; i++)
                    {
                        var tr = Get(blobs, l.bottomNames[i]);
                        parts[i] = tr;
                        w = tr.w;
                        h = tr.h;
                        sumP += tr.packs;
                    }

                    var outArr = RentTempArray(w, h, sumP, RenderTextureFormat.ARGBHalf);
                    var off = 0;
                    for (var i = 0; i < parts.Length; i++)
                    {
                        _ops.CopyPack4(parts[i].t1, 0, outArr, off, parts[i].packs);
                        off += parts[i].packs;
                    }

                    blobs[l.topNames[0]] = new TensorRef { t1 = outArr, w = w, h = h, packs = sumP, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Reshape", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    blobs[l.topNames[0]] = src;
                    src.refs++;
                    Consume(blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Padding", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var top = l.GetInt(0, 0);
                    var bottom = l.GetInt(1, 0);
                    var left = l.GetInt(2, 0);
                    var right = l.GetInt(3, 0);
                    var type = l.GetInt(4, 0);
                    var value = l.GetFloat(5, 0f);

                    var outW = src.w + left + right;
                    var outH = src.h + top + bottom;
                    if (outW <= 0 || outH <= 0)
                        throw new InvalidOperationException("Padding invalid out size: " + outW + "x" + outH);

                    var outArr = RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.PaddingPack4(src.t1, src.packs, left, right, top, bottom, type, new Vector4(value, value, value, value), outArr);
                    blobs[l.topNames[0]] = new TensorRef { t1 = outArr, w = outW, h = outH, packs = src.packs, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Pooling", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
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

                    var outW = (src.w + padLeft * 2 - kernelW) / strideW + 1;
                    var outH = (src.h + padTop * 2 - kernelH) / strideH + 1;
                    outW = Mathf.Max(1, outW);
                    outH = Mathf.Max(1, outH);
                    var outArr = RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.PoolingPack4(src.t1, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, poolingType, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t1 = outArr, w = outW, h = outH, packs = src.packs, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Softmax", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var axis = l.GetInt(0, 0);
                    if (axis != 0)
                        throw new InvalidOperationException("Softmax axis not supported: " + axis);
                    var outArr = RentTempArray(src.w, src.h, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SoftmaxChannelPack4(src.t1, src.packs, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t1 = outArr, w = src.w, h = src.h, packs = src.packs, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Convolution", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var pack = _conv[l.name];
                    if (src.packs != pack.inPacks)
                        throw new InvalidOperationException("unexpected in packs for " + l.name + ": " + src.packs + " vs " + pack.inPacks);

                    var outArr = RentTempArray(src.w, src.h, pack.outPacks, RenderTextureFormat.ARGBHalf);
                    var swGpu = ShouldUseWinograd23(pack, src.w, src.h);
                    var profileGpu = OnConvComplete != null;
                    var swStopwatch = default(System.Diagnostics.Stopwatch);
                    string convMode;

                    if (profileGpu)
                    {
                        _ops.DebugSyncGpu();
                        swStopwatch = System.Diagnostics.Stopwatch.StartNew();
                    }

                    if (pack.kernel == 1)
                    {
                        convMode = "direct1x1";
                        _ops.Conv1x1Pack4(src.t1, pack.inPacks, pack.w4, pack.b4, pack.outPacks, pack.activationType, pack.activationSlope, outArr);
                    }
                    else if (pack.kernel == 3)
                    {
                        if (swGpu)
                        {
                            convMode = "winograd23";
                            _ops.Conv3x3Pack4Winograd23(src.t1, pack.inPacks, pack.wTm23, pack.b4, pack.outPacks, pack.biasTerm, pack.activationType, pack.activationSlope, outArr);
                        }
                        else
                        {
                            convMode = "direct";
                            _ops.Conv3x3Pack4(src.t1, pack.inPacks, pack.w4, pack.b4, pack.outPacks, pack.pad, pack.activationType, pack.activationSlope, outArr);
                        }
                    }
                    else
                    {
                        throw new InvalidOperationException("unsupported kernel size: " + pack.kernel);
                    }

                    if (profileGpu)
                    {
                        _ops.DebugSyncGpu();
                        swStopwatch.Stop();
                        NotifyConvComplete(l.name, convMode, src.w, src.h, pack.inPacks, pack.outPacks, swStopwatch.Elapsed.TotalMilliseconds);
                    }

                    blobs[l.topNames[0]] = new TensorRef { t1 = outArr, w = src.w, h = src.h, packs = pack.outPacks, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Eltwise", StringComparison.Ordinal))
                {
                    var a = Get(blobs, l.bottomNames[0]);
                    var b = Get(blobs, l.bottomNames[1]);
                    var coeff = ParseEltwiseCoeff(l);
                    var outArr = RentTempArray(a.w, a.h, a.packs, RenderTextureFormat.ARGBHalf);
                    _ops.AddPack4(a.t1, b.t1, coeff.coeffA, coeff.coeffB, a.packs, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t1 = outArr, w = a.w, h = a.h, packs = a.packs, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "BinaryOp", StringComparison.Ordinal))
                {
                    var opType = l.GetInt(0, 0);
                    var withScalar = l.GetInt(1, 0);
                    var scalarB = l.GetFloat(2, 0f);
                    var a = Get(blobs, l.bottomNames[0]);
                    var outArr = RentTempArray(a.w, a.h, a.packs, RenderTextureFormat.ARGBHalf);
                    if (withScalar != 0)
                    {
                        _ops.BinaryOpScalarPack4(a.t1, scalarB, a.packs, opType, outArr);
                    }
                    else
                    {
                        var b = Get(blobs, l.bottomNames[1]);
                        if (a.w != b.w || a.h != b.h || a.packs != b.packs)
                            throw new InvalidOperationException("BinaryOp broadcast not supported: " + l.name);
                        _ops.BinaryOpPack4(a.t1, b.t1, a.packs, opType, outArr);
                    }
                    blobs[l.topNames[0]] = new TensorRef { t1 = outArr, w = a.w, h = a.h, packs = a.packs, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "UnaryOp", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var opType = l.GetInt(0, 0);
                    var outArr = RentTempArray(src.w, src.h, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.UnaryOpPack4(src.t1, src.packs, opType, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t1 = outArr, w = src.w, h = src.h, packs = src.packs, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Swish", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var outArr = RentTempArray(src.w, src.h, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SwishPack4(src.t1, src.packs, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t1 = outArr, w = src.w, h = src.h, packs = src.packs, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Sigmoid", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var outArr = RentTempArray(src.w, src.h, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SigmoidPack4(src.t1, src.packs, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t1 = outArr, w = src.w, h = src.h, packs = src.packs, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "GELU", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var fast = l.GetInt(0, 0) != 0;
                    var outArr = RentTempArray(src.w, src.h, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.GeluPack4(src.t1, src.packs, fast, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t1 = outArr, w = src.w, h = src.h, packs = src.packs, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Interp", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var resizeType = l.GetInt(0, 2);
                    var sx = l.GetFloat(1, 1f);
                    var sy = l.GetFloat(2, 1f);

                    if (Mathf.Abs(sx - 2f) < 1e-3f && Mathf.Abs(sy - 2f) < 1e-3f)
                    {
                        var outArr = RentTempArray(src.w * 2, src.h * 2, src.packs, RenderTextureFormat.ARGBHalf);
                        if (resizeType == 1)
                            _ops.Interp2xNearestPack4(src.t1, src.packs, outArr);
                        else
                            _ops.Interp2xPack4(src.t1, src.packs, outArr);
                        blobs[l.topNames[0]] = new TensorRef { t1 = outArr, w = src.w * 2, h = src.h * 2, packs = src.packs, refs = 1, owned = true };
                        Consume(blobs, remaining, l.bottomNames, pinnedNames);
                        continue;
                    }

                    if (Mathf.Abs(sx - 0.5f) < 1e-3f && Mathf.Abs(sy - 0.5f) < 1e-3f)
                    {
                        var outArr = RentTempArray(src.w / 2, src.h / 2, src.packs, RenderTextureFormat.ARGBHalf);
                        if (resizeType == 1)
                            _ops.InterpDown2NearestPack4(src.t1, src.packs, outArr);
                        else
                            _ops.InterpDown2Pack4(src.t1, src.packs, outArr);
                        blobs[l.topNames[0]] = new TensorRef { t1 = outArr, w = src.w / 2, h = src.h / 2, packs = src.packs, refs = 1, owned = true };
                        Consume(blobs, remaining, l.bottomNames, pinnedNames);
                        continue;
                    }

                    throw new InvalidOperationException("unsupported interp scale: " + sx.ToString("0.###", CultureInfo.InvariantCulture) + "," + sy.ToString("0.###", CultureInfo.InvariantCulture));
                }

                throw new InvalidOperationException("unsupported layer type: " + l.type);
            }

            var outRef = Get(blobs, "output");
            var keep = outRef.t1;
            outRef.t1 = null;
            outRef.owned = false;

            var visited = new HashSet<TensorRef>();
            foreach (var kv in blobs)
            {
                var tr = kv.Value;
                if (tr == null || !visited.Add(tr))
                    continue;
                if (tr.owned && tr.t1 != null)
                    ReturnTempArray(tr.t1);
            }

            return keep;
        }

        public ComputeTexture ForwardPack4(CommandBuffer cmd, ComputeTexture inputPack4, int inputPacks, string inputBlobName = "data", ICollection<string> pinnedNames = null)
        {
            var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
            var blobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);

            var inputRef = new TensorRef { t2 = inputPack4, w = inputPack4.width, h = inputPack4.height, packs = inputPacks, refs = 1, owned = false };
            blobs[inputBlobName] = inputRef;

            for (var li = 0; li < Model.layers.Count; li++)
            {
                var l = Model.layers[li];
                if (string.Equals(l.type, "Input", StringComparison.Ordinal))
                    continue;

                if (string.Equals(l.type, "Split", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    for (var i = 0; i < l.topNames.Length; i++)
                    {
                        blobs[l.topNames[i]] = src;
                        src.refs++;
                    }
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Concat", StringComparison.Ordinal))
                {
                    var parts = new TensorRef[l.bottomNames.Length];
                    var sumP = 0;
                    var w = 0;
                    var h = 0;
                    for (var i = 0; i < l.bottomNames.Length; i++)
                    {
                        var tr = Get(blobs, l.bottomNames[i]);
                        parts[i] = tr;
                        w = tr.w;
                        h = tr.h;
                        sumP += tr.packs;
                    }

                    var outArr = RentTempArray(cmd, w, h, sumP, RenderTextureFormat.ARGBHalf);
                    var off = 0;
                    for (var i = 0; i < parts.Length; i++)
                    {
                        _ops.CopyPack4(cmd, parts[i].t2, 0, outArr, off, parts[i].packs);
                        off += parts[i].packs;
                    }

                    blobs[l.topNames[0]] = new TensorRef { t2 = outArr, w = w, h = h, packs = sumP, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Reshape", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    blobs[l.topNames[0]] = src;
                    src.refs++;
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Padding", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var top = l.GetInt(0, 0);
                    var bottom = l.GetInt(1, 0);
                    var left = l.GetInt(2, 0);
                    var right = l.GetInt(3, 0);
                    var type = l.GetInt(4, 0);
                    var value = l.GetFloat(5, 0f);

                    var outW = src.w + left + right;
                    var outH = src.h + top + bottom;
                    if (outW <= 0 || outH <= 0)
                        throw new InvalidOperationException("Padding invalid out size: " + outW + "x" + outH);

                    var outArr = RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.PaddingPack4(cmd, src.t2, src.packs, left, right, top, bottom, type, new Vector4(value, value, value, value), outArr);
                    blobs[l.topNames[0]] = new TensorRef { t2 = outArr, w = outW, h = outH, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Pooling", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
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

                    var outW = (src.w + padLeft * 2 - kernelW) / strideW + 1;
                    var outH = (src.h + padTop * 2 - kernelH) / strideH + 1;
                    outW = Mathf.Max(1, outW);
                    outH = Mathf.Max(1, outH);
                    var outArr = RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.PoolingPack4(cmd, src.t2, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, poolingType, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t2 = outArr, w = outW, h = outH, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Softmax", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var axis = l.GetInt(0, 0);
                    if (axis != 0)
                        throw new InvalidOperationException("Softmax axis not supported: " + axis);
                    var outArr = RentTempArray(cmd, src.w, src.h, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SoftmaxChannelPack4(cmd, src.t2, src.packs, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t2 = outArr, w = src.w, h = src.h, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Convolution", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var pack = _conv[l.name];
                    if (src.packs != pack.inPacks)
                        throw new InvalidOperationException("unexpected in packs for " + l.name + ": " + src.packs + " vs " + pack.inPacks);

                    var outArr = RentTempArray(cmd, src.w, src.h, pack.outPacks, RenderTextureFormat.ARGBHalf);

                    if (pack.kernel == 1)
                    {
                        _ops.Conv1x1Pack4(cmd, src.t2, pack.inPacks, pack.w4, pack.b4, pack.outPacks, pack.activationType, pack.activationSlope, outArr);
                    }
                    else if (pack.kernel == 3)
                    {
                        if (ShouldUseWinograd23(pack, src.w, src.h))
                        {
                            _ops.Conv3x3Pack4Winograd23(cmd, src.t2, pack.inPacks, pack.wTm23, pack.b4, pack.outPacks, pack.biasTerm, pack.activationType, pack.activationSlope, outArr);
                        }
                        else
                        {
                            _ops.Conv3x3Pack4(cmd, src.t2, pack.inPacks, pack.w4, pack.b4, pack.outPacks, pack.pad, pack.activationType, pack.activationSlope, outArr);
                        }
                    }
                    else
                    {
                        throw new InvalidOperationException("unsupported kernel size: " + pack.kernel);
                    }

                    blobs[l.topNames[0]] = new TensorRef { t2 = outArr, w = src.w, h = src.h, packs = pack.outPacks, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Eltwise", StringComparison.Ordinal))
                {
                    var a = Get(blobs, l.bottomNames[0]);
                    var b = Get(blobs, l.bottomNames[1]);
                    var coeff = ParseEltwiseCoeff(l);
                    var outArr = RentTempArray(cmd, a.w, a.h, a.packs, RenderTextureFormat.ARGBHalf);
                    _ops.AddPack4(cmd, a.t2, b.t2, coeff.coeffA, coeff.coeffB, a.packs, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t2 = outArr, w = a.w, h = a.h, packs = a.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "BinaryOp", StringComparison.Ordinal))
                {
                    var opType = l.GetInt(0, 0);
                    var withScalar = l.GetInt(1, 0);
                    var scalarB = l.GetFloat(2, 0f);
                    var a = Get(blobs, l.bottomNames[0]);
                    var outArr = RentTempArray(cmd, a.w, a.h, a.packs, RenderTextureFormat.ARGBHalf);
                    if (withScalar != 0)
                    {
                        _ops.BinaryOpScalarPack4(cmd, a.t2, scalarB, a.packs, opType, outArr);
                    }
                    else
                    {
                        var b = Get(blobs, l.bottomNames[1]);
                        if (a.w != b.w || a.h != b.h || a.packs != b.packs)
                            throw new InvalidOperationException("BinaryOp broadcast not supported: " + l.name);
                        _ops.BinaryOpPack4(cmd, a.t2, b.t2, a.packs, opType, outArr);
                    }
                    blobs[l.topNames[0]] = new TensorRef { t2 = outArr, w = a.w, h = a.h, packs = a.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "UnaryOp", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var opType = l.GetInt(0, 0);
                    var outArr = RentTempArray(cmd, src.w, src.h, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.UnaryOpPack4(cmd, src.t2, src.packs, opType, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t2 = outArr, w = src.w, h = src.h, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Swish", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var outArr = RentTempArray(cmd, src.w, src.h, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SwishPack4(cmd, src.t2, src.packs, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t2 = outArr, w = src.w, h = src.h, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Sigmoid", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var outArr = RentTempArray(cmd, src.w, src.h, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SigmoidPack4(cmd, src.t2, src.packs, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t2 = outArr, w = src.w, h = src.h, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "GELU", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var fast = l.GetInt(0, 0) != 0;
                    var outArr = RentTempArray(cmd, src.w, src.h, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.GeluPack4(cmd, src.t2, src.packs, fast, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t2 = outArr, w = src.w, h = src.h, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(l.type, "Interp", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var resizeType = l.GetInt(0, 2);
                    var sx = l.GetFloat(1, 1f);
                    var sy = l.GetFloat(2, 1f);

                    if (Mathf.Abs(sx - 2f) < 1e-3f && Mathf.Abs(sy - 2f) < 1e-3f)
                    {
                        var outArr = RentTempArray(cmd, src.w * 2, src.h * 2, src.packs, RenderTextureFormat.ARGBHalf);
                        if (resizeType == 1)
                            _ops.Interp2xNearestPack4(cmd, src.t2, src.packs, outArr);
                        else
                            _ops.Interp2xPack4(cmd, src.t2, src.packs, outArr);
                        blobs[l.topNames[0]] = new TensorRef { t2 = outArr, w = src.w * 2, h = src.h * 2, packs = src.packs, refs = 1, owned = true };
                        ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                        continue;
                    }

                    if (Mathf.Abs(sx - 0.5f) < 1e-3f && Mathf.Abs(sy - 0.5f) < 1e-3f)
                    {
                        var outArr = RentTempArray(cmd, src.w / 2, src.h / 2, src.packs, RenderTextureFormat.ARGBHalf);
                        if (resizeType == 1)
                            _ops.InterpDown2NearestPack4(cmd, src.t2, src.packs, outArr);
                        else
                            _ops.InterpDown2Pack4(cmd, src.t2, src.packs, outArr);
                        blobs[l.topNames[0]] = new TensorRef { t2 = outArr, w = src.w / 2, h = src.h / 2, packs = src.packs, refs = 1, owned = true };
                        ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                        continue;
                    }

                    throw new InvalidOperationException("unsupported interp scale: " + sx.ToString("0.###", CultureInfo.InvariantCulture) + "," + sy.ToString("0.###", CultureInfo.InvariantCulture));
                }

                throw new InvalidOperationException("unsupported layer type: " + l.type);
            }

            var outRef = Get(blobs, "output");
            var keep = outRef.t2;
            outRef.t2 = null;
            outRef.owned = false;

            var visited = new HashSet<TensorRef>();
            foreach (var kv in blobs)
            {
                var tr = kv.Value;
                if (tr == null || !visited.Add(tr))
                    continue;
                if (tr.owned && tr.t2 != null)
                    ReturnTempArray(cmd, tr.t2);
            }

            return keep;
        }

        private static TensorRef Get(Dictionary<string, TensorRef> blobs, string name)
        {
            if (!blobs.TryGetValue(name, out var tr) || tr == null)
                throw new InvalidOperationException("blob not found: " + name);
            return tr;
        }

        private void ConsumeCmd(CommandBuffer cmd, Dictionary<string, TensorRef> blobs, Dictionary<string, int> remaining, string[] bottomNames, ICollection<string> pinnedNames)
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
                        if (tr.owned && tr.t2 != null)
                        {
                            try { ReturnTempArray(cmd, tr.t2); } catch { }
                        }
                        tr.t2 = null;
                        tr.owned = false;
                    }
                }
                blobs.Remove(b);
            }
        }

        private void Consume(Dictionary<string, TensorRef> blobs, Dictionary<string, int> remaining, string[] bottomNames, ICollection<string> pinnedNames)
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
                        if (tr.owned && tr.t1 != null)
                        {
                            try { ReturnTempArray(tr.t1); } catch { }
                        }
                        tr.t1 = null;
                        tr.owned = false;
                    }
                }
                blobs.Remove(b);
            }
        }

        public static Dictionary<string, int> BuildBlobUseCount(NcnnParamModel model)
        {
            var use = new Dictionary<string, int>(StringComparer.Ordinal);
            for (var i = 0; i < model.layers.Count; i++)
            {
                var l = model.layers[i];
                if (l.bottomNames == null)
                    continue;
                for (var b = 0; b < l.bottomNames.Length; b++)
                {
                    var n = l.bottomNames[b];
                    if (string.IsNullOrEmpty(n))
                        continue;
                    use.TryGetValue(n, out var c);
                    use[n] = c + 1;
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
            var r = new Vector4[outPacks];
            for (var op = 0; op < outPacks; op++)
            {
                var oc0 = op * 4 + 0;
                var oc1 = op * 4 + 1;
                var oc2 = op * 4 + 2;
                var oc3 = op * 4 + 3;
                r[op] = new Vector4(
                    oc0 < outC ? b[oc0] : 0f,
                    oc1 < outC ? b[oc1] : 0f,
                    oc2 < outC ? b[oc2] : 0f,
                    oc3 < outC ? b[oc3] : 0f);
            }
            return r;
        }

        public static Vector4[] PackWeightsToO4I4K3(float[] w, int outC, int inC, int outPacks, int inPacks)
        {
            var r = new Vector4[outPacks * inPacks * 3 * 3 * 4];
            var idx = 0;
            for (var op = 0; op < outPacks; op++)
            {
                for (var ip = 0; ip < inPacks; ip++)
                {
                    for (var ky = 0; ky < 3; ky++)
                    {
                        for (var kx = 0; kx < 3; kx++)
                        {
                            for (var ol = 0; ol < 4; ol++)
                            {
                                var oc = op * 4 + ol;
                                var il0 = ip * 4 + 0;
                                var il1 = ip * 4 + 1;
                                var il2 = ip * 4 + 2;
                                var il3 = ip * 4 + 3;
                                var k = ky * 3 + kx;

                                float GetW(int ic)
                                {
                                    if (oc >= outC || ic >= inC)
                                        return 0f;
                                    return w[(oc * inC + ic) * 9 + k];
                                }

                                r[idx++] = new Vector4(GetW(il0), GetW(il1), GetW(il2), GetW(il3));
                            }
                        }
                    }
                }
            }
            return r;
        }

        public static Vector4[] PackWeightsToO4I4K(float[] w, int outC, int inC, int k, int outPacks, int inPacks)
        {
            var r = new Vector4[outPacks * inPacks * k * k * 4];
            var idx = 0;
            for (var op = 0; op < outPacks; op++)
            {
                for (var ip = 0; ip < inPacks; ip++)
                {
                    for (var ky = 0; ky < k; ky++)
                    {
                        for (var kx = 0; kx < k; kx++)
                        {
                            for (var ol = 0; ol < 4; ol++)
                            {
                                var oc = op * 4 + ol;
                                var il0 = ip * 4 + 0;
                                var il1 = ip * 4 + 1;
                                var il2 = ip * 4 + 2;
                                var il3 = ip * 4 + 3;
                                var kk = ky * k + kx;

                                float GetW(int ic)
                                {
                                    if (oc >= outC || ic >= inC)
                                        return 0f;
                                    return w[(oc * inC + ic) * (k * k) + kk];
                                }

                                r[idx++] = new Vector4(GetW(il0), GetW(il1), GetW(il2), GetW(il3));
                            }
                        }
                    }
                }
            }
            return r;
        }

        public RenderTexture RentTempArray(int w, int h, int depth, RenderTextureFormat format)
        {
            if (!_useTempPool)
                return CreateTempArray(w, h, depth, format);

            var key = new RtKey(w, h, Mathf.Max(1, depth), format);
            if (_rtPool.TryGetValue(key, out var stack) && stack.Count > 0)
            {
                var keep = new Stack<PooledRt>(stack.Count);
                RenderTexture hit = null;
                while (stack.Count > 0)
                {
                    var p = stack.Pop();
                    if (hit == null && p.rt != null && IsFencePassedOrAged(p))
                    {
                        hit = p.rt;
                        break;
                    }
                    keep.Push(p);
                }
                while (keep.Count > 0)
                    stack.Push(keep.Pop());
                if (hit != null)
                    return hit;
            }
            return CreateTempArray(w, h, depth, format);
        }

        private static RenderTexture CreateTempArray(int w, int h, int depth, RenderTextureFormat format)
        {
            var desc = new RenderTextureDescriptor(w, h, format, 0)
            {
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = Mathf.Max(1, depth),
                msaaSamples = 1,
                sRGB = false,
                enableRandomWrite = true
            };
            return RenderTexture.GetTemporary(desc);
        }

        public ComputeTexture RentTempArray(CommandBuffer cmd, int w, int h, int depth, RenderTextureFormat format)
        {
            var desc = new RenderTextureDescriptor(w, h, format, 0)
            {
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = Mathf.Max(1, depth),
                msaaSamples = 1,
                sRGB = false,
                enableRandomWrite = true
            };

            var guid = Guid.NewGuid();
            int id = Shader.PropertyToID(guid.ToString());
            cmd.GetTemporaryRT(id, desc);
            ComputeTexture t = new ComputeTexture();
            t.nameID = id;
            t.width = w;
            t.height = h;
            _cmdSets.Add(t);
            return t;
        }

        public void ReturnTempArray(RenderTexture rt)
        {
            if (rt == null)
                return;
            if (!_useTempPool)
            {
                RenderTexture.ReleaseTemporary(rt);
                return;
            }

            var key = new RtKey(rt.width, rt.height, rt.volumeDepth, rt.format);
            if (!_rtPool.TryGetValue(key, out var stack))
            {
                stack = new Stack<PooledRt>();
                _rtPool[key] = stack;
            }
            var cap = Mathf.Max(0, _maxPooledPerShape);
            if (stack.Count >= cap)
            {
                RenderTexture.ReleaseTemporary(rt);
                return;
            }
            try
            {
                var fence = Graphics.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.ComputeProcessing);
                stack.Push(new PooledRt(rt, fence, Time.frameCount));
            }
            catch
            {
                RenderTexture.ReleaseTemporary(rt);
            }
        }

        public void ReturnTempArray(CommandBuffer cmd, ComputeTexture t)
        {
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
                var stack = kv.Value;
                while (stack.Count > 0)
                {
                    var rt = stack.Pop().rt;
                    try { RenderTexture.ReleaseTemporary(rt); } catch { }
                }
            }
            _rtPool.Clear();
        }

        private static bool IsFencePassedOrAged(PooledRt p)
        {
            try
            {
                return p.fence.passed;
            }
            catch
            {
                return (Time.frameCount - p.frame) >= 2;
            }
        }

        public void Release()
        {
            foreach (var kv in _conv)
                kv.Value?.Dispose();
            _conv.Clear();
            ClearTempPool();
            Model = null;
            _blobUseCount = null;
        }

        public void Dispose()
        {
            Release();
        }
    }
}