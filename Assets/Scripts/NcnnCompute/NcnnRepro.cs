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

        public sealed class InnerProductPack : IDisposable
        {
            public int inFeatures;
            public int outFeatures;
            public int weightSize;
            public int biasTerm;
            public ComputeBuffer w;
            public ComputeBuffer b;

            public void Dispose()
            {
                try { w?.Dispose(); } catch { }
                try { b?.Dispose(); } catch { }
            }
        }

        public sealed class MemoryDataPack : IDisposable
        {
            public ComputeBuffer data;

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

        public abstract class InferResultBase : IDisposable
        {
            internal readonly Dictionary<string, TensorRef> Blobs;
            internal readonly NcnnRepro Repro;
            internal readonly HashSet<TensorRef> VisitedTextures;

            internal InferResultBase(Dictionary<string, TensorRef> blobs, NcnnRepro repro)
            {
                Blobs = blobs;
                Repro = repro;
                VisitedTextures = new HashSet<TensorRef>();
            }

            public RenderTexture GetTexture(string name)
            {
                if (!Blobs.TryGetValue(name, out var tr) || tr == null || tr.t1 == null)
                    throw new InvalidOperationException("blob not found: " + name);
                return tr.t1;
            }

            public RenderTexture ExtractTexture(string name)
            {
                if (!Blobs.TryGetValue(name, out var tr) || tr == null || tr.t1 == null)
                    throw new InvalidOperationException("blob not found: " + name);
                tr.owned = false;
                var rt = tr.t1;
                tr.t1 = null;
                return rt;
            }

            public virtual void Dispose()
            {
                foreach (var kv in Blobs)
                {
                    var tr = kv.Value;
                    if (tr == null || !VisitedTextures.Add(tr))
                        continue;
                    if (tr.owned && tr.t1 != null)
                        try { Repro.ReturnTempArray(tr.t1); } catch { }
                }
            }
        }

        public sealed class RealEsrganInferResult : InferResultBase
        {
            internal RealEsrganInferResult(Dictionary<string, TensorRef> blobs, NcnnRepro repro)
                : base(blobs, repro)
            {
            }
        }

        public sealed class GFPGANInferResult : InferResultBase
        {
            internal readonly Dictionary<string, ComputeBuffer> BufferBlobs;
            internal readonly List<ComputeBuffer> TempBuffers;

            internal GFPGANInferResult(Dictionary<string, TensorRef> blobs, Dictionary<string, ComputeBuffer> bufferBlobs, NcnnRepro repro)
                : base(blobs, repro)
            {
                BufferBlobs = bufferBlobs;
                TempBuffers = new List<ComputeBuffer>();
            }

            public ComputeBuffer GetBuffer(string name)
            {
                if (!BufferBlobs.TryGetValue(name, out var buf) || buf == null)
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

            public ComputeBuffer ExtractBuffer(string name)
            {
                if (!BufferBlobs.TryGetValue(name, out var buf) || buf == null)
                    throw new InvalidOperationException("buffer blob not found: " + name);
                BufferBlobs.Remove(name);
                return buf;
            }

            public override void Dispose()
            {
                for (var i = 0; i < TempBuffers.Count; i++)
                {
                    try { TempBuffers[i]?.Dispose(); } catch { }
                }

                foreach (var kv in BufferBlobs)
                {
                    try { kv.Value?.Dispose(); } catch { }
                }
                BufferBlobs.Clear();

                base.Dispose();
            }
        }

        public NcnnParamModel Model { get; private set; }
        public Dictionary<string, ConvPack> Conv => _conv;
        private readonly Dictionary<string, ConvPack> _conv = new Dictionary<string, ConvPack>(StringComparer.Ordinal);
        public Dictionary<string, InnerProductPack> InnerProduct => _innerProduct;
        private readonly Dictionary<string, InnerProductPack> _innerProduct = new Dictionary<string, InnerProductPack>(StringComparer.Ordinal);
        public Dictionary<string, MemoryDataPack> MemoryData => _memoryData;
        private readonly Dictionary<string, MemoryDataPack> _memoryData = new Dictionary<string, MemoryDataPack>(StringComparer.Ordinal);
        public Dictionary<string, EmbedPack> Embed => _embed;
        private readonly Dictionary<string, EmbedPack> _embed = new Dictionary<string, EmbedPack>(StringComparer.Ordinal);
        public Dictionary<string, LayerNormPack> LayerNorm => _layerNorm;
        private readonly Dictionary<string, LayerNormPack> _layerNorm = new Dictionary<string, LayerNormPack>(StringComparer.Ordinal);
        public Dictionary<string, GroupNormPack> GroupNorm => _groupNorm;
        private readonly Dictionary<string, GroupNormPack> _groupNorm = new Dictionary<string, GroupNormPack>(StringComparer.Ordinal);
        public Dictionary<string, MultiHeadAttentionPack> MultiHeadAttention => _multiHeadAttention;
        private readonly Dictionary<string, MultiHeadAttentionPack> _multiHeadAttention = new Dictionary<string, MultiHeadAttentionPack>(StringComparer.Ordinal);
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
                if (string.Equals(layer.type, "Convolution", StringComparison.Ordinal))
                {
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
                    continue;
                }

                if (string.Equals(layer.type, "InnerProduct", StringComparison.Ordinal))
                {
                    var ip = new InnerProductPack();
                    ip.outFeatures = layer.GetInt(0, 0);
                    ip.biasTerm = layer.GetInt(1, 0);
                    ip.weightSize = layer.GetInt(2, 0);
                    ip.inFeatures = ip.outFeatures > 0 ? (ip.weightSize / ip.outFeatures) : 0;
                    if (ip.outFeatures <= 0 || ip.inFeatures <= 0)
                        throw new InvalidOperationException("invalid InnerProduct shape for " + layer.name);

                    var tag = br.ReadInt32();
                    if (tag != 0x01306B47)
                        throw new InvalidOperationException("unexpected weight tag at " + br.Position + ": 0x" + tag.ToString("X8", CultureInfo.InvariantCulture));

                    var w = br.ReadFp16ArrayAsFloat32(ip.weightSize);
                    var b = ip.biasTerm != 0 ? br.ReadFloat32Array(ip.outFeatures) : new float[ip.outFeatures];
                    ip.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                    ip.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                    ip.w.SetData(w);
                    ip.b.SetData(b);
                    _innerProduct[layer.name] = ip;
                    continue;
                }

                if (string.Equals(layer.type, "MemoryData", StringComparison.Ordinal))
                {
                    var w = layer.GetInt(0, 0);
                    var h = layer.GetInt(1, 0);
                    var d = layer.GetInt(11, 0);
                    var c = layer.GetInt(2, 0);
                    var loadType = layer.GetInt(21, 1);
                    var a = br.ReadNcnnMatAsFloat32(w, h, d, c, loadType);
                    var buf = new ComputeBuffer(a.Length, sizeof(float), ComputeBufferType.Structured);
                    buf.SetData(a);
                    _memoryData[layer.name] = new MemoryDataPack { data = buf };
                    continue;
                }

                if (string.Equals(layer.type, "Embed", StringComparison.Ordinal))
                {
                    var ep = new EmbedPack();
                    ep.numOutput = layer.GetInt(0, 0);
                    ep.inputDim = layer.GetInt(1, 0);
                    ep.biasTerm = layer.GetInt(2, 0);
                    ep.weightSize = layer.GetInt(3, 0);
                    if (ep.numOutput <= 0 || ep.inputDim <= 0 || ep.weightSize <= 0)
                        throw new InvalidOperationException("Embed invalid params: " + layer.name);

                    var embedFlagPos = br.Position;
                    var flag = br.ReadUInt32();
                    var sum = (byte)(flag & 0xFF) + (byte)((flag >> 8) & 0xFF) + (byte)((flag >> 16) & 0xFF) + (byte)((flag >> 24) & 0xFF);
                    if (flag == 0x01306B47)
                    {
                        var w = br.ReadFp16ArrayAsFloat32(ep.weightSize);
                        ep.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                        ep.w.SetData(w);
                    }
                    else if (sum == 0)
                    {
                        var w = br.ReadFloat32Array(ep.weightSize);
                        ep.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                        ep.w.SetData(w);
                    }
                    else
                    {
                        throw new InvalidOperationException("Embed unexpected flag at " + embedFlagPos + ": 0x" + flag.ToString("X8", CultureInfo.InvariantCulture));
                    }

                    if (ep.biasTerm != 0)
                    {
                        var b = br.ReadNcnnMatAsFloat32(ep.numOutput, 0, 0, 0, 1);
                        ep.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                        ep.b.SetData(b);
                    }

                    _embed[layer.name] = ep;
                    continue;
                }

                if (string.Equals(layer.type, "LayerNorm", StringComparison.Ordinal))
                {
                    var lp = new LayerNormPack();
                    lp.affineSize = layer.GetInt(0, 0);
                    lp.eps = layer.GetFloat(1, 1e-5f);
                    lp.affine = layer.GetInt(2, 1) != 0;
                    if (lp.affine && lp.affineSize > 0)
                    {
                        var gamma = br.ReadNcnnMatAsFloat32(lp.affineSize, 0, 0, 0, 1);
                        var beta = br.ReadNcnnMatAsFloat32(lp.affineSize, 0, 0, 0, 1);
                        lp.gamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
                        lp.beta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
                        lp.gamma.SetData(gamma);
                        lp.beta.SetData(beta);
                    }
                    _layerNorm[layer.name] = lp;
                    continue;
                }

                if (string.Equals(layer.type, "GroupNorm", StringComparison.Ordinal))
                {
                    var gp = new GroupNormPack();
                    gp.group = layer.GetInt(0, 1);
                    gp.channels = layer.GetInt(1, 0);
                    gp.eps = layer.GetFloat(2, 1e-5f);
                    gp.affine = layer.GetInt(3, 1) != 0;
                    if (gp.affine && gp.channels > 0)
                    {
                        var gamma = br.ReadNcnnMatAsFloat32(gp.channels, 0, 0, 0, 1);
                        var beta = br.ReadNcnnMatAsFloat32(gp.channels, 0, 0, 0, 1);
                        gp.gamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
                        gp.beta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
                        gp.gamma.SetData(gamma);
                        gp.beta.SetData(beta);
                    }
                    _groupNorm[layer.name] = gp;
                    continue;
                }

                if (string.Equals(layer.type, "MultiHeadAttention", StringComparison.Ordinal))
                {
                    var mp = new MultiHeadAttentionPack();
                    mp.embedDim = layer.GetInt(0, 0);
                    mp.numHeads = layer.GetInt(1, 1);
                    mp.weightDataSize = layer.GetInt(2, 0);
                    mp.kdim = layer.GetInt(3, mp.embedDim);
                    mp.vdim = layer.GetInt(4, mp.embedDim);
                    mp.scale = layer.GetFloat(6, 1f / Mathf.Sqrt(Mathf.Max(1, mp.embedDim / Mathf.Max(1, mp.numHeads))));
                    mp.qdim = mp.embedDim > 0 ? (mp.weightDataSize / Math.Max(1, mp.embedDim)) : 0;

                    var qW = br.ReadNcnnMatAsFloat32(mp.embedDim * mp.qdim, 0, 0, 0, 0);
                    var qB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                    var kW = br.ReadNcnnMatAsFloat32(mp.embedDim * mp.kdim, 0, 0, 0, 0);
                    var kB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                    var vW = br.ReadNcnnMatAsFloat32(mp.embedDim * mp.vdim, 0, 0, 0, 0);
                    var vB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                    var oW = br.ReadNcnnMatAsFloat32(mp.qdim * mp.embedDim, 0, 0, 0, 0);
                    var oB = br.ReadNcnnMatAsFloat32(mp.qdim, 0, 0, 0, 1);

                    mp.qW = new ComputeBuffer(qW.Length, sizeof(float), ComputeBufferType.Structured);
                    mp.qB = new ComputeBuffer(qB.Length, sizeof(float), ComputeBufferType.Structured);
                    mp.kW = new ComputeBuffer(kW.Length, sizeof(float), ComputeBufferType.Structured);
                    mp.kB = new ComputeBuffer(kB.Length, sizeof(float), ComputeBufferType.Structured);
                    mp.vW = new ComputeBuffer(vW.Length, sizeof(float), ComputeBufferType.Structured);
                    mp.vB = new ComputeBuffer(vB.Length, sizeof(float), ComputeBufferType.Structured);
                    mp.oW = new ComputeBuffer(oW.Length, sizeof(float), ComputeBufferType.Structured);
                    mp.oB = new ComputeBuffer(oB.Length, sizeof(float), ComputeBufferType.Structured);
                    mp.qW.SetData(qW); mp.qB.SetData(qB);
                    mp.kW.SetData(kW); mp.kB.SetData(kB);
                    mp.vW.SetData(vW); mp.vB.SetData(vB);
                    mp.oW.SetData(oW); mp.oB.SetData(oB);
                    _multiHeadAttention[layer.name] = mp;
                    continue;
                }
            }
        }

        public GFPGANInferResult Infer(RenderTexture inputPack4, int inputPacks, string inputBlobName = "data", ICollection<string> pinnedNames = null)
        {
            var bufferBlobs = new Dictionary<string, ComputeBuffer>(StringComparer.Ordinal);
            var tempBuffers = new List<ComputeBuffer>();
            return (GFPGANInferResult)InferCore(inputPack4, inputPacks, inputBlobName, pinnedNames, bufferBlobs, tempBuffers);
        }

        public RenderTexture ForwardPack4(RenderTexture inputPack4, int inputPacks, string inputBlobName = "data", ICollection<string> pinnedNames = null)
        {
            var result = InferCore(inputPack4, inputPacks, inputBlobName, pinnedNames, null, null);
            var rt = result.ExtractTexture("output");
            result.Dispose();
            return rt;
        }

        public GFPGANInferResult InferFromBuffers(Dictionary<string, ComputeBuffer> inputBuffers, string stopAfterTopName = null)
        {
            if (inputBuffers == null)
                throw new ArgumentNullException(nameof(inputBuffers));
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");

            var bufferBlobs = new Dictionary<string, ComputeBuffer>(inputBuffers, StringComparer.Ordinal);
            var blobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);
            var tempBuffers = new List<ComputeBuffer>();

            try
            {
                for (var li = 0; li < Model.layers.Count; li++)
                {
                    var l = Model.layers[li];

                    if (stopAfterTopName != null && bufferBlobs.ContainsKey(stopAfterTopName))
                        break;

                    if (string.Equals(l.type, "Input", StringComparison.Ordinal))
                        continue;

                    if (string.Equals(l.type, "MemoryData", StringComparison.Ordinal))
                    {
                        if (!_memoryData.TryGetValue(l.name, out var mp) || mp.data == null)
                            throw new InvalidOperationException("MemoryData not found: " + l.name);
                        bufferBlobs[l.topNames[0]] = mp.data;
                        continue;
                    }

                    if (string.Equals(l.type, "Embed", StringComparison.Ordinal))
                    {
                        if (!_embed.TryGetValue(l.name, out var ep) || ep.w == null)
                            throw new InvalidOperationException("Embed not found: " + l.name);

                        if (!bufferBlobs.TryGetValue(l.bottomNames[0], out var indicesBuf) || indicesBuf == null)
                            throw new InvalidOperationException("Embed input buffer not found: " + l.bottomNames[0]);

                        var words = indicesBuf.count;
                        var outBuf = new ComputeBuffer(words * ep.numOutput, sizeof(float), ComputeBufferType.Structured);
                        _ops.Embed(indicesBuf, words, ep.w, ep.b, ep.numOutput, ep.inputDim, ep.biasTerm != 0, outBuf);
                        bufferBlobs[l.topNames[0]] = outBuf;
                        continue;
                    }

                    if (string.Equals(l.type, "BinaryOp", StringComparison.Ordinal))
                    {
                        var opType = l.GetInt(0, 0);
                        var withScalar = l.GetInt(1, 0);
                        var scalarB = l.GetFloat(2, 0f);

                        if (!bufferBlobs.TryGetValue(l.bottomNames[0], out var aBuf) || aBuf == null)
                            throw new InvalidOperationException("BinaryOp input buffer not found: " + l.bottomNames[0]);

                        var total = aBuf.count;
                        var outBuf = new ComputeBuffer(total, sizeof(float), ComputeBufferType.Structured);

                        if (withScalar != 0)
                        {
                            _ops.BinaryOpScalarBuf(aBuf, scalarB, total, opType, outBuf);
                        }
                        else
                        {
                            if (!bufferBlobs.TryGetValue(l.bottomNames[1], out var bBuf) || bBuf == null)
                                throw new InvalidOperationException("BinaryOp input buffer not found: " + l.bottomNames[1]);
                            if (bBuf.count != total)
                                throw new InvalidOperationException("BinaryOp broadcast not supported: " + l.name);
                            _ops.BinaryOpBuf(aBuf, bBuf, total, opType, outBuf);
                        }

                        bufferBlobs[l.topNames[0]] = outBuf;
                        continue;
                    }

                    if (string.Equals(l.type, "Split", StringComparison.Ordinal))
                    {
                        if (bufferBlobs.TryGetValue(l.bottomNames[0], out var srcBuf) && srcBuf != null)
                        {
                            for (var t = 0; t < l.topNames.Length; t++)
                                bufferBlobs[l.topNames[t]] = srcBuf;
                        }
                        else if (blobs.TryGetValue(l.bottomNames[0], out var srcTex) && srcTex != null)
                        {
                            for (var t = 0; t < l.topNames.Length; t++)
                            {
                                blobs[l.topNames[t]] = srcTex;
                                srcTex.refs++;
                            }
                        }
                        else
                        {
                            throw new InvalidOperationException("Split source not found: " + l.bottomNames[0]);
                        }
                        continue;
                    }

                    if (string.Equals(l.type, "LayerNorm", StringComparison.Ordinal))
                    {
                        if (!_layerNorm.TryGetValue(l.name, out var lp))
                            throw new InvalidOperationException("LayerNorm not found: " + l.name);

                        if (!bufferBlobs.TryGetValue(l.bottomNames[0], out var srcBuf) || srcBuf == null)
                            throw new InvalidOperationException("LayerNorm input buffer not found: " + l.bottomNames[0]);

                        var rows = srcBuf.count / Mathf.Max(1, lp.affineSize);
                        var outBuf = new ComputeBuffer(srcBuf.count, sizeof(float), ComputeBufferType.Structured);
                        _ops.CopyBuf(srcBuf, outBuf, srcBuf.count);
                        _ops.LayerNorm2DInplace(outBuf, rows, lp.affineSize, lp.eps, lp.affine, lp.gamma, lp.beta);
                        bufferBlobs[l.topNames[0]] = outBuf;
                        continue;
                    }

                    if (string.Equals(l.type, "GroupNorm", StringComparison.Ordinal))
                    {
                        if (!_groupNorm.TryGetValue(l.name, out var gp))
                            throw new InvalidOperationException("GroupNorm not found: " + l.name);

                        if (!bufferBlobs.TryGetValue(l.bottomNames[0], out var srcBuf) || srcBuf == null)
                            throw new InvalidOperationException("GroupNorm input buffer not found: " + l.bottomNames[0]);

                        var outBuf = new ComputeBuffer(srcBuf.count, sizeof(float), ComputeBufferType.Structured);
                        _ops.CopyBuf(srcBuf, outBuf, srcBuf.count);
                        var spatial = srcBuf.count / Mathf.Max(1, gp.channels);
                        _ops.GroupNormInplace(outBuf, spatial, 1, gp.channels, gp.group, gp.eps, gp.affine, gp.gamma, gp.beta);
                        bufferBlobs[l.topNames[0]] = outBuf;
                        continue;
                    }

                    if (string.Equals(l.type, "MultiHeadAttention", StringComparison.Ordinal))
                    {
                        if (!_multiHeadAttention.TryGetValue(l.name, out var mp) || mp.qW == null)
                            throw new InvalidOperationException("MultiHeadAttention not found: " + l.name);

                        ComputeBuffer qBuf, kBuf, vBuf;
                        var qOk = bufferBlobs.TryGetValue(l.bottomNames[0], out qBuf) && qBuf != null;
                        var kOk = bufferBlobs.TryGetValue(l.bottomNames.Length > 1 ? l.bottomNames[1] : l.bottomNames[0], out kBuf) && kBuf != null;
                        var vOk = bufferBlobs.TryGetValue(l.bottomNames.Length > 2 ? l.bottomNames[2] : l.bottomNames[0], out vBuf) && vBuf != null;
                        if (!qOk || !kOk || !vOk)
                            throw new InvalidOperationException("MultiHeadAttention input buffer not found: " + l.name);

                        var srcLen = qBuf.count / Mathf.Max(1, mp.qdim);
                        var dstLen = kBuf.count / Mathf.Max(1, mp.kdim);

                        var qAff = new ComputeBuffer(srcLen * mp.embedDim, sizeof(float), ComputeBufferType.Structured);
                        var kAff = new ComputeBuffer(dstLen * mp.embedDim, sizeof(float), ComputeBufferType.Structured);
                        var vAff = new ComputeBuffer(dstLen * mp.embedDim, sizeof(float), ComputeBufferType.Structured);
                        _ops.InnerProduct2D(qBuf, srcLen, mp.qdim, mp.qW, mp.qB, mp.embedDim, qAff);
                        _ops.InnerProduct2D(kBuf, dstLen, mp.kdim, mp.kW, mp.kB, mp.embedDim, kAff);
                        _ops.InnerProduct2D(vBuf, dstLen, mp.vdim, mp.vW, mp.vB, mp.embedDim, vAff);

                        var qScaled = new ComputeBuffer(srcLen * mp.embedDim, sizeof(float), ComputeBufferType.Structured);
                        _ops.BinaryOpScalarBuf(qAff, mp.scale, qAff.count, 2, qScaled);
                        qAff.Dispose();

                        var ctx = new ComputeBuffer(srcLen * mp.embedDim, sizeof(float), ComputeBufferType.Structured);
                        _ops.MhaAttention(qScaled, kAff, vAff, srcLen, dstLen, mp.embedDim, mp.numHeads, 1f, ctx);
                        qScaled.Dispose();
                        kAff.Dispose();
                        vAff.Dispose();

                        var outBuf = new ComputeBuffer(srcLen * mp.qdim, sizeof(float), ComputeBufferType.Structured);
                        _ops.InnerProduct2D(ctx, srcLen, mp.embedDim, mp.oW, mp.oB, mp.qdim, outBuf);
                        ctx.Dispose();

                        bufferBlobs[l.topNames[0]] = outBuf;
                        continue;
                    }

                    if (string.Equals(l.type, "Reshape", StringComparison.Ordinal) ||
                        string.Equals(l.type, "Permute", StringComparison.Ordinal) ||
                        string.Equals(l.type, "ExpandDims", StringComparison.Ordinal))
                    {
                        if (bufferBlobs.TryGetValue(l.bottomNames[0], out var srcBuf) && srcBuf != null)
                        {
                            bufferBlobs[l.topNames[0]] = srcBuf;
                        }
                        else if (blobs.TryGetValue(l.bottomNames[0], out var srcTex) && srcTex != null)
                        {
                            blobs[l.topNames[0]] = srcTex;
                            srcTex.refs++;
                        }
                        else
                        {
                            throw new InvalidOperationException(l.type + " source not found: " + l.bottomNames[0]);
                        }
                        continue;
                    }

                    if (string.Equals(l.type, "Slice", StringComparison.Ordinal))
                    {
                        if (!bufferBlobs.TryGetValue(l.bottomNames[0], out var srcBuf) || srcBuf == null)
                            throw new InvalidOperationException("Slice source not found: " + l.bottomNames[0]);

                        var total = srcBuf.count;
                        for (var t = 0; t < l.topNames.Length; t++)
                        {
                            var start = t > 0 ? l.GetInt(-23329 + (t - 1) * 2, 0) : 0;
                            var end = t > 0 ? l.GetInt(-23328 + (t - 1) * 2, total) : total;
                            var len = end - start;
                            if (len <= 0)
                                throw new InvalidOperationException("Slice invalid range for " + l.name);
                            var sliceBuf = new ComputeBuffer(len, sizeof(float), ComputeBufferType.Structured);
                            _ops.CopyBuf(srcBuf, sliceBuf, len);
                            bufferBlobs[l.topNames[t]] = sliceBuf;
                        }
                        continue;
                    }

                    throw new InvalidOperationException("unsupported buffer layer type: " + l.type);
                }

                var result = new GFPGANInferResult(blobs, bufferBlobs, this);
                result.TempBuffers.AddRange(tempBuffers);
                return result;
            }
            catch
            {
                foreach (var kv in blobs)
                {
                    var tr = kv.Value;
                    if (tr != null && tr.owned && tr.t1 != null)
                        ReturnTempArray(tr.t1);
                }
                for (var i = 0; i < tempBuffers.Count; i++)
                {
                    try { tempBuffers[i]?.Dispose(); } catch { }
                }
                foreach (var kv in bufferBlobs)
                {
                    try { kv.Value?.Dispose(); } catch { }
                }
                bufferBlobs.Clear();
                throw;
            }
        }

        private InferResultBase InferCore(RenderTexture inputPack4, int inputPacks, string inputBlobName, ICollection<string> pinnedNames, Dictionary<string, ComputeBuffer> bufferBlobs, List<ComputeBuffer> tempBuffers)
        {
            var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
            var blobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);

            var inputRef = new TensorRef { t1 = inputPack4, w = inputPack4.width, h = inputPack4.height, packs = inputPacks, refs = 1, owned = false };
            blobs[inputBlobName] = inputRef;

            try
            {
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
                        if (blobs.TryGetValue(l.bottomNames[0], out var srcTex) && srcTex != null)
                        {
                            blobs[l.topNames[0]] = srcTex;
                            srcTex.refs++;
                        }
                        else if (bufferBlobs.TryGetValue(l.bottomNames[0], out var srcBuf) && srcBuf != null)
                        {
                            bufferBlobs[l.topNames[0]] = srcBuf;
                        }
                        else
                        {
                            throw new InvalidOperationException("blob not found for Reshape " + l.name);
                        }
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

                    if (string.Equals(l.type, "InnerProduct", StringComparison.Ordinal))
                    {
                        var ip = _innerProduct[l.name];
                        var src = Get(blobs, l.bottomNames[0]);
                        if (src.w * src.h * src.packs * 4 != ip.inFeatures)
                            throw new InvalidOperationException("InnerProduct inFeatures mismatch for " + l.name);

                        var inBuf = new ComputeBuffer(ip.inFeatures, sizeof(float), ComputeBufferType.Structured);
                        tempBuffers.Add(inBuf);
                        _ops.Pack4ToBufferCHW(src.t1, src.w, src.h, src.packs * 4, inBuf);

                        var outBuf = new ComputeBuffer(ip.outFeatures, sizeof(float), ComputeBufferType.Structured);
                        _ops.InnerProduct(inBuf, ip.inFeatures, ip.w, ip.b, ip.outFeatures, outBuf);
                        bufferBlobs[l.topNames[0]] = outBuf;
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

                    if (string.Equals(l.type, "MemoryData", StringComparison.Ordinal))
                    {
                        if (!_memoryData.TryGetValue(l.name, out var mp) || mp.data == null)
                            throw new InvalidOperationException("MemoryData not found: " + l.name);
                        bufferBlobs[l.topNames[0]] = mp.data;
                        continue;
                    }

                    if (string.Equals(l.type, "Embed", StringComparison.Ordinal))
                    {
                        if (!_embed.TryGetValue(l.name, out var ep) || ep.w == null)
                            throw new InvalidOperationException("Embed not found: " + l.name);

                        ComputeBuffer indicesBuf;
                        if (!bufferBlobs.TryGetValue(l.bottomNames[0], out indicesBuf) || indicesBuf == null)
                            throw new InvalidOperationException("Embed input buffer not found: " + l.bottomNames[0]);

                        var words = indicesBuf.count;
                        var outBuf = new ComputeBuffer(words * ep.numOutput, sizeof(float), ComputeBufferType.Structured);
                        _ops.Embed(indicesBuf, words, ep.w, ep.b, ep.numOutput, ep.inputDim, ep.biasTerm != 0, outBuf);
                        bufferBlobs[l.topNames[0]] = outBuf;
                        Consume(blobs, remaining, l.bottomNames, pinnedNames);
                        continue;
                    }

                    if (string.Equals(l.type, "LayerNorm", StringComparison.Ordinal))
                    {
                        if (!_layerNorm.TryGetValue(l.name, out var lp))
                            throw new InvalidOperationException("LayerNorm not found: " + l.name);

                        ComputeBuffer srcBuf;
                        if (!bufferBlobs.TryGetValue(l.bottomNames[0], out srcBuf) || srcBuf == null)
                            throw new InvalidOperationException("LayerNorm input buffer not found: " + l.bottomNames[0]);

                        var outBuf = new ComputeBuffer(srcBuf.count, sizeof(float), ComputeBufferType.Structured);
                        _ops.CopyBuf(srcBuf, outBuf, srcBuf.count);
                        _ops.LayerNorm2DInplace(outBuf, srcBuf.count / Mathf.Max(1, lp.affineSize), lp.affineSize, lp.eps, lp.affine, lp.gamma, lp.beta);
                        bufferBlobs[l.topNames[0]] = outBuf;
                        Consume(blobs, remaining, l.bottomNames, pinnedNames);
                        continue;
                    }

                    if (string.Equals(l.type, "GroupNorm", StringComparison.Ordinal))
                    {
                        if (!_groupNorm.TryGetValue(l.name, out var gp))
                            throw new InvalidOperationException("GroupNorm not found: " + l.name);

                        ComputeBuffer srcBuf;
                        if (!bufferBlobs.TryGetValue(l.bottomNames[0], out srcBuf) || srcBuf == null)
                            throw new InvalidOperationException("GroupNorm input buffer not found: " + l.bottomNames[0]);

                        var outBuf = new ComputeBuffer(srcBuf.count, sizeof(float), ComputeBufferType.Structured);
                        _ops.CopyBuf(srcBuf, outBuf, srcBuf.count);
                        var spatial = srcBuf.count / Mathf.Max(1, gp.channels);
                        _ops.GroupNormInplace(outBuf, spatial, 1, gp.channels, gp.group, gp.eps, gp.affine, gp.gamma, gp.beta);
                        bufferBlobs[l.topNames[0]] = outBuf;
                        Consume(blobs, remaining, l.bottomNames, pinnedNames);
                        continue;
                    }

                    if (string.Equals(l.type, "MultiHeadAttention", StringComparison.Ordinal))
                    {
                        if (!_multiHeadAttention.TryGetValue(l.name, out var mp) || mp.qW == null)
                            throw new InvalidOperationException("MultiHeadAttention not found: " + l.name);

                        ComputeBuffer qBuf, kBuf, vBuf;
                        var qOk = bufferBlobs.TryGetValue(l.bottomNames[0], out qBuf) && qBuf != null;
                        var kOk = bufferBlobs.TryGetValue(l.bottomNames.Length > 1 ? l.bottomNames[1] : l.bottomNames[0], out kBuf) && kBuf != null;
                        var vOk = bufferBlobs.TryGetValue(l.bottomNames.Length > 2 ? l.bottomNames[2] : l.bottomNames[0], out vBuf) && vBuf != null;
                        if (!qOk || !kOk || !vOk)
                            throw new InvalidOperationException("MultiHeadAttention input buffer not found: " + l.name);

                        var srcLen = qBuf.count / Mathf.Max(1, mp.qdim);
                        var dstLen = kBuf.count / Mathf.Max(1, mp.kdim);

                        var qAff = new ComputeBuffer(srcLen * mp.embedDim, sizeof(float), ComputeBufferType.Structured);
                        var kAff = new ComputeBuffer(dstLen * mp.embedDim, sizeof(float), ComputeBufferType.Structured);
                        var vAff = new ComputeBuffer(dstLen * mp.embedDim, sizeof(float), ComputeBufferType.Structured);
                        _ops.InnerProduct2D(qBuf, srcLen, mp.qdim, mp.qW, mp.qB, mp.embedDim, qAff);
                        _ops.InnerProduct2D(kBuf, dstLen, mp.kdim, mp.kW, mp.kB, mp.embedDim, kAff);
                        _ops.InnerProduct2D(vBuf, dstLen, mp.vdim, mp.vW, mp.vB, mp.embedDim, vAff);

                        var qScaled = new ComputeBuffer(srcLen * mp.embedDim, sizeof(float), ComputeBufferType.Structured);
                        _ops.BinaryOpScalarBuf(qAff, mp.scale, qAff.count, 2, qScaled);
                        qAff.Dispose();

                        var ctx = new ComputeBuffer(srcLen * mp.embedDim, sizeof(float), ComputeBufferType.Structured);
                        _ops.MhaAttention(qScaled, kAff, vAff, srcLen, dstLen, mp.embedDim, mp.numHeads, 1f, ctx);
                        qScaled.Dispose();
                        kAff.Dispose();
                        vAff.Dispose();

                        var outBuf = new ComputeBuffer(srcLen * mp.qdim, sizeof(float), ComputeBufferType.Structured);
                        _ops.InnerProduct2D(ctx, srcLen, mp.embedDim, mp.oW, mp.oB, mp.qdim, outBuf);
                        ctx.Dispose();

                        bufferBlobs[l.topNames[0]] = outBuf;
                        Consume(blobs, remaining, l.bottomNames, pinnedNames);
                        continue;
                    }

                    throw new InvalidOperationException("unsupported layer type: " + l.type);
                }

                var result = bufferBlobs != null
                    ? (InferResultBase)new GFPGANInferResult(blobs, bufferBlobs, this)
                    : new RealEsrganInferResult(blobs, this);
                if (bufferBlobs != null)
                    ((GFPGANInferResult)result).TempBuffers.AddRange(tempBuffers);
                return result;
            }
            catch
            {
                var visited = new HashSet<TensorRef>();
                foreach (var kv in blobs)
                {
                    var tr = kv.Value;
                    if (tr == null || !visited.Add(tr))
                        continue;
                    if (tr.owned && tr.t1 != null)
                        ReturnTempArray(tr.t1);
                }
                if (tempBuffers != null)
                {
                    for (var i = 0; i < tempBuffers.Count; i++)
                    {
                        try { tempBuffers[i]?.Dispose(); } catch { }
                    }
                }
                if (bufferBlobs != null)
                {
                    foreach (var kv in bufferBlobs)
                    {
                        try { kv.Value?.Dispose(); } catch { }
                    }
                    bufferBlobs.Clear();
                }
                throw;
            }
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
            foreach (var kv in _innerProduct)
                kv.Value?.Dispose();
            _innerProduct.Clear();
            foreach (var kv in _memoryData)
                kv.Value?.Dispose();
            _memoryData.Clear();
            foreach (var kv in _embed)
                kv.Value?.Dispose();
            _embed.Clear();
            foreach (var kv in _layerNorm)
                kv.Value?.Dispose();
            _layerNorm.Clear();
            foreach (var kv in _groupNorm)
                kv.Value?.Dispose();
            _groupNorm.Clear();
            foreach (var kv in _multiHeadAttention)
                kv.Value?.Dispose();
            _multiHeadAttention.Clear();
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