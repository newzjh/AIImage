using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnRepro2 : IDisposable
    {
        private static readonly HashSet<string> CodeFormerSftMulLayers = new HashSet<string>(StringComparer.Ordinal)
        {
            "Mul_581",
            "Mul_687",
            "Mul_794",
            "Mul_900"
        };

        private static readonly HashSet<string> CodeFormerSftAddLayers = new HashSet<string>(StringComparer.Ordinal)
        {
            "Add_582",
            "Add_688",
            "Add_795",
            "Add_901"
        };

        private static readonly HashSet<string> CodeFormerSftResidualLayers = new HashSet<string>(StringComparer.Ordinal)
        {
            "Add_585",
            "Add_691",
            "Add_798",
            "Add_904"
        };

        internal readonly struct BufferShape
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
            public ComputeBuffer rawWeight;
            public ComputeBuffer rawBias;

            public void Dispose()
            {
                try { packedWeight4?.Dispose(); } catch { }
                try { packedBias4?.Dispose(); } catch { }
                try { packedWeightTm23?.Dispose(); } catch { }
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

        public sealed class InferResult : IDisposable
        {
            private readonly Dictionary<string, TensorRef> _textureBlobs;
            private readonly Dictionary<string, BufferShape> _textureShapes;
            private readonly Dictionary<string, ComputeBuffer> _bufferBlobs;
            private readonly Dictionary<string, NcnnTensorBuffer> _bufferViews;
            private readonly List<IDisposable> _tempOwned;
            private readonly NcnnRepro2 _owner;
            private readonly HashSet<TensorRef> _visitedTextures = new HashSet<TensorRef>();

            internal InferResult(
                Dictionary<string, TensorRef> textureBlobs,
                Dictionary<string, BufferShape> textureShapes,
                Dictionary<string, ComputeBuffer> bufferBlobs,
                Dictionary<string, NcnnTensorBuffer> bufferViews,
                List<IDisposable> tempOwned,
                NcnnRepro2 owner)
            {
                _textureBlobs = textureBlobs;
                _textureShapes = textureShapes;
                _bufferBlobs = bufferBlobs;
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
                    try { owned?.Dispose(); } catch { }
                }

                var seenBuffers = new HashSet<ComputeBuffer>();
                foreach (var kv in _bufferBlobs)
                {
                    var buf = kv.Value;
                    if (buf == null || !seenBuffers.Add(buf))
                        continue;
                    try { buf.Dispose(); } catch { }
                }
                _bufferBlobs.Clear();
                _bufferViews.Clear();
                _tempOwned.Clear();
            }
        }

        public NcnnParamModel Model { get; private set; }

        private readonly Dictionary<string, ConvPack> _conv = new Dictionary<string, ConvPack>(StringComparer.Ordinal);
        private readonly Dictionary<string, InnerProductPack> _innerProduct = new Dictionary<string, InnerProductPack>(StringComparer.Ordinal);
        private readonly Dictionary<string, MemoryDataPack> _memoryData = new Dictionary<string, MemoryDataPack>(StringComparer.Ordinal);
        private readonly Dictionary<string, EmbedPack> _embed = new Dictionary<string, EmbedPack>(StringComparer.Ordinal);
        private readonly Dictionary<string, LayerNormPack> _layerNorm = new Dictionary<string, LayerNormPack>(StringComparer.Ordinal);
        private readonly Dictionary<string, GroupNormPack> _groupNorm = new Dictionary<string, GroupNormPack>(StringComparer.Ordinal);
        private readonly Dictionary<string, MultiHeadAttentionPack> _multiHeadAttention = new Dictionary<string, MultiHeadAttentionPack>(StringComparer.Ordinal);
        private Dictionary<string, int> _blobUseCount;
        private readonly Dictionary<RtKey, Stack<RenderTexture>> _rtPool = new Dictionary<RtKey, Stack<RenderTexture>>();

        private readonly NcnnOps _ops;
        private bool _useTempPool;
        private int _maxPooledPerShape = 2;

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

        public bool EnableWinograd23 { get; set; }
        public float CodeFormerSftMulScale { get; set; } = 1f;
        public float CodeFormerSftAddScale { get; set; } = 1f;
        public bool CodeFormerBypassSftMul { get; set; }
        public string CodeFormerTargetSftMulLayer { get; set; }
        public string CodeFormerTargetSftAddLayer { get; set; }
        public string CodeFormerTargetSftResidualLayer { get; set; }

        public NcnnRepro2(NcnnOps ops)
        {
            _ops = ops ?? throw new ArgumentNullException(nameof(ops));
        }

        public void LoadModel(string paramText, NcnnBinReader br)
        {
            Release();

            Model = NcnnParamParser.Parse(paramText);
            _blobUseCount = BuildBlobUseCount(Model);

            foreach (var layer in Model.layers)
            {
                if (string.Equals(layer.type, "Convolution", StringComparison.Ordinal)
                    || string.Equals(layer.type, "ConvolutionDepthWise", StringComparison.Ordinal))
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
                    pack.isDepthWise = string.Equals(layer.type, "ConvolutionDepthWise", StringComparison.Ordinal);

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

                    var tag = br.ReadInt32();
                    if (tag != 0x01306B47)
                        throw new InvalidOperationException("unexpected weight tag at " + br.Position + ": 0x" + tag.ToString("X8", CultureInfo.InvariantCulture));

                    var w = br.ReadFp16ArrayAsFloat32(pack.weightSize);
                    var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];

                    pack.rawWeight = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                    pack.rawBias = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                    pack.rawWeight.SetData(w);
                    pack.rawBias.SetData(b);

                    if (!pack.useBufferPath && !pack.isDepthWise && pack.group == 1)
                    {
                        var w4 = PackWeightsToO4I4K(w, pack.outC, pack.inC, pack.kernelW, pack.outPacks, pack.inPacks);
                        var b4 = PackBiasToO4(b, pack.outC, pack.outPacks);
                        pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                        pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                        pack.packedWeight4.SetData(w4);
                        pack.packedBias4.SetData(b4);

                        if (pack.kernelW == 3
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
                    ip.inFeatures = ip.outFeatures > 0 ? ip.weightSize / ip.outFeatures : 0;

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
                    continue;
                }

                if (string.Equals(layer.type, "Embed", StringComparison.Ordinal))
                {
                    var ep = new EmbedPack();
                    ep.numOutput = layer.GetInt(0, 0);
                    ep.inputDim = layer.GetInt(1, 0);
                    ep.biasTerm = layer.GetInt(2, 0);
                    ep.weightSize = layer.GetInt(3, 0);

                    var embedFlagPos = br.Position;
                    var flag = br.ReadUInt32();
                    var sum = (byte)(flag & 0xFF) + (byte)((flag >> 8) & 0xFF) + (byte)((flag >> 16) & 0xFF) + (byte)((flag >> 24) & 0xFF);
                    float[] w;
                    if (flag == 0x01306B47)
                    {
                        w = br.ReadFp16ArrayAsFloat32(ep.weightSize);
                    }
                    else if (sum == 0)
                    {
                        w = br.ReadFloat32Array(ep.weightSize);
                    }
                    else
                    {
                        throw new InvalidOperationException("Embed unexpected flag at " + embedFlagPos + ": 0x" + flag.ToString("X8", CultureInfo.InvariantCulture));
                    }

                    ep.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                    ep.w.SetData(w);

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
                    mp.qdim = mp.embedDim > 0 ? mp.weightDataSize / Mathf.Max(1, mp.embedDim) : 0;

                    var qW = br.ReadNcnnMatAsFloat32(mp.embedDim * mp.qdim, 0, 0, 0, 0);
                    var qB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                    var kW = br.ReadNcnnMatAsFloat32(mp.embedDim * mp.kdim, 0, 0, 0, 0);
                    var kB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                    var vW = br.ReadNcnnMatAsFloat32(mp.embedDim * mp.vdim, 0, 0, 0, 0);
                    var vB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                    var oW = br.ReadNcnnMatAsFloat32(mp.qdim * mp.embedDim, 0, 0, 0, 0);
                    var oB = br.ReadNcnnMatAsFloat32(mp.qdim, 0, 0, 0, 1);

                    mp.qW = NewBuffer(qW);
                    mp.qB = NewBuffer(qB);
                    mp.kW = NewBuffer(kW);
                    mp.kB = NewBuffer(kB);
                    mp.vW = NewBuffer(vW);
                    mp.vB = NewBuffer(vB);
                    mp.oW = NewBuffer(oW);
                    mp.oB = NewBuffer(oB);
                    _multiHeadAttention[layer.name] = mp;
                }
            }
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

        public InferResult InferWithMultiInputs(
            Dictionary<string, RenderTexture> textureInputs,
            Dictionary<string, NcnnTensorBuffer> bufferInputs,
            ICollection<string> pinnedNames = null)
        {
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");
            if ((textureInputs == null || textureInputs.Count == 0) && (bufferInputs == null || bufferInputs.Count == 0))
                throw new ArgumentNullException(nameof(textureInputs));

            var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
            var textureBlobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);
            var textureShapes = new Dictionary<string, BufferShape>(StringComparer.Ordinal);
            var bufferBlobs = new Dictionary<string, ComputeBuffer>(StringComparer.Ordinal);
            var bufferViews = new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal);
            var tempOwned = new List<IDisposable>();

            if (textureInputs != null)
            {
                foreach (var kv in textureInputs)
                {
                    if (kv.Value == null)
                        throw new ArgumentNullException("textureInputs[\"" + kv.Key + "\"]");
                    var rt = kv.Value;
                    var packs = rt.volumeDepth > 0 ? rt.volumeDepth : 1;
                    var useCount = _blobUseCount.TryGetValue(kv.Key, out var c) ? c : 1;
                    var logicalChannels = ResolveInputLogicalChannels(kv.Key, packs * 4);
                    textureBlobs[kv.Key] = new TensorRef
                    {
                        texture = rt,
                        width = rt.width,
                        height = rt.height,
                        packs = packs,
                        refs = useCount,
                        owned = false
                    };
                    textureShapes[kv.Key] = new BufferShape(3, rt.width, rt.height, 1, logicalChannels);
                }
            }

            if (bufferInputs != null)
            {
                foreach (var kv in bufferInputs)
                {
                    if (kv.Value == null || kv.Value.buffer == null)
                        throw new ArgumentNullException("bufferInputs[\"" + kv.Key + "\"]");
                    bufferBlobs[kv.Key] = kv.Value.buffer;
                    bufferViews[kv.Key] = kv.Value;
                }
            }

            for (var li = 0; li < Model.layers.Count; li++)
            {
                var layer = Model.layers[li];
                if (string.Equals(layer.type, "Input", StringComparison.Ordinal))
                    continue;

                if (string.Equals(layer.type, "Split", StringComparison.Ordinal))
                {
                    if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var srcBuf) && srcBuf != null)
                    {
                        var srcTensor = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                        for (var i = 0; i < layer.topNames.Length; i++)
                        {
                            bufferBlobs[layer.topNames[i]] = srcBuf;
                            if (srcTensor != null)
                                bufferViews[layer.topNames[i]] = srcTensor;
                        }
                    }
                    else
                    {
                        var src = GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                        var shape = GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                        for (var i = 0; i < layer.topNames.Length; i++)
                        {
                            textureBlobs[layer.topNames[i]] = src;
                            textureShapes[layer.topNames[i]] = shape;
                            src.refs++;
                        }
                    }

                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "MemoryData", StringComparison.Ordinal))
                {
                    if (!_memoryData.TryGetValue(layer.name, out var mp) || mp.data == null)
                        throw new InvalidOperationException("MemoryData not found: " + layer.name);
                    bufferBlobs[layer.topNames[0]] = mp.data;
                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(mp.data, mp.dims, mp.w, mp.h, mp.d, mp.c, false);
                    continue;
                }

                if (string.Equals(layer.type, "Embed", StringComparison.Ordinal))
                {
                    if (!_embed.TryGetValue(layer.name, out var ep) || ep.w == null)
                        throw new InvalidOperationException("Embed not found: " + layer.name);
                    if (!bufferBlobs.TryGetValue(layer.bottomNames[0], out var indicesBuf) || indicesBuf == null)
                        throw new InvalidOperationException("Embed input buffer not found: " + layer.bottomNames[0]);

                    var words = indicesBuf.count;
                    var outBuf = new ComputeBuffer(words * ep.numOutput, sizeof(float), ComputeBufferType.Structured);
                    _ops.Embed(indicesBuf, words, ep.w, ep.b, ep.numOutput, ep.inputDim, ep.biasTerm != 0, outBuf);
                    bufferBlobs[layer.topNames[0]] = outBuf;
                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, 2, ep.numOutput, words, 1, 1, false);
                    tempOwned.Add(outBuf);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Reshape", StringComparison.Ordinal))
                {
                    if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var reshapeBuf) && reshapeBuf != null)
                    {
                        bufferBlobs[layer.topNames[0]] = reshapeBuf;
                        var srcTensor = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                        if (srcTensor != null)
                            bufferViews[layer.topNames[0]] = ResolveReshapeTensor(srcTensor, layer);
                    }
                    else
                    {
                        var src = GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                        var srcShape = GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                        var outShape = ResolveReshapeShape(srcShape, layer);

                        // If logical channels do not fill whole pack4 lanes, keeping the texture view
                        // would preserve padded channels and break later buffer consumers such as Permute.
                        if (srcShape.dims == 3 && (srcShape.c % 4) != 0)
                        {
                            var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                            if (srcBuf == null)
                                throw new InvalidOperationException("Reshape source not found: " + layer.bottomNames[0]);
                            bufferBlobs[layer.topNames[0]] = srcBuf;
                            if (TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews) is { } srcTensor)
                                bufferViews[layer.topNames[0]] = ResolveReshapeTensor(srcTensor, layer);
                        }
                        else
                        {
                            textureBlobs[layer.topNames[0]] = src;
                            textureShapes[layer.topNames[0]] = outShape;
                            src.refs++;
                        }
                    }

                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "ShuffleChannel", StringComparison.Ordinal))
                {
                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    var srcView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                    if (srcBuf == null || srcView == null)
                        throw new InvalidOperationException("ShuffleChannel source not found: " + layer.name);

                    var outBuf = ShuffleChannelCpu(srcBuf, srcView, layer.GetInt(0, 1), layer.GetInt(1, 0) != 0);
                    bufferBlobs[layer.topNames[0]] = outBuf.buffer;
                    bufferViews[layer.topNames[0]] = outBuf;
                    tempOwned.Add(outBuf);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Permute", StringComparison.Ordinal))
                {
                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    if (srcBuf == null)
                        throw new InvalidOperationException("Permute source not found: " + layer.bottomNames[0]);

                    var srcTensor = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                    if (srcTensor == null)
                        throw new InvalidOperationException("Permute shape not resolved: " + layer.name);

                    var orderType = layer.GetInt(0, 0);
                    var dims = Mathf.Clamp(srcTensor.dims, 2, 4);
                    var axes = ResolvePermuteAxes(dims, orderType, layer.name);
                    var outShape = ResolvePermuteShape(srcTensor, dims, axes);
                    var outBuf = new ComputeBuffer(outShape.w * outShape.h * outShape.d * outShape.c, sizeof(float), ComputeBufferType.Structured);
                    _ops.Permute(srcBuf, dims, srcTensor.w, srcTensor.h, srcTensor.d, srcTensor.c, orderType, outBuf);

                    bufferBlobs[layer.topNames[0]] = outBuf;
                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, outShape.dims, outShape.w, outShape.h, outShape.d, outShape.c, false);
                    tempOwned.Add(outBuf);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Crop", StringComparison.Ordinal))
                {
                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    var srcView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                    if (srcBuf == null || srcView == null)
                        throw new InvalidOperationException("Crop source not found: " + layer.name);

                    var cropResult = ApplyCropSlices(srcBuf, srcView, layer, tempOwned);
                    bufferBlobs[layer.topNames[0]] = cropResult.buffer;
                    bufferViews[layer.topNames[0]] = cropResult;
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Reduction", StringComparison.Ordinal))
                {
                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    if (srcBuf == null)
                        throw new InvalidOperationException("Reduction source not found: " + layer.bottomNames[0]);

                    var srcTensor = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                    var reduceAll = layer.GetInt(1, 1) != 0;
                    var coeff = layer.GetFloat(2, 1f);
                    var axes = layer.GetInts(-23303, null);
                    var keepDims = layer.GetInt(4, 0) != 0;

                    if (srcTensor == null || srcTensor.dims != 2)
                        throw new InvalidOperationException("Reduction currently expects dims=2 buffer input: " + layer.name);

                    if (reduceAll)
                    {
                        var outBufAll = new ComputeBuffer(1, sizeof(float), ComputeBufferType.Structured);
                        _ops.ReductionBuf(srcBuf, srcBuf.count, 1, layer.GetInt(0, 0), coeff, outBufAll);
                        bufferBlobs[layer.topNames[0]] = outBufAll;
                        bufferViews[layer.topNames[0]] = keepDims
                            ? new NcnnTensorBuffer(outBufAll, 2, 1, 1, 1, 1, false)
                            : new NcnnTensorBuffer(outBufAll, 1, 1, 1, 1, 1, false);
                        tempOwned.Add(outBufAll);
                        Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                        continue;
                    }

                    if (axes == null || axes.Length == 0)
                        throw new InvalidOperationException("Reduction axes missing: " + layer.name);
                    if (axes.Length != 1)
                        throw new InvalidOperationException("Reduction axes length > 1 not supported yet: " + layer.name);

                    var axis = axes[0];
                    var positiveAxis = axis < 0 ? axis + srcTensor.dims : axis;
                    int reduceElems;
                    int outCount;
                    NcnnTensorBuffer outView;
                    if (positiveAxis == 1)
                    {
                        reduceElems = srcTensor.w;
                        outCount = srcTensor.h;
                        outView = keepDims ? null : null;
                    }
                    else if (positiveAxis == 0)
                    {
                        reduceElems = srcTensor.h;
                        outCount = srcTensor.w;
                        var tempTranspose = new ComputeBuffer(srcTensor.buffer.count, sizeof(float), ComputeBufferType.Structured);
                        _ops.Permute(srcBuf, 2, srcTensor.w, srcTensor.h, 1, 1, 1, tempTranspose);
                        srcBuf = tempTranspose;
                        tempOwned.Add(tempTranspose);
                        outView = keepDims ? null : null;
                    }
                    else
                    {
                        throw new InvalidOperationException("Reduction axis not supported for dims=2: " + axis + " | " + layer.name);
                    }

                    var outBuf = new ComputeBuffer(outCount, sizeof(float), ComputeBufferType.Structured);
                    _ops.ReductionBuf(srcBuf, reduceElems, outCount, layer.GetInt(0, 0), coeff, outBuf);
                    bufferBlobs[layer.topNames[0]] = outBuf;
                    bufferViews[layer.topNames[0]] = positiveAxis == 1
                        ? (keepDims
                            ? new NcnnTensorBuffer(outBuf, 2, 1, srcTensor.h, 1, 1, false)
                            : new NcnnTensorBuffer(outBuf, 1, srcTensor.h, 1, 1, 1, false))
                        : (keepDims
                            ? new NcnnTensorBuffer(outBuf, 2, srcTensor.w, 1, 1, 1, false)
                            : new NcnnTensorBuffer(outBuf, 1, srcTensor.w, 1, 1, 1, false));
                    tempOwned.Add(outBuf);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Concat", StringComparison.Ordinal))
                {
                    var first = GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                    var axis = layer.GetInt(0, 0);
                    if (axis != 0)
                        throw new InvalidOperationException("Concat only supports channel axis for texture tensors: " + layer.name);

                    var totalPacks = 0;
                    var totalLogicalChannels = 0;
                    for (var i = 0; i < layer.bottomNames.Length; i++)
                    {
                        var tr = GetOrMaterializeTexture(layer.bottomNames[i], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                        if (tr.width != first.width || tr.height != first.height)
                            throw new InvalidOperationException("Concat shape mismatch: " + layer.name);
                        totalPacks += tr.packs;
                        var logicalShape = GetTextureShape(textureShapes, tr, layer.bottomNames[i]);
                        totalLogicalChannels += logicalShape.c;
                    }

                    var outRt = RentTempArray(first.width, first.height, totalPacks, RenderTextureFormat.ARGBHalf);
                    var packOffset = 0;
                    for (var i = 0; i < layer.bottomNames.Length; i++)
                    {
                        var part = GetOrMaterializeTexture(layer.bottomNames[i], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                        _ops.CopyPack4(part.texture, 0, outRt, packOffset, part.packs);
                        packOffset += part.packs;
                    }

                    textureBlobs[layer.topNames[0]] = new TensorRef
                    {
                        texture = outRt,
                        width = first.width,
                        height = first.height,
                        packs = totalPacks,
                        refs = 1,
                        owned = true
                    };
                    textureShapes[layer.topNames[0]] = new BufferShape(3, first.width, first.height, 1, totalLogicalChannels);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Padding", StringComparison.Ordinal))
                {
                    var src = GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                    var top = layer.GetInt(0, 0);
                    var bottom = layer.GetInt(1, 0);
                    var left = layer.GetInt(2, 0);
                    var right = layer.GetInt(3, 0);
                    var type = layer.GetInt(4, 0);
                    var value = layer.GetFloat(5, 0f);

                    var outRt = RentTempArray(src.width + left + right, src.height + top + bottom, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.PaddingPack4(src.texture, src.packs, left, right, top, bottom, type, new Vector4(value, value, value, value), outRt);
                    textureBlobs[layer.topNames[0]] = new TensorRef
                    {
                        texture = outRt,
                        width = outRt.width,
                        height = outRt.height,
                        packs = src.packs,
                        refs = 1,
                        owned = true
                    };
                    var srcShape = GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                    textureShapes[layer.topNames[0]] = new BufferShape(3, outRt.width, outRt.height, 1, srcShape.c);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Pooling", StringComparison.Ordinal))
                {
                    var src = GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                    var kernelW = layer.GetInt(1, 0);
                    var kernelH = layer.GetInt(11, kernelW);
                    var strideW = layer.GetInt(2, 1);
                    var strideH = layer.GetInt(12, strideW);
                    var padLeft = layer.GetInt(3, 0);
                    var padTop = layer.GetInt(13, padLeft);
                    var outW = Mathf.Max(1, (src.width + padLeft * 2 - kernelW) / strideW + 1);
                    var outH = Mathf.Max(1, (src.height + padTop * 2 - kernelH) / strideH + 1);

                    var outRt = RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.PoolingPack4(src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, layer.GetInt(0, 0), outRt);
                    textureBlobs[layer.topNames[0]] = new TensorRef
                    {
                        texture = outRt,
                        width = outW,
                        height = outH,
                        packs = src.packs,
                        refs = 1,
                        owned = true
                    };
                    var poolSrcShape = GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                    textureShapes[layer.topNames[0]] = new BufferShape(3, outW, outH, 1, poolSrcShape.c);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Softmax", StringComparison.Ordinal))
                {
                    if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var softBuf) && softBuf != null)
                    {
                        var srcTensor = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                        var rows = srcTensor != null && srcTensor.dims == 2 ? srcTensor.h : 1;
                        var cols = srcTensor != null && srcTensor.dims == 2 ? srcTensor.w : softBuf.count;
                        var outBuf = new ComputeBuffer(softBuf.count, sizeof(float), ComputeBufferType.Structured);
                        _ops.Softmax2D(softBuf, outBuf, rows, cols);
                        bufferBlobs[layer.topNames[0]] = outBuf;
                        if (srcTensor != null)
                            bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcTensor.dims, srcTensor.w, srcTensor.h, srcTensor.d, srcTensor.c, false);
                        tempOwned.Add(outBuf);
                    }
                    else
                    {
                        var src = GetTexture(textureBlobs, layer.bottomNames[0]);
                        var outRt = RentTempArray(src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                        _ops.SoftmaxChannelPack4(src.texture, src.packs, outRt);
                        textureBlobs[layer.topNames[0]] = new TensorRef
                        {
                            texture = outRt,
                            width = src.width,
                            height = src.height,
                            packs = src.packs,
                            refs = 1,
                            owned = true
                        };
                        var softShape = GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                        textureShapes[layer.topNames[0]] = new BufferShape(3, src.width, src.height, 1, softShape.c);
                    }
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "InnerProduct", StringComparison.Ordinal))
                {
                    if (!_innerProduct.TryGetValue(layer.name, out var ip))
                        throw new InvalidOperationException("InnerProduct not found: " + layer.name);

                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    if (srcBuf == null)
                        throw new InvalidOperationException("InnerProduct source not found: " + layer.bottomNames[0]);

                    var srcTensor = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                    var rows = srcTensor != null && srcTensor.dims == 2 && srcTensor.w == ip.inFeatures ? srcTensor.h : 1;
                    var outBuf = new ComputeBuffer(ip.outFeatures * rows, sizeof(float), ComputeBufferType.Structured);
                    if (rows > 1)
                        _ops.InnerProduct2D(srcBuf, rows, ip.inFeatures, ip.w, ip.b, ip.outFeatures, outBuf);
                    else
                        _ops.InnerProduct(srcBuf, ip.inFeatures, ip.w, ip.b, ip.outFeatures, outBuf);

                    bufferBlobs[layer.topNames[0]] = outBuf;
                    bufferViews[layer.topNames[0]] = rows > 1
                        ? new NcnnTensorBuffer(outBuf, 2, ip.outFeatures, rows, 1, 1, false)
                        : new NcnnTensorBuffer(outBuf, 1, ip.outFeatures, 1, 1, 1, false);
                    tempOwned.Add(outBuf);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Convolution", StringComparison.Ordinal)
                    || string.Equals(layer.type, "ConvolutionDepthWise", StringComparison.Ordinal))
                {
                    if (!_conv.TryGetValue(layer.name, out var conv))
                        throw new InvalidOperationException("Convolution not found: " + layer.name);

                    if (conv.useBufferPath)
                    {
                        var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                        var srcTensor = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                        if (srcBuf == null || srcTensor == null || srcTensor.dims != 3)
                            throw new InvalidOperationException("Buffer convolution expects dims=3 tensor input: " + layer.name);

                        var outW = ComputeConvOut(srcTensor.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
                        var outH = ComputeConvOut(srcTensor.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
                        var outTensor = new NcnnTensorBuffer(outW, outH, conv.outC);
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
                                outTensor);
                        }
                        else
                        {
                            _ops.Conv3x3(srcTensor, conv.rawWeight, conv.rawBias, conv.outC, conv.strideW, conv.padLeft, conv.activationType, conv.activationSlope, outTensor);
                        }

                        bufferBlobs[layer.topNames[0]] = outTensor.buffer;
                        bufferViews[layer.topNames[0]] = outTensor;
                        tempOwned.Add(outTensor);
                        Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                        continue;
                    }

                    TensorRef src;
                    RenderTexture tempInputTex = null;
                        if (textureBlobs.TryGetValue(layer.bottomNames[0], out src) && src != null && src.texture != null)
                        {
                        }
                        else
                    {
                        if (!bufferBlobs.TryGetValue(layer.bottomNames[0], out var convInputBuf) || convInputBuf == null)
                            throw new InvalidOperationException("Convolution source not found: " + layer.name);
                        var convInputView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                        if (convInputView == null || convInputView.dims != 3)
                            throw new InvalidOperationException("Convolution texture path expects dims=3 buffer input: " + layer.name);

                        var inPacks = (convInputView.c + 3) / 4;
                        tempInputTex = RentTempArray(convInputView.w, convInputView.h, inPacks, RenderTextureFormat.ARGBHalf);
                        _ops.FillPack4FromBufferCHW(convInputBuf, convInputView.w, convInputView.h, convInputView.c, tempInputTex);
                        src = new TensorRef
                        {
                            texture = tempInputTex,
                            width = convInputView.w,
                            height = convInputView.h,
                            packs = inPacks,
                            refs = 1,
                            owned = false
                        };
                    }
                    var outWTex = ComputeConvOut(src.width, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
                    var outHTex = ComputeConvOut(src.height, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
                    var outRt = RentTempArray(outWTex, outHTex, conv.outPacks, RenderTextureFormat.ARGBHalf);

                    if (conv.kernelW == 1 && conv.kernelH == 1)
                    {
                        if (src.width != outWTex || src.height != outHTex)
                            throw new InvalidOperationException("Conv1x1 texture path does not support spatial resize: " + layer.name);
                        _ops.Conv1x1Pack4(src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.activationType, conv.activationSlope, outRt);
                    }
                    else if (conv.kernelW == 3
                             && conv.kernelH == 3
                             && conv.strideW == 1
                             && conv.strideH == 1
                             && conv.padLeft == conv.padRight
                             && conv.padTop == conv.padBottom
                             && conv.padLeft == conv.padTop)
                    {
                        if (EnableWinograd23 && conv.useWinograd23)
                            _ops.Conv3x3Pack4Winograd23(src.texture, conv.inPacks, conv.packedWeightTm23, conv.packedBias4, conv.outPacks, conv.biasTerm, conv.activationType, conv.activationSlope, outRt);
                        else
                            _ops.Conv3x3Pack4(src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.padLeft, conv.activationType, conv.activationSlope, outRt);
                    }
                    else
                    {
                        throw new InvalidOperationException("Texture convolution path unsupported config: " + layer.name);
                    }

                    textureBlobs[layer.topNames[0]] = new TensorRef
                    {
                        texture = outRt,
                        width = outWTex,
                        height = outHTex,
                        packs = conv.outPacks,
                        refs = 1,
                        owned = true
                    };
                    textureShapes[layer.topNames[0]] = new BufferShape(3, outWTex, outHTex, 1, conv.outC);
                    if (tempInputTex != null)
                        ReturnTempArray(tempInputTex);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Eltwise", StringComparison.Ordinal))
                {
                    var a = GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                    var b = GetOrMaterializeTexture(layer.bottomNames[1], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                    if (a.width != b.width || a.height != b.height || a.packs != b.packs)
                        throw new InvalidOperationException("Eltwise shape mismatch: " + layer.name);
                    var coeff = ParseEltwiseCoeff(layer);
                    var isTargetSftResidualLayer = string.IsNullOrEmpty(CodeFormerTargetSftResidualLayer)
                        ? CodeFormerSftResidualLayers.Contains(layer.name)
                        : string.Equals(layer.name, CodeFormerTargetSftResidualLayer, StringComparison.Ordinal);
                    if (CodeFormerSftMulScale != 1f && isTargetSftResidualLayer)
                        coeff = (coeff.coeffA, coeff.coeffB * CodeFormerSftMulScale);
                    var outRt = RentTempArray(a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                    _ops.AddPack4(a.texture, b.texture, coeff.coeffA, coeff.coeffB, a.packs, outRt);
                    textureBlobs[layer.topNames[0]] = new TensorRef
                    {
                        texture = outRt,
                        width = a.width,
                        height = a.height,
                        packs = a.packs,
                        refs = 1,
                        owned = true
                    };
                    var aShape = GetTextureShape(textureShapes, a, layer.bottomNames[0]);
                    textureShapes[layer.topNames[0]] = new BufferShape(3, a.width, a.height, 1, aShape.c);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "BinaryOp", StringComparison.Ordinal))
                {
                    var opType = layer.GetInt(0, 0);
                    var withScalar = layer.GetInt(1, 0);
                    var scalarB = layer.GetFloat(2, 0f);

                    var aBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    var aView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                    if (aBuf == null)
                        throw new InvalidOperationException("BinaryOp source not found: " + layer.name);

                    if (withScalar != 0)
                    {
                        var outBuf = new ComputeBuffer(aBuf.count, sizeof(float), ComputeBufferType.Structured);
                        _ops.BinaryOpScalarBuf(aBuf, scalarB, aBuf.count, opType, outBuf);
                        bufferBlobs[layer.topNames[0]] = outBuf;
                        if (aView != null)
                            bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, aView.dims, aView.w, aView.h, aView.d, aView.c, false);
                        tempOwned.Add(outBuf);
                    }
                    else
                    {
                        var bBuf = GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                        var bView = TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
                        if (bBuf == null)
                            throw new InvalidOperationException("BinaryOp second source not found: " + layer.name);

                        // ncnn reduction + binary-op chains in CodeFormer frequently mix [h,w] with [1,w] or [h,1].
                        // Expanding the smaller 2d tensor explicitly avoids ambiguous modulo-based broadcasting.
                        if (aView != null && bView != null && aView.dims == 2 && bView.dims == 2 && aBuf.count != bBuf.count)
                        {
                            if (TryExpand2DBroadcastBuffer(bBuf, bView, aView, out var expandedB, out var expandedBView))
                            {
                                bBuf = expandedB;
                                bView = expandedBView;
                                tempOwned.Add(expandedB);
                            }
                            else if (TryExpand2DBroadcastBuffer(aBuf, aView, bView, out var expandedA, out var expandedAView))
                            {
                                aBuf = expandedA;
                                aView = expandedAView;
                                tempOwned.Add(expandedA);
                            }
                        }
                        else if (aView != null && bView != null && aView.dims == 1 && bView.dims == 2)
                        {
                            if (TryExpand1DTo2DBroadcastBuffer(aBuf, aView, bView, out var expandedA, out var expandedAView))
                            {
                                aBuf = expandedA;
                                aView = expandedAView;
                                tempOwned.Add(expandedA);
                            }
                        }
                        else if (aView != null && bView != null && aView.dims == 2 && bView.dims == 1)
                        {
                            if (TryExpand1DTo2DBroadcastBuffer(bBuf, bView, aView, out var expandedB, out var expandedBView))
                            {
                                bBuf = expandedB;
                                bView = expandedBView;
                                tempOwned.Add(expandedB);
                            }
                        }

                        var isTargetSftAddLayer = string.IsNullOrEmpty(CodeFormerTargetSftAddLayer)
                            ? CodeFormerSftAddLayers.Contains(layer.name)
                            : string.Equals(layer.name, CodeFormerTargetSftAddLayer, StringComparison.Ordinal);

                        if (CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                        {
                            var scaledB = new ComputeBuffer(bBuf.count, sizeof(float), ComputeBufferType.Structured);
                            _ops.CopyBuf(bBuf, scaledB, bBuf.count);
                            _ops.MulScalarInplace(scaledB, CodeFormerSftAddScale, scaledB.count);
                            bBuf = scaledB;
                            tempOwned.Add(scaledB);
                        }

                        var isTargetSftMulLayer = string.IsNullOrEmpty(CodeFormerTargetSftMulLayer)
                            ? CodeFormerSftMulLayers.Contains(layer.name)
                            : string.Equals(layer.name, CodeFormerTargetSftMulLayer, StringComparison.Ordinal);
                        var isCodeFormerSftMul = opType == 2 && isTargetSftMulLayer;
                        var broadcast = ResolveBinaryBroadcast(aView, bView, aBuf.count, bBuf.count, layer.name);
                        var outBuf = new ComputeBuffer(broadcast.total, sizeof(float), ComputeBufferType.Structured);
                        _ops.BinaryOpBuf(aBuf, bBuf, broadcast.total, opType, outBuf, broadcast.mode, broadcast.size);
                        if (isCodeFormerSftMul)
                        {
                            if (CodeFormerBypassSftMul)
                            {
                                _ops.CopyBuf(aBuf, outBuf, broadcast.total);
                            }
                            else if (CodeFormerSftMulScale != 1f)
                            {
                                _ops.MulScalarInplace(outBuf, CodeFormerSftMulScale, outBuf.count);
                            }
                        }
                        bufferBlobs[layer.topNames[0]] = outBuf;
                        if (broadcast.outputView != null)
                            bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, broadcast.outputView.dims, broadcast.outputView.w, broadcast.outputView.h, broadcast.outputView.d, broadcast.outputView.c, false);
                        tempOwned.Add(outBuf);
                    }

                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "UnaryOp", StringComparison.Ordinal))
                {
                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    var srcView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                    if (srcBuf == null)
                        throw new InvalidOperationException("UnaryOp source not found: " + layer.name);
                    var outBuf = new ComputeBuffer(srcBuf.count, sizeof(float), ComputeBufferType.Structured);
                    _ops.UnaryOpBuf(srcBuf, srcBuf.count, layer.GetInt(0, 0), outBuf);
                    bufferBlobs[layer.topNames[0]] = outBuf;
                    if (srcView != null)
                        bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
                    tempOwned.Add(outBuf);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Swish", StringComparison.Ordinal))
                {
                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    var srcView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                    if (srcBuf == null)
                        throw new InvalidOperationException("Swish source not found: " + layer.name);
                    var outBuf = new ComputeBuffer(srcBuf.count, sizeof(float), ComputeBufferType.Structured);
                    _ops.SwishBuf(srcBuf, srcBuf.count, outBuf);
                    bufferBlobs[layer.topNames[0]] = outBuf;
                    if (srcView != null)
                        bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
                    tempOwned.Add(outBuf);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Sigmoid", StringComparison.Ordinal))
                {
                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    var srcView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                    if (srcBuf == null)
                        throw new InvalidOperationException("Sigmoid source not found: " + layer.name);
                    var outBuf = new ComputeBuffer(srcBuf.count, sizeof(float), ComputeBufferType.Structured);
                    _ops.SigmoidBuf(srcBuf, srcBuf.count, outBuf);
                    bufferBlobs[layer.topNames[0]] = outBuf;
                    if (srcView != null)
                        bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
                    tempOwned.Add(outBuf);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "GELU", StringComparison.Ordinal))
                {
                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    var srcView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                    if (srcBuf == null)
                        throw new InvalidOperationException("GELU source not found: " + layer.name);
                    var outBuf = new ComputeBuffer(srcBuf.count, sizeof(float), ComputeBufferType.Structured);
                    _ops.GeluBuf(srcBuf, srcBuf.count, outBuf);
                    bufferBlobs[layer.topNames[0]] = outBuf;
                    if (srcView != null)
                        bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
                    tempOwned.Add(outBuf);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Interp", StringComparison.Ordinal))
                {
                    var src = GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                    var resizeType = layer.GetInt(0, 2);
                    var sx = layer.GetFloat(1, 1f);
                    var sy = layer.GetFloat(2, 1f);

                    if (Mathf.Abs(sx - 2f) < 1e-3f && Mathf.Abs(sy - 2f) < 1e-3f)
                    {
                        var outRt = RentTempArray(src.width * 2, src.height * 2, src.packs, RenderTextureFormat.ARGBHalf);
                        if (resizeType == 1)
                            _ops.Interp2xNearestPack4(src.texture, src.packs, outRt);
                        else
                            _ops.Interp2xPack4(src.texture, src.packs, outRt);
                        textureBlobs[layer.topNames[0]] = new TensorRef
                        {
                            texture = outRt,
                            width = outRt.width,
                            height = outRt.height,
                            packs = src.packs,
                            refs = 1,
                            owned = true
                        };
                        var interpShape = GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                        textureShapes[layer.topNames[0]] = new BufferShape(3, outRt.width, outRt.height, 1, interpShape.c);
                        Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                        continue;
                    }

                    if (Mathf.Abs(sx - 0.5f) < 1e-3f && Mathf.Abs(sy - 0.5f) < 1e-3f)
                    {
                        var outRt = RentTempArray(Mathf.Max(1, src.width / 2), Mathf.Max(1, src.height / 2), src.packs, RenderTextureFormat.ARGBHalf);
                        if (resizeType == 1)
                            _ops.InterpDown2NearestPack4(src.texture, src.packs, outRt);
                        else
                            _ops.InterpDown2Pack4(src.texture, src.packs, outRt);
                        textureBlobs[layer.topNames[0]] = new TensorRef
                        {
                            texture = outRt,
                            width = outRt.width,
                            height = outRt.height,
                            packs = src.packs,
                            refs = 1,
                            owned = true
                        };
                        var interpDownShape = GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                        textureShapes[layer.topNames[0]] = new BufferShape(3, outRt.width, outRt.height, 1, interpDownShape.c);
                        Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                        continue;
                    }

                    throw new InvalidOperationException("unsupported interp scale: " + sx.ToString("0.###", CultureInfo.InvariantCulture) + "," + sy.ToString("0.###", CultureInfo.InvariantCulture));
                }

                if (string.Equals(layer.type, "LayerNorm", StringComparison.Ordinal))
                {
                    if (!_layerNorm.TryGetValue(layer.name, out var lp))
                        throw new InvalidOperationException("LayerNorm not found: " + layer.name);
                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    var srcView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                    if (srcBuf == null || srcView == null || srcView.dims != 2)
                        throw new InvalidOperationException("LayerNorm expects dims=2 buffer input: " + layer.name);
                    var outBuf = new ComputeBuffer(srcBuf.count, sizeof(float), ComputeBufferType.Structured);
                    _ops.CopyBuf(srcBuf, outBuf, srcBuf.count);
                    _ops.LayerNorm2DInplace(outBuf, srcView.h, srcView.w, lp.eps, lp.affine, lp.gamma, lp.beta);
                    bufferBlobs[layer.topNames[0]] = outBuf;
                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
                    tempOwned.Add(outBuf);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "GroupNorm", StringComparison.Ordinal))
                {
                    if (!_groupNorm.TryGetValue(layer.name, out var gp))
                        throw new InvalidOperationException("GroupNorm not found: " + layer.name);
                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    var srcView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                    if (srcBuf == null || srcView == null)
                        throw new InvalidOperationException("GroupNorm source not found: " + layer.name);
                    var outBuf = new ComputeBuffer(srcBuf.count, sizeof(float), ComputeBufferType.Structured);
                    _ops.CopyBuf(srcBuf, outBuf, srcBuf.count);
                    var spatial = srcBuf.count / Mathf.Max(1, gp.channels);
                    _ops.GroupNormInplace(outBuf, spatial, 1, gp.channels, gp.group, gp.eps, gp.affine, gp.gamma, gp.beta);
                    bufferBlobs[layer.topNames[0]] = outBuf;
                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
                    tempOwned.Add(outBuf);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "MultiHeadAttention", StringComparison.Ordinal))
                {
                    if (!_multiHeadAttention.TryGetValue(layer.name, out var mp))
                        throw new InvalidOperationException("MultiHeadAttention not found: " + layer.name);

                    if (!bufferBlobs.TryGetValue(layer.bottomNames[0], out var qBuf) || qBuf == null)
                        throw new InvalidOperationException("MultiHeadAttention q input not found: " + layer.name);
                    if (!bufferBlobs.TryGetValue(layer.bottomNames.Length > 1 ? layer.bottomNames[1] : layer.bottomNames[0], out var kBuf) || kBuf == null)
                        throw new InvalidOperationException("MultiHeadAttention k input not found: " + layer.name);
                    if (!bufferBlobs.TryGetValue(layer.bottomNames.Length > 2 ? layer.bottomNames[2] : layer.bottomNames[0], out var vBuf) || vBuf == null)
                        throw new InvalidOperationException("MultiHeadAttention v input not found: " + layer.name);

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

                    var ctx = new ComputeBuffer(srcLen * mp.embedDim, sizeof(float), ComputeBufferType.Structured);
                    _ops.MhaAttention(qScaled, kAff, vAff, srcLen, dstLen, mp.embedDim, mp.numHeads, 1f, ctx);

                    var outBuf = new ComputeBuffer(srcLen * mp.qdim, sizeof(float), ComputeBufferType.Structured);
                    _ops.InnerProduct2D(ctx, srcLen, mp.embedDim, mp.oW, mp.oB, mp.qdim, outBuf);

                    bufferBlobs[layer.topNames[0]] = outBuf;
                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, 2, mp.qdim, srcLen, 1, 1, false);
                    tempOwned.Add(qAff);
                    tempOwned.Add(kAff);
                    tempOwned.Add(vAff);
                    tempOwned.Add(qScaled);
                    tempOwned.Add(ctx);
                    tempOwned.Add(outBuf);
                    Consume(textureBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                throw new InvalidOperationException("unsupported layer type: " + layer.type);
            }

            return new InferResult(textureBlobs, textureShapes, bufferBlobs, bufferViews, tempOwned, this);
        }

        public RenderTexture RentTempArray(int w, int h, int depth, RenderTextureFormat format)
        {
            w = Mathf.Max(1, w);
            h = Mathf.Max(1, h);
            depth = Mathf.Max(1, depth);

            var key = new RtKey(w, h, depth, format);
            if (_useTempPool && _rtPool.TryGetValue(key, out var pool))
            {
                while (pool.Count > 0)
                {
                    var rt = pool.Pop();
                    if (rt != null)
                        return rt;
                }
            }

            var created = new RenderTexture(w, h, 0, format, RenderTextureReadWrite.Linear)
            {
                volumeDepth = depth,
                dimension = TextureDimension.Tex2DArray,
                enableRandomWrite = true,
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear
            };
            created.Create();
            return created;
        }

        public void ReturnTempArray(RenderTexture rt)
        {
            if (rt == null)
                return;

            if (!_useTempPool || _maxPooledPerShape <= 0)
            {
                try { rt.Release(); } catch { }
                UnityEngine.Object.Destroy(rt);
                return;
            }

            var key = new RtKey(rt.width, rt.height, rt.volumeDepth > 0 ? rt.volumeDepth : 1, rt.format);
            if (!_rtPool.TryGetValue(key, out var pool))
            {
                pool = new Stack<RenderTexture>();
                _rtPool[key] = pool;
            }

            if (pool.Count >= _maxPooledPerShape)
            {
                try { rt.Release(); } catch { }
                UnityEngine.Object.Destroy(rt);
                return;
            }

            pool.Push(rt);
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
                    try { rt.Release(); } catch { }
                    UnityEngine.Object.Destroy(rt);
                }
            }
            _rtPool.Clear();
        }

        public void Release()
        {
            foreach (var kv in _conv) kv.Value?.Dispose();
            foreach (var kv in _innerProduct) kv.Value?.Dispose();
            foreach (var kv in _memoryData) kv.Value?.Dispose();
            foreach (var kv in _embed) kv.Value?.Dispose();
            foreach (var kv in _layerNorm) kv.Value?.Dispose();
            foreach (var kv in _groupNorm) kv.Value?.Dispose();
            foreach (var kv in _multiHeadAttention) kv.Value?.Dispose();

            _conv.Clear();
            _innerProduct.Clear();
            _memoryData.Clear();
            _embed.Clear();
            _layerNorm.Clear();
            _groupNorm.Clear();
            _multiHeadAttention.Clear();
            Model = null;
            _blobUseCount = null;
            ClearTempPool();
        }

        public void Dispose()
        {
            Release();
        }

        private RenderTexture MaterializeTextureFromBuffer(
            string name,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews)
        {
            if (!bufferBlobs.TryGetValue(name, out var buffer) || buffer == null)
                return null;
            if (!bufferViews.TryGetValue(name, out var view) || view == null)
                return null;
            if (view.dims != 3)
                return null;

            var packs = Mathf.CeilToInt(view.c / 4f);
            var rt = RentTempArray(view.w, view.h, packs, RenderTextureFormat.ARGBHalf);
            _ops.FillPack4FromBufferCHW(buffer, view.w, view.h, view.c, rt);
            return rt;
        }

        private TensorRef GetOrMaterializeTexture(
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
                ? new BufferShape(3, view.w, view.h, 1, view.c)
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

        private static int ComputeConvOut(int inSize, int kernel, int dilation, int stride, int padBefore, int padAfter)
        {
            var kernelExtent = dilation * (kernel - 1) + 1;
            return Mathf.Max(1, (inSize + padBefore + padAfter - kernelExtent) / Mathf.Max(1, stride) + 1);
        }

        private static ComputeBuffer NewBuffer(float[] data)
        {
            var buf = new ComputeBuffer(data.Length, sizeof(float), ComputeBufferType.Structured);
            buf.SetData(data);
            return buf;
        }

        private static TensorRef GetTexture(Dictionary<string, TensorRef> blobs, string name)
        {
            if (!blobs.TryGetValue(name, out var tr) || tr == null || tr.texture == null)
                throw new InvalidOperationException("blob not found: " + name);
            return tr;
        }

        private static NcnnTensorBuffer TryGetBufferView(
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

        private ComputeBuffer GetOrConvertToBuffer(
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
                var convertedExact = new ComputeBuffer(logicalCount, sizeof(float), ComputeBufferType.Structured);
                _ops.Pack4ToBufferCHW(tr.texture, tr.width, tr.height, physicalChannels, convertedExact);
                bufferBlobs[name] = convertedExact;
                bufferViews[name] = new NcnnTensorBuffer(convertedExact, shape.dims, shape.w, shape.h, shape.d, shape.c, false);
                tempOwned.Add(convertedExact);
                return convertedExact;
            }

            if (logicalCount > 0 && logicalCount < physicalCount)
            {
                var physicalBuffer = new ComputeBuffer(physicalCount, sizeof(float), ComputeBufferType.Structured);
                _ops.Pack4ToBufferCHW(tr.texture, tr.width, tr.height, physicalChannels, physicalBuffer);
                tempOwned.Add(physicalBuffer);

                var converted = new ComputeBuffer(logicalCount, sizeof(float), ComputeBufferType.Structured);
                _ops.CopyBufPartial(physicalBuffer, 0, converted, logicalCount);

                bufferBlobs[name] = converted;
                bufferViews[name] = new NcnnTensorBuffer(converted, shape.dims, shape.w, shape.h, shape.d, shape.c, false);
                tempOwned.Add(converted);
                return converted;
            }

            throw new InvalidOperationException("texture logical shape mismatch: " + name + " | physical=" + physicalCount + " logical=" + logicalCount);
        }

        private static void Consume(
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
                }
                textureBlobs.Remove(name);
            }
        }

        private static NcnnTensorBuffer ResolveReshapeTensor(NcnnTensorBuffer src, NcnnParamModel.Layer layer)
        {
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

        private static BufferShape ResolveReshapeShape(BufferShape src, NcnnParamModel.Layer layer)
        {
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

        private static BufferShape GetTextureShape(Dictionary<string, BufferShape> textureShapes, TensorRef tr, string name)
        {
            if (textureShapes.TryGetValue(name, out var shape))
                return shape;
            return new BufferShape(3, tr.width, tr.height, 1, tr.packs * 4);
        }

        private int ResolveInputLogicalChannels(string inputBlobName, int fallbackChannels)
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
                if (!string.Equals(layer.type, "Convolution", StringComparison.Ordinal)
                    && !string.Equals(layer.type, "ConvolutionDepthWise", StringComparison.Ordinal))
                    continue;
                if (_conv.TryGetValue(layer.name, out var conv) && conv != null && conv.inC > 0)
                    return conv.inC;
            }

            return fallbackChannels;
        }

        private static Vector4Int ResolvePermuteAxes(int dims, int orderType, string layerName)
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

        private static BufferShape ResolvePermuteShape(NcnnTensorBuffer src, int dims, Vector4Int axes)
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

        private static int MapNcnnAxisToTensorAxis(int dims, int axis)
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

        private static int GetAxisSize(int dims, int w, int h, int d, int c, int axis)
        {
            if (axis == 0) return w;
            if (axis == 1) return h;
            if (axis == 2) return dims == 4 ? d : c;
            if (axis == 3) return c;
            throw new ArgumentOutOfRangeException(nameof(axis));
        }

        private NcnnTensorBuffer ApplyCropSlices(
            ComputeBuffer srcBuf,
            NcnnTensorBuffer srcView,
            NcnnParamModel.Layer layer,
            List<IDisposable> tempOwned)
        {
            var starts = layer.GetInts(-23309, null);
            var ends = layer.GetInts(-23310, null);
            var axes = layer.GetInts(-23311, null);

            if (starts == null || ends == null || starts.Length == 0 || ends.Length == 0)
            {
                throw new InvalidOperationException("Crop without starts/ends arrays is not supported yet: " + layer.name);
            }

            if (axes == null || axes.Length == 0)
            {
                axes = new int[starts.Length];
                for (var i = 0; i < axes.Length; i++)
                    axes[i] = i;
            }

            var currentBuf = srcBuf;
            var currentView = srcView;

            for (var i = 0; i < starts.Length; i++)
            {
                var ncnnAxis = axes[Mathf.Min(i, axes.Length - 1)];
                if (ncnnAxis < 0)
                    ncnnAxis += currentView.dims;
                var axis = MapNcnnAxisToTensorAxis(currentView.dims, ncnnAxis);

                var begin = starts[i];
                var end = ends[Mathf.Min(i, ends.Length - 1)];
                var axisSize = GetAxisSize(currentView.dims, currentView.w, currentView.h, currentView.d, currentView.c, axis);
                if (begin == -233) begin = 0;
                if (end == -233) end = axisSize;
                if (begin < 0) begin = axisSize + begin;
                if (end <= 0) end = axisSize + end;
                begin = Mathf.Clamp(begin, 0, axisSize);
                end = Mathf.Clamp(end, begin, axisSize);
                var outSize = Mathf.Max(0, end - begin);
                if (outSize <= 0)
                    throw new InvalidOperationException("Crop produced empty output: " + layer.name);

                var outW = currentView.w;
                var outH = currentView.h;
                var outD = currentView.d;
                var outC = currentView.c;
                if (axis == 0) outW = outSize;
                else if (axis == 1) outH = outSize;
                else if (axis == 2 && currentView.dims == 4) outD = outSize;
                else if (axis == 2 || axis == 3) outC = outSize;

                var outCount = outW * outH * outD * outC;
                var outBuf = new ComputeBuffer(outCount, sizeof(float), ComputeBufferType.Structured);
                _ops.Slice(currentBuf, currentView.dims, currentView.w, currentView.h, currentView.d, currentView.c, axis, begin, outW, outH, outD, outC, outBuf);
                var outView = new NcnnTensorBuffer(outBuf, currentView.dims, outW, outH, outD, outC, false);

                if (!ReferenceEquals(currentBuf, srcBuf))
                    tempOwned.Add(currentBuf);
                tempOwned.Add(outBuf);
                currentBuf = outBuf;
                currentView = outView;
            }

            return currentView;
        }

        private static NcnnTensorBuffer ShuffleChannelCpu(ComputeBuffer srcBuffer, NcnnTensorBuffer srcView, int group, bool reverse)
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

            var outBuffer = new ComputeBuffer(dstData.Length, sizeof(float), ComputeBufferType.Structured);
            outBuffer.SetData(dstData);
            return new NcnnTensorBuffer(outBuffer, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
        }

        private static (int mode, int size, int total, NcnnTensorBuffer outputView) ResolveBinaryBroadcast(
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

        private static bool TryExpand2DBroadcastBuffer(
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

            expandedBuffer = new ComputeBuffer(expandedData.Length, sizeof(float), ComputeBufferType.Structured);
            expandedBuffer.SetData(expandedData);
            expandedView = new NcnnTensorBuffer(expandedBuffer, 2, targetView.w, targetView.h, 1, 1, false);
            return true;
        }

        private static bool TryExpand1DTo2DBroadcastBuffer(
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

            expandedBuffer = new ComputeBuffer(expandedData.Length, sizeof(float), ComputeBufferType.Structured);
            expandedBuffer.SetData(expandedData);
            expandedView = new NcnnTensorBuffer(expandedBuffer, 2, targetView.w, targetView.h, 1, 1, false);
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
    }
}
