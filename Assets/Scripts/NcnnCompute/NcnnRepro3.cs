using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnRepro3 : IDisposable
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

        private readonly struct TensorShape
        {
            public readonly int w;
            public readonly int h;
            public readonly int c;

            public TensorShape(int w, int h, int c)
            {
                this.w = w;
                this.h = h;
                this.c = c;
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
            public int width;
            public int height;
            public int packs;
            public int sourceWidth;
            public int sourceHeight;
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
            public bool useWinograd23;
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

        public sealed class InferResult : IDisposable
        {
            private readonly Dictionary<string, TensorRef> _textureBlobs;
            private readonly Dictionary<string, IndexRef> _indexBlobs;
            private readonly NcnnRepro3 _owner;
            private readonly HashSet<object> _visited = new HashSet<object>();

            internal InferResult(
                Dictionary<string, TensorRef> textureBlobs,
                Dictionary<string, IndexRef> indexBlobs,
                NcnnRepro3 owner)
            {
                _textureBlobs = textureBlobs;
                _indexBlobs = indexBlobs;
                _owner = owner;
            }

            public RenderTexture GetTexture(string name)
            {
                if (_textureBlobs.TryGetValue(name, out var tr) && tr != null && tr.texture != null)
                    return tr.texture;
                throw new InvalidOperationException("blob not found: " + name);
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
                throw new InvalidOperationException("blob not found: " + name);
            }

            public void Dispose()
            {
                foreach (var kv in _textureBlobs)
                {
                    var tr = kv.Value;
                    if (tr == null || !_visited.Add(tr))
                        continue;
                    if (tr.owned && tr.texture != null)
                    {
                        try { _owner.ReturnTempArray(tr.texture); } catch { }
                    }
                }

                foreach (var kv in _indexBlobs)
                {
                    var ir = kv.Value;
                    if (ir == null || !_visited.Add(ir))
                        continue;
                    if (ir.owned && ir.texture != null)
                    {
                        try { _owner.ReturnTempArray(ir.texture); } catch { }
                    }
                }

                _textureBlobs.Clear();
                _indexBlobs.Clear();
            }
        }

        private readonly Dictionary<string, ConvPack> _conv = new Dictionary<string, ConvPack>(StringComparer.Ordinal);
        private readonly Dictionary<RtKey, Stack<RenderTexture>> _rtPool = new Dictionary<RtKey, Stack<RenderTexture>>();
        private readonly NcnnOps _ops;
        private Dictionary<string, int> _blobUseCount;
        private bool _useTempPool;
        private int _maxPooledPerShape = 2;

        public NcnnParamModel Model { get; private set; }
        public bool EnableWinograd23 { get; set; }
        public bool ForceBufferConvolution { get; set; }
        public RenderTextureFormat TensorTextureFormat { get; set; } = RenderTextureFormat.ARGBHalf;

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

        public NcnnRepro3(NcnnOps ops)
        {
            _ops = ops ?? throw new ArgumentNullException(nameof(ops));
        }

        public void LoadModel(string paramText, NcnnBinReader br)
        {
            if (br == null)
                throw new ArgumentNullException(nameof(br));

            Release();
            Model = NcnnParamParser.Parse(paramText);
            _blobUseCount = NcnnRepro2.BuildBlobUseCount(Model);

            foreach (var layer in Model.layers)
            {
                if (!string.Equals(layer.type, "Convolution", StringComparison.Ordinal))
                    continue;

                var pack = new ConvPack
                {
                    outC = layer.GetInt(0, 0),
                    group = Mathf.Max(1, layer.GetInt(7, 1)),
                    kernelW = layer.GetInt(1, 0),
                    kernelH = layer.GetInt(11, layer.GetInt(1, 0)),
                    dilationW = layer.GetInt(2, 1),
                    dilationH = layer.GetInt(12, layer.GetInt(2, 1)),
                    strideW = layer.GetInt(3, 1),
                    strideH = layer.GetInt(13, layer.GetInt(3, 1)),
                    padLeft = layer.GetInt(4, 0),
                    padRight = layer.GetInt(15, layer.GetInt(4, 0)),
                    padTop = layer.GetInt(14, layer.GetInt(4, 0)),
                    padBottom = layer.GetInt(16, layer.GetInt(14, layer.GetInt(4, 0))),
                    biasTerm = layer.GetInt(5, 0),
                    weightSize = layer.GetInt(6, 0),
                    activationType = layer.GetInt(9, 0),
                    activationSlope = NcnnRepro2.ParseLeakySlope(layer)
                };

                var kernelArea = Mathf.Max(1, pack.kernelW * pack.kernelH);
                pack.inC = Mathf.Max(1, pack.weightSize / Mathf.Max(1, pack.outC * kernelArea));
                pack.inPacks = (pack.inC + 3) / 4;
                pack.outPacks = (pack.outC + 3) / 4;

                var tag = br.ReadInt32();
                if (tag != 0x01306B47)
                    throw new InvalidOperationException("unexpected weight tag at " + br.Position + ": 0x" + tag.ToString("X8", CultureInfo.InvariantCulture));

                var weight = br.ReadFp16ArrayAsFloat32(pack.weightSize, true);
                var bias = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];

                pack.rawWeight = NewFloatBuffer(weight);
                pack.rawBias = NewFloatBuffer(bias);

                if (pack.group == 1
                    && pack.dilationW == 1
                    && pack.dilationH == 1
                    && ((pack.kernelW == 1 && pack.kernelH == 1 && pack.strideW == 1 && pack.strideH == 1)
                        || (pack.kernelW == 3 && pack.kernelH == 3 && pack.strideW == 1 && pack.strideH == 1
                            && pack.padLeft == pack.padRight && pack.padTop == pack.padBottom && pack.padLeft == pack.padTop)))
                {
                    var w4 = NcnnRepro2.PackWeightsToO4I4K(weight, pack.outC, pack.inC, pack.kernelW, pack.outPacks, pack.inPacks);
                    var b4 = NcnnRepro2.PackBiasToO4(bias, pack.outC, pack.outPacks);
                    pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    pack.packedWeight4.SetData(w4);
                    pack.packedBias4.SetData(b4);

                    if (pack.kernelW == 3
                        && pack.kernelH == 3
                        && pack.padLeft == 1
                        && pack.padRight == 1
                        && pack.padTop == 1
                        && pack.padBottom == 1
                        && NcnnWinograd23.CanUse(pack.kernelW, pack.padLeft, pack.inPacks, pack.outPacks))
                    {
                        pack.useWinograd23 = true;
                        var wTm = NcnnWinograd23.PackWeightTm23(weight, pack.outC, pack.inC, pack.outPacks, pack.inPacks);
                        pack.packedWeightTm23 = new ComputeBuffer(wTm.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                        pack.packedWeightTm23.SetData(wTm);
                    }
                }

                _conv[layer.name] = pack;
            }
        }

        public InferResult Infer(RenderTexture inputPack4, int inputPacks, string inputBlobName = "input", ICollection<string> pinnedNames = null)
        {
            if (inputPack4 == null)
                throw new ArgumentNullException(nameof(inputPack4));
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");

            var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
            var textureBlobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);
            var textureShapes = new Dictionary<string, TensorShape>(StringComparer.Ordinal);
            var indexBlobs = new Dictionary<string, IndexRef>(StringComparer.Ordinal);

            var inputUseCount = _blobUseCount.TryGetValue(inputBlobName, out var initialUseCount) ? initialUseCount : 1;
            var logicalInputChannels = ResolveInputLogicalChannels(inputBlobName, inputPacks * 4);
            textureBlobs[inputBlobName] = new TensorRef
            {
                texture = inputPack4,
                width = inputPack4.width,
                height = inputPack4.height,
                packs = inputPacks,
                refs = inputUseCount,
                owned = false
            };
            textureShapes[inputBlobName] = new TensorShape(inputPack4.width, inputPack4.height, logicalInputChannels);

            foreach (var layer in Model.layers)
            {
                if (string.Equals(layer.type, "Input", StringComparison.Ordinal))
                    continue;

                if (string.Equals(layer.type, "Split", StringComparison.Ordinal))
                {
                    if (textureBlobs.TryGetValue(layer.bottomNames[0], out var srcTex) && srcTex != null && srcTex.texture != null)
                    {
                        var shape = GetTextureShape(textureShapes, layer.bottomNames[0], srcTex);
                        for (var i = 0; i < layer.topNames.Length; i++)
                        {
                            textureBlobs[layer.topNames[i]] = srcTex;
                            textureShapes[layer.topNames[i]] = shape;
                            srcTex.refs++;
                        }
                    }
                    else if (indexBlobs.TryGetValue(layer.bottomNames[0], out var srcIdx) && srcIdx != null && srcIdx.texture != null)
                    {
                        for (var i = 0; i < layer.topNames.Length; i++)
                        {
                            indexBlobs[layer.topNames[i]] = srcIdx;
                            srcIdx.refs++;
                        }
                    }
                    else
                    {
                        throw new InvalidOperationException("Split source not found: " + layer.name);
                    }

                    Consume(textureBlobs, indexBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Convolution", StringComparison.Ordinal))
                {
                    if (!_conv.TryGetValue(layer.name, out var conv) || conv == null)
                        throw new InvalidOperationException("Convolution not found: " + layer.name);

                    var src = GetTexture(textureBlobs, layer.bottomNames[0]);
                    var srcShape = GetTextureShape(textureShapes, layer.bottomNames[0], src);
                    var outW = ComputeConvOut(src.width, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
                    var outH = ComputeConvOut(src.height, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
                    var outRt = RentTempArray(outW, outH, conv.outPacks, RenderTextureFormat.ARGBHalf);

                    var canUseTexturePath = !ForceBufferConvolution
                                            && conv.group == 1
                                            && conv.dilationW == 1
                                            && conv.dilationH == 1
                                            && ((conv.kernelW == 1 && conv.kernelH == 1 && conv.strideW == 1 && conv.strideH == 1)
                                                || (conv.kernelW == 3 && conv.kernelH == 3 && conv.strideW == 1 && conv.strideH == 1
                                                    && conv.padLeft == conv.padRight && conv.padTop == conv.padBottom && conv.padLeft == conv.padTop));

                    if (canUseTexturePath)
                    {
                        if (conv.kernelW == 1)
                        {
                            _ops.Conv1x1Pack4(src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.activationType, conv.activationSlope, outRt);
                        }
                        else if (EnableWinograd23 && conv.useWinograd23)
                        {
                            _ops.Conv3x3Pack4Winograd23(src.texture, conv.inPacks, conv.packedWeightTm23, conv.packedBias4, conv.outPacks, conv.biasTerm, conv.activationType, conv.activationSlope, outRt);
                        }
                        else
                        {
                            _ops.Conv3x3Pack4(src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.padLeft, conv.activationType, conv.activationSlope, outRt);
                        }
                    }
                    else
                    {
                        using var srcBuffer = new ComputeBuffer(srcShape.w * srcShape.h * srcShape.c, sizeof(float), ComputeBufferType.Structured);
                        _ops.Pack4ToBufferCHW(src.texture, srcShape.w, srcShape.h, srcShape.c, srcBuffer);
                        var srcTensor = new NcnnTensorBuffer(srcBuffer, 3, srcShape.w, srcShape.h, 1, srcShape.c, false);
                        using var outTensor = new NcnnTensorBuffer(outW, outH, conv.outC);
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
                        _ops.FillPack4FromBufferCHW(outTensor.buffer, outW, outH, conv.outC, outRt);
                    }

                    SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, new TensorShape(outW, outH, conv.outC));
                    Consume(textureBlobs, indexBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "BinaryOp", StringComparison.Ordinal))
                {
                    var opType = layer.GetInt(0, 0);
                    var withScalar = layer.GetInt(1, 0);
                    if (withScalar != 0)
                        throw new InvalidOperationException("BinaryOp scalar path is not supported in NcnnRepro3: " + layer.name);

                    var a = GetTexture(textureBlobs, layer.bottomNames[0]);
                    var b = GetTexture(textureBlobs, layer.bottomNames[1]);
                    var aShape = GetTextureShape(textureShapes, layer.bottomNames[0], a);
                    var bShape = GetTextureShape(textureShapes, layer.bottomNames[1], b);
                    if (a.width != b.width || a.height != b.height || a.packs != b.packs || aShape.c != bShape.c)
                        throw new InvalidOperationException("BinaryOp shape mismatch: " + layer.name);

                    var outRt = RentTempArray(a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                    _ops.BinaryOpPack4(a.texture, b.texture, a.packs, opType, outRt);
                    SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, aShape);
                    Consume(textureBlobs, indexBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "ReLU", StringComparison.Ordinal))
                {
                    var src = GetTexture(textureBlobs, layer.bottomNames[0]);
                    var srcShape = GetTextureShape(textureShapes, layer.bottomNames[0], src);
                    var outRt = RentTempArray(src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.LeakyReluPack4(src.texture, 0f, src.packs, outRt);
                    SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
                    Consume(textureBlobs, indexBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Pooling", StringComparison.Ordinal))
                {
                    var src = GetTexture(textureBlobs, layer.bottomNames[0]);
                    var srcShape = GetTextureShape(textureShapes, layer.bottomNames[0], src);
                    var poolType = layer.GetInt(0, 0);
                    var kernelW = layer.GetInt(1, 0);
                    var kernelH = layer.GetInt(11, kernelW);
                    var strideW = layer.GetInt(2, 1);
                    var strideH = layer.GetInt(12, strideW);
                    var padLeft = layer.GetInt(3, 0);
                    var padRight = layer.GetInt(14, padLeft);
                    var padTop = layer.GetInt(13, padLeft);
                    var padBottom = layer.GetInt(15, padTop);

                    var outW = Mathf.Max(1, (src.width + padLeft + padRight - kernelW) / Mathf.Max(1, strideW) + 1);
                    var outH = Mathf.Max(1, (src.height + padTop + padBottom - kernelH) / Mathf.Max(1, strideH) + 1);
                    var outRt = RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.PoolingPack4(src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, poolType, outRt);
                    SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, new TensorShape(outW, outH, srcShape.c));
                    Consume(textureBlobs, indexBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "MaxPoolingInd", StringComparison.Ordinal))
                {
                    var src = GetTexture(textureBlobs, layer.bottomNames[0]);
                    var srcShape = GetTextureShape(textureShapes, layer.bottomNames[0], src);
                    var kernelW = layer.GetInt(1, 0);
                    var kernelH = layer.GetInt(11, kernelW);
                    var strideW = layer.GetInt(2, 1);
                    var strideH = layer.GetInt(12, strideW);
                    var padLeft = layer.GetInt(3, 0);
                    var padRight = layer.GetInt(14, padLeft);
                    var padTop = layer.GetInt(13, padLeft);
                    var padBottom = layer.GetInt(15, padTop);

                    var outW = Mathf.Max(1, (src.width + padLeft + padRight - kernelW) / Mathf.Max(1, strideW) + 1);
                    var outH = Mathf.Max(1, (src.height + padTop + padBottom - kernelH) / Mathf.Max(1, strideH) + 1);
                    var outRt = RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                    var idxRt = RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBFloat);
                    _ops.MaxPoolingIndPack4(src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, outRt, idxRt);
                    SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, new TensorShape(outW, outH, srcShape.c));
                    indexBlobs[layer.topNames[1]] = new IndexRef
                    {
                        texture = idxRt,
                        width = outW,
                        height = outH,
                        packs = src.packs,
                        sourceWidth = src.width,
                        sourceHeight = src.height,
                        refs = _blobUseCount.TryGetValue(layer.topNames[1], out var idxUseCount) ? idxUseCount : 1,
                        owned = true
                    };
                    Consume(textureBlobs, indexBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Interp", StringComparison.Ordinal))
                {
                    var src = GetTexture(textureBlobs, layer.bottomNames[0]);
                    var srcShape = GetTextureShape(textureShapes, layer.bottomNames[0], src);
                    var resizeType = layer.GetInt(0, 2);
                    var sx = layer.GetFloat(1, 1f);
                    var sy = layer.GetFloat(2, 1f);
                    if (resizeType == 1)
                        throw new InvalidOperationException("nearest interp is not used by the matting model and is not supported in NcnnRepro3: " + layer.name);

                    var outW = Mathf.Max(1, Mathf.RoundToInt(src.width * sx));
                    var outH = Mathf.Max(1, Mathf.RoundToInt(src.height * sy));
                    var outRt = RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.InterpPack4(src.texture, src.packs, sx, sy, outRt);
                    SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, new TensorShape(outW, outH, srcShape.c));
                    Consume(textureBlobs, indexBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "MaxUnPooling", StringComparison.Ordinal))
                {
                    var src = GetTexture(textureBlobs, layer.bottomNames[0]);
                    if (!indexBlobs.TryGetValue(layer.bottomNames[1], out var idx) || idx == null || idx.texture == null)
                        throw new InvalidOperationException("MaxUnPooling index source not found: " + layer.name);
                    var srcShape = GetTextureShape(textureShapes, layer.bottomNames[0], src);
                    var kernelW = layer.GetInt(1, 3);
                    var kernelH = layer.GetInt(11, kernelW);
                    var strideW = layer.GetInt(2, 2);
                    var strideH = layer.GetInt(12, strideW);
                    var padLeft = layer.GetInt(3, 1);
                    var padTop = layer.GetInt(13, padLeft);

                    var outRt = RentTempArray(idx.sourceWidth, idx.sourceHeight, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.MaxUnPoolingPack4(src.texture, idx.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, outRt);
                    SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, new TensorShape(idx.sourceWidth, idx.sourceHeight, srcShape.c));
                    Consume(textureBlobs, indexBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                if (string.Equals(layer.type, "Concat", StringComparison.Ordinal))
                {
                    var first = GetTexture(textureBlobs, layer.bottomNames[0]);
                    var firstShape = GetTextureShape(textureShapes, layer.bottomNames[0], first);
                    var totalPacks = first.packs;
                    var totalChannels = firstShape.c;

                    for (var i = 1; i < layer.bottomNames.Length; i++)
                    {
                        var part = GetTexture(textureBlobs, layer.bottomNames[i]);
                        var partShape = GetTextureShape(textureShapes, layer.bottomNames[i], part);
                        if (part.width != first.width || part.height != first.height)
                            throw new InvalidOperationException("Concat shape mismatch: " + layer.name);
                        totalPacks += part.packs;
                        totalChannels += partShape.c;
                    }

                    var outRt = RentTempArray(first.width, first.height, totalPacks, RenderTextureFormat.ARGBHalf);
                    var packOffset = 0;
                    for (var i = 0; i < layer.bottomNames.Length; i++)
                    {
                        var part = GetTexture(textureBlobs, layer.bottomNames[i]);
                        _ops.CopyPack4(part.texture, 0, outRt, packOffset, part.packs);
                        packOffset += part.packs;
                    }

                    SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, new TensorShape(first.width, first.height, totalChannels));
                    Consume(textureBlobs, indexBlobs, remaining, layer.bottomNames, pinnedNames);
                    continue;
                }

                throw new InvalidOperationException("unsupported layer in NcnnRepro3: " + layer.type + " | " + layer.name);
            }

            return new InferResult(textureBlobs, indexBlobs, this);
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

            var allocLabel = "NcnnRepro3.RentTempArray(" + (callerMember ?? "?") + ":" + callerLine.ToString(CultureInfo.InvariantCulture) + ")";
            var key = new RtKey(w, h, depth, format);
            if (_useTempPool && _rtPool.TryGetValue(key, out var pool))
            {
                while (pool.Count > 0)
                {
                    var pooled = pool.Pop();
                    if (pooled != null)
                    {
                        NcnnGpuResourceTracker.RegisterTexture(pooled, allocLabel + "|pool");
                        return pooled;
                    }
                }
            }

            var desc = new RenderTextureDescriptor(w, h, format, 0)
            {
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = depth,
                enableRandomWrite = true,
                msaaSamples = 1
            };
            var rt = RenderTexture.GetTemporary(desc);
            NcnnGpuResourceTracker.RegisterTexture(rt, allocLabel + "|new");
            return rt;
        }

        public void ReturnTempArray(RenderTexture rt)
        {
            if (rt == null)
                return;

            if (!_useTempPool || _maxPooledPerShape <= 0)
            {
                NcnnGpuResourceTracker.ReleaseTexture(rt, "NcnnRepro3.ReturnTempArray");
                RenderTexture.ReleaseTemporary(rt);
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
                NcnnGpuResourceTracker.ReleaseTexture(rt, "NcnnRepro3.ReturnTempArray(pool-full)");
                RenderTexture.ReleaseTemporary(rt);
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
                    NcnnGpuResourceTracker.ReleaseTexture(rt, "NcnnRepro3.ClearTempPool");
                    try { RenderTexture.ReleaseTemporary(rt); } catch { }
                }
            }
            _rtPool.Clear();
        }

        public void Release()
        {
            foreach (var kv in _conv)
                kv.Value?.Dispose();
            _conv.Clear();
            Model = null;
            _blobUseCount = null;
            ClearTempPool();
        }

        public void Dispose()
        {
            Release();
        }

        private static void SetTextureBlob(
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, TensorShape> textureShapes,
            string name,
            RenderTexture texture,
            TensorShape shape)
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
            textureShapes[name] = shape;
        }

        private static TensorRef GetTexture(Dictionary<string, TensorRef> textureBlobs, string name)
        {
            if (!textureBlobs.TryGetValue(name, out var tr) || tr == null || tr.texture == null)
                throw new InvalidOperationException("blob not found: " + name);
            return tr;
        }

        private static TensorShape GetTextureShape(Dictionary<string, TensorShape> textureShapes, string name, TensorRef texture)
        {
            if (textureShapes.TryGetValue(name, out var shape))
                return shape;
            return new TensorShape(texture.width, texture.height, texture.packs * 4);
        }

        private void Consume(
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, IndexRef> indexBlobs,
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
                        NcnnGpuResourceTracker.ReleaseTexture(tr.texture, "NcnnRepro3.Consume(texture)");
                        try { RenderTexture.ReleaseTemporary(tr.texture); } catch { }
                        tr.texture = null;
                    }
                    textureBlobs.Remove(name);
                }

                if (indexBlobs.TryGetValue(name, out var ir) && ir != null)
                {
                    ir.refs--;
                    if (ir.refs <= 0 && ir.owned && ir.texture != null)
                    {
                        NcnnGpuResourceTracker.ReleaseTexture(ir.texture, "NcnnRepro3.Consume(index)");
                        try { RenderTexture.ReleaseTemporary(ir.texture); } catch { }
                        ir.texture = null;
                    }
                    indexBlobs.Remove(name);
                }
            }
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
                if (!string.Equals(layer.type, "Convolution", StringComparison.Ordinal))
                    continue;
                if (_conv.TryGetValue(layer.name, out var conv) && conv != null && conv.inC > 0)
                    return conv.inC;
            }

            return fallbackChannels;
        }

        private static int ComputeConvOut(int inSize, int kernel, int dilation, int stride, int padBefore, int padAfter)
        {
            var kernelExtent = dilation * (kernel - 1) + 1;
            return Mathf.Max(1, (inSize + padBefore + padAfter - kernelExtent) / Mathf.Max(1, stride) + 1);
        }

        private static ComputeBuffer NewFloatBuffer(float[] data)
        {
            var buffer = new ComputeBuffer(data.Length, sizeof(float), ComputeBufferType.Structured);
            buffer.SetData(data);
            return buffer;
        }
    }
}
