// Auto-generated kernel implementation group: AexisKernels.BufferTensor.hlsl

void AexisEmbed_Impl(uint3 id)
{
    int p = (int)id.x;
    int q = (int)id.y;
    if (q < 0 || p < 0 || q >= _EmbedWords || p >= _EmbedNumOutput) return;

    int word_index = _EmbedIdx[q];
    word_index = clamp(word_index, 0, _EmbedInputDim - 1);
    int weightIndex = word_index * _EmbedNumOutput + p;
    float v;
    if (_UseInt4EmbedWeights != 0)
    {
        uint packed = _EmbedWInt4Packed[weightIndex >> 3];
        uint raw = (packed >> ((weightIndex & 7) * 4)) & 0xfu;
        int signedValue = raw >= 8u ? (int)raw - 16 : (int)raw;
        v = (float)signedValue * _EmbedWInt4Scales[weightIndex / _EmbedWInt4ScaleBlockSize];
    }
    else if (_UseInt8EmbedWeights != 0)
    {
        uint packed = _EmbedWInt8Packed[weightIndex >> 2];
        uint raw = (packed >> ((weightIndex & 3) * 8)) & 0xffu;
        int signedValue = raw >= 128u ? (int)raw - 256 : (int)raw;
        v = (float)signedValue * _EmbedWInt8Scales[word_index];
    }
    else
    {
        v = _EmbedW[weightIndex];
    }
    if (_EmbedBiasTerm != 0)
        v += _EmbedB[p];
    _EmbedOut[q * _EmbedNumOutput + p] = v;
}

float AexisEmbedValue(int wordIndex, int p)
{
    wordIndex = clamp(wordIndex, 0, _EmbedInputDim - 1);
    int weightIndex = wordIndex * _EmbedNumOutput + p;
    float v;
    if (_UseInt4EmbedWeights != 0)
    {
        uint packed = _EmbedWInt4Packed[weightIndex >> 3];
        uint raw = (packed >> ((weightIndex & 7) * 4)) & 0xfu;
        int signedValue = raw >= 8u ? (int)raw - 16 : (int)raw;
        v = (float)signedValue * _EmbedWInt4Scales[weightIndex / _EmbedWInt4ScaleBlockSize];
    }
    else if (_UseInt8EmbedWeights != 0)
    {
        uint packed = _EmbedWInt8Packed[weightIndex >> 2];
        uint raw = (packed >> ((weightIndex & 3) * 8)) & 0xffu;
        int signedValue = raw >= 128u ? (int)raw - 256 : (int)raw;
        v = (float)signedValue * _EmbedWInt8Scales[wordIndex];
    }
    else
    {
        v = _EmbedW[weightIndex];
    }
    if (_EmbedBiasTerm != 0)
        v += _EmbedB[p];
    return v;
}

void AexisEmbedTexture_Impl(uint3 id)
{
    int p = (int)id.x;
    int q = (int)id.y;
    if (q < 0 || p < 0 || q >= _EmbedWords || p >= _EmbedNumOutput) return;

    _LinearOut0[int2(p, q)] = AexisEmbedValue(_EmbedIdx[q], p);
}

void AexisEmbedTextureLinearIndex_Impl(uint3 id)
{
    int p = (int)id.x;
    int q = (int)id.y;
    if (q < 0 || p < 0 || q >= _EmbedWords || p >= _EmbedNumOutput) return;

    int storageW = max(1, _EmbedIndexStorageW);
    int x = q % storageW;
    int y = q / storageW;
    int wordIndex = (int)round(_LinearIn0[int2(x, y)]);
    _LinearOut0[int2(p, q)] = AexisEmbedValue(wordIndex, p);
}

void AexisEmbedTexturePack4Index_Impl(uint3 id)
{
    int p = (int)id.x;
    int q = (int)id.y;
    if (q < 0 || p < 0 || q >= _EmbedWords || p >= _EmbedNumOutput) return;

    int storageW = max(1, _EmbedIndexStorageW);
    int x = q % storageW;
    int y = q / storageW;
    float4 packed = _TexIn0Arr[int3(x, y, 0)];
    int wordIndex = (int)round(packed.x);
    _LinearOut0[int2(p, q)] = AexisEmbedValue(wordIndex, p);
}

void AexisPermute_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    int dims = _PermuteDims;

    if (dims == 2)
    {
        uint total = (uint)(_PermuteOutW * _PermuteOutH);
        if (idx >= total) return;

        int ow = (int)(idx % (uint)_PermuteOutW);
        int oh = (int)(idx / (uint)_PermuteOutW);

        int ax0 = _PermuteAxes.x;
        int ax1 = _PermuteAxes.y;
        int inv0 = ax0 == 0 ? 0 : 1;
        int inv1 = ax0 == 1 ? 0 : 1;

        int out0 = ow;
        int out1 = oh;
        int iw = (inv0 == 0 ? out0 : out1);
        int ih = (inv1 == 0 ? out0 : out1);

        uint inIdx = (uint)(ih * _PermuteInW + iw);
        _PermuteOut[idx] = _PermuteIn[inIdx];
        return;
    }

    if (dims == 3)
    {
        uint total = (uint)(_PermuteOutW * _PermuteOutH * _PermuteOutC);
        if (idx >= total) return;

        int ow = (int)(idx % (uint)_PermuteOutW);
        uint t1 = idx / (uint)_PermuteOutW;
        int oh = (int)(t1 % (uint)_PermuteOutH);
        int oc = (int)(t1 / (uint)_PermuteOutH);

        int ax0 = _PermuteAxes.x;
        int ax1 = _PermuteAxes.y;
        int ax2 = _PermuteAxes.z;
        int inv0 = ax0 == 0 ? 0 : (ax1 == 0 ? 1 : 2);
        int inv1 = ax0 == 1 ? 0 : (ax1 == 1 ? 1 : 2);
        int inv2 = ax0 == 2 ? 0 : (ax1 == 2 ? 1 : 2);

        int out0 = ow;
        int out1 = oh;
        int out2 = oc;
        int iw = (inv0 == 0 ? out0 : (inv0 == 1 ? out1 : out2));
        int ih = (inv1 == 0 ? out0 : (inv1 == 1 ? out1 : out2));
        int ic = (inv2 == 0 ? out0 : (inv2 == 1 ? out1 : out2));

        uint inIdx = (uint)((ic * _PermuteInH + ih) * _PermuteInW + iw);
        _PermuteOut[idx] = _PermuteIn[inIdx];
        return;
    }

    if (dims == 4)
    {
        uint total = (uint)(_PermuteOutW * _PermuteOutH * _PermuteOutD * _PermuteOutC);
        if (idx >= total) return;

        int ow = (int)(idx % (uint)_PermuteOutW);
        uint t1 = idx / (uint)_PermuteOutW;
        int oh = (int)(t1 % (uint)_PermuteOutH);
        uint t2 = t1 / (uint)_PermuteOutH;
        int od = (int)(t2 % (uint)_PermuteOutD);
        int oc = (int)(t2 / (uint)_PermuteOutD);

        int ax0 = _PermuteAxes.x;
        int ax1 = _PermuteAxes.y;
        int ax2 = _PermuteAxes.z;
        int ax3 = _PermuteAxes.w;
        int inv0 = ax0 == 0 ? 0 : (ax1 == 0 ? 1 : (ax2 == 0 ? 2 : 3));
        int inv1 = ax0 == 1 ? 0 : (ax1 == 1 ? 1 : (ax2 == 1 ? 2 : 3));
        int inv2 = ax0 == 2 ? 0 : (ax1 == 2 ? 1 : (ax2 == 2 ? 2 : 3));
        int inv3 = ax0 == 3 ? 0 : (ax1 == 3 ? 1 : (ax2 == 3 ? 2 : 3));

        int out0 = ow;
        int out1 = oh;
        int out2 = od;
        int out3 = oc;

        int iw = (inv0 == 0 ? out0 : (inv0 == 1 ? out1 : (inv0 == 2 ? out2 : out3)));
        int ih = (inv1 == 0 ? out0 : (inv1 == 1 ? out1 : (inv1 == 2 ? out2 : out3)));
        int idd = (inv2 == 0 ? out0 : (inv2 == 1 ? out1 : (inv2 == 2 ? out2 : out3)));
        int ic = (inv3 == 0 ? out0 : (inv3 == 1 ? out1 : (inv3 == 2 ? out2 : out3)));

        uint inIdx = (uint)((((ic * _PermuteInD + idd) * _PermuteInH + ih) * _PermuteInW) + iw);
        _PermuteOut[idx] = _PermuteIn[inIdx];
        return;
    }
}

void AexisSlice_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    int dims = _SliceDims;
    int ow = _SliceOutW;
    int oh = _SliceOutH;
    int od = _SliceOutD;
    int oc = _SliceOutC;
    uint total = (uint)(ow * oh * od * oc);
    if (idx >= total) return;

    int x = (int)(idx % (uint)ow);
    uint t1 = idx / (uint)ow;
    int y = (int)(t1 % (uint)oh);
    uint t2 = t1 / (uint)oh;
    int z = (int)(t2 % (uint)od);
    int c = (int)(t2 / (uint)od);

    int ix = x;
    int iy = y;
    int iz = z;
    int ic = c;
    int begin = _SliceBegin;
    int axis = _SliceAxis;
    if (dims == 2)
    {
        if (axis == 0) ix = x + begin;
        else if (axis == 1) iy = y + begin;
        uint inIdx2 = (uint)(iy * _SliceInW + ix);
        _SliceOut[idx] = _SliceIn[inIdx2];
        return;
    }

    if (dims == 3)
    {
        if (axis == 0) ix = x + begin;
        else if (axis == 1) iy = y + begin;
        else if (axis == 2) ic = c + begin;
        uint inIdx3 = (uint)((ic * _SliceInH + iy) * _SliceInW + ix);
        _SliceOut[idx] = _SliceIn[inIdx3];
        return;
    }

    // dims == 4
    if (axis == 0) ix = x + begin;
    else if (axis == 1) iy = y + begin;
    else if (axis == 2) iz = z + begin;
    else if (axis == 3) ic = c + begin;
    uint inIdx4 = (uint)((((ic * _SliceInD + iz) * _SliceInH + iy) * _SliceInW) + ix);
    _SliceOut[idx] = _SliceIn[inIdx4];
}

void AexisTile_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    int dims = _TileDims;
    int ow = _TileOutW;
    int oh = _TileOutH;
    int od = _TileOutD;
    int oc = _TileOutC;
    uint total = (uint)(ow * oh * od * oc);
    if (idx >= total) return;

    int x = (int)(idx % (uint)ow);
    uint t1 = idx / (uint)ow;
    int y = (int)(t1 % (uint)oh);
    uint t2 = t1 / (uint)oh;
    int z = (int)(t2 % (uint)od);
    int c = (int)(t2 / (uint)od);

    int ix = x;
    int iy = y;
    int iz = z;
    int ic = c;
    int axis = _TileAxis;
    if (dims == 1)
    {
        uint inW1 = (uint)max(1, _TileInW);
        ix = (int)((uint)x % inW1);
        _TileOut[idx] = _TileIn[ix];
        return;
    }
    if (dims == 2)
    {
        uint inW2 = (uint)max(1, _TileInW);
        uint inH2 = (uint)max(1, _TileInH);
        if (axis == 0) iy = (int)((uint)y % inH2);
        else if (axis == 1) ix = (int)((uint)x % inW2);
        uint inIdx2 = (uint)(iy * _TileInW + ix);
        _TileOut[idx] = _TileIn[inIdx2];
        return;
    }
    if (dims == 3)
    {
        uint inW3 = (uint)max(1, _TileInW);
        uint inH3 = (uint)max(1, _TileInH);
        uint inC3 = (uint)max(1, _TileInC);
        if (axis == 0) ic = (int)((uint)c % inC3);
        else if (axis == 1) iy = (int)((uint)y % inH3);
        else if (axis == 2) ix = (int)((uint)x % inW3);
        uint inIdx3 = (uint)((ic * _TileInH + iy) * _TileInW + ix);
        _TileOut[idx] = _TileIn[inIdx3];
        return;
    }

    // dims == 4
    uint inW4 = (uint)max(1, _TileInW);
    uint inH4 = (uint)max(1, _TileInH);
    uint inD4 = (uint)max(1, _TileInD);
    uint inC4 = (uint)max(1, _TileInC);
    if (axis == 0) ic = (int)((uint)c % inC4);
    else if (axis == 1) iz = (int)((uint)z % inD4);
    else if (axis == 2) iy = (int)((uint)y % inH4);
    else if (axis == 3) ix = (int)((uint)x % inW4);
    uint inIdx4 = (uint)((((ic * _TileInD + iz) * _TileInH + iy) * _TileInW) + ix);
    _TileOut[idx] = _TileIn[inIdx4];
}

void AexisReduceSum256_Impl(uint3 groupId, uint3 groupThreadId)
{
    uint tid = groupThreadId.x;
    uint group = groupId.x;
    uint base = group * 256u;
    uint idx = base + tid;

    float v = 0.0;
    if (idx < (uint)_ReduceTotal)
        v = _ReduceIn[idx];
    _ReduceSum256_sdata[tid] = v;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u)
    {
        if (tid < stride)
            _ReduceSum256_sdata[tid] += _ReduceSum256_sdata[tid + stride];
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u)
        _ReduceOut[group] = _ReduceSum256_sdata[0];
}

void AexisMulScalarBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_MulTotal) return;
    _MulOut[idx] = _MulIn[idx] * _MulK;
}

void AexisCopyBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    _BufOut[idx] = _BufA[idx];
}

void AexisCopyBufPartial_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    _BufOut[idx + (uint)_DstOffset] = _BufA[idx + (uint)_SrcOffset];
}

void AexisBinaryOpBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;

    uint broadcastSize = (uint)max(_BinaryBroadcastSize, 1);
    uint channelIndex = broadcastSize > 0 ? (idx / broadcastSize) : 0;
    float a = _BinaryBroadcastMode == 1
        ? _BufA[idx % broadcastSize]
        : (_BinaryBroadcastMode == 3 ? _BufA[channelIndex] : _BufA[idx]);
    float b = _BinaryWithScalar != 0
        ? _BinaryScalar
        : (_BinaryBroadcastMode == 1
            ? _BufB[idx]
            : (_BinaryBroadcastMode == 2
                ? _BufB[idx % broadcastSize]
                : (_BinaryBroadcastMode == 4
                    ? _BufB[channelIndex]
                    : _BufB[idx])));

    _BufOut[idx] = AexisApplyBinaryOpScalar(a, b, _BinaryOpType);
}

void AexisUnaryOpBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;

    float x = _BufA[idx];
    float y = x;
    int t = _UnaryOpType;
    if (t == 0) y = abs(x);
    else if (t == 1) y = -x;
    else if (t == 2) y = floor(x);
    else if (t == 3) y = ceil(x);
    else if (t == 4) y = x * x;
    else if (t == 5) y = sqrt(x);
    else if (t == 6) y = rsqrt(x);
    else if (t == 7) y = exp(x);
    else if (t == 8) y = log(x);
    else if (t == 9) y = sin(x);
    else if (t == 10) y = cos(x);
    else if (t == 11) y = tan(x);
    else if (t == 12) y = asin(x);
    else if (t == 13) y = acos(x);
    else if (t == 14) y = atan(x);
    else if (t == 15) y = 1.0 / x;
    else if (t == 16) y = tanh(x);
    else if (t == 17) y = log(x) / log(10.0);
    else if (t == 18) y = round(x);
    else if (t == 19) y = trunc(x);
    else if (t == 20) y = sign(x);
    else if (t == 21) y = exp(x) - 1.0;
    else if (t == 22) y = sinh(x);
    else if (t == 23) y = log(x + sqrt(x * x + 1.0));
    else if (t == 24) y = cosh(x);
    else if (t == 25) y = log(x + sqrt(x * x - 1.0));
    else if (t == 26) y = 0.5 * log((1.0 + x) / (1.0 - x));
    else if (t == 27) y = log(1.0 + x);
    else if (t == 28) y = x == 0.0 ? 1.0 : 0.0; // Aexis ONNX logical Not extension.
    _BufOut[idx] = y;
}

void AexisSigmoidBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    float x = _BufA[idx];
    _BufOut[idx] = 1.0 / (1.0 + exp(-x));
}

void AexisSwishBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    float x = _BufA[idx];
    float s = 1.0 / (1.0 + exp(-x));
    _BufOut[idx] = x * s;
}

void AexisGeluBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    float x = _BufA[idx];
    float x3 = x * x * x;
    float t = clamp(0.7978845608 * (x + 0.044715 * x3), -10.0, 10.0);
    float y = 0.5 * x * (1.0 + tanh(t));
    _BufOut[idx] = y;
}

void AexisPointwiseBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    _BufOut[idx] = AexisApplyPointwiseScalar(_BufA[idx]);
}

void AexisCastBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    _QuantOut[idx] = AexisCastScalar(_QuantIn[idx]);
}

void AexisScaleBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    int axis = AexisBufferAxisIndex(_ScaleDims, _ScaleW, _ScaleH, _ScaleD, _ScaleC, (int)idx);
    float scale = AexisResolveScaleValue(axis);
    float bias = AexisResolveScaleBias(axis);
    _ScaleOutBuf[idx] = _ScaleInBuf[idx] * scale + bias;
}

void AexisPReluBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    int axis = AexisBufferAxisIndex(_PReluDims, _PReluW, _PReluH, _PReluD, _PReluC, (int)idx);
    float slope = AexisResolvePReluSlope(axis);
    float v = _PReluInBuf[idx];
    _PReluOutBuf[idx] = v < 0.0 ? v * slope : v;
}

void AexisReorgBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;

    int outW = max(1, _ReorgInW / max(1, _ReorgStride));
    int outH = max(1, _ReorgInH / max(1, _ReorgStride));
    int outPlane = outW * outH;
    int p = (int)(idx / (uint)outPlane);
    int rem = (int)(idx - (uint)(p * outPlane));
    int y = rem / outW;
    int x = rem - y * outW;

    int stride = max(1, _ReorgStride);
    int q;
    int sh;
    int sw;
    if (_ReorgMode == 0)
    {
        q = p / (stride * stride);
        int off = p - q * stride * stride;
        sh = off / stride;
        sw = off - sh * stride;
    }
    else
    {
        int off = p / max(1, _ReorgInC);
        q = p - off * _ReorgInC;
        sh = off / stride;
        sw = off - sh * stride;
    }

    int srcIdx = (q * _ReorgInH + y * stride + sh) * _ReorgInW + x * stride + sw;
    _ReorgOutBuf[idx] = _ReorgInBuf[srcIdx];
}

void AexisReductionRowsBuf_Impl(uint3 id)
{
    uint row = id.x;
    if (row >= (uint)_ReductionRowsOutCount) return;

    int elemCount = max(1, _ReductionRowsReduceElems);
    int baseIndex = (int)row * elemCount;
    float acc = _ReductionRowsIn[baseIndex];
    int opType = _ReductionRowsOpType;

    if (opType == 4)
    {
        for (int i = 1; i < elemCount; i++)
            acc = max(acc, _ReductionRowsIn[baseIndex + i]);
        _ReductionRowsOut[row] = acc * _ReductionRowsCoeff;
        return;
    }

    if (opType == 5)
    {
        for (int i = 1; i < elemCount; i++)
            acc = min(acc, _ReductionRowsIn[baseIndex + i]);
        _ReductionRowsOut[row] = acc * _ReductionRowsCoeff;
        return;
    }

    if (opType == 0 || opType == 3)
    {
        float sum = 0.0;
        for (int i = 0; i < elemCount; i++)
            sum += _ReductionRowsIn[baseIndex + i];
        if (opType == 3)
            sum /= elemCount;
        _ReductionRowsOut[row] = sum * _ReductionRowsCoeff;
        return;
    }

    if (opType == 2)
    {
        float sumSq = 0.0;
        for (int i = 0; i < elemCount; i++)
        {
            float v = _ReductionRowsIn[baseIndex + i];
            sumSq += v * v;
        }
        _ReductionRowsOut[row] = sumSq * _ReductionRowsCoeff;
        return;
    }

    if (opType == 1)
    {
        float sumAbs = 0.0;
        for (int i = 0; i < elemCount; i++)
            sumAbs += abs(_ReductionRowsIn[baseIndex + i]);
        _ReductionRowsOut[row] = sumAbs * _ReductionRowsCoeff;
        return;
    }

    if (opType == 6 || opType == 10)
    {
        float sumLog = 0.0;
        for (int i = 0; i < elemCount; i++)
            sumLog += log(max(_ReductionRowsIn[baseIndex + i], 1e-12));
        _ReductionRowsOut[row] = exp(sumLog) * _ReductionRowsCoeff;
        return;
    }

    if (opType == 9)
    {
        float sumVal = 0.0;
        for (int i = 0; i < elemCount; i++)
            sumVal += _ReductionRowsIn[baseIndex + i];
        _ReductionRowsOut[row] = log(max(sumVal, 1e-12)) * _ReductionRowsCoeff;
        return;
    }

    if (opType == 7)
    {
        float sumAbs = 0.0;
        for (int i = 0; i < elemCount; i++)
            sumAbs += abs(_ReductionRowsIn[baseIndex + i]);
        _ReductionRowsOut[row] = sumAbs * _ReductionRowsCoeff;
        return;
    }

    if (opType == 8)
    {
        float sumSq = 0.0;
        for (int i = 0; i < elemCount; i++)
        {
            float v = _ReductionRowsIn[baseIndex + i];
            sumSq += v * v;
        }
        _ReductionRowsOut[row] = sqrt(max(sumSq, 0.0)) * _ReductionRowsCoeff;
        return;
    }

    _ReductionRowsOut[row] = acc * _ReductionRowsCoeff;
}

void AexisConv1dBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;

    int outW = max(1, _Conv1dOutW);
    int oc = (int)(idx / (uint)outW);
    int x = (int)(idx - (uint)(oc * outW));
    float sum = _Conv1dB[oc];

    for (int ic = 0; ic < _Conv1dInC; ic++)
    {
        int srcBase = ic * _Conv1dInW;
        int weightBase = (oc * _Conv1dInC + ic) * _Conv1dKernelW;
        for (int k = 0; k < _Conv1dKernelW; k++)
        {
            int sx = x * _Conv1dStrideW - _Conv1dPadLeft + k * _Conv1dDilationW;
            if ((uint)sx >= (uint)_Conv1dInW)
                continue;
            sum += _Conv1dIn[srcBase + sx] * _Conv1dW[weightBase + k];
        }
    }

    _Conv1dOut[idx] = AexisApplyActivationScalar(sum);
}
