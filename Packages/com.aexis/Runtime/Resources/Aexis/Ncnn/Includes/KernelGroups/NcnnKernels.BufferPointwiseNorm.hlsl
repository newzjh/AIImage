// Auto-generated kernel implementation group: NcnnKernels.BufferPointwiseNorm.hlsl

void NcnnLayerNorm2D_Impl(uint3 groupId, uint3 groupThreadId)
{
    int row = (int)groupId.x;
    if (row < 0 || row >= _LnRows) return;
    int tid = (int)groupThreadId.x;
    int cols = max(1, _LnCols);
    int baseIndex = row * cols;

    float sum = 0.0;
    float sqsum = 0.0;
    for (int c0 = tid; c0 < cols; c0 += 256)
    {
        float x = _LnInOut[baseIndex + c0];
        sum += x;
        sqsum += x * x;
    }

    _Red0[tid] = sum;
    _Red1[tid] = sqsum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (int stLn = 128; stLn > 0; stLn >>= 1)
    {
        if (tid < stLn)
        {
            _Red0[tid] += _Red0[tid + stLn];
            _Red1[tid] += _Red1[tid + stLn];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float mean = _Red0[0] / cols;
    float var = _Red1[0] / cols - mean * mean;
    float invstd = rsqrt(max(var + _LnEps, 1e-20));

    for (int c1 = tid; c1 < cols; c1 += 256)
    {
        float x = _LnInOut[baseIndex + c1];
        float y = (x - mean) * invstd;
        if (_LnAffine != 0)
            y = y * _LnGamma[c1] + _LnBeta[c1];
        _LnInOut[baseIndex + c1] = y;
    }
}

void NcnnSoftmax2D_Impl(uint3 groupId, uint3 groupThreadId)
{
    int row = (int)groupId.x;
    if (row < 0 || row >= _SoftRows) return;
    int tid = (int)groupThreadId.x;
    int cols = max(1, _SoftCols);
    int baseIndex = row * cols;

    float maxv = -3.402823466e+38;
    for (int c0 = tid; c0 < cols; c0 += 256)
    {
        float x = _SoftIn[baseIndex + c0];
        maxv = max(maxv, x);
    }
    _Red0[tid] = maxv;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (int stSm0 = 128; stSm0 > 0; stSm0 >>= 1)
    {
        if (tid < stSm0)
            _Red0[tid] = max(_Red0[tid], _Red0[tid + stSm0]);
        GroupMemoryBarrierWithGroupSync();
    }
    float rowMax = _Red0[0];

    if (_SoftmaxMode == 2)
    {
        int firstMaximum = 0;
        for (int c = 0; c < cols; c++)
        {
            if (_SoftIn[baseIndex + c] == rowMax)
            {
                firstMaximum = c;
                break;
            }
        }
        for (int c = tid; c < cols; c += 256)
            _SoftOut[baseIndex + c] = c == firstMaximum ? 1.0 : 0.0;
        return;
    }

    float sum = 0.0;
    for (int c1 = tid; c1 < cols; c1 += 256)
    {
        float e = exp(_SoftIn[baseIndex + c1] - rowMax);
        sum += e;
        _SoftOut[baseIndex + c1] = e;
    }
    _Red1[tid] = sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (int stSm1 = 128; stSm1 > 0; stSm1 >>= 1)
    {
        if (tid < stSm1)
            _Red1[tid] += _Red1[tid + stSm1];
        GroupMemoryBarrierWithGroupSync();
    }
    float invSum = 1.0 / max(_Red1[0], 1e-20);

    for (int c2 = tid; c2 < cols; c2 += 256)
    {
        _SoftOut[baseIndex + c2] = _SoftmaxMode == 1
            ? _SoftIn[baseIndex + c2] - rowMax - log(max(_Red1[0], 1e-20))
            : _SoftOut[baseIndex + c2] * invSum;
    }
}

void NcnnQuantizeBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    int axis = NcnnQuantAxisIndex((int)idx);
    float scale = NcnnResolveQuantScaleIn(axis);
    _QuantOut[idx] = clamp(NcnnRoundToNearest(_QuantIn[idx] * scale), -127.0, 127.0);
}

void NcnnDequantizeBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    int axis = NcnnQuantAxisIndex((int)idx);
    float scale = NcnnResolveQuantScaleIn(axis);
    float bias = NcnnResolveQuantBias(axis);
    _QuantOut[idx] = NcnnRoundToNearest(_QuantIn[idx]) * scale + bias;
}

void NcnnRequantizeBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    int axis = NcnnQuantAxisIndex((int)idx);
    float scaleIn = NcnnResolveQuantScaleIn(axis);
    float scaleOut = NcnnResolveQuantScaleOut(axis);
    float bias = NcnnResolveQuantBias(axis);
    float v = NcnnRoundToNearest(_QuantIn[idx]) * scaleIn + bias;
    v = NcnnApplyActivationScalarQuant(v);
    _QuantOut[idx] = clamp(NcnnRoundToNearest(v * scaleOut), -127.0, 127.0);
}

void NcnnPixelShuffleBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;

    int outW = max(1, _PixelShuffleOutW);
    int outH = max(1, _PixelShuffleOutH);
    int outPlane = outW * outH;
    int p = (int)(idx / (uint)outPlane);
    int rem = (int)(idx - (uint)(p * outPlane));
    int y = rem / outW;
    int x = rem - y * outW;

    int scale = max(1, _PixelShuffleScale);
    int inX = x / scale;
    int inY = y / scale;
    int sh = y % scale;
    int sw = x % scale;
    int q;
    if (_PixelShuffleMode == 0)
        q = p * scale * scale + sh * scale + sw;
    else
        q = (sh * scale + sw) * _PixelShuffleOutC + p;

    int srcIdx = (q * _PixelShuffleInH + inY) * _PixelShuffleInW + inX;
    _PixelShuffleOut[idx] = _PixelShuffleIn[srcIdx];
}

void NcnnRotaryEmbedBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;

    int halfDim = max(1, _RotaryEmbedDim / 2);
    int headStride = _RotaryEmbedDim * _RotarySeqLen;
    int head = (int)(idx / (uint)(halfDim * _RotarySeqLen));
    int rem = (int)(idx - (uint)(head * halfDim * _RotarySeqLen));
    int seq = rem / halfDim;
    int j = rem - seq * halfDim;

    int rowBase = head * headStride + seq * _RotaryEmbedDim;
    int cacheBase = seq * halfDim + j;
    float cosVal = _RotaryCos[cacheBase];
    float sinVal = _RotarySin[cacheBase];

    if (_RotaryInterleaved != 0)
    {
        int i0 = rowBase + j * 2;
        int i1 = i0 + 1;
        if (i1 >= rowBase + _RotaryEmbedDim)
            return;
        float x0 = _RotaryIn[i0];
        float x1 = _RotaryIn[i1];
        _RotaryOut[i0] = x0 * cosVal - x1 * sinVal;
        _RotaryOut[i1] = x0 * sinVal + x1 * cosVal;
    }
    else
    {
        int i0 = rowBase + j;
        int i1 = rowBase + halfDim + j;
        if (i1 >= rowBase + _RotaryEmbedDim)
            return;
        float x0 = _RotaryIn[i0];
        float x1 = _RotaryIn[i1];
        _RotaryOut[i0] = x0 * cosVal - x1 * sinVal;
        _RotaryOut[i1] = x0 * sinVal + x1 * cosVal;
    }
}

void NcnnNormalizeBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;

    int plane = max(1, _NormW * _NormH * _NormD);
    int channels = _NormDims >= 3 ? max(1, _NormC) : 1;
    int q = (int)(idx / (uint)plane);
    int s = (int)(idx - (uint)(q * plane));

    float sumSquare = 0.0;

    if (_NormAcrossSpatial != 0 && _NormAcrossChannel != 0)
    {
        for (int i = 0; i < _Total; i++)
        {
            float v = _NormIn[i];
            sumSquare += v * v;
        }
    }
    else if (_NormAcrossSpatial != 0)
    {
        int baseIndex = q * plane;
        for (int i = 0; i < plane; i++)
        {
            float v = _NormIn[baseIndex + i];
            sumSquare += v * v;
        }
    }
    else if (_NormAcrossChannel != 0)
    {
        for (int c = 0; c < channels; c++)
        {
            float v = _NormIn[c * plane + s];
            sumSquare += v * v;
        }
    }
    else
    {
        _NormOut[idx] = _NormIn[idx];
        return;
    }

    float scale = NcnnResolveNormScale(q);
    float a = NcnnComputeNormAlpha(sumSquare) * scale;
    _NormOut[idx] = _NormIn[idx] * a;
}

void NcnnLrnBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;

    int w = max(1, _LrnW);
    int h = max(1, _LrnH);
    int c = max(1, _LrnC);
    int size = w * h;
    int q = (int)(idx / (uint)size);
    int rem = (int)(idx - (uint)(q * size));
    int y = rem / w;
    int x = rem - y * w;

    float sumSquare = 0.0;

    if (_LrnRegionType == 0)
    {
        int half = _LrnLocalSize / 2;
        int p0 = max(0, q - half);
        int p1 = min(c - 1, q + half);
        for (int p = p0; p <= p1; p++)
        {
            float v = _LrnIn[p * size + rem];
            sumSquare += v * v;
        }

        float norm = pow(_LrnBias + (_LrnAlpha / max(1, _LrnLocalSize)) * sumSquare, -_LrnBeta);
        _LrnOut[idx] = _LrnIn[idx] * norm;
        return;
    }

    int pad = _LrnLocalSize / 2;
    for (int ky = 0; ky < _LrnLocalSize; ky++)
    {
        int sy = y + ky - pad;
        if (sy < 0 || sy >= h)
            continue;

        for (int kx = 0; kx < _LrnLocalSize; kx++)
        {
            int sx = x + kx - pad;
            if (sx < 0 || sx >= w)
                continue;

            float v = _LrnIn[q * size + sy * w + sx];
            sumSquare += v * v;
        }
    }

    float maxk = max(1, _LrnLocalSize * _LrnLocalSize);
    float normWithin = pow(_LrnBias + (_LrnAlpha / maxk) * sumSquare, -_LrnBeta);
    _LrnOut[idx] = _LrnIn[idx] * normWithin;
}

void NcnnRmsNormBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;

    int segmentBase;
    int segmentSize;
    int innerIndex;
    NcnnDecodeRmsNormSegment((int)idx, segmentBase, segmentSize, innerIndex);

    float sqsum = 0.0;
    for (int i = 0; i < segmentSize; i++)
    {
        float v = _RmsNormIn[segmentBase + i];
        sqsum += v * v;
    }

    float rms = sqsum / max(1, segmentSize);
    float y = _RmsNormIn[idx] * rsqrt(max(rms + _RmsNormEps, 1e-20));
    if (_RmsNormAffine != 0 && _RmsNormAffineSize > 0)
        y *= _RmsNormGamma[clamp(innerIndex, 0, _RmsNormAffineSize - 1)];
    _RmsNormOut[idx] = y;
}

void NcnnUnfoldBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;

    int outw = max(1, _UnfoldOutW);
    int outh = max(1, _UnfoldOutH);
    int size = outw * outh;
    int maxk = max(1, _UnfoldKernelW * _UnfoldKernelH);

    int outRow = (int)(idx / (uint)size);
    int rem = (int)(idx - (uint)(outRow * size));
    int oy = rem / outw;
    int ox = rem - oy * outw;

    int c = outRow / maxk;
    int k = outRow - c * maxk;
    int u = k / max(1, _UnfoldKernelW);
    int v = k - u * _UnfoldKernelW;

    int inY = oy * _UnfoldStrideH + u * _UnfoldDilationH - _UnfoldPadTop;
    int inX = ox * _UnfoldStrideW + v * _UnfoldDilationW - _UnfoldPadLeft;

    float value = _UnfoldPadValue;
    if ((uint)inX < (uint)_UnfoldInW && (uint)inY < (uint)_UnfoldInH && c < _UnfoldInC)
    {
        int srcIdx = (c * _UnfoldInH + inY) * _UnfoldInW + inX;
        value = _UnfoldIn[srcIdx];
    }

    _UnfoldOut[idx] = value;
}

void NcnnTouchU32_Impl(uint3 id)
{
    _TouchOut[0] = (uint)_TouchValue;
}
