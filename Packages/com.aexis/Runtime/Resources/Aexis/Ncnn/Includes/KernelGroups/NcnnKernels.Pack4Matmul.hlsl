// Auto-generated kernel implementation group: NcnnKernels.Pack4Matmul.hlsl

#define NCNN_SDPA_Q_CACHE_FLOATS 1024
groupshared float _SdpaQCache[NCNN_SDPA_Q_CACHE_FLOATS];

float NcnnApplyUnaryOpLinearScalar(float x)
{
    float y = x;
    int t = _UnaryOpType;
    if (t == 0) y = abs(x);
    else if (t == 1) y = -x;
    else if (t == 2) y = floor(x);
    else if (t == 3) y = ceil(x);
    else if (t == 4) y = x * x;
    else if (t == 5) y = sqrt(max(x, 0.0));
    else if (t == 6) y = rsqrt(max(x, 1e-12));
    else if (t == 7) y = exp(x);
    else if (t == 8) y = log(max(x, 1e-12));
    else if (t == 9) y = sin(x);
    else if (t == 10) y = cos(x);
    else if (t == 11) y = tan(x);
    else if (t == 15) y = 1.0 / max(x, 1e-12);
    else if (t == 16) y = tanh(x);
    return y;
}

float NcnnApplyBinaryOpLinearScalar(float a, float b)
{
    float o = a;
    int t = _BinaryOpType;
    if (t == 0) o = a + b;
    else if (t == 1) o = a - b;
    else if (t == 2) o = a * b;
    else if (t == 3) o = a / b;
    else if (t == 4) o = max(a, b);
    else if (t == 5) o = min(a, b);
    else if (t == 6) o = pow(abs(a), b);
    else if (t == 7) o = b - a;
    else if (t == 8) o = b / a;
    else if (t == 9) o = pow(abs(b), a);
    return o;
}

float NcnnApplySwishLinearScalar(float x)
{
    return x / (1.0 + exp(-x));
}

float NcnnApplySigmoidLinearScalar(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

float NcnnApplyGeluLinearScalar(float x)
{
    float x3 = x * x * x;
    float t = clamp(0.79788452 * (x + 0.044715 * x3), -10.0, 10.0);
    return 0.5 * x * (1.0 + tanh(t));
}

void NcnnSwishPack4_Impl(uint3 id)
{
    uint w, h, d;
    _ActOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 x = _ActInArr[int3((int)id.x, (int)id.y, p)];
    float4 y = x / (1.0 + exp(-x));
    _ActOutArr[int3((int)id.x, (int)id.y, p)] = y;
}

void NcnnSwishLinearMat_Impl(uint3 id)
{
    uint w, h;
    _LinearOut0.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;

    int2 coord = int2((int)id.x, (int)id.y);
    float x = _LinearIn0[coord];
    _LinearOut0[coord] = NcnnApplySwishLinearScalar(x);
}

void NcnnSigmoidPack4_Impl(uint3 id)
{
    uint w, h, d;
    _ActOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 x = _ActInArr[int3((int)id.x, (int)id.y, p)];
    float4 y = 1.0 / (1.0 + exp(-x));
    _ActOutArr[int3((int)id.x, (int)id.y, p)] = y;
}

void NcnnSigmoidLinearMat_Impl(uint3 id)
{
    uint w, h;
    _LinearOut0.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;

    int2 coord = int2((int)id.x, (int)id.y);
    float x = _LinearIn0[coord];
    _LinearOut0[coord] = NcnnApplySigmoidLinearScalar(x);
}

void NcnnGeluPack4_Impl(uint3 id)
{
    uint w, h, d;
    _ActOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 x = _ActInArr[int3((int)id.x, (int)id.y, p)];
    float4 x3 = x * x * x;
    float4 t = clamp(0.79788452 * (x + 0.044715 * x3), -10.0, 10.0);
    float4 y = 0.5 * x * (1.0 + tanh(t));
    _ActOutArr[int3((int)id.x, (int)id.y, p)] = y;
}

void NcnnGeluLinearMat_Impl(uint3 id)
{
    uint w, h;
    _LinearOut0.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;

    int2 coord = int2((int)id.x, (int)id.y);
    float x = _LinearIn0[coord];
    _LinearOut0[coord] = NcnnApplyGeluLinearScalar(x);
}

void NcnnUnaryOpLinearMat_Impl(uint3 id)
{
    uint w, h;
    _LinearOut0.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;

    int2 coord = int2((int)id.x, (int)id.y);
    float x = _LinearIn0[coord];
    _LinearOut0[coord] = NcnnApplyUnaryOpLinearScalar(x);
}

void NcnnBinaryOpLinearMat_Impl(uint3 id)
{
    uint w, h;
    _LinearOut0.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;

    int2 coord = int2((int)id.x, (int)id.y);
    float a = _LinearIn0[coord];
    float b = _BinaryWithScalar != 0 ? _BinaryScalar : _LinearIn1[coord];
    _LinearOut0[coord] = NcnnApplyBinaryOpLinearScalar(a, b);
}

// The scalar is a fixed graph input, uploaded once in a ComputeBuffer. The
// activation and output remain Texture2D LinearMat resources.
void NcnnBinaryOpLinearMatFixedInputScalar_Impl(uint3 id)
{
    uint w, h;
    _LinearOut0.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;

    int2 coord = int2((int)id.x, (int)id.y);
    float textureValue = _LinearIn0[coord];
    float scalarValue = _BufB[0];
    float a = _BinaryPack4BufferScalarMode == 1 ? scalarValue : textureValue;
    float b = _BinaryPack4BufferScalarMode == 1 ? textureValue : scalarValue;
    _LinearOut0[coord] = NcnnApplyBinaryOpLinearScalar(a, b);
}

void NcnnBinaryOpPack4LinearMixed_Impl(uint3 id)
{
    uint w, h, d;
    _TexOut0Arr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h || id.z >= d)
        return;

    int packX = (int)id.x;
    int row = (int)id.y;
    int baseCol = (((int)id.z * (int)w) + packX) * 4;
    float4 a = _TexIn0Arr[int3(packX, row, (int)id.z)];
    float4 o = 0.0;

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int col = baseCol + lane;
        if (_BinaryWithScalar == 0)
        {
            uint linearW, linearH;
            _LinearIn1.GetDimensions(linearW, linearH);
            if (col >= (int)linearW || row >= (int)linearH)
                continue;
        }
        float packValue = NcnnReadLane(a, lane);
        float linearValue = _BinaryWithScalar != 0 ? _BinaryScalar : _LinearIn1[int2(col, row)];
        float lhs = _BinaryPack4LinearMixedMode == 2 ? linearValue : packValue;
        float rhs = _BinaryPack4LinearMixedMode == 2 ? packValue : linearValue;
        NcnnWriteLane(o, lane, NcnnApplyBinaryOpLinearScalar(lhs, rhs));
    }

    _TexOut0Arr[int3(packX, row, (int)id.z)] = o;
}

void NcnnMatMul2D_Impl(uint3 groupId, uint3 groupThreadId)
{
    int lx = (int)groupThreadId.x;
    int ly = (int)groupThreadId.y;
    int row = (int)groupId.y * TILE + ly;
    int col = (int)groupId.x * TILE + lx;
    bool valid = (row < _MatM && col < _MatN);

    float acc = 0.0;
    for (int k0 = 0; k0 < _MatK; k0 += TILE)
    {
        int ak = k0 + lx;
        int bk = k0 + ly;
        float a = 0.0;
        float b = 0.0;

        if (row < _MatM && ak < _MatK)
            a = _MatA[row * _MatK + ak];
        if (bk < _MatK && col < _MatN)
        {
            if (_MatTransB != 0)
                b = _MatB[col * _MatK + bk];
            else
                b = _MatB[bk * _MatN + col];
        }

        _MatAs[ly * TILE + lx] = a;
        _MatBs[ly * TILE + lx] = b;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (int k1 = 0; k1 < TILE; k1++)
        {
            int kk = k0 + k1;
            if (kk < _MatK)
                acc += _MatAs[ly * TILE + k1] * _MatBs[k1 * TILE + lx];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (valid)
        _MatOut[row * _MatN + col] = acc;
}

void NcnnMatMulPack4CDHW_Impl(uint3 id)
{
    uint ow, oh, od;
    _MatPack4OutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int outCol = (int)id.x;
    int outRow = (int)id.y;
    int slice = (int)id.z;
    int outPackCount = max(1, (_MatPack4OutBatchC + 3) / 4);
    int outBatchD = slice / outPackCount;
    int outBatchCPack = slice - outBatchD * outPackCount;
    if (outBatchD < 0 || outBatchD >= _MatPack4OutBatchD)
    {
        _MatPack4OutArr[int3(outCol, outRow, slice)] = 0.0;
        return;
    }

    float4 o = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int outBatchC = outBatchCPack * 4 + lane;
        if (outBatchC >= _MatPack4OutBatchC)
            continue;

        int aBatchD = min(_MatPack4ABatchD - 1, (_MatPack4ABatchD > 1) ? outBatchD : 0);
        int aBatchC = min(_MatPack4ABatchC - 1, (_MatPack4ABatchC > 1) ? outBatchC : 0);
        int bBatchD = min(_MatPack4BBatchD - 1, (_MatPack4BBatchD > 1) ? outBatchD : 0);
        int bBatchC = min(_MatPack4BBatchC - 1, (_MatPack4BBatchC > 1) ? outBatchC : 0);

        float acc = 0.0;
        [loop]
        for (int kk = 0; kk < _MatK; kk++)
        {
            float a = NcnnReadPack4ChannelCDHW(_MatPack4AInArr, kk, outRow, aBatchD, aBatchC, _MatPack4ABatchC);
            float b = _MatTransB != 0
                ? NcnnReadPack4ChannelCDHW(_MatPack4BInArr, kk, outCol, bBatchD, bBatchC, _MatPack4BBatchC)
                : NcnnReadPack4ChannelCDHW(_MatPack4BInArr, outCol, kk, bBatchD, bBatchC, _MatPack4BBatchC);
            acc += a * b;
        }

        NcnnWriteLane(o, lane, acc);
    }

    _MatPack4OutArr[int3(outCol, outRow, slice)] = o;
}

void NcnnSdpaAttentionPack4CDHW_Impl(uint3 groupId, uint3 groupThreadId)
{
    int row = (int)groupId.x;
    int headPack = (int)groupId.y;
    int outChunk = (int)groupId.z;
    uint tid = groupThreadId.x;
    if (row < 0 || row >= _SdpaSrcLen || headPack < 0)
        return;

    int dst = _SdpaDstLen;
    int embedDim = _SdpaEmbedDim;
    int outEmbedDim = _SdpaOutEmbedDim;
    if (dst <= 0 || dst > 4096 || embedDim <= 0 || outEmbedDim <= 0)
        return;

    int headsPerGroup = max(1, _SdpaNumHeadsPerGroup);
    int outX = outChunk * 64 + (int)tid;
    float4 outValue = 0.0;
    bool useQCache = embedDim * 4 <= NCNN_SDPA_Q_CACHE_FLOATS;

    if (useQCache)
    {
        int cacheCount = embedDim * 4;
        for (int cacheIndex = (int)tid; cacheIndex < cacheCount; cacheIndex += 64)
        {
            int laneForCache = cacheIndex / embedDim;
            int iForCache = cacheIndex - laneForCache * embedDim;
            int headForCache = headPack * 4 + laneForCache;
            bool validHeadForCache = headForCache >= 0 && headForCache < _SdpaNumHeads;
            _SdpaQCache[cacheIndex] = validHeadForCache
                ? NcnnReadPack4ChannelCDHW(_SdpaQArr, iForCache, row, 0, headForCache, _SdpaNumHeads)
                : 0.0;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int head = headPack * 4 + lane;
        bool validHead = head >= 0 && head < _SdpaNumHeads;
        int keyHead = validHead ? min(max(0, _SdpaNumGroups - 1), head / headsPerGroup) : 0;

        for (int colScore = (int)tid; colScore < dst; colScore += 64)
        {
            float s = 0.0;
            if (validHead)
            {
                [loop]
                for (int i = 0; i < embedDim; i++)
                {
                    float q = useQCache
                        ? _SdpaQCache[lane * embedDim + i]
                        : NcnnReadPack4ChannelCDHW(_SdpaQArr, i, row, 0, head, _SdpaNumHeads);
                    float k = NcnnReadPack4ChannelCDHW(_SdpaKArr, i, colScore, 0, keyHead, _SdpaNumGroups);
                    s += q * k;
                }
                s *= _SdpaScale;
                if (_SdpaCausal != 0 && colScore > row)
                    s = -1.0e20;
                else if (_SdpaHasTextureMask != 0
                    && colScore < _SdpaTextureMaskW
                    && row < _SdpaTextureMaskH)
                    s += NcnnReadPack4ChannelCDHW(_SdpaMaskArr, colScore, row, 0, 0, 1);
            }
            _SdpaScoresFast[colScore] = s;
        }
        GroupMemoryBarrierWithGroupSync();

        float localMax = -3.402823466e+38;
        for (int colMax = (int)tid; colMax < dst; colMax += 64)
            localMax = max(localMax, _SdpaScoresFast[colMax]);
        _SdpaReduce[tid] = localMax;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint stride = 32; stride > 0; stride >>= 1)
        {
            if (tid < stride)
                _SdpaReduce[tid] = max(_SdpaReduce[tid], _SdpaReduce[tid + stride]);
            GroupMemoryBarrierWithGroupSync();
        }

        float maxValue = _SdpaReduce[0];
        float localSum = 0.0;
        for (int colExp = (int)tid; colExp < dst; colExp += 64)
        {
            float e = exp(_SdpaScoresFast[colExp] - maxValue);
            _SdpaScoresFast[colExp] = e;
            localSum += e;
        }
        _SdpaReduce[tid] = localSum;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint stride2 = 32; stride2 > 0; stride2 >>= 1)
        {
            if (tid < stride2)
                _SdpaReduce[tid] += _SdpaReduce[tid + stride2];
            GroupMemoryBarrierWithGroupSync();
        }

        float invSum = 1.0 / max(_SdpaReduce[0], 1e-20);
        for (int colNorm = (int)tid; colNorm < dst; colNorm += 64)
            _SdpaScoresFast[colNorm] *= invSum;
        GroupMemoryBarrierWithGroupSync();

        if (validHead && outX < outEmbedDim)
        {
            float sum = 0.0;
            [loop]
            for (int colValue = 0; colValue < dst; colValue++)
            {
                float v = NcnnReadPack4ChannelCDHW(_SdpaVArr, outX, colValue, 0, keyHead, _SdpaNumGroups);
                sum += _SdpaScoresFast[colValue] * v;
            }
            NcnnWriteLane(outValue, lane, sum);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (outX < outEmbedDim)
        _SdpaOutArr[int3(outX, row, headPack)] = outValue;
}

void NcnnVistaTailPromptDotPack4_Impl(uint3 id)
{
    uint w, h, slices;
    _VistaTailOutArr.GetDimensions(w, h, slices);
    if (id.x >= w || id.y >= h || id.z >= slices)
        return;

    int x = (int)id.x;
    int y = (int)id.y;
    int z = (int)id.z;
    if (x < 0 || x >= _VistaTailW || y < 0 || y >= _VistaTailH || z < 0 || z >= _VistaTailD)
        return;

    float sum = 0.0;
    int promptBase = 0;
    int featureSliceBase = z * _VistaTailPacks;
    [loop]
    for (int pack = 0; pack < _VistaTailPacks; pack++)
    {
        float4 feat = _VistaTailInArr[int3(x, y, featureSliceBase + pack)];
        float4 prompt4 = float4(
            _VistaTailPrompt[promptBase + 0],
            _VistaTailPrompt[promptBase + 1],
            _VistaTailPrompt[promptBase + 2],
            _VistaTailPrompt[promptBase + 3]);
        sum += dot(feat, prompt4);
        promptBase += 4;
    }

    _VistaTailOutArr[int3(x, y, z)] = float4(sum, 0.0, 0.0, 0.0);
}

void NcnnVistaTailPromptDotPack4Tex_Impl(uint3 id)
{
    uint w, h, slices;
    _VistaTailOutArr.GetDimensions(w, h, slices);
    if (id.x >= w || id.y >= h || id.z >= slices)
        return;

    int x = (int)id.x;
    int y = (int)id.y;
    int z = (int)id.z;
    if (x < 0 || x >= _VistaTailW || y < 0 || y >= _VistaTailH || z < 0 || z >= _VistaTailD)
        return;

    float sum = 0.0;
    int featureSliceBase = z * _VistaTailPacks;
    [loop]
    for (int pack = 0; pack < _VistaTailPacks; pack++)
    {
        float4 feat = _VistaTailInArr[int3(x, y, featureSliceBase + pack)];
        float4 prompt4 = _VistaTailPromptTex[int3(0, 0, pack)];
        sum += dot(feat, prompt4);
    }

    _VistaTailOutArr[int3(x, y, z)] = float4(sum, 0.0, 0.0, 0.0);
}

void NcnnArgmaxUpdatePack4CDHW_Impl(uint3 id)
{
    uint w, h, outputDepth;
    _ArgmaxBestValueArr.GetDimensions(w, h, outputDepth);
    if ((int)id.x >= (int)w || (int)id.y >= (int)h || (int)id.z >= _ArgmaxInD || (int)id.z >= (int)outputDepth)
        return;

    int packCount = max(1, (_ArgmaxInC + 3) / 4);
    int z = (int)id.z;
    float bestValue = _ArgmaxInitialize != 0
        ? -3.402823466e+38
        : _ArgmaxBestValueArr[int3((int)id.x, (int)id.y, z)].x;
    float bestLabel = _ArgmaxInitialize != 0
        ? (float)_ArgmaxChannelOffset
        : _ArgmaxBestLabelArr[int3((int)id.x, (int)id.y, z)].x;

    [loop]
    for (int pack = 0; pack < packCount; pack++)
    {
        int slice = z * packCount + pack;
        float4 logits = _ArgmaxPack4InArr[int3((int)id.x, (int)id.y, slice)];
        [unroll]
        for (int lane = 0; lane < 4; lane++)
        {
            int localChannel = pack * 4 + lane;
            if (localChannel < 0 || localChannel >= _ArgmaxInC)
                continue;

            float value = NcnnReadLane(logits, lane);
            if (value > bestValue)
            {
                bestValue = value;
                bestLabel = (float)(_ArgmaxChannelOffset + localChannel);
            }
        }
    }

    _ArgmaxBestValueArr[int3((int)id.x, (int)id.y, z)] = float4(bestValue, 0.0, 0.0, 0.0);
    _ArgmaxBestLabelArr[int3((int)id.x, (int)id.y, z)] = float4(bestLabel, 0.0, 0.0, 0.0);
}

void NcnnArgMaxPack4LinearMat_Impl(uint3 id)
{
    int row = (int)id.y;
    if (id.x != 0 || row < 0 || row >= _ArgmaxLogicalHeight)
        return;

    uint storageWidth, storageHeight, storageSlices;
    _ArgmaxPack4InArr.GetDimensions(storageWidth, storageHeight, storageSlices);
    int inputWidth = max(1, (int)storageWidth);
    int tileRows = max(1, _ArgmaxTileRowsPerMatrix);
    int packCount = max(1, (_ArgmaxLogicalWidth + 3) / 4);
    int bestIndex = 0;
    float bestValue = 0.0;

    [loop]
    for (int pack = 0; pack < packCount; pack++)
    {
        int tile = pack / inputWidth;
        int x = pack - tile * inputWidth;
        int y = tileRows > 1 ? row * tileRows + tile : row;
        int slice = tileRows > 1 ? 0 : tile;
        if (y < 0 || y >= (int)storageHeight || slice < 0 || slice >= (int)storageSlices)
            continue;

        float4 values = _ArgmaxPack4InArr[int3(x, y, slice)];
        [unroll]
        for (int lane = 0; lane < 4; lane++)
        {
            int index = pack * 4 + lane;
            if (index >= _ArgmaxLogicalWidth)
                break;
            float value = NcnnReadLane(values, lane);
            if (index == 0 || value > bestValue)
            {
                bestValue = value;
                bestIndex = index;
            }
        }
    }

    _LinearOut0[int2(0, row)] = (float)bestIndex;
}

float NcnnResolveGemm2DBias(int row, int col)
{
    float sum = 0.0;
    if (_MatUseC != 0)
    {
        if (_MatBroadcastTypeC == 0)
            sum = _MatC[0];
        else if (_MatBroadcastTypeC == 1)
            sum = _MatC[row];
        else if (_MatBroadcastTypeC == 3)
            sum = _MatC[row * _MatN + col];
        else if (_MatBroadcastTypeC == 4)
            sum = _MatC[col];
        sum *= _MatBeta;
    }

    return sum;
}

float NcnnGemm2DReadB(int col, int kk)
{
    int index = _MatTransB != 0
        ? col * _MatK + kk
        : kk * _MatN + col;
    if (_UseInt4GemmWeights != 0)
    {
        uint packed = _MatBInt4Packed[index >> 3];
        uint raw = (packed >> ((index & 7) * 4)) & 0xfu;
        int signedValue = raw >= 8u ? (int)raw - 16 : (int)raw;
        return (float)signedValue * _MatBInt4Scales[col];
    }
    if (_UseInt8GemmWeights != 0)
    {
        uint packed = _MatBInt8Packed[index >> 2];
        uint raw = (packed >> ((index & 3) * 8)) & 0xffu;
        int signedValue = raw >= 128u ? (int)raw - 256 : (int)raw;
        return (float)signedValue * _MatBInt8Scales[col];
    }
    if (_UseFp16GemmWeights == 0)
        return _MatB[index];

    uint packed = _MatBFp16[index >> 1];
    return f16tof32((index & 1) == 0 ? (packed & 0xffffu) : (packed >> 16));
}

int NcnnGemmPack4LinearOutPacksPerThread()
{
    return clamp(_GemmTexOutsPerThread, 1, 2);
}

float4 NcnnGemm2DReadB4(int baseCol, int kk)
{
    float4 b = 0.0;
    if (baseCol < _MatN) b.x = NcnnGemm2DReadB(baseCol, kk);
    if (baseCol + 1 < _MatN) b.y = NcnnGemm2DReadB(baseCol + 1, kk);
    if (baseCol + 2 < _MatN) b.z = NcnnGemm2DReadB(baseCol + 2, kk);
    if (baseCol + 3 < _MatN) b.w = NcnnGemm2DReadB(baseCol + 3, kk);
    return b;
}

bool NcnnCanReadPackedInt8TransB4()
{
    return _UseInt8GemmWeights != 0
        && _MatTransB != 0
        && (_MatK & 3) == 0;
}

uint4 NcnnLoadPackedInt8TransB4(int baseCol, int kkBase)
{
    return uint4(
        _MatBInt8Packed[(baseCol * _MatK + kkBase) >> 2],
        _MatBInt8Packed[((baseCol + 1) * _MatK + kkBase) >> 2],
        _MatBInt8Packed[((baseCol + 2) * _MatK + kkBase) >> 2],
        _MatBInt8Packed[((baseCol + 3) * _MatK + kkBase) >> 2]);
}

float NcnnDecodePackedInt8Weight(uint packed, int shift)
{
    uint raw = (packed >> shift) & 0xffu;
    return (float)(raw >= 128u ? (int)raw - 256 : (int)raw);
}

float4 NcnnReadPackedInt8Scales4(int baseCol)
{
    return float4(
        _MatBInt8Scales[baseCol],
        _MatBInt8Scales[baseCol + 1],
        _MatBInt8Scales[baseCol + 2],
        _MatBInt8Scales[baseCol + 3]);
}

float4 NcnnDecodePackedInt8TransB4(uint4 packed, int kLane, float4 scales)
{
    int shift = kLane << 3;
    return float4(
        NcnnDecodePackedInt8Weight(packed.x, shift),
        NcnnDecodePackedInt8Weight(packed.y, shift),
        NcnnDecodePackedInt8Weight(packed.z, shift),
        NcnnDecodePackedInt8Weight(packed.w, shift)) * scales;
}

float NcnnQuantizeGemmActivationForInt8(float value)
{
    if (_UseInt8Activations == 0)
        return value;
    int quantized = clamp((int)round(value / _Int8ActivationScale) + _Int8ActivationZeroPoint, -128, 127);
    return (float)(quantized - _Int8ActivationZeroPoint) * _Int8ActivationScale;
}

float4 NcnnQuantizeGemmActivationForInt8(float4 value)
{
    if (_UseInt8Activations == 0)
        return value;
    return float4(
        NcnnQuantizeGemmActivationForInt8(value.x),
        NcnnQuantizeGemmActivationForInt8(value.y),
        NcnnQuantizeGemmActivationForInt8(value.z),
        NcnnQuantizeGemmActivationForInt8(value.w));
}

float4 NcnnResolveGemm2DBias4(int row, int baseCol)
{
    float4 bias = 0.0;
    if (baseCol < _MatN) bias.x = NcnnResolveGemm2DBias(row, baseCol);
    if (baseCol + 1 < _MatN) bias.y = NcnnResolveGemm2DBias(row, baseCol + 1);
    if (baseCol + 2 < _MatN) bias.z = NcnnResolveGemm2DBias(row, baseCol + 2);
    if (baseCol + 3 < _MatN) bias.w = NcnnResolveGemm2DBias(row, baseCol + 3);
    return bias;
}

float NcnnGemm2DTextureAValue(int row, int col)
{
    float acc = 0.0;
    [loop]
    for (int kk = 0; kk < _MatK; kk++)
    {
        float a = 0.0;
        if (kk >= 0 && kk < _GemmTexAInW && row >= 0 && row < _GemmTexAInH)
            a = _GemmTexAInArr[int3(kk, row, 0)].x;

        a = NcnnQuantizeGemmActivationForInt8(a);
        acc += a * NcnnGemm2DReadB(col, kk);
    }

    return NcnnResolveGemm2DBias(row, col) + acc * _MatAlpha;
}

float NcnnGemm2DLinearTextureAValue(int row, int col)
{
    float acc = 0.0;
    [loop]
    for (int kk = 0; kk < _MatK; kk++)
    {
        float a = 0.0;
        if (kk >= 0 && kk < _GemmTexAInW && row >= 0 && row < _GemmTexAInH)
            a = _LinearIn0[int2(kk, row)];

        a = NcnnQuantizeGemmActivationForInt8(a);
        acc += a * NcnnGemm2DReadB(col, kk);
    }

    return NcnnResolveGemm2DBias(row, col) + acc * _MatAlpha;
}

float NcnnGemm2DPack4LinearTextureARead(int row, int kk)
{
    if (kk < 0 || kk >= _MatK || row < 0 || row >= _GemmTexAInH)
        return 0.0;

    if (_GemmTexAInW == _MatK)
        return NcnnQuantizeGemmActivationForInt8(_LinearIn0[int2(kk, row)]);

    int logicalPack = kk >> 2;
    int packX = logicalPack % max(1, _GemmTexAInW);
    int packSlice = logicalPack / max(1, _GemmTexAInW);
    if (packX < 0 || packX >= _GemmTexAInW)
        return 0.0;

    return NcnnQuantizeGemmActivationForInt8(NcnnReadLane(_GemmTexAInArr[int3(packX, row, packSlice)], kk & 3));
}

float NcnnGemm2DPack4LinearTextureAReadLinear(int row, int kk)
{
    if (kk < 0 || kk >= _MatK || kk >= _GemmTexAInW || row < 0 || row >= _GemmTexAInH)
        return 0.0;
    return NcnnQuantizeGemmActivationForInt8(_LinearIn0[int2(kk, row)]);
}

float NcnnGemm2DPack4LinearTextureAReadPack4(int row, int kk)
{
    if (kk < 0 || kk >= _MatK || row < 0 || row >= _GemmTexAInH)
        return 0.0;

    int logicalPack = kk >> 2;
    int packX = logicalPack % max(1, _GemmTexAInW);
    int packSlice = logicalPack / max(1, _GemmTexAInW);
    if (packX < 0 || packX >= _GemmTexAInW)
        return 0.0;

    return NcnnQuantizeGemmActivationForInt8(NcnnReadLane(_GemmTexAInArr[int3(packX, row, packSlice)], kk & 3));
}

void NcnnGemm2DTextureA_Impl(uint3 id)
{
    uint ow, oh, od;
    _GemmTexOutArr.GetDimensions(ow, oh, od);
    int outsPerThread = clamp(_GemmTexOutsPerThread, 1, 4);
    int baseCol = (int)id.x * outsPerThread;
    if (baseCol >= (int)ow || id.y >= oh || id.z >= od)
        return;

    int row = (int)id.y;
    if (row < 0 || row >= _MatM)
        return;

    if (outsPerThread <= 1)
    {
        int col = (int)id.x;
        if (col >= 0 && col < _MatN && col < (int)ow)
        {
            float sum = NcnnGemm2DTextureAValue(row, col);
            _GemmTexOutArr[int3(col, row, 0)] = float4(sum, 0.0, 0.0, 0.0);
        }
        return;
    }

    int maxCol = min(_MatN, (int)ow);
    bool valid0 = baseCol < maxCol;
    bool valid1 = outsPerThread > 1 && baseCol + 1 < maxCol;
    bool valid2 = outsPerThread > 2 && baseCol + 2 < maxCol;
    bool valid3 = outsPerThread > 3 && baseCol + 3 < maxCol;
    float acc0 = 0.0;
    float acc1 = 0.0;
    float acc2 = 0.0;
    float acc3 = 0.0;

    [loop]
    for (int kk = 0; kk < _MatK; kk++)
    {
        float a = 0.0;
        if (kk >= 0 && kk < _GemmTexAInW && row >= 0 && row < _GemmTexAInH)
            a = _GemmTexAInArr[int3(kk, row, 0)].x;

        a = NcnnQuantizeGemmActivationForInt8(a);
        if (valid0) acc0 += a * NcnnGemm2DReadB(baseCol, kk);
        if (valid1) acc1 += a * NcnnGemm2DReadB(baseCol + 1, kk);
        if (valid2) acc2 += a * NcnnGemm2DReadB(baseCol + 2, kk);
        if (valid3) acc3 += a * NcnnGemm2DReadB(baseCol + 3, kk);
    }

    if (valid0) _GemmTexOutArr[int3(baseCol, row, 0)] = float4(NcnnResolveGemm2DBias(row, baseCol) + acc0 * _MatAlpha, 0.0, 0.0, 0.0);
    if (valid1) _GemmTexOutArr[int3(baseCol + 1, row, 0)] = float4(NcnnResolveGemm2DBias(row, baseCol + 1) + acc1 * _MatAlpha, 0.0, 0.0, 0.0);
    if (valid2) _GemmTexOutArr[int3(baseCol + 2, row, 0)] = float4(NcnnResolveGemm2DBias(row, baseCol + 2) + acc2 * _MatAlpha, 0.0, 0.0, 0.0);
    if (valid3) _GemmTexOutArr[int3(baseCol + 3, row, 0)] = float4(NcnnResolveGemm2DBias(row, baseCol + 3) + acc3 * _MatAlpha, 0.0, 0.0, 0.0);
}

void NcnnGemm2DLinearTextureA_Impl(uint3 id)
{
    uint ow, oh;
    _LinearOut0.GetDimensions(ow, oh);
    int outsPerThread = clamp(_GemmTexOutsPerThread, 1, 4);
    int baseCol = (int)id.x * outsPerThread;
    if (baseCol >= (int)ow || id.y >= oh)
        return;

    int row = (int)id.y;
    if (row < 0 || row >= _MatM)
        return;

    if (outsPerThread <= 1)
    {
        int col = (int)id.x;
        if (col >= 0 && col < _MatN && col < (int)ow)
            _LinearOut0[int2(col, row)] = NcnnGemm2DLinearTextureAValue(row, col);
        return;
    }

    int maxCol = min(_MatN, (int)ow);
    bool valid0 = baseCol < maxCol;
    bool valid1 = outsPerThread > 1 && baseCol + 1 < maxCol;
    bool valid2 = outsPerThread > 2 && baseCol + 2 < maxCol;
    bool valid3 = outsPerThread > 3 && baseCol + 3 < maxCol;
    float acc0 = 0.0;
    float acc1 = 0.0;
    float acc2 = 0.0;
    float acc3 = 0.0;

    [loop]
    for (int kk = 0; kk < _MatK; kk++)
    {
        float a = 0.0;
        if (kk >= 0 && kk < _GemmTexAInW && row >= 0 && row < _GemmTexAInH)
            a = _LinearIn0[int2(kk, row)];

        a = NcnnQuantizeGemmActivationForInt8(a);
        if (valid0) acc0 += a * NcnnGemm2DReadB(baseCol, kk);
        if (valid1) acc1 += a * NcnnGemm2DReadB(baseCol + 1, kk);
        if (valid2) acc2 += a * NcnnGemm2DReadB(baseCol + 2, kk);
        if (valid3) acc3 += a * NcnnGemm2DReadB(baseCol + 3, kk);
    }

    if (valid0) _LinearOut0[int2(baseCol, row)] = NcnnResolveGemm2DBias(row, baseCol) + acc0 * _MatAlpha;
    if (valid1) _LinearOut0[int2(baseCol + 1, row)] = NcnnResolveGemm2DBias(row, baseCol + 1) + acc1 * _MatAlpha;
    if (valid2) _LinearOut0[int2(baseCol + 2, row)] = NcnnResolveGemm2DBias(row, baseCol + 2) + acc2 * _MatAlpha;
    if (valid3) _LinearOut0[int2(baseCol + 3, row)] = NcnnResolveGemm2DBias(row, baseCol + 3) + acc3 * _MatAlpha;
}

void NcnnGemm2DPack4LinearTextureA_Impl(uint3 id)
{
    uint ow, oh, od;
    _GemmTexOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int row = (int)id.y;
    int baseCol = (int)id.x * 4;
    if (row < 0 || row >= _MatM || baseCol >= _MatN)
        return;

    bool valid0 = baseCol < _MatN;
    bool valid1 = baseCol + 1 < _MatN;
    bool valid2 = baseCol + 2 < _MatN;
    bool valid3 = baseCol + 3 < _MatN;
    float acc0 = 0.0;
    float acc1 = 0.0;
    float acc2 = 0.0;
    float acc3 = 0.0;

    [loop]
    for (int kk = 0; kk < _MatK; kk++)
    {
        float a = NcnnGemm2DPack4LinearTextureARead(row, kk);
        if (valid0) acc0 += a * NcnnGemm2DReadB(baseCol, kk);
        if (valid1) acc1 += a * NcnnGemm2DReadB(baseCol + 1, kk);
        if (valid2) acc2 += a * NcnnGemm2DReadB(baseCol + 2, kk);
        if (valid3) acc3 += a * NcnnGemm2DReadB(baseCol + 3, kk);
    }

    float4 o = 0.0;
    if (valid0) o.x = NcnnResolveGemm2DBias(row, baseCol) + acc0 * _MatAlpha;
    if (valid1) o.y = NcnnResolveGemm2DBias(row, baseCol + 1) + acc1 * _MatAlpha;
    if (valid2) o.z = NcnnResolveGemm2DBias(row, baseCol + 2) + acc2 * _MatAlpha;
    if (valid3) o.w = NcnnResolveGemm2DBias(row, baseCol + 3) + acc3 * _MatAlpha;
    _GemmTexOutArr[int3((int)id.x, row, 0)] = o;
}

void NcnnGemm2DPack4LinearTextureAFromLinear_Impl(uint3 id)
{
    uint ow, oh, od;
    _GemmTexOutArr.GetDimensions(ow, oh, od);
    int packsPerThread = NcnnGemmPack4LinearOutPacksPerThread();
    int outPackX = (int)id.x * packsPerThread;
    if (outPackX >= (int)ow || id.y >= oh || id.z >= od)
        return;

    int row = (int)id.y;
    int baseCol0 = outPackX * 4;
    int baseCol1 = baseCol0 + 4;
    if (row < 0 || row >= _MatM || baseCol0 >= _MatN)
        return;

    bool writeSecondPack = packsPerThread > 1 && outPackX + 1 < (int)ow && baseCol1 < _MatN;
    // ncnn initializes Gemm accumulation with beta * C. Keeping the bias in
    // the running FP32 sum is numerically observable for large cancellation.
    float4 acc0 = NcnnResolveGemm2DBias4(row, baseCol0);
    float4 acc1 = writeSecondPack ? NcnnResolveGemm2DBias4(row, baseCol1) : 0.0;

    bool usePackedInt8 = NcnnCanReadPackedInt8TransB4()
        && baseCol0 + 3 < _MatN
        && (!writeSecondPack || baseCol1 + 3 < _MatN);
    if (usePackedInt8)
    {
        [loop]
        for (int kk = 0; kk < _MatK; kk += 4)
        {
            float4 a = float4(
                NcnnGemm2DPack4LinearTextureAReadLinear(row, kk),
                NcnnGemm2DPack4LinearTextureAReadLinear(row, kk + 1),
                NcnnGemm2DPack4LinearTextureAReadLinear(row, kk + 2),
                NcnnGemm2DPack4LinearTextureAReadLinear(row, kk + 3));
            uint4 packed0 = NcnnLoadPackedInt8TransB4(baseCol0, kk);
            float4 scales0 = NcnnReadPackedInt8Scales4(baseCol0);
            acc0 += a.x * NcnnDecodePackedInt8TransB4(packed0, 0, scales0);
            acc0 += a.y * NcnnDecodePackedInt8TransB4(packed0, 1, scales0);
            acc0 += a.z * NcnnDecodePackedInt8TransB4(packed0, 2, scales0);
            acc0 += a.w * NcnnDecodePackedInt8TransB4(packed0, 3, scales0);
            if (writeSecondPack)
            {
                uint4 packed1 = NcnnLoadPackedInt8TransB4(baseCol1, kk);
                float4 scales1 = NcnnReadPackedInt8Scales4(baseCol1);
                acc1 += a.x * NcnnDecodePackedInt8TransB4(packed1, 0, scales1);
                acc1 += a.y * NcnnDecodePackedInt8TransB4(packed1, 1, scales1);
                acc1 += a.z * NcnnDecodePackedInt8TransB4(packed1, 2, scales1);
                acc1 += a.w * NcnnDecodePackedInt8TransB4(packed1, 3, scales1);
            }
        }
    }
    else
    {
        [loop]
        for (int kk = 0; kk < _MatK; kk++)
        {
            float a = NcnnGemm2DPack4LinearTextureAReadLinear(row, kk);
            acc0 += a * NcnnGemm2DReadB4(baseCol0, kk);
            if (writeSecondPack)
                acc1 += a * NcnnGemm2DReadB4(baseCol1, kk);
        }
    }

    _GemmTexOutArr[int3(outPackX, row, 0)] = acc0 * _MatAlpha;
    if (writeSecondPack)
        _GemmTexOutArr[int3(outPackX + 1, row, 0)] = acc1 * _MatAlpha;
}

void NcnnGemm2DPack4LinearTextureAFromPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _GemmTexOutArr.GetDimensions(ow, oh, od);
    int packsPerThread = NcnnGemmPack4LinearOutPacksPerThread();
    int outPackX = (int)id.x * packsPerThread;
    if (outPackX >= (int)ow || id.y >= oh || id.z >= od)
        return;

    int row = (int)id.y;
    int baseCol0 = outPackX * 4;
    int baseCol1 = baseCol0 + 4;
    if (row < 0 || row >= _MatM || row >= _GemmTexAInH || baseCol0 >= _MatN)
        return;

    bool writeSecondPack = packsPerThread > 1 && outPackX + 1 < (int)ow && baseCol1 < _MatN;
    float4 acc0 = NcnnResolveGemm2DBias4(row, baseCol0);
    float4 acc1 = writeSecondPack ? NcnnResolveGemm2DBias4(row, baseCol1) : 0.0;
    bool usePackedInt8 = NcnnCanReadPackedInt8TransB4()
        && baseCol0 + 3 < _MatN
        && (!writeSecondPack || baseCol1 + 3 < _MatN);

    uint inputWidth, inputHeight, inputSlices;
    _GemmTexAInArr.GetDimensions(inputWidth, inputHeight, inputSlices);
    [loop]
    for (int logicalPack = 0; logicalPack * 4 < _MatK; logicalPack++)
    {
        int packX = logicalPack % max(1, (int)inputWidth);
        int packSlice = logicalPack / max(1, (int)inputWidth);
        if (packSlice >= (int)inputSlices)
            break;
        int kkBase = logicalPack << 2;
        float4 a = NcnnQuantizeGemmActivationForInt8(_GemmTexAInArr[int3(packX, row, packSlice)]);
        if (usePackedInt8 && kkBase + 3 < _MatK)
        {
            uint4 packed0 = NcnnLoadPackedInt8TransB4(baseCol0, kkBase);
            float4 scales0 = NcnnReadPackedInt8Scales4(baseCol0);
            acc0 += a.x * NcnnDecodePackedInt8TransB4(packed0, 0, scales0);
            acc0 += a.y * NcnnDecodePackedInt8TransB4(packed0, 1, scales0);
            acc0 += a.z * NcnnDecodePackedInt8TransB4(packed0, 2, scales0);
            acc0 += a.w * NcnnDecodePackedInt8TransB4(packed0, 3, scales0);
            if (writeSecondPack)
            {
                uint4 packed1 = NcnnLoadPackedInt8TransB4(baseCol1, kkBase);
                float4 scales1 = NcnnReadPackedInt8Scales4(baseCol1);
                acc1 += a.x * NcnnDecodePackedInt8TransB4(packed1, 0, scales1);
                acc1 += a.y * NcnnDecodePackedInt8TransB4(packed1, 1, scales1);
                acc1 += a.z * NcnnDecodePackedInt8TransB4(packed1, 2, scales1);
                acc1 += a.w * NcnnDecodePackedInt8TransB4(packed1, 3, scales1);
            }
            continue;
        }
        if (kkBase + 3 < _MatK)
        {
            acc0 += a.x * NcnnGemm2DReadB4(baseCol0, kkBase);
            acc0 += a.y * NcnnGemm2DReadB4(baseCol0, kkBase + 1);
            acc0 += a.z * NcnnGemm2DReadB4(baseCol0, kkBase + 2);
            acc0 += a.w * NcnnGemm2DReadB4(baseCol0, kkBase + 3);
            if (writeSecondPack)
            {
                acc1 += a.x * NcnnGemm2DReadB4(baseCol1, kkBase);
                acc1 += a.y * NcnnGemm2DReadB4(baseCol1, kkBase + 1);
                acc1 += a.z * NcnnGemm2DReadB4(baseCol1, kkBase + 2);
                acc1 += a.w * NcnnGemm2DReadB4(baseCol1, kkBase + 3);
            }
            continue;
        }

        [unroll]
        for (int lane = 0; lane < 4; lane++)
        {
            int kk = kkBase + lane;
            if (kk >= _MatK)
                break;
            float av = NcnnReadLane(a, lane);
            acc0 += av * NcnnGemm2DReadB4(baseCol0, kk);
            if (writeSecondPack)
                acc1 += av * NcnnGemm2DReadB4(baseCol1, kk);
        }
    }

    _GemmTexOutArr[int3(outPackX, row, 0)] = acc0 * _MatAlpha;
    if (writeSecondPack)
        _GemmTexOutArr[int3(outPackX + 1, row, 0)] = acc1 * _MatAlpha;
}

void NcnnGemm2DAttentionQkvLinearTextureA_Impl(uint3 id)
{
    uint ow, oh, od;
    _GemmTexOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int outX = (int)id.x;
    int row = (int)id.y;
    int headPack = (int)id.z;
    int headDim = max(1, _AttentionGemmHeadDim);
    int numHeads = max(1, _AttentionGemmNumHeads);
    if (outX < 0 || outX >= headDim || row < 0 || row >= _MatM)
        return;

    float4 o = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int head = headPack * 4 + lane;
        if (head < 0 || head >= numHeads)
            continue;

        int col = head * headDim + outX;
        if (col < 0 || col >= _MatN)
            continue;

        float acc = NcnnResolveGemm2DBias(row, col);
        [loop]
        for (int kk = 0; kk < _MatK; kk++)
        {
            float a = 0.0;
            if (kk >= 0 && kk < _GemmTexAInW && row >= 0 && row < _GemmTexAInH)
                a = _LinearIn0[int2(kk, row)];
            a = NcnnQuantizeGemmActivationForInt8(a);
            acc += a * NcnnGemm2DReadB(col, kk);
        }

        NcnnWriteLane(o, lane, acc * _MatAlpha);
    }

    _GemmTexOutArr[int3(outX, row, headPack)] = o;
}

void NcnnGemm2DAttentionQkvPack4LinearTextureA_Impl(uint3 id)
{
    uint ow, oh, od;
    _GemmTexOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int outX = (int)id.x;
    int row = (int)id.y;
    int headPack = (int)id.z;
    int headDim = max(1, _AttentionGemmHeadDim);
    int numHeads = max(1, _AttentionGemmNumHeads);
    if (outX < 0 || outX >= headDim || row < 0 || row >= _MatM)
        return;

    float4 o = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int head = headPack * 4 + lane;
        if (head < 0 || head >= numHeads)
            continue;

        int col = head * headDim + outX;
        if (col < 0 || col >= _MatN)
            continue;

        float acc = NcnnResolveGemm2DBias(row, col);
        [loop]
        for (int kk = 0; kk < _MatK; kk++)
        {
            int packX = kk >> 2;
            float a = NcnnQuantizeGemmActivationForInt8(_GemmTexAInArr[int3(packX, row, 0)][kk & 3]);
            acc += a * NcnnGemm2DReadB(col, kk);
        }

        NcnnWriteLane(o, lane, acc * _MatAlpha);
    }

    _GemmTexOutArr[int3(outX, row, headPack)] = o;
}

void NcnnGemm2DAttentionQkvTextureA_Impl(uint3 id)
{
    uint ow, oh, od;
    _GemmTexOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int outX = (int)id.x;
    int row = (int)id.y;
    int headPack = (int)id.z;
    int headDim = max(1, _AttentionGemmHeadDim);
    int numHeads = max(1, _AttentionGemmNumHeads);
    if (outX < 0 || outX >= headDim || row < 0 || row >= _MatM)
        return;

    float4 o = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int head = headPack * 4 + lane;
        if (head < 0 || head >= numHeads)
            continue;

        int col = head * headDim + outX;
        if (col < 0 || col >= _MatN)
            continue;

        float acc = NcnnResolveGemm2DBias(row, col);
        [loop]
        for (int kk = 0; kk < _MatK; kk++)
        {
            float a = 0.0;
            if (kk >= 0 && kk < _GemmTexAInW && row >= 0 && row < _GemmTexAInH)
                a = _GemmTexAInArr[int3(kk, row, 0)].x;
            a = NcnnQuantizeGemmActivationForInt8(a);
            acc += a * NcnnGemm2DReadB(col, kk);
        }

        NcnnWriteLane(o, lane, acc * _MatAlpha);
    }

    _GemmTexOutArr[int3(outX, row, headPack)] = o;
}

void NcnnGemm2DAttentionPack4ToLinearTextureA_Impl(uint3 id)
{
    uint ow, oh;
    _LinearOut0.GetDimensions(ow, oh);
    if (id.x >= ow || id.y >= oh)
        return;

    int col = (int)id.x;
    int row = (int)id.y;
    int headDim = max(1, _AttentionGemmHeadDim);
    if (col < 0 || col >= _MatN || row < 0 || row >= _MatM)
        return;

    float acc = NcnnResolveGemm2DBias(row, col);
    [loop]
    for (int kk = 0; kk < _MatK; kk++)
    {
        int head = kk / headDim;
        int dim = kk - head * headDim;
        float a = NcnnQuantizeGemmActivationForInt8(NcnnReadPack4Channel(_GemmTexAInArr, dim, row, head));
        acc += a * NcnnGemm2DReadB(col, kk);
    }

    _LinearOut0[int2(col, row)] = acc * _MatAlpha;
}

void NcnnGemm2DAttentionPack4ToPack4LinearTextureA_Impl(uint3 id)
{
    uint ow, oh, od;
    _GemmTexOutArr.GetDimensions(ow, oh, od);
    int packsPerThread = NcnnGemmPack4LinearOutPacksPerThread();
    int outPackX = (int)id.x * packsPerThread;
    if (outPackX >= (int)ow || id.y >= oh || id.z >= od)
        return;

    int row = (int)id.y;
    int baseCol0 = outPackX * 4;
    int baseCol1 = baseCol0 + 4;
    int headDim = max(1, _AttentionGemmHeadDim);
    int headCount = max(1, _AttentionGemmNumHeads);
    if (row < 0 || row >= _MatM || baseCol0 >= _MatN)
        return;

    bool writeSecondPack = packsPerThread > 1 && outPackX + 1 < (int)ow && baseCol1 < _MatN;
    float4 acc0 = NcnnResolveGemm2DBias4(row, baseCol0);
    float4 acc1 = writeSecondPack ? NcnnResolveGemm2DBias4(row, baseCol1) : 0.0;

    [loop]
    for (int head = 0; head < headCount; head++)
    {
        [loop]
        for (int dim = 0; dim < headDim; dim++)
        {
            int kk = head * headDim + dim;
            if (kk >= _MatK)
                break;

            float a = NcnnQuantizeGemmActivationForInt8(NcnnReadPack4Channel(_GemmTexAInArr, dim, row, head));
            acc0 += a * NcnnGemm2DReadB4(baseCol0, kk);
            if (writeSecondPack)
                acc1 += a * NcnnGemm2DReadB4(baseCol1, kk);
        }
    }

    _GemmTexOutArr[int3(outPackX, row, 0)] = acc0 * _MatAlpha;
    if (writeSecondPack)
        _GemmTexOutArr[int3(outPackX + 1, row, 0)] = acc1 * _MatAlpha;
}

void NcnnGemm2D_Impl(uint3 groupId, uint3 groupThreadId)
{
    int lx = (int)groupThreadId.x;
    int ly = (int)groupThreadId.y;
    int row = (int)groupId.y * TILE + ly;
    int col = (int)groupId.x * TILE + lx;
    bool valid = (row < _MatM && col < _MatN);

    float sum = 0.0;
    if (valid && _MatUseC != 0)
    {
        if (_MatBroadcastTypeC == 0)
            sum = _MatC[0];
        else if (_MatBroadcastTypeC == 1)
            sum = _MatC[row];
        else if (_MatBroadcastTypeC == 3)
            sum = _MatC[row * _MatN + col];
        else if (_MatBroadcastTypeC == 4)
            sum = _MatC[col];
        sum *= _MatBeta;
    }

    float acc = 0.0;
    for (int k0 = 0; k0 < _MatK; k0 += TILE)
    {
        int ak = k0 + lx;
        int bk = k0 + ly;
        float a = 0.0;
        float b = 0.0;

        if (row < _MatM && ak < _MatK)
            a = _MatA[row * _MatK + ak];
        if (bk < _MatK && col < _MatN)
        {
            if (_MatTransB != 0)
                b = _MatB[col * _MatK + bk];
            else
                b = _MatB[bk * _MatN + col];
        }

        _MatAs[ly * TILE + lx] = a;
        _MatBs[ly * TILE + lx] = b;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (int k1 = 0; k1 < TILE; k1++)
        {
            int kk = k0 + k1;
            if (kk < _MatK)
                acc += _MatAs[ly * TILE + k1] * _MatBs[k1 * TILE + lx];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (valid)
    {
        sum += acc * _MatAlpha;
        _MatOut[row * _MatN + col] = sum;
    }
}

void NcnnGemm2D16_Impl(uint3 groupId, uint3 groupThreadId)
{
    int lx = (int)groupThreadId.x;
    int ly = (int)groupThreadId.y;
    int row = (int)groupId.y * TILE16 + ly;
    int col = (int)groupId.x * TILE16 + lx;
    bool valid = (row < _MatM && col < _MatN);

    float sum = 0.0;
    if (valid && _MatUseC != 0)
    {
        if (_MatBroadcastTypeC == 0)
            sum = _MatC[0];
        else if (_MatBroadcastTypeC == 1)
            sum = _MatC[row];
        else if (_MatBroadcastTypeC == 3)
            sum = _MatC[row * _MatN + col];
        else if (_MatBroadcastTypeC == 4)
            sum = _MatC[col];
        sum *= _MatBeta;
    }

    float acc = 0.0;
    for (int k0 = 0; k0 < _MatK; k0 += TILE16)
    {
        int ak = k0 + lx;
        int bk = k0 + ly;
        float a = 0.0;
        float b = 0.0;

        if (row < _MatM && ak < _MatK)
            a = _MatA[row * _MatK + ak];
        if (bk < _MatK && col < _MatN)
        {
            if (_MatTransB != 0)
                b = _MatB[col * _MatK + bk];
            else
                b = _MatB[bk * _MatN + col];
        }

        _MatAs16[ly * TILE16 + lx] = a;
        _MatBs16[ly * TILE16 + lx] = b;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (int k1 = 0; k1 < TILE16; k1++)
        {
            int kk = k0 + k1;
            if (kk < _MatK)
                acc += _MatAs16[ly * TILE16 + k1] * _MatBs16[k1 * TILE16 + lx];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (valid)
    {
        sum += acc * _MatAlpha;
        _MatOut[row * _MatN + col] = sum;
    }
}

void NcnnLayerNormPack4WidthTex_Impl(uint3 id)
{
    uint ow, oh, od;
    _LnTexOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int x = (int)id.x;
    int y = (int)id.y;
    int slice = (int)id.z;
    int packs = max(1, (_LnC + 3) / 4);
    int z = slice / packs;
    int pack = slice - z * packs;
    if (z < 0 || z >= max(1, _LnD))
    {
        _LnTexOutArr[int3(x, y, slice)] = 0.0;
        return;
    }

    if (_LnC == 1 && packs == 1)
    {
        float4 sumLo = 0.0;
        float4 sumHi = 0.0;
        int groupCount = _LnW / 8;
        for (int group = 0; group < groupCount; group++)
        {
            int baseX = group * 8;
            sumLo += float4(
                _LnTexInArr[int3(baseX, y, 0)].x,
                _LnTexInArr[int3(baseX + 1, y, 0)].x,
                _LnTexInArr[int3(baseX + 2, y, 0)].x,
                _LnTexInArr[int3(baseX + 3, y, 0)].x);
            sumHi += float4(
                _LnTexInArr[int3(baseX + 4, y, 0)].x,
                _LnTexInArr[int3(baseX + 5, y, 0)].x,
                _LnTexInArr[int3(baseX + 6, y, 0)].x,
                _LnTexInArr[int3(baseX + 7, y, 0)].x);
        }
        float tailSum = 0.0;
        for (int tailX = groupCount * 8; tailX < _LnW; tailX++)
            tailSum += _LnTexInArr[int3(tailX, y, 0)].x;
        float4 mergedSum = sumLo + sumHi;
        float2 pairedSum = mergedSum.xy + mergedSum.zw;
        float sum = (pairedSum.x + pairedSum.y) + tailSum;
        float mean = sum / max(1, _LnW);

        float4 varianceLo = 0.0;
        float4 varianceHi = 0.0;
        for (int group2 = 0; group2 < groupCount; group2++)
        {
            int baseX2 = group2 * 8;
            float4 centeredLo = float4(
                _LnTexInArr[int3(baseX2, y, 0)].x,
                _LnTexInArr[int3(baseX2 + 1, y, 0)].x,
                _LnTexInArr[int3(baseX2 + 2, y, 0)].x,
                _LnTexInArr[int3(baseX2 + 3, y, 0)].x) - mean;
            float4 centeredHi = float4(
                _LnTexInArr[int3(baseX2 + 4, y, 0)].x,
                _LnTexInArr[int3(baseX2 + 5, y, 0)].x,
                _LnTexInArr[int3(baseX2 + 6, y, 0)].x,
                _LnTexInArr[int3(baseX2 + 7, y, 0)].x) - mean;
            varianceLo += centeredLo * centeredLo;
            varianceHi += centeredHi * centeredHi;
        }
        float tailVariance = 0.0;
        for (int tailX2 = groupCount * 8; tailX2 < _LnW; tailX2++)
        {
            float centeredTail = _LnTexInArr[int3(tailX2, y, 0)].x - mean;
            tailVariance += centeredTail * centeredTail;
        }
        float4 mergedVariance = varianceLo + varianceHi;
        float2 pairedVariance = mergedVariance.xy + mergedVariance.zw;
        float variance = ((pairedVariance.x + pairedVariance.y) + tailVariance) / max(1, _LnW);
        float invstd = rsqrt(max(variance + _LnEps, 1e-20));
        float scaledMean = mean * invstd;
        float normalized = _LnTexInArr[int3(x, y, 0)].x * invstd - scaledMean;
        if (_LnAffine != 0)
            normalized = normalized * _LnGamma[x] + _LnBeta[x];
        _LnTexOutArr[int3(x, y, slice)] = float4(normalized, 0.0, 0.0, 0.0);
        return;
    }

    float4 sum = 0.0;
    for (int ix = 0; ix < _LnW; ix++)
    {
        float4 v = _LnTexInArr[int3(ix, y, slice)];
        sum += v;
    }

    float widthCount = max(1, _LnW);
    float4 mean = sum / widthCount;
    float4 sqsum = 0.0;
    for (int ix2 = 0; ix2 < _LnW; ix2++)
    {
        float4 centered = _LnTexInArr[int3(ix2, y, slice)] - mean;
        sqsum += centered * centered;
    }
    float4 var = sqsum / widthCount;
    float4 invstd = rsqrt(max(var + _LnEps, 1e-20));

    float4 o = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int channel = pack * 4 + lane;
        if (channel >= _LnC)
            continue;
        float scalar = NcnnReadLane(_LnTexInArr[int3(x, y, slice)], lane);
        float normalized = (scalar - NcnnReadLane(mean, lane)) * NcnnReadLane(invstd, lane);
        if (_LnAffine != 0)
            normalized = normalized * _LnGamma[x] + _LnBeta[x];
        NcnnWriteLane(o, lane, normalized);
    }

    _LnTexOutArr[int3(x, y, slice)] = o;
}

void NcnnLayerNormPack4Linear2D_Impl(uint3 id)
{
    uint ow, oh, od;
    _LnTexOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int packX = (int)id.x;
    int row = (int)id.y;
    int logicalW = max(1, _LnW);
    int storageW = max(1, (logicalW + 3) / 4);
    if (row < 0 || row >= _LnH || packX < 0 || packX >= storageW)
        return;

    float4 sumLo = 0.0;
    float4 sumHi = 0.0;
    int pairCount = storageW / 2;
    for (int pair = 0; pair < pairCount; pair++)
    {
        sumLo += _LnTexInArr[int3(pair * 2, row, 0)];
        sumHi += _LnTexInArr[int3(pair * 2 + 1, row, 0)];
    }
    float tailSum = 0.0;
    for (int tailPack = pairCount * 2; tailPack < storageW; tailPack++)
    {
        float4 tail = _LnTexInArr[int3(tailPack, row, 0)];
        [unroll]
        for (int tailLane = 0; tailLane < 4; tailLane++)
        {
            int tailCol = tailPack * 4 + tailLane;
            if (tailCol < logicalW)
                tailSum += NcnnReadLane(tail, tailLane);
        }
    }

    float invCount = 1.0 / (float)logicalW;
    float4 mergedSum = sumLo + sumHi;
    float2 pairedSum = mergedSum.xy + mergedSum.zw;
    float sum = (pairedSum.x + pairedSum.y) + tailSum;
    float mean = sum * invCount;
    float4 varianceLo = 0.0;
    float4 varianceHi = 0.0;
    for (int pair2 = 0; pair2 < pairCount; pair2++)
    {
        float4 centeredLo = _LnTexInArr[int3(pair2 * 2, row, 0)] - mean;
        float4 centeredHi = _LnTexInArr[int3(pair2 * 2 + 1, row, 0)] - mean;
        varianceLo += centeredLo * centeredLo;
        varianceHi += centeredHi * centeredHi;
    }
    float tailVariance = 0.0;
    for (int tailPack2 = pairCount * 2; tailPack2 < storageW; tailPack2++)
    {
        float4 tail2 = _LnTexInArr[int3(tailPack2, row, 0)];
        [unroll]
        for (int tailLane2 = 0; tailLane2 < 4; tailLane2++)
        {
            int tailCol2 = tailPack2 * 4 + tailLane2;
            if (tailCol2 < logicalW)
            {
                float centeredTail = NcnnReadLane(tail2, tailLane2) - mean;
                tailVariance += centeredTail * centeredTail;
            }
        }
    }
    float4 mergedVariance = varianceLo + varianceHi;
    float2 pairedVariance = mergedVariance.xy + mergedVariance.zw;
    float sqsum = (pairedVariance.x + pairedVariance.y) + tailVariance;
    float variance = sqsum * invCount;
    // Match ncnn's FP32 elempack=1 LayerNorm path (1 / sqrtf), rather than
    // the approximate reciprocal-square-root used by packed image norms.
    float invstd = 1.0 / sqrt(max(variance + _LnEps, 1e-20));
    float scaledMean = mean * invstd;

    float4 src = _LnTexInArr[int3(packX, row, 0)];
    float4 o = 0.0;
    [unroll]
    for (int outLane = 0; outLane < 4; outLane++)
    {
        int col = packX * 4 + outLane;
        if (col >= logicalW)
            continue;
        float normalized = NcnnReadLane(src, outLane) * invstd - scaledMean;
        if (_LnAffine != 0)
            normalized = normalized * _LnGamma[col] + _LnBeta[col];
        NcnnWriteLane(o, outLane, normalized);
    }

    _LnTexOutArr[int3(packX, row, 0)] = o;
}

void NcnnSoftmaxPack4CDHW_Impl(uint3 id)
{
    uint w, h, d;
    _SoftmaxPack4CDHWOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h || id.z >= d)
        return;

    int outX = (int)id.x;
    int outY = (int)id.y;
    int slice = (int)id.z;
    int packCount = max(1, (_SoftmaxPack4CDHWC + 3) / 4);
    int outZ = slice / packCount;
    int outPack = slice - outZ * packCount;
    if (outZ < 0 || outZ >= _SoftmaxPack4CDHWD)
    {
        _SoftmaxPack4CDHWOutArr[int3(outX, outY, slice)] = 0.0;
        return;
    }

    float4 o = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int c = outPack * 4 + lane;
        if (c >= _SoftmaxPack4CDHWC)
            continue;

        int axis = _SoftmaxAxis;
        int axisLength = axis == 0 ? _SoftmaxPack4CDHWW
            : axis == 1 ? _SoftmaxPack4CDHWH
            : axis == 2 ? _SoftmaxPack4CDHWD
            : _SoftmaxPack4CDHWC;
        if (axis < 0 || axis > 3 || axisLength <= 0)
            continue;

        float maxv = -3.402823466e+38;
        for (int i = 0; i < axisLength; i++)
        {
            int x = axis == 0 ? i : outX;
            int y = axis == 1 ? i : outY;
            int z = axis == 2 ? i : outZ;
            int channel = axis == 3 ? i : c;
            float v = NcnnReadPack4ChannelCDHW(_SoftmaxPack4CDHWInArr, x, y, z, channel, _SoftmaxPack4CDHWC);
            maxv = max(maxv, v);
        }

        float sum = 0.0;
        for (int i = 0; i < axisLength; i++)
        {
            int x = axis == 0 ? i : outX;
            int y = axis == 1 ? i : outY;
            int z = axis == 2 ? i : outZ;
            int channel = axis == 3 ? i : c;
            float v = NcnnReadPack4ChannelCDHW(_SoftmaxPack4CDHWInArr, x, y, z, channel, _SoftmaxPack4CDHWC);
            sum += exp(v - maxv);
        }

        float current = NcnnReadPack4ChannelCDHW(_SoftmaxPack4CDHWInArr, outX, outY, outZ, c, _SoftmaxPack4CDHWC);
        float value = sum > 0.0 ? exp(current - maxv) / sum : 0.0;
        NcnnWriteLane(o, lane, value);
    }

    _SoftmaxPack4CDHWOutArr[int3(outX, outY, slice)] = o;
}

void NcnnSoftmaxLinearMat2D_Impl(uint3 id)
{
    uint ow, oh;
    _LinearOut0.GetDimensions(ow, oh);
    if (id.x >= ow || id.y >= oh)
        return;

    int x = (int)id.x;
    int row = (int)id.y;
    int inW = max(1, _SoftmaxPack4CDHWW);
    int inH = max(1, _SoftmaxPack4CDHWH);
    if (x < 0 || x >= inW || row < 0 || row >= inH)
    {
        _LinearOut0[int2(x, row)] = 0.0;
        return;
    }

    int axis = _SoftmaxAxis;
    if (axis < 0 || axis > 1)
    {
        _LinearOut0[int2(x, row)] = 0.0;
        return;
    }
    int axisLength = axis == 0 ? inW : inH;
    float maxv = axis == 0 ? _LinearIn0[int2(0, row)] : _LinearIn0[int2(x, 0)];
    for (int i = 1; i < axisLength; i++)
    {
        float value = axis == 0 ? _LinearIn0[int2(i, row)] : _LinearIn0[int2(x, i)];
        maxv = max(maxv, value);
    }

    float sum = 0.0;
    for (int i = 0; i < axisLength; i++)
    {
        float value = axis == 0 ? _LinearIn0[int2(i, row)] : _LinearIn0[int2(x, i)];
        sum += exp(value - maxv);
    }

    float current = _LinearIn0[int2(x, row)];
    _LinearOut0[int2(x, row)] = sum > 0.0 ? exp(current - maxv) / sum : 0.0;
}

void NcnnReductionScalar2D_Impl(uint3 id)
{
    uint ow, oh, od;
    _ReduceScalar2DOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int outX = (int)id.x;
    int outY = (int)id.y;
    int axis = _ReduceScalar2DAxis;
    int inW = max(1, _ReduceScalar2DInW);
    int inH = max(1, _ReduceScalar2DInH);
    float coeff = _ReduceScalar2DCoeff;
    int opType = _ReduceScalar2DOpType;

    float result = 0.0;

    if (axis == 2)
    {
        if (outX != 0 || outY != 0)
        {
            _ReduceScalar2DOutArr[int3(outX, outY, (int)id.z)] = 0.0;
            return;
        }

        int total = max(1, inW * inH);
        float acc = _ReduceScalar2DInArr[int3(0, 0, 0)].x;

        if (opType == 4)
        {
            for (int i = 1; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                acc = max(acc, _ReduceScalar2DInArr[int3(x, y, 0)].x);
            }
            result = acc * coeff;
        }
        else if (opType == 5)
        {
            for (int i = 1; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                acc = min(acc, _ReduceScalar2DInArr[int3(x, y, 0)].x);
            }
            result = acc * coeff;
        }
        else if (opType == 0 || opType == 3)
        {
            float sum = 0.0;
            for (int i = 0; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                sum += _ReduceScalar2DInArr[int3(x, y, 0)].x;
            }
            if (opType == 3)
                sum /= total;
            result = sum * coeff;
        }
        else if (opType == 2)
        {
            float sumSq = 0.0;
            for (int i = 0; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                float v = _ReduceScalar2DInArr[int3(x, y, 0)].x;
                sumSq += v * v;
            }
            result = sumSq * coeff;
        }
        else if (opType == 1 || opType == 7)
        {
            float sumAbs = 0.0;
            for (int i = 0; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                sumAbs += abs(_ReduceScalar2DInArr[int3(x, y, 0)].x);
            }
            result = sumAbs * coeff;
        }
        else if (opType == 6 || opType == 10)
        {
            float sumLog = 0.0;
            for (int i = 0; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                sumLog += log(max(_ReduceScalar2DInArr[int3(x, y, 0)].x, 1e-12));
            }
            result = exp(sumLog) * coeff;
        }
        else if (opType == 9)
        {
            float sumVal = 0.0;
            for (int i = 0; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                sumVal += _ReduceScalar2DInArr[int3(x, y, 0)].x;
            }
            result = log(max(sumVal, 1e-12)) * coeff;
        }
        else if (opType == 8)
        {
            float sumSq = 0.0;
            for (int i = 0; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                float v = _ReduceScalar2DInArr[int3(x, y, 0)].x;
                sumSq += v * v;
            }
            result = sqrt(max(sumSq, 0.0)) * coeff;
        }
        else
        {
            result = acc * coeff;
        }

        _ReduceScalar2DOutArr[int3(outX, outY, (int)id.z)] = float4(result, 0.0, 0.0, 0.0);
        return;
    }

    if (axis == 1)
    {
        int row = oh == 1 ? outX : outY;
        if (row < 0 || row >= inH)
        {
            _ReduceScalar2DOutArr[int3(outX, outY, (int)id.z)] = 0.0;
            return;
        }

        float acc = _ReduceScalar2DInArr[int3(0, row, 0)].x;
        if (opType == 4)
        {
            for (int x = 1; x < inW; x++)
                acc = max(acc, _ReduceScalar2DInArr[int3(x, row, 0)].x);
            result = acc * coeff;
        }
        else if (opType == 5)
        {
            for (int x = 1; x < inW; x++)
                acc = min(acc, _ReduceScalar2DInArr[int3(x, row, 0)].x);
            result = acc * coeff;
        }
        else if (opType == 0 || opType == 3)
        {
            float sum = 0.0;
            for (int x = 0; x < inW; x++)
                sum += _ReduceScalar2DInArr[int3(x, row, 0)].x;
            if (opType == 3)
                sum /= inW;
            result = sum * coeff;
        }
        else if (opType == 2)
        {
            float sumSq = 0.0;
            for (int x = 0; x < inW; x++)
            {
                float v = _ReduceScalar2DInArr[int3(x, row, 0)].x;
                sumSq += v * v;
            }
            result = sumSq * coeff;
        }
        else if (opType == 1 || opType == 7)
        {
            float sumAbs = 0.0;
            for (int x = 0; x < inW; x++)
                sumAbs += abs(_ReduceScalar2DInArr[int3(x, row, 0)].x);
            result = sumAbs * coeff;
        }
        else if (opType == 6 || opType == 10)
        {
            float sumLog = 0.0;
            for (int x = 0; x < inW; x++)
                sumLog += log(max(_ReduceScalar2DInArr[int3(x, row, 0)].x, 1e-12));
            result = exp(sumLog) * coeff;
        }
        else if (opType == 9)
        {
            float sumVal = 0.0;
            for (int x = 0; x < inW; x++)
                sumVal += _ReduceScalar2DInArr[int3(x, row, 0)].x;
            result = log(max(sumVal, 1e-12)) * coeff;
        }
        else if (opType == 8)
        {
            float sumSq = 0.0;
            for (int x = 0; x < inW; x++)
            {
                float v = _ReduceScalar2DInArr[int3(x, row, 0)].x;
                sumSq += v * v;
            }
            result = sqrt(max(sumSq, 0.0)) * coeff;
        }
        else
        {
            result = acc * coeff;
        }

        _ReduceScalar2DOutArr[int3(outX, outY, (int)id.z)] = float4(result, 0.0, 0.0, 0.0);
        return;
    }

    int col = ow == 1 ? outY : outX;
    if (col < 0 || col >= inW)
    {
        _ReduceScalar2DOutArr[int3(outX, outY, (int)id.z)] = 0.0;
        return;
    }

    float acc = _ReduceScalar2DInArr[int3(col, 0, 0)].x;
    if (opType == 4)
    {
        for (int y = 1; y < inH; y++)
            acc = max(acc, _ReduceScalar2DInArr[int3(col, y, 0)].x);
        result = acc * coeff;
    }
    else if (opType == 5)
    {
        for (int y = 1; y < inH; y++)
            acc = min(acc, _ReduceScalar2DInArr[int3(col, y, 0)].x);
        result = acc * coeff;
    }
    else if (opType == 0 || opType == 3)
    {
        float sum = 0.0;
        for (int y = 0; y < inH; y++)
            sum += _ReduceScalar2DInArr[int3(col, y, 0)].x;
        if (opType == 3)
            sum /= inH;
        result = sum * coeff;
    }
    else if (opType == 2)
    {
        float sumSq = 0.0;
        for (int y = 0; y < inH; y++)
        {
            float v = _ReduceScalar2DInArr[int3(col, y, 0)].x;
            sumSq += v * v;
        }
        result = sumSq * coeff;
    }
    else if (opType == 1 || opType == 7)
    {
        float sumAbs = 0.0;
        for (int y = 0; y < inH; y++)
            sumAbs += abs(_ReduceScalar2DInArr[int3(col, y, 0)].x);
        result = sumAbs * coeff;
    }
    else if (opType == 6 || opType == 10)
    {
        float sumLog = 0.0;
        for (int y = 0; y < inH; y++)
            sumLog += log(max(_ReduceScalar2DInArr[int3(col, y, 0)].x, 1e-12));
        result = exp(sumLog) * coeff;
    }
    else if (opType == 9)
    {
        float sumVal = 0.0;
        for (int y = 0; y < inH; y++)
            sumVal += _ReduceScalar2DInArr[int3(col, y, 0)].x;
        result = log(max(sumVal, 1e-12)) * coeff;
    }
    else if (opType == 8)
    {
        float sumSq = 0.0;
        for (int y = 0; y < inH; y++)
        {
            float v = _ReduceScalar2DInArr[int3(col, y, 0)].x;
            sumSq += v * v;
        }
        result = sqrt(max(sumSq, 0.0)) * coeff;
    }
    else
    {
        result = acc * coeff;
    }

    _ReduceScalar2DOutArr[int3(outX, outY, (int)id.z)] = float4(result, 0.0, 0.0, 0.0);
}

void NcnnGemm2DPack4TiledTextureAFromLinear_Impl(uint3 id)
{
    uint ow, oh, od;
    _GemmTexOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int tileRows = max(1, _GemmPack4TiledRowsPerMatrix);
    int row = (int)id.y / tileRows;
    int tileRow = (int)id.y - row * tileRows;
    int outPack = tileRow * (int)ow + (int)id.x;
    int baseCol = outPack * 4;
    if (row < 0 || row >= _MatM || baseCol >= _MatN)
        return;

    float4 acc = 0.0;
    [loop]
    for (int kk = 0; kk < _MatK; ++kk)
    {
        float a = NcnnQuantizeGemmActivationForInt8(_LinearIn0[int2(kk, row)]);
        acc += a * NcnnGemm2DReadB4(baseCol, kk);
    }

    _GemmTexOutArr[int3((int)id.x, (int)id.y, 0)] = NcnnResolveGemm2DBias4(row, baseCol) + acc * _MatAlpha;
}

void NcnnGemm2DPack4TiledTextureAFromPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _GemmTexOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int tileRows = max(1, _GemmPack4TiledRowsPerMatrix);
    int row = (int)id.y / tileRows;
    int tileRow = (int)id.y - row * tileRows;
    int outPack = tileRow * (int)ow + (int)id.x;
    int baseCol = outPack * 4;
    if (row < 0 || row >= _MatM || baseCol >= _MatN)
        return;

    float4 acc = 0.0;
    [loop]
    for (int packX = 0; packX < _GemmTexAInW; ++packX)
    {
        int kkBase = packX << 2;
        float4 a = NcnnQuantizeGemmActivationForInt8(_GemmTexAInArr[int3(packX, row, 0)]);
        [unroll]
        for (int lane = 0; lane < 4; ++lane)
        {
            int kk = kkBase + lane;
            if (kk < _MatK)
                acc += NcnnReadLane(a, lane) * NcnnGemm2DReadB4(baseCol, kk);
        }
    }

    _GemmTexOutArr[int3((int)id.x, (int)id.y, 0)] = NcnnResolveGemm2DBias4(row, baseCol) + acc * _MatAlpha;
}

void NcnnReductionPack4Width_Impl(uint3 id)
{
    uint ow, oh, od;
    _ReduceScalar2DOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int inW = max(1, _ReduceScalar2DInW);
    int inH = max(1, _ReduceScalar2DInH);
    int outX = (int)id.x;
    int row = (int)id.y;
    int pack = (int)id.z;
    if (outX != 0 || row < 0 || row >= inH)
    {
        _ReduceScalar2DOutArr[int3(outX, row, pack)] = 0.0;
        return;
    }

    int opType = _ReduceScalar2DOpType;
    float coeff = _ReduceScalar2DCoeff;
    float4 result = 0.0;

    if (opType == 4)
    {
        float4 acc = _ReduceScalar2DInArr[int3(0, row, pack)];
        for (int x = 1; x < inW; x++)
            acc = max(acc, _ReduceScalar2DInArr[int3(x, row, pack)]);
        result = acc * coeff;
    }
    else if (opType == 5)
    {
        float4 acc = _ReduceScalar2DInArr[int3(0, row, pack)];
        for (int x = 1; x < inW; x++)
            acc = min(acc, _ReduceScalar2DInArr[int3(x, row, pack)]);
        result = acc * coeff;
    }
    else if (opType == 0 || opType == 3)
    {
        float4 sum = 0.0;
        for (int x = 0; x < inW; x++)
            sum += _ReduceScalar2DInArr[int3(x, row, pack)];
        if (opType == 3)
            sum /= (float)inW;
        result = sum * coeff;
    }
    else if (opType == 2)
    {
        float4 sumSq = 0.0;
        for (int x = 0; x < inW; x++)
        {
            float4 v = _ReduceScalar2DInArr[int3(x, row, pack)];
            sumSq += v * v;
        }
        result = sumSq * coeff;
    }
    else if (opType == 1 || opType == 7)
    {
        float4 sumAbs = 0.0;
        for (int x = 0; x < inW; x++)
            sumAbs += abs(_ReduceScalar2DInArr[int3(x, row, pack)]);
        result = sumAbs * coeff;
    }
    else if (opType == 6 || opType == 10)
    {
        float4 sumLog = 0.0;
        for (int x = 0; x < inW; x++)
            sumLog += log(max(_ReduceScalar2DInArr[int3(x, row, pack)], 1e-12));
        result = exp(sumLog) * coeff;
    }
    else if (opType == 9)
    {
        float4 sumVal = 0.0;
        for (int x = 0; x < inW; x++)
            sumVal += _ReduceScalar2DInArr[int3(x, row, pack)];
        result = log(max(sumVal, 1e-12)) * coeff;
    }
    else if (opType == 8)
    {
        float4 sumSq = 0.0;
        for (int x = 0; x < inW; x++)
        {
            float4 v = _ReduceScalar2DInArr[int3(x, row, pack)];
            sumSq += v * v;
        }
        result = sqrt(max(sumSq, 0.0)) * coeff;
    }
    else
    {
        result = _ReduceScalar2DInArr[int3(0, row, pack)] * coeff;
    }

    _ReduceScalar2DOutArr[int3(outX, row, pack)] = result;
}

void NcnnReductionPack4Channel_Impl(uint3 id)
{
    uint ow, oh, od;
    _ReduceScalar2DOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int inW = max(1, _ReduceScalar2DInW);
    int inH = max(1, _ReduceScalar2DInH);
    int inC = max(1, _ReducePack4InC);
    int outX = (int)id.x;
    int outY = (int)id.y;
    int outPack = (int)id.z;
    if (outPack != 0 || outX < 0 || outX >= inW || outY < 0 || outY >= inH)
    {
        _ReduceScalar2DOutArr[int3(outX, outY, outPack)] = 0.0;
        return;
    }

    int opType = _ReduceScalar2DOpType;
    float coeff = _ReduceScalar2DCoeff;
    float result = 0.0;

    if (opType == 4)
    {
        float acc = NcnnReadPack4Channel(_ReduceScalar2DInArr, outX, outY, 0);
        for (int c = 1; c < inC; c++)
            acc = max(acc, NcnnReadPack4Channel(_ReduceScalar2DInArr, outX, outY, c));
        result = acc * coeff;
    }
    else if (opType == 5)
    {
        float acc = NcnnReadPack4Channel(_ReduceScalar2DInArr, outX, outY, 0);
        for (int c = 1; c < inC; c++)
            acc = min(acc, NcnnReadPack4Channel(_ReduceScalar2DInArr, outX, outY, c));
        result = acc * coeff;
    }
    else if (opType == 0 || opType == 3)
    {
        float sum = 0.0;
        for (int c = 0; c < inC; c++)
            sum += NcnnReadPack4Channel(_ReduceScalar2DInArr, outX, outY, c);
        if (opType == 3)
            sum /= (float)inC;
        result = sum * coeff;
    }
    else if (opType == 2)
    {
        float sumSq = 0.0;
        for (int c = 0; c < inC; c++)
        {
            float v = NcnnReadPack4Channel(_ReduceScalar2DInArr, outX, outY, c);
            sumSq += v * v;
        }
        result = sumSq * coeff;
    }
    else if (opType == 1 || opType == 7)
    {
        float sumAbs = 0.0;
        for (int c = 0; c < inC; c++)
            sumAbs += abs(NcnnReadPack4Channel(_ReduceScalar2DInArr, outX, outY, c));
        result = sumAbs * coeff;
    }
    else if (opType == 6 || opType == 10)
    {
        float sumLog = 0.0;
        for (int c = 0; c < inC; c++)
            sumLog += log(max(NcnnReadPack4Channel(_ReduceScalar2DInArr, outX, outY, c), 1e-12));
        result = exp(sumLog) * coeff;
    }
    else if (opType == 9)
    {
        float sumVal = 0.0;
        for (int c = 0; c < inC; c++)
            sumVal += NcnnReadPack4Channel(_ReduceScalar2DInArr, outX, outY, c);
        result = log(max(sumVal, 1e-12)) * coeff;
    }
    else if (opType == 8)
    {
        float sumSq = 0.0;
        for (int c = 0; c < inC; c++)
        {
            float v = NcnnReadPack4Channel(_ReduceScalar2DInArr, outX, outY, c);
            sumSq += v * v;
        }
        result = sqrt(max(sumSq, 0.0)) * coeff;
    }
    else
    {
        result = NcnnReadPack4Channel(_ReduceScalar2DInArr, outX, outY, 0) * coeff;
    }

    _ReduceScalar2DOutArr[int3(outX, outY, outPack)] = float4(result, 0.0, 0.0, 0.0);
}

void NcnnReductionLinearMat2D_Impl(uint3 id)
{
    uint ow, oh;
    _LinearOut0.GetDimensions(ow, oh);
    if (id.x >= ow || id.y >= oh)
        return;

    int outX = (int)id.x;
    int outY = (int)id.y;
    int axis = _ReduceScalar2DAxis;
    int inW = max(1, _ReduceScalar2DInW);
    int inH = max(1, _ReduceScalar2DInH);
    float coeff = _ReduceScalar2DCoeff;
    int opType = _ReduceScalar2DOpType;

    float result = 0.0;

    if (axis == 2)
    {
        if (outX != 0 || outY != 0)
        {
            _LinearOut0[int2(outX, outY)] = 0.0;
            return;
        }

        int total = max(1, inW * inH);
        float acc = _LinearIn0[int2(0, 0)];

        if (opType == 4)
        {
            for (int i = 1; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                acc = max(acc, _LinearIn0[int2(x, y)]);
            }
            result = acc * coeff;
        }
        else if (opType == 5)
        {
            for (int i = 1; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                acc = min(acc, _LinearIn0[int2(x, y)]);
            }
            result = acc * coeff;
        }
        else if (opType == 0 || opType == 3)
        {
            float sum = 0.0;
            for (int i = 0; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                sum += _LinearIn0[int2(x, y)];
            }
            if (opType == 3)
                sum /= total;
            result = sum * coeff;
        }
        else if (opType == 2)
        {
            float sumSq = 0.0;
            for (int i = 0; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                float v = _LinearIn0[int2(x, y)];
                sumSq += v * v;
            }
            result = sumSq * coeff;
        }
        else if (opType == 1 || opType == 7)
        {
            float sumAbs = 0.0;
            for (int i = 0; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                sumAbs += abs(_LinearIn0[int2(x, y)]);
            }
            result = sumAbs * coeff;
        }
        else if (opType == 6 || opType == 10)
        {
            float sumLog = 0.0;
            for (int i = 0; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                sumLog += log(max(_LinearIn0[int2(x, y)], 1e-12));
            }
            result = exp(sumLog) * coeff;
        }
        else if (opType == 9)
        {
            float sumVal = 0.0;
            for (int i = 0; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                sumVal += _LinearIn0[int2(x, y)];
            }
            result = log(max(sumVal, 1e-12)) * coeff;
        }
        else if (opType == 8)
        {
            float sumSq = 0.0;
            for (int i = 0; i < total; i++)
            {
                int x = i % inW;
                int y = i / inW;
                float v = _LinearIn0[int2(x, y)];
                sumSq += v * v;
            }
            result = sqrt(max(sumSq, 0.0)) * coeff;
        }
        else
        {
            result = acc * coeff;
        }

        _LinearOut0[int2(outX, outY)] = result;
        return;
    }

    if (axis == 1)
    {
        int row = oh == 1 ? outX : outY;
        if (row < 0 || row >= inH)
        {
            _LinearOut0[int2(outX, outY)] = 0.0;
            return;
        }

        float acc = _LinearIn0[int2(0, row)];
        if (opType == 4)
        {
            for (int x = 1; x < inW; x++)
                acc = max(acc, _LinearIn0[int2(x, row)]);
            result = acc * coeff;
        }
        else if (opType == 5)
        {
            for (int x = 1; x < inW; x++)
                acc = min(acc, _LinearIn0[int2(x, row)]);
            result = acc * coeff;
        }
        else if (opType == 0 || opType == 3)
        {
            float sum = 0.0;
            for (int x = 0; x < inW; x++)
                sum += _LinearIn0[int2(x, row)];
            if (opType == 3)
                sum /= inW;
            result = sum * coeff;
        }
        else if (opType == 2)
        {
            float sumSq = 0.0;
            for (int x = 0; x < inW; x++)
            {
                float v = _LinearIn0[int2(x, row)];
                sumSq += v * v;
            }
            result = sumSq * coeff;
        }
        else if (opType == 1 || opType == 7)
        {
            float sumAbs = 0.0;
            for (int x = 0; x < inW; x++)
                sumAbs += abs(_LinearIn0[int2(x, row)]);
            result = sumAbs * coeff;
        }
        else if (opType == 6 || opType == 10)
        {
            float sumLog = 0.0;
            for (int x = 0; x < inW; x++)
                sumLog += log(max(_LinearIn0[int2(x, row)], 1e-12));
            result = exp(sumLog) * coeff;
        }
        else if (opType == 9)
        {
            float sumVal = 0.0;
            for (int x = 0; x < inW; x++)
                sumVal += _LinearIn0[int2(x, row)];
            result = log(max(sumVal, 1e-12)) * coeff;
        }
        else if (opType == 8)
        {
            float sumSq = 0.0;
            for (int x = 0; x < inW; x++)
            {
                float v = _LinearIn0[int2(x, row)];
                sumSq += v * v;
            }
            result = sqrt(max(sumSq, 0.0)) * coeff;
        }
        else
        {
            result = acc * coeff;
        }

        _LinearOut0[int2(outX, outY)] = result;
        return;
    }

    int col = ow == 1 ? outY : outX;
    if (col < 0 || col >= inW)
    {
        _LinearOut0[int2(outX, outY)] = 0.0;
        return;
    }

    float acc = _LinearIn0[int2(col, 0)];
    if (opType == 4)
    {
        for (int y = 1; y < inH; y++)
            acc = max(acc, _LinearIn0[int2(col, y)]);
        result = acc * coeff;
    }
    else if (opType == 5)
    {
        for (int y = 1; y < inH; y++)
            acc = min(acc, _LinearIn0[int2(col, y)]);
        result = acc * coeff;
    }
    else if (opType == 0 || opType == 3)
    {
        float sum = 0.0;
        for (int y = 0; y < inH; y++)
            sum += _LinearIn0[int2(col, y)];
        if (opType == 3)
            sum /= inH;
        result = sum * coeff;
    }
    else if (opType == 2)
    {
        float sumSq = 0.0;
        for (int y = 0; y < inH; y++)
        {
            float v = _LinearIn0[int2(col, y)];
            sumSq += v * v;
        }
        result = sumSq * coeff;
    }
    else if (opType == 1 || opType == 7)
    {
        float sumAbs = 0.0;
        for (int y = 0; y < inH; y++)
            sumAbs += abs(_LinearIn0[int2(col, y)]);
        result = sumAbs * coeff;
    }
    else if (opType == 6 || opType == 10)
    {
        float sumLog = 0.0;
        for (int y = 0; y < inH; y++)
            sumLog += log(max(_LinearIn0[int2(col, y)], 1e-12));
        result = exp(sumLog) * coeff;
    }
    else if (opType == 9)
    {
        float sumVal = 0.0;
        for (int y = 0; y < inH; y++)
            sumVal += _LinearIn0[int2(col, y)];
        result = log(max(sumVal, 1e-12)) * coeff;
    }
    else if (opType == 8)
    {
        float sumSq = 0.0;
        for (int y = 0; y < inH; y++)
        {
            float v = _LinearIn0[int2(col, y)];
            sumSq += v * v;
        }
        result = sqrt(max(sumSq, 0.0)) * coeff;
    }
    else
    {
        result = acc * coeff;
    }

    _LinearOut0[int2(outX, outY)] = result;
}
