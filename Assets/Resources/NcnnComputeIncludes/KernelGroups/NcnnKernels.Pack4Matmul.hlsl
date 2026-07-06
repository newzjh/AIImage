// Auto-generated kernel implementation group: NcnnKernels.Pack4Matmul.hlsl

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

void NcnnGemm2DTextureA_Impl(uint3 id)
{
    uint ow, oh, od;
    _GemmTexOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int row = (int)id.y;
    int col = (int)id.x;
    if (row < 0 || row >= _MatM || col < 0 || col >= _MatN)
        return;

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

    float acc = 0.0;
    [loop]
    for (int kk = 0; kk < _MatK; kk++)
    {
        float a = 0.0;
        if (kk >= 0 && kk < _GemmTexAInW && row >= 0 && row < _GemmTexAInH)
            a = _GemmTexAInArr[int3(kk, row, 0)].x;

        float b = _MatTransB != 0
            ? _MatB[col * _MatK + kk]
            : _MatB[kk * _MatN + col];
        acc += a * b;
    }

    sum += acc * _MatAlpha;
    _GemmTexOutArr[int3(col, row, 0)] = float4(sum, 0.0, 0.0, 0.0);
}

void NcnnGemm2DLinearTextureA_Impl(uint3 id)
{
    uint ow, oh;
    _LinearOut0.GetDimensions(ow, oh);
    if (id.x >= ow || id.y >= oh)
        return;

    int row = (int)id.y;
    int col = (int)id.x;
    if (row < 0 || row >= _MatM || col < 0 || col >= _MatN)
        return;

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

    float acc = 0.0;
    [loop]
    for (int kk = 0; kk < _MatK; kk++)
    {
        float a = 0.0;
        if (kk >= 0 && kk < _GemmTexAInW && row >= 0 && row < _GemmTexAInH)
            a = _LinearIn0[int2(kk, row)];

        float b = _MatTransB != 0
            ? _MatB[col * _MatK + kk]
            : _MatB[kk * _MatN + col];
        acc += a * b;
    }

    sum += acc * _MatAlpha;
    _LinearOut0[int2(col, row)] = sum;
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

    float4 sum = 0.0;
    float4 sqsum = 0.0;
    for (int ix = 0; ix < _LnW; ix++)
    {
        float4 v = _LnTexInArr[int3(ix, y, slice)];
        sum += v;
        sqsum += v * v;
    }

    float widthCount = max(1, _LnW);
    float4 mean = sum / widthCount;
    float4 var = sqsum / widthCount - mean * mean;
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

        float maxv = -3.402823466e+38;
        for (int xx = 0; xx < _SoftmaxPack4CDHWW; xx++)
        {
            float v = NcnnReadPack4ChannelCDHW(_SoftmaxPack4CDHWInArr, xx, outY, outZ, c, _SoftmaxPack4CDHWC);
            maxv = max(maxv, v);
        }

        float sum = 0.0;
        for (int xx = 0; xx < _SoftmaxPack4CDHWW; xx++)
        {
            float v = NcnnReadPack4ChannelCDHW(_SoftmaxPack4CDHWInArr, xx, outY, outZ, c, _SoftmaxPack4CDHWC);
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

    float maxv = _LinearIn0[int2(0, row)];
    for (int i = 1; i < inW; i++)
        maxv = max(maxv, _LinearIn0[int2(i, row)]);

    float sum = 0.0;
    for (int i = 0; i < inW; i++)
        sum += exp(_LinearIn0[int2(i, row)] - maxv);

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
