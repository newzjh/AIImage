// Auto-generated kernel implementation group: AexisKernels.BufferAttention.hlsl

void AexisInnerProduct2D_Impl(uint3 id)
{
    int o = (int)id.x;
    int r = (int)id.y;
    if (o < 0 || r < 0) return;
    if (o >= _IP2OutFeatures || r >= _IP2Rows) return;
    float sum = _IP2B[o];
    uint inBase = (uint)r * (uint)_IP2InFeatures;
    uint wBase = (uint)o * (uint)_IP2InFeatures;
    for (int i = 0; i < _IP2InFeatures; i++)
    {
        sum += _IP2In[inBase + (uint)i] * _IP2W[wBase + (uint)i];
    }
    _IP2Out[(uint)r * (uint)_IP2OutFeatures + (uint)o] = sum;
}

void AexisMhaAttention_Impl(uint3 groupId, uint3 groupThreadId)
{
    int q = (int)groupId.x;
    int head = (int)groupId.y;
    uint tid = groupThreadId.x;
    if (q < 0 || head < 0) return;
    if (q >= _MhaSrcLen || head >= _MhaNumHeads) return;
    int dst = _MhaDstLen;
    if (dst <= 0 || dst > 4096) return;

    int headDim = _MhaHeadDim;
    int embedDim = _MhaEmbedDim;
    int qBase = q * embedDim + head * headDim;
    int kHeadBase = head * headDim;

    // Split score dots over keys. Softmax stays sequential on tid 0 so accumulation order stays
    // close to ncnn/current Unity while avoiding one-thread dot products for the whole row.
    for (int scoreIndex = (int)tid; scoreIndex < dst; scoreIndex += 64)
    {
        int kBase = scoreIndex * embedDim + kHeadBase;
        float s = 0.0;
        for (int dotIndex = 0; dotIndex < headDim; dotIndex++)
        {
            s += (_MhaQ[qBase + dotIndex] * _MhaScale) * _MhaK[kBase + dotIndex];
        }
        s += AexisResolveSdpaMask(head, q, scoreIndex);
        _MhaScores[scoreIndex] = s;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0)
    {
        float maxScore = -3.402823466e+38;
        for (int maxIndex = 0; maxIndex < dst; maxIndex++)
        {
            maxScore = max(maxScore, _MhaScores[maxIndex]);
        }

        float sumExp = 0.0;
        for (int expIndex = 0; expIndex < dst; expIndex++)
        {
            float w = exp(_MhaScores[expIndex] - maxScore);
            _MhaScores[expIndex] = w;
            sumExp += w;
        }

        float invSum = 1.0 / max(sumExp, 1e-20);
        for (int normIndex = 0; normIndex < dst; normIndex++)
        {
            _MhaScores[normIndex] *= invSum;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (int lane = (int)tid; lane < headDim; lane += 64)
    {
        float outv = 0.0;
        int vHeadBase = head * headDim + lane;
        for (int valueIndex = 0; valueIndex < dst; valueIndex++)
        {
            outv += _MhaScores[valueIndex] * _MhaV[valueIndex * embedDim + vHeadBase];
        }
        _MhaOut[q * embedDim + head * headDim + lane] = outv;
    }
}

void AexisMhaAttentionFast_Impl(uint3 groupId, uint3 groupThreadId)
{
    int q = (int)groupId.x;
    int head = (int)groupId.y;
    uint tid = groupThreadId.x;
    if (q < 0 || head < 0) return;
    if (q >= _MhaSrcLen || head >= _MhaNumHeads) return;
    int dst = _MhaDstLen;
    if (dst <= 0 || dst > 4096) return;

    int headDim = _MhaHeadDim;
    int embedDim = _MhaEmbedDim;
    int qBase = q * embedDim + head * headDim;
    int kHeadBase = head * headDim;

    for (int scoreIndex = (int)tid; scoreIndex < dst; scoreIndex += 64)
    {
        int kBase = scoreIndex * embedDim + kHeadBase;
        float s = 0.0;
        for (int dotIndex = 0; dotIndex < headDim; dotIndex++)
        {
            s += (_MhaQ[qBase + dotIndex] * _MhaScale) * _MhaK[kBase + dotIndex];
        }
        s += AexisResolveSdpaMask(head, q, scoreIndex);
        _MhaScores[scoreIndex] = s;
    }
    GroupMemoryBarrierWithGroupSync();

    float localMax = -3.402823466e+38;
    for (int maxIndex = (int)tid; maxIndex < dst; maxIndex += 64)
    {
        localMax = max(localMax, _MhaScores[maxIndex]);
    }
    _MhaReduce[tid] = localMax;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint maxStride = 32; maxStride > 0; maxStride >>= 1)
    {
        if (tid < maxStride)
            _MhaReduce[tid] = max(_MhaReduce[tid], _MhaReduce[tid + maxStride]);
        GroupMemoryBarrierWithGroupSync();
    }
    float maxScore = _MhaReduce[0];

    float localSum = 0.0;
    for (int expIndex = (int)tid; expIndex < dst; expIndex += 64)
    {
        float w = exp(_MhaScores[expIndex] - maxScore);
        _MhaScores[expIndex] = w;
        localSum += w;
    }
    _MhaReduce[tid] = localSum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint sumStride = 32; sumStride > 0; sumStride >>= 1)
    {
        if (tid < sumStride)
            _MhaReduce[tid] += _MhaReduce[tid + sumStride];
        GroupMemoryBarrierWithGroupSync();
    }
    float invSum = 1.0 / max(_MhaReduce[0], 1e-20);

    for (int normIndex = (int)tid; normIndex < dst; normIndex += 64)
    {
        _MhaScores[normIndex] *= invSum;
    }
    GroupMemoryBarrierWithGroupSync();

    for (int lane = (int)tid; lane < headDim; lane += 64)
    {
        float outv = 0.0;
        int vHeadBase = head * headDim + lane;
        for (int valueIndex = 0; valueIndex < dst; valueIndex++)
        {
            outv += _MhaScores[valueIndex] * _MhaV[valueIndex * embedDim + vHeadBase];
        }
        _MhaOut[q * embedDim + head * headDim + lane] = outv;
    }
}

void AexisMhaAttentionQkvFast_Impl(uint3 groupId, uint3 groupThreadId)
{
    int q = (int)groupId.x;
    int head = (int)groupId.y;
    uint tid = groupThreadId.x;
    if (q < 0 || head < 0) return;
    if (q >= _MhaSrcLen || head >= _MhaNumHeads) return;
    int dst = _MhaDstLen;
    if (dst <= 0 || dst > 4096) return;

    int headDim = _MhaHeadDim;
    int embedDim = _MhaEmbedDim;
    int qkvStride = embedDim * 3;
    int qBase = q * qkvStride + head * headDim;
    int kHeadBase = embedDim + head * headDim;

    for (int scoreIndex = (int)tid; scoreIndex < dst; scoreIndex += 64)
    {
        int kBase = scoreIndex * qkvStride + kHeadBase;
        float s = 0.0;
        for (int dotIndex = 0; dotIndex < headDim; dotIndex++)
        {
            s += (_MhaQkv[qBase + dotIndex] * _MhaScale) * _MhaQkv[kBase + dotIndex];
        }
        _MhaScores[scoreIndex] = s;
    }
    GroupMemoryBarrierWithGroupSync();

    float localMax = -3.402823466e+38;
    for (int maxIndex = (int)tid; maxIndex < dst; maxIndex += 64)
    {
        localMax = max(localMax, _MhaScores[maxIndex]);
    }
    _MhaReduce[tid] = localMax;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint maxStride = 32; maxStride > 0; maxStride >>= 1)
    {
        if (tid < maxStride)
            _MhaReduce[tid] = max(_MhaReduce[tid], _MhaReduce[tid + maxStride]);
        GroupMemoryBarrierWithGroupSync();
    }
    float maxScore = _MhaReduce[0];

    float localSum = 0.0;
    for (int expIndex = (int)tid; expIndex < dst; expIndex += 64)
    {
        float w = exp(_MhaScores[expIndex] - maxScore);
        _MhaScores[expIndex] = w;
        localSum += w;
    }
    _MhaReduce[tid] = localSum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint sumStride = 32; sumStride > 0; sumStride >>= 1)
    {
        if (tid < sumStride)
            _MhaReduce[tid] += _MhaReduce[tid + sumStride];
        GroupMemoryBarrierWithGroupSync();
    }
    float invSum = 1.0 / max(_MhaReduce[0], 1e-20);

    for (int normIndex = (int)tid; normIndex < dst; normIndex += 64)
    {
        _MhaScores[normIndex] *= invSum;
    }
    GroupMemoryBarrierWithGroupSync();

    for (int lane = (int)tid; lane < headDim; lane += 64)
    {
        float outv = 0.0;
        int vHeadBase = embedDim * 2 + head * headDim + lane;
        for (int valueIndex = 0; valueIndex < dst; valueIndex++)
        {
            outv += _MhaScores[valueIndex] * _MhaQkv[valueIndex * qkvStride + vHeadBase];
        }
        _MhaOut[q * embedDim + head * headDim + lane] = outv;
    }
}

void AexisMhaProjectQkv2D_Impl(uint3 groupId, uint3 groupThreadId)
{
    int lx = (int)groupThreadId.x;
    int ly = (int)groupThreadId.y;
    int row = (int)groupId.y * 16 + ly;
    int col = (int)groupId.x * 16 + lx;
    int projection = (int)groupId.z;
    bool valid = row < _MhaProjectRows
                 && col < _MhaProjectOutFeatures
                 && projection >= 0
                 && projection < 3;

    float sum = 0.0;
    if (valid)
    {
        if (projection == 0)
            sum = _MhaProjectQB[col];
        else if (projection == 1)
            sum = _MhaProjectKB[col];
        else
            sum = _MhaProjectVB[col];
    }

    float acc = 0.0;
    for (int k0 = 0; k0 < _MhaProjectInFeatures; k0 += 16)
    {
        int ak = k0 + lx;
        int bk = k0 + ly;
        float a = 0.0;
        float b = 0.0;

        if (row < _MhaProjectRows && ak < _MhaProjectInFeatures)
            a = _MhaProjectIn[row * _MhaProjectInFeatures + ak];
        if (col < _MhaProjectOutFeatures && bk < _MhaProjectInFeatures)
        {
            if (projection == 0)
                b = _MhaProjectQW[col * _MhaProjectInFeatures + bk];
            else if (projection == 1)
                b = _MhaProjectKW[col * _MhaProjectInFeatures + bk];
            else
                b = _MhaProjectVW[col * _MhaProjectInFeatures + bk];
        }

        _MhaProjectAs[ly * 16 + lx] = a;
        _MhaProjectBs[ly * 16 + lx] = b;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (int k1 = 0; k1 < 16; k1++)
        {
            int kk = k0 + k1;
            if (kk < _MhaProjectInFeatures)
                acc += _MhaProjectAs[ly * 16 + k1] * _MhaProjectBs[k1 * 16 + lx];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (valid)
    {
        int stride = _MhaProjectOutFeatures * 3;
        _MhaProjectOut[row * stride + projection * _MhaProjectOutFeatures + col] = sum + acc;
    }
}

void AexisSdpaQkBuf_Impl(uint3 groupId, uint3 groupThreadId)
{
    int row = (int)groupId.x;
    int head = (int)groupId.y;
    uint tid = groupThreadId.x;
    if (row < 0 || head < 0) return;
    if (row >= _SdpaSrcLen || head >= _SdpaNumHeads) return;

    int dst = _SdpaDstLen;
    if (dst <= 0 || dst > 4096) return;

    int embedDim = _SdpaEmbedDim;
    int keyHead = head / max(1, _SdpaNumHeadsPerGroup);
    int qBase = (head * _SdpaSrcLen + row) * embedDim;
    int kBaseHead = keyHead * dst * embedDim;
    int scoreBase = (head * _SdpaSrcLen + row) * dst;

    for (int col = (int)tid; col < dst; col += 64)
    {
        int kBase = kBaseHead + col * embedDim;
        float s = 0.0;
        for (int i = 0; i < embedDim; i++)
            s += _SdpaQ[qBase + i] * _SdpaK[kBase + i];
        s *= _SdpaScale;
        s += AexisResolveSdpaMask(head, row, col);
        _SdpaScores[scoreBase + col] = s;
    }
}

void AexisSdpaSoftmaxBuf_Impl(uint3 groupId, uint3 groupThreadId)
{
    int row = (int)groupId.x;
    int head = (int)groupId.y;
    uint tid = groupThreadId.x;
    if (row < 0 || head < 0) return;
    if (row >= _SdpaSrcLen || head >= _SdpaNumHeads) return;

    int dst = _SdpaDstLen;
    if (dst <= 0 || dst > 4096) return;

    int scoreBase = (head * _SdpaSrcLen + row) * dst;
    float localMax = -3.402823466e+38;
    for (int col = (int)tid; col < dst; col += 64)
        localMax = max(localMax, _SdpaScores[scoreBase + col]);
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
    for (int col = (int)tid; col < dst; col += 64)
    {
        float e = exp(_SdpaScores[scoreBase + col] - maxValue);
        _SdpaScores[scoreBase + col] = e;
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
    for (int col = (int)tid; col < dst; col += 64)
        _SdpaScores[scoreBase + col] *= invSum;
}

void AexisSdpaQkvBuf_Impl(uint3 groupId, uint3 groupThreadId)
{
    int row = (int)groupId.x;
    int head = (int)groupId.y;
    uint tid = groupThreadId.x;
    if (row < 0 || head < 0) return;
    if (row >= _SdpaSrcLen || head >= _SdpaNumHeads) return;

    int dst = _SdpaDstLen;
    int outEmbedDim = _SdpaOutEmbedDim;
    if (dst <= 0 || dst > 4096 || outEmbedDim <= 0) return;

    int keyHead = head / max(1, _SdpaNumHeadsPerGroup);
    int scoreBase = (head * _SdpaSrcLen + row) * dst;
    int valueBase = keyHead * dst * outEmbedDim;
    int outBase = (head * _SdpaSrcLen + row) * outEmbedDim;

    for (int lane = (int)tid; lane < outEmbedDim; lane += 64)
    {
        float sum = 0.0;
        for (int col = 0; col < dst; col++)
            sum += _SdpaScores[scoreBase + col] * _SdpaV[valueBase + col * outEmbedDim + lane];
        _SdpaOut[outBase + lane] = sum;
    }
}

void AexisSdpaAttentionFast_Impl(uint3 groupId, uint3 groupThreadId)
{
    int row = (int)groupId.x;
    int head = (int)groupId.y;
    uint tid = groupThreadId.x;
    if (row < 0 || head < 0) return;
    if (row >= _SdpaSrcLen || head >= _SdpaNumHeads) return;

    int dst = _SdpaDstLen;
    int embedDim = _SdpaEmbedDim;
    int outEmbedDim = _SdpaOutEmbedDim;
    if (dst <= 0 || dst > 4096 || embedDim <= 0 || outEmbedDim <= 0) return;

    int keyHead = head / max(1, _SdpaNumHeadsPerGroup);
    int qBase = (head * _SdpaSrcLen + row) * embedDim;
    int kBaseHead = keyHead * dst * embedDim;
    int valueBase = keyHead * dst * outEmbedDim;
    int outBase = (head * _SdpaSrcLen + row) * outEmbedDim;

    for (int col = (int)tid; col < dst; col += 64)
    {
        int kBase = kBaseHead + col * embedDim;
        float s = 0.0;
        for (int i = 0; i < embedDim; i++)
            s += _SdpaQ[qBase + i] * _SdpaK[kBase + i];
        s *= _SdpaScale;
        s += AexisResolveSdpaMask(head, row, col);
        _SdpaScoresFast[col] = s;
    }
    GroupMemoryBarrierWithGroupSync();

    float localMax = -3.402823466e+38;
    for (int col = (int)tid; col < dst; col += 64)
        localMax = max(localMax, _SdpaScoresFast[col]);
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
    for (int col = (int)tid; col < dst; col += 64)
    {
        float e = exp(_SdpaScoresFast[col] - maxValue);
        _SdpaScoresFast[col] = e;
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
    for (int col = (int)tid; col < dst; col += 64)
        _SdpaScoresFast[col] *= invSum;
    GroupMemoryBarrierWithGroupSync();

    for (int lane = (int)tid; lane < outEmbedDim; lane += 64)
    {
        float sum = 0.0;
        for (int col = 0; col < dst; col++)
            sum += _SdpaScoresFast[col] * _SdpaV[valueBase + col * outEmbedDim + lane];
        _SdpaOut[outBase + lane] = sum;
    }
}
