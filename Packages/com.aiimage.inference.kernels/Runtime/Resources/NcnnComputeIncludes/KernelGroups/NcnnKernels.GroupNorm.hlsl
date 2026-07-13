// Auto-generated kernel implementation group: NcnnKernels.GroupNorm.hlsl

void NcnnGroupNormStats_Impl(uint3 groupId, uint3 groupThreadId)
{
    int g = (int)groupId.x;
    int tid = (int)groupThreadId.x;
    if (g < 0 || g >= _GnGroup) return;

    uint size = (uint)(_GnW * _GnH);
    uint total = (uint)(_GnChannelsG) * size;
    uint chBase = (uint)g * (uint)_GnChannelsG;

    float sum = 0.0;
    float sqsum = 0.0;
    for (uint t = (uint)tid; t < total; t += 256u)
    {
        uint ch = t / size;
        uint s = t - ch * size;
        uint c = chBase + ch;
        float x = _GnInOut[c * size + s];
        sum += x;
        sqsum += x * x;
    }

    red0[tid] = sum;
    red1[tid] = sqsum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (int stGn = 128; stGn > 0; stGn >>= 1)
    {
        if (tid < stGn)
        {
            red0[tid] += red0[tid + stGn];
            red1[tid] += red1[tid + stGn];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
        _GnStatsOut[g] = float4(red0[0], red1[0], 0.0, 0.0);
}

void NcnnGroupNormApply_Impl(uint3 groupId, uint3 groupThreadId)
{
    int g = (int)groupId.x;
    int tid = (int)groupThreadId.x;
    if (g < 0 || g >= _GnGroup) return;

    uint size = (uint)(_GnW * _GnH);
    uint total = (uint)(_GnChannelsG) * size;
    uint chBase = (uint)g * (uint)_GnChannelsG;

    float2 stats = _GnStatsOut[g].xy;
    float denom = (float)max(1u, total);
    float mean = stats.x / denom;
    float var = stats.y / denom - mean * mean;
    float invstd = rsqrt(max(var + _GnEps, 1e-20));

    for (uint t = (uint)tid; t < total; t += 256u)
    {
        uint ch = t / size;
        uint s = t - ch * size;
        uint c = chBase + ch;
        float x = _GnInOut[c * size + s];
        float y = (x - mean) * invstd;
        if (_GnAffine != 0)
            y = y * _GnGamma[c] + _GnBeta[c];
        _GnInOut[c * size + s] = y;
    }
}

void NcnnGroupNormMean_Impl(uint3 groupId, uint3 groupThreadId)
{
    int g = (int)groupId.x;
    int tid = (int)groupThreadId.x;
    if (g < 0 || g >= _GnGroup) return;

    uint size = (uint)(_GnW * _GnH);
    uint total = (uint)(_GnChannelsG) * size;
    uint chBase = (uint)g * (uint)_GnChannelsG;

    float sum = 0.0;
    for (uint t = (uint)tid; t < total; t += 256u)
    {
        uint ch = t / size;
        uint s = t - ch * size;
        uint c = chBase + ch;
        sum += _GnInOut[c * size + s];
    }

    red0[tid] = sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (int stGn = 128; stGn > 0; stGn >>= 1)
    {
        if (tid < stGn)
            red0[tid] += red0[tid + stGn];
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
    {
        float denom = (float)max(1u, total);
        _GnStatsOut[g] = float4(red0[0] / denom, 0.0, 0.0, 0.0);
    }
}

void NcnnGroupNormVariance_Impl(uint3 groupId, uint3 groupThreadId)
{
    int g = (int)groupId.x;
    int tid = (int)groupThreadId.x;
    if (g < 0 || g >= _GnGroup) return;

    uint size = (uint)(_GnW * _GnH);
    uint total = (uint)(_GnChannelsG) * size;
    uint chBase = (uint)g * (uint)_GnChannelsG;
    float mean = _GnStatsOut[g].x;

    float sqsum = 0.0;
    for (uint t = (uint)tid; t < total; t += 256u)
    {
        uint ch = t / size;
        uint s = t - ch * size;
        uint c = chBase + ch;
        float v = _GnInOut[c * size + s] - mean;
        sqsum += v * v;
    }

    red0[tid] = sqsum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (int stGn = 128; stGn > 0; stGn >>= 1)
    {
        if (tid < stGn)
            red0[tid] += red0[tid + stGn];
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
    {
        float denom = (float)max(1u, total);
        float mean0 = _GnStatsOut[g].x;
        _GnStatsOut[g] = float4(mean0, red0[0] / denom, 0.0, 0.0);
    }
}

void NcnnGroupNormApplyMeanVar_Impl(uint3 groupId, uint3 groupThreadId)
{
    int g = (int)groupId.x;
    int tid = (int)groupThreadId.x;
    if (g < 0 || g >= _GnGroup) return;

    uint size = (uint)(_GnW * _GnH);
    uint total = (uint)(_GnChannelsG) * size;
    uint chBase = (uint)g * (uint)_GnChannelsG;

    float2 stats = _GnStatsOut[g].xy;
    float mean = stats.x;
    float invstd = rsqrt(max(stats.y + _GnEps, 1e-20));

    for (uint t = (uint)tid; t < total; t += 256u)
    {
        uint ch = t / size;
        uint s = t - ch * size;
        uint c = chBase + ch;
        float y = (_GnInOut[c * size + s] - mean) * invstd;
        if (_GnAffine != 0)
            y = y * _GnGamma[c] + _GnBeta[c];
        _GnInOut[c * size + s] = y;
    }
}

void NcnnGroupNormPack4Mean_Impl(uint3 groupId, uint3 groupThreadId)
{
    int g = (int)groupId.x;
    int tid = (int)groupThreadId.x;
    if (g < 0 || g >= _GnGroup) return;

    uint size = (uint)(_GnW * _GnH * max(_GnD, 1));
    uint total = (uint)_GnChannelsG * size;
    uint chBase = (uint)g * (uint)_GnChannelsG;

    float sum = 0.0;
    for (uint t = (uint)tid; t < total; t += 256u)
    {
        uint ch = t / size;
        uint s = t - ch * size;
        uint c = chBase + ch;
        sum += NcnnGnReadPack4(c, s);
    }

    red0[tid] = sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (int stGn = 128; stGn > 0; stGn >>= 1)
    {
        if (tid < stGn)
            red0[tid] += red0[tid + stGn];
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
    {
        float denom = (float)max(1u, total);
        _GnStatsOut[g] = float4(red0[0] / denom, 0.0, 0.0, 0.0);
    }
}

void NcnnGroupNormPack4Variance_Impl(uint3 groupId, uint3 groupThreadId)
{
    int g = (int)groupId.x;
    int tid = (int)groupThreadId.x;
    if (g < 0 || g >= _GnGroup) return;

    uint size = (uint)(_GnW * _GnH * max(_GnD, 1));
    uint total = (uint)_GnChannelsG * size;
    uint chBase = (uint)g * (uint)_GnChannelsG;
    float mean = _GnStatsOut[g].x;

    float sqsum = 0.0;
    for (uint t = (uint)tid; t < total; t += 256u)
    {
        uint ch = t / size;
        uint s = t - ch * size;
        uint c = chBase + ch;
        float v = NcnnGnReadPack4(c, s) - mean;
        sqsum += v * v;
    }

    red0[tid] = sqsum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (int stGn = 128; stGn > 0; stGn >>= 1)
    {
        if (tid < stGn)
            red0[tid] += red0[tid + stGn];
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
    {
        float denom = (float)max(1u, total);
        float mean0 = _GnStatsOut[g].x;
        _GnStatsOut[g] = float4(mean0, red0[0] / denom, 0.0, 0.0);
    }
}

void NcnnGroupNormPack4ApplyMeanVar_Impl(uint3 id)
{
    uint w, h, d;
    _GnTexOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    int packCount = max(1, (_GnC + 3) / 4);
    int z = p / packCount;
    int pack = p - z * packCount;
    float4 v = _GnTexInArr[int3((int)id.x, (int)id.y, p)];
    int c0 = pack * 4 + 0;
    int c1 = pack * 4 + 1;
    int c2 = pack * 4 + 2;
    int c3 = pack * 4 + 3;
    _GnTexOutArr[int3((int)id.x, (int)id.y, p)] = float4(
        NcnnGnApplyOne(v.x, c0),
        NcnnGnApplyOne(v.y, c1),
        NcnnGnApplyOne(v.z, c2),
        NcnnGnApplyOne(v.w, c3));
}

void NcnnGroupNormPack4MeanTex_Impl(uint3 groupId, uint3 groupThreadId)
{
    int g = (int)groupId.x;
    int tid = (int)groupThreadId.x;
    if (g < 0 || g >= _GnGroup) return;

    uint size = (uint)(_GnW * _GnH * max(_GnD, 1));
    uint total = (uint)_GnChannelsG * size;
    uint chBase = (uint)g * (uint)_GnChannelsG;

    float sum = 0.0;
    for (uint t = (uint)tid; t < total; t += 256u)
    {
        uint ch = t / size;
        uint s = t - ch * size;
        uint c = chBase + ch;
        sum += NcnnGnReadPack4(c, s);
    }

    red0[tid] = sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (int stGn = 128; stGn > 0; stGn >>= 1)
    {
        if (tid < stGn)
            red0[tid] += red0[tid + stGn];
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
    {
        float denom = (float)max(1u, total);
        _GnStatsTexOut[int3(g, 0, 0)] = float4(red0[0] / denom, 0.0, 0.0, 0.0);
    }
}

void NcnnGroupNormPack4VarianceTex_Impl(uint3 groupId, uint3 groupThreadId)
{
    int g = (int)groupId.x;
    int tid = (int)groupThreadId.x;
    if (g < 0 || g >= _GnGroup) return;

    uint size = (uint)(_GnW * _GnH * max(_GnD, 1));
    uint total = (uint)_GnChannelsG * size;
    uint chBase = (uint)g * (uint)_GnChannelsG;
    float mean = _GnStatsTexIn[int3(g, 0, 0)].x;

    float sqsum = 0.0;
    for (uint t = (uint)tid; t < total; t += 256u)
    {
        uint ch = t / size;
        uint s = t - ch * size;
        uint c = chBase + ch;
        float v = NcnnGnReadPack4(c, s) - mean;
        sqsum += v * v;
    }

    red0[tid] = sqsum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (int stGn = 128; stGn > 0; stGn >>= 1)
    {
        if (tid < stGn)
            red0[tid] += red0[tid + stGn];
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
        _GnStatsTexOut[int3(g, 0, 0)] = float4(mean, red0[0] / (float)max(1u, total), 0.0, 0.0);
}

void NcnnGroupNormPack4ApplyMeanVarTex_Impl(uint3 id)
{
    uint w, h, d;
    _GnTexOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    int packCount = max(1, (_GnC + 3) / 4);
    int z = p / packCount;
    int pack = p - z * packCount;
    float4 v = _GnTexInArr[int3((int)id.x, (int)id.y, p)];
    int c0 = pack * 4 + 0;
    int c1 = pack * 4 + 1;
    int c2 = pack * 4 + 2;
    int c3 = pack * 4 + 3;
    _GnTexOutArr[int3((int)id.x, (int)id.y, p)] = float4(
        NcnnGnApplyOneTex(v.x, c0),
        NcnnGnApplyOneTex(v.y, c1),
        NcnnGnApplyOneTex(v.z, c2),
        NcnnGnApplyOneTex(v.w, c3));
}
