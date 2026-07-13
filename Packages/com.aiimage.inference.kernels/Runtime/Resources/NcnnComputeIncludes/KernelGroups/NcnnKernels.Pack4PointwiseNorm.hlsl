// Pack4 texture variants for ncnn pointwise, quantization, and normalization layers.

int NcnnPack4TextureChannelCount(int dims, int c)
{
    return dims >= 3 ? max(1, c) : 1;
}

int NcnnPack4TexturePackCount(int dims, int c)
{
    return max(1, (NcnnPack4TextureChannelCount(dims, c) + 3) / 4);
}

void NcnnDecodePack4TextureSlice(int dims, int c, int slice, out int z, out int pack)
{
    int packs = NcnnPack4TexturePackCount(dims, c);
    if (dims == 4)
    {
        z = slice / packs;
        pack = slice - z * packs;
        return;
    }

    z = 0;
    pack = slice;
}

bool NcnnIsPack4TextureLaneValid(int dims, int c, int channel)
{
    if (dims <= 2)
        return channel == 0;
    return channel >= 0 && channel < c;
}

float NcnnReadPack4TensorScalar(Texture2DArray<float4> tex, int dims, int w, int h, int d, int c, int x, int y, int z, int channel)
{
    if (x < 0 || x >= w || y < 0 || y >= max(1, h))
        return 0.0;

    if (dims <= 1)
        return tex[int3(x, 0, 0)].x;

    if (dims == 2)
        return tex[int3(x, y, 0)].x;

    if (dims == 3)
        return NcnnReadPack4Channel(tex, x, y, channel);

    if (z < 0 || z >= max(1, d))
        return 0.0;
    return NcnnReadPack4ChannelCDHW(tex, x, y, z, channel, c);
}

int NcnnPack4TensorLinearIndex(int dims, int w, int h, int d, int c, int x, int y, int z, int channel)
{
    int width = max(1, w);
    int height = max(1, h);
    int depth = max(1, d);

    if (dims <= 1)
        return x;
    if (dims == 2)
        return y * width + x;
    if (dims == 3)
        return (channel * height + y) * width + x;
    return ((channel * depth + z) * height + y) * width + x;
}

float NcnnReadPack4TensorScalarByLinear(Texture2DArray<float4> tex, int linearIndex, int dims, int w, int h, int d, int c)
{
    int width = max(1, w);
    int height = max(1, h);
    int depth = max(1, d);

    if (dims <= 1)
    {
        if (linearIndex < 0 || linearIndex >= width)
            return 0.0;
        return tex[int3(linearIndex, 0, 0)].x;
    }

    if (dims == 2)
    {
        int y = linearIndex / width;
        int x = linearIndex - y * width;
        if (y < 0 || y >= height)
            return 0.0;
        return tex[int3(x, y, 0)].x;
    }

    int plane = width * height;
    if (dims == 3)
    {
        int channel = linearIndex / plane;
        int rem = linearIndex - channel * plane;
        int y = rem / width;
        int x = rem - y * width;
        if (channel < 0 || channel >= c)
            return 0.0;
        return NcnnReadPack4Channel(tex, x, y, channel);
    }

    int volume = plane * depth;
    int channel4 = linearIndex / volume;
    int rem4 = linearIndex - channel4 * volume;
    int z = rem4 / plane;
    rem4 -= z * plane;
    int y4 = rem4 / width;
    int x4 = rem4 - y4 * width;
    if (channel4 < 0 || channel4 >= c || z < 0 || z >= depth)
        return 0.0;
    return NcnnReadPack4ChannelCDHW(tex, x4, y4, z, channel4, c);
}

int NcnnPack4QuantAxis(int dims, int w, int h, int d, int x, int y, int z, int channel)
{
    if (dims <= 1)
        return x;
    if (dims == 2)
        return y;
    return channel;
}

void NcnnCastPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _TexOut0Arr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int z;
    int pack;
    NcnnDecodePack4TextureSlice(_QuantDims, _QuantC, (int)id.z, z, pack);
    float4 src = _TexIn0Arr[int3((int)id.x, (int)id.y, (int)id.z)];
    float4 dst = 0.0;

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int channel = _QuantDims >= 3 ? pack * 4 + lane : lane;
        if (!NcnnIsPack4TextureLaneValid(_QuantDims, _QuantC, channel))
            continue;
        NcnnWriteLane(dst, lane, NcnnCastScalar(NcnnReadLane(src, lane)));
    }

    _TexOut0Arr[int3((int)id.x, (int)id.y, (int)id.z)] = dst;
}

void NcnnQuantizePack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _TexOut0Arr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int z;
    int pack;
    NcnnDecodePack4TextureSlice(_QuantDims, _QuantC, (int)id.z, z, pack);
    float4 src = _TexIn0Arr[int3((int)id.x, (int)id.y, (int)id.z)];
    float4 dst = 0.0;

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int channel = _QuantDims >= 3 ? pack * 4 + lane : lane;
        if (!NcnnIsPack4TextureLaneValid(_QuantDims, _QuantC, channel))
            continue;

        int axis = NcnnPack4QuantAxis(_QuantDims, _QuantW, _QuantH, _QuantD, (int)id.x, (int)id.y, z, channel);
        float scale = NcnnResolveQuantScaleIn(axis);
        float value = clamp(NcnnRoundToNearest(NcnnReadLane(src, lane) * scale), -127.0, 127.0);
        NcnnWriteLane(dst, lane, value);
    }

    _TexOut0Arr[int3((int)id.x, (int)id.y, (int)id.z)] = dst;
}

void NcnnDequantizePack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _TexOut0Arr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int z;
    int pack;
    NcnnDecodePack4TextureSlice(_QuantDims, _QuantC, (int)id.z, z, pack);
    float4 src = _TexIn0Arr[int3((int)id.x, (int)id.y, (int)id.z)];
    float4 dst = 0.0;

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int channel = _QuantDims >= 3 ? pack * 4 + lane : lane;
        if (!NcnnIsPack4TextureLaneValid(_QuantDims, _QuantC, channel))
            continue;

        int axis = NcnnPack4QuantAxis(_QuantDims, _QuantW, _QuantH, _QuantD, (int)id.x, (int)id.y, z, channel);
        float scale = NcnnResolveQuantScaleIn(axis);
        float bias = NcnnResolveQuantBias(axis);
        float value = NcnnRoundToNearest(NcnnReadLane(src, lane)) * scale + bias;
        NcnnWriteLane(dst, lane, value);
    }

    _TexOut0Arr[int3((int)id.x, (int)id.y, (int)id.z)] = dst;
}

void NcnnRequantizePack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _TexOut0Arr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int z;
    int pack;
    NcnnDecodePack4TextureSlice(_QuantDims, _QuantC, (int)id.z, z, pack);
    float4 src = _TexIn0Arr[int3((int)id.x, (int)id.y, (int)id.z)];
    float4 dst = 0.0;

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int channel = _QuantDims >= 3 ? pack * 4 + lane : lane;
        if (!NcnnIsPack4TextureLaneValid(_QuantDims, _QuantC, channel))
            continue;

        int axis = NcnnPack4QuantAxis(_QuantDims, _QuantW, _QuantH, _QuantD, (int)id.x, (int)id.y, z, channel);
        float scaleIn = NcnnResolveQuantScaleIn(axis);
        float scaleOut = NcnnResolveQuantScaleOut(axis);
        float bias = NcnnResolveQuantBias(axis);
        float value = NcnnRoundToNearest(NcnnReadLane(src, lane)) * scaleIn + bias;
        value = NcnnApplyActivationScalarQuant(value);
        value = clamp(NcnnRoundToNearest(value * scaleOut), -127.0, 127.0);
        NcnnWriteLane(dst, lane, value);
    }

    _TexOut0Arr[int3((int)id.x, (int)id.y, (int)id.z)] = dst;
}

void NcnnNormalizePack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _TexOut0Arr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int z;
    int pack;
    NcnnDecodePack4TextureSlice(_NormDims, _NormC, (int)id.z, z, pack);
    float4 src = _TexIn0Arr[int3((int)id.x, (int)id.y, (int)id.z)];
    float4 dst = 0.0;
    int plane = max(1, _NormW * _NormH * _NormD);
    int channels = _NormDims >= 3 ? max(1, _NormC) : 1;
    int s = (z * max(1, _NormH) + (int)id.y) * max(1, _NormW) + (int)id.x;

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int channel = _NormDims >= 3 ? pack * 4 + lane : lane;
        if (!NcnnIsPack4TextureLaneValid(_NormDims, _NormC, channel))
            continue;

        float value = NcnnReadLane(src, lane);
        float sumSquare = 0.0;

        if (_NormAcrossSpatial != 0 && _NormAcrossChannel != 0)
        {
            for (int c0 = 0; c0 < channels; c0++)
            {
                for (int i = 0; i < plane; i++)
                {
                    float v = NcnnReadPack4TensorScalarByLinear(_TexIn0Arr, c0 * plane + i, _NormDims, _NormW, _NormH, _NormD, _NormC);
                    sumSquare += v * v;
                }
            }
        }
        else if (_NormAcrossSpatial != 0)
        {
            int baseIndex = channel * plane;
            for (int i1 = 0; i1 < plane; i1++)
            {
                float v1 = NcnnReadPack4TensorScalarByLinear(_TexIn0Arr, baseIndex + i1, _NormDims, _NormW, _NormH, _NormD, _NormC);
                sumSquare += v1 * v1;
            }
        }
        else if (_NormAcrossChannel != 0)
        {
            for (int c1 = 0; c1 < channels; c1++)
            {
                float v2 = NcnnReadPack4TensorScalarByLinear(_TexIn0Arr, c1 * plane + s, _NormDims, _NormW, _NormH, _NormD, _NormC);
                sumSquare += v2 * v2;
            }
        }
        else
        {
            NcnnWriteLane(dst, lane, value);
            continue;
        }

        float scale = NcnnResolveNormScale(channel);
        NcnnWriteLane(dst, lane, value * NcnnComputeNormAlpha(sumSquare) * scale);
    }

    _TexOut0Arr[int3((int)id.x, (int)id.y, (int)id.z)] = dst;
}

void NcnnLrnPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _TexOut0Arr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int pack = (int)id.z;
    float4 src = _TexIn0Arr[int3((int)id.x, (int)id.y, pack)];
    float4 dst = 0.0;
    int channels = max(1, _LrnC);
    int width = max(1, _LrnW);
    int height = max(1, _LrnH);

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int channel = pack * 4 + lane;
        if (channel < 0 || channel >= channels)
            continue;

        float sumSquare = 0.0;
        if (_LrnRegionType == 0)
        {
            int half = _LrnLocalSize / 2;
            int p0 = max(0, channel - half);
            int p1 = min(channels - 1, channel + half);
            for (int p = p0; p <= p1; p++)
            {
                float v = NcnnReadPack4Channel(_TexIn0Arr, (int)id.x, (int)id.y, p);
                sumSquare += v * v;
            }

            float norm = pow(_LrnBias + (_LrnAlpha / max(1, _LrnLocalSize)) * sumSquare, -_LrnBeta);
            NcnnWriteLane(dst, lane, NcnnReadLane(src, lane) * norm);
            continue;
        }

        int pad = _LrnLocalSize / 2;
        for (int ky = 0; ky < _LrnLocalSize; ky++)
        {
            int sy = (int)id.y + ky - pad;
            if (sy < 0 || sy >= height)
                continue;

            for (int kx = 0; kx < _LrnLocalSize; kx++)
            {
                int sx = (int)id.x + kx - pad;
                if (sx < 0 || sx >= width)
                    continue;

                float v2 = NcnnReadPack4Channel(_TexIn0Arr, sx, sy, channel);
                sumSquare += v2 * v2;
            }
        }

        float maxk = max(1, _LrnLocalSize * _LrnLocalSize);
        float normWithin = pow(_LrnBias + (_LrnAlpha / maxk) * sumSquare, -_LrnBeta);
        NcnnWriteLane(dst, lane, NcnnReadLane(src, lane) * normWithin);
    }

    _TexOut0Arr[int3((int)id.x, (int)id.y, pack)] = dst;
}

void NcnnRmsNormPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _TexOut0Arr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int z;
    int pack;
    NcnnDecodePack4TextureSlice(_RmsNormDims, _RmsNormC, (int)id.z, z, pack);
    float4 src = _TexIn0Arr[int3((int)id.x, (int)id.y, (int)id.z)];
    float4 dst = 0.0;

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int channel = _RmsNormDims >= 3 ? pack * 4 + lane : lane;
        if (!NcnnIsPack4TextureLaneValid(_RmsNormDims, _RmsNormC, channel))
            continue;

        int idx = NcnnPack4TensorLinearIndex(_RmsNormDims, _RmsNormW, _RmsNormH, _RmsNormD, _RmsNormC, (int)id.x, (int)id.y, z, channel);
        int segmentBase;
        int segmentSize;
        int innerIndex;
        NcnnDecodeRmsNormSegment(idx, segmentBase, segmentSize, innerIndex);

        float sqsum = 0.0;
        for (int i = 0; i < segmentSize; i++)
        {
            float v = NcnnReadPack4TensorScalarByLinear(_TexIn0Arr, segmentBase + i, _RmsNormDims, _RmsNormW, _RmsNormH, _RmsNormD, _RmsNormC);
            sqsum += v * v;
        }

        float rms = sqsum / max(1, segmentSize);
        float value = NcnnReadLane(src, lane) * rsqrt(max(rms + _RmsNormEps, 1e-20));
        if (_RmsNormAffine != 0 && _RmsNormAffineSize > 0)
            value *= _RmsNormGamma[clamp(innerIndex, 0, _RmsNormAffineSize - 1)];
        NcnnWriteLane(dst, lane, value);
    }

    _TexOut0Arr[int3((int)id.x, (int)id.y, (int)id.z)] = dst;
}

void NcnnRotaryEmbedPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _TexOut0Arr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int embedDim = max(1, _RotaryEmbedDim);
    int seqLen = max(1, _RotarySeqLen);
    int numHeads = max(1, _RotaryNumHeads);
    int halfDim = max(1, embedDim / 2);
    int embed = (int)id.x;
    int seq = (int)id.y;
    int pack = (int)id.z;
    float4 dst = 0.0;

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int head = pack * 4 + lane;
        if (embed >= embedDim || seq >= seqLen || head >= numHeads)
            continue;

        int pair0;
        int pair1;
        int cacheIndex;
        bool first;
        if (_RotaryInterleaved != 0)
        {
            int j = embed / 2;
            pair0 = j * 2;
            pair1 = pair0 + 1;
            cacheIndex = seq * halfDim + j;
            first = (embed & 1) == 0;
        }
        else
        {
            int j2 = embed < halfDim ? embed : embed - halfDim;
            pair0 = j2;
            pair1 = halfDim + j2;
            cacheIndex = seq * halfDim + j2;
            first = embed < halfDim;
        }

        if (pair1 >= embedDim)
        {
            NcnnWriteLane(dst, lane, NcnnReadPack4TensorScalar(_TexIn0Arr, 3, embedDim, seqLen, 1, numHeads, embed, seq, 0, head));
            continue;
        }

        float cosVal = NcnnReadPack4TensorScalarByLinear(_TexIn1Arr, cacheIndex, _RotaryCosDims, _RotaryCosW, _RotaryCosH, _RotaryCosD, _RotaryCosC);
        float sinVal = NcnnReadPack4TensorScalarByLinear(_TexIn2Arr, cacheIndex, _RotarySinDims, _RotarySinW, _RotarySinH, _RotarySinD, _RotarySinC);
        float x0 = NcnnReadPack4TensorScalar(_TexIn0Arr, 3, embedDim, seqLen, 1, numHeads, pair0, seq, 0, head);
        float x1 = NcnnReadPack4TensorScalar(_TexIn0Arr, 3, embedDim, seqLen, 1, numHeads, pair1, seq, 0, head);
        float value = first ? x0 * cosVal - x1 * sinVal : x0 * sinVal + x1 * cosVal;
        NcnnWriteLane(dst, lane, value);
    }

    _TexOut0Arr[int3((int)id.x, (int)id.y, (int)id.z)] = dst;
}
