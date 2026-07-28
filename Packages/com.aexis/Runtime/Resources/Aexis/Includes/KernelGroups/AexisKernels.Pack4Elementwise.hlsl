// Auto-generated kernel implementation group: AexisKernels.Pack4Elementwise.hlsl

void AexisAddPack4_Impl(uint3 id)
{
    uint w, h, d;
    _AddOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 a = _AddA[int3((int)id.x, (int)id.y, p)];
    float4 b = _AddB[int3((int)id.x, (int)id.y, p)];
    _AddOutArr[int3((int)id.x, (int)id.y, p)] = a * _CoeffA + b * _CoeffB;
}

void AexisCopyPack4_Impl(uint3 id)
{
    uint w, h, d;
    _CopyOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= _CopyPacks) return;
    int inP = _CopyInOffset + p;
    int outP = _CopyOutOffset + p;
    float4 v = _CopyInArr[int3((int)id.x, (int)id.y, inP)];
    _CopyOutArr[int3((int)id.x, (int)id.y, outP)] = v;
}

void AexisConcatPack4CDHW_Impl(uint3 id)
{
    uint w, h, d;
    _ConcatPack4CDHWOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h || id.z >= d)
        return;

    int outX = (int)id.x;
    int outY = (int)id.y;
    int slice = (int)id.z;
    int outPacks = max(1, (_ConcatPack4CDHWOutC + 3) / 4);
    int outZ = slice / outPacks;
    int outPack = slice - outZ * outPacks;
    if (outZ < 0 || outZ >= _ConcatPack4CDHWD)
    {
        _ConcatPack4CDHWOutArr[int3(outX, outY, slice)] = 0.0;
        return;
    }

    float4 o = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int outC = outPack * 4 + lane;
        if (outC >= _ConcatPack4CDHWOutC)
            continue;

        float scalar = outC < _ConcatPack4CDHWAC
            ? AexisReadPack4ChannelCDHW(_ConcatPack4CDHWAInArr, outX, outY, outZ, outC, _ConcatPack4CDHWAC)
            : AexisReadPack4ChannelCDHW(_ConcatPack4CDHWBInArr, outX, outY, outZ, outC - _ConcatPack4CDHWAC, _ConcatPack4CDHWBC);
        AexisWriteLane(o, lane, scalar);
    }

    _ConcatPack4CDHWOutArr[int3(outX, outY, slice)] = o;
}

void AexisConcatSequencePack4CDHW_Impl(uint3 id)
{
    uint w, h, d;
    _ConcatSequenceOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h || id.z >= d)
        return;

    int y = (int)id.y;
    if (y >= _ConcatSequencePack4CDHWAH + _ConcatSequencePack4CDHWBH)
    {
        _ConcatSequenceOutArr[int3(id.x, id.y, id.z)] = 0.0;
        return;
    }
    float4 value = y < _ConcatSequencePack4CDHWAH
        ? _ConcatSequenceAInArr[int3(id.x, id.y, id.z)]
        : _ConcatSequenceBInArr[int3(id.x, y - _ConcatSequencePack4CDHWAH, id.z)];
    _ConcatSequenceOutArr[int3(id.x, id.y, id.z)] = value;
}

// Writes only the newly generated sequence into a capacity-backed cache.
void AexisAppendSequencePack4CDHW_Impl(uint3 id)
{
    uint w, h, d;
    _AppendSequenceInArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h || id.z >= d)
        return;

    uint outW, outH, outD;
    _AppendSequenceOutArr.GetDimensions(outW, outH, outD);
    int destinationY = _AppendSequencePack4CDHWOffset + (int)id.y;
    if (id.x >= outW || destinationY < 0 || destinationY >= (int)outH || id.z >= outD)
        return;

    _AppendSequenceOutArr[int3(id.x, destinationY, id.z)] = _AppendSequenceInArr[int3(id.x, id.y, id.z)];
}

// Stateless coordinate hashing makes the generated stream independent of dispatch
// partitioning. The declared seed and logical channel count are the complete RNG
// contract; Pack4 padding lanes are always zeroed.
uint AexisRandomHash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float AexisRandomUniform(uint coordinate)
{
    return (float)(AexisRandomHash(coordinate) & 0x00ffffffu) * (1.0 / 16777216.0);
}

void AexisDeterministicRandomPack4_Impl(uint3 id)
{
    uint width, height, depth;
    _TexOut0Arr.GetDimensions(width, height, depth);
    if (id.x >= width || id.y >= height || id.z >= depth)
        return;

    int packs = max(1, _RandomPacks);
    int channelPack = (int)id.z % packs;
    float4 source = _TexIn0Arr[int3(id)];
    float4 output = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int channel = channelPack * 4 + lane;
        if (channel >= _RandomChannels)
            continue;
        uint coordinate = (uint)_RandomSeed
            ^ ((uint)id.x * 0x9e3779b9u)
            ^ ((uint)id.y * 0x85ebca6bu)
            ^ ((uint)id.z * 0xc2b2ae35u)
            ^ ((uint)lane * 0x27d4eb2du);
        float randomValue = AexisRandomUniform(coordinate);
        float value = 0.0;
        if (_RandomMode == 0)
            value = _RandomParam0 + randomValue * (_RandomParam1 - _RandomParam0);
        else if (_RandomMode == 1)
        {
            float second = max(1e-7, AexisRandomUniform(coordinate ^ 0xa511e9b3u));
            value = _RandomParam0 + _RandomParam1 * sqrt(-2.0 * log(max(1e-7, randomValue))) * cos(6.28318530718 * second);
        }
        else
            value = randomValue < saturate(source[lane]) ? 1.0 : 0.0;
        AexisWriteLane(output, lane, value);
    }
    _TexOut0Arr[int3(id)] = output;
}

void AexisStaticRandomPack4_Impl(uint3 id)
{
    uint width, height, depth;
    _TexOut0Arr.GetDimensions(width, height, depth);
    if (id.x >= width || id.y >= height || id.z >= depth)
        return;

    int packs = max(1, _RandomPacks);
    int channelPack = (int)id.z % packs;
    float4 output = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; ++lane)
    {
        int channel = channelPack * 4 + lane;
        if (channel >= _RandomChannels)
            continue;
        uint coordinate = (uint)_RandomSeed
            ^ ((uint)id.x * 0x9e3779b9u)
            ^ ((uint)id.y * 0x85ebca6bu)
            ^ ((uint)id.z * 0xc2b2ae35u)
            ^ ((uint)lane * 0x27d4eb2du);
        float randomValue = AexisRandomUniform(coordinate);
        float value = _RandomMode == 0
            ? _RandomParam0 + randomValue * (_RandomParam1 - _RandomParam0)
            : _RandomParam0 + _RandomParam1 * sqrt(-2.0 * log(max(1e-7, randomValue)))
                * cos(6.28318530718 * max(1e-7, AexisRandomUniform(coordinate ^ 0xa511e9b3u)));
        AexisWriteLane(output, lane, value);
    }
    _TexOut0Arr[int3(id)] = output;
}

float AexisMultinomialLogit(int batch, int category)
{
    float4 packed = _MultinomialLogitsArr[int3(0, batch, category / 4)];
    return packed[category & 3];
}

// One invocation owns all four lanes of a sampled-index pack. This is important:
// it makes the output race-free while preserving deterministic seed/coordinate
// semantics independently of the graphics API dispatch partitioning.
void AexisMultinomialPack4_Impl(uint3 id)
{
    int samplePack = (int)id.z;
    int batch = (int)id.y;
    int sampleBase = samplePack * 4;
    if (id.x != 0 || batch < 0 || batch >= _MultinomialBatch || sampleBase >= _MultinomialSamples)
        return;

    float maxLogit = -3.402823466e+38f;
    [loop]
    for (int category = 0; category < _MultinomialClasses; ++category)
    {
        float value = AexisMultinomialLogit(batch, category);
        if (isfinite(value))
            maxLogit = max(maxLogit, value);
    }

    float normalizer = 0.0;
    [loop]
    for (int category = 0; category < _MultinomialClasses; ++category)
    {
        float value = AexisMultinomialLogit(batch, category);
        if (isfinite(value) && maxLogit > -3.4e+38f)
            normalizer += exp(clamp(value - maxLogit, -80.0, 0.0));
    }

    float4 result = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; ++lane)
    {
        int sample = sampleBase + lane;
        if (sample >= _MultinomialSamples)
            continue;

        uint coordinate = (uint)_MultinomialSeed
            ^ ((uint)batch * 0x9e3779b9u)
            ^ ((uint)sample * 0x85ebca6bu)
            ^ ((uint)_MultinomialClasses * 0xc2b2ae35u);
        float target = AexisRandomUniform(coordinate) * normalizer;
        float cumulative = 0.0;
        int selected = 0;
        if (normalizer > 0.0)
        {
            [loop]
            for (int category = 0; category < _MultinomialClasses; ++category)
            {
                float value = AexisMultinomialLogit(batch, category);
                if (isfinite(value))
                    cumulative += exp(clamp(value - maxLogit, -80.0, 0.0));
                if (cumulative > target || category == _MultinomialClasses - 1)
                {
                    selected = category;
                    break;
                }
            }
        }
        AexisWriteLane(result, lane, (float)selected);
    }
    _MultinomialOutputArr[int3(0, batch, samplePack)] = result;
}

void AexisBuildSdInpaintInput9Pack4_Impl(uint3 id)
{
    uint w, h, d;
    _SdInpaintInputOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    int x = (int)id.x;
    int y = (int)id.y;
    float4 outv = 0.0;

    if (p == 0)
    {
        outv = _SdInpaintLatentsArr[int3(x, y, 0)];
    }
    else if (p == 1)
    {
        float4 masked = _SdInpaintMaskedLatentsArr[int3(x, y, 0)];
        float mask = _SdInpaintMaskArr[int3(x, y, 0)].x;
        outv = float4(mask, masked.x, masked.y, masked.z);
    }
    else if (p == 2)
    {
        float4 masked = _SdInpaintMaskedLatentsArr[int3(x, y, 0)];
        outv = float4(masked.w, 0.0, 0.0, 0.0);
    }

    _SdInpaintInputOutArr[int3(x, y, p)] = outv;
}

void AexisFillPack4FromBufferCHW_Impl(uint3 id)
{
    uint w, h, d;
    _FillOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    uint wh = w * h;
    int c0 = p * 4 + 0;
    int c1 = p * 4 + 1;
    int c2 = p * 4 + 2;
    int c3 = p * 4 + 3;
    uint idx0 = (uint)c0 * wh + id.y * w + id.x;
    uint idx1 = (uint)c1 * wh + id.y * w + id.x;
    uint idx2 = (uint)c2 * wh + id.y * w + id.x;
    uint idx3 = (uint)c3 * wh + id.y * w + id.x;
    float x0 = c0 < _FillC ? _FillIn[idx0] : 0.0;
    float x1 = c1 < _FillC ? _FillIn[idx1] : 0.0;
    float x2 = c2 < _FillC ? _FillIn[idx2] : 0.0;
    float x3 = c3 < _FillC ? _FillIn[idx3] : 0.0;
    _FillOutArr[int3((int)id.x, (int)id.y, p)] = float4(x0, x1, x2, x3);
}

void AexisFillPack4FromBufferCDHW_Impl(uint3 id)
{
    uint w, h, d;
    _FillOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int slice = (int)id.z;
    if (slice < 0 || slice >= (int)d) return;

    int packCount = max(1, (_FillC + 3) / 4);
    int z = slice / packCount;
    int pack = slice - z * packCount;
    if (z < 0 || z >= _FillD)
    {
        _FillOutArr[int3((int)id.x, (int)id.y, slice)] = 0.0;
        return;
    }

    uint wh = w * h;
    uint spatial = id.y * w + id.x;
    int c0 = pack * 4 + 0;
    int c1 = pack * 4 + 1;
    int c2 = pack * 4 + 2;
    int c3 = pack * 4 + 3;
    uint idx0 = ((uint)c0 * (uint)_FillD + (uint)z) * wh + spatial;
    uint idx1 = ((uint)c1 * (uint)_FillD + (uint)z) * wh + spatial;
    uint idx2 = ((uint)c2 * (uint)_FillD + (uint)z) * wh + spatial;
    uint idx3 = ((uint)c3 * (uint)_FillD + (uint)z) * wh + spatial;
    float x0 = c0 < _FillC ? _FillIn[idx0] : 0.0;
    float x1 = c1 < _FillC ? _FillIn[idx1] : 0.0;
    float x2 = c2 < _FillC ? _FillIn[idx2] : 0.0;
    float x3 = c3 < _FillC ? _FillIn[idx3] : 0.0;
    _FillOutArr[int3((int)id.x, (int)id.y, slice)] = float4(x0, x1, x2, x3);
}

void AexisFillScalarTexture_Impl(uint3 id)
{
    uint w, h, d;
    _FillScalarOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h || id.z >= d)
        return;

    uint linearIndex = id.x + id.y * w + id.z * w * h;
    float value = 0.0;
    if (linearIndex < (uint)_FillScalarValueCount)
    {
        if (linearIndex == 0) value = _FillScalarValues4.x;
        else if (linearIndex == 1) value = _FillScalarValues4.y;
        else if (linearIndex == 2) value = _FillScalarValues4.z;
        else if (linearIndex == 3) value = _FillScalarValues4.w;
    }

    _FillScalarOutArr[int3((int)id.x, (int)id.y, (int)id.z)] = float4(value, 0.0, 0.0, 0.0);
}

void AexisFillScalarLinearMat_Impl(uint3 id)
{
    uint w, h;
    _LinearOut0.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;

    uint linearIndex = id.x + id.y * w;
    float value = 0.0;
    if (linearIndex < (uint)_FillScalarValueCount)
    {
        if (linearIndex == 0) value = _FillScalarValues4.x;
        else if (linearIndex == 1) value = _FillScalarValues4.y;
        else if (linearIndex == 2) value = _FillScalarValues4.z;
        else if (linearIndex == 3) value = _FillScalarValues4.w;
    }

    _LinearOut0[int2((int)id.x, (int)id.y)] = value;
}

void AexisScalePack4_Impl(uint3 id)
{
    uint w, h, d;
    _ScaleOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 v = _ScaleInArr[int3((int)id.x, (int)id.y, p)];
    _ScaleOutArr[int3((int)id.x, (int)id.y, p)] = v * _ScaleK;
}

void AexisAddBiasPack4_Impl(uint3 id)
{
    uint w, h, d;
    _BiasOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 v = _BiasInArr[int3((int)id.x, (int)id.y, p)];
    float4 b = _Bias4[p % max(1, _BiasPacks)];
    _BiasOutArr[int3((int)id.x, (int)id.y, p)] = v + b;
}

void AexisBatchNormPack4_Impl(uint3 id)
{
    uint w, h, d;
    _BatchNormOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 v = _BatchNormInArr[int3((int)id.x, (int)id.y, p)];
    float4 a = _BatchNormA4[p];
    float4 b = _BatchNormB4[p];
    _BatchNormOutArr[int3((int)id.x, (int)id.y, p)] = v * b + a;
}

void AexisLeakyReluPack4_Impl(uint3 id)
{
    uint w, h, d;
    _LreluOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 v = _LreluInArr[int3((int)id.x, (int)id.y, p)];
    float4 o = max(v, 0.0) + min(v, 0.0) * _LreluSlope;
    _LreluOutArr[int3((int)id.x, (int)id.y, p)] = o;
}

void AexisPReluPack4_Impl(uint3 id)
{
    uint w, h, d;
    _LreluOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    float4 v = _LreluInArr[int3((int)id.x, (int)id.y, p)];
    int baseChannel = (p % max(1, _PReluSlopePack4Packs)) * 4;
    float4 slope = float4(
        _PReluSlopePack4Count > 0 ? _PReluSlopePack4Buf[clamp(baseChannel + 0, 0, _PReluSlopePack4Count - 1)] : _LreluSlope,
        _PReluSlopePack4Count > 0 ? _PReluSlopePack4Buf[clamp(baseChannel + 1, 0, _PReluSlopePack4Count - 1)] : _LreluSlope,
        _PReluSlopePack4Count > 0 ? _PReluSlopePack4Buf[clamp(baseChannel + 2, 0, _PReluSlopePack4Count - 1)] : _LreluSlope,
        _PReluSlopePack4Count > 0 ? _PReluSlopePack4Buf[clamp(baseChannel + 3, 0, _PReluSlopePack4Count - 1)] : _LreluSlope
    );
    float4 o = max(v, 0.0) + min(v, 0.0) * slope;
    _LreluOutArr[int3((int)id.x, (int)id.y, p)] = o;
}

void AexisAddNoiseBroadcastPack4_Impl(uint3 id)
{
    uint w, h, d;
    _NoiseInOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    uint idx = id.y * w + id.x;
    float n = _Noise[idx] * _NoiseWeight;
    float4 v = _NoiseInOutArr[int3((int)id.x, (int)id.y, p)];
    _NoiseInOutArr[int3((int)id.x, (int)id.y, p)] = v + n;
}

void AexisClipPack4_Impl(uint3 id)
{
    uint w, h, d;
    _ClipOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 v = _ClipInArr[int3((int)id.x, (int)id.y, p)];
    _ClipOutArr[int3((int)id.x, (int)id.y, p)] = clamp(v, _ClipMin, _ClipMax);
}

void AexisSftPack4_Impl(uint3 id)
{
    uint w, h, d;
    _SftOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 v = _SftInArr[int3((int)id.x, (int)id.y, p)];
    if (p >= _SftHalfPacks)
    {
        int cp = p - _SftHalfPacks;
        float4 m = _SftCondMulArr[int3((int)id.x, (int)id.y, cp)];
        float4 a = _SftCondAddArr[int3((int)id.x, (int)id.y, cp)];
        v = v * m + a;
    }
    _SftOutArr[int3((int)id.x, (int)id.y, p)] = v;
}

void AexisPack4ToRgb01_Impl(uint3 id)
{
    uint w, h;
    _RgbOut.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    int sy = _FlipY != 0 ? (int)(h - 1 - id.y) : (int)id.y;
    float4 v = _RgbInArr[int3((int)id.x, sy, 0)];
    v.xyz = v.xyz * 0.5 + 0.5;
    _RgbOut[int2((int)id.x, (int)id.y)] = float4(v.x, v.y, v.z, 1.0);
}

void AexisPack4ToRgbScaled_Impl(uint3 id)
{
    uint w, h;
    _RgbOut.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    int sy = _FlipY != 0 ? (int)(h - 1 - id.y) : (int)id.y;
    float4 v = _RgbInArr[int3((int)id.x, sy, 0)];
    v.xyz = saturate(v.xyz * _OutputValueScale + _OutputValueBias);
    _RgbOut[int2((int)id.x, (int)id.y)] = float4(v.x, v.y, v.z, 1.0);
}

void AexisNhwcPack4ToRgbScaled_Impl(uint3 id)
{
    uint w, h;
    _RgbOut.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;

    int logicalY = _FlipY != 0 ? (int)(h - 1 - id.y) : (int)id.y;
    int pack = logicalY >> 2;
    int lane = logicalY & 3;
    int x = (int)id.x;

    float r = AexisReadLane(_RgbInArr[int3(0, x, pack)], lane);
    float g = AexisReadLane(_RgbInArr[int3(1, x, pack)], lane);
    float b = AexisReadLane(_RgbInArr[int3(2, x, pack)], lane);
    _RgbOut[int2((int)id.x, (int)id.y)] = float4(
        saturate(r * _OutputValueScale + _OutputValueBias),
        saturate(g * _OutputValueScale + _OutputValueBias),
        saturate(b * _OutputValueScale + _OutputValueBias),
        1.0);
}

void AexisProbeTilePack4_Impl(uint3 id)
{
    uint iw, ih, idd;
    _ProbeInArr.GetDimensions(iw, ih, idd);
    if (idd == 0) return;
    int pad = max(0, _ProbePad);
    int cw = max(1, _ProbeCoreW);
    int ch = max(1, _ProbeCoreH);
    int x0 = clamp(pad, 0, (int)iw - 1);
    int y0 = clamp(pad, 0, (int)ih - 1);
    int x1 = clamp(pad + cw - 1, 0, (int)iw - 1);
    int y1 = clamp(pad + ch - 1, 0, (int)ih - 1);
    int spanX = max(1, x1 - x0);
    int spanY = max(1, y1 - y0);

    float4 sum = 0.0;
    [unroll]
    for (int uy = 0; uy < 4; uy++)
    {
        int sy = y0 + (int)((float)((uy + 1) * spanY) * 0.2);
        [unroll]
        for (int ux = 0; ux < 4; ux++)
        {
            int sx = x0 + (int)((float)((ux + 1) * spanX) * 0.2);
            sum += _ProbeInArr[int3(sx, sy, 0)];
        }
    }
    _ProbeOut[_ProbeIndex] = sum * (1.0 / 16.0);
}

void AexisProbeSeams_Impl(uint3 id)
{
    int seamsV = max(0, _SeamTilesX - 1);
    int seamsH = max(0, _SeamTilesY - 1);
    int seamCount = seamsV + seamsH;
    int i = (int)id.x;
    if (i < 0 || i >= seamCount) return;

    int samples = max(4, _SeamSamples);
    float sumScore = 0.0;
    int valid = 0;

    if (i < seamsV)
    {
        int seamX = (i + 1) * _SeamStepX;
        if (seamX - 2 < 0 || seamX + 1 >= _SeamW) { _SeamOut4[i] = float4(0.0, 0.0, (float)seamX, 0.0); return; }

        [loop]
        for (int s = 0; s < samples; s++)
        {
            int y = (int)((((float)s + 0.5) / (float)samples) * (float)(_SeamH - 1));
            y = clamp(y, 0, _SeamH - 1);
            float3 cLL = _SeamTex[int2(seamX - 2, y)].xyz;
            float3 cL  = _SeamTex[int2(seamX - 1, y)].xyz;
            float3 cR  = _SeamTex[int2(seamX + 0, y)].xyz;
            float3 cRR = _SeamTex[int2(seamX + 1, y)].xyz;
            float d0 = abs(cL.x - cR.x) + abs(cL.y - cR.y) + abs(cL.z - cR.z);
            float d1 = abs(cLL.x - cL.x) + abs(cLL.y - cL.y) + abs(cLL.z - cL.z);
            float d2 = abs(cR.x - cRR.x) + abs(cR.y - cRR.y) + abs(cR.z - cRR.z);
            float local = 0.5 * (d1 + d2);
            sumScore += max(0.0, d0 - local);
            valid++;
        }

        float score = valid > 0 ? (sumScore / (float)valid) : 0.0;
        _SeamOut4[i] = float4(score, 0.0, (float)seamX, 0.0);
        return;
    }

    int j = i - seamsV;
    int seamY = (j + 1) * _SeamStepY;
    if (seamY - 2 < 0 || seamY + 1 >= _SeamH) { _SeamOut4[i] = float4(0.0, 1.0, (float)seamY, 0.0); return; }

    [loop]
    for (int s = 0; s < samples; s++)
    {
        int x = (int)((((float)s + 0.5) / (float)samples) * (float)(_SeamW - 1));
        x = clamp(x, 0, _SeamW - 1);
        float3 cUU = _SeamTex[int2(x, seamY - 2)].xyz;
        float3 cU  = _SeamTex[int2(x, seamY - 1)].xyz;
        float3 cD  = _SeamTex[int2(x, seamY + 0)].xyz;
        float3 cDD = _SeamTex[int2(x, seamY + 1)].xyz;
        float d0 = abs(cU.x - cD.x) + abs(cU.y - cD.y) + abs(cU.z - cD.z);
        float d1 = abs(cUU.x - cU.x) + abs(cUU.y - cU.y) + abs(cUU.z - cU.z);
        float d2 = abs(cD.x - cDD.x) + abs(cD.y - cDD.y) + abs(cD.z - cDD.z);
        float local = 0.5 * (d1 + d2);
        sumScore += max(0.0, d0 - local);
        valid++;
    }

    float score = valid > 0 ? (sumScore / (float)valid) : 0.0;
    _SeamOut4[i] = float4(score, 1.0, (float)seamY, 0.0);
}
