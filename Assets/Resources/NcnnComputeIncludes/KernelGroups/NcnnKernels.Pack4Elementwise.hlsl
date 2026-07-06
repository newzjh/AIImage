// Auto-generated kernel implementation group: NcnnKernels.Pack4Elementwise.hlsl

void NcnnAddPack4_Impl(uint3 id)
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

void NcnnCopyPack4_Impl(uint3 id)
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

void NcnnConcatPack4CDHW_Impl(uint3 id)
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
            ? NcnnReadPack4ChannelCDHW(_ConcatPack4CDHWAInArr, outX, outY, outZ, outC, _ConcatPack4CDHWAC)
            : NcnnReadPack4ChannelCDHW(_ConcatPack4CDHWBInArr, outX, outY, outZ, outC - _ConcatPack4CDHWAC, _ConcatPack4CDHWBC);
        NcnnWriteLane(o, lane, scalar);
    }

    _ConcatPack4CDHWOutArr[int3(outX, outY, slice)] = o;
}

void NcnnBuildSdInpaintInput9Pack4_Impl(uint3 id)
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

void NcnnFillPack4FromBufferCHW_Impl(uint3 id)
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

void NcnnFillPack4FromBufferCDHW_Impl(uint3 id)
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

void NcnnFillScalarTexture_Impl(uint3 id)
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

void NcnnScalePack4_Impl(uint3 id)
{
    uint w, h, d;
    _ScaleOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 v = _ScaleInArr[int3((int)id.x, (int)id.y, p)];
    _ScaleOutArr[int3((int)id.x, (int)id.y, p)] = v * _ScaleK;
}

void NcnnAddBiasPack4_Impl(uint3 id)
{
    uint w, h, d;
    _BiasOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 v = _BiasInArr[int3((int)id.x, (int)id.y, p)];
    float4 b = _Bias4[p];
    _BiasOutArr[int3((int)id.x, (int)id.y, p)] = v + b;
}

void NcnnBatchNormPack4_Impl(uint3 id)
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

void NcnnLeakyReluPack4_Impl(uint3 id)
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

void NcnnPReluPack4_Impl(uint3 id)
{
    uint w, h, d;
    _LreluOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    float4 v = _LreluInArr[int3((int)id.x, (int)id.y, p)];
    int baseChannel = p * 4;
    float4 slope = float4(
        _PReluSlopePack4Count > 0 ? _PReluSlopePack4Buf[clamp(baseChannel + 0, 0, _PReluSlopePack4Count - 1)] : _LreluSlope,
        _PReluSlopePack4Count > 0 ? _PReluSlopePack4Buf[clamp(baseChannel + 1, 0, _PReluSlopePack4Count - 1)] : _LreluSlope,
        _PReluSlopePack4Count > 0 ? _PReluSlopePack4Buf[clamp(baseChannel + 2, 0, _PReluSlopePack4Count - 1)] : _LreluSlope,
        _PReluSlopePack4Count > 0 ? _PReluSlopePack4Buf[clamp(baseChannel + 3, 0, _PReluSlopePack4Count - 1)] : _LreluSlope
    );
    float4 o = max(v, 0.0) + min(v, 0.0) * slope;
    _LreluOutArr[int3((int)id.x, (int)id.y, p)] = o;
}

void NcnnAddNoiseBroadcastPack4_Impl(uint3 id)
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

void NcnnClipPack4_Impl(uint3 id)
{
    uint w, h, d;
    _ClipOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 v = _ClipInArr[int3((int)id.x, (int)id.y, p)];
    _ClipOutArr[int3((int)id.x, (int)id.y, p)] = clamp(v, _ClipMin, _ClipMax);
}

void NcnnSftPack4_Impl(uint3 id)
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

void NcnnPack4ToRgb01_Impl(uint3 id)
{
    uint w, h;
    _RgbOut.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    int sy = _FlipY != 0 ? (int)(h - 1 - id.y) : (int)id.y;
    float4 v = _RgbInArr[int3((int)id.x, sy, 0)];
    v.xyz = v.xyz * 0.5 + 0.5;
    _RgbOut[int2((int)id.x, (int)id.y)] = float4(v.x, v.y, v.z, 1.0);
}

void NcnnProbeTilePack4_Impl(uint3 id)
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

void NcnnProbeSeams_Impl(uint3 id)
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
