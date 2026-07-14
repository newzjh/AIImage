// Auto-generated kernel implementation group: NcnnKernels.Pack4Conv.hlsl

float4 NcnnReadConvWeight4(int index)
{
    if (_UseFp16ConvWeights == 0)
        return _ConvW4[index];

    uint2 packed = _ConvW4Fp16[index];
    return float4(
        f16tof32(packed.x & 0xffffu),
        f16tof32(packed.x >> 16),
        f16tof32(packed.y & 0xffffu),
        f16tof32(packed.y >> 16));
}

float4 NcnnReadDepthWiseConvWeight4(int index)
{
    if (_UseFp16DepthWiseWeights == 0)
        return _DwConvW4[index];

    uint2 packed = _DwConvW4Fp16[index];
    return float4(
        f16tof32(packed.x & 0xffffu),
        f16tof32(packed.x >> 16),
        f16tof32(packed.y & 0xffffu),
        f16tof32(packed.y >> 16));
}

float4 NcnnMaskConvOutputTail(float4 value, int outputPack)
{
    int firstChannel = outputPack * 4;
    if (firstChannel + 0 >= _OutC) value.x = 0.0;
    if (firstChannel + 1 >= _OutC) value.y = 0.0;
    if (firstChannel + 2 >= _OutC) value.z = 0.0;
    if (firstChannel + 3 >= _OutC) value.w = 0.0;
    return value;
}

void NcnnPackRgbToPack4_Impl(uint3 id)
{
    uint w, h, d;
    _NcnnOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    uint tw, th;
    _NcnnIn.GetDimensions(tw, th);
    int sx = (int) (id.x + _OffsetX) * _ScaleX;
    int sy = (int) (id.y + _OffsetY) * _ScaleY;
    sx = clamp(sx, 0, (int)tw - 1);
    sy = clamp(sy, 0, (int)th - 1);
    if (_FlipY != 0)
        sy = (int)th - 1 - sy;
    float4 v = _NcnnIn[int2(sx, sy)];
    _NcnnOutArr[int3((int)id.x, (int)id.y, 0)] = float4(v.x, v.y, v.z, 0.0);
}

void NcnnUnpackPack4ToRgb_Impl(uint3 id)
{
    uint w, h;
    _NcnnOut.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    float4 v = _NcnnInArr[int3((int)id.x, (int)id.y, 0)];
    _NcnnOut[int2((int)id.x, (int)id.y)] = float4(v.x, v.y, v.z, 1.0);
}

void NcnnConv3x3Pack4_Impl(uint3 id, uint3 groupId, uint3 groupThreadId)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    int ox = (int)id.x * 2;
    int oy = (int)id.y * 2;
    int op = (int)id.z * 2;
    bool active = ox < (int)ow && oy < (int)oh && op >= 0 && op < _OutPacks;
    int op1 = op + 1;
    bool hasOp1 = op1 < _OutPacks;

    float4 bias0 = active ? _ConvB4[op] : 0.0;
    float4 bias1 = hasOp1 ? _ConvB4[op1] : 0.0;
    float4 sum00 = bias0;
    float4 sum01 = bias0;
    float4 sum10 = bias0;
    float4 sum11 = bias0;
    float4 sum20 = bias1;
    float4 sum21 = bias1;
    float4 sum30 = bias1;
    float4 sum31 = bias1;

    uint iw, ih, idd;
    _ConvInArr.GetDimensions(iw, ih, idd);
    int groupOx = (int)groupId.x * 16;
    int groupOy = (int)groupId.y * 16;
    int groupSx = groupOx - _Pad;
    int groupSy = groupOy - _Pad;
    int lane = (int)groupThreadId.y * 8 + (int)groupThreadId.x;
    int localOx = (int)groupThreadId.x * 2;
    int localOy = (int)groupThreadId.y * 2;

    bool hasOp0 = op >= 0 && op < _OutPacks;

    for (int ip0 = 0; ip0 < _InPacks; ip0++)
    {
        for (int li = lane; li < 18 * 18; li += 64)
        {
            int tx = li % 18;
            int ty = li / 18;
            int sx = groupSx + tx;
            int sy = groupSy + ty;
            float4 v = 0.0;
            if (sx >= 0 && sy >= 0 && sx < (int)iw && sy < (int)ih)
                v = _ConvInArr[int3(sx, sy, ip0)];
            _ConvTile[li] = v;
        }

        for (int wi = lane; wi < 9 * 4; wi += 64)
        {
            _ConvWeight0[wi] = hasOp0 ? _ConvW4[((op * _InPacks + ip0) * 9) * 4 + wi] : 0.0;
            _ConvWeight1[wi] = hasOp1 ? _ConvW4[((op1 * _InPacks + ip0) * 9) * 4 + wi] : 0.0;
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (int ky = 0; ky < 3; ky++)
        {
            [unroll]
            for (int kx = 0; kx < 3; kx++)
            {
                float4 v00 = _ConvTile[(localOy + ky) * 18 + (localOx + kx)];
                float4 v01 = _ConvTile[(localOy + ky) * 18 + (localOx + kx + 1)];
                float4 v10 = _ConvTile[(localOy + ky + 1) * 18 + (localOx + kx)];
                float4 v11 = _ConvTile[(localOy + ky + 1) * 18 + (localOx + kx + 1)];

                int k = ky * 3 + kx;
                int wbase = k * 4;
                float4 w00 = _ConvWeight0[wbase + 0];
                float4 w01 = _ConvWeight0[wbase + 1];
                float4 w02 = _ConvWeight0[wbase + 2];
                float4 w03 = _ConvWeight0[wbase + 3];

                sum00.x += dot(v00, w00);
                sum00.y += dot(v00, w01);
                sum00.z += dot(v00, w02);
                sum00.w += dot(v00, w03);

                sum01.x += dot(v01, w00);
                sum01.y += dot(v01, w01);
                sum01.z += dot(v01, w02);
                sum01.w += dot(v01, w03);

                sum10.x += dot(v10, w00);
                sum10.y += dot(v10, w01);
                sum10.z += dot(v10, w02);
                sum10.w += dot(v10, w03);

                sum11.x += dot(v11, w00);
                sum11.y += dot(v11, w01);
                sum11.z += dot(v11, w02);
                sum11.w += dot(v11, w03);

                if (hasOp1)
                {
                    float4 w10 = _ConvWeight1[wbase + 0];
                    float4 w11 = _ConvWeight1[wbase + 1];
                    float4 w12 = _ConvWeight1[wbase + 2];
                    float4 w13 = _ConvWeight1[wbase + 3];

                    sum20.x += dot(v00, w10);
                    sum20.y += dot(v00, w11);
                    sum20.z += dot(v00, w12);
                    sum20.w += dot(v00, w13);

                    sum21.x += dot(v01, w10);
                    sum21.y += dot(v01, w11);
                    sum21.z += dot(v01, w12);
                    sum21.w += dot(v01, w13);

                    sum30.x += dot(v10, w10);
                    sum30.y += dot(v10, w11);
                    sum30.z += dot(v10, w12);
                    sum30.w += dot(v10, w13);

                    sum31.x += dot(v11, w10);
                    sum31.y += dot(v11, w11);
                    sum31.z += dot(v11, w12);
                    sum31.w += dot(v11, w13);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    sum00 = NcnnApplyActivation(sum00);
    sum01 = NcnnApplyActivation(sum01);
    sum10 = NcnnApplyActivation(sum10);
    sum11 = NcnnApplyActivation(sum11);
    if (hasOp1)
    {
        sum20 = NcnnApplyActivation(sum20);
        sum21 = NcnnApplyActivation(sum21);
        sum30 = NcnnApplyActivation(sum30);
        sum31 = NcnnApplyActivation(sum31);
    }

    if (active)
    {
        _ConvOutArr[int3(ox, oy, op)] = sum00;
        if (ox + 1 < (int)ow) _ConvOutArr[int3(ox + 1, oy, op)] = sum01;
        if (oy + 1 < (int)oh) _ConvOutArr[int3(ox, oy + 1, op)] = sum10;
        if (ox + 1 < (int)ow && oy + 1 < (int)oh) _ConvOutArr[int3(ox + 1, oy + 1, op)] = sum11;
        if (hasOp1)
        {
            _ConvOutArr[int3(ox, oy, op1)] = sum20;
            if (ox + 1 < (int)ow) _ConvOutArr[int3(ox + 1, oy, op1)] = sum21;
            if (oy + 1 < (int)oh) _ConvOutArr[int3(ox, oy + 1, op1)] = sum30;
            if (ox + 1 < (int)ow && oy + 1 < (int)oh) _ConvOutArr[int3(ox + 1, oy + 1, op1)] = sum31;
        }
    }
}

void NcnnConvPack4General_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    int ox = (int)id.x * 2;
    int oy = (int)id.y * 2;
    int op = (int)id.z * 2;
    if (ox >= (int)ow || oy >= (int)oh || op < 0 || op >= _OutPacks)
        return;

    int op1 = op + 1;
    bool hasOp1 = op1 < _OutPacks;
    bool hasX1 = (ox + 1) < (int)ow;
    bool hasY1 = (oy + 1) < (int)oh;
    int kernelArea = _KernelWVar * _KernelHVar;
    float4 bias0 = _ConvB4[op];
    float4 bias1 = hasOp1 ? _ConvB4[op1] : 0.0;
    float4 sum00 = bias0;
    float4 sum01 = bias0;
    float4 sum10 = bias0;
    float4 sum11 = bias0;
    float4 sum20 = bias1;
    float4 sum21 = bias1;
    float4 sum30 = bias1;
    float4 sum31 = bias1;

    for (int ip0 = 0; ip0 < _InPacks; ip0++)
    {
        int weightBase0 = ((op * _InPacks + ip0) * kernelArea) * 4;
        int weightBase1 = ((op1 * _InPacks + ip0) * kernelArea) * 4;

        for (int ky = 0; ky < _KernelHVar; ky++)
        {
            int iy0 = oy * _StrideHVar - _PadTopVar + ky * _DilationHVar;
            int iy1 = iy0 + _StrideHVar;
            bool validY0 = iy0 >= 0 && iy0 < _InH;
            bool validY1 = hasY1 && iy1 >= 0 && iy1 < _InH;
            if (!validY0 && !validY1)
                continue;

            for (int kx = 0; kx < _KernelWVar; kx++)
            {
                int ix0 = ox * _StrideWVar - _PadLeftVar + kx * _DilationWVar;
                int ix1 = ix0 + _StrideWVar;
                bool validX0 = ix0 >= 0 && ix0 < _InW;
                bool validX1 = hasX1 && ix1 >= 0 && ix1 < _InW;
                if (!validX0 && !validX1)
                    continue;

                float4 v00 = (validX0 && validY0) ? _ConvInArr[int3(ix0, iy0, ip0)] : 0.0;
                float4 v01 = (validX1 && validY0) ? _ConvInArr[int3(ix1, iy0, ip0)] : 0.0;
                float4 v10 = (validX0 && validY1) ? _ConvInArr[int3(ix0, iy1, ip0)] : 0.0;
                float4 v11 = (validX1 && validY1) ? _ConvInArr[int3(ix1, iy1, ip0)] : 0.0;

                int kernelIndex = ky * _KernelWVar + kx;
                int baseIndex0 = weightBase0 + kernelIndex * 4;
                float4 w00 = NcnnReadConvWeight4(baseIndex0 + 0);
                float4 w01 = NcnnReadConvWeight4(baseIndex0 + 1);
                float4 w02 = NcnnReadConvWeight4(baseIndex0 + 2);
                float4 w03 = NcnnReadConvWeight4(baseIndex0 + 3);
                sum00.x += dot(v00, w00);
                sum00.y += dot(v00, w01);
                sum00.z += dot(v00, w02);
                sum00.w += dot(v00, w03);
                sum01.x += dot(v01, w00);
                sum01.y += dot(v01, w01);
                sum01.z += dot(v01, w02);
                sum01.w += dot(v01, w03);
                sum10.x += dot(v10, w00);
                sum10.y += dot(v10, w01);
                sum10.z += dot(v10, w02);
                sum10.w += dot(v10, w03);
                sum11.x += dot(v11, w00);
                sum11.y += dot(v11, w01);
                sum11.z += dot(v11, w02);
                sum11.w += dot(v11, w03);

                if (hasOp1)
                {
                    int baseIndex1 = weightBase1 + kernelIndex * 4;
                    float4 w10 = NcnnReadConvWeight4(baseIndex1 + 0);
                    float4 w11 = NcnnReadConvWeight4(baseIndex1 + 1);
                    float4 w12 = NcnnReadConvWeight4(baseIndex1 + 2);
                    float4 w13 = NcnnReadConvWeight4(baseIndex1 + 3);
                    sum20.x += dot(v00, w10);
                    sum20.y += dot(v00, w11);
                    sum20.z += dot(v00, w12);
                    sum20.w += dot(v00, w13);
                    sum21.x += dot(v01, w10);
                    sum21.y += dot(v01, w11);
                    sum21.z += dot(v01, w12);
                    sum21.w += dot(v01, w13);
                    sum30.x += dot(v10, w10);
                    sum30.y += dot(v10, w11);
                    sum30.z += dot(v10, w12);
                    sum30.w += dot(v10, w13);
                    sum31.x += dot(v11, w10);
                    sum31.y += dot(v11, w11);
                    sum31.z += dot(v11, w12);
                    sum31.w += dot(v11, w13);
                }
            }
        }
    }

    sum00 = NcnnApplyActivation(sum00);
    if (hasX1) sum01 = NcnnApplyActivation(sum01);
    if (hasY1) sum10 = NcnnApplyActivation(sum10);
    if (hasX1 && hasY1) sum11 = NcnnApplyActivation(sum11);
    sum00 = NcnnMaskConvOutputTail(sum00, op);
    if (hasX1) sum01 = NcnnMaskConvOutputTail(sum01, op);
    if (hasY1) sum10 = NcnnMaskConvOutputTail(sum10, op);
    if (hasX1 && hasY1) sum11 = NcnnMaskConvOutputTail(sum11, op);

    _ConvOutArr[int3(ox, oy, op)] = sum00;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, op)] = sum01;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, op)] = sum10;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, op)] = sum11;

    if (!hasOp1)
        return;

    sum20 = NcnnApplyActivation(sum20);
    if (hasX1) sum21 = NcnnApplyActivation(sum21);
    if (hasY1) sum30 = NcnnApplyActivation(sum30);
    if (hasX1 && hasY1) sum31 = NcnnApplyActivation(sum31);
    sum20 = NcnnMaskConvOutputTail(sum20, op1);
    if (hasX1) sum21 = NcnnMaskConvOutputTail(sum21, op1);
    if (hasY1) sum30 = NcnnMaskConvOutputTail(sum30, op1);
    if (hasX1 && hasY1) sum31 = NcnnMaskConvOutputTail(sum31, op1);

    _ConvOutArr[int3(ox, oy, op1)] = sum20;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, op1)] = sum21;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, op1)] = sum30;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, op1)] = sum31;
}

void NcnnDeconvolutionPack4General_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int op = (int)id.z;
    if (op < 0 || op >= _OutPacks) return;

    int kernelArea = _KernelWVar * _KernelHVar;
    int borderedX = (int)id.x + _PadLeftVar;
    int borderedY = (int)id.y + _PadTopVar;
    float4 sum = _ConvB4[op];

    for (int ip0 = 0; ip0 < _InPacks; ip0++)
    {
        int weightBase = ((op * _InPacks + ip0) * kernelArea) * 4;

        for (int ky = 0; ky < _KernelHVar; ky++)
        {
            int iyNumerator = borderedY - ky * _DilationHVar;
            if (iyNumerator < 0) continue;
            if ((iyNumerator % _StrideHVar) != 0) continue;
            int iy = iyNumerator / _StrideHVar;
            if (iy < 0 || iy >= _InH) continue;

            for (int kx = 0; kx < _KernelWVar; kx++)
            {
                int ixNumerator = borderedX - kx * _DilationWVar;
                if (ixNumerator < 0) continue;
                if ((ixNumerator % _StrideWVar) != 0) continue;
                int ix = ixNumerator / _StrideWVar;
                if (ix < 0 || ix >= _InW) continue;

                float4 inV = _ConvInArr[int3(ix, iy, ip0)];
                int baseIndex = weightBase + (ky * _KernelWVar + kx) * 4;
                float4 w0 = _ConvW4[baseIndex + 0];
                float4 w1 = _ConvW4[baseIndex + 1];
                float4 w2 = _ConvW4[baseIndex + 2];
                float4 w3 = _ConvW4[baseIndex + 3];
                sum.x += dot(inV, w0);
                sum.y += dot(inV, w1);
                sum.z += dot(inV, w2);
                sum.w += dot(inV, w3);
            }
        }
    }

    sum = NcnnApplyActivation(sum);
    _ConvOutArr[int3((int)id.x, (int)id.y, op)] = sum;
}

// General 2D Pack4 convolution uses immutable scalar ncnn weights. Activations and
// results remain Texture2DArray-backed; scalar reads are necessary because group
// boundaries and channel tails are not required to align to a float4 pack.
void NcnnConv2dGroupPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int op = (int)id.z;
    int group = max(1, _ConvGroup);
    int inchG = _InC / group;
    int outchG = _OutC / group;
    int kernelArea = _KernelWVar * _KernelHVar;
    float4 sum = 0.0;

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int oc = op * 4 + lane;
        if (oc >= _OutC)
            continue;

        int g = min(group - 1, oc / outchG);
        float value = _ConvB[oc];
        int weightBase = (oc * inchG) * kernelArea;
        for (int icLocal = 0; icLocal < inchG; icLocal++)
        {
            int ic = g * inchG + icLocal;
            int kernelBase = weightBase + icLocal * kernelArea;
            for (int ky = 0; ky < _KernelHVar; ky++)
            {
                int iy = (int)id.y * _StrideHVar - _PadTopVar + ky * _DilationHVar;
                if (iy < 0 || iy >= _InH)
                    continue;
                for (int kx = 0; kx < _KernelWVar; kx++)
                {
                    int ix = (int)id.x * _StrideWVar - _PadLeftVar + kx * _DilationWVar;
                    if (ix < 0 || ix >= _InW)
                        continue;
                    value += NcnnReadPack4Channel(_ConvInArr, ix, iy, ic) * _ConvW[kernelBase + ky * _KernelWVar + kx];
                }
            }
        }
        NcnnWriteLane(sum, lane, value);
    }

    sum = NcnnApplyActivation(sum);
    [unroll]
    for (int tailLane = 0; tailLane < 4; tailLane++)
    {
        if (op * 4 + tailLane >= _OutC)
            NcnnWriteLane(sum, tailLane, 0.0);
    }
    _ConvOutArr[int3((int)id.x, (int)id.y, op)] = sum;
}

void NcnnDeconvolution2dGroupPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int op = (int)id.z;
    int group = max(1, _ConvGroup);
    int inchG = _InC / group;
    int outchG = _OutC / group;
    int kernelArea = _KernelWVar * _KernelHVar;
    int borderedX = (int)id.x + _PadLeftVar;
    int borderedY = (int)id.y + _PadTopVar;
    float4 sum = 0.0;

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int oc = op * 4 + lane;
        if (oc >= _OutC)
            continue;

        int g = min(group - 1, oc / outchG);
        float value = _ConvB[oc];
        int weightBase = (oc * inchG) * kernelArea;
        for (int icLocal = 0; icLocal < inchG; icLocal++)
        {
            int ic = g * inchG + icLocal;
            int kernelBase = weightBase + icLocal * kernelArea;
            for (int ky = 0; ky < _KernelHVar; ky++)
            {
                int iyNumerator = borderedY - ky * _DilationHVar;
                if (iyNumerator < 0 || (iyNumerator % _StrideHVar) != 0)
                    continue;
                int iy = iyNumerator / _StrideHVar;
                if (iy < 0 || iy >= _InH)
                    continue;
                for (int kx = 0; kx < _KernelWVar; kx++)
                {
                    int ixNumerator = borderedX - kx * _DilationWVar;
                    if (ixNumerator < 0 || (ixNumerator % _StrideWVar) != 0)
                        continue;
                    int ix = ixNumerator / _StrideWVar;
                    if (ix < 0 || ix >= _InW)
                        continue;
                    value += NcnnReadPack4Channel(_ConvInArr, ix, iy, ic) * _ConvW[kernelBase + ky * _KernelWVar + kx];
                }
            }
        }
        NcnnWriteLane(sum, lane, value);
    }

    sum = NcnnApplyActivation(sum);
    [unroll]
    for (int tailLane = 0; tailLane < 4; tailLane++)
    {
        if (op * 4 + tailLane >= _OutC)
            NcnnWriteLane(sum, tailLane, 0.0);
    }
    _ConvOutArr[int3((int)id.x, (int)id.y, op)] = sum;
}

void NcnnDeconvolutionDepthWisePack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int pack = (int)id.z;
    if (pack < 0 || pack >= _OutPacks) return;

    int group = max(1, _ConvGroup);
    int outch_g = max(1, _OutC / group);
    bool isOneToOneDepthWise = _OutC == group && _InC == group;
    int borderedX = (int)id.x + _PadLeftVar;
    int borderedY = (int)id.y + _PadTopVar;
    float4 sum = _DwConvB4[pack];

    for (int ky = 0; ky < _KernelHVar; ky++)
    {
        int iyNumerator = borderedY - ky * _DilationHVar;
        if (iyNumerator < 0)
            continue;
        if ((iyNumerator % _StrideHVar) != 0)
            continue;
        int iy = iyNumerator / _StrideHVar;
        if (iy < 0 || iy >= _InH)
            continue;

        for (int kx = 0; kx < _KernelWVar; kx++)
        {
            int ixNumerator = borderedX - kx * _DilationWVar;
            if (ixNumerator < 0)
                continue;
            if ((ixNumerator % _StrideWVar) != 0)
                continue;
            int ix = ixNumerator / _StrideWVar;
            if (ix < 0 || ix >= _InW)
                continue;

            float4 w = _DwConvW4[(pack * _KernelHVar + ky) * _KernelWVar + kx];
            if (isOneToOneDepthWise)
            {
                sum += _ConvInArr[int3(ix, iy, pack)] * w;
                continue;
            }

            [unroll]
            for (int lane = 0; lane < 4; lane++)
            {
                int oc = pack * 4 + lane;
                if (oc < 0 || oc >= _OutC)
                    continue;

                int g = min(group - 1, oc / outch_g);
                float value = NcnnReadLane(sum, lane) + NcnnReadPack4Channel(_ConvInArr, ix, iy, g) * NcnnReadLane(w, lane);
                NcnnWriteLane(sum, lane, value);
            }
        }
    }

    sum = NcnnApplyActivation(sum);
    _ConvOutArr[int3((int)id.x, (int)id.y, pack)] = sum;
}

void NcnnConvDepthWisePack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    int ox = (int)id.x * 2;
    int oy = (int)id.y * 2;
    int pack = (int)id.z;
    if (ox >= (int)ow || oy >= (int)oh || pack < 0 || pack >= _OutPacks) return;

    bool hasX1 = (ox + 1) < (int)ow;
    bool hasY1 = (oy + 1) < (int)oh;
    float4 bias = _DwConvB4[pack];
    float4 sum00 = bias;
    float4 sum01 = bias;
    float4 sum10 = bias;
    float4 sum11 = bias;
    int group = max(1, _ConvGroup);
    bool isOneToOneDepthWise = _OutC == group;
    int outch_g = max(1, _OutC / group);

    for (int ky = 0; ky < _KernelHVar; ky++)
    {
        int iy0 = oy * _StrideHVar - _PadTopVar + ky * _DilationHVar;
        int iy1 = iy0 + _StrideHVar;
        bool validY0 = iy0 >= 0 && iy0 < _InH;
        bool validY1 = hasY1 && iy1 >= 0 && iy1 < _InH;
        if (!validY0 && !validY1) continue;

        for (int kx = 0; kx < _KernelWVar; kx++)
        {
            int ix0 = ox * _StrideWVar - _PadLeftVar + kx * _DilationWVar;
            int ix1 = ix0 + _StrideWVar;
            bool validX0 = ix0 >= 0 && ix0 < _InW;
            bool validX1 = hasX1 && ix1 >= 0 && ix1 < _InW;
            if (!validX0 && !validX1) continue;

            float4 w = NcnnReadDepthWiseConvWeight4((pack * _KernelHVar + ky) * _KernelWVar + kx);
            if (isOneToOneDepthWise)
            {
                if (validX0 && validY0) sum00 += _ConvInArr[int3(ix0, iy0, pack)] * w;
                if (validX1 && validY0) sum01 += _ConvInArr[int3(ix1, iy0, pack)] * w;
                if (validX0 && validY1) sum10 += _ConvInArr[int3(ix0, iy1, pack)] * w;
                if (validX1 && validY1) sum11 += _ConvInArr[int3(ix1, iy1, pack)] * w;
                continue;
            }

            [unroll]
            for (int lane = 0; lane < 4; lane++)
            {
                int oc = pack * 4 + lane;
                if (oc < 0 || oc >= _OutC)
                    continue;

                int g = min(group - 1, oc / outch_g);
                float weightValue = NcnnReadLane(w, lane);
                if (validX0 && validY0)
                    NcnnWriteLane(sum00, lane, NcnnReadLane(sum00, lane) + NcnnReadPack4Channel(_ConvInArr, ix0, iy0, g) * weightValue);
                if (validX1 && validY0)
                    NcnnWriteLane(sum01, lane, NcnnReadLane(sum01, lane) + NcnnReadPack4Channel(_ConvInArr, ix1, iy0, g) * weightValue);
                if (validX0 && validY1)
                    NcnnWriteLane(sum10, lane, NcnnReadLane(sum10, lane) + NcnnReadPack4Channel(_ConvInArr, ix0, iy1, g) * weightValue);
                if (validX1 && validY1)
                    NcnnWriteLane(sum11, lane, NcnnReadLane(sum11, lane) + NcnnReadPack4Channel(_ConvInArr, ix1, iy1, g) * weightValue);
            }
        }
    }

    sum00 = NcnnApplyActivation(sum00);
    if (hasX1) sum01 = NcnnApplyActivation(sum01);
    if (hasY1) sum10 = NcnnApplyActivation(sum10);
    if (hasX1 && hasY1) sum11 = NcnnApplyActivation(sum11);

    _ConvOutArr[int3(ox, oy, pack)] = sum00;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, pack)] = sum01;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, pack)] = sum10;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, pack)] = sum11;
}

void NcnnWinograd23TransformInputPack4_Impl(uint3 id)
{
    int gx = (int)id.x;
    int gy = (int)id.y;
    int gz = (int)id.z;
    if (gx >= _WinoBlockX || gy >= _WinoBlockY || gz >= _InPacks)
        return;

    // ncnn winograd23 samples from the padded input tensor.
    // Repro path keeps the source unpadded, so emulate pad=1 here.
    int sx = gx * 2 - 1;
    int sy = gy * 2 - 1;
    int cstep = _WinoInCstep;

    float4 v00 = (sx + 0 >= 0 && sy + 0 >= 0 && sx + 0 < _WinoW && sy + 0 < _WinoH) ? _WinoInArr[int3(sx + 0, sy + 0, gz)] : 0.0;
    float4 v01 = sx + 1 < _WinoW ? _WinoInArr[int3(sx + 1, sy + 0, gz)] : 0.0;
    float4 v02 = sx + 2 < _WinoW ? _WinoInArr[int3(sx + 2, sy + 0, gz)] : 0.0;
    float4 v03 = sx + 3 < _WinoW ? _WinoInArr[int3(sx + 3, sy + 0, gz)] : 0.0;

    float4 v10 = sy + 1 < _WinoH ? _WinoInArr[int3(sx + 0, sy + 1, gz)] : 0.0;
    float4 v11 = sy + 1 < _WinoH && sx + 1 < _WinoW ? _WinoInArr[int3(sx + 1, sy + 1, gz)] : 0.0;
    float4 v12 = sy + 1 < _WinoH && sx + 2 < _WinoW ? _WinoInArr[int3(sx + 2, sy + 1, gz)] : 0.0;
    float4 v13 = sy + 1 < _WinoH && sx + 3 < _WinoW ? _WinoInArr[int3(sx + 3, sy + 1, gz)] : 0.0;

    float4 v20 = sy + 2 < _WinoH ? _WinoInArr[int3(sx + 0, sy + 2, gz)] : 0.0;
    float4 v21 = sy + 2 < _WinoH && sx + 1 < _WinoW ? _WinoInArr[int3(sx + 1, sy + 2, gz)] : 0.0;
    float4 v22 = sy + 2 < _WinoH && sx + 2 < _WinoW ? _WinoInArr[int3(sx + 2, sy + 2, gz)] : 0.0;
    float4 v23 = sy + 2 < _WinoH && sx + 3 < _WinoW ? _WinoInArr[int3(sx + 3, sy + 2, gz)] : 0.0;

    float4 v30 = sy + 3 < _WinoH ? _WinoInArr[int3(sx + 0, sy + 3, gz)] : 0.0;
    float4 v31 = sy + 3 < _WinoH && sx + 1 < _WinoW ? _WinoInArr[int3(sx + 1, sy + 3, gz)] : 0.0;
    float4 v32 = sy + 3 < _WinoH && sx + 2 < _WinoW ? _WinoInArr[int3(sx + 2, sy + 3, gz)] : 0.0;
    float4 v33 = sy + 3 < _WinoH && sx + 3 < _WinoW ? _WinoInArr[int3(sx + 3, sy + 3, gz)] : 0.0;

    float4 m00 = v00 - v02;
    float4 m01 = v10 - v12;
    float4 m02 = v20 - v22;
    float4 m03 = v30 - v32;

    float4 m10 = v02 + v01;
    float4 m11 = v12 + v11;
    float4 m12 = v22 + v21;
    float4 m13 = v32 + v31;

    float4 m20 = v02 - v01;
    float4 m21 = v12 - v11;
    float4 m22 = v22 - v21;
    float4 m23 = v32 - v31;

    float4 m30 = v03 - v01;
    float4 m31 = v13 - v11;
    float4 m32 = v23 - v21;
    float4 m33 = v33 - v31;

    v00 = m00 - m02;
    v10 = m10 - m12;
    v20 = m20 - m22;
    v30 = m30 - m32;

    v01 = m02 + m01;
    v11 = m12 + m11;
    v21 = m22 + m21;
    v31 = m32 + m31;

    v02 = m02 - m01;
    v12 = m12 - m11;
    v22 = m22 - m21;
    v32 = m32 - m31;

    v03 = m03 - m01;
    v13 = m13 - m11;
    v23 = m23 - m21;
    v33 = m33 - m31;

    int vTmOffset = gz * cstep + gy * _WinoBlockX + gx;
    int packC = _InPacks;
    _WinoBottomTm[vTmOffset + 0 * cstep * packC] = v00;
    _WinoBottomTm[vTmOffset + 1 * cstep * packC] = v01;
    _WinoBottomTm[vTmOffset + 2 * cstep * packC] = v02;
    _WinoBottomTm[vTmOffset + 3 * cstep * packC] = v03;
    _WinoBottomTm[vTmOffset + 4 * cstep * packC] = v10;
    _WinoBottomTm[vTmOffset + 5 * cstep * packC] = v11;
    _WinoBottomTm[vTmOffset + 6 * cstep * packC] = v12;
    _WinoBottomTm[vTmOffset + 7 * cstep * packC] = v13;
    _WinoBottomTm[vTmOffset + 8 * cstep * packC] = v20;
    _WinoBottomTm[vTmOffset + 9 * cstep * packC] = v21;
    _WinoBottomTm[vTmOffset + 10 * cstep * packC] = v22;
    _WinoBottomTm[vTmOffset + 11 * cstep * packC] = v23;
    _WinoBottomTm[vTmOffset + 12 * cstep * packC] = v30;
    _WinoBottomTm[vTmOffset + 13 * cstep * packC] = v31;
    _WinoBottomTm[vTmOffset + 14 * cstep * packC] = v32;
    _WinoBottomTm[vTmOffset + 15 * cstep * packC] = v33;
}

void NcnnWinograd23GemmPack4_Impl(uint3 id)
{
    int gx = (int)id.x * 4;
    int gy = (int)id.y;
    int gz = (int)id.z;
    int tiles = _WinoInCstep;
    int outc = _OutPacks;
    if (gx >= tiles || gy >= outc || gz >= 16)
        return;

    float4 sum0 = 0.0;
    float4 sum1 = 0.0;
    float4 sum2 = 0.0;
    float4 sum3 = 0.0;

    // bottom_tm layout is [k][in_pack][tile], matching ncnn's
    // gz * c * cstep + gx base address in convolution_pack4_3x3s1d1_winograd_gemm.
    int vOffset = gz * _InPacks * tiles + gx;
    int wBase = (gz * outc + gy) * _InPacks * 4;

    for (int z = 0; z < _InPacks; z++)
    {
        int wOffset = wBase + z * 4;
        float4 v0 = _WinoBottomTm[vOffset + 0];
        float4 v1 = _WinoBottomTm[vOffset + 1];
        float4 v2 = _WinoBottomTm[vOffset + 2];
        float4 v3 = _WinoBottomTm[vOffset + 3];

        float4 k0 = _WinoWeightTm[wOffset + 0];
        float4 k1 = _WinoWeightTm[wOffset + 1];
        float4 k2 = _WinoWeightTm[wOffset + 2];
        float4 k3 = _WinoWeightTm[wOffset + 3];

        sum0.x += dot(v0, k0); sum0.y += dot(v0, k1); sum0.z += dot(v0, k2); sum0.w += dot(v0, k3);
        sum1.x += dot(v1, k0); sum1.y += dot(v1, k1); sum1.z += dot(v1, k2); sum1.w += dot(v1, k3);
        sum2.x += dot(v2, k0); sum2.y += dot(v2, k1); sum2.z += dot(v2, k2); sum2.w += dot(v2, k3);
        sum3.x += dot(v3, k0); sum3.y += dot(v3, k1); sum3.z += dot(v3, k2); sum3.w += dot(v3, k3);

        vOffset += tiles;
    }

    int base = gz * outc * _WinoOutCstep + gy * _WinoOutCstep + gx;
    _WinoTopTm[base + 0] = sum0;
    if (gx + 1 < tiles) _WinoTopTm[base + 1] = sum1;
    if (gx + 2 < tiles) _WinoTopTm[base + 2] = sum2;
    if (gx + 3 < tiles) _WinoTopTm[base + 3] = sum3;
}

void NcnnWinograd23TransformOutputPack4_Impl(uint3 id)
{
    int gx = (int)id.x;
    int gy = (int)id.y;
    int gz = (int)id.z;
    if (gx >= _WinoBlockX || gy >= _WinoBlockY || gz >= _OutPacks)
        return;

    int vTmOffset = gz * _WinoOutCstep + gy * _WinoBlockX + gx;
    int packC = _OutPacks;

    float4 v00 = _WinoTopTm[vTmOffset + 0 * _WinoOutCstep * packC];
    float4 v01 = _WinoTopTm[vTmOffset + 1 * _WinoOutCstep * packC];
    float4 v02 = _WinoTopTm[vTmOffset + 2 * _WinoOutCstep * packC];
    float4 v03 = _WinoTopTm[vTmOffset + 3 * _WinoOutCstep * packC];
    float4 v10 = _WinoTopTm[vTmOffset + 4 * _WinoOutCstep * packC];
    float4 v11 = _WinoTopTm[vTmOffset + 5 * _WinoOutCstep * packC];
    float4 v12 = _WinoTopTm[vTmOffset + 6 * _WinoOutCstep * packC];
    float4 v13 = _WinoTopTm[vTmOffset + 7 * _WinoOutCstep * packC];
    float4 v20 = _WinoTopTm[vTmOffset + 8 * _WinoOutCstep * packC];
    float4 v21 = _WinoTopTm[vTmOffset + 9 * _WinoOutCstep * packC];
    float4 v22 = _WinoTopTm[vTmOffset + 10 * _WinoOutCstep * packC];
    float4 v23 = _WinoTopTm[vTmOffset + 11 * _WinoOutCstep * packC];
    float4 v30 = _WinoTopTm[vTmOffset + 12 * _WinoOutCstep * packC];
    float4 v31 = _WinoTopTm[vTmOffset + 13 * _WinoOutCstep * packC];
    float4 v32 = _WinoTopTm[vTmOffset + 14 * _WinoOutCstep * packC];
    float4 v33 = _WinoTopTm[vTmOffset + 15 * _WinoOutCstep * packC];

    float4 m00 = v00 + v01 + v02;
    float4 m01 = v10 + v11 + v12;
    float4 m02 = v20 + v21 + v22;
    float4 m03 = v30 + v31 + v32;

    float4 m10 = v01 - v02 + v03;
    float4 m11 = v11 - v12 + v13;
    float4 m12 = v21 - v22 + v23;
    float4 m13 = v31 - v32 + v33;

    if (_WinoBiasTerm != 0)
    {
        float4 biasValue = _WinoBias4[gz];
        v00 = biasValue + m00 + m01 + m02;
        v10 = biasValue + m10 + m11 + m12;
        v01 = biasValue + m01 - m02 + m03;
        v11 = biasValue + m11 - m12 + m13;
    }
    else
    {
        v00 = m00 + m01 + m02;
        v10 = m10 + m11 + m12;
        v01 = m01 - m02 + m03;
        v11 = m11 - m12 + m13;
    }

    v00 = NcnnApplyActivation(v00);
    v01 = NcnnApplyActivation(v01);
    v10 = NcnnApplyActivation(v10);
    v11 = NcnnApplyActivation(v11);

    int x = gx * 2;
    int y = gy * 2;
    _WinoOutArr[int3(x + 0, y + 0, gz)] = v00;
    if (x + 1 < _WinoOutW) _WinoOutArr[int3(x + 1, y + 0, gz)] = v01;
    if (y + 1 < _WinoOutH) _WinoOutArr[int3(x + 0, y + 1, gz)] = v10;
    if (y + 1 < _WinoOutH && x + 1 < _WinoOutW) _WinoOutArr[int3(x + 1, y + 1, gz)] = v11;
}

void NcnnConv1x1Pack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);

    int ox = (int)id.x * 2;
    int oy = (int)id.y * 2;
    int op = (int)id.z * 2;
    if (ox >= (int)ow || oy >= (int)oh || op < 0 || op >= _OutPacks)
        return;

    int op1 = op + 1;
    bool hasX1 = ox + 1 < (int)ow;
    bool hasY1 = oy + 1 < (int)oh;
    bool hasOp1 = op1 < _OutPacks;

    float4 bias0 = _ConvB4[op];
    float4 bias1 = hasOp1 ? _ConvB4[op1] : 0.0;

    float4 sum00 = bias0;
    float4 sum01 = bias0;
    float4 sum10 = bias0;
    float4 sum11 = bias0;

    float4 sum20 = bias1;
    float4 sum21 = bias1;
    float4 sum30 = bias1;
    float4 sum31 = bias1;

    for (int ip0 = 0; ip0 < _InPacks; ip0++)
    {
        float4 v00 = _ConvInArr[int3(ox, oy, ip0)];
        float4 v01 = hasX1 ? _ConvInArr[int3(ox + 1, oy, ip0)] : 0.0;
        float4 v10 = hasY1 ? _ConvInArr[int3(ox, oy + 1, ip0)] : 0.0;
        float4 v11 = (hasX1 && hasY1) ? _ConvInArr[int3(ox + 1, oy + 1, ip0)] : 0.0;

        int base0 = ((op * _InPacks + ip0) * 4);
        float4 w00 = _ConvW4[base0 + 0];
        float4 w01 = _ConvW4[base0 + 1];
        float4 w02 = _ConvW4[base0 + 2];
        float4 w03 = _ConvW4[base0 + 3];

        sum00.x += dot(v00, w00);
        sum00.y += dot(v00, w01);
        sum00.z += dot(v00, w02);
        sum00.w += dot(v00, w03);

        if (hasX1)
        {
            sum01.x += dot(v01, w00);
            sum01.y += dot(v01, w01);
            sum01.z += dot(v01, w02);
            sum01.w += dot(v01, w03);
        }

        if (hasY1)
        {
            sum10.x += dot(v10, w00);
            sum10.y += dot(v10, w01);
            sum10.z += dot(v10, w02);
            sum10.w += dot(v10, w03);
        }

        if (hasX1 && hasY1)
        {
            sum11.x += dot(v11, w00);
            sum11.y += dot(v11, w01);
            sum11.z += dot(v11, w02);
            sum11.w += dot(v11, w03);
        }

        if (!hasOp1)
            continue;

        int base1 = ((op1 * _InPacks + ip0) * 4);
        float4 w10 = _ConvW4[base1 + 0];
        float4 w11 = _ConvW4[base1 + 1];
        float4 w12 = _ConvW4[base1 + 2];
        float4 w13 = _ConvW4[base1 + 3];

        sum20.x += dot(v00, w10);
        sum20.y += dot(v00, w11);
        sum20.z += dot(v00, w12);
        sum20.w += dot(v00, w13);

        if (hasX1)
        {
            sum21.x += dot(v01, w10);
            sum21.y += dot(v01, w11);
            sum21.z += dot(v01, w12);
            sum21.w += dot(v01, w13);
        }

        if (hasY1)
        {
            sum30.x += dot(v10, w10);
            sum30.y += dot(v10, w11);
            sum30.z += dot(v10, w12);
            sum30.w += dot(v10, w13);
        }

        if (hasX1 && hasY1)
        {
            sum31.x += dot(v11, w10);
            sum31.y += dot(v11, w11);
            sum31.z += dot(v11, w12);
            sum31.w += dot(v11, w13);
        }
    }

    sum00 = NcnnApplyActivation(sum00);
    if (hasX1) sum01 = NcnnApplyActivation(sum01);
    if (hasY1) sum10 = NcnnApplyActivation(sum10);
    if (hasX1 && hasY1) sum11 = NcnnApplyActivation(sum11);

    _ConvOutArr[int3(ox, oy, op)] = sum00;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, op)] = sum01;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, op)] = sum10;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, op)] = sum11;

    if (!hasOp1)
        return;

    sum20 = NcnnApplyActivation(sum20);
    if (hasX1) sum21 = NcnnApplyActivation(sum21);
    if (hasY1) sum30 = NcnnApplyActivation(sum30);
    if (hasX1 && hasY1) sum31 = NcnnApplyActivation(sum31);

    _ConvOutArr[int3(ox, oy, op1)] = sum20;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, op1)] = sum21;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, op1)] = sum30;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, op1)] = sum31;
}

void NcnnPackRgbToPack4Gfpgan_Impl(uint3 id)
{
    uint w, h, d;
    _NcnnOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    uint tw, th;
    _NcnnIn.GetDimensions(tw, th);
    int sx = (int) (id.x + _OffsetX) * _ScaleX;
    int sy = (int) (id.y + _OffsetY) * _ScaleY;
    sx = clamp(sx, 0, (int)tw - 1);
    sy = clamp(sy, 0, (int)th - 1);
    if (_FlipY != 0)
        sy = (int)th - 1 - sy;
    float4 v = _NcnnIn[int2(sx, sy)];
    v.xyz = v.xyz * 2.0 - 1.0;
    _NcnnOutArr[int3((int)id.x, (int)id.y, 0)] = float4(v.x, v.y, v.z, 0.0);
}
