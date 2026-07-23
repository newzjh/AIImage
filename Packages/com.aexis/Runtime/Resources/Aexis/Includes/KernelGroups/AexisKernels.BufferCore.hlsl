// Auto-generated kernel implementation group: AexisKernels.BufferCore.hlsl

void AexisPassthrough_Impl(uint3 id)
{
    uint w, h;
    _AexisOut.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    _AexisOut[int2(id.x, id.y)] = _AexisIn[int2(id.x, id.y)];
}

void AexisLeakyRelu_Impl(uint3 id)
{
    uint w, h;
    _AexisOut.GetDimensions(w, h);
    uint idx = _BaseIndex + id.x;
    uint n = w * h;
    if (idx >= n) return;
    uint x = idx % w;
    uint y = idx / w;
    float4 v = _AexisIn[int2(x, y)];
    float slope = 0.2;
    v.xyz = (v.xyz >= 0) ? v.xyz : v.xyz * slope;
    _AexisOut[int2(x, y)] = v;
}

void AexisConv3x3_Impl(uint3 id)
{
    int outW = _OutW;
    int outH = _OutH;
    int ox = (int)id.x;
    int oy = (int)id.y;
    int oc = (int)id.z;
    if (ox < 0 || oy < 0 || oc < 0) return;
    if (ox >= outW || oy >= outH || oc >= _OutC) return;

    float sum = _ConvB[oc];

    for (int ic = 0; ic < _InC; ic++)
    {
        for (int ky = 0; ky < 3; ky++)
        {
            for (int kx = 0; kx < 3; kx++)
            {
                int ix = ox * _Stride + kx - _Pad;
                int iy = oy * _Stride + ky - _Pad;
                if (ix < 0 || iy < 0 || ix >= _InW || iy >= _InH)
                    continue;

                uint inIdx = (uint)(ic * _InW * _InH + iy * _InW + ix);
                uint wIdx = (uint)(((oc * _InC + ic) * 3 + ky) * 3 + kx);
                sum += _ConvIn[inIdx] * _ConvW[wIdx];
            }
        }
    }

    sum = AexisApplyActivationScalar(sum);

    uint outIndex = (uint)((oc * outH + oy) * outW + ox);
    _ConvOut[outIndex] = sum;
}

void AexisConv3dBuf_Impl(uint3 id)
{
    int outChannelDepth = _OutD * _OutC;
    if ((int)id.x >= _OutW || (int)id.y >= _OutH || (int)id.z >= outChannelDepth) return;

    int channelDepth = (int)id.z;
    int oc = channelDepth / max(1, _OutD);
    int oz = channelDepth - oc * max(1, _OutD);
    if (oc < 0 || oc >= _OutC || oz < 0 || oz >= _OutD) return;

    float sum = _ConvB[oc];
    int inPlane = _InW * _InH;
    int inVolume = inPlane * _InD;
    int kernelPlane = _KernelWVar * _KernelHVar;
    int kernelVolume = kernelPlane * _KernelDVar;

    for (int ic = 0; ic < _InC; ic++)
    {
        int inputChannelBase = ic * inVolume;
        int weightChannelBase = (oc * _InC + ic) * kernelVolume;
        for (int kz = 0; kz < _KernelDVar; kz++)
        {
            int sz = oz * _StrideDVar - _PadFrontVar + kz * _DilationDVar;
            if (sz < 0 || sz >= _InD) continue;
            int inputDepthBase = inputChannelBase + sz * inPlane;
            int weightDepthBase = weightChannelBase + kz * kernelPlane;
            for (int ky = 0; ky < _KernelHVar; ky++)
            {
                int sy = (int)id.y * _StrideHVar - _PadTopVar + ky * _DilationHVar;
                if (sy < 0 || sy >= _InH) continue;
                int inputRowBase = inputDepthBase + sy * _InW;
                int weightRowBase = weightDepthBase + ky * _KernelWVar;
                for (int kx = 0; kx < _KernelWVar; kx++)
                {
                    int sx = (int)id.x * _StrideWVar - _PadLeftVar + kx * _DilationWVar;
                    if (sx < 0 || sx >= _InW) continue;
                    sum += _ConvIn[inputRowBase + sx] * _ConvW[weightRowBase + kx];
                }
            }
        }
    }

    int outIndex = (((oc * _OutD) + oz) * _OutH + (int)id.y) * _OutW + (int)id.x;
    _ConvOut[outIndex] = AexisApplyActivationScalar(sum);
}

void AexisConv3dPack4CDHW16x4_Impl(uint3 id)
{
    AexisConv3dPack4CDHWBody(id);
}

void AexisConv3dPack4CDHWTile3x3_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);

    int ox = (int)id.x * 2;
    int oy = (int)id.y * 2;
    int outPairCount = max(1, (_OutPacks + 1) / 2);
    int oz = (int)id.z / outPairCount;
    int op = ((int)id.z - oz * outPairCount) * 2;
    if (ox >= (int)ow || oy >= (int)oh || oz < 0 || oz >= _OutD || op < 0 || op >= _OutPacks)
        return;

    int op1 = op + 1;
    bool hasOp1 = op1 < _OutPacks;
    int globalOp = _OutPackOffset + op;
    int globalOp1 = _OutPackOffset + op1;
    bool hasX1 = (ox + 1) < (int)ow;
    bool hasY1 = (oy + 1) < (int)oh;
    int sxBase = ox - _PadLeftVar;
    int syBase = oy - _PadTopVar;
    int szBase = oz - _PadFrontVar;
    bool interior = hasX1
        && hasY1
        && sxBase >= 0
        && syBase >= 0
        && szBase >= 0
        && (sxBase + 3) < _InW
        && (syBase + 3) < _InH
        && (szBase + 2) < _InD;
    if (!interior)
    {
        AexisConv3dPack4CDHWBody(id);
        return;
    }

    float4 bias0 = _ConvB4[globalOp];
    float4 bias1 = hasOp1 ? _ConvB4[globalOp1] : 0.0;
    float4 sum00 = bias0;
    float4 sum01 = bias0;
    float4 sum10 = bias0;
    float4 sum11 = bias0;
    float4 sum20 = bias1;
    float4 sum21 = bias1;
    float4 sum30 = bias1;
    float4 sum31 = bias1;

    [loop]
    for (int ip = 0; ip < _InPacks; ip++)
    {
        int weightBase0 = ((globalOp * _InPacks + ip) * 27) * 4;
        int weightBase1 = hasOp1 ? ((globalOp1 * _InPacks + ip) * 27) * 4 : 0;

        [unroll]
        for (int kz = 0; kz < 3; kz++)
        {
            int inSlice = (szBase + kz) * _InPacks + ip;
            [unroll]
            for (int ky = 0; ky < 3; ky++)
            {
                int sy = syBase + ky;
                [unroll]
                for (int kx = 0; kx < 3; kx++)
                {
                    int sx = sxBase + kx;
                    float4 v00 = _ConvInArr[int3(sx, sy, inSlice)];
                    float4 v01 = _ConvInArr[int3(sx + 1, sy, inSlice)];
                    float4 v10 = _ConvInArr[int3(sx, sy + 1, inSlice)];
                    float4 v11 = _ConvInArr[int3(sx + 1, sy + 1, inSlice)];

                    int wbase = weightBase0 + (((kz * 9) + (ky * 3) + kx) * 4);
                    float4 w00 = _ConvW4[wbase + 0];
                    float4 w01 = _ConvW4[wbase + 1];
                    float4 w02 = _ConvW4[wbase + 2];
                    float4 w03 = _ConvW4[wbase + 3];

                    AexisAccumulateDot4(sum00, v00, w00, w01, w02, w03);
                    AexisAccumulateDot4(sum01, v01, w00, w01, w02, w03);
                    AexisAccumulateDot4(sum10, v10, w00, w01, w02, w03);
                    AexisAccumulateDot4(sum11, v11, w00, w01, w02, w03);

                    if (hasOp1)
                    {
                        int qbase = weightBase1 + (((kz * 9) + (ky * 3) + kx) * 4);
                        float4 q00 = _ConvW4[qbase + 0];
                        float4 q01 = _ConvW4[qbase + 1];
                        float4 q02 = _ConvW4[qbase + 2];
                        float4 q03 = _ConvW4[qbase + 3];

                        AexisAccumulateDot4(sum20, v00, q00, q01, q02, q03);
                        AexisAccumulateDot4(sum21, v01, q00, q01, q02, q03);
                        AexisAccumulateDot4(sum30, v10, q00, q01, q02, q03);
                        AexisAccumulateDot4(sum31, v11, q00, q01, q02, q03);
                    }
                }
            }
        }
    }

    int outSlice0 = oz * _OutPacks + op;
    sum00 = AexisApplyActivation(sum00);
    sum01 = AexisApplyActivation(sum01);
    sum10 = AexisApplyActivation(sum10);
    sum11 = AexisApplyActivation(sum11);
    _ConvOutArr[int3(ox, oy, outSlice0)] = sum00;
    _ConvOutArr[int3(ox + 1, oy, outSlice0)] = sum01;
    _ConvOutArr[int3(ox, oy + 1, outSlice0)] = sum10;
    _ConvOutArr[int3(ox + 1, oy + 1, outSlice0)] = sum11;

    if (!hasOp1)
        return;

    int outSlice1 = oz * _OutPacks + op1;
    sum20 = AexisApplyActivation(sum20);
    sum21 = AexisApplyActivation(sum21);
    sum30 = AexisApplyActivation(sum30);
    sum31 = AexisApplyActivation(sum31);
    _ConvOutArr[int3(ox, oy, outSlice1)] = sum20;
    _ConvOutArr[int3(ox + 1, oy, outSlice1)] = sum21;
    _ConvOutArr[int3(ox, oy + 1, outSlice1)] = sum30;
    _ConvOutArr[int3(ox + 1, oy + 1, outSlice1)] = sum31;
}

void AexisConv3dPack4CDHWZ2_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    int ox = (int)id.x * 2;
    int oy = (int)id.y * 2;
    int outPairCount = max(1, (_OutPacks + 1) / 2);
    int ozBlock = (int)id.z / outPairCount;
    int oz0 = ozBlock * 2;
    int op = ((int)id.z - ozBlock * outPairCount) * 2;
    if (ox >= (int)ow || oy >= (int)oh || oz0 < 0 || oz0 >= _OutD || op < 0 || op >= _OutPacks) return;

    int oz1 = oz0 + 1;
    int op1 = op + 1;
    int globalOp = _OutPackOffset + op;
    int globalOp1 = _OutPackOffset + op1;
    bool hasZ1 = oz1 < _OutD;
    bool hasOp1 = op1 < _OutPacks;
    bool hasX1 = (ox + 1) < (int)ow;
    bool hasY1 = (oy + 1) < (int)oh;

    float4 bias0 = _ConvB4[globalOp];
    float4 bias1 = hasOp1 ? _ConvB4[globalOp1] : 0.0;

    float4 z0sum00 = bias0;
    float4 z0sum01 = bias0;
    float4 z0sum10 = bias0;
    float4 z0sum11 = bias0;
    float4 z0sum20 = bias1;
    float4 z0sum21 = bias1;
    float4 z0sum30 = bias1;
    float4 z0sum31 = bias1;

    float4 z1sum00 = bias0;
    float4 z1sum01 = bias0;
    float4 z1sum10 = bias0;
    float4 z1sum11 = bias0;
    float4 z1sum20 = bias1;
    float4 z1sum21 = bias1;
    float4 z1sum30 = bias1;
    float4 z1sum31 = bias1;

    [loop]
    for (int ip = 0; ip < _InPacks; ip++)
    {
        int weightBase0 = ((globalOp * _InPacks + ip) * 27) * 4;
        int weightBase1 = ((globalOp1 * _InPacks + ip) * 27) * 4;
        int srcZBase = oz0 - _PadFrontVar;

        [unroll]
        for (int ky = 0; ky < 3; ky++)
        {
            int sy0 = oy - _PadTopVar + ky;
            int sy1 = sy0 + 1;
            bool validY0 = sy0 >= 0 && sy0 < _InH;
            bool validY1 = hasY1 && sy1 >= 0 && sy1 < _InH;
            if (!validY0 && !validY1)
                continue;

            [unroll]
            for (int kx = 0; kx < 3; kx++)
            {
                int sx0 = ox - _PadLeftVar + kx;
                int sx1 = sx0 + 1;
                bool validX0 = sx0 >= 0 && sx0 < _InW;
                bool validX1 = hasX1 && sx1 >= 0 && sx1 < _InW;
                if (!validX0 && !validX1)
                    continue;

                int kernelBase = (ky * 3 + kx) * 4;
                float4 w0_0 = _ConvW4[weightBase0 + kernelBase + 0];
                float4 w0_1 = _ConvW4[weightBase0 + kernelBase + 1];
                float4 w0_2 = _ConvW4[weightBase0 + kernelBase + 2];
                float4 w0_3 = _ConvW4[weightBase0 + kernelBase + 3];
                float4 w1_0 = _ConvW4[weightBase0 + 9 * 4 + kernelBase + 0];
                float4 w1_1 = _ConvW4[weightBase0 + 9 * 4 + kernelBase + 1];
                float4 w1_2 = _ConvW4[weightBase0 + 9 * 4 + kernelBase + 2];
                float4 w1_3 = _ConvW4[weightBase0 + 9 * 4 + kernelBase + 3];
                float4 w2_0 = _ConvW4[weightBase0 + 18 * 4 + kernelBase + 0];
                float4 w2_1 = _ConvW4[weightBase0 + 18 * 4 + kernelBase + 1];
                float4 w2_2 = _ConvW4[weightBase0 + 18 * 4 + kernelBase + 2];
                float4 w2_3 = _ConvW4[weightBase0 + 18 * 4 + kernelBase + 3];

                float4 u0_0 = 0.0;
                float4 u0_1 = 0.0;
                float4 u0_2 = 0.0;
                float4 u0_3 = 0.0;
                float4 u1_0 = 0.0;
                float4 u1_1 = 0.0;
                float4 u1_2 = 0.0;
                float4 u1_3 = 0.0;
                float4 u2_0 = 0.0;
                float4 u2_1 = 0.0;
                float4 u2_2 = 0.0;
                float4 u2_3 = 0.0;
                float4 u3_0 = 0.0;
                float4 u3_1 = 0.0;
                float4 u3_2 = 0.0;
                float4 u3_3 = 0.0;

                int sz0 = srcZBase + 0;
                int sz1 = srcZBase + 1;
                int sz2 = srcZBase + 2;
                int sz3 = srcZBase + 3;

                if (sz0 >= 0 && sz0 < _InD)
                {
                    int slice = sz0 * _InPacks + ip;
                    u0_0 = (validX0 && validY0) ? _ConvInArr[int3(sx0, sy0, slice)] : 0.0;
                    u0_1 = (validX1 && validY0) ? _ConvInArr[int3(sx1, sy0, slice)] : 0.0;
                    u0_2 = (validX0 && validY1) ? _ConvInArr[int3(sx0, sy1, slice)] : 0.0;
                    u0_3 = (validX1 && validY1) ? _ConvInArr[int3(sx1, sy1, slice)] : 0.0;
                }
                if (sz1 >= 0 && sz1 < _InD)
                {
                    int slice = sz1 * _InPacks + ip;
                    u1_0 = (validX0 && validY0) ? _ConvInArr[int3(sx0, sy0, slice)] : 0.0;
                    u1_1 = (validX1 && validY0) ? _ConvInArr[int3(sx1, sy0, slice)] : 0.0;
                    u1_2 = (validX0 && validY1) ? _ConvInArr[int3(sx0, sy1, slice)] : 0.0;
                    u1_3 = (validX1 && validY1) ? _ConvInArr[int3(sx1, sy1, slice)] : 0.0;
                }
                if (sz2 >= 0 && sz2 < _InD)
                {
                    int slice = sz2 * _InPacks + ip;
                    u2_0 = (validX0 && validY0) ? _ConvInArr[int3(sx0, sy0, slice)] : 0.0;
                    u2_1 = (validX1 && validY0) ? _ConvInArr[int3(sx1, sy0, slice)] : 0.0;
                    u2_2 = (validX0 && validY1) ? _ConvInArr[int3(sx0, sy1, slice)] : 0.0;
                    u2_3 = (validX1 && validY1) ? _ConvInArr[int3(sx1, sy1, slice)] : 0.0;
                }
                if (hasZ1 && sz3 >= 0 && sz3 < _InD)
                {
                    int slice = sz3 * _InPacks + ip;
                    u3_0 = (validX0 && validY0) ? _ConvInArr[int3(sx0, sy0, slice)] : 0.0;
                    u3_1 = (validX1 && validY0) ? _ConvInArr[int3(sx1, sy0, slice)] : 0.0;
                    u3_2 = (validX0 && validY1) ? _ConvInArr[int3(sx0, sy1, slice)] : 0.0;
                    u3_3 = (validX1 && validY1) ? _ConvInArr[int3(sx1, sy1, slice)] : 0.0;
                }

                z0sum00 += float4(dot(u0_0, w0_0), dot(u0_0, w0_1), dot(u0_0, w0_2), dot(u0_0, w0_3));
                z0sum00 += float4(dot(u1_0, w1_0), dot(u1_0, w1_1), dot(u1_0, w1_2), dot(u1_0, w1_3));
                z0sum00 += float4(dot(u2_0, w2_0), dot(u2_0, w2_1), dot(u2_0, w2_2), dot(u2_0, w2_3));
                z0sum01 += float4(dot(u0_1, w0_0), dot(u0_1, w0_1), dot(u0_1, w0_2), dot(u0_1, w0_3));
                z0sum01 += float4(dot(u1_1, w1_0), dot(u1_1, w1_1), dot(u1_1, w1_2), dot(u1_1, w1_3));
                z0sum01 += float4(dot(u2_1, w2_0), dot(u2_1, w2_1), dot(u2_1, w2_2), dot(u2_1, w2_3));
                z0sum10 += float4(dot(u0_2, w0_0), dot(u0_2, w0_1), dot(u0_2, w0_2), dot(u0_2, w0_3));
                z0sum10 += float4(dot(u1_2, w1_0), dot(u1_2, w1_1), dot(u1_2, w1_2), dot(u1_2, w1_3));
                z0sum10 += float4(dot(u2_2, w2_0), dot(u2_2, w2_1), dot(u2_2, w2_2), dot(u2_2, w2_3));
                z0sum11 += float4(dot(u0_3, w0_0), dot(u0_3, w0_1), dot(u0_3, w0_2), dot(u0_3, w0_3));
                z0sum11 += float4(dot(u1_3, w1_0), dot(u1_3, w1_1), dot(u1_3, w1_2), dot(u1_3, w1_3));
                z0sum11 += float4(dot(u2_3, w2_0), dot(u2_3, w2_1), dot(u2_3, w2_2), dot(u2_3, w2_3));

                if (hasZ1)
                {
                    z1sum00 += float4(dot(u1_0, w0_0), dot(u1_0, w0_1), dot(u1_0, w0_2), dot(u1_0, w0_3));
                    z1sum00 += float4(dot(u2_0, w1_0), dot(u2_0, w1_1), dot(u2_0, w1_2), dot(u2_0, w1_3));
                    z1sum00 += float4(dot(u3_0, w2_0), dot(u3_0, w2_1), dot(u3_0, w2_2), dot(u3_0, w2_3));
                    z1sum01 += float4(dot(u1_1, w0_0), dot(u1_1, w0_1), dot(u1_1, w0_2), dot(u1_1, w0_3));
                    z1sum01 += float4(dot(u2_1, w1_0), dot(u2_1, w1_1), dot(u2_1, w1_2), dot(u2_1, w1_3));
                    z1sum01 += float4(dot(u3_1, w2_0), dot(u3_1, w2_1), dot(u3_1, w2_2), dot(u3_1, w2_3));
                    z1sum10 += float4(dot(u1_2, w0_0), dot(u1_2, w0_1), dot(u1_2, w0_2), dot(u1_2, w0_3));
                    z1sum10 += float4(dot(u2_2, w1_0), dot(u2_2, w1_1), dot(u2_2, w1_2), dot(u2_2, w1_3));
                    z1sum10 += float4(dot(u3_2, w2_0), dot(u3_2, w2_1), dot(u3_2, w2_2), dot(u3_2, w2_3));
                    z1sum11 += float4(dot(u1_3, w0_0), dot(u1_3, w0_1), dot(u1_3, w0_2), dot(u1_3, w0_3));
                    z1sum11 += float4(dot(u2_3, w1_0), dot(u2_3, w1_1), dot(u2_3, w1_2), dot(u2_3, w1_3));
                    z1sum11 += float4(dot(u3_3, w2_0), dot(u3_3, w2_1), dot(u3_3, w2_2), dot(u3_3, w2_3));
                }

                if (hasOp1)
                {
                    float4 q0_0 = _ConvW4[weightBase1 + kernelBase + 0];
                    float4 q0_1 = _ConvW4[weightBase1 + kernelBase + 1];
                    float4 q0_2 = _ConvW4[weightBase1 + kernelBase + 2];
                    float4 q0_3 = _ConvW4[weightBase1 + kernelBase + 3];
                    float4 q1_0 = _ConvW4[weightBase1 + 9 * 4 + kernelBase + 0];
                    float4 q1_1 = _ConvW4[weightBase1 + 9 * 4 + kernelBase + 1];
                    float4 q1_2 = _ConvW4[weightBase1 + 9 * 4 + kernelBase + 2];
                    float4 q1_3 = _ConvW4[weightBase1 + 9 * 4 + kernelBase + 3];
                    float4 q2_0 = _ConvW4[weightBase1 + 18 * 4 + kernelBase + 0];
                    float4 q2_1 = _ConvW4[weightBase1 + 18 * 4 + kernelBase + 1];
                    float4 q2_2 = _ConvW4[weightBase1 + 18 * 4 + kernelBase + 2];
                    float4 q2_3 = _ConvW4[weightBase1 + 18 * 4 + kernelBase + 3];

                    z0sum20 += float4(dot(u0_0, q0_0), dot(u0_0, q0_1), dot(u0_0, q0_2), dot(u0_0, q0_3));
                    z0sum20 += float4(dot(u1_0, q1_0), dot(u1_0, q1_1), dot(u1_0, q1_2), dot(u1_0, q1_3));
                    z0sum20 += float4(dot(u2_0, q2_0), dot(u2_0, q2_1), dot(u2_0, q2_2), dot(u2_0, q2_3));
                    z0sum21 += float4(dot(u0_1, q0_0), dot(u0_1, q0_1), dot(u0_1, q0_2), dot(u0_1, q0_3));
                    z0sum21 += float4(dot(u1_1, q1_0), dot(u1_1, q1_1), dot(u1_1, q1_2), dot(u1_1, q1_3));
                    z0sum21 += float4(dot(u2_1, q2_0), dot(u2_1, q2_1), dot(u2_1, q2_2), dot(u2_1, q2_3));
                    z0sum30 += float4(dot(u0_2, q0_0), dot(u0_2, q0_1), dot(u0_2, q0_2), dot(u0_2, q0_3));
                    z0sum30 += float4(dot(u1_2, q1_0), dot(u1_2, q1_1), dot(u1_2, q1_2), dot(u1_2, q1_3));
                    z0sum30 += float4(dot(u2_2, q2_0), dot(u2_2, q2_1), dot(u2_2, q2_2), dot(u2_2, q2_3));
                    z0sum31 += float4(dot(u0_3, q0_0), dot(u0_3, q0_1), dot(u0_3, q0_2), dot(u0_3, q0_3));
                    z0sum31 += float4(dot(u1_3, q1_0), dot(u1_3, q1_1), dot(u1_3, q1_2), dot(u1_3, q1_3));
                    z0sum31 += float4(dot(u2_3, q2_0), dot(u2_3, q2_1), dot(u2_3, q2_2), dot(u2_3, q2_3));

                    if (hasZ1)
                    {
                        z1sum20 += float4(dot(u1_0, q0_0), dot(u1_0, q0_1), dot(u1_0, q0_2), dot(u1_0, q0_3));
                        z1sum20 += float4(dot(u2_0, q1_0), dot(u2_0, q1_1), dot(u2_0, q1_2), dot(u2_0, q1_3));
                        z1sum20 += float4(dot(u3_0, q2_0), dot(u3_0, q2_1), dot(u3_0, q2_2), dot(u3_0, q2_3));
                        z1sum21 += float4(dot(u1_1, q0_0), dot(u1_1, q0_1), dot(u1_1, q0_2), dot(u1_1, q0_3));
                        z1sum21 += float4(dot(u2_1, q1_0), dot(u2_1, q1_1), dot(u2_1, q1_2), dot(u2_1, q1_3));
                        z1sum21 += float4(dot(u3_1, q2_0), dot(u3_1, q2_1), dot(u3_1, q2_2), dot(u3_1, q2_3));
                        z1sum30 += float4(dot(u1_2, q0_0), dot(u1_2, q0_1), dot(u1_2, q0_2), dot(u1_2, q0_3));
                        z1sum30 += float4(dot(u2_2, q1_0), dot(u2_2, q1_1), dot(u2_2, q1_2), dot(u2_2, q1_3));
                        z1sum30 += float4(dot(u3_2, q2_0), dot(u3_2, q2_1), dot(u3_2, q2_2), dot(u3_2, q2_3));
                        z1sum31 += float4(dot(u1_3, q0_0), dot(u1_3, q0_1), dot(u1_3, q0_2), dot(u1_3, q0_3));
                        z1sum31 += float4(dot(u2_3, q1_0), dot(u2_3, q1_1), dot(u2_3, q1_2), dot(u2_3, q1_3));
                        z1sum31 += float4(dot(u3_3, q2_0), dot(u3_3, q2_1), dot(u3_3, q2_2), dot(u3_3, q2_3));
                    }
                }
            }
        }
    }

    int outSlice0 = oz0 * _OutPacks + op;
    z0sum00 = AexisApplyActivation(z0sum00);
    if (hasX1) z0sum01 = AexisApplyActivation(z0sum01);
    if (hasY1) z0sum10 = AexisApplyActivation(z0sum10);
    if (hasX1 && hasY1) z0sum11 = AexisApplyActivation(z0sum11);
    _ConvOutArr[int3(ox, oy, outSlice0)] = z0sum00;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, outSlice0)] = z0sum01;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, outSlice0)] = z0sum10;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, outSlice0)] = z0sum11;

    if (hasOp1)
    {
        int outSlice0b = oz0 * _OutPacks + op1;
        z0sum20 = AexisApplyActivation(z0sum20);
        if (hasX1) z0sum21 = AexisApplyActivation(z0sum21);
        if (hasY1) z0sum30 = AexisApplyActivation(z0sum30);
        if (hasX1 && hasY1) z0sum31 = AexisApplyActivation(z0sum31);
        _ConvOutArr[int3(ox, oy, outSlice0b)] = z0sum20;
        if (hasX1) _ConvOutArr[int3(ox + 1, oy, outSlice0b)] = z0sum21;
        if (hasY1) _ConvOutArr[int3(ox, oy + 1, outSlice0b)] = z0sum30;
        if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, outSlice0b)] = z0sum31;
    }

    if (!hasZ1)
        return;

    int outSlice1 = oz1 * _OutPacks + op;
    z1sum00 = AexisApplyActivation(z1sum00);
    if (hasX1) z1sum01 = AexisApplyActivation(z1sum01);
    if (hasY1) z1sum10 = AexisApplyActivation(z1sum10);
    if (hasX1 && hasY1) z1sum11 = AexisApplyActivation(z1sum11);
    _ConvOutArr[int3(ox, oy, outSlice1)] = z1sum00;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, outSlice1)] = z1sum01;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, outSlice1)] = z1sum10;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, outSlice1)] = z1sum11;

    if (!hasOp1)
        return;

    int outSlice1b = oz1 * _OutPacks + op1;
    z1sum20 = AexisApplyActivation(z1sum20);
    if (hasX1) z1sum21 = AexisApplyActivation(z1sum21);
    if (hasY1) z1sum30 = AexisApplyActivation(z1sum30);
    if (hasX1 && hasY1) z1sum31 = AexisApplyActivation(z1sum31);
    _ConvOutArr[int3(ox, oy, outSlice1b)] = z1sum20;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, outSlice1b)] = z1sum21;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, outSlice1b)] = z1sum30;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, outSlice1b)] = z1sum31;
}

void AexisConvDepthWise_Impl(uint3 id)
{
    if ((int)id.x >= _OutW || (int)id.y >= _OutH || (int)id.z >= _OutC) return;

    int group = max(1, _ConvGroup);
    int inch_g = max(1, _InC / group);
    int outch_g = max(1, _OutC / group);
    int kernelArea = _KernelWVar * _KernelHVar;
    int oc = (int)id.z;
    int g = min(group - 1, oc / outch_g);
    int ocLocal = oc - g * outch_g;

    float sum = _ConvB[oc];
    int weightBase = ((g * outch_g + ocLocal) * inch_g) * kernelArea;

    for (int icLocal = 0; icLocal < inch_g; icLocal++)
    {
        int ic = g * inch_g + icLocal;
        int kernelBase = weightBase + icLocal * kernelArea;

        for (int ky = 0; ky < _KernelHVar; ky++)
        {
            int sy = (int)id.y * _StrideHVar - _PadTopVar + ky * _DilationHVar;
            if (sy < 0 || sy >= _InH) continue;

            for (int kx = 0; kx < _KernelWVar; kx++)
            {
                int sx = (int)id.x * _StrideWVar - _PadLeftVar + kx * _DilationWVar;
                if (sx < 0 || sx >= _InW) continue;

                int inputIndex = (ic * _InH + sy) * _InW + sx;
                int weightIndex = kernelBase + ky * _KernelWVar + kx;
                sum += _ConvIn[inputIndex] * _ConvW[weightIndex];
            }
        }
    }

    int outIndex = (oc * _OutH + (int)id.y) * _OutW + (int)id.x;
    _ConvOut[outIndex] = AexisApplyActivationScalar(sum);
}

void AexisDeconvolutionBuf_Impl(uint3 id)
{
    if ((int)id.x >= _OutW || (int)id.y >= _OutH || (int)id.z >= _OutC) return;

    int group = max(1, _ConvGroup);
    int inch_g = max(1, _InC / group);
    int outch_g = max(1, _OutC / group);
    int kernelArea = _KernelWVar * _KernelHVar;
    int oc = (int)id.z;
    int g = min(group - 1, oc / outch_g);
    int ocLocal = oc - g * outch_g;

    int borderedX = (int)id.x + _PadLeftVar;
    int borderedY = (int)id.y + _PadTopVar;
    float sum = _ConvB[oc];
    int weightBase = ((g * outch_g + ocLocal) * inch_g) * kernelArea;

    for (int icLocal = 0; icLocal < inch_g; icLocal++)
    {
        int ic = g * inch_g + icLocal;
        int kernelBase = weightBase + icLocal * kernelArea;

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

                int inputIndex = (ic * _InH + iy) * _InW + ix;
                int weightIndex = kernelBase + ky * _KernelWVar + kx;
                sum += _ConvIn[inputIndex] * _ConvW[weightIndex];
            }
        }
    }

    int outIndex = (oc * _OutH + (int)id.y) * _OutW + (int)id.x;
    _ConvOut[outIndex] = AexisApplyActivationScalar(sum);
}

void AexisDeconvolution3dPack4CDHW_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;

    int outSlice = (int)id.z;
    if (outSlice < 0 || outSlice >= (int)od) return;

    int outPackCount = max(1, _OutPacks);
    int oz = outSlice / outPackCount;
    int op = outSlice - oz * outPackCount;
    if (oz < 0 || oz >= _OutD || op < 0 || op >= _OutPacks) return;

    int borderedX = (int)id.x + _PadLeftVar;
    int borderedY = (int)id.y + _PadTopVar;
    int borderedZ = oz + _PadFrontVar;
    int kernelPlane = _KernelWVar * _KernelHVar;
    int kernelVolume = kernelPlane * _KernelDVar;
    float4 sum = _ConvB4[op];

    for (int ip = 0; ip < _InPacks; ip++)
    {
        for (int kz = 0; kz < _KernelDVar; kz++)
        {
            int izNumerator = borderedZ - kz * _DilationDVar;
            if (izNumerator < 0) continue;
            if ((izNumerator % _StrideDVar) != 0) continue;
            int iz = izNumerator / _StrideDVar;
            if (iz < 0 || iz >= _InD) continue;

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

                    int inSlice = iz * _InPacks + ip;
                    float4 v = _ConvInArr[int3(ix, iy, inSlice)];
                    int kernelIndex = (kz * kernelPlane) + (ky * _KernelWVar) + kx;
                    int weightBase = ((((op * _InPacks) + ip) * kernelVolume) + kernelIndex) * 4;
                    float4 w0 = _ConvW4[weightBase + 0];
                    float4 w1 = _ConvW4[weightBase + 1];
                    float4 w2 = _ConvW4[weightBase + 2];
                    float4 w3 = _ConvW4[weightBase + 3];
                    sum.x += dot(v, w0);
                    sum.y += dot(v, w1);
                    sum.z += dot(v, w2);
                    sum.w += dot(v, w3);
                }
            }
        }
    }

    _ConvOutArr[int3((int)id.x, (int)id.y, outSlice)] = AexisApplyActivation(sum);
}

void AexisDeconvolution3dBuf_Impl(uint3 id)
{
    int outChannelDepth = _OutD * _OutC;
    if ((int)id.x >= _OutW || (int)id.y >= _OutH || (int)id.z >= outChannelDepth) return;

    int channelDepth = (int)id.z;
    int oc = channelDepth / max(1, _OutD);
    int oz = channelDepth - oc * max(1, _OutD);
    if (oc < 0 || oc >= _OutC || oz < 0 || oz >= _OutD) return;

    int borderedX = (int)id.x + _PadLeftVar;
    int borderedY = (int)id.y + _PadTopVar;
    int borderedZ = oz + _PadFrontVar;
    float sum = _ConvB[oc];
    int inPlane = _InW * _InH;
    int inVolume = inPlane * _InD;
    int kernelPlane = _KernelWVar * _KernelHVar;
    int kernelVolume = kernelPlane * _KernelDVar;

    for (int ic = 0; ic < _InC; ic++)
    {
        int inputChannelBase = ic * inVolume;
        int weightChannelBase = (oc * _InC + ic) * kernelVolume;

        for (int kz = 0; kz < _KernelDVar; kz++)
        {
            int izNumerator = borderedZ - kz * _DilationDVar;
            if (izNumerator < 0) continue;
            if ((izNumerator % _StrideDVar) != 0) continue;
            int iz = izNumerator / _StrideDVar;
            if (iz < 0 || iz >= _InD) continue;

            int inputDepthBase = inputChannelBase + iz * inPlane;
            int weightDepthBase = weightChannelBase + kz * kernelPlane;

            for (int ky = 0; ky < _KernelHVar; ky++)
            {
                int iyNumerator = borderedY - ky * _DilationHVar;
                if (iyNumerator < 0) continue;
                if ((iyNumerator % _StrideHVar) != 0) continue;
                int iy = iyNumerator / _StrideHVar;
                if (iy < 0 || iy >= _InH) continue;

                int inputRowBase = inputDepthBase + iy * _InW;
                int weightRowBase = weightDepthBase + ky * _KernelWVar;

                for (int kx = 0; kx < _KernelWVar; kx++)
                {
                    int ixNumerator = borderedX - kx * _DilationWVar;
                    if (ixNumerator < 0) continue;
                    if ((ixNumerator % _StrideWVar) != 0) continue;
                    int ix = ixNumerator / _StrideWVar;
                    if (ix < 0 || ix >= _InW) continue;

                    int inputIndex = inputRowBase + ix;
                    int weightIndex = weightRowBase + kx;
                    sum += _ConvIn[inputIndex] * _ConvW[weightIndex];
                }
            }
        }
    }

    int outIndex = (((oc * _OutD) + oz) * _OutH + (int)id.y) * _OutW + (int)id.x;
    _ConvOut[outIndex] = AexisApplyActivationScalar(sum);
}

void AexisTexToBuf3_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    uint w = (uint)_InW;
    uint h = (uint)_InH;
    uint n = w * h;
    if (idx >= n) return;
    uint x = idx % w;
    uint y = idx / w;
    int sx = (int)x + _OffsetX;
    int sy = (int)y + _OffsetY;
    uint tw, th;
    _AexisIn.GetDimensions(tw, th);
    sx = clamp(sx, 0, (int)tw - 1);
    sy = clamp(sy, 0, (int)th - 1);
    float4 c = _AexisIn[int2(sx, sy)];
    _BufOut[idx] = c.x;
    _BufOut[n + idx] = c.y;
    _BufOut[n * 2 + idx] = c.z;
}

void AexisBufToTex3_Impl(uint3 id)
{
    uint w, h;
    _AexisOut.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    uint idx = id.y * w + id.x;
    uint n = w * h;
    float r = _BufA[idx];
    float g = _BufA[n + idx];
    float b = _BufA[n * 2 + idx];
    _AexisOut[int2(id.x, id.y)] = float4(r, g, b, 1.0);
}

void AexisLeakyReluBuf_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    float v = _BufOut[idx];
    _BufOut[idx] = v >= 0.0 ? v : v * _CoeffA;
}

void AexisAddWeighted_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_Total) return;
    _BufOut[idx] = _BufA[idx] * _CoeffA + _BufB[idx] * _CoeffB;
}

void AexisCopyC_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    if (idx >= (uint)_CopyTotal) return;
    uint wh = (uint)(_InW * _InH);
    uint ic = idx / wh;
    uint rem = idx - ic * wh;
    uint oc = ic + (uint)_ChanOffset;
    uint outIdx = oc * wh + rem;
    _BufOut[outIdx] = _BufA[idx];
}

void AexisInterp2x_Impl(uint3 id)
{
    uint outIndex = _BaseIndex + id.x;
    uint inW = (uint)_InW;
    uint inH = (uint)_InH;
    uint inWH = inW * inH;
    uint outW = inW * 2;
    uint outH = inH * 2;
    uint outWH = outW * outH;
    uint outN = outWH * (uint)_InC;
    if (outIndex >= outN) return;

    uint c = outIndex / outWH;
    uint rem = outIndex - c * outWH;
    uint oy = rem / outW;
    uint ox = rem - oy * outW;

    float fx = ((float)ox + 0.5) * 0.5 - 0.5;
    float fy = ((float)oy + 0.5) * 0.5 - 0.5;
    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float tx = fx - x0;
    float ty = fy - y0;
    x0 = clamp(x0, 0, (int)inW - 1);
    x1 = clamp(x1, 0, (int)inW - 1);
    y0 = clamp(y0, 0, (int)inH - 1);
    y1 = clamp(y1, 0, (int)inH - 1);

    uint baseC = c * inWH;
    float v00 = _BufA[baseC + (uint)(y0 * (int)inW + x0)];
    float v10 = _BufA[baseC + (uint)(y0 * (int)inW + x1)];
    float v01 = _BufA[baseC + (uint)(y1 * (int)inW + x0)];
    float v11 = _BufA[baseC + (uint)(y1 * (int)inW + x1)];
    float vx0 = lerp(v00, v10, tx);
    float vx1 = lerp(v01, v11, tx);
    float v = lerp(vx0, vx1, ty);
    _BufOut[outIndex] = v;
}

void AexisBlitTileToDst_Impl(uint3 id)
{
    if ((int)id.x >= _BlitW || (int)id.y >= _BlitH) return;

    int2 Dst;
    Dst.x = _DstX + (int)id.x;
    Dst.y = _DstY + (int)id.y;

    float2 srcF;
    srcF.x = (float)_TilePadX + ((float)id.x + 0.5f) * (float)_TileCoreW / (float)_BlitW - 0.5f;
    srcF.y = (float)_TilePadY + ((float)id.y + 0.5f) * (float)_TileCoreH / (float)_BlitH - 0.5f;

    int2 src0 = (int2)floor(srcF);
    int2 src1 = src0 + 1;
    float2 t = srcF - (float2)src0;

    int2 srcLimit = int2(_TileW - 1, _TileH - 1);
    src0 = clamp(src0, int2(0, 0), srcLimit);
    src1 = clamp(src1, int2(0, 0), srcLimit);

    float4 v00 = _AexisInArr[int3(src0.x, src0.y, 0)];
    float4 v10 = _AexisInArr[int3(src1.x, src0.y, 0)];
    float4 v01 = _AexisInArr[int3(src0.x, src1.y, 0)];
    float4 v11 = _AexisInArr[int3(src1.x, src1.y, 0)];

    float4 v = lerp(lerp(v00, v10, t.x), lerp(v01, v11, t.x), t.y);
    v.w = 1.0;
    _AexisOut[Dst] = v;
}

void AexisInnerProduct_Impl(uint3 id)
{
    uint o = _BaseIndex + id.x;
    if (o >= (uint)_IPOutFeatures) return;
    float sum = _IPB[o];
    uint wbase = o * (uint)_IPInFeatures;
    for (int i = 0; i < _IPInFeatures; i++)
        sum += _IPW[wbase + (uint)i] * _IPIn[i];
    _IPOut[o] = sum;
}
