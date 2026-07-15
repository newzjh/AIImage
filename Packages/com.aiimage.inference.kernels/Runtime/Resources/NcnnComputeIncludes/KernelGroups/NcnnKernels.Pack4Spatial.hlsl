// Auto-generated kernel implementation group: NcnnKernels.Pack4Spatial.hlsl

void NcnnInterpPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _InterpOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)od) return;

    uint iw, ih, idd;
    _InterpInArr.GetDimensions(iw, ih, idd);

    if (iw == 0 || ih == 0)
    {
        _InterpOutArr[int3((int)id.x, (int)id.y, p)] = 0.0;
        return;
    }

    float sxScale = max(_InterpScaleFactorX, 1e-6);
    float syScale = max(_InterpScaleFactorY, 1e-6);
    float fx = ((float)id.x + 0.5) / sxScale - 0.5;
    float fy = ((float)id.y + 0.5) / syScale - 0.5;
    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    float tx = fx - x0;
    float ty = fy - y0;

    int x1;
    if (iw <= 1)
    {
        x0 = 0;
        x1 = 0;
        tx = 0.0;
    }
    else
    {
        if (x0 < 0) { x0 = 0; tx = 0.0; }
        if (x0 >= (int)iw - 1) { x0 = max(0, (int)iw - 2); tx = 1.0; }
        x1 = x0 + 1;
    }

    int y1;
    if (ih <= 1)
    {
        y0 = 0;
        y1 = 0;
        ty = 0.0;
    }
    else
    {
        if (y0 < 0) { y0 = 0; ty = 0.0; }
        if (y0 >= (int)ih - 1) { y0 = max(0, (int)ih - 2); ty = 1.0; }
        y1 = y0 + 1;
    }

    float4 v00 = _InterpInArr[int3(x0, y0, p)];
    float4 v10 = _InterpInArr[int3(x1, y0, p)];
    float4 v01 = _InterpInArr[int3(x0, y1, p)];
    float4 v11 = _InterpInArr[int3(x1, y1, p)];
    float4 vx0 = lerp(v00, v10, tx);
    float4 vx1 = lerp(v01, v11, tx);
    _InterpOutArr[int3((int)id.x, (int)id.y, p)] = lerp(vx0, vx1, ty);
}

void NcnnInterpPack4Nearest_Impl(uint3 id)
{
    uint ow, oh, od;
    _InterpOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)od) return;

    uint iw, ih, idd;
    _InterpInArr.GetDimensions(iw, ih, idd);
    if (iw == 0 || ih == 0)
    {
        _InterpOutArr[int3((int)id.x, (int)id.y, p)] = 0.0;
        return;
    }

    float sxScale = max(_InterpScaleFactorX, 1e-6);
    float syScale = max(_InterpScaleFactorY, 1e-6);
    int sx = min((int)(((float)id.x + 0.5) / sxScale), (int)iw - 1);
    int sy = min((int)(((float)id.y + 0.5) / syScale), (int)ih - 1);
    sx = max(0, sx);
    sy = max(0, sy);
    _InterpOutArr[int3((int)id.x, (int)id.y, p)] = _InterpInArr[int3(sx, sy, p)];
}

void NcnnInterpPack4CDHW_Impl(uint3 id)
{
    uint ow, oh, od;
    _InterpOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;

    int outSlice = (int)id.z;
    if (outSlice < 0 || outSlice >= (int)od) return;

    int outPacks = max(1, _OutPacks);
    int srcPacks = max(1, _InPacks);
    int oz = outSlice / outPacks;
    int op = outSlice - oz * outPacks;
    if (oz < 0 || oz >= _OutD || op < 0 || op >= outPacks) return;

    float scaleX = _InterpScaleFactorX > 0.0 ? _InterpScaleFactorX : ((float)_OutW / max(1.0, (float)_InW));
    float scaleY = _InterpScaleFactorY > 0.0 ? _InterpScaleFactorY : ((float)_OutH / max(1.0, (float)_InH));
    float scaleZ = _InterpScaleFactorZ > 0.0 ? _InterpScaleFactorZ : ((float)_OutD / max(1.0, (float)_InD));

    if (_InterpResizeType == 1)
    {
        int sxn = min((int)((float)id.x / max(scaleX, 1e-6)), max(0, _InW - 1));
        int syn = min((int)((float)id.y / max(scaleY, 1e-6)), max(0, _InH - 1));
        int szn = min((int)((float)oz / max(scaleZ, 1e-6)), max(0, _InD - 1));
        _InterpOutArr[int3((int)id.x, (int)id.y, outSlice)] = NcnnReadInterpPack4CDHW(sxn, syn, szn, op, _InW, _InH, _InD, srcPacks);
        return;
    }

    float fx = _InterpAlignCorners != 0 && _OutW > 1 && _InW > 1
        ? (float)id.x * ((float)(_InW - 1) / (float)(_OutW - 1))
        : ((float)id.x + 0.5) / max(scaleX, 1e-6) - 0.5;
    float fy = _InterpAlignCorners != 0 && _OutH > 1 && _InH > 1
        ? (float)id.y * ((float)(_InH - 1) / (float)(_OutH - 1))
        : ((float)id.y + 0.5) / max(scaleY, 1e-6) - 0.5;
    float fz = _InterpAlignCorners != 0 && _OutD > 1 && _InD > 1
        ? (float)oz * ((float)(_InD - 1) / (float)(_OutD - 1))
        : ((float)oz + 0.5) / max(scaleZ, 1e-6) - 0.5;

    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    int z0 = (int)floor(fz);
    float tx = fx - x0;
    float ty = fy - y0;
    float tz = fz - z0;

    if (_InW <= 1) { x0 = 0; tx = 0.0; }
    else
    {
        if (x0 < 0) { x0 = 0; tx = 0.0; }
        if (x0 >= _InW - 1) { x0 = max(0, _InW - 2); tx = 1.0; }
    }

    if (_InH <= 1) { y0 = 0; ty = 0.0; }
    else
    {
        if (y0 < 0) { y0 = 0; ty = 0.0; }
        if (y0 >= _InH - 1) { y0 = max(0, _InH - 2); ty = 1.0; }
    }

    if (_InD <= 1) { z0 = 0; tz = 0.0; }
    else
    {
        if (z0 < 0) { z0 = 0; tz = 0.0; }
        if (z0 >= _InD - 1) { z0 = max(0, _InD - 2); tz = 1.0; }
    }

    int x1 = min(x0 + 1, max(0, _InW - 1));
    int y1 = min(y0 + 1, max(0, _InH - 1));
    int z1 = min(z0 + 1, max(0, _InD - 1));

    float4 c000 = NcnnReadInterpPack4CDHW(x0, y0, z0, op, _InW, _InH, _InD, srcPacks);
    float4 c100 = NcnnReadInterpPack4CDHW(x1, y0, z0, op, _InW, _InH, _InD, srcPacks);
    float4 c010 = NcnnReadInterpPack4CDHW(x0, y1, z0, op, _InW, _InH, _InD, srcPacks);
    float4 c110 = NcnnReadInterpPack4CDHW(x1, y1, z0, op, _InW, _InH, _InD, srcPacks);
    float4 c001 = NcnnReadInterpPack4CDHW(x0, y0, z1, op, _InW, _InH, _InD, srcPacks);
    float4 c101 = NcnnReadInterpPack4CDHW(x1, y0, z1, op, _InW, _InH, _InD, srcPacks);
    float4 c011 = NcnnReadInterpPack4CDHW(x0, y1, z1, op, _InW, _InH, _InD, srcPacks);
    float4 c111 = NcnnReadInterpPack4CDHW(x1, y1, z1, op, _InW, _InH, _InD, srcPacks);

    float4 c00 = lerp(c000, c100, tx);
    float4 c10 = lerp(c010, c110, tx);
    float4 c01 = lerp(c001, c101, tx);
    float4 c11 = lerp(c011, c111, tx);
    float4 c0 = lerp(c00, c10, ty);
    float4 c1 = lerp(c01, c11, ty);
    _InterpOutArr[int3((int)id.x, (int)id.y, outSlice)] = lerp(c0, c1, tz);
}

void NcnnInterp2xPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _InterpOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)od) return;
    uint iw, ih, idd;
    _InterpInArr.GetDimensions(iw, ih, idd);

    float fx = ((float)id.x + 0.5) * 0.5 - 0.5;
    float fy = ((float)id.y + 0.5) * 0.5 - 0.5;
    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    float tx = fx - x0;
    float ty = fy - y0;
    if (x0 < 0) { x0 = 0; tx = 0.0; }
    if (y0 < 0) { y0 = 0; ty = 0.0; }
    if (x0 >= (int)iw - 1) { x0 = max(0, (int)iw - 2); tx = 1.0; }
    if (y0 >= (int)ih - 1) { y0 = max(0, (int)ih - 2); ty = 1.0; }
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float4 v00 = _InterpInArr[int3(x0, y0, p)];
    float4 v10 = _InterpInArr[int3(x1, y0, p)];
    float4 v01 = _InterpInArr[int3(x0, y1, p)];
    float4 v11 = _InterpInArr[int3(x1, y1, p)];
    float4 vx0 = lerp(v00, v10, tx);
    float4 vx1 = lerp(v01, v11, tx);
    float4 v = lerp(vx0, vx1, ty);
    _InterpOutArr[int3((int)id.x, (int)id.y, p)] = v;
}

void NcnnInterp2xNearestPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _InterpNnOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)od) return;

    uint iw, ih, idd;
    _InterpNnInArr.GetDimensions(iw, ih, idd);
    int sx = min((int)(id.x >> 1), (int)iw - 1);
    int sy = min((int)(id.y >> 1), (int)ih - 1);
    _InterpNnOutArr[int3((int)id.x, (int)id.y, p)] = _InterpNnInArr[int3(sx, sy, p)];
}

void NcnnInterpDown2Pack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _InterpDownOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)od) return;
    uint iw, ih, idd;
    _InterpDownInArr.GetDimensions(iw, ih, idd);

    float fx = ((float)id.x + 0.5) / 0.5 - 0.5;
    float fy = ((float)id.y + 0.5) / 0.5 - 0.5;
    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    float tx = fx - x0;
    float ty = fy - y0;
    if (x0 < 0) { x0 = 0; tx = 0.0; }
    if (y0 < 0) { y0 = 0; ty = 0.0; }
    if (x0 >= (int)iw - 1) { x0 = max(0, (int)iw - 2); tx = 1.0; }
    if (y0 >= (int)ih - 1) { y0 = max(0, (int)ih - 2); ty = 1.0; }
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float4 v00 = _InterpDownInArr[int3(x0, y0, p)];
    float4 v10 = _InterpDownInArr[int3(x1, y0, p)];
    float4 v01 = _InterpDownInArr[int3(x0, y1, p)];
    float4 v11 = _InterpDownInArr[int3(x1, y1, p)];
    float4 vx0 = lerp(v00, v10, tx);
    float4 vx1 = lerp(v01, v11, tx);
    float4 v = lerp(vx0, vx1, ty);
    _InterpDownOutArr[int3((int)id.x, (int)id.y, p)] = v;
}

void NcnnInterpDown2NearestPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _InterpDownNnOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)od) return;

    uint iw, ih, idd;
    _InterpDownNnInArr.GetDimensions(iw, ih, idd);
    int sx = min((int)(id.x * 2), (int)iw - 1);
    int sy = min((int)(id.y * 2), (int)ih - 1);
    _InterpDownNnOutArr[int3((int)id.x, (int)id.y, p)] = _InterpDownNnInArr[int3(sx, sy, p)];
}

void NcnnPaddingPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _PadOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)od) return;

    uint iw, ih, idd;
    _PadInArr.GetDimensions(iw, ih, idd);
    int sx = (int)id.x - _PadLeft;
    int sy = (int)id.y - _PadTop;

    if (_PadType == 0)
    {
        if (sx < 0 || sy < 0 || sx >= (int)iw || sy >= (int)ih || p >= (int)idd)
        {
            _PadOutArr[int3((int)id.x, (int)id.y, p)] = _PadValue4;
            return;
        }
        _PadOutArr[int3((int)id.x, (int)id.y, p)] = _PadInArr[int3(sx, sy, p)];
        return;
    }

    if (sx < 0 || sx >= (int)iw)
    {
        if (_PadType == 1)
            sx = clamp(sx, 0, (int)iw - 1);
        else
            sx = Reflect101Index(sx, (int)iw);
    }
    if (sy < 0 || sy >= (int)ih)
    {
        if (_PadType == 1)
            sy = clamp(sy, 0, (int)ih - 1);
        else
            sy = Reflect101Index(sy, (int)ih);
    }
    int sp = p;
    if (sp >= (int)idd) sp = (int)idd - 1;
    _PadOutArr[int3((int)id.x, (int)id.y, p)] = _PadInArr[int3(sx, sy, sp)];
}

void NcnnPoolingPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _PoolOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)od) return;

    uint iw, ih, idd;
    _PoolInArr.GetDimensions(iw, ih, idd);
    int ox = (int)id.x;
    int oy = (int)id.y;

    int sx0 = ox * _PoolStrideW - _PoolPadLeft;
    int sy0 = oy * _PoolStrideH - _PoolPadTop;
    int sx1 = sx0 + _PoolKernelW;
    int sy1 = sy0 + _PoolKernelH;

    float4 acc;
    if (_PoolType == 0)
        acc = float4(-3.402823466e+38, -3.402823466e+38, -3.402823466e+38, -3.402823466e+38);
    else
        acc = 0.0;

    int count = 0;
    for (int ky = 0; ky < _PoolKernelH; ky++)
    {
        int yy = sy0 + ky;
        if (yy < 0 || yy >= (int)ih) continue;

        for (int kx = 0; kx < _PoolKernelW; kx++)
        {
            int xx = sx0 + kx;
            if (xx < 0 || yy < 0 || xx >= (int)iw || yy >= (int)ih) continue;
            if (p >= (int)idd) continue;
            float4 v = _PoolInArr[int3(xx, yy, p)];
            if (_PoolType == 0)
                acc = max(acc, v);
            else
            {
                acc += v;
                count++;
            }
        }
    }

    if (_PoolType != 0)
    {
        float inv = count > 0 ? (1.0 / (float)count) : 0.0;
        acc *= inv;
    }

    _PoolOutArr[int3(ox, oy, p)] = acc;
}

void NcnnPoolingPack4CDHW_Impl(uint3 id)
{
    uint ow, oh, od;
    _Pool4DOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int ox = (int)id.x;
    int oy = (int)id.y;
    int slice = (int)id.z;
    int packCount = max(1, (_Pool4DOutC + 3) / 4);
    int oz = slice / packCount;
    int outPack = slice - oz * packCount;
    if (oz < 0 || oz >= _Pool4DOutD)
    {
        _Pool4DOutArr[int3(ox, oy, slice)] = 0.0;
        return;
    }

    float4 acc = _Pool4DPoolType == 0
        ? float4(-3.402823466e+38, -3.402823466e+38, -3.402823466e+38, -3.402823466e+38)
        : 0.0;
    int count = 0;

    int sx0;
    int sy0;
    int sz0;
    int sx1;
    int sy1;
    int sz1;

    if (_Pool4DGlobal != 0)
    {
        sx0 = 0; sy0 = 0; sz0 = 0;
        sx1 = _Pool4DInW; sy1 = _Pool4DInH; sz1 = _Pool4DInD;
    }
    else if (_Pool4DAdaptive != 0)
    {
        sx0 = (_Pool4DInW * ox) / max(1, _Pool4DOutW);
        sx1 = (_Pool4DInW * (ox + 1) + max(1, _Pool4DOutW) - 1) / max(1, _Pool4DOutW);
        sy0 = (_Pool4DInH * oy) / max(1, _Pool4DOutH);
        sy1 = (_Pool4DInH * (oy + 1) + max(1, _Pool4DOutH) - 1) / max(1, _Pool4DOutH);
        sz0 = (_Pool4DInD * oz) / max(1, _Pool4DOutD);
        sz1 = (_Pool4DInD * (oz + 1) + max(1, _Pool4DOutD) - 1) / max(1, _Pool4DOutD);
    }
    else
    {
        sx0 = ox * _Pool4DStrideW - _Pool4DPadLeft;
        sy0 = oy * _Pool4DStrideH - _Pool4DPadTop;
        sz0 = oz * _Pool4DStrideD - _Pool4DPadFront;
        sx1 = sx0 + _Pool4DKernelW;
        sy1 = sy0 + _Pool4DKernelH;
        sz1 = sz0 + _Pool4DKernelD;
    }

    if (_Pool4DPoolType == 0)
    {
        int kz0 = max(sz0, 0);
        int kz1 = min(sz1, _Pool4DInD);
        int ky0 = max(sy0, 0);
        int ky1 = min(sy1, _Pool4DInH);
        int kx0 = max(sx0, 0);
        int kx1 = min(sx1, _Pool4DInW);

        for (int kz = kz0; kz < kz1; kz++)
        {
            int inSlice = kz * packCount + outPack;
            for (int ky = ky0; ky < ky1; ky++)
            {
                for (int kx = kx0; kx < kx1; kx++)
                {
                    float4 v = _Pool4DInArr[int3(kx, ky, inSlice)];
                    acc = max(acc, v);
                }
            }
        }
    }
    else
    {
        for (int kz = sz0; kz < sz1; kz++)
        {
            bool validZ = kz >= 0 && kz < _Pool4DInD;
            int inSlice = kz * packCount + outPack;
            for (int ky = sy0; ky < sy1; ky++)
            {
                bool validZY = validZ && ky >= 0 && ky < _Pool4DInH;
                for (int kx = sx0; kx < sx1; kx++)
                {
                    bool valid = validZY && kx >= 0 && kx < _Pool4DInW;
                    if (valid)
                        acc += _Pool4DInArr[int3(kx, ky, inSlice)];

                    if (_Pool4DAdaptive != 0 || _Pool4DIncludePad != 0 || valid)
                        count += 1;
                }
            }
        }

        if (count > 0)
            acc /= (float)count;
        else
            acc = 0.0;
    }

    _Pool4DOutArr[int3(ox, oy, slice)] = acc;
}

void NcnnMaxPoolingIndPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _MaxPoolOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)od) return;

    uint iw, ih, idd;
    _MaxPoolInArr.GetDimensions(iw, ih, idd);
    int ox = (int)id.x;
    int oy = (int)id.y;

    int sx0 = ox * _PoolStrideW - _PoolPadLeft;
    int sy0 = oy * _PoolStrideH - _PoolPadTop;
    float4 bestValue = float4(-3.402823466e+38, -3.402823466e+38, -3.402823466e+38, -3.402823466e+38);
    float4 bestIndex = float4(-1.0, -1.0, -1.0, -1.0);

    for (int ky = 0; ky < _PoolKernelH; ky++)
    {
        int yy = sy0 + ky;
        if (yy < 0 || yy >= (int)ih) continue;

        for (int kx = 0; kx < _PoolKernelW; kx++)
        {
            int xx = sx0 + kx;
            if (xx < 0 || yy < 0 || xx >= (int)iw || yy >= (int)ih) continue;
            if (p >= (int)idd) continue;

            float4 v = _MaxPoolInArr[int3(xx, yy, p)];
            float idxv = (float)(yy * (int)iw + xx);

            if (v.x > bestValue.x) { bestValue.x = v.x; bestIndex.x = idxv; }
            if (v.y > bestValue.y) { bestValue.y = v.y; bestIndex.y = idxv; }
            if (v.z > bestValue.z) { bestValue.z = v.z; bestIndex.z = idxv; }
            if (v.w > bestValue.w) { bestValue.w = v.w; bestIndex.w = idxv; }
        }
    }

    _MaxPoolOutArr[int3(ox, oy, p)] = bestValue;
    _MaxPoolIndicesArr[int3(ox, oy, p)] = bestIndex;
}

void NcnnMaxPoolingIndicesFromValuePack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _MaxPoolIndicesArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)od) return;

    uint iw, ih, idd;
    _MaxPoolInArr.GetDimensions(iw, ih, idd);
    int ox = (int)id.x;
    int oy = (int)id.y;

    int sx0 = ox * _PoolStrideW - _PoolPadLeft;
    int sy0 = oy * _PoolStrideH - _PoolPadTop;
    float4 bestValue = float4(-3.402823466e+38, -3.402823466e+38, -3.402823466e+38, -3.402823466e+38);
    float4 bestIndex = float4(-1.0, -1.0, -1.0, -1.0);

    for (int ky = 0; ky < _PoolKernelH; ky++)
    {
        int yy = sy0 + ky;
        if (yy < 0 || yy >= (int)ih) continue;

        for (int kx = 0; kx < _PoolKernelW; kx++)
        {
            int xx = sx0 + kx;
            if (xx < 0 || yy < 0 || xx >= (int)iw || yy >= (int)ih) continue;
            if (p >= (int)idd) continue;

            float4 v = _MaxPoolInArr[int3(xx, yy, p)];
            float idxv = (float)(yy * (int)iw + xx);
            if (v.x > bestValue.x) { bestValue.x = v.x; bestIndex.x = idxv; }
            if (v.y > bestValue.y) { bestValue.y = v.y; bestIndex.y = idxv; }
            if (v.z > bestValue.z) { bestValue.z = v.z; bestIndex.z = idxv; }
            if (v.w > bestValue.w) { bestValue.w = v.w; bestIndex.w = idxv; }
        }
    }

    _MaxPoolIndicesArr[int3(ox, oy, p)] = bestIndex;
}

void NcnnMaxUnPoolingPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _MaxUnpoolOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)od) return;

    uint iw, ih, idd;
    _MaxUnpoolInArr.GetDimensions(iw, ih, idd);

    int x = (int)id.x;
    int y = (int)id.y;
    int targetIndex = y * (int)ow + x;
    int strideW = max(1, _PoolStrideW);
    int strideH = max(1, _PoolStrideH);
    int oxMin = (int)ceil(((float)(x + _PoolPadLeft - _PoolKernelW + 1)) / (float)strideW);
    int oxMax = (int)floor(((float)(x + _PoolPadLeft)) / (float)strideW);
    int oyMin = (int)ceil(((float)(y + _PoolPadTop - _PoolKernelH + 1)) / (float)strideH);
    int oyMax = (int)floor(((float)(y + _PoolPadTop)) / (float)strideH);

    oxMin = clamp(oxMin, 0, max(0, (int)iw - 1));
    oxMax = clamp(oxMax, 0, max(0, (int)iw - 1));
    oyMin = clamp(oyMin, 0, max(0, (int)ih - 1));
    oyMax = clamp(oyMax, 0, max(0, (int)ih - 1));

    float4 outValue = 0.0;
    int4 hasMatch = 0;

    for (int oy = oyMin; oy <= oyMax; oy++)
    {
        for (int ox = oxMin; ox <= oxMax; ox++)
        {
            float4 pooled = _MaxUnpoolInArr[int3(ox, oy, p)];
            float4 indices = _MaxUnpoolIndicesArr[int3(ox, oy, p)];

            if (abs(indices.x - (float)targetIndex) < 0.5)
            {
                outValue.x = hasMatch.x != 0 ? max(outValue.x, pooled.x) : pooled.x;
                hasMatch.x = 1;
            }
            if (abs(indices.y - (float)targetIndex) < 0.5)
            {
                outValue.y = hasMatch.y != 0 ? max(outValue.y, pooled.y) : pooled.y;
                hasMatch.y = 1;
            }
            if (abs(indices.z - (float)targetIndex) < 0.5)
            {
                outValue.z = hasMatch.z != 0 ? max(outValue.z, pooled.z) : pooled.z;
                hasMatch.z = 1;
            }
            if (abs(indices.w - (float)targetIndex) < 0.5)
            {
                outValue.w = hasMatch.w != 0 ? max(outValue.w, pooled.w) : pooled.w;
                hasMatch.w = 1;
            }
        }
    }

    _MaxUnpoolOutArr[int3(x, y, p)] = outValue;
}

void NcnnSoftmaxChannelPack4_Impl(uint3 id)
{
    uint w, h, d;
    _SoftmaxOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    float4 v = _SoftmaxInArr[int3((int)id.x, (int)id.y, p)];

    float maxv = -3.402823466e+38;
    for (int ip0 = 0; ip0 < _SoftmaxPacks; ip0++)
    {
        float4 t = _SoftmaxInArr[int3((int)id.x, (int)id.y, ip0)];
        maxv = max(maxv, max(max(t.x, t.y), max(t.z, t.w)));
    }

    float sum = 0.0;
    for (int ip1 = 0; ip1 < _SoftmaxPacks; ip1++)
    {
        float4 t = _SoftmaxInArr[int3((int)id.x, (int)id.y, ip1)];
        float4 e = exp(t - maxv);
        sum += e.x + e.y + e.z + e.w;
    }
    float inv = sum > 0.0 ? (1.0 / sum) : 0.0;
    float4 o = exp(v - maxv) * inv;
    _SoftmaxOutArr[int3((int)id.x, (int)id.y, p)] = o;
}

void NcnnUnaryOpPack4_Impl(uint3 id)
{
    uint w, h, d;
    _UnaryOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 x = _UnaryInArr[int3((int)id.x, (int)id.y, p)];
    float4 y = x;
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
    _UnaryOutArr[int3((int)id.x, (int)id.y, p)] = y;
}

void NcnnBinaryOpPack4_Impl(uint3 id)
{
    uint w, h, d;
    _BinaryOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    float4 a = _BinaryA[int3((int)id.x, (int)id.y, p)];
    float4 b = _BinaryWithScalar != 0 ? _BinaryScalar : _BinaryB[int3((int)id.x, (int)id.y, p)];

    float4 o = a;
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
    _BinaryOutArr[int3((int)id.x, (int)id.y, p)] = o;
}

void NcnnBinaryOpPack4Broadcast_Impl(uint3 id)
{
    uint w, h, d;
    _BinaryOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    int broadcastMode = _BinaryPack4BroadcastMode;
    int3 aCoord = broadcastMode == 1
        ? int3(0, 0, p)
        : int3((int)id.x, (int)id.y, p);
    int3 bCoord = broadcastMode == 2
        ? int3(0, 0, p)
        : int3((int)id.x, (int)id.y, p);
    if (broadcastMode == 3)
        aCoord.z = 0;
    else if (broadcastMode == 4)
        bCoord.z = 0;

    float4 a = _BinaryA[aCoord];
    float4 b = _BinaryB[bCoord];
    if (broadcastMode == 3)
        a = a.xxxx;
    else if (broadcastMode == 4)
        b = b.xxxx;

    float4 o = a;
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
    _BinaryOutArr[int3((int)id.x, (int)id.y, p)] = o;
}

void NcnnBinaryOpPack4BufferScalar_Impl(uint3 id)
{
    uint w, h, d;
    _BinaryOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    float4 tex = _BinaryA[int3((int)id.x, (int)id.y, p)];
    float s = _BufB[0];
    float4 scalar = float4(s, s, s, s);
    float4 a = _BinaryPack4BufferScalarMode == 1 ? scalar : tex;
    float4 b = _BinaryPack4BufferScalarMode == 1 ? tex : scalar;

    float4 o = a;
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
    _BinaryOutArr[int3((int)id.x, (int)id.y, p)] = o;
}

void NcnnBinaryOpPack4ChannelVectorTex_Impl(uint3 id)
{
    uint w, h, d;
    _BinaryOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int slice = (int)id.z;
    if (slice < 0 || slice >= (int)d) return;

    uint vw, vh, vd;
    _BinaryB.GetDimensions(vw, vh, vd);
    int vectorPacks = max(_BinaryPack4ChannelVectorPacks, 1);
    int pack = slice % vectorPacks;
    bool isColumnVector = vw == 1 && vh > 1;

    float4 vec = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int c = pack * 4 + lane;
        float scalar = 0.0;
        if (isColumnVector)
        {
            if (c >= 0 && c < (int)vh)
                scalar = _BinaryB[int3(0, c, 0)].x;
        }
        else
        {
            if (c >= 0 && c < (int)vw)
                scalar = _BinaryB[int3(c, 0, 0)].x;
        }

        if (lane == 0) vec.x = scalar;
        else if (lane == 1) vec.y = scalar;
        else if (lane == 2) vec.z = scalar;
        else vec.w = scalar;
    }

    float4 tex = _BinaryA[int3((int)id.x, (int)id.y, slice)];
    float4 a = _BinaryPack4ChannelVectorMode == 1 ? vec : tex;
    float4 b = _BinaryPack4ChannelVectorMode == 1 ? tex : vec;

    float4 o = a;
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
    _BinaryOutArr[int3((int)id.x, (int)id.y, slice)] = o;
}

void NcnnBinaryOpScalarSingleBroadcast_Impl(uint3 id)
{
    uint w, h, d;
    _BinaryOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h || id.z >= d)
        return;

    int x = (int)id.x;
    int y = (int)id.y;
    int3 aCoord = int3(x, y, 0);
    int3 bCoord = int3(x, y, 0);
    int mode = _BinaryScalarSingleBroadcastMode;
    if (mode == 1)
    {
        aCoord = int3(x, 0, 0);
    }
    else if (mode == 2)
    {
        aCoord = int3(0, y, 0);
    }
    else if (mode == 3)
    {
        bCoord = int3(x, 0, 0);
    }
    else if (mode == 4)
    {
        bCoord = int3(0, y, 0);
    }
    else if (mode == 5)
    {
        aCoord = int3(0, 0, 0);
    }
    else if (mode == 6)
    {
        bCoord = int3(0, 0, 0);
    }
    else if (mode == 7)
    {
        aCoord = int3(y, 0, 0);
    }
    else if (mode == 8)
    {
        bCoord = int3(y, 0, 0);
    }

    float a = _BinaryA[aCoord].x;
    float b = _BinaryB[bCoord].x;
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
    _BinaryOutArr[int3(x, y, (int)id.z)] = float4(o, 0.0, 0.0, 0.0);
}

void NcnnCodeFormerMinEncodingFromSoftOneHot_Impl(uint3 id)
{
    uint ow, oh, od;
    _CodeFormerMinEncodingArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int outX = (int)id.x;
    int outY = (int)id.y;
    if (outX < 0 || outY < 0 || outX >= _CodeFormerCodebookSize || outY >= _CodeFormerTokenCount)
    {
        _CodeFormerMinEncodingArr[int3(outX, outY, (int)id.z)] = 0.0;
        return;
    }

    float bestValue = _CodeFormerSoftOneHot[int2(0, outY)].x;
    int bestIndex = 0;
    [loop]
    for (int i = 1; i < _CodeFormerCodebookSize; i++)
    {
        float value = _CodeFormerSoftOneHot[int2(i, outY)].x;
        if (value > bestValue)
        {
            bestValue = value;
            bestIndex = i;
        }
    }

    float oneHot = outX == bestIndex ? 1.0 : 0.0;
    _CodeFormerMinEncodingArr[int3(outX, outY, (int)id.z)] = float4(oneHot, 0.0, 0.0, 0.0);
}
