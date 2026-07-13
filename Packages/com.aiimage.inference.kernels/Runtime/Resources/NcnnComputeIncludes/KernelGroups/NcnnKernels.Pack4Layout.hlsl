// Auto-generated kernel implementation group: NcnnKernels.Pack4Layout.hlsl

void NcnnPack4ToBufferCHW_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    uint wh = (uint)(_Pack4W * _Pack4H);
    uint c = idx / wh;
    if (c >= (uint)_Pack4C) return;
    uint rem = idx - c * wh;
    uint y = rem / (uint)_Pack4W;
    uint x = rem - y * (uint)_Pack4W;
    int pack = (int)(c >> 2);
    int lane = (int)(c & 3);
    float4 v = _Pack4InArr[int3((int)x, (int)y, pack)];
    float o = lane == 0 ? v.x : (lane == 1 ? v.y : (lane == 2 ? v.z : v.w));
    _Pack4Out[idx] = o;
}

void NcnnLinearMatToBuffer_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    uint w = (uint)max(_Pack4W, 1);
    uint h = (uint)max(_Pack4H, 1);
    uint total = w * h;
    if (idx >= total)
        return;

    uint y = idx / w;
    uint x = idx - y * w;
    _Pack4Out[idx] = _LinearIn0[int2((int)x, (int)y)];
}

void NcnnFillLinearMatFromBuffer_Impl(uint3 id)
{
    uint w, h;
    _LinearOut0.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;

    uint logicalW = (uint)max(_FillW, 1);
    uint logicalH = (uint)max(_FillH, 1);
    uint idx = id.y * logicalW + id.x;
    uint total = logicalW * logicalH;
    _LinearOut0[int2((int)id.x, (int)id.y)] = idx < total ? _FillIn[idx] : 0.0;
}

void NcnnPack4ChannelsToWidth_Impl(uint3 id)
{
    uint ow, oh, od;
    _Pack4ChannelOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int outX = (int)id.x;
    if (outX < 0 || outX >= _Pack4ChannelCount)
    {
        _Pack4ChannelOutArr[int3((int)id.x, (int)id.y, (int)id.z)] = 0.0;
        return;
    }

    float scalar = NcnnReadPack4Channel(_Pack4ChannelInArr, 0, 0, outX);
    _Pack4ChannelOutArr[int3(outX, 0, 0)] = float4(scalar, 0.0, 0.0, 0.0);
}

void NcnnPack4ToBufferCDHW_Impl(uint3 id)
{
    uint idx = _BaseIndex + id.x;
    uint wh = (uint)(_Pack4W * _Pack4H);
    uint whd = wh * (uint)max(_Pack4D, 1);
    uint c = idx / whd;
    if (c >= (uint)_Pack4C) return;

    uint rem = idx - c * whd;
    uint z = rem / wh;
    rem -= z * wh;
    uint y = rem / (uint)_Pack4W;
    uint x = rem - y * (uint)_Pack4W;

    int packCount = max(1, (_Pack4C + 3) / 4);
    int pack = (int)(c >> 2);
    int lane = (int)(c & 3);
    int slice = (int)(z * (uint)packCount) + pack;
    float4 v = _Pack4InArr[int3((int)x, (int)y, slice)];
    float o = lane == 0 ? v.x : (lane == 1 ? v.y : (lane == 2 ? v.z : v.w));
    _Pack4Out[idx] = o;
}

void NcnnShuffleChannelPack4_Impl(uint3 id)
{
    uint w, h, d;
    _ShuffleOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    if (_ShuffleGroup <= 0 || _ShuffleChannels <= 0) return;

    int channelsPerGroup = _ShuffleChannels / _ShuffleGroup;
    int4 gz4 = p * 4 + int4(0, 1, 2, 3);
    float4 o = 0.0;
    if (gz4.x < _ShuffleChannels)
    {
        int srcChannel = (gz4.x % _ShuffleGroup) * channelsPerGroup + (gz4.x / _ShuffleGroup);
        o.x = NcnnReadPack4Channel(_ShuffleInArr, (int)id.x, (int)id.y, srcChannel);
    }
    if (gz4.y < _ShuffleChannels)
    {
        int srcChannel = (gz4.y % _ShuffleGroup) * channelsPerGroup + (gz4.y / _ShuffleGroup);
        o.y = NcnnReadPack4Channel(_ShuffleInArr, (int)id.x, (int)id.y, srcChannel);
    }
    if (gz4.z < _ShuffleChannels)
    {
        int srcChannel = (gz4.z % _ShuffleGroup) * channelsPerGroup + (gz4.z / _ShuffleGroup);
        o.z = NcnnReadPack4Channel(_ShuffleInArr, (int)id.x, (int)id.y, srcChannel);
    }
    if (gz4.w < _ShuffleChannels)
    {
        int srcChannel = (gz4.w % _ShuffleGroup) * channelsPerGroup + (gz4.w / _ShuffleGroup);
        o.w = NcnnReadPack4Channel(_ShuffleInArr, (int)id.x, (int)id.y, srcChannel);
    }

    _ShuffleOutArr[int3((int)id.x, (int)id.y, p)] = o;
}

void NcnnCropPack4_Impl(uint3 id)
{
    uint w, h, d;
    _CropPack4OutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    int x = (int)id.x + _CropPack4OffsetW;
    int y = (int)id.y + _CropPack4OffsetH;
    int4 oc4 = p * 4 + int4(0, 1, 2, 3);
    float4 o = 0.0;
    if (oc4.x < _CropPack4OutC) o.x = NcnnReadPack4Channel(_CropPack4InArr, x, y, oc4.x + _CropPack4OffsetC);
    if (oc4.y < _CropPack4OutC) o.y = NcnnReadPack4Channel(_CropPack4InArr, x, y, oc4.y + _CropPack4OffsetC);
    if (oc4.z < _CropPack4OutC) o.z = NcnnReadPack4Channel(_CropPack4InArr, x, y, oc4.z + _CropPack4OffsetC);
    if (oc4.w < _CropPack4OutC) o.w = NcnnReadPack4Channel(_CropPack4InArr, x, y, oc4.w + _CropPack4OffsetC);
    _CropPack4OutArr[int3((int)id.x, (int)id.y, p)] = o;
}

void NcnnSlicePack4_Impl(uint3 id)
{
    uint w, h, d;
    _SlicePack4OutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    int x = (int)id.x;
    int y = (int)id.y;
    int4 oc4 = p * 4 + int4(0, 1, 2, 3);
    float4 o = 0.0;

    if (_SlicePack4Axis == 0)
        x += _SlicePack4Begin;
    else if (_SlicePack4Axis == 1)
        y += _SlicePack4Begin;

    if (oc4.x < _SlicePack4OutC)
    {
        int sc = _SlicePack4Axis == 2 ? oc4.x + _SlicePack4Begin : oc4.x;
        o.x = NcnnReadPack4Channel(_SlicePack4InArr, x, y, sc);
    }
    if (oc4.y < _SlicePack4OutC)
    {
        int sc = _SlicePack4Axis == 2 ? oc4.y + _SlicePack4Begin : oc4.y;
        o.y = NcnnReadPack4Channel(_SlicePack4InArr, x, y, sc);
    }
    if (oc4.z < _SlicePack4OutC)
    {
        int sc = _SlicePack4Axis == 2 ? oc4.z + _SlicePack4Begin : oc4.z;
        o.z = NcnnReadPack4Channel(_SlicePack4InArr, x, y, sc);
    }
    if (oc4.w < _SlicePack4OutC)
    {
        int sc = _SlicePack4Axis == 2 ? oc4.w + _SlicePack4Begin : oc4.w;
        o.w = NcnnReadPack4Channel(_SlicePack4InArr, x, y, sc);
    }

    _SlicePack4OutArr[int3((int)id.x, (int)id.y, p)] = o;
}

void NcnnSliceLinearMat2D_Impl(uint3 id)
{
    uint w, h;
    _LinearOut0.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;

    int x = (int)id.x;
    int y = (int)id.y;
    if (_SlicePack4Axis == 0)
        x += _SlicePack4Begin;
    else if (_SlicePack4Axis == 1)
        y += _SlicePack4Begin;

    _LinearOut0[int2((int)id.x, (int)id.y)] = _LinearIn0[int2(x, y)];
}

void NcnnPermutePack4_Impl(uint3 id)
{
    uint w, h, d;
    _PermutePack4OutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    int x = (int)id.x;
    int y = (int)id.y;
    int4 oc4 = p * 4 + int4(0, 1, 2, 3);
    float4 o = 0.0;

    if (oc4.x < _PermutePack4OutC)
    {
        int srcW = _PermutePack4Axis0 == 0 ? x : (_PermutePack4Axis1 == 0 ? y : oc4.x);
        int srcH = _PermutePack4Axis0 == 1 ? x : (_PermutePack4Axis1 == 1 ? y : oc4.x);
        int srcC = _PermutePack4Axis0 == 2 ? x : (_PermutePack4Axis1 == 2 ? y : oc4.x);
        o.x = NcnnReadPack4Channel(_PermutePack4InArr, srcW, srcH, srcC);
    }
    if (oc4.y < _PermutePack4OutC)
    {
        int srcW = _PermutePack4Axis0 == 0 ? x : (_PermutePack4Axis1 == 0 ? y : oc4.y);
        int srcH = _PermutePack4Axis0 == 1 ? x : (_PermutePack4Axis1 == 1 ? y : oc4.y);
        int srcC = _PermutePack4Axis0 == 2 ? x : (_PermutePack4Axis1 == 2 ? y : oc4.y);
        o.y = NcnnReadPack4Channel(_PermutePack4InArr, srcW, srcH, srcC);
    }
    if (oc4.z < _PermutePack4OutC)
    {
        int srcW = _PermutePack4Axis0 == 0 ? x : (_PermutePack4Axis1 == 0 ? y : oc4.z);
        int srcH = _PermutePack4Axis0 == 1 ? x : (_PermutePack4Axis1 == 1 ? y : oc4.z);
        int srcC = _PermutePack4Axis0 == 2 ? x : (_PermutePack4Axis1 == 2 ? y : oc4.z);
        o.z = NcnnReadPack4Channel(_PermutePack4InArr, srcW, srcH, srcC);
    }
    if (oc4.w < _PermutePack4OutC)
    {
        int srcW = _PermutePack4Axis0 == 0 ? x : (_PermutePack4Axis1 == 0 ? y : oc4.w);
        int srcH = _PermutePack4Axis0 == 1 ? x : (_PermutePack4Axis1 == 1 ? y : oc4.w);
        int srcC = _PermutePack4Axis0 == 2 ? x : (_PermutePack4Axis1 == 2 ? y : oc4.w);
        o.w = NcnnReadPack4Channel(_PermutePack4InArr, srcW, srcH, srcC);
    }

    _PermutePack4OutArr[int3(x, y, p)] = o;
}

void NcnnPermutePack4CDHW_Impl(uint3 id)
{
    uint w, h, d;
    _PermutePack4CDHWOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int slice = (int)id.z;
    if (slice < 0 || slice >= (int)d) return;

    int outPackCount = max(1, (_PermutePack4CDHWOutC + 3) / 4);
    int outZ = slice / outPackCount;
    int outPack = slice - outZ * outPackCount;
    if (outZ < 0 || outZ >= _PermutePack4CDHWOutD)
    {
        _PermutePack4CDHWOutArr[int3((int)id.x, (int)id.y, slice)] = 0.0;
        return;
    }

    int outX = (int)id.x;
    int outY = (int)id.y;
    int4 oc4 = outPack * 4 + int4(0, 1, 2, 3);
    float4 o = 0.0;

    if (oc4.x < _PermutePack4CDHWOutC)
    {
        int srcW = _PermutePack4CDHWAxis0 == 0 ? outX : (_PermutePack4CDHWAxis1 == 0 ? outY : (_PermutePack4CDHWAxis2 == 0 ? outZ : oc4.x));
        int srcH = _PermutePack4CDHWAxis0 == 1 ? outX : (_PermutePack4CDHWAxis1 == 1 ? outY : (_PermutePack4CDHWAxis2 == 1 ? outZ : oc4.x));
        int srcD = _PermutePack4CDHWAxis0 == 2 ? outX : (_PermutePack4CDHWAxis1 == 2 ? outY : (_PermutePack4CDHWAxis2 == 2 ? outZ : oc4.x));
        int srcC = _PermutePack4CDHWAxis0 == 3 ? outX : (_PermutePack4CDHWAxis1 == 3 ? outY : (_PermutePack4CDHWAxis2 == 3 ? outZ : oc4.x));
        o.x = NcnnReadPack4ChannelCDHW(_PermutePack4CDHWInArr, srcW, srcH, srcD, srcC, _PermutePack4CDHWInC);
    }
    if (oc4.y < _PermutePack4CDHWOutC)
    {
        int srcW = _PermutePack4CDHWAxis0 == 0 ? outX : (_PermutePack4CDHWAxis1 == 0 ? outY : (_PermutePack4CDHWAxis2 == 0 ? outZ : oc4.y));
        int srcH = _PermutePack4CDHWAxis0 == 1 ? outX : (_PermutePack4CDHWAxis1 == 1 ? outY : (_PermutePack4CDHWAxis2 == 1 ? outZ : oc4.y));
        int srcD = _PermutePack4CDHWAxis0 == 2 ? outX : (_PermutePack4CDHWAxis1 == 2 ? outY : (_PermutePack4CDHWAxis2 == 2 ? outZ : oc4.y));
        int srcC = _PermutePack4CDHWAxis0 == 3 ? outX : (_PermutePack4CDHWAxis1 == 3 ? outY : (_PermutePack4CDHWAxis2 == 3 ? outZ : oc4.y));
        o.y = NcnnReadPack4ChannelCDHW(_PermutePack4CDHWInArr, srcW, srcH, srcD, srcC, _PermutePack4CDHWInC);
    }
    if (oc4.z < _PermutePack4CDHWOutC)
    {
        int srcW = _PermutePack4CDHWAxis0 == 0 ? outX : (_PermutePack4CDHWAxis1 == 0 ? outY : (_PermutePack4CDHWAxis2 == 0 ? outZ : oc4.z));
        int srcH = _PermutePack4CDHWAxis0 == 1 ? outX : (_PermutePack4CDHWAxis1 == 1 ? outY : (_PermutePack4CDHWAxis2 == 1 ? outZ : oc4.z));
        int srcD = _PermutePack4CDHWAxis0 == 2 ? outX : (_PermutePack4CDHWAxis1 == 2 ? outY : (_PermutePack4CDHWAxis2 == 2 ? outZ : oc4.z));
        int srcC = _PermutePack4CDHWAxis0 == 3 ? outX : (_PermutePack4CDHWAxis1 == 3 ? outY : (_PermutePack4CDHWAxis2 == 3 ? outZ : oc4.z));
        o.z = NcnnReadPack4ChannelCDHW(_PermutePack4CDHWInArr, srcW, srcH, srcD, srcC, _PermutePack4CDHWInC);
    }
    if (oc4.w < _PermutePack4CDHWOutC)
    {
        int srcW = _PermutePack4CDHWAxis0 == 0 ? outX : (_PermutePack4CDHWAxis1 == 0 ? outY : (_PermutePack4CDHWAxis2 == 0 ? outZ : oc4.w));
        int srcH = _PermutePack4CDHWAxis0 == 1 ? outX : (_PermutePack4CDHWAxis1 == 1 ? outY : (_PermutePack4CDHWAxis2 == 1 ? outZ : oc4.w));
        int srcD = _PermutePack4CDHWAxis0 == 2 ? outX : (_PermutePack4CDHWAxis1 == 2 ? outY : (_PermutePack4CDHWAxis2 == 2 ? outZ : oc4.w));
        int srcC = _PermutePack4CDHWAxis0 == 3 ? outX : (_PermutePack4CDHWAxis1 == 3 ? outY : (_PermutePack4CDHWAxis2 == 3 ? outZ : oc4.w));
        o.w = NcnnReadPack4ChannelCDHW(_PermutePack4CDHWInArr, srcW, srcH, srcD, srcC, _PermutePack4CDHWInC);
    }

    _PermutePack4CDHWOutArr[int3(outX, outY, slice)] = o;
}

void NcnnPermuteLinearMat2D_Impl(uint3 id)
{
    uint ow, oh;
    _LinearOut0.GetDimensions(ow, oh);
    if (id.x >= ow || id.y >= oh)
        return;

    int outX = (int)id.x;
    int outY = (int)id.y;
    int srcX = _PermutePack4Axis0 == 0 ? outX : outY;
    int srcY = _PermutePack4Axis0 == 1 ? outX : outY;

    float value = 0.0;
    if (srcX >= 0 && srcX < _PermutePack4InW && srcY >= 0 && srcY < _PermutePack4InH)
        value = _LinearIn0[int2(srcX, srcY)];

    _LinearOut0[int2(outX, outY)] = value;
}

void NcnnWindowPartitionPack4_Impl(uint3 id)
{
    uint w, h, d;
    _WindowPartitionPack4OutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int pack = (int)id.z;
    if (pack < 0 || pack >= (int)d) return;

    int outX = (int)id.x;
    int outToken = (int)id.y;
    int4 outWindow4 = pack * 4 + int4(0, 1, 2, 3);
    float4 o = 0.0;

    int tokenStrideBC = _WindowPartitionPack4TokensB * _WindowPartitionPack4TokensC;
    int tokenA = outToken / max(1, tokenStrideBC);
    int tokenRem = outToken - tokenA * tokenStrideBC;
    int tokenB = tokenRem / max(1, _WindowPartitionPack4TokensC);
    int tokenC = tokenRem - tokenB * _WindowPartitionPack4TokensC;

    if (outWindow4.x < _WindowPartitionPack4OutC)
    {
        int groupA = outWindow4.x / max(1, _WindowPartitionPack4GroupsB * _WindowPartitionPack4GroupsC);
        int groupRem = outWindow4.x - groupA * _WindowPartitionPack4GroupsB * _WindowPartitionPack4GroupsC;
        int groupB = groupRem / max(1, _WindowPartitionPack4GroupsC);
        int groupC = groupRem - groupB * _WindowPartitionPack4GroupsC;
        int srcC = groupA * _WindowPartitionPack4TokensA + tokenA;
        int srcD = groupB * _WindowPartitionPack4TokensB + tokenB;
        int srcH = groupC * _WindowPartitionPack4TokensC + tokenC;
        o.x = NcnnReadPack4ChannelCDHW(_WindowPartitionPack4InArr, outX, srcH, srcD, srcC, _WindowPartitionPack4InC);
    }
    if (outWindow4.y < _WindowPartitionPack4OutC)
    {
        int groupA = outWindow4.y / max(1, _WindowPartitionPack4GroupsB * _WindowPartitionPack4GroupsC);
        int groupRem = outWindow4.y - groupA * _WindowPartitionPack4GroupsB * _WindowPartitionPack4GroupsC;
        int groupB = groupRem / max(1, _WindowPartitionPack4GroupsC);
        int groupC = groupRem - groupB * _WindowPartitionPack4GroupsC;
        int srcC = groupA * _WindowPartitionPack4TokensA + tokenA;
        int srcD = groupB * _WindowPartitionPack4TokensB + tokenB;
        int srcH = groupC * _WindowPartitionPack4TokensC + tokenC;
        o.y = NcnnReadPack4ChannelCDHW(_WindowPartitionPack4InArr, outX, srcH, srcD, srcC, _WindowPartitionPack4InC);
    }
    if (outWindow4.z < _WindowPartitionPack4OutC)
    {
        int groupA = outWindow4.z / max(1, _WindowPartitionPack4GroupsB * _WindowPartitionPack4GroupsC);
        int groupRem = outWindow4.z - groupA * _WindowPartitionPack4GroupsB * _WindowPartitionPack4GroupsC;
        int groupB = groupRem / max(1, _WindowPartitionPack4GroupsC);
        int groupC = groupRem - groupB * _WindowPartitionPack4GroupsC;
        int srcC = groupA * _WindowPartitionPack4TokensA + tokenA;
        int srcD = groupB * _WindowPartitionPack4TokensB + tokenB;
        int srcH = groupC * _WindowPartitionPack4TokensC + tokenC;
        o.z = NcnnReadPack4ChannelCDHW(_WindowPartitionPack4InArr, outX, srcH, srcD, srcC, _WindowPartitionPack4InC);
    }
    if (outWindow4.w < _WindowPartitionPack4OutC)
    {
        int groupA = outWindow4.w / max(1, _WindowPartitionPack4GroupsB * _WindowPartitionPack4GroupsC);
        int groupRem = outWindow4.w - groupA * _WindowPartitionPack4GroupsB * _WindowPartitionPack4GroupsC;
        int groupB = groupRem / max(1, _WindowPartitionPack4GroupsC);
        int groupC = groupRem - groupB * _WindowPartitionPack4GroupsC;
        int srcC = groupA * _WindowPartitionPack4TokensA + tokenA;
        int srcD = groupB * _WindowPartitionPack4TokensB + tokenB;
        int srcH = groupC * _WindowPartitionPack4TokensC + tokenC;
        o.w = NcnnReadPack4ChannelCDHW(_WindowPartitionPack4InArr, outX, srcH, srcD, srcC, _WindowPartitionPack4InC);
    }

    _WindowPartitionPack4OutArr[int3(outX, outToken, pack)] = o;
}

void NcnnWindowUnpartitionPack4_Impl(uint3 id)
{
    uint w, h, d;
    _WindowUnpartitionPack4OutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int slice = (int)id.z;
    if (slice < 0 || slice >= (int)d) return;

    int outPackCount = max(1, (_WindowUnpartitionPack4OutC + 3) / 4);
    int outZ = slice / outPackCount;
    int outPack = slice - outZ * outPackCount;
    if (outZ < 0 || outZ >= _WindowUnpartitionPack4OutD)
    {
        _WindowUnpartitionPack4OutArr[int3((int)id.x, (int)id.y, slice)] = 0.0;
        return;
    }

    int outX = (int)id.x;
    int outY = (int)id.y;
    int4 outC4 = outPack * 4 + int4(0, 1, 2, 3);
    float4 o = 0.0;

    if (outC4.x < _WindowUnpartitionPack4OutC)
    {
        int groupA = outC4.x / max(1, _WindowUnpartitionPack4TokensA);
        int tokenA = outC4.x - groupA * _WindowUnpartitionPack4TokensA;
        int groupB = outZ / max(1, _WindowUnpartitionPack4TokensB);
        int tokenB = outZ - groupB * _WindowUnpartitionPack4TokensB;
        int groupC = outY / max(1, _WindowUnpartitionPack4TokensC);
        int tokenC = outY - groupC * _WindowUnpartitionPack4TokensC;
        int inputGroup = ((groupA * _WindowUnpartitionPack4GroupsB) + groupB) * _WindowUnpartitionPack4GroupsC + groupC;
        int inputToken = ((tokenA * _WindowUnpartitionPack4TokensB) + tokenB) * _WindowUnpartitionPack4TokensC + tokenC;
        o.x = NcnnReadPack4Channel(_WindowUnpartitionPack4InArr, outX, inputToken, inputGroup);
    }
    if (outC4.y < _WindowUnpartitionPack4OutC)
    {
        int groupA = outC4.y / max(1, _WindowUnpartitionPack4TokensA);
        int tokenA = outC4.y - groupA * _WindowUnpartitionPack4TokensA;
        int groupB = outZ / max(1, _WindowUnpartitionPack4TokensB);
        int tokenB = outZ - groupB * _WindowUnpartitionPack4TokensB;
        int groupC = outY / max(1, _WindowUnpartitionPack4TokensC);
        int tokenC = outY - groupC * _WindowUnpartitionPack4TokensC;
        int inputGroup = ((groupA * _WindowUnpartitionPack4GroupsB) + groupB) * _WindowUnpartitionPack4GroupsC + groupC;
        int inputToken = ((tokenA * _WindowUnpartitionPack4TokensB) + tokenB) * _WindowUnpartitionPack4TokensC + tokenC;
        o.y = NcnnReadPack4Channel(_WindowUnpartitionPack4InArr, outX, inputToken, inputGroup);
    }
    if (outC4.z < _WindowUnpartitionPack4OutC)
    {
        int groupA = outC4.z / max(1, _WindowUnpartitionPack4TokensA);
        int tokenA = outC4.z - groupA * _WindowUnpartitionPack4TokensA;
        int groupB = outZ / max(1, _WindowUnpartitionPack4TokensB);
        int tokenB = outZ - groupB * _WindowUnpartitionPack4TokensB;
        int groupC = outY / max(1, _WindowUnpartitionPack4TokensC);
        int tokenC = outY - groupC * _WindowUnpartitionPack4TokensC;
        int inputGroup = ((groupA * _WindowUnpartitionPack4GroupsB) + groupB) * _WindowUnpartitionPack4GroupsC + groupC;
        int inputToken = ((tokenA * _WindowUnpartitionPack4TokensB) + tokenB) * _WindowUnpartitionPack4TokensC + tokenC;
        o.z = NcnnReadPack4Channel(_WindowUnpartitionPack4InArr, outX, inputToken, inputGroup);
    }
    if (outC4.w < _WindowUnpartitionPack4OutC)
    {
        int groupA = outC4.w / max(1, _WindowUnpartitionPack4TokensA);
        int tokenA = outC4.w - groupA * _WindowUnpartitionPack4TokensA;
        int groupB = outZ / max(1, _WindowUnpartitionPack4TokensB);
        int tokenB = outZ - groupB * _WindowUnpartitionPack4TokensB;
        int groupC = outY / max(1, _WindowUnpartitionPack4TokensC);
        int tokenC = outY - groupC * _WindowUnpartitionPack4TokensC;
        int inputGroup = ((groupA * _WindowUnpartitionPack4GroupsB) + groupB) * _WindowUnpartitionPack4GroupsC + groupC;
        int inputToken = ((tokenA * _WindowUnpartitionPack4TokensB) + tokenB) * _WindowUnpartitionPack4TokensC + tokenC;
        o.w = NcnnReadPack4Channel(_WindowUnpartitionPack4InArr, outX, inputToken, inputGroup);
    }

    _WindowUnpartitionPack4OutArr[int3(outX, outY, slice)] = o;
}

void NcnnReshapePack4ToScalar2D_Impl(uint3 id)
{
    uint ow, oh, od;
    _ReshapePack4ToScalar2DOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    uint linearIndex = id.y * ow + id.x;
    float scalar = NcnnReadPack4LinearScalar2DInput(
        linearIndex,
        _ReshapePack4ToScalar2DInDims,
        _ReshapePack4ToScalar2DInW,
        _ReshapePack4ToScalar2DInH,
        _ReshapePack4ToScalar2DInD,
        _ReshapePack4ToScalar2DInC);
    _ReshapePack4ToScalar2DOutArr[int3((int)id.x, (int)id.y, (int)id.z)] = float4(scalar, 0.0, 0.0, 0.0);
}

void NcnnReshapePack4ToLinearMat_Impl(uint3 id)
{
    uint ow, oh;
    _LinearOut0.GetDimensions(ow, oh);
    if (id.x >= ow || id.y >= oh)
        return;

    uint linearIndex = id.y * ow + id.x;
    float scalar = NcnnReadPack4LinearScalar2DInput(
        linearIndex,
        _ReshapePack4ToScalar2DInDims,
        _ReshapePack4ToScalar2DInW,
        _ReshapePack4ToScalar2DInH,
        _ReshapePack4ToScalar2DInD,
        _ReshapePack4ToScalar2DInC);
    _LinearOut0[int2((int)id.x, (int)id.y)] = scalar;
}

void NcnnReshapePack4ToPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ReshapePack4ToPack4OutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int slice = (int)id.z;
    int outPacks = max(1, (_ReshapePack4ToPack4OutC + 3) / 4);
    int outZ = 0;
    int outPack = slice;
    if (_ReshapePack4ToPack4OutDims >= 4)
    {
        outZ = slice / outPacks;
        outPack = slice - outZ * outPacks;
        if (outZ < 0 || outZ >= max(1, _ReshapePack4ToPack4OutD))
        {
            _ReshapePack4ToPack4OutArr[int3((int)id.x, (int)id.y, slice)] = 0.0;
            return;
        }
    }

    float4 value = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int outC = outPack * 4 + lane;
        if (outC >= _ReshapePack4ToPack4OutC)
            continue;

        uint linearIndex = _ReshapePack4ToPack4OutDims >= 4
            ? (((uint)outC * (uint)max(1, _ReshapePack4ToPack4OutD) + (uint)outZ) * (uint)_ReshapePack4ToPack4OutH + id.y) * (uint)_ReshapePack4ToPack4OutW + id.x
            : ((uint)outC * (uint)_ReshapePack4ToPack4OutH + id.y) * (uint)_ReshapePack4ToPack4OutW + id.x;
        float scalar = NcnnReadPack4LinearPack4Input(
            linearIndex,
            _ReshapePack4ToPack4InDims,
            _ReshapePack4ToPack4InW,
            _ReshapePack4ToPack4InH,
            _ReshapePack4ToPack4InD,
            _ReshapePack4ToPack4InC);
        if (lane == 0) value.x = scalar;
        else if (lane == 1) value.y = scalar;
        else if (lane == 2) value.z = scalar;
        else value.w = scalar;
    }

    _ReshapePack4ToPack4OutArr[int3((int)id.x, (int)id.y, slice)] = value;
}

void NcnnReshapeScalar2DToPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ReshapeScalar2DOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int slice = (int)id.z;
    int outPacks = max(1, (_ReshapeScalar2DOutC + 3) / 4);
    int z = 0;
    int pack = slice;
    if (_ReshapeScalar2DOutDims >= 4)
    {
        z = slice / outPacks;
        pack = slice - z * outPacks;
        if (z < 0 || z >= max(1, _ReshapeScalar2DOutD))
        {
            _ReshapeScalar2DOutArr[int3((int)id.x, (int)id.y, slice)] = 0.0;
            return;
        }
    }

    float4 value = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int c = pack * 4 + lane;
        if (c >= _ReshapeScalar2DOutC)
            continue;

        uint linearIndex = _ReshapeScalar2DOutDims >= 4
            ? (((uint)c * (uint)max(1, _ReshapeScalar2DOutD) + (uint)z) * (uint)_ReshapeScalar2DOutH + id.y) * (uint)_ReshapeScalar2DOutW + id.x
            : ((uint)c * (uint)_ReshapeScalar2DOutH + id.y) * (uint)_ReshapeScalar2DOutW + id.x;

        float scalar = NcnnReadScalar2DLinear(linearIndex, _ReshapeScalar2DInW, _ReshapeScalar2DInH);
        if (lane == 0) value.x = scalar;
        else if (lane == 1) value.y = scalar;
        else if (lane == 2) value.z = scalar;
        else value.w = scalar;
    }

    _ReshapeScalar2DOutArr[int3((int)id.x, (int)id.y, slice)] = value;
}

void NcnnReshapeLinearMatToPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ReshapeScalar2DOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int slice = (int)id.z;
    int outPacks = max(1, (_ReshapeScalar2DOutC + 3) / 4);
    int z = 0;
    int pack = slice;
    if (_ReshapeScalar2DOutDims >= 4)
    {
        z = slice / outPacks;
        pack = slice - z * outPacks;
        if (z < 0 || z >= max(1, _ReshapeScalar2DOutD))
        {
            _ReshapeScalar2DOutArr[int3((int)id.x, (int)id.y, slice)] = 0.0;
            return;
        }
    }

    float4 value = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int c = pack * 4 + lane;
        if (c >= _ReshapeScalar2DOutC)
            continue;

        uint linearIndex = _ReshapeScalar2DOutDims >= 4
            ? (((uint)c * (uint)max(1, _ReshapeScalar2DOutD) + (uint)z) * (uint)_ReshapeScalar2DOutH + id.y) * (uint)_ReshapeScalar2DOutW + id.x
            : ((uint)c * (uint)_ReshapeScalar2DOutH + id.y) * (uint)_ReshapeScalar2DOutW + id.x;

        float scalar = NcnnReadLinearMatScalar(_LinearIn0, linearIndex, _ReshapeScalar2DInW, _ReshapeScalar2DInH);
        if (lane == 0) value.x = scalar;
        else if (lane == 1) value.y = scalar;
        else if (lane == 2) value.z = scalar;
        else value.w = scalar;
    }

    _ReshapeScalar2DOutArr[int3((int)id.x, (int)id.y, slice)] = value;
}

void NcnnAttentionReshapePack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _AttentionReshapeOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int outX = (int)id.x;
    int outY = (int)id.y;
    int slice = (int)id.z;
    int outPacks = max(1, (_AttentionReshapeOutC + 3) / 4);
    int outZ = slice / outPacks;
    int outPack = slice - outZ * outPacks;
    if (outZ < 0 || outZ >= _AttentionReshapeOutD)
    {
        _AttentionReshapeOutArr[int3(outX, outY, slice)] = 0.0;
        return;
    }

    float4 o = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int outC = outPack * 4 + lane;
        if (outC >= _AttentionReshapeOutC)
            continue;

        int srcWindow = outZ;
        int srcToken = outY;
        int srcChannel = outC * _AttentionReshapeHeadDim + outX;
        if (outX < 0 || outX >= _AttentionReshapeHeadDim)
            continue;
        if (srcWindow < 0 || srcWindow >= _AttentionReshapeInC)
            continue;
        if (srcToken < 0 || srcToken >= _AttentionReshapeInH)
            continue;
        if (srcChannel < 0 || srcChannel >= _AttentionReshapeInW)
            continue;

        float scalar = NcnnReadPack4Channel(_AttentionReshapeInArr, srcChannel, srcToken, srcWindow);
        NcnnWriteLane(o, lane, scalar);
    }

    _AttentionReshapeOutArr[int3(outX, outY, slice)] = o;
}

void NcnnAttentionContextFlattenPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _AttentionContextFlattenOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int outX = (int)id.x;
    int outY = (int)id.y;
    int slice = (int)id.z;
    int logicalOutChannels = _AttentionContextFlattenOutDims == 3
        ? max(1, _AttentionContextFlattenOutC)
        : 1;
    int outPacks = max(1, (logicalOutChannels + 3) / 4);
    if (slice < 0 || slice >= outPacks)
    {
        _AttentionContextFlattenOutArr[int3(outX, outY, slice)] = 0.0;
        return;
    }

    float4 o = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int outChannel = slice * 4 + lane;
        if (outChannel >= logicalOutChannels)
            continue;

        int srcWindow;
        int srcToken;
        if (_AttentionContextFlattenOutDims == 3 && _AttentionContextFlattenOutC > 1)
        {
            srcWindow = outChannel;
            srcToken = outY;
        }
        else
        {
            if (outChannel != 0 || _AttentionContextFlattenInH <= 0)
                continue;
            srcWindow = outY / _AttentionContextFlattenInH;
            srcToken = outY - srcWindow * _AttentionContextFlattenInH;
        }

        if (srcWindow < 0 || srcWindow >= _AttentionContextFlattenInD)
            continue;
        if (srcToken < 0 || srcToken >= _AttentionContextFlattenInH)
            continue;
        if (_AttentionContextFlattenInC <= 0 || _AttentionContextFlattenInW <= 0)
            continue;

        int srcHead = outX % _AttentionContextFlattenInC;
        int srcDim = outX / _AttentionContextFlattenInC;
        if (srcHead < 0 || srcHead >= _AttentionContextFlattenInC)
            continue;
        if (srcDim < 0 || srcDim >= _AttentionContextFlattenInW)
            continue;

        float scalar = NcnnReadPack4ChannelCDHW(
            _AttentionContextFlattenInArr,
            srcDim,
            srcToken,
            srcWindow,
            srcHead,
            _AttentionContextFlattenInC);
        NcnnWriteLane(o, lane, scalar);
    }

    _AttentionContextFlattenOutArr[int3(outX, outY, slice)] = o;
}

void NcnnSlicePack4CDHW_Impl(uint3 id)
{
    uint w, h, d;
    _SlicePack4CDHWOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h || id.z >= d)
        return;

    int outX = (int)id.x;
    int outY = (int)id.y;
    int slice = (int)id.z;
    int outPacks = max(1, (_SlicePack4CDHWOutC + 3) / 4);
    int outZ = slice / outPacks;
    int outPack = slice - outZ * outPacks;
    if (outZ < 0 || outZ >= _SlicePack4CDHWOutD)
    {
        _SlicePack4CDHWOutArr[int3(outX, outY, slice)] = 0.0;
        return;
    }

    int srcX = outX;
    int srcY = outY;
    int srcZ = outZ;
    if (_SlicePack4CDHWAxis == 0)
        srcX += _SlicePack4CDHWBegin;
    else if (_SlicePack4CDHWAxis == 1)
        srcY += _SlicePack4CDHWBegin;
    else if (_SlicePack4CDHWAxis == 2)
        srcZ += _SlicePack4CDHWBegin;

    float4 o = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int outC = outPack * 4 + lane;
        if (outC >= _SlicePack4CDHWOutC)
            continue;

        int srcC = _SlicePack4CDHWAxis == 3
            ? outC + _SlicePack4CDHWBegin
            : outC;
        float scalar = NcnnReadPack4ChannelCDHW(
            _SlicePack4CDHWInArr,
            srcX,
            srcY,
            srcZ,
            srcC,
            _SlicePack4CDHWInC);
        NcnnWriteLane(o, lane, scalar);
    }

    _SlicePack4CDHWOutArr[int3(outX, outY, slice)] = o;
}

void NcnnReorgPack4_Impl(uint3 id)
{
    uint w, h, d;
    _ReorgOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    int srcP = p / 4;
    int off = p % 4;

    int srcX = (int)id.x * 2 + (off % 2);
    int srcY = (int)id.y * 2 + (off / 2);

    _ReorgOutArr[int3((int)id.x, (int)id.y, p)] = _ReorgInArr[int3(srcX, srcY, srcP)];
}

void NcnnPointwisePack4_Impl(uint3 id)
{
    uint w, h, d;
    _PointwiseOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;
    float4 x = _PointwiseInArr[int3((int)id.x, (int)id.y, p)];
    _PointwiseOutArr[int3((int)id.x, (int)id.y, p)] = NcnnApplyPointwise4(x);
}

void NcnnPixelShufflePack4_Impl(uint3 id)
{
    uint w, h, d;
    _PixelShufflePack4OutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    int p = (int)id.z;
    if (p < 0 || p >= (int)d) return;

    int scale = max(1, _PixelShufflePack4Scale);
    int inX = (int)id.x / scale;
    int inY = (int)id.y / scale;
    int sh = (int)id.y % scale;
    int sw = (int)id.x % scale;
    int4 oc4 = p * 4 + int4(0, 1, 2, 3);
    float4 o = 0.0;

    if (oc4.x < _PixelShufflePack4OutC)
    {
        int srcChannel = _PixelShufflePack4Mode == 0
            ? oc4.x * scale * scale + sh * scale + sw
            : (sh * scale + sw) * _PixelShufflePack4OutC + oc4.x;
        o.x = NcnnReadPack4Channel(_PixelShufflePack4InArr, inX, inY, srcChannel);
    }
    if (oc4.y < _PixelShufflePack4OutC)
    {
        int srcChannel = _PixelShufflePack4Mode == 0
            ? oc4.y * scale * scale + sh * scale + sw
            : (sh * scale + sw) * _PixelShufflePack4OutC + oc4.y;
        o.y = NcnnReadPack4Channel(_PixelShufflePack4InArr, inX, inY, srcChannel);
    }
    if (oc4.z < _PixelShufflePack4OutC)
    {
        int srcChannel = _PixelShufflePack4Mode == 0
            ? oc4.z * scale * scale + sh * scale + sw
            : (sh * scale + sw) * _PixelShufflePack4OutC + oc4.z;
        o.z = NcnnReadPack4Channel(_PixelShufflePack4InArr, inX, inY, srcChannel);
    }
    if (oc4.w < _PixelShufflePack4OutC)
    {
        int srcChannel = _PixelShufflePack4Mode == 0
            ? oc4.w * scale * scale + sh * scale + sw
            : (sh * scale + sw) * _PixelShufflePack4OutC + oc4.w;
        o.w = NcnnReadPack4Channel(_PixelShufflePack4InArr, inX, inY, srcChannel);
    }

    _PixelShufflePack4OutArr[int3((int)id.x, (int)id.y, p)] = o;
}

void NcnnUnfoldPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _TexOut0Arr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh)
        return;

    int outw = max(1, _UnfoldOutW);
    int outh = max(1, _UnfoldOutH);
    int size = outw * outh;
    int maxk = max(1, _UnfoldKernelW * _UnfoldKernelH);
    int outRow = (int)id.y;
    int rem = (int)id.x;
    if (rem < 0 || rem >= size || outRow < 0 || outRow >= maxk * max(1, _UnfoldInC))
        return;

    int oy = rem / outw;
    int ox = rem - oy * outw;
    int c = outRow / maxk;
    int k = outRow - c * maxk;
    int u = k / max(1, _UnfoldKernelW);
    int v = k - u * max(1, _UnfoldKernelW);
    int inY = oy * _UnfoldStrideH + u * _UnfoldDilationH - _UnfoldPadTop;
    int inX = ox * _UnfoldStrideW + v * _UnfoldDilationW - _UnfoldPadLeft;

    float value = _UnfoldPadValue;
    if ((uint)inX < (uint)_UnfoldInW && (uint)inY < (uint)_UnfoldInH && c < _UnfoldInC)
        value = NcnnReadPack4Channel(_TexIn0Arr, inX, inY, c);

    _TexOut0Arr[int3((int)id.x, (int)id.y, 0)] = float4(value, 0.0, 0.0, 0.0);
}
