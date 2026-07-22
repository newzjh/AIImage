// Texture-backed Sentis-style shape/index operators.

int NcnnSentisTotalCount(int w, int h, int d, int c)
{
    return max(1, w) * max(1, h) * max(1, d) * max(1, c);
}

int NcnnSentisLinearIndex(int x, int y, int z, int c, int w, int h, int d)
{
    return (((c * max(1, d) + z) * max(1, h) + y) * max(1, w)) + x;
}

void NcnnSentisDecodeLinear(int linearIndex, int dims, int w, int h, int d, int c, out int x, out int y, out int z, out int ch)
{
    x = 0;
    y = 0;
    z = 0;
    ch = 0;

    int width = max(1, w);
    x = linearIndex % width;
    int rem = linearIndex / width;
    if (dims <= 1)
        return;

    y = rem % max(1, h);
    rem /= max(1, h);
    if (dims == 2)
        return;

    if (dims == 3)
    {
        ch = rem;
        return;
    }

    z = rem % max(1, d);
    ch = rem / max(1, d);
}

int NcnnSentisAxisSize(int dims, int w, int h, int d, int c, int axis)
{
    if (axis < 0)
        axis += dims;
    if (dims <= 1)
        return max(1, w);
    if (dims == 2)
        return axis == 0 ? max(1, h) : max(1, w);
    if (dims == 3)
        return axis == 0 ? max(1, c) : (axis == 1 ? max(1, h) : max(1, w));
    return axis == 0 ? max(1, c) : (axis == 1 ? max(1, d) : (axis == 2 ? max(1, h) : max(1, w)));
}

int NcnnSentisGetAxisCoord(int dims, int x, int y, int z, int ch, int axis)
{
    if (axis < 0)
        axis += dims;
    if (dims <= 1)
        return x;
    if (dims == 2)
        return axis == 0 ? y : x;
    if (dims == 3)
        return axis == 0 ? ch : (axis == 1 ? y : x);
    return axis == 0 ? ch : (axis == 1 ? z : (axis == 2 ? y : x));
}

void NcnnSentisSetAxisCoord(inout int x, inout int y, inout int z, inout int ch, int dims, int axis, int value)
{
    if (axis < 0)
        axis += dims;
    if (dims <= 1)
    {
        x = value;
        return;
    }
    if (dims == 2)
    {
        if (axis == 0) y = value;
        else x = value;
        return;
    }
    if (dims == 3)
    {
        if (axis == 0) ch = value;
        else if (axis == 1) y = value;
        else x = value;
        return;
    }
    if (axis == 0) ch = value;
    else if (axis == 1) z = value;
    else if (axis == 2) y = value;
    else x = value;
}

float NcnnSentisReadLinear(Texture2D<float> tex, int linearIndex, int storageW, int storageH)
{
    int sw = max(1, storageW);
    int x = linearIndex % sw;
    int y = linearIndex / sw;
    if (y < 0 || y >= max(1, storageH))
        return 0.0;
    return tex[int2(x, y)];
}

float NcnnSentisReadAt(Texture2D<float> tex, int x, int y, int z, int ch, int w, int h, int d, int storageW, int storageH)
{
    int linearIndex = NcnnSentisLinearIndex(x, y, z, ch, w, h, d);
    return NcnnSentisReadLinear(tex, linearIndex, storageW, storageH);
}

float NcnnSentisReadBroadcastIn0(int ox, int oy, int oz, int oc)
{
    int ix = 0, iy = 0, iz = 0, ic = 0;
    int shift = _SentisOutDims - _SentisIn0Dims;
    [loop]
    for (int outAxis = 0; outAxis < _SentisOutDims; outAxis++)
    {
        int inAxis = outAxis - shift;
        if (inAxis < 0)
            continue;
        int coord = NcnnSentisGetAxisCoord(_SentisOutDims, ox, oy, oz, oc, outAxis);
        int size = NcnnSentisAxisSize(_SentisIn0Dims, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0C, inAxis);
        if (size == 1)
            coord = 0;
        NcnnSentisSetAxisCoord(ix, iy, iz, ic, _SentisIn0Dims, inAxis, clamp(coord, 0, size - 1));
    }
    return NcnnSentisReadAt(_LinearIn0, ix, iy, iz, ic, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0StorageW, _SentisIn0StorageH);
}

float NcnnSentisReadBroadcastIn1(int ox, int oy, int oz, int oc)
{
    int ix = 0, iy = 0, iz = 0, ic = 0;
    int shift = _SentisOutDims - _SentisIn1Dims;
    [loop]
    for (int outAxis = 0; outAxis < _SentisOutDims; outAxis++)
    {
        int inAxis = outAxis - shift;
        if (inAxis < 0)
            continue;
        int coord = NcnnSentisGetAxisCoord(_SentisOutDims, ox, oy, oz, oc, outAxis);
        int size = NcnnSentisAxisSize(_SentisIn1Dims, _SentisIn1W, _SentisIn1H, _SentisIn1D, _SentisIn1C, inAxis);
        if (size == 1)
            coord = 0;
        NcnnSentisSetAxisCoord(ix, iy, iz, ic, _SentisIn1Dims, inAxis, clamp(coord, 0, size - 1));
    }
    return NcnnSentisReadAt(_LinearIn1, ix, iy, iz, ic, _SentisIn1W, _SentisIn1H, _SentisIn1D, _SentisIn1StorageW, _SentisIn1StorageH);
}

float NcnnSentisReadBroadcastIn2(int ox, int oy, int oz, int oc)
{
    int ix = 0, iy = 0, iz = 0, ic = 0;
    int shift = _SentisOutDims - _SentisIn2Dims;
    [loop]
    for (int outAxis = 0; outAxis < _SentisOutDims; outAxis++)
    {
        int inAxis = outAxis - shift;
        if (inAxis < 0)
            continue;
        int coord = NcnnSentisGetAxisCoord(_SentisOutDims, ox, oy, oz, oc, outAxis);
        int size = NcnnSentisAxisSize(_SentisIn2Dims, _SentisIn2W, _SentisIn2H, _SentisIn2D, _SentisIn2C, inAxis);
        if (size == 1)
            coord = 0;
        NcnnSentisSetAxisCoord(ix, iy, iz, ic, _SentisIn2Dims, inAxis, clamp(coord, 0, size - 1));
    }
    return NcnnSentisReadAt(_LinearIn2, ix, iy, iz, ic, _SentisIn2W, _SentisIn2H, _SentisIn2D, _SentisIn2StorageW, _SentisIn2StorageH);
}

void NcnnSentisConstantLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int linearIndex = (int)id.y * (int)sw + (int)id.x;
    _LinearOut0[int2((int)id.x, (int)id.y)] = linearIndex < _SentisTotal ? _SentisValue0 : 0.0;
}

void NcnnSentisRangeLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int linearIndex = (int)id.y * (int)sw + (int)id.x;
    float value = linearIndex < _SentisTotal ? (_SentisValue0 + _SentisValue1 * (float)linearIndex) : 0.0;
    _LinearOut0[int2((int)id.x, (int)id.y)] = value;
}

void NcnnSentisExpandLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _SentisOutStorageW + (int)id.x;
    int total = NcnnSentisTotalCount(_SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }
    int ox, oy, oz, oc;
    NcnnSentisDecodeLinear(outLinear, _SentisOutDims, _SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC, ox, oy, oz, oc);
    _LinearOut0[int2((int)id.x, (int)id.y)] = NcnnSentisReadBroadcastIn0(ox, oy, oz, oc);
}

void NcnnSentisWhereLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _SentisOutStorageW + (int)id.x;
    int total = NcnnSentisTotalCount(_SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }
    int ox, oy, oz, oc;
    NcnnSentisDecodeLinear(outLinear, _SentisOutDims, _SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC, ox, oy, oz, oc);
    float cond = NcnnSentisReadBroadcastIn0(ox, oy, oz, oc);
    float a = NcnnSentisReadBroadcastIn1(ox, oy, oz, oc);
    float b = NcnnSentisReadBroadcastIn2(ox, oy, oz, oc);
    _LinearOut0[int2((int)id.x, (int)id.y)] = cond != 0.0 ? a : b;
}

void NcnnSentisGatherLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _SentisOutStorageW + (int)id.x;
    int total = NcnnSentisTotalCount(_SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }

    int axis = _SentisAxis < 0 ? _SentisAxis + _SentisIn0Dims : _SentisAxis;
    int ox, oy, oz, oc;
    NcnnSentisDecodeLinear(outLinear, _SentisOutDims, _SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC, ox, oy, oz, oc);

    int dx = 0, dy = 0, dz = 0, dc = 0;
    int ix = 0, iy = 0, iz = 0, ic = 0;
    [loop]
    for (int outAxis = 0; outAxis < _SentisOutDims; outAxis++)
    {
        int coord = NcnnSentisGetAxisCoord(_SentisOutDims, ox, oy, oz, oc, outAxis);
        if (outAxis < axis)
        {
            NcnnSentisSetAxisCoord(dx, dy, dz, dc, _SentisIn0Dims, outAxis, coord);
        }
        else if (outAxis < axis + _SentisIn1Dims)
        {
            NcnnSentisSetAxisCoord(ix, iy, iz, ic, _SentisIn1Dims, outAxis - axis, coord);
        }
        else
        {
            NcnnSentisSetAxisCoord(dx, dy, dz, dc, _SentisIn0Dims, outAxis - _SentisIn1Dims + 1, coord);
        }
    }

    int indexLinear = NcnnSentisLinearIndex(ix, iy, iz, ic, _SentisIn1W, _SentisIn1H, _SentisIn1D);
    int gathered = (int)round(NcnnSentisReadLinear(_LinearIn1, indexLinear, _SentisIn1StorageW, _SentisIn1StorageH));
    int axisSize = NcnnSentisAxisSize(_SentisIn0Dims, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0C, axis);
    if (gathered < 0)
        gathered += axisSize;
    NcnnSentisSetAxisCoord(dx, dy, dz, dc, _SentisIn0Dims, axis, clamp(gathered, 0, axisSize - 1));
    _LinearOut0[int2((int)id.x, (int)id.y)] = NcnnSentisReadAt(_LinearIn0, dx, dy, dz, dc, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0StorageW, _SentisIn0StorageH);
}

void NcnnSentisGatherElementsLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _SentisOutStorageW + (int)id.x;
    int total = NcnnSentisTotalCount(_SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }

    int axis = _SentisAxis < 0 ? _SentisAxis + _SentisIn0Dims : _SentisAxis;
    int ox, oy, oz, oc;
    NcnnSentisDecodeLinear(outLinear, _SentisOutDims, _SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC, ox, oy, oz, oc);
    int gathered = (int)round(NcnnSentisReadAt(_LinearIn1, ox, oy, oz, oc, _SentisIn1W, _SentisIn1H, _SentisIn1D, _SentisIn1StorageW, _SentisIn1StorageH));
    int axisSize = NcnnSentisAxisSize(_SentisIn0Dims, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0C, axis);
    if (gathered < 0)
        gathered += axisSize;

    int dx = ox, dy = oy, dz = oz, dc = oc;
    NcnnSentisSetAxisCoord(dx, dy, dz, dc, _SentisIn0Dims, axis, clamp(gathered, 0, axisSize - 1));
    _LinearOut0[int2((int)id.x, (int)id.y)] = NcnnSentisReadAt(_LinearIn0, dx, dy, dz, dc, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0StorageW, _SentisIn0StorageH);
}

void NcnnSentisArgReduceLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _SentisOutStorageW + (int)id.x;
    int total = NcnnSentisTotalCount(_SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }

    int axis = _SentisAxis < 0 ? _SentisAxis + _SentisIn0Dims : _SentisAxis;
    int axisSize = NcnnSentisAxisSize(_SentisIn0Dims, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0C, axis);
    int ox, oy, oz, oc;
    NcnnSentisDecodeLinear(outLinear, _SentisOutDims, _SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC, ox, oy, oz, oc);

    int bestIndex = 0;
    float bestValue = 0.0;
    [loop]
    for (int i = 0; i < axisSize; i++)
    {
        int ix = 0, iy = 0, iz = 0, ic = 0;
        [loop]
        for (int inAxis = 0; inAxis < _SentisIn0Dims; inAxis++)
        {
            int coord = i;
            if (inAxis != axis)
            {
                int outAxis = _SentisKeepDims != 0 ? inAxis : (inAxis < axis ? inAxis : inAxis - 1);
                coord = NcnnSentisGetAxisCoord(_SentisOutDims, ox, oy, oz, oc, outAxis);
            }
            NcnnSentisSetAxisCoord(ix, iy, iz, ic, _SentisIn0Dims, inAxis, coord);
        }

        float value = NcnnSentisReadAt(_LinearIn0, ix, iy, iz, ic, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0StorageW, _SentisIn0StorageH);
        if (i == 0
            || (_SentisMode != 0 && (value > bestValue || (_SentisSelectLast != 0 && value == bestValue)))
            || (_SentisMode == 0 && (value < bestValue || (_SentisSelectLast != 0 && value == bestValue))))
        {
            bestValue = value;
            bestIndex = i;
        }
    }

    _LinearOut0[int2((int)id.x, (int)id.y)] = (float)bestIndex;
}

float NcnnSentisTopKReadCandidate(int ox, int oy, int oz, int oc, int axis, int candidate)
{
    int ix = ox, iy = oy, iz = oz, ic = oc;
    NcnnSentisSetAxisCoord(ix, iy, iz, ic, _SentisIn0Dims, axis, candidate);
    return NcnnSentisReadAt(_LinearIn0, ix, iy, iz, ic, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0StorageW, _SentisIn0StorageH);
}

void NcnnSentisTopKLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _SentisOutStorageW + (int)id.x;
    int total = NcnnSentisTotalCount(_SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        if (_SentisHasIndices != 0)
            _LinearOut1[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }

    int axis = _SentisAxis < 0 ? _SentisAxis + _SentisIn0Dims : _SentisAxis;
    int axisSize = NcnnSentisAxisSize(_SentisIn0Dims, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0C, axis);
    int ox, oy, oz, oc;
    NcnnSentisDecodeLinear(outLinear, _SentisOutDims, _SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC, ox, oy, oz, oc);
    int rank = NcnnSentisGetAxisCoord(_SentisOutDims, ox, oy, oz, oc, axis);

    int bestIndex = 0;
    float bestValue = 0.0;
    bool found = false;
    [loop]
    for (int candidate = 0; candidate < axisSize; candidate++)
    {
        float value = NcnnSentisTopKReadCandidate(ox, oy, oz, oc, axis, candidate);
        int better = 0;
        [loop]
        for (int other = 0; other < axisSize; other++)
        {
            float otherValue = NcnnSentisTopKReadCandidate(ox, oy, oz, oc, axis, other);
            bool isBetter = _SentisMode != 0
                ? (otherValue > value || (otherValue == value && other < candidate))
                : (otherValue < value || (otherValue == value && other < candidate));
            if (isBetter)
                better++;
        }

        if (better == rank)
        {
            bestIndex = candidate;
            bestValue = value;
            found = true;
            break;
        }
    }

    if (!found)
        bestValue = NcnnSentisTopKReadCandidate(ox, oy, oz, oc, axis, bestIndex);

    _LinearOut0[int2((int)id.x, (int)id.y)] = bestValue;
    if (_SentisHasIndices != 0)
        _LinearOut1[int2((int)id.x, (int)id.y)] = (float)bestIndex;
}

void NcnnSentisOneHotLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _SentisOutStorageW + (int)id.x;
    int total = NcnnSentisTotalCount(_SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }

    int ox, oy, oz, oc;
    NcnnSentisDecodeLinear(outLinear, _SentisOutDims, _SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC, ox, oy, oz, oc);
    int depthCoord = NcnnSentisGetAxisCoord(_SentisOutDims, ox, oy, oz, oc, _SentisAxis);
    int ix = 0, iy = 0, iz = 0, ic = 0;
    [loop]
    for (int outAxis = 0; outAxis < _SentisOutDims; outAxis++)
    {
        if (outAxis == _SentisAxis)
            continue;
        int indexAxis = outAxis < _SentisAxis ? outAxis : outAxis - 1;
        int coord = NcnnSentisGetAxisCoord(_SentisOutDims, ox, oy, oz, oc, outAxis);
        NcnnSentisSetAxisCoord(ix, iy, iz, ic, _SentisIn0Dims, indexAxis, coord);
    }
    int indexValue = (int)round(NcnnSentisReadAt(_LinearIn0, ix, iy, iz, ic, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0StorageW, _SentisIn0StorageH));
    if (indexValue < 0)
        indexValue += max(1, _SentisK);
    _LinearOut0[int2((int)id.x, (int)id.y)] = indexValue == depthCoord ? _SentisValue1 : _SentisValue0;
}

void NcnnSentisCumSumLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _SentisOutStorageW + (int)id.x;
    int total = NcnnSentisTotalCount(_SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }

    int axis = _SentisAxis < 0 ? _SentisAxis + _SentisIn0Dims : _SentisAxis;
    int ox, oy, oz, oc;
    NcnnSentisDecodeLinear(outLinear, _SentisOutDims, _SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC, ox, oy, oz, oc);
    int current = NcnnSentisGetAxisCoord(_SentisOutDims, ox, oy, oz, oc, axis);
    int axisSize = NcnnSentisAxisSize(_SentisIn0Dims, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0C, axis);
    int begin = _SentisReverse != 0 ? current : 0;
    int end = _SentisReverse != 0 ? axisSize - 1 : current;
    if (_SentisExclusive != 0)
    {
        if (_SentisReverse != 0) begin = min(axisSize, begin + 1);
        else end = max(-1, end - 1);
    }

    float sum = 0.0;
    if (_SentisReverse != 0)
    {
        [loop]
        for (int i = begin; i <= end; i++)
        {
            int ix = ox, iy = oy, iz = oz, ic = oc;
            NcnnSentisSetAxisCoord(ix, iy, iz, ic, _SentisIn0Dims, axis, i);
            sum += NcnnSentisReadAt(_LinearIn0, ix, iy, iz, ic, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0StorageW, _SentisIn0StorageH);
        }
    }
    else
    {
        [loop]
        for (int i = begin; i <= end; i++)
        {
            int ix = ox, iy = oy, iz = oz, ic = oc;
            NcnnSentisSetAxisCoord(ix, iy, iz, ic, _SentisIn0Dims, axis, i);
            sum += NcnnSentisReadAt(_LinearIn0, ix, iy, iz, ic, _SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0StorageW, _SentisIn0StorageH);
        }
    }

    _LinearOut0[int2((int)id.x, (int)id.y)] = sum;
}

// Data-dependent ONNX nodes use a fixed LinearMat capacity. _LinearOut1[0]
// carries the actual count and stays GPU-resident; no CPU readback is involved.
void NcnnSentisNonZeroLinearMat_Impl(uint3 id)
{
    int sourceTotal = NcnnSentisTotalCount(_SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0C);
    int capacity = max(1, _SentisTotal);
    int count = 0;
    [loop] for (int i = 0; i < capacity; ++i) _LinearOut0[int2(i % _SentisOutStorageW, i / _SentisOutStorageW)] = -1.0;
    [loop] for (int i = 0; i < sourceTotal; ++i)
    {
        if (NcnnSentisReadLinear(_LinearIn0, i, _SentisIn0StorageW, _SentisIn0StorageH) != 0.0)
        {
            if (count < capacity) _LinearOut0[int2(count % _SentisOutStorageW, count / _SentisOutStorageW)] = (float)i;
            count++;
        }
    }
    _LinearOut1[int2(0, 0)] = (float)min(count, capacity);
}

void NcnnSentisCompressLinearMat_Impl(uint3 id)
{
    int sourceTotal = NcnnSentisTotalCount(_SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0C);
    int conditionTotal = NcnnSentisTotalCount(_SentisIn1W, _SentisIn1H, _SentisIn1D, _SentisIn1C);
    int capacity = max(1, _SentisTotal);
    int count = 0;
    [loop] for (int i = 0; i < capacity; ++i) _LinearOut0[int2(i % _SentisOutStorageW, i / _SentisOutStorageW)] = 0.0;
    [loop] for (int i = 0; i < min(sourceTotal, conditionTotal); ++i)
    {
        if (NcnnSentisReadLinear(_LinearIn1, i, _SentisIn1StorageW, _SentisIn1StorageH) != 0.0)
        {
            if (count < capacity) _LinearOut0[int2(count % _SentisOutStorageW, count / _SentisOutStorageW)] = NcnnSentisReadLinear(_LinearIn0, i, _SentisIn0StorageW, _SentisIn0StorageH);
            count++;
        }
    }
    _LinearOut1[int2(0, 0)] = (float)min(count, capacity);
}

void NcnnSentisGatherNDLinearMat_Impl(uint3 id)
{
    uint sw, sh; _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh) return;
    int outputIndex = (int)id.y * _SentisOutStorageW + (int)id.x;
    int total = NcnnSentisTotalCount(_SentisOutW, _SentisOutH, _SentisOutD, _SentisOutC);
    if (outputIndex >= total) { _LinearOut0[int2(id.xy)] = 0.0; return; }
    int sourceTotal = NcnnSentisTotalCount(_SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0C);
    int source = (int)round(NcnnSentisReadLinear(_LinearIn1, outputIndex, _SentisIn1StorageW, _SentisIn1StorageH));
    if (source < 0) source += sourceTotal;
    _LinearOut0[int2((int)id.x, (int)id.y)] = source >= 0 && source < sourceTotal ? NcnnSentisReadLinear(_LinearIn0, source, _SentisIn0StorageW, _SentisIn0StorageH) : 0.0;
}

void NcnnSentisScatterLinearMat_Impl(uint3 id)
{
    int total = NcnnSentisTotalCount(_SentisIn0W, _SentisIn0H, _SentisIn0D, _SentisIn0C);
    [loop] for (int i = 0; i < total; ++i) _LinearOut0[int2(i % _SentisOutStorageW, i / _SentisOutStorageW)] = NcnnSentisReadLinear(_LinearIn0, i, _SentisIn0StorageW, _SentisIn0StorageH);
    int updates = min(_SentisTotal, NcnnSentisTotalCount(_SentisIn2W, _SentisIn2H, _SentisIn2D, _SentisIn2C));
    [loop] for (int i = 0; i < updates; ++i)
    {
        int target = (int)round(NcnnSentisReadLinear(_LinearIn1, i, _SentisIn1StorageW, _SentisIn1StorageH));
        if (target < 0) target += total;
        if (target >= 0 && target < total) _LinearOut0[int2(target % _SentisOutStorageW, target / _SentisOutStorageW)] = NcnnSentisReadLinear(_LinearIn2, i, _SentisIn2StorageW, _SentisIn2StorageH);
    }
}
