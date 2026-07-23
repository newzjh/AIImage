// Texture-backed Aexis-style shape/index operators.

int AexisTotalCount(int w, int h, int d, int c)
{
    return max(1, w) * max(1, h) * max(1, d) * max(1, c);
}

int AexisLinearIndex(int x, int y, int z, int c, int w, int h, int d)
{
    return (((c * max(1, d) + z) * max(1, h) + y) * max(1, w)) + x;
}

void AexisDecodeLinear(int linearIndex, int dims, int w, int h, int d, int c, out int x, out int y, out int z, out int ch)
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

int AexisAxisSize(int dims, int w, int h, int d, int c, int axis)
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

int AexisGetAxisCoord(int dims, int x, int y, int z, int ch, int axis)
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

void AexisSetAxisCoord(inout int x, inout int y, inout int z, inout int ch, int dims, int axis, int value)
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

float AexisReadLinear(Texture2D<float> tex, int linearIndex, int storageW, int storageH)
{
    int sw = max(1, storageW);
    int x = linearIndex % sw;
    int y = linearIndex / sw;
    if (y < 0 || y >= max(1, storageH))
        return 0.0;
    return tex[int2(x, y)];
}

float AexisReadAt(Texture2D<float> tex, int x, int y, int z, int ch, int w, int h, int d, int storageW, int storageH)
{
    int linearIndex = AexisLinearIndex(x, y, z, ch, w, h, d);
    return AexisReadLinear(tex, linearIndex, storageW, storageH);
}

float AexisReadBroadcastIn0(int ox, int oy, int oz, int oc)
{
    int ix = 0, iy = 0, iz = 0, ic = 0;
    int shift = _AexisOutDims - _AexisIn0Dims;
    [loop]
    for (int outAxis = 0; outAxis < _AexisOutDims; outAxis++)
    {
        int inAxis = outAxis - shift;
        if (inAxis < 0)
            continue;
        int coord = AexisGetAxisCoord(_AexisOutDims, ox, oy, oz, oc, outAxis);
        int size = AexisAxisSize(_AexisIn0Dims, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0C, inAxis);
        if (size == 1)
            coord = 0;
        AexisSetAxisCoord(ix, iy, iz, ic, _AexisIn0Dims, inAxis, clamp(coord, 0, size - 1));
    }
    return AexisReadAt(_LinearIn0, ix, iy, iz, ic, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0StorageW, _AexisIn0StorageH);
}

float AexisReadBroadcastIn1(int ox, int oy, int oz, int oc)
{
    int ix = 0, iy = 0, iz = 0, ic = 0;
    int shift = _AexisOutDims - _AexisIn1Dims;
    [loop]
    for (int outAxis = 0; outAxis < _AexisOutDims; outAxis++)
    {
        int inAxis = outAxis - shift;
        if (inAxis < 0)
            continue;
        int coord = AexisGetAxisCoord(_AexisOutDims, ox, oy, oz, oc, outAxis);
        int size = AexisAxisSize(_AexisIn1Dims, _AexisIn1W, _AexisIn1H, _AexisIn1D, _AexisIn1C, inAxis);
        if (size == 1)
            coord = 0;
        AexisSetAxisCoord(ix, iy, iz, ic, _AexisIn1Dims, inAxis, clamp(coord, 0, size - 1));
    }
    return AexisReadAt(_LinearIn1, ix, iy, iz, ic, _AexisIn1W, _AexisIn1H, _AexisIn1D, _AexisIn1StorageW, _AexisIn1StorageH);
}

float AexisReadBroadcastIn2(int ox, int oy, int oz, int oc)
{
    int ix = 0, iy = 0, iz = 0, ic = 0;
    int shift = _AexisOutDims - _AexisIn2Dims;
    [loop]
    for (int outAxis = 0; outAxis < _AexisOutDims; outAxis++)
    {
        int inAxis = outAxis - shift;
        if (inAxis < 0)
            continue;
        int coord = AexisGetAxisCoord(_AexisOutDims, ox, oy, oz, oc, outAxis);
        int size = AexisAxisSize(_AexisIn2Dims, _AexisIn2W, _AexisIn2H, _AexisIn2D, _AexisIn2C, inAxis);
        if (size == 1)
            coord = 0;
        AexisSetAxisCoord(ix, iy, iz, ic, _AexisIn2Dims, inAxis, clamp(coord, 0, size - 1));
    }
    return AexisReadAt(_LinearIn2, ix, iy, iz, ic, _AexisIn2W, _AexisIn2H, _AexisIn2D, _AexisIn2StorageW, _AexisIn2StorageH);
}

void AexisConstantLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int linearIndex = (int)id.y * (int)sw + (int)id.x;
    _LinearOut0[int2((int)id.x, (int)id.y)] = linearIndex < _AexisTotal ? _AexisValue0 : 0.0;
}

void AexisRangeLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int linearIndex = (int)id.y * (int)sw + (int)id.x;
    float value = linearIndex < _AexisTotal ? (_AexisValue0 + _AexisValue1 * (float)linearIndex) : 0.0;
    _LinearOut0[int2((int)id.x, (int)id.y)] = value;
}

void AexisExpandLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _AexisOutStorageW + (int)id.x;
    int total = AexisTotalCount(_AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }
    int ox, oy, oz, oc;
    AexisDecodeLinear(outLinear, _AexisOutDims, _AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC, ox, oy, oz, oc);
    _LinearOut0[int2((int)id.x, (int)id.y)] = AexisReadBroadcastIn0(ox, oy, oz, oc);
}

void AexisWhereLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _AexisOutStorageW + (int)id.x;
    int total = AexisTotalCount(_AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }
    int ox, oy, oz, oc;
    AexisDecodeLinear(outLinear, _AexisOutDims, _AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC, ox, oy, oz, oc);
    float cond = AexisReadBroadcastIn0(ox, oy, oz, oc);
    float a = AexisReadBroadcastIn1(ox, oy, oz, oc);
    float b = AexisReadBroadcastIn2(ox, oy, oz, oc);
    _LinearOut0[int2((int)id.x, (int)id.y)] = cond != 0.0 ? a : b;
}

void AexisGatherLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _AexisOutStorageW + (int)id.x;
    int total = AexisTotalCount(_AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }

    int axis = _AexisAxis < 0 ? _AexisAxis + _AexisIn0Dims : _AexisAxis;
    int ox, oy, oz, oc;
    AexisDecodeLinear(outLinear, _AexisOutDims, _AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC, ox, oy, oz, oc);

    int dx = 0, dy = 0, dz = 0, dc = 0;
    int ix = 0, iy = 0, iz = 0, ic = 0;
    [loop]
    for (int outAxis = 0; outAxis < _AexisOutDims; outAxis++)
    {
        int coord = AexisGetAxisCoord(_AexisOutDims, ox, oy, oz, oc, outAxis);
        if (outAxis < axis)
        {
            AexisSetAxisCoord(dx, dy, dz, dc, _AexisIn0Dims, outAxis, coord);
        }
        else if (outAxis < axis + _AexisIn1Dims)
        {
            AexisSetAxisCoord(ix, iy, iz, ic, _AexisIn1Dims, outAxis - axis, coord);
        }
        else
        {
            AexisSetAxisCoord(dx, dy, dz, dc, _AexisIn0Dims, outAxis - _AexisIn1Dims + 1, coord);
        }
    }

    int indexLinear = AexisLinearIndex(ix, iy, iz, ic, _AexisIn1W, _AexisIn1H, _AexisIn1D);
    int gathered = (int)round(AexisReadLinear(_LinearIn1, indexLinear, _AexisIn1StorageW, _AexisIn1StorageH));
    int axisSize = AexisAxisSize(_AexisIn0Dims, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0C, axis);
    if (gathered < 0)
        gathered += axisSize;
    AexisSetAxisCoord(dx, dy, dz, dc, _AexisIn0Dims, axis, clamp(gathered, 0, axisSize - 1));
    _LinearOut0[int2((int)id.x, (int)id.y)] = AexisReadAt(_LinearIn0, dx, dy, dz, dc, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0StorageW, _AexisIn0StorageH);
}

void AexisGatherElementsLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _AexisOutStorageW + (int)id.x;
    int total = AexisTotalCount(_AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }

    int axis = _AexisAxis < 0 ? _AexisAxis + _AexisIn0Dims : _AexisAxis;
    int ox, oy, oz, oc;
    AexisDecodeLinear(outLinear, _AexisOutDims, _AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC, ox, oy, oz, oc);
    int gathered = (int)round(AexisReadAt(_LinearIn1, ox, oy, oz, oc, _AexisIn1W, _AexisIn1H, _AexisIn1D, _AexisIn1StorageW, _AexisIn1StorageH));
    int axisSize = AexisAxisSize(_AexisIn0Dims, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0C, axis);
    if (gathered < 0)
        gathered += axisSize;

    int dx = ox, dy = oy, dz = oz, dc = oc;
    AexisSetAxisCoord(dx, dy, dz, dc, _AexisIn0Dims, axis, clamp(gathered, 0, axisSize - 1));
    _LinearOut0[int2((int)id.x, (int)id.y)] = AexisReadAt(_LinearIn0, dx, dy, dz, dc, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0StorageW, _AexisIn0StorageH);
}

void AexisArgReduceLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _AexisOutStorageW + (int)id.x;
    int total = AexisTotalCount(_AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }

    int axis = _AexisAxis < 0 ? _AexisAxis + _AexisIn0Dims : _AexisAxis;
    int axisSize = AexisAxisSize(_AexisIn0Dims, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0C, axis);
    int ox, oy, oz, oc;
    AexisDecodeLinear(outLinear, _AexisOutDims, _AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC, ox, oy, oz, oc);

    int bestIndex = 0;
    float bestValue = 0.0;
    [loop]
    for (int i = 0; i < axisSize; i++)
    {
        int ix = 0, iy = 0, iz = 0, ic = 0;
        [loop]
        for (int inAxis = 0; inAxis < _AexisIn0Dims; inAxis++)
        {
            int coord = i;
            if (inAxis != axis)
            {
                int outAxis = _AexisKeepDims != 0 ? inAxis : (inAxis < axis ? inAxis : inAxis - 1);
                coord = AexisGetAxisCoord(_AexisOutDims, ox, oy, oz, oc, outAxis);
            }
            AexisSetAxisCoord(ix, iy, iz, ic, _AexisIn0Dims, inAxis, coord);
        }

        float value = AexisReadAt(_LinearIn0, ix, iy, iz, ic, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0StorageW, _AexisIn0StorageH);
        if (i == 0
            || (_AexisMode != 0 && (value > bestValue || (_AexisSelectLast != 0 && value == bestValue)))
            || (_AexisMode == 0 && (value < bestValue || (_AexisSelectLast != 0 && value == bestValue))))
        {
            bestValue = value;
            bestIndex = i;
        }
    }

    _LinearOut0[int2((int)id.x, (int)id.y)] = (float)bestIndex;
}

float AexisTopKReadCandidate(int ox, int oy, int oz, int oc, int axis, int candidate)
{
    int ix = ox, iy = oy, iz = oz, ic = oc;
    AexisSetAxisCoord(ix, iy, iz, ic, _AexisIn0Dims, axis, candidate);
    return AexisReadAt(_LinearIn0, ix, iy, iz, ic, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0StorageW, _AexisIn0StorageH);
}

void AexisTopKLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _AexisOutStorageW + (int)id.x;
    int total = AexisTotalCount(_AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        if (_AexisHasIndices != 0)
            _LinearOut1[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }

    int axis = _AexisAxis < 0 ? _AexisAxis + _AexisIn0Dims : _AexisAxis;
    int axisSize = AexisAxisSize(_AexisIn0Dims, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0C, axis);
    int ox, oy, oz, oc;
    AexisDecodeLinear(outLinear, _AexisOutDims, _AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC, ox, oy, oz, oc);
    int rank = AexisGetAxisCoord(_AexisOutDims, ox, oy, oz, oc, axis);

    int bestIndex = 0;
    float bestValue = 0.0;
    bool found = false;
    [loop]
    for (int candidate = 0; candidate < axisSize; candidate++)
    {
        float value = AexisTopKReadCandidate(ox, oy, oz, oc, axis, candidate);
        int better = 0;
        [loop]
        for (int other = 0; other < axisSize; other++)
        {
            float otherValue = AexisTopKReadCandidate(ox, oy, oz, oc, axis, other);
            bool isBetter = _AexisMode != 0
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
        bestValue = AexisTopKReadCandidate(ox, oy, oz, oc, axis, bestIndex);

    _LinearOut0[int2((int)id.x, (int)id.y)] = bestValue;
    if (_AexisHasIndices != 0)
        _LinearOut1[int2((int)id.x, (int)id.y)] = (float)bestIndex;
}

void AexisOneHotLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _AexisOutStorageW + (int)id.x;
    int total = AexisTotalCount(_AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }

    int ox, oy, oz, oc;
    AexisDecodeLinear(outLinear, _AexisOutDims, _AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC, ox, oy, oz, oc);
    int depthCoord = AexisGetAxisCoord(_AexisOutDims, ox, oy, oz, oc, _AexisAxis);
    int ix = 0, iy = 0, iz = 0, ic = 0;
    [loop]
    for (int outAxis = 0; outAxis < _AexisOutDims; outAxis++)
    {
        if (outAxis == _AexisAxis)
            continue;
        int indexAxis = outAxis < _AexisAxis ? outAxis : outAxis - 1;
        int coord = AexisGetAxisCoord(_AexisOutDims, ox, oy, oz, oc, outAxis);
        AexisSetAxisCoord(ix, iy, iz, ic, _AexisIn0Dims, indexAxis, coord);
    }
    int indexValue = (int)round(AexisReadAt(_LinearIn0, ix, iy, iz, ic, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0StorageW, _AexisIn0StorageH));
    if (indexValue < 0)
        indexValue += max(1, _AexisK);
    _LinearOut0[int2((int)id.x, (int)id.y)] = indexValue == depthCoord ? _AexisValue1 : _AexisValue0;
}

void AexisCumSumLinearMat_Impl(uint3 id)
{
    uint sw, sh;
    _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh)
        return;
    int outLinear = (int)id.y * _AexisOutStorageW + (int)id.x;
    int total = AexisTotalCount(_AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC);
    if (outLinear >= total)
    {
        _LinearOut0[int2((int)id.x, (int)id.y)] = 0.0;
        return;
    }

    int axis = _AexisAxis < 0 ? _AexisAxis + _AexisIn0Dims : _AexisAxis;
    int ox, oy, oz, oc;
    AexisDecodeLinear(outLinear, _AexisOutDims, _AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC, ox, oy, oz, oc);
    int current = AexisGetAxisCoord(_AexisOutDims, ox, oy, oz, oc, axis);
    int axisSize = AexisAxisSize(_AexisIn0Dims, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0C, axis);
    int begin = _AexisReverse != 0 ? current : 0;
    int end = _AexisReverse != 0 ? axisSize - 1 : current;
    if (_AexisExclusive != 0)
    {
        if (_AexisReverse != 0) begin = min(axisSize, begin + 1);
        else end = max(-1, end - 1);
    }

    float sum = 0.0;
    if (_AexisReverse != 0)
    {
        [loop]
        for (int i = begin; i <= end; i++)
        {
            int ix = ox, iy = oy, iz = oz, ic = oc;
            AexisSetAxisCoord(ix, iy, iz, ic, _AexisIn0Dims, axis, i);
            sum += AexisReadAt(_LinearIn0, ix, iy, iz, ic, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0StorageW, _AexisIn0StorageH);
        }
    }
    else
    {
        [loop]
        for (int i = begin; i <= end; i++)
        {
            int ix = ox, iy = oy, iz = oz, ic = oc;
            AexisSetAxisCoord(ix, iy, iz, ic, _AexisIn0Dims, axis, i);
            sum += AexisReadAt(_LinearIn0, ix, iy, iz, ic, _AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0StorageW, _AexisIn0StorageH);
        }
    }

    _LinearOut0[int2((int)id.x, (int)id.y)] = sum;
}

// Data-dependent ONNX nodes use a fixed LinearMat capacity. _LinearOut1[0]
// carries the actual count and stays GPU-resident; no CPU readback is involved.
void AexisNonZeroLinearMat_Impl(uint3 id)
{
    int sourceTotal = AexisTotalCount(_AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0C);
    int capacity = max(1, _AexisTotal);
    int count = 0;
    [loop] for (int i = 0; i < capacity; ++i) _LinearOut0[int2(i % _AexisOutStorageW, i / _AexisOutStorageW)] = -1.0;
    [loop] for (int i = 0; i < sourceTotal; ++i)
    {
        if (AexisReadLinear(_LinearIn0, i, _AexisIn0StorageW, _AexisIn0StorageH) != 0.0)
        {
            if (count < capacity) _LinearOut0[int2(count % _AexisOutStorageW, count / _AexisOutStorageW)] = (float)i;
            count++;
        }
    }
    _LinearOut1[int2(0, 0)] = (float)min(count, capacity);
}

void AexisCompressLinearMat_Impl(uint3 id)
{
    int sourceTotal = AexisTotalCount(_AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0C);
    int conditionTotal = AexisTotalCount(_AexisIn1W, _AexisIn1H, _AexisIn1D, _AexisIn1C);
    int capacity = max(1, _AexisTotal);
    int count = 0;
    [loop] for (int i = 0; i < capacity; ++i) _LinearOut0[int2(i % _AexisOutStorageW, i / _AexisOutStorageW)] = 0.0;
    [loop] for (int i = 0; i < min(sourceTotal, conditionTotal); ++i)
    {
        if (AexisReadLinear(_LinearIn1, i, _AexisIn1StorageW, _AexisIn1StorageH) != 0.0)
        {
            if (count < capacity) _LinearOut0[int2(count % _AexisOutStorageW, count / _AexisOutStorageW)] = AexisReadLinear(_LinearIn0, i, _AexisIn0StorageW, _AexisIn0StorageH);
            count++;
        }
    }
    _LinearOut1[int2(0, 0)] = (float)min(count, capacity);
}

void AexisGatherNDLinearMat_Impl(uint3 id)
{
    uint sw, sh; _LinearOut0.GetDimensions(sw, sh);
    if (id.x >= sw || id.y >= sh) return;
    int outputIndex = (int)id.y * _AexisOutStorageW + (int)id.x;
    int total = AexisTotalCount(_AexisOutW, _AexisOutH, _AexisOutD, _AexisOutC);
    if (outputIndex >= total) { _LinearOut0[int2(id.xy)] = 0.0; return; }
    int sourceTotal = AexisTotalCount(_AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0C);
    int source = (int)round(AexisReadLinear(_LinearIn1, outputIndex, _AexisIn1StorageW, _AexisIn1StorageH));
    if (source < 0) source += sourceTotal;
    _LinearOut0[int2((int)id.x, (int)id.y)] = source >= 0 && source < sourceTotal ? AexisReadLinear(_LinearIn0, source, _AexisIn0StorageW, _AexisIn0StorageH) : 0.0;
}

void AexisScatterLinearMat_Impl(uint3 id)
{
    int total = AexisTotalCount(_AexisIn0W, _AexisIn0H, _AexisIn0D, _AexisIn0C);
    [loop] for (int i = 0; i < total; ++i) _LinearOut0[int2(i % _AexisOutStorageW, i / _AexisOutStorageW)] = AexisReadLinear(_LinearIn0, i, _AexisIn0StorageW, _AexisIn0StorageH);
    int updates = min(_AexisTotal, AexisTotalCount(_AexisIn2W, _AexisIn2H, _AexisIn2D, _AexisIn2C));
    [loop] for (int i = 0; i < updates; ++i)
    {
        int target = (int)round(AexisReadLinear(_LinearIn1, i, _AexisIn1StorageW, _AexisIn1StorageH));
        if (target < 0) target += total;
        if (target >= 0 && target < total) _LinearOut0[int2(target % _AexisOutStorageW, target / _AexisOutStorageW)] = AexisReadLinear(_LinearIn2, i, _AexisIn2StorageW, _AexisIn2StorageH);
    }
}
