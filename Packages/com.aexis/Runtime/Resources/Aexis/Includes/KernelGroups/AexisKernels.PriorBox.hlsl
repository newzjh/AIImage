// Texture-backed PriorBox generation for scalar linear-mat outputs.

float AexisPriorBoxCoord(float centerX, float centerY, float boxW, float boxH, int imageW, int imageH, int coord)
{
    float value;
    if (coord == 0)
        value = (centerX - boxW * 0.5) / max(1, imageW);
    else if (coord == 1)
        value = (centerY - boxH * 0.5) / max(1, imageH);
    else if (coord == 2)
        value = (centerX + boxW * 0.5) / max(1, imageW);
    else
        value = (centerY + boxH * 0.5) / max(1, imageH);
    return _PriorClip != 0 ? clamp(value, 0.0, 1.0) : value;
}

float AexisPriorBoxMxNetValue(int x)
{
    int featW = max(1, _PriorFeatW);
    int featH = max(1, _PriorFeatH);
    int numPrior = max(1, _PriorNumPrior);
    int boxValuesPerCell = max(1, numPrior * 4);
    int cell = x / boxValuesPerCell;
    int elem = x - cell * boxValuesPerCell;
    int boxIndex = elem / 4;
    int coord = elem - boxIndex * 4;
    int row = cell / featW;
    int col = cell - row * featW;
    if (row < 0 || row >= featH || _PriorNumMinSizes <= 0)
        return 0.0;

    float centerX = _PriorOffset * _PriorStepW + col * _PriorStepW;
    float centerY = _PriorOffset * _PriorStepH + row * _PriorStepH;
    float boxW;
    float boxH;
    if (boxIndex < _PriorNumMinSizes)
    {
        float size = _PriorMinSizes[boxIndex];
        boxW = size * featH / (float)featW;
        boxH = size;
    }
    else
    {
        int ratioIndex = boxIndex - _PriorNumMinSizes + 1;
        if (ratioIndex < 0 || ratioIndex >= _PriorNumAspectRatios)
            return 0.0;
        float baseSize = _PriorMinSizes[0];
        float ratio = sqrt(max(_PriorAspectRatios[ratioIndex], 1e-20));
        boxW = baseSize * featH / (float)featW * ratio;
        boxH = baseSize / ratio;
    }

    float value;
    if (coord == 0)
        value = centerX - boxW * 0.5;
    else if (coord == 1)
        value = centerY - boxH * 0.5;
    else if (coord == 2)
        value = centerX + boxW * 0.5;
    else
        value = centerY + boxH * 0.5;
    return _PriorClip != 0 ? clamp(value, 0.0, 1.0) : value;
}

float AexisPriorBoxCaffeValue(int x, int y)
{
    if (y == 1)
        return _PriorVariances[x - (x / 4) * 4];
    if (y != 0)
        return 0.0;

    int featW = max(1, _PriorFeatW);
    int featH = max(1, _PriorFeatH);
    int imageW = max(1, _PriorImageW);
    int imageH = max(1, _PriorImageH);
    int numPrior = max(1, _PriorNumPrior);
    int boxValuesPerCell = max(1, numPrior * 4);
    int cell = x / boxValuesPerCell;
    int elem = x - cell * boxValuesPerCell;
    int boxIndex = elem / 4;
    int coord = elem - boxIndex * 4;
    int row = cell / featW;
    int col = cell - row * featW;
    if (row < 0 || row >= featH || _PriorNumMinSizes <= 0)
        return 0.0;

    float centerX = _PriorOffset * _PriorStepW + col * _PriorStepW;
    float centerY = _PriorOffset * _PriorStepH + row * _PriorStepH;
    if (_PriorCenterMmdetection != 0)
    {
        centerX = _PriorOffset * (_PriorStepW - 1.0) + col * _PriorStepW;
        centerY = _PriorOffset * (_PriorStepH - 1.0) + row * _PriorStepH;
    }

    int emitted = 0;
    for (int k = 0; k < _PriorNumMinSizes; k++)
    {
        float minSize = _PriorMinSizes[k];
        if (boxIndex == emitted)
            return AexisPriorBoxCoord(centerX, centerY, minSize, minSize, imageW, imageH, coord);
        emitted++;

        if (_PriorNumMaxSizes > 0 && k < _PriorNumMaxSizes)
        {
            float boxSize = sqrt(max(minSize * _PriorMaxSizes[k], 0.0));
            if (boxIndex == emitted)
                return AexisPriorBoxCoord(centerX, centerY, boxSize, boxSize, imageW, imageH, coord);
            emitted++;
        }

        for (int p = 0; p < _PriorNumAspectRatios; p++)
        {
            float arSqrt = sqrt(max(_PriorAspectRatios[p], 1e-20));
            float boxW = minSize * arSqrt;
            float boxH = minSize / arSqrt;
            if (boxIndex == emitted)
                return AexisPriorBoxCoord(centerX, centerY, boxW, boxH, imageW, imageH, coord);
            emitted++;

            if (_PriorFlip != 0)
            {
                if (boxIndex == emitted)
                    return AexisPriorBoxCoord(centerX, centerY, boxH, boxW, imageW, imageH, coord);
                emitted++;
            }
        }
    }

    return 0.0;
}

void AexisPriorBox_Impl(uint3 id)
{
    uint ow, oh;
    _LinearOut0.GetDimensions(ow, oh);
    if (id.x >= ow || id.y >= oh)
        return;

    int x = (int)id.x;
    int y = (int)id.y;
    float value = _PriorMode == 0 ? AexisPriorBoxMxNetValue(x) : AexisPriorBoxCaffeValue(x, y);
    _LinearOut0[int2(x, y)] = value;
}
