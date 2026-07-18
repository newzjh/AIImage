Texture2DArray<float4> _DeepFillFeatureArr;
Texture2DArray<float4> _DeepFillMaskArr;
Texture2DArray<float4> _DeepFillPatchStatsArr;
Texture2DArray<float4> _DeepFillScoresArr;
Texture2DArray<float4> _DeepFillWeightsArr;
RWTexture2DArray<float4> _DeepFillPatchStatsOutArr;
RWTexture2DArray<float4> _DeepFillScoresOutArr;
RWTexture2DArray<float4> _DeepFillWeightsOutArr;
RWTexture2DArray<float4> _DeepFillOutputArr;

int _DeepFillFeatureW;
int _DeepFillFeatureH;
int _DeepFillFeatureChannels;
int _DeepFillFeaturePacks;
int _DeepFillMatchW;
int _DeepFillMatchH;
int _DeepFillSourceCount;
int _DeepFillSourcePacks;
int _DeepFillMaskW;
int _DeepFillMaskH;
int _DeepFillMaskDownsample;
float _DeepFillPatchEpsilon;
float _DeepFillSoftmaxScale;

groupshared float _DeepFillReduce[128];

float4 NcnnDeepFillReadFeature(int x, int y, int pack)
{
    if (x < 0 || y < 0 || x >= _DeepFillFeatureW || y >= _DeepFillFeatureH
        || pack < 0 || pack >= _DeepFillFeaturePacks)
        return 0.0;
    return _DeepFillFeatureArr[int3(x, y, pack)];
}

float NcnnDeepFillReadMaskDownsampled(int x, int y)
{
    if (x < 0 || y < 0 || x >= _DeepFillMatchW || y >= _DeepFillMatchH)
        return 0.0;
    int mx = x * _DeepFillMaskDownsample;
    int my = y * _DeepFillMaskDownsample;
    if (mx < 0 || my < 0 || mx >= _DeepFillMaskW || my >= _DeepFillMaskH)
        return 0.0;
    return _DeepFillMaskArr[int3(mx, my, 0)].x;
}

float NcnnDeepFillPatchValid(int sourceX, int sourceY)
{
    [unroll]
    for (int ky = -1; ky <= 1; ky++)
    {
        [unroll]
        for (int kx = -1; kx <= 1; kx++)
        {
            if (NcnnDeepFillReadMaskDownsampled(sourceX + kx, sourceY + ky) > 0.5)
                return 0.0;
        }
    }
    return 1.0;
}

void NcnnDeepFillV2PatchStats_Impl(uint3 id)
{
    int sourceX = (int)id.x;
    int sourceY = (int)id.y;
    if (sourceX >= _DeepFillMatchW || sourceY >= _DeepFillMatchH)
        return;

    float sumSq = _DeepFillPatchEpsilon * (float)(_DeepFillFeatureChannels * 9);
    [unroll]
    for (int ky = -1; ky <= 1; ky++)
    {
        [unroll]
        for (int kx = -1; kx <= 1; kx++)
        {
            int fx = (sourceX + kx) * 2;
            int fy = (sourceY + ky) * 2;
            for (int pack = 0; pack < _DeepFillFeaturePacks; pack++)
            {
                float4 value = NcnnDeepFillReadFeature(fx, fy, pack);
                sumSq += dot(value, value);
            }
        }
    }

    float norm = sqrt(max(sumSq, 1.0e-20));
    _DeepFillPatchStatsOutArr[int3(sourceX, sourceY, 0)] = float4(norm, NcnnDeepFillPatchValid(sourceX, sourceY), 0.0, 0.0);
}

float NcnnDeepFillPatchScore(int targetX, int targetY, int sourceIndex)
{
    if (sourceIndex < 0 || sourceIndex >= _DeepFillSourceCount)
        return 0.0;
    int sourceX = sourceIndex % _DeepFillMatchW;
    int sourceY = sourceIndex / _DeepFillMatchW;
    float4 stats = _DeepFillPatchStatsArr[int3(sourceX, sourceY, 0)];
    if (stats.y <= 0.0)
        return 0.0;

    float sum = 0.0;
    [unroll]
    for (int ky = -1; ky <= 1; ky++)
    {
        [unroll]
        for (int kx = -1; kx <= 1; kx++)
        {
            int targetFeatureX = (targetX + kx) * 2;
            int targetFeatureY = (targetY + ky) * 2;
            int sourceFeatureX = (sourceX + kx) * 2;
            int sourceFeatureY = (sourceY + ky) * 2;
            for (int pack = 0; pack < _DeepFillFeaturePacks; pack++)
            {
                float4 targetValue = NcnnDeepFillReadFeature(targetFeatureX, targetFeatureY, pack);
                float4 sourceValue = NcnnDeepFillReadFeature(sourceFeatureX, sourceFeatureY, pack);
                sum += dot(targetValue, sourceValue);
            }
        }
    }
    return sum / stats.x;
}

void NcnnDeepFillV2Scores_Impl(uint3 id)
{
    int targetX = (int)id.x;
    int targetY = (int)id.y;
    int sourcePack = (int)id.z;
    if (targetX >= _DeepFillMatchW || targetY >= _DeepFillMatchH || sourcePack >= _DeepFillSourcePacks)
        return;

    int sourceBase = sourcePack * 4;
    float4 score;
    score.x = NcnnDeepFillPatchScore(targetX, targetY, sourceBase + 0);
    score.y = NcnnDeepFillPatchScore(targetX, targetY, sourceBase + 1);
    score.z = NcnnDeepFillPatchScore(targetX, targetY, sourceBase + 2);
    score.w = NcnnDeepFillPatchScore(targetX, targetY, sourceBase + 3);
    _DeepFillScoresOutArr[int3(targetX, targetY, sourcePack)] = score;
}

float4 NcnnDeepFillSourceValid4(int sourcePack)
{
    int sourceBase = sourcePack * 4;
    float4 valid = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int sourceIndex = sourceBase + lane;
        if (sourceIndex >= _DeepFillSourceCount)
            continue;
        int sourceX = sourceIndex % _DeepFillMatchW;
        int sourceY = sourceIndex / _DeepFillMatchW;
        NcnnWriteLane(valid, lane, _DeepFillPatchStatsArr[int3(sourceX, sourceY, 0)].y);
    }
    return valid;
}

void NcnnDeepFillV2Softmax_Impl(uint3 groupId, uint3 groupThreadId)
{
    int targetIndex = (int)groupId.x;
    int lane = (int)groupThreadId.x;
    if (targetIndex >= _DeepFillMatchW * _DeepFillMatchH)
        return;
    int targetX = targetIndex % _DeepFillMatchW;
    int targetY = targetIndex / _DeepFillMatchW;

    float localMax = -3.402823466e+38;
    for (int sourcePack = lane; sourcePack < _DeepFillSourcePacks; sourcePack += 128)
    {
        float4 score = _DeepFillScoresArr[int3(targetX, targetY, sourcePack)] * _DeepFillSoftmaxScale;
        localMax = max(localMax, max(max(score.x, score.y), max(score.z, score.w)));
    }
    _DeepFillReduce[lane] = localMax;
    GroupMemoryBarrierWithGroupSync();
    for (int step = 64; step > 0; step >>= 1)
    {
        if (lane < step)
            _DeepFillReduce[lane] = max(_DeepFillReduce[lane], _DeepFillReduce[lane + step]);
        GroupMemoryBarrierWithGroupSync();
    }
    float maxValue = _DeepFillReduce[0];

    float localSum = 0.0;
    for (int sourcePack = lane; sourcePack < _DeepFillSourcePacks; sourcePack += 128)
    {
        float4 score = _DeepFillScoresArr[int3(targetX, targetY, sourcePack)] * _DeepFillSoftmaxScale;
        float4 exponent = exp(score - maxValue);
        localSum += exponent.x + exponent.y + exponent.z + exponent.w;
    }
    _DeepFillReduce[lane] = localSum;
    GroupMemoryBarrierWithGroupSync();
    for (int step = 64; step > 0; step >>= 1)
    {
        if (lane < step)
            _DeepFillReduce[lane] += _DeepFillReduce[lane + step];
        GroupMemoryBarrierWithGroupSync();
    }
    float invSum = rcp(max(_DeepFillReduce[0], 1.0e-20));

    for (int sourcePack = lane; sourcePack < _DeepFillSourcePacks; sourcePack += 128)
    {
        float4 score = _DeepFillScoresArr[int3(targetX, targetY, sourcePack)] * _DeepFillSoftmaxScale;
        float4 weight = exp(score - maxValue) * invSum;
        weight *= NcnnDeepFillSourceValid4(sourcePack);
        _DeepFillWeightsOutArr[int3(targetX, targetY, sourcePack)] = weight;
    }
}

void NcnnDeepFillV2Reconstruct_Impl(uint3 id)
{
    int outputX = (int)id.x;
    int outputY = (int)id.y;
    int outputPack = (int)id.z;
    if (outputX >= _DeepFillFeatureW || outputY >= _DeepFillFeatureH || outputPack >= _DeepFillFeaturePacks)
        return;

    float4 sum = 0.0;
    [unroll]
    for (int ky = 0; ky < 4; ky++)
    {
        int targetNumeratorY = outputY + 1 - ky;
        if ((targetNumeratorY & 1) != 0)
            continue;
        int targetY = targetNumeratorY / 2;
        if (targetY < 0 || targetY >= _DeepFillMatchH)
            continue;

        [unroll]
        for (int kx = 0; kx < 4; kx++)
        {
            int targetNumeratorX = outputX + 1 - kx;
            if ((targetNumeratorX & 1) != 0)
                continue;
            int targetX = targetNumeratorX / 2;
            if (targetX < 0 || targetX >= _DeepFillMatchW)
                continue;

            for (int sourcePack = 0; sourcePack < _DeepFillSourcePacks; sourcePack++)
            {
                float4 weights = _DeepFillWeightsArr[int3(targetX, targetY, sourcePack)];
                int sourceBase = sourcePack * 4;
                [unroll]
                for (int lane = 0; lane < 4; lane++)
                {
                    int sourceIndex = sourceBase + lane;
                    if (sourceIndex >= _DeepFillSourceCount)
                        continue;
                    int sourceX = sourceIndex % _DeepFillMatchW;
                    int sourceY = sourceIndex / _DeepFillMatchW;
                    int featureX = sourceX * 2 + kx - 1;
                    int featureY = sourceY * 2 + ky - 1;
                    sum += NcnnReadLane(weights, lane) * NcnnDeepFillReadFeature(featureX, featureY, outputPack);
                }
            }
        }
    }
    _DeepFillOutputArr[int3(outputX, outputY, outputPack)] = sum * 0.25;
}
