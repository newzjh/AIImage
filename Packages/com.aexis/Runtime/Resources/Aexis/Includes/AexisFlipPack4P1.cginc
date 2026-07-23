void AexisFlipPack4P1_Impl(uint3 id)
{
    if (id.x >= (uint)_P1Width || id.y >= (uint)_P1Height || id.z >= (uint)(_P1Depth * _P1Packs))
        return;

    const int pack = (int)id.z % _P1Packs;
    const int outputDepth = (int)id.z / _P1Packs;
    const int sourceX = _P1FlipWidth != 0 ? _P1Width - 1 - (int)id.x : (int)id.x;
    const int sourceY = _P1FlipHeight != 0 ? _P1Height - 1 - (int)id.y : (int)id.y;
    const int sourceDepth = _P1FlipDepth != 0 ? _P1Depth - 1 - outputDepth : outputDepth;

    float4 outputValue = 0.f;
    for (int lane = 0; lane < 4; lane++)
    {
        const int outputChannel = pack * 4 + lane;
        if (outputChannel >= _P1Channels)
            continue;
        const int sourceChannel = _P1FlipChannels != 0 ? _P1Channels - 1 - outputChannel : outputChannel;
        AexisWriteLane(outputValue, lane, AexisReadPack4ChannelCDHW(_AexisInArr, sourceX, sourceY, sourceDepth, sourceChannel, _P1Channels));
    }
    _AexisOutArr[int3(id.x, id.y, id.z)] = outputValue;
}
