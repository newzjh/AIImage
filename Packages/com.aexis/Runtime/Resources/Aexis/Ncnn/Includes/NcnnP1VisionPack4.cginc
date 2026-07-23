// Built-in P1 vision kernels. Activations and outputs stay in Pack4 texture arrays;
// the only buffers here are immutable DeformableConv2D weights and bias.

int P1PackCount(int channels)
{
    return max(1, (channels + 3) >> 2);
}

float P1Read(Texture2DArray<float4> tex, int x, int y, int z, int c, int w, int h, int d, int channels)
{
    if (x < 0 || y < 0 || z < 0 || c < 0 || x >= w || y >= h || z >= d || c >= channels)
        return 0.0;
    return NcnnReadPack4ChannelCDHW(tex, x, y, z, c, channels);
}

float P1Read0(int x, int y, int z, int c)
{
    return P1Read(_P1In0Arr, x, y, z, c, _P1In0W, _P1In0H, _P1In0D, _P1In0C);
}

float P1Read1(int x, int y, int z, int c)
{
    return P1Read(_P1In1Arr, x, y, z, c, _P1In1W, _P1In1H, _P1In1D, _P1In1C);
}

float P1Read2(int x, int y, int z, int c)
{
    return P1Read(_P1In2Arr, x, y, z, c, _P1In2W, _P1In2H, _P1In2D, _P1In2C);
}

float P1ReadLinear(Texture2DArray<float4> tex, int index, int dims, int w, int h, int d, int channels)
{
    if (index < 0)
        return 0.0;
    if (dims <= 1)
        return P1Read(tex, index, 0, 0, 0, w, 1, 1, 1);
    if (dims == 2)
    {
        int x = index % w;
        int y = index / w;
        return P1Read(tex, x, y, 0, 0, w, h, 1, 1);
    }
    int plane = w * h;
    if (dims == 3)
    {
        int c = index / plane;
        int spatial = index - c * plane;
        return P1Read(tex, spatial % w, spatial / w, 0, c, w, h, 1, channels);
    }
    int volume = plane * d;
    int c4 = index / volume;
    int rem = index - c4 * volume;
    int z4 = rem / plane;
    int spatial4 = rem - z4 * plane;
    return P1Read(tex, spatial4 % w, spatial4 / w, z4, c4, w, h, d, channels);
}

float P1Read0Linear(int index)
{
    return P1ReadLinear(_P1In0Arr, index, _P1In0Dims, _P1In0W, _P1In0H, _P1In0D, _P1In0C);
}

float P1Read1Linear(int index)
{
    return P1ReadLinear(_P1In1Arr, index, _P1In1Dims, _P1In1W, _P1In1H, _P1In1D, _P1In1C);
}

void P1DecodeOutput(uint3 id, out int x, out int y, out int z, out int pack)
{
    int packs = P1PackCount(_P1OutC);
    x = (int)id.x;
    y = (int)id.y;
    pack = (int)id.z % packs;
    z = (int)id.z / packs;
}

bool P1OutputInRange(uint3 id)
{
    return id.x < (uint)_P1OutW && id.y < (uint)_P1OutH && id.z < (uint)(_P1OutD * P1PackCount(_P1OutC));
}

float P1AdjustGridCoordinate(float v, int size)
{
    if (_P1PaddingMode == 2)
        return clamp(v, 0.0, (float)(size - 1));
    if (_P1PaddingMode == 3)
    {
        int high = _P1AlignCorners != 0 ? size - 1 : size;
        float reflected = (float)high - abs(abs(v + (_P1AlignCorners != 0 ? 0.0 : 0.5)) - (float)high);
        if (_P1AlignCorners == 0)
            reflected -= 0.5;
        return clamp(reflected, 0.0, (float)(size - 1));
    }
    return v;
}

float P1GridUnnormalize(float v, int size)
{
    return _P1AlignCorners != 0 ? (v + 1.0) * 0.5 * (size - 1) : ((v + 1.0) * size - 1.0) * 0.5;
}

float P1Sample2D(int channel, float fx, float fy)
{
    fx = P1AdjustGridCoordinate(fx, _P1In0W);
    fy = P1AdjustGridCoordinate(fy, _P1In0H);
    if (_P1Mode == 2)
        return P1Read0((int)floor(fx + 0.5), (int)floor(fy + 0.5), 0, channel);

    int x1 = (int)floor(fx);
    int y1 = (int)floor(fy);
    if (_P1Mode == 3)
    {
        float tx = fx - x1;
        float ty = fy - y1;
        float ax0 = -0.75 * (tx + 1.0) * (tx + 1.0) * (tx + 1.0) + 3.75 * (tx + 1.0) * (tx + 1.0) - 6.0 * (tx + 1.0) + 3.0;
        float ax1 = 1.25 * tx * tx * tx - 2.25 * tx * tx + 1.0;
        float ax2 = 1.25 * (1.0 - tx) * (1.0 - tx) * (1.0 - tx) - 2.25 * (1.0 - tx) * (1.0 - tx) + 1.0;
        float ax3 = 1.0 - ax0 - ax1 - ax2;
        float ay0 = -0.75 * (ty + 1.0) * (ty + 1.0) * (ty + 1.0) + 3.75 * (ty + 1.0) * (ty + 1.0) - 6.0 * (ty + 1.0) + 3.0;
        float ay1 = 1.25 * ty * ty * ty - 2.25 * ty * ty + 1.0;
        float ay2 = 1.25 * (1.0 - ty) * (1.0 - ty) * (1.0 - ty) - 2.25 * (1.0 - ty) * (1.0 - ty) + 1.0;
        float ay3 = 1.0 - ay0 - ay1 - ay2;
        float rows[4];
        [unroll] for (int ry = 0; ry < 4; ++ry)
        {
            int sy = y1 + ry - 1;
            rows[ry] = P1Read0((int)floor(P1AdjustGridCoordinate(x1 - 1, _P1In0W)), (int)floor(P1AdjustGridCoordinate(sy, _P1In0H)), 0, channel) * ax0
                + P1Read0((int)floor(P1AdjustGridCoordinate(x1, _P1In0W)), (int)floor(P1AdjustGridCoordinate(sy, _P1In0H)), 0, channel) * ax1
                + P1Read0((int)floor(P1AdjustGridCoordinate(x1 + 1, _P1In0W)), (int)floor(P1AdjustGridCoordinate(sy, _P1In0H)), 0, channel) * ax2
                + P1Read0((int)floor(P1AdjustGridCoordinate(x1 + 2, _P1In0W)), (int)floor(P1AdjustGridCoordinate(sy, _P1In0H)), 0, channel) * ax3;
        }
        return rows[0] * ay0 + rows[1] * ay1 + rows[2] * ay2 + rows[3] * ay3;
    }

    int x2 = x1 + 1;
    int y2 = y1 + 1;
    float ax = fx - x1;
    float ay = fy - y1;
    float v0 = P1Read0(x1, y1, 0, channel) * (1.0 - ax) + P1Read0(x2, y1, 0, channel) * ax;
    float v1 = P1Read0(x1, y2, 0, channel) * (1.0 - ax) + P1Read0(x2, y2, 0, channel) * ax;
    return v0 * (1.0 - ay) + v1 * ay;
}

float P1Sample3D(int channel, float fx, float fy, float fz)
{
    fx = P1AdjustGridCoordinate(fx, _P1In0W);
    fy = P1AdjustGridCoordinate(fy, _P1In0H);
    fz = P1AdjustGridCoordinate(fz, _P1In0D);
    if (_P1Mode == 2)
        return P1Read0((int)floor(fx + 0.5), (int)floor(fy + 0.5), (int)floor(fz + 0.5), channel);
    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    int z0 = (int)floor(fz);
    float ax = fx - x0;
    float ay = fy - y0;
    float az = fz - z0;
    float a = P1Read0(x0, y0, z0, channel) * (1.0 - ax) + P1Read0(x0 + 1, y0, z0, channel) * ax;
    float b = P1Read0(x0, y0 + 1, z0, channel) * (1.0 - ax) + P1Read0(x0 + 1, y0 + 1, z0, channel) * ax;
    float c = P1Read0(x0, y0, z0 + 1, channel) * (1.0 - ax) + P1Read0(x0 + 1, y0, z0 + 1, channel) * ax;
    float d = P1Read0(x0, y0 + 1, z0 + 1, channel) * (1.0 - ax) + P1Read0(x0 + 1, y0 + 1, z0 + 1, channel) * ax;
    return (a * (1.0 - ay) + b * ay) * (1.0 - az) + (c * (1.0 - ay) + d * ay) * az;
}

void NcnnP1GridSamplePack4_Impl(uint3 id)
{
    if (!P1OutputInRange(id)) return;
    int x, y, z, pack;
    P1DecodeOutput(id, x, y, z, pack);
    float gx;
    float gy;
    float gz = 0.0;
    if (_P1GridPermute != 0)
    {
        gx = P1Read1(x, y, z, 0);
        gy = P1Read1(x, y, z, 1);
        if (_P1OutDims == 4) gz = P1Read1(x, y, z, 2);
    }
    else
    {
        int baseIndex = (z * _P1OutH * _P1OutW + y * _P1OutW + x) * (_P1OutDims == 4 ? 3 : 2);
        gx = P1Read1Linear(baseIndex);
        gy = P1Read1Linear(baseIndex + 1);
        if (_P1OutDims == 4) gz = P1Read1Linear(baseIndex + 2);
    }
    gx = P1GridUnnormalize(gx, _P1In0W);
    gy = P1GridUnnormalize(gy, _P1In0H);
    if (_P1OutDims == 4) gz = P1GridUnnormalize(gz, _P1In0D);
    float4 dst = 0.0;
    [unroll] for (int lane = 0; lane < 4; ++lane)
    {
        int c = pack * 4 + lane;
        if (c < _P1OutC)
            NcnnWriteLane(dst, lane, _P1OutDims == 4 ? P1Sample3D(c, gx, gy, gz) : P1Sample2D(c, gx, gy));
    }
    _P1OutArr[int3(x, y, id.z)] = dst;
}

void NcnnP1GluPack4_Impl(uint3 id)
{
    if (!P1OutputInRange(id)) return;
    int x, y, z, pack;
    P1DecodeOutput(id, x, y, z, pack);
    float4 dst = 0.0;
    [unroll] for (int lane = 0; lane < 4; ++lane)
    {
        int c = pack * 4 + lane;
        if (c >= _P1OutC) continue;
        int sx = x;
        int sy = y;
        int sz = z;
        int sc = c;
        int offset = _P1Axis == 0 ? _P1OutC : (_P1Axis == 1 ? _P1OutD : (_P1Axis == 2 ? _P1OutH : _P1OutW));
        int gx = sx;
        int gy = sy;
        int gz = sz;
        int gc = sc;
        if (_P1Axis == 0) gc += offset;
        else if (_P1Axis == 1) gz += offset;
        else if (_P1Axis == 2) gy += offset;
        else gx += offset;
        float a = P1Read0(sx, sy, sz, sc);
        float b = P1Read0(gx, gy, gz, gc);
        NcnnWriteLane(dst, lane, a / (1.0 + exp(-b)));
    }
    _P1OutArr[int3(x, y, id.z)] = dst;
}

void NcnnP1DiagPack4_Impl(uint3 id)
{
    if (!P1OutputInRange(id)) return;
    int x, y, z, pack;
    P1DecodeOutput(id, x, y, z, pack);
    float4 dst = 0.0;
    if (_P1In0Dims <= 1)
    {
        if (x - y == _P1Diagonal)
        {
            int index = _P1Diagonal >= 0 ? y : x;
            if (index < _P1In0W) dst.x = P1Read0Linear(index);
        }
    }
    else if (_P1In0Dims == 2 && y == 0 && x < _P1OutW)
    {
        int row = _P1Diagonal < 0 ? x - _P1Diagonal : x;
        int col = _P1Diagonal > 0 ? x + _P1Diagonal : x;
        if (row >= 0 && col >= 0 && row < _P1In0H && col < _P1In0W)
            dst.x = P1Read0(col, row, 0, 0);
    }
    _P1OutArr[int3(x, y, id.z)] = dst;
}

void NcnnP1FoldPack4_Impl(uint3 id)
{
    if (!P1OutputInRange(id)) return;
    int x, y, z, pack;
    P1DecodeOutput(id, x, y, z, pack);
    int kernelArea = _P1KernelW * _P1KernelH;
    int paddedW = _P1OutW + _P1PadLeft + _P1PadRight;
    int paddedH = _P1OutH + _P1PadTop + _P1PadBottom;
    int extentW = _P1DilationW * (_P1KernelW - 1) + 1;
    int extentH = _P1DilationH * (_P1KernelH - 1) + 1;
    int inW = (paddedW - extentW) / _P1StrideW + 1;
    int inH = (paddedH - extentH) / _P1StrideH + 1;
    float4 dst = 0.0;
    [unroll] for (int lane = 0; lane < 4; ++lane)
    {
        int c = pack * 4 + lane;
        if (c >= _P1OutC) continue;
        float sum = 0.0;
        for (int ky = 0; ky < _P1KernelH; ++ky)
        for (int kx = 0; kx < _P1KernelW; ++kx)
        {
            int px = x + _P1PadLeft - kx * _P1DilationW;
            int py = y + _P1PadTop - ky * _P1DilationH;
            if (px < 0 || py < 0 || px % _P1StrideW != 0 || py % _P1StrideH != 0) continue;
            int ix = px / _P1StrideW;
            int iy = py / _P1StrideH;
            if (ix >= inW || iy >= inH) continue;
            int row = c * kernelArea + ky * _P1KernelW + kx;
            sum += P1Read0Linear(row * _P1In0W + iy * inW + ix);
        }
        NcnnWriteLane(dst, lane, sum);
    }
    _P1OutArr[int3(x, y, id.z)] = dst;
}

void NcnnP1SppPack4_Impl(uint3 id)
{
    if (!P1OutputInRange(id)) return;
    int x, y, z, pack;
    P1DecodeOutput(id, x, y, z, pack);
    int level = 0;
    int offset = 0;
    int bins = 1;
    while (level + 1 < _P1PyramidHeight && x >= offset + bins * bins)
    {
        offset += bins * bins;
        bins <<= 1;
        level++;
    }
    int local = x - offset;
    int bx = local % bins;
    int by = local / bins;
    int kernelW = (int)ceil((float)_P1In0W / bins);
    int kernelH = (int)ceil((float)_P1In0H / bins);
    int padW = (kernelW * bins - _P1In0W + 1) / 2;
    int padH = (kernelH * bins - _P1In0H + 1) / 2;
    int channel = y;
    float value = _P1PoolingType == 0 ? -3.402823e38 : 0.0;
    for (int ky = 0; ky < kernelH; ++ky)
    for (int kx = 0; kx < kernelW; ++kx)
    {
        float sample = P1Read0(bx * kernelW + kx - padW, by * kernelH + ky - padH, 0, channel);
        if (_P1PoolingType == 0) value = max(value, sample);
        else value += sample;
    }
    if (_P1PoolingType != 0) value /= (float)(kernelW * kernelH);
    _P1OutArr[int3(x, y, id.z)] = float4(value, 0.0, 0.0, 0.0);
}

float P1BilinearFeature(int channel, float x, float y)
{
    if (x < -1.0 || y < -1.0 || x > _P1In0W || y > _P1In0H) return 0.0;
    x = max(x, 0.0);
    y = max(y, 0.0);
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int x1 = min(x0 + 1, _P1In0W - 1);
    int y1 = min(y0 + 1, _P1In0H - 1);
    float dx = x - x0;
    float dy = y - y0;
    float v0 = P1Read0(x0, y0, 0, channel) * (1.0 - dx) + P1Read0(x1, y0, 0, channel) * dx;
    float v1 = P1Read0(x0, y1, 0, channel) * (1.0 - dx) + P1Read0(x1, y1, 0, channel) * dx;
    return v0 * (1.0 - dy) + v1 * dy;
}

void P1ReadRoi(out float x1, out float y1, out float x2, out float y2)
{
    x1 = P1Read1Linear(0) * _P1SpatialScale;
    y1 = P1Read1Linear(1) * _P1SpatialScale;
    x2 = P1Read1Linear(2) * _P1SpatialScale;
    y2 = P1Read1Linear(3) * _P1SpatialScale;
}

void NcnnP1RoiAlignPack4_Impl(uint3 id)
{
    if (!P1OutputInRange(id)) return;
    int x, y, z, pack;
    P1DecodeOutput(id, x, y, z, pack);
    float x1, y1, x2, y2;
    P1ReadRoi(x1, y1, x2, y2);
    if (_P1Aligned != 0) { x1 -= 0.5; y1 -= 0.5; x2 -= 0.5; y2 -= 0.5; }
    float rw = x2 - x1;
    float rh = y2 - y1;
    if (_P1Aligned == 0) { rw = max(rw, 1.0); rh = max(rh, 1.0); }
    float bw = rw / _P1PooledW;
    float bh = rh / _P1PooledH;
    int gridW = _P1SamplingRatio > 0 ? _P1SamplingRatio : max(1, (int)ceil(rw / _P1PooledW));
    int gridH = _P1SamplingRatio > 0 ? _P1SamplingRatio : max(1, (int)ceil(rh / _P1PooledH));
    float4 dst = 0.0;
    [unroll] for (int lane = 0; lane < 4; ++lane)
    {
        int c = pack * 4 + lane;
        if (c >= _P1OutC) continue;
        float sum = 0.0;
        for (int iy = 0; iy < gridH; ++iy)
        for (int ix = 0; ix < gridW; ++ix)
            sum += P1BilinearFeature(c, x1 + (x + (ix + 0.5) / gridW) * bw, y1 + (y + (iy + 0.5) / gridH) * bh);
        NcnnWriteLane(dst, lane, sum / (gridW * gridH));
    }
    _P1OutArr[int3(x, y, id.z)] = dst;
}

void NcnnP1RoiPoolPack4_Impl(uint3 id)
{
    if (!P1OutputInRange(id)) return;
    int x, y, z, pack;
    P1DecodeOutput(id, x, y, z, pack);
    float x1, y1, x2, y2;
    P1ReadRoi(x1, y1, x2, y2);
    int rx1 = (int)floor(x1 + 0.5);
    int ry1 = (int)floor(y1 + 0.5);
    int rx2 = (int)floor(x2 + 0.5);
    int ry2 = (int)floor(y2 + 0.5);
    float rw = max((float)(rx2 - rx1 + 1), 1.0);
    float rh = max((float)(ry2 - ry1 + 1), 1.0);
    int ws = clamp(rx1 + (int)floor(x * rw / _P1PooledW), 0, _P1In0W);
    int we = clamp(rx1 + (int)ceil((x + 1) * rw / _P1PooledW), 0, _P1In0W);
    int hs = clamp(ry1 + (int)floor(y * rh / _P1PooledH), 0, _P1In0H);
    int he = clamp(ry1 + (int)ceil((y + 1) * rh / _P1PooledH), 0, _P1In0H);
    float4 dst = 0.0;
    [unroll] for (int lane = 0; lane < 4; ++lane)
    {
        int c = pack * 4 + lane;
        if (c >= _P1OutC) continue;
        float value = hs >= he || ws >= we ? 0.0 : -3.402823e38;
        for (int py = hs; py < he; ++py)
        for (int px = ws; px < we; ++px)
            value = max(value, P1Read0(px, py, 0, c));
        NcnnWriteLane(dst, lane, value);
    }
    _P1OutArr[int3(x, y, id.z)] = dst;
}

void NcnnP1PsRoiPoolPack4_Impl(uint3 id)
{
    if (!P1OutputInRange(id)) return;
    int x, y, z, pack;
    P1DecodeOutput(id, x, y, z, pack);
    float x1, y1, x2, y2;
    P1ReadRoi(x1, y1, x2, y2);
    x1 = floor(x1 + 0.5) * _P1SpatialScale;
    y1 = floor(y1 + 0.5) * _P1SpatialScale;
    x2 = floor(x2 + 1.5) * _P1SpatialScale;
    y2 = floor(y2 + 1.5) * _P1SpatialScale;
    float rw = max(x2 - x1, 0.1);
    float rh = max(y2 - y1, 0.1);
    int ws = clamp((int)floor(x1 + x * rw / _P1PooledW), 0, _P1In0W);
    int we = clamp((int)ceil(x1 + (x + 1) * rw / _P1PooledW), 0, _P1In0W);
    int hs = clamp((int)floor(y1 + y * rh / _P1PooledH), 0, _P1In0H);
    int he = clamp((int)ceil(y1 + (y + 1) * rh / _P1PooledH), 0, _P1In0H);
    float4 dst = 0.0;
    [unroll] for (int lane = 0; lane < 4; ++lane)
    {
        int c = pack * 4 + lane;
        if (c >= _P1OutC) continue;
        int sourceChannel = (c * _P1PooledH + y) * _P1PooledW + x;
        float sum = 0.0;
        for (int py = hs; py < he; ++py)
        for (int px = ws; px < we; ++px)
            sum += P1Read0(px, py, 0, sourceChannel);
        int area = (he - hs) * (we - ws);
        NcnnWriteLane(dst, lane, area > 0 ? sum / area : 0.0);
    }
    _P1OutArr[int3(x, y, id.z)] = dst;
}

float P1DeformableSample(int channel, float x, float y)
{
    if (x <= -1.0 || y <= -1.0 || x >= _P1In0W || y >= _P1In0H) return 0.0;
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    float dx = x - x0;
    float dy = y - y0;
    return P1Read0(x0, y0, 0, channel) * (1.0 - dx) * (1.0 - dy)
        + P1Read0(x0 + 1, y0, 0, channel) * dx * (1.0 - dy)
        + P1Read0(x0, y0 + 1, 0, channel) * (1.0 - dx) * dy
        + P1Read0(x0 + 1, y0 + 1, 0, channel) * dx * dy;
}

float P1ApplyActivation(float v)
{
    if (_P1ActivationType == 1) return max(v, 0.0);
    if (_P1ActivationType == 2) return v >= 0.0 ? v : v * _P1ActivationSlope;
    if (_P1ActivationType == 4) return 1.0 / (1.0 + exp(-v));
    return v;
}

void NcnnP1DeformableConv2dPack4_Impl(uint3 id)
{
    if (!P1OutputInRange(id)) return;
    int x, y, z, pack;
    P1DecodeOutput(id, x, y, z, pack);
    float4 dst = 0.0;
    [unroll] for (int lane = 0; lane < 4; ++lane)
    {
        int oc = pack * 4 + lane;
        if (oc >= _P1OutC) continue;
        float sum = _P1BiasTerm != 0 ? _P1Bias[oc] : 0.0;
        for (int ky = 0; ky < _P1KernelH; ++ky)
        for (int kx = 0; kx < _P1KernelW; ++kx)
        {
            int kernelIndex = ky * _P1KernelW + kx;
            float offY = P1Read1(x, y, 0, kernelIndex * 2);
            float offX = P1Read1(x, y, 0, kernelIndex * 2 + 1);
            float mask = _P1In2C > 0 ? P1Read2(x, y, 0, kernelIndex) : 1.0;
            float sy = y * _P1StrideH - _P1PadTop + ky * _P1DilationH + offY;
            float sx = x * _P1StrideW - _P1PadLeft + kx * _P1DilationW + offX;
            for (int ic = 0; ic < _P1In0C; ++ic)
            {
                int weightIndex = ((oc * _P1In0C + ic) * _P1KernelH + ky) * _P1KernelW + kx;
                sum += P1DeformableSample(ic, sx, sy) * mask * _P1Weights[weightIndex];
            }
        }
        NcnnWriteLane(dst, lane, P1ApplyActivation(sum));
    }
    _P1OutArr[int3(x, y, id.z)] = dst;
}

float P1OutputScalar(int x, int y)
{
    return _P1OutArr.Load(int4(x, y, 0, 0)).x;
}

void P1WriteOutputScalar(int x, int y, float value)
{
    _P1OutArr[int3(x, y, 0)] = float4(value, 0.0, 0.0, 0.0);
}

void P1ClearDetections()
{
    for (int row = 0; row < _P1DetectionCapacity; ++row)
    for (int column = 0; column < 6; ++column)
        P1WriteOutputScalar(column, row, 0.0);
}

void P1ClearProposals()
{
    for (int row = 0; row < _P1DetectionCapacity; ++row)
    for (int column = 0; column < 4; ++column)
        P1WriteOutputScalar(column, row, 0.0);
}

float P1Iou(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2)
{
    float iw = max(0.0, min(ax2, bx2) - max(ax1, bx1));
    float ih = max(0.0, min(ay2, by2) - max(ay1, by1));
    float inter = iw * ih;
    float areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1);
    float areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1);
    return inter / max(areaA + areaB - inter, 1e-20);
}

bool P1SuppressedByOutput(float x1, float y1, float x2, float y2, int label, int count, bool classAware)
{
    for (int slot = 0; slot < count; ++slot)
    {
        float selectedScore = P1OutputScalar(1, slot);
        if (selectedScore <= 0.0) continue;
        int selectedLabel = (int)P1OutputScalar(0, slot);
        if (classAware && selectedLabel != label) continue;
        if (P1Iou(x1, y1, x2, y2, P1OutputScalar(2, slot), P1OutputScalar(3, slot), P1OutputScalar(4, slot), P1OutputScalar(5, slot)) > _P1NmsThreshold)
            return true;
    }
    return false;
}

void P1WriteDetection(int slot, int label, float score, float x1, float y1, float x2, float y2)
{
    P1WriteOutputScalar(0, slot, (float)label);
    P1WriteOutputScalar(1, slot, score);
    P1WriteOutputScalar(2, slot, x1);
    P1WriteOutputScalar(3, slot, y1);
    P1WriteOutputScalar(4, slot, x2);
    P1WriteOutputScalar(5, slot, y2);
}

bool P1ProposalCandidate(int candidate, out float score, out float x1, out float y1, out float x2, out float y2)
{
    int anchors = 9;
    int plane = _P1In0W * _P1In0H;
    int anchor = candidate / plane;
    int position = candidate - anchor * plane;
    int x = position % _P1In0W;
    int y = position / _P1In0W;
    score = P1Read0(x, y, 0, anchor + anchors);
    float ratio = anchor / 3 == 0 ? 0.5 : (anchor / 3 == 1 ? 1.0 : 2.0);
    float scale = anchor % 3 == 0 ? 8.0 : (anchor % 3 == 1 ? 16.0 : 32.0);
    float rw = round(_P1BaseSize / sqrt(ratio));
    float rh = round(rw * ratio);
    float aw = rw * scale;
    float ah = rh * scale;
    float ax = _P1BaseSize * 0.5 + x * _P1FeatStride;
    float ay = _P1BaseSize * 0.5 + y * _P1FeatStride;
    float dx = P1Read1(x, y, 0, anchor * 4);
    float dy = P1Read1(x, y, 0, anchor * 4 + 1);
    float dw = P1Read1(x, y, 0, anchor * 4 + 2);
    float dh = P1Read1(x, y, 0, anchor * 4 + 3);
    float cx = ax + aw * 0.5 + aw * dx;
    float cy = ay + ah * 0.5 + ah * dy;
    float bw = aw * exp(dw);
    float bh = ah * exp(dh);
    float imH = P1Read2Linear(0);
    float imW = P1Read2Linear(1);
    float imScale = P1Read2Linear(2);
    x1 = clamp(cx - bw * 0.5, 0.0, imW - 1.0);
    y1 = clamp(cy - bh * 0.5, 0.0, imH - 1.0);
    x2 = clamp(cx + bw * 0.5, 0.0, imW - 1.0);
    y2 = clamp(cy + bh * 0.5, 0.0, imH - 1.0);
    return x2 - x1 + 1.0 >= _P1MinSize * imScale && y2 - y1 + 1.0 >= _P1MinSize * imScale;
}

void NcnnP1ProposalPack4_Impl(uint3 id)
{
    if (any(id != uint3(0, 0, 0))) return;
    P1ClearProposals();
    int candidateCount = _P1In0W * _P1In0H * 9;
    for (int slot = 0; slot < _P1DetectionCapacity; ++slot)
    {
        float best = -3.402823e38;
        float bx1 = 0.0, by1 = 0.0, bx2 = 0.0, by2 = 0.0;
        for (int candidate = 0; candidate < candidateCount; ++candidate)
        {
            float score, x1, y1, x2, y2;
            if (!P1ProposalCandidate(candidate, score, x1, y1, x2, y2) || score <= best) continue;
            if (!P1SuppressedByOutput(x1, y1, x2, y2, 0, slot, false))
            {
                best = score; bx1 = x1; by1 = y1; bx2 = x2; by2 = y2;
            }
        }
        if (best <= -3.0e38) break;
        P1WriteOutputScalar(0, slot, bx1);
        P1WriteOutputScalar(1, slot, by1);
        P1WriteOutputScalar(2, slot, bx2);
        P1WriteOutputScalar(3, slot, by2);
    }
}

bool P1SsdCandidate(int candidate, out int label, out float score, out float x1, out float y1, out float x2, out float y2)
{
    int priors = (_P1In0W * _P1In0H * _P1In0D * max(1, _P1In0C)) / 4;
    label = candidate / priors + 1;
    int prior = candidate - (label - 1) * priors;
    score = P1Read1Linear(prior * _P1NumClasses + label);
    float px1 = P1Read2Linear(prior * 4);
    float py1 = P1Read2Linear(prior * 4 + 1);
    float px2 = P1Read2Linear(prior * 4 + 2);
    float py2 = P1Read2Linear(prior * 4 + 3);
    float vx = _P1Variance0;
    float vy = _P1Variance1;
    float vw = _P1Variance2;
    float vh = _P1Variance3;
    int priorElements = priors * 4;
    if (_P1In2W * _P1In2H * _P1In2D * max(1, _P1In2C) >= priorElements * 2)
    {
        vx = P1Read2Linear(priorElements + prior * 4);
        vy = P1Read2Linear(priorElements + prior * 4 + 1);
        vw = P1Read2Linear(priorElements + prior * 4 + 2);
        vh = P1Read2Linear(priorElements + prior * 4 + 3);
    }
    float pw = px2 - px1;
    float ph = py2 - py1;
    float pcx = (px1 + px2) * 0.5;
    float pcy = (py1 + py2) * 0.5;
    float cx = vx * P1Read0Linear(prior * 4) * pw + pcx;
    float cy = vy * P1Read0Linear(prior * 4 + 1) * ph + pcy;
    float bw = exp(vw * P1Read0Linear(prior * 4 + 2)) * pw;
    float bh = exp(vh * P1Read0Linear(prior * 4 + 3)) * ph;
    x1 = cx - bw * 0.5; y1 = cy - bh * 0.5; x2 = cx + bw * 0.5; y2 = cy + bh * 0.5;
    return score > _P1ConfidenceThreshold;
}

void NcnnP1DetectionOutputPack4_Impl(uint3 id)
{
    if (any(id != uint3(0, 0, 0))) return;
    P1ClearDetections();
    int priors = (_P1In0W * _P1In0H * _P1In0D * max(1, _P1In0C)) / 4;
    int candidateCount = priors * max(0, _P1NumClasses - 1);
    for (int slot = 0; slot < _P1DetectionCapacity; ++slot)
    {
        float best = -3.402823e38;
        int bestLabel = 0;
        float bx1 = 0.0, by1 = 0.0, bx2 = 0.0, by2 = 0.0;
        for (int candidate = 0; candidate < candidateCount; ++candidate)
        {
            int label; float score, x1, y1, x2, y2;
            if (!P1SsdCandidate(candidate, label, score, x1, y1, x2, y2) || score <= best) continue;
            if (!P1SuppressedByOutput(x1, y1, x2, y2, label, slot, true))
            {
                best = score; bestLabel = label; bx1 = x1; by1 = y1; bx2 = x2; by2 = y2;
            }
        }
        if (best <= -3.0e38) break;
        P1WriteDetection(slot, bestLabel, best, bx1, by1, bx2, by2);
    }
}

float P1YoloRead(int which, int x, int y, int channel)
{
    return which == 0 ? P1Read0(x, y, 0, channel) : (which == 1 ? P1Read1(x, y, 0, channel) : P1Read2(x, y, 0, channel));
}

void P1YoloShape(int which, out int w, out int h, out int c)
{
    if (which == 0) { w = _P1In0W; h = _P1In0H; c = _P1In0C; }
    else if (which == 1) { w = _P1In1W; h = _P1In1H; c = _P1In1C; }
    else { w = _P1In2W; h = _P1In2H; c = _P1In2C; }
}

bool P1YoloCandidate(int candidate, out int label, out float score, out float x1, out float y1, out float x2, out float y2)
{
    int which = 0;
    int w0 = _P1In0W * _P1In0H * _P1NumBoxes;
    int w1 = _P1In1C > 0 ? _P1In1W * _P1In1H * _P1NumBoxes : 0;
    if (candidate >= w0) { which = 1; candidate -= w0; }
    if (candidate >= w1 && _P1In2C > 0) { which = 2; candidate -= w1; }
    int w, h, channels;
    P1YoloShape(which, w, h, channels);
    int plane = w * h;
    int anchor = candidate / plane;
    int pos = candidate - anchor * plane;
    int x = pos % w;
    int y = pos / w;
    int stride = 5 + _P1NumClasses;
    int baseChannel = anchor * stride;
    float clsMax = -3.402823e38;
    float clsSum = 0.0;
    label = 0;
    for (int c = 0; c < _P1NumClasses; ++c)
    {
        float raw = P1YoloRead(which, x, y, baseChannel + 5 + c);
        if (raw > clsMax) { clsMax = raw; label = c; }
    }
    if (_P1YoloV3 == 0)
    {
        for (int c = 0; c < _P1NumClasses; ++c) clsSum += exp(P1YoloRead(which, x, y, baseChannel + 5 + c) - clsMax);
        clsMax = exp(clsMax) / max(clsSum, 1e-20);
    }
    else clsMax = 1.0 / (1.0 + exp(-clsMax));
    float objectness = 1.0 / (1.0 + exp(-P1YoloRead(which, x, y, baseChannel + 4)));
    score = objectness * clsMax;
    float biasW = _P1DetectionBiases[(which * _P1NumBoxes + anchor) * 2];
    float biasH = _P1DetectionBiases[(which * _P1NumBoxes + anchor) * 2 + 1];
    float cx = (x + 1.0 / (1.0 + exp(-P1YoloRead(which, x, y, baseChannel)))) / w;
    float cy = (y + 1.0 / (1.0 + exp(-P1YoloRead(which, x, y, baseChannel + 1)))) / h;
    float bw = exp(P1YoloRead(which, x, y, baseChannel + 2)) * biasW / w;
    float bh = exp(P1YoloRead(which, x, y, baseChannel + 3)) * biasH / h;
    x1 = cx - bw * 0.5; y1 = cy - bh * 0.5; x2 = cx + bw * 0.5; y2 = cy + bh * 0.5;
    return score >= _P1ConfidenceThreshold;
}

void NcnnP1YoloDetectionOutputPack4_Impl(uint3 id)
{
    if (any(id != uint3(0, 0, 0))) return;
    P1ClearDetections();
    int candidates = _P1In0W * _P1In0H * _P1NumBoxes;
    if (_P1In1C > 0) candidates += _P1In1W * _P1In1H * _P1NumBoxes;
    if (_P1In2C > 0) candidates += _P1In2W * _P1In2H * _P1NumBoxes;
    for (int slot = 0; slot < _P1DetectionCapacity; ++slot)
    {
        float best = -3.402823e38;
        int bestLabel = 0;
        float bx1 = 0.0, by1 = 0.0, bx2 = 0.0, by2 = 0.0;
        for (int candidate = 0; candidate < candidates; ++candidate)
        {
            int label; float score, x1, y1, x2, y2;
            if (!P1YoloCandidate(candidate, label, score, x1, y1, x2, y2) || score <= best) continue;
            if (!P1SuppressedByOutput(x1, y1, x2, y2, label, slot, false))
            {
                best = score; bestLabel = label; bx1 = x1; by1 = y1; bx2 = x2; by2 = y2;
            }
        }
        if (best <= -3.0e38) break;
        P1WriteDetection(slot, bestLabel + 1, best, bx1, by1, bx2, by2);
    }
}

int P1EinsumVectorValue(int kind, int index)
{
    if (kind == 0)
    {
        if (index == 0) return _P1EinsumDim0; if (index == 1) return _P1EinsumDim1; if (index == 2) return _P1EinsumDim2; if (index == 3) return _P1EinsumDim3;
        if (index == 4) return _P1EinsumDim4; if (index == 5) return _P1EinsumDim5; if (index == 6) return _P1EinsumDim6; return _P1EinsumDim7;
    }
    if (kind == 1)
    {
        if (index == 0) return _P1EinsumA0; if (index == 1) return _P1EinsumA1; if (index == 2) return _P1EinsumA2; return _P1EinsumA3;
    }
    if (kind == 2)
    {
        if (index == 0) return _P1EinsumB0; if (index == 1) return _P1EinsumB1; if (index == 2) return _P1EinsumB2; return _P1EinsumB3;
    }
    if (kind == 3)
    {
        if (index == 0) return _P1EinsumC0; if (index == 1) return _P1EinsumC1; if (index == 2) return _P1EinsumC2; return _P1EinsumC3;
    }
    if (kind == 4)
    {
        if (index == 0) return _P1EinsumO0; if (index == 1) return _P1EinsumO1; if (index == 2) return _P1EinsumO2; return _P1EinsumO3;
    }
    if (index == 0) return _P1EinsumR0; if (index == 1) return _P1EinsumR1; if (index == 2) return _P1EinsumR2; return _P1EinsumR3;
}

int P1EinsumCoord(int coords[8], int label)
{
    return label < 0 ? 0 : coords[label];
}

float P1EinsumReadOperand(int operand, int coords[8])
{
    int c = P1EinsumCoord(coords, P1EinsumVectorValue(operand + 1, 0));
    int d = P1EinsumCoord(coords, P1EinsumVectorValue(operand + 1, 1));
    int h = P1EinsumCoord(coords, P1EinsumVectorValue(operand + 1, 2));
    int w = P1EinsumCoord(coords, P1EinsumVectorValue(operand + 1, 3));
    int lc = P1EinsumVectorValue(operand + 1, 0);
    int ld = P1EinsumVectorValue(operand + 1, 1);
    int lh = P1EinsumVectorValue(operand + 1, 2);
    int lw = P1EinsumVectorValue(operand + 1, 3);
    if (lc >= 0 && P1EinsumVectorValue(0, lc) == 1) c = 0;
    if (ld >= 0 && P1EinsumVectorValue(0, ld) == 1) d = 0;
    if (lh >= 0 && P1EinsumVectorValue(0, lh) == 1) h = 0;
    if (lw >= 0 && P1EinsumVectorValue(0, lw) == 1) w = 0;
    return operand == 0 ? P1Read0(w, h, d, c) : (operand == 1 ? P1Read1(w, h, d, c) : P1Read2(w, h, d, c));
}

void NcnnP1EinsumPack4_Impl(uint3 id)
{
    if (!P1OutputInRange(id)) return;
    int x, y, z, pack;
    P1DecodeOutput(id, x, y, z, pack);
    float4 dst = 0.0;
    [unroll] for (int lane = 0; lane < 4; ++lane)
    {
        int channel = pack * 4 + lane;
        if (channel >= _P1OutC) continue;
        int coords[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        int outputCoordinates[4] = { channel, z, y, x };
        [unroll] for (int axis = 0; axis < 4; ++axis)
        {
            int label = P1EinsumVectorValue(4, axis);
            if (label >= 0) coords[label] = outputCoordinates[axis];
        }
        int reductions = 1;
        for (int r = 0; r < _P1EinsumReductionCount; ++r)
            reductions *= P1EinsumVectorValue(0, P1EinsumVectorValue(5, r));
        float sum = 0.0;
        for (int flat = 0; flat < reductions; ++flat)
        {
            int value = flat;
            for (int r = 0; r < _P1EinsumReductionCount; ++r)
            {
                int label = P1EinsumVectorValue(5, r);
                int extent = P1EinsumVectorValue(0, label);
                coords[label] = value % extent;
                value /= extent;
            }
            float term = P1EinsumReadOperand(0, coords) * P1EinsumReadOperand(1, coords);
            if (_P1EinsumOperandCount == 3) term *= P1EinsumReadOperand(2, coords);
            sum += term;
        }
        NcnnWriteLane(dst, lane, sum);
    }
    _P1OutArr[int3(x, y, id.z)] = dst;
}
