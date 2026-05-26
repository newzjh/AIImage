using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;
using UnityEngine.Rendering;

public sealed class RealEsrganNcnnReproRunner : MonoBehaviour
{
    public string modelName = "realesrgan-x4plus";
    public int tileSize = 128;
    public int tilePad = 10;
    public int maxInputLongSide = 2048;
    public bool enableTileProbe = false;
    public bool enableSeamProbe = false;
    public bool useCommandBuffer = false;
    public bool enableWinograd23 = false;

    public event Action<float, string> ProgressChanged;

    private NcnnParamModel _model;
    private readonly Dictionary<string, ConvPack> _conv = new Dictionary<string, ConvPack>(StringComparer.Ordinal);
    private Dictionary<string, int> _blobUseCount;
    private NcnnOps _ops;
    private NcnnBufferPool _bufferPool;
    private bool _loaded;
    private bool _useCmdThisRun;

    private readonly struct RtKey : IEquatable<RtKey>
    {
        public readonly int w;
        public readonly int h;
        public readonly int d;
        public readonly RenderTextureFormat format;

        public RtKey(int w, int h, int d, RenderTextureFormat format)
        {
            this.w = w;
            this.h = h;
            this.d = d;
            this.format = format;
        }

        public bool Equals(RtKey other) => w == other.w && h == other.h && d == other.d && format == other.format;
        public override bool Equals(object obj) => obj is RtKey other && Equals(other);
        public override int GetHashCode()
        {
            unchecked
            {
                var hash = w;
                hash = (hash * 397) ^ h;
                hash = (hash * 397) ^ d;
                hash = (hash * 397) ^ (int)format;
                return hash;
            }
        }
    }

    private sealed class ConvPack : IDisposable
    {
        public int outC;
        public int inC;
        public int kernel;
        public int stride;
        public int pad;
        public int biasTerm;
        public int weightSize;
        public int activationType;
        public float activationSlope;
        public int inPacks;
        public int outPacks;
        public ComputeBuffer wOihw;
        public ComputeBuffer bOihw;
        public ComputeBuffer w4;
        public ComputeBuffer b4;

        public void Dispose()
        {
            try { wOihw?.Dispose(); } catch { }
            try { bOihw?.Dispose(); } catch { }
            try { w4?.Dispose(); } catch { }
            try { b4?.Dispose(); } catch { }
        }
    }

    private sealed class TensorRef
    {
        public NcnnTensorBuffer t;
        public int w;
        public int h;
        public int c;
        public int refs;
        public bool owned;
    }


    private void Awake()
    {
        _ops = new NcnnOps();
        _bufferPool = new NcnnBufferPool(maxFreeBuffers: 32, sizeCompareRatio: 0.875f);
    }

    private void OnDestroy()
    {
        foreach (var kv in _conv)
            kv.Value?.Dispose();
        _conv.Clear();

        _loaded = false;
        _model = null;
        _blobUseCount = null;
        try { _ops?.ReleaseWinogradWorkspace(); } catch { }
        _bufferPool?.Dispose();
        _bufferPool = null;
    }

    public async UniTask<RealEsrganResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (src == null)
            return default;

        EnsureLoaded();

        var originalW = src.width;
        var originalH = src.height;
        var totalSw = Stopwatch.StartNew();

        RealEsrganResult Finish(RealEsrganResult r)
        {
            r.elapsedMs = totalSw.ElapsedMilliseconds;
            try
            {
                UnityEngine.Debug.Log("[TIMING] ESRGAN(repro) " + r.elapsedMs + " ms | in=" + originalW + "x" + originalH + " | model=" + (modelName ?? "") + " | err=" + (r.error ?? ""));
            }
            catch
            {
            }
            return r;
        }
        var runFactor = 4;
        var limit = Mathf.Max(256, maxInputLongSide);
        var maxSide = Mathf.Max(originalW, originalH);
        var runInW = originalW;
        var runInH = originalH;
        if (maxSide > limit)
        {
            var s = (float)limit / maxSide;
            runInW = Mathf.Max(1, Mathf.RoundToInt(originalW * s));
            runInH = Mathf.Max(1, Mathf.RoundToInt(originalH * s));
        }
        var baseTileSize = tileSize > 0 ? tileSize : Mathf.Max(runInW, runInH);
        var effectiveTilePad = tilePad > 0 ? tilePad : 10;

        ReportProgress(0f, "准备输入");
        await UniTask.Yield();

        var isVulkan = SystemInfo.graphicsDeviceType == GraphicsDeviceType.Vulkan;
        _useCmdThisRun = useCommandBuffer && IsComputeCmdSupported(SystemInfo.graphicsDeviceType);
        var attemptTiles = new[]
        {
            Mathf.Max(16, baseTileSize),
            Mathf.Max(16, baseTileSize / 2),
            Mathf.Max(16, baseTileSize / 4)
        };

        Exception lastErr = null;
        for (var attempt = 0; attempt < attemptTiles.Length; attempt++)
        {
            var effectiveTileSize = attemptTiles[attempt];
            try
            {
                var r = await ProcessOnceAsync(src, ct, originalW, originalH, runInW, runInH, runFactor, effectiveTileSize, effectiveTilePad);
                return Finish(r);
            }
            catch (Exception e)
            {
                lastErr = e;
                if (!IsLikelyVulkanOom(e))
                    break;
                await UniTask.Yield();
            }
        }
        return Finish(new RealEsrganResult { error = lastErr != null ? lastErr.Message : "unknown error" });
    }

    public async UniTask<RealEsrganResult> ProcessAsyncNv(Texture2D src, CancellationToken ct)
    {
        if (src == null)
            return default;

        EnsureLoaded();

        var originalW = src.width;
        var originalH = src.height;
        var totalSw = Stopwatch.StartNew();

        RealEsrganResult Finish(RealEsrganResult r)
        {
            r.elapsedMs = totalSw.ElapsedMilliseconds;
            try
            {
                UnityEngine.Debug.Log("[TIMING] ESRGAN(repro-ncnn) " + r.elapsedMs + " ms | in=" + originalW + "x" + originalH + " | model=" + (modelName ?? "") + " | err=" + (r.error ?? ""));
            }
            catch
            {
            }
            return r;
        }
        var runFactor = 4;
        var limit = Mathf.Max(256, maxInputLongSide);
        var maxSide = Mathf.Max(originalW, originalH);
        var runInW = originalW;
        var runInH = originalH;
        if (maxSide > limit)
        {
            var s = (float)limit / maxSide;
            runInW = Mathf.Max(1, Mathf.RoundToInt(originalW * s));
            runInH = Mathf.Max(1, Mathf.RoundToInt(originalH * s));
        }

        ReportProgress(0f, "准备输入");
        await UniTask.Yield();

        try
        {
            var r = await ProcessOnceAsyncNv(src, ct, originalW, originalH, runInW, runInH, runFactor);
            return Finish(r);
        }
        catch (Exception e)
        {
            return Finish(new RealEsrganResult { error = e.Message ?? "unknown error" });
        }
    }

    private async UniTask<RealEsrganResult> ProcessOnceAsync(Texture2D src, CancellationToken ct, int originalW, int originalH, int runInW, int runInH, int runFactor, int effectiveTileSize, int effectiveTilePad)
    {
        Texture2D runInput = null;
        var ownsRunInput = false;
        RenderTexture scaledOutRt = null;
        RenderTexture outRt = null;
        ComputeBuffer probeBuf = null;
        Vector4[] probeData = null;
        ComputeBuffer probeInBuf = null;

        try
        {
            if (runInW != originalW || runInH != originalH)
            {
                runInput = ResizeTextureBilinear(src, runInW, runInH);
                ownsRunInput = true;
                if (runInput == null)
                    return new RealEsrganResult { error = "resize input failed" };
            }
            else
            {
                runInput = src;
            }

            var scaledOutW = runInW * runFactor;
            var scaledOutH = runInH * runFactor;
            scaledOutRt = new RenderTexture(scaledOutW, scaledOutH, 0, RenderTextureFormat.ARGB32);
            scaledOutRt.enableRandomWrite = true;
            scaledOutRt.wrapMode = TextureWrapMode.Clamp;
            scaledOutRt.filterMode = FilterMode.Bilinear;
            scaledOutRt.Create();
            if (!scaledOutRt.IsCreated())
                throw new InvalidOperationException("failed to create scaledOutRt " + scaledOutW + "x" + scaledOutH);

            if (scaledOutW != originalW || scaledOutH != originalH)
            {
                outRt = new RenderTexture(originalW, originalH, 0, RenderTextureFormat.ARGB32);
                outRt.wrapMode = TextureWrapMode.Clamp;
                outRt.filterMode = FilterMode.Bilinear;
                outRt.Create();
                if (!outRt.IsCreated())
                    throw new InvalidOperationException("failed to create outRt " + originalW + "x" + originalH);
            }

            var tilesX = Mathf.CeilToInt(runInW / (float)Mathf.Max(1, effectiveTileSize));
            var tilesY = Mathf.CeilToInt(runInH / (float)Mathf.Max(1, effectiveTileSize));
            var tileCount = Mathf.Max(1, tilesX * tilesY);
            var tileIndex = 0;
            var packMs = 0L;
            var forwardMs = 0L;
            var blitMs = 0L;
            var rentMs = 0L;
            var returnMs = 0L;
            var yieldMs = 0L;
            var cmdMs = 0L;
            var swTileAll = Stopwatch.StartNew();
            if (enableTileProbe)
            {
                probeBuf = new ComputeBuffer(tileCount, sizeof(float) * 4, ComputeBufferType.Structured);
                probeData = new Vector4[tileCount];
            }

            Vector4[] probeInData = null;
            if (enableTileProbe)
            {
                probeInBuf = new ComputeBuffer(tileCount, sizeof(float) * 4, ComputeBufferType.Structured);
                probeInData = new Vector4[tileCount];
            }


            for (var ty = 0; ty < runInH; ty += Mathf.Max(1, effectiveTileSize))
            {
                CommandBuffer rowCmd = null;
                if (_useCmdThisRun)
                {
                    rowCmd = new CommandBuffer();
                    rowCmd.name = "TileRow_" + ty;
                }

                for (var tx = 0; tx < runInW; tx += Mathf.Max(1, effectiveTileSize))
                {
                    ct.ThrowIfCancellationRequested();

                    var tw = Mathf.Min(effectiveTileSize, runInW - tx);
                    var th = Mathf.Min(effectiveTileSize, runInH - ty);
                    var cw = tw + effectiveTilePad * 2;
                    var ch = th + effectiveTilePad * 2;
                    var ox = tx - effectiveTilePad;
                    var oy = ty - effectiveTilePad;
          

                    var swRent = Stopwatch.StartNew();
                    NcnnTensorBuffer inBuf = null;
                    inBuf = RentTempBuffer(cw, ch, 3);
                    rentMs += swRent.ElapsedMilliseconds;
                    NcnnTensorBuffer outBuf = null;
                    RenderTexture outTex = null;
                    try
                    {
                        var swPack = Stopwatch.StartNew();
                        if (_useCmdThisRun)
                            _ops.TextureToBuffer3Cmd(rowCmd, runInput, ox, oy, inBuf);
                        else
                            _ops.TextureToBuffer3(runInput, ox, oy, inBuf);
                        packMs += swPack.ElapsedMilliseconds;

                        var swFwd = Stopwatch.StartNew();
                        if (_useCmdThisRun)
                            outBuf = ForwardNcnnCmd(rowCmd, inBuf);
                        else
                            outBuf = ForwardNcnn(inBuf);
                        forwardMs += swFwd.ElapsedMilliseconds;

                        var dstX = tx * runFactor;
                        var dstY = ty * runFactor;
                        var dstW = tw * runFactor;
                        var dstH = th * runFactor;
                        var tileOutOriginX = ox * runFactor;
                        var tileOutOriginY = oy * runFactor;
                        var swBlit = Stopwatch.StartNew();

                        if (_useCmdThisRun)
                        {
                            outTex = RenderTexture.GetTemporary(cw * runFactor, ch * runFactor, 0, RenderTextureFormat.ARGBHalf);
                            outTex.enableRandomWrite = true;
                            outTex.Create();
                            _ops.BufferToTexture3Cmd(rowCmd, outBuf, outTex);
                            _ops.BlitTileToDst(rowCmd, outTex, scaledOutRt, dstX, dstY, tileOutOriginX, tileOutOriginY, dstW, dstH, 1f);
                        }
                        else
                        {
                            outTex = RenderTexture.GetTemporary(cw * runFactor, ch * runFactor, 0, RenderTextureFormat.ARGBHalf);
                            outTex.enableRandomWrite = true;
                            outTex.Create();
                            _ops.BufferToTexture3(outBuf, outTex);
                            _ops.BlitTileToDst(outTex, scaledOutRt, dstX, dstY, tileOutOriginX, tileOutOriginY, dstW, dstH, 1f);
                        }
                        blitMs += swBlit.ElapsedMilliseconds;

                    }
                    finally
                    {
                        var swRet = Stopwatch.StartNew();
                        if (inBuf != null) ReturnTempBuffer(inBuf);
                        if (outBuf != null) ReturnTempBuffer(outBuf);
                        if (outTex != null) RenderTexture.ReleaseTemporary(outTex);
                        returnMs += swRet.ElapsedMilliseconds;
                    }

                    //if (tileIndex % 4 == 0)
                    //await UniTask.Yield();
                    await UniTask.NextFrame();

                    tileIndex++;
                }

                if (_useCmdThisRun)
                {
                    var sC = Stopwatch.StartNew();
                    Graphics.ExecuteCommandBufferAsync(rowCmd, ComputeQueueType.Default);
                    //Graphics.ExecuteCommandBuffer(rowCmd);
                    rowCmd.Dispose();
                    cmdMs += sC.ElapsedMilliseconds;
                }

                var tileProgress = (float)tileIndex / tileCount;
                ReportProgress(tileProgress, "推理分块 " + (tileIndex + 1) + "/" + tileCount);
                var sw = Stopwatch.StartNew();
                await UniTask.Yield();
                yieldMs += sw.ElapsedMilliseconds;
            }

            ReportProgress(0.98f, "后处理");
            var finalRt = scaledOutRt;
            if (outRt != null)
            {
                Graphics.Blit(scaledOutRt, outRt);
                finalRt = outRt;
            }

            if (enableSeamProbe)
            {
                var seamCount = Mathf.Max(0, tilesX - 1) + Mathf.Max(0, tilesY - 1);
                if (seamCount > 0)
                {
                    using (var seamBuf = new ComputeBuffer(seamCount, sizeof(float) * 4, ComputeBufferType.Structured))
                    {
                        _ops.ProbeSeams(scaledOutRt, tilesX, tilesY, effectiveTileSize * runFactor, effectiveTileSize * runFactor, 32, seamBuf);
                        var seamData = new Vector4[seamCount];
                        seamBuf.GetData(seamData);
                        var maxScore = 0f;
                        var maxType = 0f;
                        var maxPos = 0f;
                        for (var i = 0; i < seamData.Length; i++)
                        {
                            var v = seamData[i];
                            if (v.x > maxScore)
                            {
                                maxScore = v.x;
                                maxType = v.y;
                                maxPos = v.z;
                            }
                        }
                        var seamType = maxType < 0.5f ? "V" : "H";
                        UnityEngine.Debug.Log("[SEAM] ESRGAN(repro) maxScore=" + maxScore.ToString("0.######", CultureInfo.InvariantCulture) + " type=" + seamType + " pos=" + ((int)maxPos));
                    }
                }
            }

            ReportProgress(0.99f, "读取结果");
            var scaledTex = await ReadbackTextureAsync(finalRt, finalRt.width, finalRt.height, ct);
            if (scaledTex == null)
                return new RealEsrganResult { error = "readback failed" };

            if (probeBuf != null && probeData != null)
            {
                try
                {
                    probeBuf.GetData(probeData);
                    float maxDiff = 0f;
                    var maxA = -1;
                    var maxB = -1;
                    for (var y = 0; y < tilesY; y++)
                    {
                        for (var x = 0; x < tilesX; x++)
                        {
                            var a = y * tilesX + x;
                            if (x + 1 < tilesX)
                            {
                                var b = y * tilesX + (x + 1);
                                var da = probeData[a];
                                var db = probeData[b];
                                var d = Mathf.Abs(da.x - db.x) + Mathf.Abs(da.y - db.y) + Mathf.Abs(da.z - db.z);
                                if (d > maxDiff) { maxDiff = d; maxA = a; maxB = b; }
                            }
                            if (y + 1 < tilesY)
                            {
                                var b = (y + 1) * tilesX + x;
                                var da = probeData[a];
                                var db = probeData[b];
                                var d = Mathf.Abs(da.x - db.x) + Mathf.Abs(da.y - db.y) + Mathf.Abs(da.z - db.z);
                                if (d > maxDiff) { maxDiff = d; maxA = a; maxB = b; }
                            }
                        }
                    }
                    var va = (maxA >= 0 && maxA < probeData.Length) ? probeData[maxA] : Vector4.zero;
                    var vb = (maxB >= 0 && maxB < probeData.Length) ? probeData[maxB] : Vector4.zero;
                    var ax = maxA >= 0 ? (maxA % tilesX) : -1;
                    var ay = maxA >= 0 ? (maxA / tilesX) : -1;
                    var bx = maxB >= 0 ? (maxB % tilesX) : -1;
                    var by = maxB >= 0 ? (maxB / tilesX) : -1;
                    UnityEngine.Debug.Log(
                        "[PROBE] ESRGAN(repro) tiles=" + tilesX + "x" + tilesY
                        + " maxAdjDiff=" + maxDiff.ToString("0.######", CultureInfo.InvariantCulture)
                        + " a=" + maxA + " (" + ax + "," + ay + ") (" + va.x.ToString("0.######", CultureInfo.InvariantCulture) + "," + va.y.ToString("0.######", CultureInfo.InvariantCulture) + "," + va.z.ToString("0.######", CultureInfo.InvariantCulture) + ")"
                        + " b=" + maxB + " (" + bx + "," + by + ") (" + vb.x.ToString("0.######", CultureInfo.InvariantCulture) + "," + vb.y.ToString("0.######", CultureInfo.InvariantCulture) + "," + vb.z.ToString("0.######", CultureInfo.InvariantCulture) + ")"
                    );
                }
                catch
                {
                }
            }
            if (probeInBuf != null && probeInData != null)
            {
                try
                {
                    probeInBuf.GetData(probeInData);
                    float maxDiff = 0f;
                    var maxA = -1;
                    var maxB = -1;
                    for (var y = 0; y < tilesY; y++)
                    {
                        for (var x = 0; x < tilesX; x++)
                        {
                            var a = y * tilesX + x;
                            if (x + 1 < tilesX)
                            {
                                var b = y * tilesX + (x + 1);
                                var da = probeInData[a];
                                var db = probeInData[b];
                                var d = Mathf.Abs(da.x - db.x) + Mathf.Abs(da.y - db.y) + Mathf.Abs(da.z - db.z);
                                if (d > maxDiff) { maxDiff = d; maxA = a; maxB = b; }
                            }
                            if (y + 1 < tilesY)
                            {
                                var b = (y + 1) * tilesX + x;
                                var da = probeInData[a];
                                var db = probeInData[b];
                                var d = Mathf.Abs(da.x - db.x) + Mathf.Abs(da.y - db.y) + Mathf.Abs(da.z - db.z);
                                if (d > maxDiff) { maxDiff = d; maxA = a; maxB = b; }
                            }
                        }
                    }

                    var va = (maxA >= 0 && maxA < probeInData.Length) ? probeInData[maxA] : Vector4.zero;
                    var vb = (maxB >= 0 && maxB < probeInData.Length) ? probeInData[maxB] : Vector4.zero;
                    var ax = maxA >= 0 ? (maxA % tilesX) : -1;
                    var ay = maxA >= 0 ? (maxA / tilesX) : -1;
                    var bx = maxB >= 0 ? (maxB % tilesX) : -1;
                    var by = maxB >= 0 ? (maxB / tilesX) : -1;
                    UnityEngine.Debug.Log(
                        "[PROBE] ESRGAN(repro.in) tiles=" + tilesX + "x" + tilesY
                        + " maxAdjDiff=" + maxDiff.ToString("0.######", CultureInfo.InvariantCulture)
                        + " a=" + maxA + " (" + ax + "," + ay + ") (" + va.x.ToString("0.######", CultureInfo.InvariantCulture) + "," + va.y.ToString("0.######", CultureInfo.InvariantCulture) + "," + va.z.ToString("0.######", CultureInfo.InvariantCulture) + ")"
                        + " b=" + maxB + " (" + bx + "," + by + ") (" + vb.x.ToString("0.######", CultureInfo.InvariantCulture) + "," + vb.y.ToString("0.######", CultureInfo.InvariantCulture) + "," + vb.z.ToString("0.######", CultureInfo.InvariantCulture) + ")"
                    );
                }
                catch
                {
                }
            }
            try
            {
                UnityEngine.Debug.Log(
                    "[TIMING] ESRGAN(repro.breakdown) tiles=" + tileCount
                    + " rent=" + rentMs + " ms | return=" + returnMs + " ms | yield=" + yieldMs + " ms"
                    + " | pack=" + packMs + " ms | forward=" + forwardMs + " ms | cmdBuf=" + cmdMs + " ms"
                    + " | tileAll=" + swTileAll.ElapsedMilliseconds + " ms | pool=0"
                );
            }
            catch
            {
            }

            Texture2D finalTex = scaledTex;
            if (finalTex == null)
                return new RealEsrganResult { error = "resize output failed" };

            ReportProgress(1f, "完成");
            return new RealEsrganResult { texture = finalTex, error = null };
        }
        catch (OperationCanceledException)
        {
            return new RealEsrganResult { error = "Cancelled" };
        }
        finally
        {
            if (ownsRunInput && runInput != null) Destroy(runInput);
            if (scaledOutRt != null)
            {
                scaledOutRt.Release();
                Destroy(scaledOutRt);
            }
            if (outRt != null)
            {
                outRt.Release();
                Destroy(outRt);
            }
            try { probeBuf?.Dispose(); } catch { }
            try { probeInBuf?.Dispose(); } catch { }
        }
    }

    private void EnsureLoaded()
    {
        if (_loaded)
            return;

        var model = string.IsNullOrWhiteSpace(modelName) ? "realesrgan-x4plus" : modelName.Trim();
        var paramPath = Path.Combine(Application.streamingAssetsPath, "RealESRGAN", "models", model + ".param");
        var binPath = Path.Combine(Application.streamingAssetsPath, "RealESRGAN", "models", model + ".bin");

        var paramText = File.ReadAllText(paramPath);
        _model = NcnnParamParser.Parse(paramText);
        _blobUseCount = BuildBlobUseCount(_model);

        using (var fs = File.OpenRead(binPath))
        using (var br = new NcnnBinReader(fs))
        {
            foreach (var layer in _model.layers)
            {
                if (!string.Equals(layer.type, "Convolution", StringComparison.Ordinal))
                    continue;

                var pack = new ConvPack();
                pack.outC = layer.GetInt(0, 0);
                pack.kernel = layer.GetInt(1, 3);
                pack.stride = layer.GetInt(2, 1);
                pack.pad = layer.GetInt(4, 0);
                pack.biasTerm = layer.GetInt(5, 0);
                pack.weightSize = layer.GetInt(6, 0);
                pack.activationType = layer.GetInt(9, 0);
                pack.activationSlope = ParseLeakySlope(layer);
                pack.inC = Mathf.Max(1, pack.weightSize / Mathf.Max(1, pack.outC * 9));
                pack.outPacks = (pack.outC + 3) / 4;
                pack.inPacks = (pack.inC + 3) / 4;

                var tag = br.ReadInt32();
                if (tag != 0x01306B47)
                    throw new InvalidOperationException("unexpected weight tag at " + br.Position + ": 0x" + tag.ToString("X8", CultureInfo.InvariantCulture));

                var w = br.ReadFp16ArrayAsFloat32(pack.weightSize);
                var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
                pack.wOihw = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                pack.bOihw = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                pack.wOihw.SetData(w);
                pack.bOihw.SetData(b);

                if (pack.kernel == 3)
                {
                    var w4 = PackWeightsToO4I4K3(w, pack.outC, pack.inC, pack.outPacks, pack.inPacks);
                    var b4 = PackBiasToO4(b, pack.outC, pack.outPacks);
                    pack.w4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    pack.b4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    pack.w4.SetData(w4);
                    pack.b4.SetData(b4);
                }

                _conv[layer.name] = pack;
            }
        }

        _loaded = true;
    }

    private NcnnTensorBuffer ForwardNcnn(NcnnTensorBuffer input)
    {
        var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
        var blobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);

        var inputRef = new TensorRef { t = input, w = input.w, h = input.h, c = input.c, refs = 1, owned = false };
        blobs["data"] = inputRef;

        for (var li = 0; li < _model.layers.Count; li++)
        {
            var l = _model.layers[li];
            if (string.Equals(l.type, "Input", StringComparison.Ordinal))
                continue;

            UnityEngine.Debug.Log($"[ForwardNcnn] Layer {li}/{_model.layers.Count}: type={l.type}, name={l.name}");

            if (string.Equals(l.type, "Split", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                for (var i = 0; i < l.topNames.Length; i++)
                {
                    blobs[l.topNames[i]] = src;
                    src.refs++;
                }
                continue;
            }

            if (string.Equals(l.type, "Concat", StringComparison.Ordinal))
            {
                var parts = new TensorRef[l.bottomNames.Length];
                var sumC = 0;
                var w = 0;
                var h = 0;
                for (var i = 0; i < l.bottomNames.Length; i++)
                {
                    var tr = Get(blobs, l.bottomNames[i]);
                    parts[i] = tr;
                    w = tr.w;
                    h = tr.h;
                    sumC += tr.c;
                    UnityEngine.Debug.Log($"[Concat] bottomNames[{i}]={l.bottomNames[i]} tr.c={tr.c} tr.t.buffer.count={tr.t?.buffer?.count}");
                }

                var outBuf = RentTempBuffer(w, h, sumC);
                UnityEngine.Debug.Log($"[Concat] outBuf.c={outBuf.c} sumC={sumC}");
                var off = 0;
                for (var i = 0; i < parts.Length; i++)
                {
                    var srcPart = RentTempBuffer(parts[i].w, parts[i].h, parts[i].c);
                    UnityEngine.Debug.Log($"[Concat] i={i} srcPart.c={srcPart.c} parts[i].c={parts[i].c} off={off} outBuf.c={outBuf.c}");
                    _ops.CopyBuf(parts[i].t.buffer, srcPart.buffer, parts[i].w * parts[i].h * parts[i].c);
                    _ops.CopyToConcat(srcPart, outBuf, off);
                    ReturnTempBuffer(srcPart);
                    off += parts[i].c;
                }

                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = w, h = h, c = sumC, refs = 1, owned = true };
                ConsumeNcnn(blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            } 

           if (string.Equals(l.type, "Convolution", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var pack = _conv[l.name];
                if (pack.kernel != 3)
                    throw new InvalidOperationException("unsupported kernel size: " + pack.kernel);

                var outW = src.w;
                var outH = src.h;
                var outBuf = RentTempBuffer(outW, outH, pack.outC);
                _ops.Conv3x3(src.t, pack.wOihw, pack.bOihw, pack.outC, 1, pack.pad, pack.activationType, pack.activationSlope, outBuf);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = outW, h = outH, c = pack.outC, refs = 1, owned = true };
                ConsumeNcnn(blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "Eltwise", StringComparison.Ordinal))
            {
                var a = Get(blobs, l.bottomNames[0]);
                var b = Get(blobs, l.bottomNames[1]);
                var coeff = ParseEltwiseCoeff(l);
                var outBuf = RentTempBuffer(a.w, a.h, a.c);
                _ops.AddWeighted(a.t, b.t, coeff.coeffA, coeff.coeffB, outBuf);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = a.w, h = a.h, c = a.c, refs = 1, owned = true };
                ConsumeNcnn(blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "Interp", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var sx = l.GetFloat(1, 1f);
                var sy = l.GetFloat(2, 1f);
                if (Mathf.Abs(sx - 2f) > 1e-3f || Mathf.Abs(sy - 2f) > 1e-3f)
                    throw new InvalidOperationException("unsupported interp scale: " + sx.ToString("0.###", CultureInfo.InvariantCulture) + "," + sy.ToString("0.###", CultureInfo.InvariantCulture));

                var outBuf = RentTempBuffer(src.w * 2, src.h * 2, src.c);
                _ops.Interp2x(src.t, outBuf);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = src.w * 2, h = src.h * 2, c = src.c, refs = 1, owned = true };
                ConsumeNcnn(blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "LeakyReLU", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var slope = l.GetFloat(0, 0.2f);
                _ops.LeakyReluInplace(src.t, slope);
                blobs[l.topNames[0]] = new TensorRef { t = src.t, w = src.w, h = src.h, c = src.c, refs = 1, owned = false };
                ConsumeNcnn(blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "Sigmoid", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var outBuf = RentTempBuffer(src.w, src.h, src.c);
                var total = src.w * src.h * src.c;
                _ops.SigmoidBuf(src.t.buffer, total, outBuf.buffer);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = src.w, h = src.h, c = src.c, refs = 1, owned = true };
                ConsumeNcnn(blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "Swish", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var outBuf = RentTempBuffer(src.w, src.h, src.c);
                var total = src.w * src.h * src.c;
                _ops.SwishBuf(src.t.buffer, total, outBuf.buffer);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = src.w, h = src.h, c = src.c, refs = 1, owned = true };
                ConsumeNcnn(blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "GELU", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var fast = l.GetInt(0, 0) != 0;
                var outBuf = RentTempBuffer(src.w, src.h, src.c);
                var total = src.w * src.h * src.c;
                _ops.GeluBuf(src.t.buffer, total, outBuf.buffer);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = src.w, h = src.h, c = src.c, refs = 1, owned = true };
                ConsumeNcnn(blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }



            if (string.Equals(l.type, "BinaryOp", StringComparison.Ordinal))
            {
                var opType = l.GetInt(0, 0);
                var withScalar = l.GetInt(1, 0);
                var scalarB = l.GetFloat(2, 0f);
                var a = Get(blobs, l.bottomNames[0]);
                var outBuf = RentTempBuffer(a.w, a.h, a.c);
                if (withScalar != 0)
                {
                    var total = a.w * a.h * a.c;
                    _ops.BinaryOpScalarBuf(a.t.buffer, scalarB, total, opType, outBuf.buffer);
                }
                else
                {
                    var b = Get(blobs, l.bottomNames[1]);
                    if (a.w != b.w || a.h != b.h || a.c != b.c)
                        throw new InvalidOperationException("BinaryOp broadcast not supported: " + l.name);
                    var total = a.w * a.h * a.c;
                    _ops.BinaryOpBuf(a.t.buffer, b.t.buffer, total, opType, outBuf.buffer);
                }
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = a.w, h = a.h, c = a.c, refs = 1, owned = true };
                ConsumeNcnn(blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            throw new InvalidOperationException("unsupported layer type: " + l.type);
        }

        var outRef = Get(blobs, "output");
        var keep = outRef.t;
        outRef.t = null;
        outRef.owned = false;

        var visited = new HashSet<TensorRef>();
        foreach (var kv in blobs)
        {
            var tr = kv.Value;
            if (tr == null || !visited.Add(tr))
                continue;
            if (tr.owned && tr.t != null)
                ReturnTempBuffer(tr.t);
        }

        return keep;
    }

    private void ReturnToPool(NcnnTensorBuffer t)
    {
        if (t != null)
            _bufferPool.Return(t);
    }

    private static void ConsumeNcnn(Dictionary<string, TensorRef> blobs, Dictionary<string, int> remaining, string[] bottomNames, Action<NcnnTensorBuffer> returnAction)
    {
        for (var i = 0; i < bottomNames.Length; i++)
        {
            var b = bottomNames[i];
            if (!remaining.TryGetValue(b, out var c))
                continue;
            c--;
            remaining[b] = c;
            if (c > 0)
                continue;

            if (blobs.TryGetValue(b, out var tr) && tr != null)
            {
                tr.refs--;
                if (tr.refs <= 0)
                {
                    if (tr.owned && tr.t != null)
                    {
                        try { returnAction?.Invoke(tr.t); } catch { }
                    }
                    tr.t = null;
                    tr.owned = false;
                }
            }
            blobs.Remove(b);
        }
    }

    private static void ConsumeNcnnCmd(CommandBuffer cmd, Dictionary<string, TensorRef> blobs, Dictionary<string, int> remaining, string[] bottomNames, Action<NcnnTensorBuffer> returnAction)
    {
        for (var i = 0; i < bottomNames.Length; i++)
        {
            var b = bottomNames[i];
            if (!remaining.TryGetValue(b, out var c))
                continue;
            c--;
            remaining[b] = c;
            if (c > 0)
                continue;

            if (blobs.TryGetValue(b, out var tr) && tr != null)
            {
                tr.refs--;
                if (tr.refs <= 0)
                {
                    if (tr.owned && tr.t != null)
                    {
                        try { returnAction?.Invoke(tr.t); } catch { }
                    }
                    tr.t = null;
                    tr.owned = false;
                }
            }
            blobs.Remove(b);
        }
    }

    private NcnnTensorBuffer ForwardNcnnCmd(CommandBuffer cmd, NcnnTensorBuffer input)
    {
        var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
        var blobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);

        var inputRef = new TensorRef { t = input, w = input.w, h = input.h, c = input.c, refs = 1, owned = false };
        blobs["data"] = inputRef;

        for (var li = 0; li < _model.layers.Count; li++)
        {
            var l = _model.layers[li];
            if (string.Equals(l.type, "Input", StringComparison.Ordinal))
                continue;

            if (string.Equals(l.type, "Split", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                for (var i = 0; i < l.topNames.Length; i++)
                {
                    blobs[l.topNames[i]] = src;
                    src.refs++;
                }
                continue;
            }

            if (string.Equals(l.type, "Convolution", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var pack = _conv[l.name];
                if (pack.kernel != 3)
                    throw new InvalidOperationException("unsupported kernel size: " + pack.kernel);

                var outW = src.w;
                var outH = src.h;
                var outBuf = RentTempBuffer(outW, outH, pack.outC);
                _ops.Conv3x3Cmd(cmd, src.t, pack.wOihw, pack.bOihw, pack.outC, pack.stride, pack.pad, pack.activationType, pack.activationSlope, outBuf);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = outW, h = outH, c = pack.outC, refs = 1, owned = true };
                ConsumeNcnnCmd(cmd, blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "Eltwise", StringComparison.Ordinal))
            {
                var a = Get(blobs, l.bottomNames[0]);
                var b = Get(blobs, l.bottomNames[1]);
                var coeff = ParseEltwiseCoeff(l);
                var outBuf = RentTempBuffer(a.w, a.h, a.c);
                _ops.AddWeightedCmd(cmd, a.t, b.t, coeff.coeffA, coeff.coeffB, outBuf);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = a.w, h = a.h, c = a.c, refs = 1, owned = true };
                ConsumeNcnnCmd(cmd, blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "Interp", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var sx = l.GetFloat(1, 1f);
                var sy = l.GetFloat(2, 1f);
                if (Mathf.Abs(sx - 2f) > 1e-3f || Mathf.Abs(sy - 2f) > 1e-3f)
                    throw new InvalidOperationException("unsupported interp scale: " + sx.ToString("0.###", CultureInfo.InvariantCulture) + "," + sy.ToString("0.###", CultureInfo.InvariantCulture));

                var outBuf = RentTempBuffer(src.w * 2, src.h * 2, src.c);
                _ops.Interp2xCmd(cmd, src.t, outBuf);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = src.w * 2, h = src.h * 2, c = src.c, refs = 1, owned = true };
                ConsumeNcnnCmd(cmd, blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "LeakyReLU", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var slope = l.GetFloat(0, 0.2f);
                var outBuf = RentTempBuffer(src.w, src.h, src.c);
                _ops.CopyBufCmd(cmd, src.t.buffer, outBuf.buffer, src.w * src.h * src.c);
                _ops.LeakyReluInplaceCmd(cmd, outBuf, slope);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = src.w, h = src.h, c = src.c, refs = 1, owned = true };
                ConsumeNcnnCmd(cmd, blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "Sigmoid", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var outBuf = RentTempBuffer(src.w, src.h, src.c);
                var total = src.w * src.h * src.c;
                _ops.SigmoidBufCmd(cmd, src.t.buffer, total, outBuf.buffer);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = src.w, h = src.h, c = src.c, refs = 1, owned = true };
                ConsumeNcnnCmd(cmd, blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "Swish", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var outBuf = RentTempBuffer(src.w, src.h, src.c);
                var total = src.w * src.h * src.c;
                _ops.SwishBufCmd(cmd, src.t.buffer, total, outBuf.buffer);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = src.w, h = src.h, c = src.c, refs = 1, owned = true };
                ConsumeNcnnCmd(cmd, blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "GELU", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var fast = l.GetInt(0, 0) != 0;
                var outBuf = RentTempBuffer(src.w, src.h, src.c);
                var total = src.w * src.h * src.c;
                _ops.GeluBufCmd(cmd, src.t.buffer, total, outBuf.buffer);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = src.w, h = src.h, c = src.c, refs = 1, owned = true };
                ConsumeNcnnCmd(cmd, blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "Concat", StringComparison.Ordinal))
            {
                var parts = new TensorRef[l.bottomNames.Length];
                var sumC = 0;
                var w = 0;
                var h = 0;
                for (var i = 0; i < l.bottomNames.Length; i++)
                {
                    var tr = Get(blobs, l.bottomNames[i]);
                    parts[i] = tr;
                    w = tr.w;
                    h = tr.h;
                    sumC += tr.c;
                }

                var outBuf = RentTempBuffer(w, h, sumC);
                var off = 0;
                for (var i = 0; i < parts.Length; i++)
                {
                    var srcPart = RentTempBuffer(parts[i].w, parts[i].h, parts[i].c);
                    var partTotal = parts[i].w * parts[i].h * parts[i].c;
                    _ops.CopyBufCmd(cmd, parts[i].t.buffer, srcPart.buffer, partTotal);
                    _ops.CopyToConcatCmd(cmd, srcPart, outBuf, off);
                    ReturnTempBuffer(srcPart);
                    off += parts[i].c;
                }

                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = w, h = h, c = sumC, refs = 1, owned = true };
                ConsumeNcnnCmd(cmd, blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            if (string.Equals(l.type, "BinaryOp", StringComparison.Ordinal))
            {
                var opType = l.GetInt(0, 0);
                var withScalar = l.GetInt(1, 0);
                var scalarB = l.GetFloat(2, 0f);
                var a = Get(blobs, l.bottomNames[0]);
                var outBuf = RentTempBuffer(a.w, a.h, a.c);
                if (withScalar != 0)
                {
                    var total = a.w * a.h * a.c;
                    _ops.BinaryOpScalarBufCmd(cmd, a.t.buffer, scalarB, total, opType, outBuf.buffer);
                }
                else
                {
                    var b = Get(blobs, l.bottomNames[1]);
                    if (a.w != b.w || a.h != b.h || a.c != b.c)
                        throw new InvalidOperationException("BinaryOp broadcast not supported: " + l.name);
                    var total = a.w * a.h * a.c;
                    _ops.BinaryOpBufCmd(cmd, a.t.buffer, b.t.buffer, total, opType, outBuf.buffer);
                }
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, w = a.w, h = a.h, c = a.c, refs = 1, owned = true };
                ConsumeNcnnCmd(cmd, blobs, remaining, l.bottomNames, ReturnToPool);
                continue;
            }

            throw new InvalidOperationException("unsupported layer type: " + l.type);
        }

        var outRef = Get(blobs, "output");
        var keep = outRef.t;
        outRef.t = null;
        outRef.owned = false;

        var visited = new HashSet<TensorRef>();
        foreach (var kv in blobs)
        {
            var tr = kv.Value;
            if (tr == null || !visited.Add(tr))
                continue;
            if (tr.owned && tr.t != null)
                ReturnTempBuffer(tr.t);
        }

        return keep;
    }

    private static TensorRef Get(Dictionary<string, TensorRef> blobs, string name)
    {
        if (!blobs.TryGetValue(name, out var tr) || tr == null)
            throw new InvalidOperationException("blob not found: " + name);
        return tr;
    }

    private static bool IsComputeCmdSupported(GraphicsDeviceType t)
    {
        return t == GraphicsDeviceType.Vulkan
               || t == GraphicsDeviceType.Direct3D11
               || t == GraphicsDeviceType.Direct3D12
               || t == GraphicsDeviceType.Metal
               || t == GraphicsDeviceType.WebGPU;
    }

    private static Dictionary<string, int> BuildBlobUseCount(NcnnParamModel model)
    {
        var use = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < model.layers.Count; i++)
        {
            var l = model.layers[i];
            if (l.bottomNames == null)
                continue;
            for (var b = 0; b < l.bottomNames.Length; b++)
            {
                var n = l.bottomNames[b];
                if (string.IsNullOrEmpty(n))
                    continue;
                use.TryGetValue(n, out var c);
                use[n] = c + 1;
            }
        }
        return use;
    }

    private static float ParseLeakySlope(NcnnParamModel.Layer layer)
    {
        if (layer.intParams == null || !layer.intParams.TryGetValue(-23310, out var s) || string.IsNullOrWhiteSpace(s))
            return 0.2f;
        var parts = s.Split(',');
        if (parts.Length >= 2 && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
            return v;
        return 0.2f;
    }

    private static (float coeffA, float coeffB) ParseEltwiseCoeff(NcnnParamModel.Layer layer)
    {
        if (layer.intParams == null || !layer.intParams.TryGetValue(-23301, out var s) || string.IsNullOrWhiteSpace(s))
            return (1f, 1f);
        var parts = s.Split(',');
        if (parts.Length >= 3
            && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var a)
            && float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var b))
            return (a, b);
        return (1f, 1f);
    }

    private static Vector4[] PackBiasToO4(float[] b, int outC, int outPacks)
    {
        var r = new Vector4[outPacks];
        for (var op = 0; op < outPacks; op++)
        {
            var oc0 = op * 4 + 0;
            var oc1 = op * 4 + 1;
            var oc2 = op * 4 + 2;
            var oc3 = op * 4 + 3;
            r[op] = new Vector4(
                oc0 < outC ? b[oc0] : 0f,
                oc1 < outC ? b[oc1] : 0f,
                oc2 < outC ? b[oc2] : 0f,
                oc3 < outC ? b[oc3] : 0f);
        }
        return r;
    }

    private static Vector4[] PackWeightsToO4I4K3(float[] w, int outC, int inC, int outPacks, int inPacks)
    {
        var r = new Vector4[outPacks * inPacks * 3 * 3 * 4];
        var idx = 0;
        for (var op = 0; op < outPacks; op++)
        {
            for (var ip = 0; ip < inPacks; ip++)
            {
                for (var ky = 0; ky < 3; ky++)
                {
                    for (var kx = 0; kx < 3; kx++)
                    {
                        for (var ol = 0; ol < 4; ol++)
                        {
                            var oc = op * 4 + ol;
                            var il0 = ip * 4 + 0;
                            var il1 = ip * 4 + 1;
                            var il2 = ip * 4 + 2;
                            var il3 = ip * 4 + 3;
                            var k = ky * 3 + kx;

                            float GetW(int ic)
                            {
                                if (oc >= outC || ic >= inC)
                                    return 0f;
                                return w[(oc * inC + ic) * 9 + k];
                            }

                            r[idx++] = new Vector4(GetW(il0), GetW(il1), GetW(il2), GetW(il3));
                        }
                    }
                }
            }
        }
        return r;
    }

    private RenderTexture RentTempArray(int w, int h, int depth, RenderTextureFormat format)
    {
        var desc = new RenderTextureDescriptor(w, h, format, 0)
        {
            dimension = TextureDimension.Tex2DArray,
            volumeDepth = Mathf.Max(1, depth),
            msaaSamples = 1,
            sRGB = false,
            enableRandomWrite = true
        };

        return RenderTexture.GetTemporary(desc);
    }

    private void ReturnTempArray(RenderTexture t)
    {
         RenderTexture.ReleaseTemporary(t);
    }

    private NcnnTensorBuffer RentTempBuffer(int w, int h, int c)
    {
        return _bufferPool.Rent(w, h, c);
    }

    private void ReturnTempBuffer(NcnnTensorBuffer t)
    {
        _bufferPool.Return(t);
    }

    private async UniTask<Texture2D> ReadbackTextureAsync(RenderTexture rt, int w, int h, CancellationToken ct)
    {
        var tcs = new UniTaskCompletionSource<AsyncGPUReadbackRequest>();
        AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, req => tcs.TrySetResult(req));
        var r = await tcs.Task;
        ct.ThrowIfCancellationRequested();
        if (r.hasError)
            return null;
        var data = r.GetData<byte>();
        var tex = new Texture2D(w, h, TextureFormat.RGBA32, false, true);
        tex.LoadRawTextureData(data);
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        return tex;
    }

    private static Texture2D ResizeTextureBilinear(Texture2D src, int w, int h)
    {
        if (src == null)
            return null;
        if (w <= 0 || h <= 0)
            return null;

        var dst = new Texture2D(w, h, TextureFormat.RGBA32, false, true);
        dst.wrapMode = TextureWrapMode.Clamp;
        dst.filterMode = FilterMode.Bilinear;

        var srcPixels = src.GetPixels32();
        var dstPixels = new Color32[w * h];
        var sw = src.width;
        var sh = src.height;
        var invW = sw > 1 ? 1f / (sw - 1f) : 0f;
        var invH = sh > 1 ? 1f / (sh - 1f) : 0f;

        for (var y = 0; y < h; y++)
        {
            var v = h > 1 ? y / (h - 1f) : 0f;
            var sy = v / Mathf.Max(1e-6f, invH);
            var y0 = Mathf.Clamp((int)sy, 0, sh - 1);
            var y1 = Mathf.Clamp(y0 + 1, 0, sh - 1);
            var ty = sy - y0;
            for (var x = 0; x < w; x++)
            {
                var u = w > 1 ? x / (w - 1f) : 0f;
                var sx = u / Mathf.Max(1e-6f, invW);
                var x0 = Mathf.Clamp((int)sx, 0, sw - 1);
                var x1 = Mathf.Clamp(x0 + 1, 0, sw - 1);
                var tx = sx - x0;

                var c00 = srcPixels[y0 * sw + x0];
                var c10 = srcPixels[y0 * sw + x1];
                var c01 = srcPixels[y1 * sw + x0];
                var c11 = srcPixels[y1 * sw + x1];

                var r0 = Mathf.Lerp(c00.r, c10.r, tx);
                var g0 = Mathf.Lerp(c00.g, c10.g, tx);
                var b0 = Mathf.Lerp(c00.b, c10.b, tx);
                var a0 = Mathf.Lerp(c00.a, c10.a, tx);

                var r1 = Mathf.Lerp(c01.r, c11.r, tx);
                var g1 = Mathf.Lerp(c01.g, c11.g, tx);
                var b1 = Mathf.Lerp(c01.b, c11.b, tx);
                var a1 = Mathf.Lerp(c01.a, c11.a, tx);

                var r2 = Mathf.Lerp(r0, r1, ty);
                var g2 = Mathf.Lerp(g0, g1, ty);
                var b2 = Mathf.Lerp(b0, b1, ty);
                var a2 = Mathf.Lerp(a0, a1, ty);

                dstPixels[y * w + x] = new Color32((byte)r2, (byte)g2, (byte)b2, (byte)a2);
            }
        }

        dst.SetPixels32(dstPixels);
        dst.Apply(false, false);
        return dst;
    }

    private void ReportProgress(float p, string t)
    {
        try { ProgressChanged?.Invoke(Mathf.Clamp01(p), t ?? ""); } catch { }
    }

    private static int AutoTileSize()
    {
        var mb = SystemInfo.graphicsMemorySize;
        if (mb > 1900)
            return 200;
        if (mb > 550)
            return 100;
        if (mb > 190)
            return 64;
        return 32;
    }

    private static bool IsLikelyVulkanOom(Exception e)
    {
        if (e == null) return false;
        var msg = e.Message ?? "";
        if (msg.IndexOf("Out of device memory", StringComparison.OrdinalIgnoreCase) >= 0) return true;
        if (msg.IndexOf("out of memory", StringComparison.OrdinalIgnoreCase) >= 0) return true;
        if (msg.IndexOf("failed to create", StringComparison.OrdinalIgnoreCase) >= 0) return true;
        return false;
    }

    private async UniTask<RealEsrganResult> ProcessOnceAsyncNv(Texture2D src, CancellationToken ct, int originalW, int originalH, int runInW, int runInH, int runFactor)
    {
        Texture2D runInput = null;
        var ownsRunInput = false;
        RenderTexture scaledOutRt = null;
        RenderTexture outRt = null;
        NcnnTensorBuffer inputBuf = null;
        NcnnTensorBuffer outputBuf = null;
        RenderTexture outputTex = null;

        try
        {
            if (runInW != originalW || runInH != originalH)
            {
                runInput = ResizeTextureBilinear(src, runInW, runInH);
                ownsRunInput = true;
                if (runInput == null)
                    return new RealEsrganResult { error = "resize input failed" };
            }
            else
            {
                runInput = src;
            }

            inputBuf = new NcnnTensorBuffer(runInW, runInH, 3);

            var cmd = new CommandBuffer();
            cmd.name = "RealESRGAN-ncnn";

            _ops.TextureToBuffer3Cmd(cmd, runInput, 0, 0, inputBuf);
            outputBuf = ForwardNcnnCmd(cmd, inputBuf);

            var scaledOutW = runInW * runFactor;
            var scaledOutH = runInH * runFactor;

            outputTex = new RenderTexture(scaledOutW, scaledOutH, 0, RenderTextureFormat.ARGBHalf);
            outputTex.enableRandomWrite = true;
            outputTex.wrapMode = TextureWrapMode.Clamp;
            outputTex.filterMode = FilterMode.Bilinear;
            outputTex.Create();

            _ops.BufferToTexture3Cmd(cmd, outputBuf, outputTex);

            Graphics.ExecuteCommandBuffer(cmd);
            cmd.Dispose();

            scaledOutRt = new RenderTexture(scaledOutW, scaledOutH, 0, RenderTextureFormat.ARGB32);
            scaledOutRt.enableRandomWrite = true;
            scaledOutRt.wrapMode = TextureWrapMode.Clamp;
            scaledOutRt.filterMode = FilterMode.Bilinear;
            scaledOutRt.Create();
            if (!scaledOutRt.IsCreated())
                throw new InvalidOperationException("failed to create scaledOutRt " + scaledOutW + "x" + scaledOutH);

            Graphics.Blit(outputTex, scaledOutRt);

            if (scaledOutW != originalW || scaledOutH != originalH)
            {
                outRt = new RenderTexture(originalW, originalH, 0, RenderTextureFormat.ARGB32);
                outRt.wrapMode = TextureWrapMode.Clamp;
                outRt.filterMode = FilterMode.Bilinear;
                outRt.Create();
                if (!outRt.IsCreated())
                    throw new InvalidOperationException("failed to create outRt " + originalW + "x" + originalH);
                Graphics.Blit(scaledOutRt, outRt);
            }

            ReportProgress(0.99f, "读取结果");
            var finalRt = outRt != null ? outRt : scaledOutRt;
            var scaledTex = await ReadbackTextureAsync(finalRt, finalRt.width, finalRt.height, ct);
            if (scaledTex == null)
                return new RealEsrganResult { error = "readback failed" };

            Texture2D finalTex = scaledTex;
            if (finalTex == null)
                return new RealEsrganResult { error = "resize output failed" };

            ReportProgress(1f, "完成");
            return new RealEsrganResult { texture = finalTex, error = null };
        }
        catch (OperationCanceledException)
        {
            return new RealEsrganResult { error = "Cancelled" };
        }
        finally
        {
            if (ownsRunInput && runInput != null) Destroy(runInput);
            if (scaledOutRt != null)
            {
                scaledOutRt.Release();
                Destroy(scaledOutRt);
            }
            if (outRt != null)
            {
                outRt.Release();
                Destroy(outRt);
            }
            if (outputTex != null)
            {
                outputTex.Release();
                Destroy(outputTex);
            }
            try { inputBuf?.Dispose(); } catch { }
            try { outputBuf?.Dispose(); } catch { }
        }
    }


}
