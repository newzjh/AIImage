using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;

public sealed class GfpganNcnnReproRunner : MonoBehaviour
{
    public bool enableGfpganRepro = true;
    public string paramRelativePath = "GFPGAN/models/encoder.param";
    public string binRelativePath = "GFPGAN/models/encoder.bin";
    public int maxInputLongSide = 2048;

    public event Action<float, string> ProgressChanged;

    private NcnnParamModel _model;
    private bool _loaded;

    public async UniTask<GfpganResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (!enableGfpganRepro)
            return new GfpganResult { error = "GFPGAN(复刻) disabled" };
        if (src == null)
            return default;

        EnsureLoaded();
        if (_model == null)
            return new GfpganResult { error = "GFPGAN(复刻) 模型不可用" };

        var originalW = src.width;
        var originalH = src.height;
        var maxSide = Mathf.Max(originalW, originalH);
        var limit = Mathf.Max(256, maxInputLongSide);
        var runInW = originalW;
        var runInH = originalH;
        if (maxSide > limit)
        {
            var s = (float)limit / maxSide;
            runInW = Mathf.Max(1, Mathf.RoundToInt(originalW * s));
            runInH = Mathf.Max(1, Mathf.RoundToInt(originalH * s));
        }

        ReportProgress(0f, "准备中");
        await UniTask.Yield();

        var sb = new StringBuilder();
        sb.Append("GFPGAN(复刻) 推理暂未实现。");
        sb.Append("当前已完成 param 解析，后续将按 ncnn Vulkan 支持的 layer/pack4 路径逐步补齐。");
        sb.Append(" 输入将限制到不超过 ");
        sb.Append(limit.ToString(CultureInfo.InvariantCulture));
        sb.Append("，当前运行输入: ");
        sb.Append(runInW.ToString(CultureInfo.InvariantCulture));
        sb.Append("x");
        sb.Append(runInH.ToString(CultureInfo.InvariantCulture));
        sb.AppendLine();

        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var l in _model.layers)
        {
            if (string.IsNullOrWhiteSpace(l.type))
                continue;
            counts.TryGetValue(l.type, out var c);
            counts[l.type] = c + 1;
        }

        sb.Append("Layer types: ");
        var first = true;
        foreach (var kv in counts)
        {
            if (!first) sb.Append(", ");
            first = false;
            sb.Append(kv.Key);
            sb.Append("=");
            sb.Append(kv.Value.ToString(CultureInfo.InvariantCulture));
        }

        ReportProgress(1f, "完成");
        return new GfpganResult { error = sb.ToString() };
    }

    private void EnsureLoaded()
    {
        if (_loaded)
            return;

        var paramPath = Path.Combine(Application.streamingAssetsPath, paramRelativePath);
        var binPath = Path.Combine(Application.streamingAssetsPath, binRelativePath);
        if (!File.Exists(paramPath))
            throw new InvalidOperationException("GFPGAN(复刻) param 不存在: " + paramPath);
        if (!File.Exists(binPath))
            throw new InvalidOperationException("GFPGAN(复刻) bin 不存在: " + binPath);

        var paramText = File.ReadAllText(paramPath);
        _model = NcnnParamParser.Parse(paramText);
        _loaded = true;
    }

    private void ReportProgress(float progress01, string text)
    {
        progress01 = Mathf.Clamp01(progress01);
        try { ProgressChanged?.Invoke(progress01, text ?? ""); } catch { }
    }
}

