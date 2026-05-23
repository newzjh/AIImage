using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;
using Debug = UnityEngine.Debug;

public sealed class RealEsrganNcnnVulkanRunner : MonoBehaviour
{
    public bool enableRealEsrgan = true;
    public int scale = 2;
    public string modelName = "realesrgan-x4plus";
    public int tileSize = 0;
    public int gpuId = -1;
    public string executablePathOverride;
    public string modelDirOverride;
    public int maxInputLongSide = 2048;
    public int inputAlignMultiple = 0;

    public event Action<float, string> ProgressChanged;
    private readonly object _progressLock = new object();
    private float _lastProgress;
    private long _lastProgressTicks;

    public async UniTask<RealEsrganResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (!enableRealEsrgan)
            return new RealEsrganResult { error = "Real-ESRGAN disabled", workDir = null };
        if (src == null)
            return default;

        var originalW = src.width;
        var originalH = src.height;

        var exePath = ResolveExecutablePath();
        if (string.IsNullOrWhiteSpace(exePath) || !File.Exists(exePath))
            return new RealEsrganResult { error = "realesrgan-ncnn-vulkan executable not found: " + (exePath ?? ""), workDir = null };

        var modelDir = ResolveModelDir();
        if (string.IsNullOrWhiteSpace(modelDir) || !Directory.Exists(modelDir))
            return new RealEsrganResult { error = "Real-ESRGAN model directory not found: " + (modelDir ?? ""), workDir = null };

        var s = Mathf.Clamp(scale, 2, 4);
        var model = string.IsNullOrWhiteSpace(modelName) ? "realesrgan-x4plus" : modelName.Trim();
        var modelFactor = InferModelFactor(model);
        var runFactor = modelFactor;

        var workDir = CreateWorkDir();
        var inputPath = Path.Combine(workDir, "input.png");
        var outputPath = Path.Combine(workDir, "output.png");
        var scaledInputPath = Path.Combine(workDir, "input_scaled.png");
        var scaledOutputPath = Path.Combine(workDir, "output_scaled.png");

        Texture2D scaledInput = null;
        try
        {
            ct.ThrowIfCancellationRequested();

            ReportProgress(0f, "准备输入");
            await UniTask.Yield();

            var maxSide = Mathf.Max(originalW, originalH);
            var limit = Mathf.Max(256, maxInputLongSide);
            var runInputPath = inputPath;
            var runOutputPath = outputPath;

            if (maxSide > limit)
            {
                var scaleDown = (float)limit / maxSide;
                var sw = Mathf.Max(1, Mathf.RoundToInt(originalW * scaleDown));
                var sh = Mathf.Max(1, Mathf.RoundToInt(originalH * scaleDown));
                if (inputAlignMultiple >= 2)
                {
                    var m = inputAlignMultiple;
                    sw = RoundToMultiple(sw, m);
                    sh = RoundToMultiple(sh, m);
                    if (originalW >= originalH)
                    {
                        if (sw > limit) sw = Mathf.Max(m, sw - m);
                    }
                    else
                    {
                        if (sh > limit) sh = Mathf.Max(m, sh - m);
                    }
                    sw = Mathf.Max(1, sw);
                    sh = Mathf.Max(1, sh);
                }
                scaledInput = ResizeTextureBilinear(src, sw, sh);
                if (scaledInput == null)
                    return new RealEsrganResult { error = "Failed to scale down input image" };

                var inputBytes = scaledInput.EncodeToPNG();
                await File.WriteAllBytesAsync(scaledInputPath, inputBytes, ct);
                runInputPath = scaledInputPath;
                runOutputPath = scaledOutputPath;
            }
            else
            {
                var inputBytes = src.EncodeToPNG();
                await File.WriteAllBytesAsync(inputPath, inputBytes, ct);
            }

            var args = BuildArgs(exePath, runInputPath, runOutputPath, runFactor, model, modelDir);

            var psi = new ProcessStartInfo
            {
                FileName = exePath,
                Arguments = args,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                WorkingDirectory = Path.GetDirectoryName(exePath) ?? ""
            };

            string threadError = null;
            byte[] outBytes = null;
            await UniTask.SwitchToThreadPool();
            try
            {
                var stdoutSb = new StringBuilder();
                var stderrSb = new StringBuilder();
                var sbLock = new object();
                int exitCode = -1;
                try
                {
                    using (var p = Process.Start(psi))
                    {
                        if (p == null)
                        {
                            threadError = "Failed to start Real-ESRGAN process";
                        }
                        else
                        {
                            ReportProgress(0.05f, "开始推理");

                            var readStdout = ConsumeStreamAsync(p.StandardOutput.BaseStream, line =>
                            {
                                lock (sbLock) stdoutSb.AppendLine(line);
                                TryReportProgressFromLine(line);
                            }, ct);

                            var readStderr = ConsumeStreamAsync(p.StandardError.BaseStream, line =>
                            {
                                lock (sbLock) stderrSb.AppendLine(line);
                                TryReportProgressFromLine(line);
                            }, ct);

                            using var pseudoCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                            var pseudoProgress = PseudoProgressAsync(pseudoCts.Token);

                            while (!p.HasExited)
                            {
                                ct.ThrowIfCancellationRequested();
                                await UniTask.Delay(50, cancellationToken: ct);
                            }

                            try { pseudoCts.Cancel(); } catch { }

                            await readStdout;
                            await readStderr;
                            try { await pseudoProgress; } catch { }
                            exitCode = p.ExitCode;
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    threadError = "Cancelled";
                }
                catch (Exception e)
                {
                    threadError = e.Message;
                }

                if (string.IsNullOrWhiteSpace(threadError))
                {
                    if (exitCode != 0)
                    {
                        string stdout;
                        string stderr;
                        lock (sbLock)
                        {
                            stdout = stdoutSb.ToString();
                            stderr = stderrSb.ToString();
                        }
                        threadError = "Real-ESRGAN failed (" + exitCode + "): " + (string.IsNullOrWhiteSpace(stderr) ? stdout : stderr);
                    }
                    else if (!File.Exists(runOutputPath))
                    {
                        threadError = "Real-ESRGAN output not found: " + runOutputPath;
                    }
                    else
                    {
                        try
                        {
                            ReportProgress(0.95f, "读取输出");
                            outBytes = await File.ReadAllBytesAsync(runOutputPath, ct);
                        }
                        catch (OperationCanceledException)
                        {
                            threadError = "Cancelled";
                        }
                        catch (Exception e)
                        {
                            threadError = e.Message;
                        }
                    }
                }
            }
            finally
            {
                await UniTask.SwitchToMainThread();
            }

            if (!string.IsNullOrWhiteSpace(threadError))
                return new RealEsrganResult { error = threadError, workDir = workDir };

            var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            if (!tex.LoadImage(outBytes, false))
            {
                Destroy(tex);
                return new RealEsrganResult { error = "Failed to decode Real-ESRGAN output image", workDir = workDir };
            }
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            tex.name = "RealESRGAN_" + runFactor + "x_scaled";

            Texture2D finalTex = tex;
            if (finalTex.width != originalW || finalTex.height != originalH)
            {
                ReportProgress(0.98f, "回缩放到原分辨率");
                var resized = ResizeTextureBilinear(finalTex, originalW, originalH);
                Destroy(finalTex);
                finalTex = resized;
                if (finalTex == null)
                    return new RealEsrganResult { error = "Failed to resize output back to original resolution" };
            }
            finalTex.name = "RealESRGAN_" + runFactor + "x";

            ReportProgress(1f, "完成");

            return new RealEsrganResult { texture = finalTex, workDir = workDir };
        }
        finally
        {
            if (scaledInput != null)
                Destroy(scaledInput);
        }
    }

    private string ResolveExecutablePath()
    {
        if (!string.IsNullOrWhiteSpace(executablePathOverride))
            return executablePathOverride;

        var root = Application.streamingAssetsPath;
        if (string.IsNullOrWhiteSpace(root))
            return null;

#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        return Path.Combine(root, "RealESRGAN", "realesrgan-ncnn-vulkan.exe");
#elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
        return Path.Combine(root, "RealESRGAN", "realesrgan-ncnn-vulkan");
#elif UNITY_STANDALONE_LINUX
        return Path.Combine(root, "RealESRGAN", "realesrgan-ncnn-vulkan");
#else
        return null;
#endif
    }

    private string ResolveModelDir()
    {
        if (!string.IsNullOrWhiteSpace(modelDirOverride))
            return modelDirOverride;
        var root = Application.streamingAssetsPath;
        if (string.IsNullOrWhiteSpace(root))
            return null;
        return Path.Combine(root, "RealESRGAN", "models");
    }

    private static string CreateWorkDir()
    {
        var root = Application.temporaryCachePath;
        if (string.IsNullOrWhiteSpace(root))
            root = Path.GetTempPath();
        var dir = Path.Combine(root, "AIImage_RealESRGAN_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
        try { Directory.CreateDirectory(dir); } catch { }
        return dir;
    }

    private string BuildArgs(string exePath, string inputPath, string outputPath, int s, string model, string modelDir)
    {
        var exeBase = (Path.GetFileNameWithoutExtension(exePath) ?? "").ToLowerInvariant();
        var isUpscayl = exeBase == "upscayl-bin";

        var exeDir = Path.GetDirectoryName(exePath) ?? "";
        var defaultModelDir = Path.Combine(exeDir, "models");
        var useModelDirArg = !isUpscayl && !string.Equals(Path.GetFullPath(modelDir), Path.GetFullPath(defaultModelDir), StringComparison.OrdinalIgnoreCase);

        var g = gpuId < 0 ? "auto" : gpuId.ToString();

        var sb = new StringBuilder();
        sb.Append("-v ");
        sb.Append("-i ").Append(QuoteArg(inputPath)).Append(' ');
        sb.Append("-o ").Append(QuoteArg(outputPath)).Append(' ');
        sb.Append("-s ").Append(s).Append(' ');
        if (isUpscayl)
            sb.Append("-z ").Append(s).Append(' ');
        sb.Append("-t ").Append(Mathf.Max(0, tileSize)).Append(' ');
        sb.Append("-n ").Append(QuoteArg(model)).Append(' ');
        sb.Append("-g ").Append(g).Append(' ');
        if (useModelDirArg)
            sb.Append("-m ").Append(QuoteArg(modelDir)).Append(' ');
        return sb.ToString().Trim();
    }

    private static string QuoteArg(string s)
    {
        if (string.IsNullOrEmpty(s))
            return "\"\"";
        if (s.IndexOfAny(new[] { ' ', '\t', '\n', '\r', '"' }) < 0)
            return s;
        return "\"" + s.Replace("\"", "\\\"") + "\"";
    }

    private void ReportProgress(float progress01, string text)
    {
        progress01 = Mathf.Clamp01(progress01);
        lock (_progressLock)
        {
            _lastProgress = progress01;
            _lastProgressTicks = DateTime.UtcNow.Ticks;
        }
        try { ProgressChanged?.Invoke(progress01, text ?? ""); } catch { }
    }

    private async Task ConsumeStreamAsync(Stream stream, Action<string> onLine, CancellationToken ct)
    {
        var buffer = new byte[4096];
        var carry = new StringBuilder();
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            int read;
            try
            {
                read = await stream.ReadAsync(buffer, 0, buffer.Length, ct);
            }
            catch (OperationCanceledException)
            {
                return;
            }
            if (read <= 0)
                break;

            for (var i = 0; i < read; i++)
            {
                var b = buffer[i];
                char c = (b >= 32 && b <= 126) ? (char)b : ' ';
                if (b == (byte)'\n' || b == (byte)'\r')
                {
                    if (carry.Length > 0)
                    {
                        var s = carry.ToString();
                        carry.Clear();
                        try { onLine?.Invoke(s); } catch { }
                    }
                }
                else
                {
                    carry.Append(c);
                }
            }
            if (carry.Length > 0)
            {
                var s = carry.ToString();
                carry.Clear();
                try { onLine?.Invoke(s); } catch { }
            }
        }
    }

    private void TryReportProgressFromLine(string line)
    {
        if (string.IsNullOrWhiteSpace(line))
            return;

        var mPct = Regex.Match(line, @"(\d+(?:[.,]\d+)?)%");
        if (mPct.Success && float.TryParse(mPct.Groups[1].Value.Replace(',', '.'), out var pct))
        {
            ReportProgress(0.10f + 0.80f * Mathf.Clamp01(pct / 100f), line);
            return;
        }

        if (Regex.IsMatch(line, @"->\s+.+?\s+done$", RegexOptions.IgnoreCase))
        {
            ReportProgress(0.90f, line);
            return;
        }

        var mFrac = Regex.Match(line, @"(\d+)\s*/\s*(\d+)");
        if (mFrac.Success &&
            int.TryParse(mFrac.Groups[1].Value, out var a) &&
            int.TryParse(mFrac.Groups[2].Value, out var b) &&
            b > 0)
        {
            var p = Mathf.Clamp01(a / (float)b);
            ReportProgress(0.10f + 0.80f * p, line);
        }
    }

    private async Task PseudoProgressAsync(CancellationToken ct)
    {
        var p = 0.10f;
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            float last;
            long ticks;
            lock (_progressLock)
            {
                last = _lastProgress;
                ticks = _lastProgressTicks;
            }
            if (last >= 0.90f)
                return;

            var age = TimeSpan.FromTicks(DateTime.UtcNow.Ticks - ticks);
            if (age.TotalMilliseconds > 700)
            {
                p = Mathf.Min(0.90f, p + 0.0035f);
                if (p > last)
                    ReportProgress(p, "推理中…");
            }

            await Task.Delay(100, ct);
        }
    }

    private static Texture2D ResizeTextureBilinear(Texture2D src, int w, int h)
    {
        if (src == null || w <= 0 || h <= 0)
            return null;
        var prev = RenderTexture.active;
        var rt = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        rt.enableRandomWrite = false;
        rt.Create();
        try
        {
            var prevFilter = src.filterMode;
            src.filterMode = FilterMode.Bilinear;
            Graphics.Blit(src, rt);
            src.filterMode = prevFilter;

            RenderTexture.active = rt;
            var tex = new Texture2D(w, h, TextureFormat.RGBA32, false, false);
            tex.ReadPixels(new Rect(0, 0, w, h), 0, 0, false);
            tex.Apply(false, false);
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            return tex;
        }
        catch
        {
            return null;
        }
        finally
        {
            RenderTexture.active = prev;
            try { rt.Release(); } catch { }
            Destroy(rt);
        }
    }

    private static int InferModelFactor(string model)
    {
        if (string.IsNullOrWhiteSpace(model))
            return 4;
        var m = Regex.Match(model, @"(?:x(\d+)|(\d+)x)", RegexOptions.IgnoreCase);
        if (!m.Success)
            return 4;
        var s = m.Groups[1].Success ? m.Groups[1].Value : m.Groups[2].Value;
        if (!int.TryParse(s, out var v))
            return 4;
        if (v < 2) v = 2;
        if (v > 4) v = 4;
        return v;
    }

    private static int RoundToMultiple(int value, int multiple)
    {
        if (multiple <= 1)
            return value;
        var r = value % multiple;
        if (r == 0)
            return value;
        var down = value - r;
        var up = down + multiple;
        if (down <= 0)
            return up;
        return (value - down) <= (up - value) ? down : up;
    }
}

public struct RealEsrganResult
{
    public Texture2D texture;
    public string workDir;
    public string error;
}
