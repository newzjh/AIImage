using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
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
    public string outputFormat = "png";

    public event Action<float, string> ProgressChanged;
    private readonly object _progressLock = new object();
    private float _lastProgress;
    private long _lastProgressTicks;

    public async UniTask<RealEsrganResult> ProcessAsync(Texture2D src, bool dumpDebugFiles, CancellationToken ct)
    {
        if (!enableRealEsrgan)
            return new RealEsrganResult { error = "Real-ESRGAN disabled", workDir = null };
        if (src == null)
            return default;

        var originalW = src.width;
        var originalH = src.height;
        var totalSw = Stopwatch.StartNew();

        RealEsrganResult Finish(RealEsrganResult r)
        {
            r.elapsedMs = totalSw.ElapsedMilliseconds;
            try
            {
                Debug.Log("[TIMING] Real-ESRGAN(exe) " + r.elapsedMs + " ms | in=" + originalW + "x" + originalH + " | model=" + (modelName ?? "") + " | err=" + (r.error ?? ""));
            }
            catch
            {
            }
            return r;
        }

        var exePath = ResolveExecutablePath();
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        exePath = NormalizeWinPath(exePath);
#endif
#if ENABLE_IL2CPP && UNITY_STANDALONE_WIN && !UNITY_EDITOR
        exePath = await PrepareExeForIl2CppAsync(exePath, ct);
#endif
        await ReportDbgAsync("A", "realesrgan.exe.resolve", "[DEBUG] ResolveExecutablePath",
            "{\"exePath\":\"" + EscapeJson(exePath ?? "") + "\",\"streamingAssetsPath\":\"" + EscapeJson(Application.streamingAssetsPath) + "\",\"exists\":" + (File.Exists(exePath ?? "") ? 1 : 0) + "}",
            "", ct);
        if (string.IsNullOrWhiteSpace(exePath) || !File.Exists(exePath))
            return Finish(new RealEsrganResult { error = "realesrgan-ncnn-vulkan executable not found: " + (exePath ?? ""), workDir = null });

        var modelDir = ResolveModelDir();
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        modelDir = NormalizeWinPath(modelDir);
#endif
#if ENABLE_IL2CPP && UNITY_STANDALONE_WIN && !UNITY_EDITOR
        modelDir = await PrepareModelsForIl2CppAsync(modelDir, exePath, ct);
#endif
        await ReportDbgAsync("A", "realesrgan.exe.modeldir", "[DEBUG] ResolveModelDir",
            "{\"modelDir\":\"" + EscapeJson(modelDir ?? "") + "\",\"exists\":" + (Directory.Exists(modelDir ?? "") ? 1 : 0) + "}",
            "", ct);
        if (string.IsNullOrWhiteSpace(modelDir) || !Directory.Exists(modelDir))
            return Finish(new RealEsrganResult { error = "Real-ESRGAN model directory not found: " + (modelDir ?? ""), workDir = null });

        var s = Mathf.Clamp(scale, 2, 4);
        var model = string.IsNullOrWhiteSpace(modelName) ? "realesrgan-x4plus" : modelName.Trim();
        var modelFactor = InferModelFactor(model);
        var runFactor = modelFactor;

        var workDir = CreateWorkDir();
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        workDir = NormalizeWinPath(workDir);
#endif
        var inputPath = Path.Combine(workDir, "input.png");
        var outputPath = Path.Combine(workDir, "output.png");
        var scaledInputPath = Path.Combine(workDir, "input_scaled.png");
        var scaledOutputPath = Path.Combine(workDir, "output_scaled.png");

        try
        {
            var testPath = Path.Combine(workDir, "__write_test.tmp");
            File.WriteAllBytes(testPath, new byte[] { 1, 2, 3 });
            File.Delete(testPath);
        }
        catch (Exception e)
        {
            await ReportDbgAsync("D", "realesrgan.exe.workdir.not_writable", "[DEBUG] workDir not writable",
                "{\"workDir\":\"" + EscapeJson(workDir) + "\",\"msg\":\"" + EscapeJson(e.Message) + "\"}",
                "", ct);
        }

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

            var sw = originalW;
            var sh = originalH;
            if (maxSide > limit)
            {
                var scaleDown = (float)limit / maxSide;
                sw = Mathf.Max(1, Mathf.RoundToInt(originalW * scaleDown));
                sh = Mathf.Max(1, Mathf.RoundToInt(originalH * scaleDown));
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
            }

            if (sw != originalW || sh != originalH)
            {
                scaledInput = ResizeTextureBilinear(src, sw, sh);
                if (scaledInput == null)
                    return Finish(new RealEsrganResult { error = "Failed to resize input image", workDir = dumpDebugFiles ? workDir : null });

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

            //await UniTask.SwitchToThreadPool();

            var fmt = string.IsNullOrWhiteSpace(outputFormat) ? "png" : outputFormat.Trim().ToLowerInvariant();
            var args = BuildArgs(exePath, runInputPath, runOutputPath, runFactor, model, modelDir, fmt);

            var psi = new ProcessStartInfo
            {
                FileName = exePath,
                Arguments = args,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                WorkingDirectory = workDir ?? ""
            };
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            psi.FileName = NormalizeWinPath(psi.FileName);
            psi.WorkingDirectory = NormalizeWinPath(psi.WorkingDirectory);
#endif
#if ENABLE_IL2CPP && UNITY_STANDALONE_WIN && !UNITY_EDITOR
            psi.RedirectStandardOutput = false;
            psi.RedirectStandardError = false;
#endif

            string threadError = null;
            byte[] outBytes = null;

            try
            {
                var stdoutSb = new StringBuilder();
                var stderrSb = new StringBuilder();
                var sbLock = new object();
                int exitCode = -1;
                string il2cppLogText = "";
                try
                {
#if ENABLE_IL2CPP && UNITY_STANDALONE_WIN && !UNITY_EDITOR
                    var logPath = Path.Combine(workDir, "realesrgan_stdout.txt");
                    using var tailCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                    var tailTask = TailProgressFromFileAsync(logPath, tailCts.Token);

                    var exitCodeLocal = await UniTask.RunOnThreadPool(() =>
                    {
                        return StartProcessAndWaitWin32(psi.FileName, psi.Arguments, psi.WorkingDirectory, workDir, ct);
                    }, cancellationToken: ct);

                    exitCode = exitCodeLocal;
                    try { tailCts.Cancel(); } catch { }
                    try { await tailTask; } catch { }
                    try
                    {
                        var logText = "";
                        try { if (File.Exists(logPath)) logText = File.ReadAllText(logPath); } catch { }
                        il2cppLogText = logText ?? "";
                        await ReportDbgAsync("A", "realesrgan.exe.win32.exit", "[DEBUG] CreateProcess exit",
                            "{\"exitCode\":" + exitCode + ",\"outputExists\":" + (File.Exists(runOutputPath) ? 1 : 0) + ",\"outputPath\":\"" + EscapeJson(runOutputPath) + "\",\"logPath\":\"" + EscapeJson(logPath) + "\",\"logText\":\"" + EscapeJson(logText) + "\"}",
                            "", ct);
                    }
                    catch
                    {
                    }
#else
                    using (var p = Process.Start(psi))
                    {
                        if (p == null)
                        {
                            threadError = "Failed to start Real-ESRGAN process";
                        }
                        else
                        {
                            ReportProgress(0.05f, "开始推理");

#if !(ENABLE_IL2CPP && UNITY_STANDALONE_WIN && !UNITY_EDITOR)
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
#endif

                            using var pseudoCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                            var pseudoProgress = PseudoProgressAsync(pseudoCts.Token);

                            while (!p.HasExited)
                            {
                                ct.ThrowIfCancellationRequested();
                                await UniTask.Delay(50, cancellationToken: ct);
                            }

                            try { pseudoCts.Cancel(); } catch { }

#if !(ENABLE_IL2CPP && UNITY_STANDALONE_WIN && !UNITY_EDITOR)
                            await readStdout;
                            await readStderr;
#endif
                            try { await pseudoProgress; } catch { }
                            exitCode = p.ExitCode;
                        }
                    }
#endif
                }
                catch (OperationCanceledException)
                {
                    threadError = "Cancelled";
                }
                catch (Exception e)
                {
                    var win32 = e as System.ComponentModel.Win32Exception;
                    threadError = e.Message;
                    try
                    {
                        var realDir = Path.Combine(Application.streamingAssetsPath, "RealESRGAN");
                        var dirExists = Directory.Exists(realDir);
                        var files = dirExists ? string.Join("|", Directory.GetFiles(realDir)) : "";
                        await ReportDbgAsync("D", "realesrgan.exe.process.start.exception", "[DEBUG] Process.Start exception",
                            "{\"msg\":\"" + EscapeJson(e.Message) + "\",\"type\":\"" + EscapeJson(e.GetType().FullName) + "\",\"nativeErrorCode\":" + (win32 != null ? win32.NativeErrorCode : -1) + ",\"exePath\":\"" + EscapeJson(exePath) + "\",\"workingDir\":\"" + EscapeJson(psi.WorkingDirectory) + "\",\"args\":\"" + EscapeJson(args) + "\",\"realEsrganDir\":\"" + EscapeJson(realDir) + "\",\"realEsrganDirExists\":" + (dirExists ? 1 : 0) + ",\"files\":\"" + EscapeJson(files) + "\"}",
                            "", ct);
                    }
                    catch
                    {
                    }
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
#if ENABLE_IL2CPP && UNITY_STANDALONE_WIN && !UNITY_EDITOR
                        try
                        {
                            var files = "";
                            try { files = string.Join("|", Directory.GetFiles(workDir)); } catch { }
                            await ReportDbgAsync("D", "realesrgan.exe.output.missing", "[DEBUG] output not found after exitCode=0",
                                "{\"workDir\":\"" + EscapeJson(workDir) + "\",\"runOutputPath\":\"" + EscapeJson(runOutputPath) + "\",\"files\":\"" + EscapeJson(files) + "\"}",
                                "", ct);
                        }
                        catch
                        {
                        }
#endif
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
                return Finish(new RealEsrganResult { error = threadError, workDir = dumpDebugFiles ? workDir : null });

            var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            if (!tex.LoadImage(outBytes, false))
            {
                Destroy(tex);
                return Finish(new RealEsrganResult { error = "Failed to decode Real-ESRGAN output image", workDir = dumpDebugFiles ? workDir : null });
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
                    return Finish(new RealEsrganResult { error = "Failed to resize output back to original resolution", workDir = dumpDebugFiles ? workDir : null });
            }
            finalTex.name = "RealESRGAN_" + runFactor + "x";

            ReportProgress(1f, "完成");

            return Finish(new RealEsrganResult { texture = finalTex, workDir = dumpDebugFiles ? workDir : null });
        }
        finally
        {
            if (scaledInput != null)
                Destroy(scaledInput);
            if (!dumpDebugFiles && !string.IsNullOrWhiteSpace(workDir))
            {
                try { Directory.Delete(workDir, true); } catch { }
            }
        }
    }

    #region debug-point windows-player-native-crash-report
    private static string _dbgUrl;
    private static string _dbgSessionId;
    private static bool _dbgLoaded;

    private static void LoadDbgEnv()
    {
        if (_dbgLoaded) return;
        _dbgLoaded = true;
        _dbgUrl = "http://127.0.0.1:7778/event";
        _dbgSessionId = "windows-player-native-crash";
        try
        {
            var envName = "windows-player-native-crash.env";
            var candidates = new[]
            {
                Path.Combine(Application.persistentDataPath, ".dbg", envName),
                Path.Combine(Environment.CurrentDirectory, ".dbg", envName),
                Path.Combine(Application.dataPath, "..", ".dbg", envName)
            };
            for (var i = 0; i < candidates.Length; i++)
            {
                var envPath = candidates[i];
                if (!File.Exists(envPath))
                    continue;
                var lines = File.ReadAllLines(envPath);
                foreach (var raw in lines)
                {
                    var line = raw?.Trim();
                    if (string.IsNullOrEmpty(line)) continue;
                    if (line.StartsWith("DEBUG_SERVER_URL=", StringComparison.Ordinal))
                        _dbgUrl = line.Substring("DEBUG_SERVER_URL=".Length).Trim();
                    else if (line.StartsWith("DEBUG_SESSION_ID=", StringComparison.Ordinal))
                        _dbgSessionId = line.Substring("DEBUG_SESSION_ID=".Length).Trim();
                }
                break;
            }
        }
        catch
        {
        }
    }

    private static async UniTask ReportDbgAsync(string hypothesisId, string type, string msg, string dataJson, string traceId, CancellationToken ct)
    {
        try
        {
            LoadDbgEnv();
            if (string.IsNullOrWhiteSpace(hypothesisId)) hypothesisId = "A";
            if (string.IsNullOrWhiteSpace(dataJson)) dataJson = "{}";
            var payload =
                "{\"sessionId\":\"" + EscapeJson(_dbgSessionId) + "\"" +
                ",\"runId\":\"pre-fix\"" +
                ",\"hypothesisId\":\"" + EscapeJson(hypothesisId) + "\"" +
                ",\"location\":\"RealEsrganNcnnVulkanRunner\"" +
                ",\"msg\":\"" + EscapeJson(msg) + "\"" +
                ",\"type\":\"" + EscapeJson(type) + "\"" +
                ",\"traceId\":\"" + EscapeJson(traceId ?? "") + "\"" +
                ",\"ts\":" + DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() +
                ",\"data\":" + dataJson +
                "}";
            var line = payload + "\n";

            var logPath = Path.Combine(Application.persistentDataPath, "trae-debug-log-windows-player-native-crash.ndjson");
            try { File.AppendAllText(logPath, line); } catch { }
            try { await File.AppendAllTextAsync(logPath, line, ct); } catch { }

            if (!string.IsNullOrWhiteSpace(_dbgUrl))
            {
                try
                {
                    await UniTask.SwitchToMainThread();
                    using var req = new UnityWebRequest(_dbgUrl, "POST");
                    var body = System.Text.Encoding.UTF8.GetBytes(payload);
                    req.uploadHandler = new UploadHandlerRaw(body);
                    req.downloadHandler = new DownloadHandlerBuffer();
                    req.SetRequestHeader("Content-Type", "application/json");
                    var op = req.SendWebRequest();
                    while (!op.isDone)
                        await UniTask.Delay(15, cancellationToken: ct);
                }
                catch
                {
                }
            }
        }
        catch
        {
        }
    }

    private static string EscapeJson(string s)
    {
        if (string.IsNullOrEmpty(s)) return "";
        return s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\r", "\\r").Replace("\n", "\\n");
    }
    #endregion

#if ENABLE_IL2CPP && UNITY_STANDALONE_WIN && !UNITY_EDITOR
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct STARTUPINFO
    {
        public int cb;
        public string lpReserved;
        public string lpDesktop;
        public string lpTitle;
        public int dwX;
        public int dwY;
        public int dwXSize;
        public int dwYSize;
        public int dwXCountChars;
        public int dwYCountChars;
        public int dwFillAttribute;
        public int dwFlags;
        public short wShowWindow;
        public short cbReserved2;
        public IntPtr lpReserved2;
        public IntPtr hStdInput;
        public IntPtr hStdOutput;
        public IntPtr hStdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PROCESS_INFORMATION
    {
        public IntPtr hProcess;
        public IntPtr hThread;
        public int dwProcessId;
        public int dwThreadId;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct SECURITY_ATTRIBUTES
    {
        public int nLength;
        public IntPtr lpSecurityDescriptor;
        public int bInheritHandle;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateProcessW(
        string lpApplicationName,
        System.Text.StringBuilder lpCommandLine,
        IntPtr lpProcessAttributes,
        IntPtr lpThreadAttributes,
        bool bInheritHandles,
        uint dwCreationFlags,
        IntPtr lpEnvironment,
        string lpCurrentDirectory,
        ref STARTUPINFO lpStartupInfo,
        out PROCESS_INFORMATION lpProcessInformation);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateFileW(
        string lpFileName,
        uint dwDesiredAccess,
        uint dwShareMode,
        ref SECURITY_ATTRIBUTES lpSecurityAttributes,
        uint dwCreationDisposition,
        uint dwFlagsAndAttributes,
        IntPtr hTemplateFile);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint WaitForSingleObject(IntPtr hHandle, uint dwMilliseconds);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetExitCodeProcess(IntPtr hProcess, out uint lpExitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateProcess(IntPtr hProcess, uint uExitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);

    private static int StartProcessAndWaitWin32(string exePath, string args, string workingDir, string workDir, CancellationToken ct)
    {
        var si = new STARTUPINFO();
        si.cb = Marshal.SizeOf(typeof(STARTUPINFO));
        si.dwFlags = 0x00000001 | 0x00000100;
        si.wShowWindow = 0;

        var logPath = Path.Combine(workDir ?? "", "realesrgan_stdout.txt");
        IntPtr hFile = IntPtr.Zero;
        try
        {
            const uint GENERIC_WRITE = 0x40000000;
            const uint FILE_SHARE_READ = 0x00000001;
            const uint FILE_SHARE_WRITE = 0x00000002;
            const uint CREATE_ALWAYS = 2;
            const uint FILE_ATTRIBUTE_NORMAL = 0x00000080;
            var sa = new SECURITY_ATTRIBUTES
            {
                nLength = Marshal.SizeOf(typeof(SECURITY_ATTRIBUTES)),
                lpSecurityDescriptor = IntPtr.Zero,
                bInheritHandle = 1
            };
            hFile = CreateFileW(logPath, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, ref sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, IntPtr.Zero);
            if (hFile != IntPtr.Zero && hFile.ToInt64() != -1)
            {
                si.hStdOutput = hFile;
                si.hStdError = hFile;
            }
        }
        catch
        {
        }

        var cmd = new System.Text.StringBuilder();
        cmd.Append('"').Append(exePath).Append('"');
        if (!string.IsNullOrWhiteSpace(args))
        {
            cmd.Append(' ').Append(args);
        }
        const uint CREATE_NO_WINDOW = 0x08000000;
        if (!CreateProcessW(exePath, cmd, IntPtr.Zero, IntPtr.Zero, true, CREATE_NO_WINDOW, IntPtr.Zero, workingDir, ref si, out var pi))
        {
            var code = Marshal.GetLastWin32Error();
            throw new System.ComponentModel.Win32Exception(code, "CreateProcessW failed: " + exePath);
        }

        try
        {
            const uint WAIT_OBJECT_0 = 0x00000000;
            const uint WAIT_TIMEOUT = 0x00000102;
            while (true)
            {
                ct.ThrowIfCancellationRequested();
                var r = WaitForSingleObject(pi.hProcess, 50);
                if (r == WAIT_OBJECT_0)
                    break;
                if (r != WAIT_TIMEOUT)
                    break;
            }

            if (!GetExitCodeProcess(pi.hProcess, out var exit))
                return -1;
            return unchecked((int)exit);
        }
        catch (OperationCanceledException)
        {
            try { TerminateProcess(pi.hProcess, 0); } catch { }
            throw;
        }
        finally
        {
            try { CloseHandle(pi.hThread); } catch { }
            try { CloseHandle(pi.hProcess); } catch { }
            try
            {
                if (hFile != IntPtr.Zero && hFile.ToInt64() != -1)
                    CloseHandle(hFile);
            }
            catch { }
        }
    }
#endif

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
        var root = Application.persistentDataPath;
        if (string.IsNullOrWhiteSpace(root))
            root = Path.GetTempPath();
        var dir = Path.Combine(root, "RealESRGAN", "Work", "AIImage_RealESRGAN_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
        try { Directory.CreateDirectory(dir); } catch { }
        return dir;
    }

#if ENABLE_IL2CPP && UNITY_STANDALONE_WIN && !UNITY_EDITOR
    private async UniTask TailProgressFromFileAsync(string logPath, CancellationToken ct)
    {
        long pos = 0;
        var carry = new StringBuilder();
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                if (File.Exists(logPath))
                {
                    using (var fs = new FileStream(logPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                    {
                        if (pos > fs.Length)
                            pos = 0;
                        fs.Seek(pos, SeekOrigin.Begin);
                        var buf = new byte[4096];
                        int read;
                        while ((read = await fs.ReadAsync(buf, 0, buf.Length, ct)) > 0)
                        {
                            pos += read;
                            for (var i = 0; i < read; i++)
                            {
                                var b = buf[i];
                                if (b == (byte)'\n' || b == (byte)'\r')
                                {
                                    if (carry.Length > 0)
                                    {
                                        var line = carry.ToString();
                                        carry.Clear();
                                        TryReportProgressFromLine(line);
                                    }
                                }
                                else
                                {
                                    char c = (b >= 32 && b <= 126) ? (char)b : ' ';
                                    carry.Append(c);
                                }
                            }
                        }
                    }
                }
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch
            {
            }

            await UniTask.Delay(80, cancellationToken: ct);
        }
    }
#endif

#if ENABLE_IL2CPP && UNITY_STANDALONE_WIN && !UNITY_EDITOR
    private static async UniTask<string> PrepareExeForIl2CppAsync(string streamingExePath, CancellationToken ct)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(streamingExePath) || !File.Exists(streamingExePath))
                return streamingExePath;

            var srcDir = Path.GetDirectoryName(streamingExePath);
            if (string.IsNullOrWhiteSpace(srcDir) || !Directory.Exists(srcDir))
                return streamingExePath;

            var dstDir = Path.Combine(Application.persistentDataPath, "RealESRGAN", "bin");
            Directory.CreateDirectory(dstDir);

            var exeName = Path.GetFileName(streamingExePath);
            var dstExe = Path.Combine(dstDir, exeName);

            void CopyIfDifferent(string src, string dst)
            {
                try
                {
                    if (!File.Exists(src))
                        return;
                    var need = true;
                    if (File.Exists(dst))
                    {
                        var a = new FileInfo(src);
                        var b = new FileInfo(dst);
                        need = a.Length != b.Length || a.LastWriteTimeUtc != b.LastWriteTimeUtc;
                    }
                    if (need)
                        File.Copy(src, dst, true);
                }
                catch
                {
                }
            }

            CopyIfDifferent(streamingExePath, dstExe);
            var dlls = Directory.GetFiles(srcDir, "*.dll");
            for (var i = 0; i < dlls.Length; i++)
            {
                ct.ThrowIfCancellationRequested();
                var src = dlls[i];
                var dst = Path.Combine(dstDir, Path.GetFileName(src));
                CopyIfDifferent(src, dst);
                await UniTask.Yield();
            }
            return dstExe;
        }
        catch
        {
            return streamingExePath;
        }
    }
#endif

#if ENABLE_IL2CPP && UNITY_STANDALONE_WIN && !UNITY_EDITOR
    private static async UniTask<string> PrepareModelsForIl2CppAsync(string streamingModelDir, string exePath, CancellationToken ct)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(exePath) || !File.Exists(exePath))
                return streamingModelDir;

            var exeDir = Path.GetDirectoryName(exePath);
            if (string.IsNullOrWhiteSpace(exeDir))
                return streamingModelDir;

            var dstModelDir = Path.Combine(exeDir, "models");
            Directory.CreateDirectory(dstModelDir);

            if (!Directory.Exists(streamingModelDir))
                return dstModelDir;

            void CopyIfDifferent(string src, string dst)
            {
                try
                {
                    if (!File.Exists(src))
                        return;
                    var need = true;
                    if (File.Exists(dst))
                    {
                        var a = new FileInfo(src);
                        var b = new FileInfo(dst);
                        need = a.Length != b.Length || a.LastWriteTimeUtc != b.LastWriteTimeUtc;
                    }
                    if (need)
                        File.Copy(src, dst, true);
                }
                catch
                {
                }
            }

            var srcFiles = Directory.GetFiles(streamingModelDir);
            for (var i = 0; i < srcFiles.Length; i++)
            {
                ct.ThrowIfCancellationRequested();
                var src = srcFiles[i];
                var dst = Path.Combine(dstModelDir, Path.GetFileName(src));
                CopyIfDifferent(src, dst);
                await UniTask.Yield();
            }

            return dstModelDir;
        }
        catch
        {
            return streamingModelDir;
        }
    }
#endif

    private string BuildArgs(string exePath, string inputPath, string outputPath, int s, string model, string modelDir, string format)
    {
        var exeBase = (Path.GetFileNameWithoutExtension(exePath) ?? "").ToLowerInvariant();
        var isUpscayl = exeBase == "upscayl-bin";

        var useModelDirArg = !isUpscayl && !string.IsNullOrWhiteSpace(modelDir);

        var g = gpuId < 0 ? "auto" : gpuId.ToString();

        var sb = new StringBuilder();
        sb.Append("-v ");
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        inputPath = NormalizeWinPath(inputPath);
        outputPath = NormalizeWinPath(outputPath);
        modelDir = NormalizeWinPath(modelDir);
#endif
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
        if (!string.IsNullOrWhiteSpace(format) && !string.Equals(format, "png", StringComparison.OrdinalIgnoreCase))
            sb.Append(" -f ").Append(QuoteArg(format));
        return sb.ToString().Trim();
    }

#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
    private static string NormalizeWinPath(string p)
    {
        if (string.IsNullOrWhiteSpace(p))
            return p;
        try
        {
            var full = Path.GetFullPath(p);
            return full.Replace('/', '\\');
        }
        catch
        {
            return p.Replace('/', '\\');
        }
    }
#endif

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
    public long elapsedMs;
}
