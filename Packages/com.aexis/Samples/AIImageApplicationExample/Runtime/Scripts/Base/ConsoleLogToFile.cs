using System;
using System.Collections.Concurrent;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// 捕获Unity控制台日志，异步写入持久化目录的日志文件
/// 日志文件路径：Application.persistentDataPath/UnityGameLog_yyyyMMdd.log
/// </summary>
public class ConsoleLogToFile : MonoBehaviour
{
    // 日志文件写入流（全局单例保持打开，提升性能）
    private StreamWriter _logWriter;
    // 日志文件路径
    private string _logFilePath;

    private readonly ConcurrentQueue<string> _pendingLogs = new ConcurrentQueue<string>();
    private readonly SemaphoreSlim _pendingSignal = new SemaphoreSlim(0);
    private CancellationTokenSource _writerCts;
    private Task _writerTask;
    private volatile bool _stopping;
    
    // 单例实例，保证全局唯一
    public static ConsoleLogToFile Instance { get; private set; }

    private void Awake()
    {
        // 单例模式，防止重复创建
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject); // 切换场景不销毁

        // 初始化日志文件
        InitializeLogFile();

        StartWriterLoop();
        
        // 订阅Unity控制台日志事件
        Application.logMessageReceivedThreaded += OnLogReceived;
    }

    /// <summary>
    /// 初始化日志文件，创建文件并写入头部信息
    /// </summary>
    private void InitializeLogFile()
    {
        // 按日期生成日志文件名，避免单个文件过大
        string logFileName = $"UnityGameLog_{DateTime.Now:yyyyMMdd}.log";
        // 持久化目录（不同平台自动适配：Windows/Mac/Android/iOS）
        _logFilePath = Path.Combine(Application.persistentDataPath, logFileName);

        try
        {
            // 创建文件流，共享读写，异步写入
            FileStream fileStream = new FileStream(
                _logFilePath, 
                FileMode.OpenOrCreate,
                FileAccess.Write, 
                FileShare.Read, 
                4096,                     // 缓冲区大小
                useAsync: true);          // 启用异步IO

            // 初始化StreamWriter，使用UTF8编码
            _logWriter = new StreamWriter(fileStream, Encoding.UTF8);
            _logWriter.AutoFlush = false; // 关闭自动刷新，提升性能

            // 写入日志开头标记
            WriteLogHeader();
        }
        catch (Exception e)
        {
            Debug.LogError($"日志文件初始化失败：{e.Message}");
        }
    }

    /// <summary>
    /// 写入日志文件头部（启动时间）
    /// </summary>
    private void WriteLogHeader()
    {
        string header = $"==================== 游戏启动 - {DateTime.Now:yyyy-MM-dd HH:mm:ss} ====================\n";
        EnqueueLog(header);
    }

    /// <summary>
    /// 日志接收回调（Unity多线程安全的回调）
    /// </summary>
    private void OnLogReceived(string logText, string stackTrace, LogType type)
    {
        // 过滤空日志
        if (string.IsNullOrEmpty(logText)) return;

        // 格式化日志：时间 + 类型 + 内容
        string log = $"[{DateTime.Now:HH:mm:ss}] [{type}] {logText}\n";

        // 如果是错误/异常，追加堆栈信息
        if (type is LogType.Error or LogType.Exception or LogType.Assert)
        {
            log += $"堆栈信息：{stackTrace}\n";
        }

        // 异步写入文件（核心：不阻塞主线程）
        EnqueueLog(log);
    }

    private void StartWriterLoop()
    {
        if (_logWriter == null || _writerTask != null)
            return;

        _writerCts = new CancellationTokenSource();
        _writerTask = Task.Run(() => WriterLoopAsync(_writerCts.Token));
    }

    private void EnqueueLog(string log)
    {
        if (_stopping || _logWriter == null || string.IsNullOrEmpty(log))
            return;

        _pendingLogs.Enqueue(log);
        try
        {
            _pendingSignal.Release();
        }
        catch
        {
        }
    }

    private async Task WriterLoopAsync(CancellationToken token)
    {
        var sb = new StringBuilder(4096);
        while (!token.IsCancellationRequested)
        {
            try
            {
                await _pendingSignal.WaitAsync(token);
            }
            catch (OperationCanceledException)
            {
                break;
            }

            if (_logWriter == null)
                break;

            sb.Clear();
            int drained = 0;
            while (drained < 256 && _pendingLogs.TryDequeue(out var msg))
            {
                sb.Append(msg);
                drained++;
            }

            if (sb.Length == 0)
                continue;

            try
            {
                await _logWriter.WriteAsync(sb.ToString());
                await _logWriter.FlushAsync();
            }
            catch
            {
            }
        }

        if (_logWriter == null)
            return;

        try
        {
            sb.Clear();
            while (_pendingLogs.TryDequeue(out var msg))
                sb.Append(msg);

            if (sb.Length > 0)
            {
                await _logWriter.WriteAsync(sb.ToString());
                await _logWriter.FlushAsync();
            }
        }
        catch
        {
        }
    }

    /// <summary>
    /// 游戏退出/销毁时，关闭文件流
    /// </summary>
    private void OnDestroy()
    {
        _stopping = true;

        // 取消订阅事件
        Application.logMessageReceivedThreaded -= OnLogReceived;

        // 安全关闭文件流
        if (_writerCts != null)
        {
            try
            {
                _writerCts.Cancel();
                _pendingSignal.Release();
            }
            catch
            {
            }

            try
            {
                _writerTask?.Wait(2000);
            }
            catch
            {
            }
        }

        if (_logWriter != null)
        {
            try
            {
                _logWriter.FlushAsync().Wait();
            }
            catch
            {
            }
            _logWriter.Close();
            _logWriter.Dispose();
            _logWriter = null;
        }

        try
        {
            _writerCts?.Dispose();
        }
        catch
        {
        }
        _writerCts = null;
        _writerTask = null;
        
        Instance = null;
    }
}
