using UnityEngine;
using UnityEditor;
using System;
using System.IO;
using ICSharpCode.SharpZipLib.Zip; // SharpZipLib 核心命名空间

// 标记为编辑器加载时初始化，打开工程自动执行
[InitializeOnLoad]
public class ZFBrowserAutoExtractor
{
    // ========== 你的文件路径 ==========
    private const string CompressedFile = "Assets/ZFBrowser/Plugins/w64/zf_cef.dll.zip";
    private const string TargetDllPath = "Assets/ZFBrowser/Plugins/w64/zf_cef.dll";
    // =================================

    // 静态构造函数：Unity 打开工程时自动执行
    static ZFBrowserAutoExtractor()
    {
        // 延迟1秒执行（等 Unity 工程加载完成）
        EditorApplication.delayCall += CheckAndExtract;
    }

    /// <summary>
    /// 检查 DLL 是否存在，不存在则自动解压
    /// </summary>
    private static void CheckAndExtract()
    {
        // 1. 检查目标 DLL 是否已存在，存在则直接返回
        if (File.Exists(TargetDllPath))
        {
            Debug.Log("zf_cef.dll 已存在，无需解压");
            return;
        }

        // 2. 检查 ZIP 压缩包是否存在
        if (!File.Exists(CompressedFile))
        {
            EditorUtility.DisplayDialog("提示", 
                $"未找到压缩包：{CompressedFile}\n请确认压缩包已上传到仓库", 
                "确定");
            return;
        }

        // 4. 使用 SharpZipLib 解压 ZIP 包
        try
        {
            // 创建目标目录（如果不存在）
            string targetDir = Path.GetDirectoryName(TargetDllPath);

            // 调用 SharpZipLib 解压
            ExtractZipWithSharpZipLib(CompressedFile, targetDir);

            // 检查解压结果
            if (File.Exists(TargetDllPath))
            {
                AssetDatabase.Refresh();
                Debug.Log("zf_cef.dll 解压成功！");
            }
            else
            {
                // 额外检查：压缩包内 DLL 路径不同的情况
                string[] dllFiles = Directory.GetFiles(targetDir, "zf_cef.dll", SearchOption.AllDirectories);
                if (dllFiles.Length > 0)
                {
                    File.Move(dllFiles[0], TargetDllPath);
                    AssetDatabase.Refresh();
                    Debug.Log($"zf_cef.dll 解压成功（路径修正）：{dllFiles[0]} → {TargetDllPath}");
                }
                else
                {
                    throw new Exception("解压完成但未找到 zf_cef.dll，请检查压缩包内容");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"解压 zf_cef.dll 出错：{e.Message}");
        }
    }

    /// <summary>
    /// 使用 SharpZipLib 解压 ZIP 包（纯工程内依赖，无外部工具）
    /// </summary>
    private static void ExtractZipWithSharpZipLib(string zipPath, string extractPath)
    {
        // 确保解压路径末尾不带分隔符
        extractPath = extractPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

        // 打开 ZIP 文件流
        using (FileStream fs = File.OpenRead(zipPath))
        using (ZipInputStream zis = new ZipInputStream(fs))
        {
            ZipEntry zipEntry;
            // 遍历 ZIP 内的所有文件/目录
            while ((zipEntry = zis.GetNextEntry()) != null)
            {
                // 跳过空条目和目录
                if (string.IsNullOrEmpty(zipEntry.Name) || zipEntry.IsDirectory)
                {
                    continue;
                }

                // 拼接目标文件路径（处理 ZIP 内的路径）
                string entryFileName = zipEntry.Name;
                // 修复跨平台路径分隔符问题
                entryFileName = entryFileName.Replace('/', Path.DirectorySeparatorChar);
                string destinationPath = Path.Combine(extractPath, entryFileName);

                // 创建目标目录（如果不存在）
                string destinationDir = Path.GetDirectoryName(destinationPath);
                if (!Directory.Exists(destinationDir))
                {
                    Directory.CreateDirectory(destinationDir);
                }

                // 解压文件到目标路径（覆盖已存在的文件）
                using (FileStream streamWriter = File.Create(destinationPath))
                {
                    byte[] buffer = new byte[4096];
                    int bytesRead;
                    while ((bytesRead = zis.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        streamWriter.Write(buffer, 0, bytesRead);
                    }
                }

                Debug.Log($"解压文件：{entryFileName} → {destinationPath}");
            }
        }
    }

    // 手动触发解压的菜单（备用）
    [MenuItem("Tools/ZFBrowser/手动解压ZFBrowser")]
    private static void ManualExtract()
    {
        CheckAndExtract();
    }
}