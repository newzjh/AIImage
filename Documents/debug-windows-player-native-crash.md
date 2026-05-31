# [OPEN] windows-player-native-crash

## Symptoms
- Windows 打包（Mono / IL2CPP）后：
  - 点击 Real-ESRGAN(ncnn原生) 或 GFPGAN(ncnn原生) 会导致 exe 直接崩溃
  - IL2CPP 下 Real-ESRGAN(外部进程) 启动进程时报“找不到文件”，Mono 下正常
- Unity Editor 内无上述问题

## Expected
- 打包后点击上述按钮不应导致崩溃
- IL2CPP/Mono 都能正确找到并启动 StreamingAssets 内的 realesrgan-ncnn-vulkan.exe

## Hypotheses
- H1: 打包后 native DLL 依赖缺失/加载路径不对（realesrgan_unity.dll / ncnn.dll / vulkan-1.dll 等），导致首次调用时 native 访问非法崩溃
- H2: 打包后 StreamingAssets 模型/权重文件路径不正确（GFPGAN: encoder/style；Real-ESRGAN: models），native 侧收到空路径或打不开文件导致崩溃
- H3: Vulkan 初始化在 Player 中失败（GPU/Vulkan loader/权限），ncnn::create_gpu_instance 或 VulkanDevice 创建阶段触发崩溃
- H4: IL2CPP 下 StreamingAssets 路径解析或 ProcessStartInfo 参数不同，导致外部进程路径实际不存在（或工作目录/引号导致）

## Evidence Plan
- 在 C#（按钮/Runner）添加运行时证据上报：记录所有关键路径、File.Exists、Directory.Exists、异常信息（DllNotFoundException/Win32Exception），并写入 persistentDataPath 以及 POST 到 Debug Server
- 在 Windows Player 上复现并收集 pre-fix 日志（runId=pre-fix）

## Status
- [OPEN] 等待采集 Player 运行时证据

