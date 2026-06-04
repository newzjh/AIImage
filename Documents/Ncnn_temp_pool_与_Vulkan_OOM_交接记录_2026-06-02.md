## 背景

这轮工作的目标有两部分：

1. 给 `NcnnRepro / NcnnRepro2 / NcnnRepro3 / NcnnRepro4` 增加 `ComputeBuffer` 临时池，替代此前只对 `RenderTexture` 有池化入口的状态。
2. 用 Unity 静默批处理验证 `CodeFormer / GFPGAN / Matting / YoloSeg / ESRGAN` 在 `02.png` 上的显存、`RenderTexture`、`ComputeBuffer` 行为，定位 `Vulkan out of memory` / `Suboptimal memory type used for buffer/image because of low memory` 的来源。

当前日期：`2026-06-02`

## 本轮已完成的工作

### 1. 新增了 `ComputeBuffer` 临时池

新增文件：

- [Assets/Scripts/NcnnCompute/NcnnTempComputeBufferPool.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnTempComputeBufferPool.cs)

作用：

- 统一管理 `ComputeBuffer` 的临时租借/归还。
- 支持按 `(count, stride)` 复用。
- 增加了容量限制：
  - `MaxSingleBufferBytes`
  - `MaxTotalPooledBytes`
- 增加了失效 buffer 容错，避免池中残留死句柄后再访问 `buffer.count / buffer.stride` 触发空引用/底层异常。

### 2. `NcnnTensorBuffer` 支持自定义归还回调

修改：

- [Assets/Scripts/NcnnCompute/NcnnTensorBuffer.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnTensorBuffer.cs)

作用：

- `ownsBuffer=true` 时可通过回调把 buffer 归还到池，而不是直接 `Dispose()`。

### 3. GPU 资源跟踪器增加了 reuse 轨迹

修改：

- [Assets/Scripts/NcnnCompute/NcnnGpuResourceTracker.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnGpuResourceTracker.cs)

新增：

- `ReuseBuffer(...)`
- `ReuseTexture(...)`

作用：

- 在报告里可区分：
  - `alloc_buffer`
  - `reuse_buffer`
  - `alloc_rt`
  - `reuse_rt`

### 4. `NcnnRepro*` 四条实现已接入 `ComputeBuffer` 池

修改文件：

- [Assets/Scripts/NcnnCompute/NcnnRepro.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro.cs)
- [Assets/Scripts/NcnnCompute/NcnnRepro2.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro2.cs)
- [Assets/Scripts/NcnnCompute/NcnnRepro3.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro3.cs)
- [Assets/Scripts/NcnnCompute/NcnnRepro4.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro4.cs)

已做的事情：

- 大量 `new ComputeBuffer(...)` 改成 `RentTempBuffer(...)`
- `InferResult.Dispose()`
- 中途 `Consume(...)`
- `GetOrConvertToBuffer(...)`
- 若干 `Crop / Shuffle / Softmax / MatMul / Attention / CompareTextureConvPath` 路径

已确认修掉的一类问题：

- 用户贴出来的这条异常：

```text
NullReferenceException
UnityEngine.ComputeBuffer.get_count ()
NcnnCompute.NcnnTempComputeBufferPool.GarbageCollect ()
```

根因是池里混入了失效的 `ComputeBuffer`，后来在 GC / Rent 时访问 `count`。  
现已在 `NcnnTempComputeBufferPool` 中做元数据缓存和失效容错。

### 5. `EnableTempPool` 语义已改为“只控制 buffer 池”

这轮按用户要求，开始移除 C# 层 `RenderTexture` 池。

当前状态：

- `EnableTempPool` 现在目标语义是：只控制 `ComputeBuffer` 池。
- `RenderTexture` 申请改回直接走 Unity 内部 `RenderTexture.GetTemporary / ReleaseTemporary`。
- `RenderTexture` 的数量和字节仍然继续通过 `NcnnGpuResourceTracker` 跟踪。

已改文件：

- [Assets/Scripts/NcnnCompute/NcnnRepro.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro.cs)
- [Assets/Scripts/NcnnCompute/NcnnRepro2.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro2.cs)
- [Assets/Scripts/NcnnCompute/NcnnRepro3.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro3.cs)
- [Assets/Scripts/NcnnCompute/NcnnRepro4.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro4.cs)

注意：

- 这部分改动已经落地，但当前工作树尚未完成最终稳定性回归，见“遗留问题”章节。

### 6. 增加了统一静默 stress 入口

修改：

- [Assets/Editor/NcnnDebugRunner.cs](/E:/Projects/AIImage/Assets/Editor/NcnnDebugRunner.cs)

新增入口：

- `RunReproSuiteStressBatch`

用途：

- 在同一 Unity/Vulkan 进程里顺序跑：
  - `ESRGAN`
  - `YoloSeg`
  - `Matting`
  - `GFPGAN`
  - `CodeFormer`
- 记录每轮：
  - `elapsed_ms`
  - `gfx_mb`
  - `managed_mb`
  - `rt_objects`
  - `NcnnGpuResourceTracker.BuildSummary()`
- 同时写每个 runner 的独立 GPU 报告

## 这轮跑出来的关键结论

### 1. `CodeFormer` 的峰值主体仍然是 `ComputeBuffer`

可信日志：

- [Logs/codeformer-bufferpool-check.log](/E:/Projects/AIImage/Logs/codeformer-bufferpool-check.log)

可信报告：

- [C:/Users/hc/AppData/Local/Temp/YanQi/AIImage/AIImage_CodeFormerRepro2_20260602_162349/codeformer_gpu_resources.txt](/C:/Users/hc/AppData/Local/Temp/YanQi/AIImage/AIImage_CodeFormerRepro2_20260602_162349/codeformer_gpu_resources.txt)

可信结论：

- `CodeFormer Debug result | error= | elapsedMs=41238`
- `reuse_buffer=82`
- `reuse_rt=309`
- 说明：
  - `ComputeBuffer` 池是实际命中的
  - `CodeFormer` 单图路径是能跑通的

### 2. 多 runner 来回跑时，Vulkan 驱动内存会明显累积

可信报告目录之一：

- [C:/Users/hc/AppData/Local/Temp/YanQi/AIImage/AIImage_ReproSuiteStress_20260602_170638](/C:/Users/hc/AppData/Local/Temp/YanQi/AIImage/AIImage_ReproSuiteStress_20260602_170638)

该轮主结论：

- `YoloSeg / Matting / CodeFormer` 路径在 tracked 资源层面可以跑完。
- 日志中出现大量：

```text
Vulkan - Suboptimal memory type used for buffer because of low memory
Vulkan - Suboptimal memory type used for image because of low memory
```

说明：

- OOM 风险并不只是“代码里没还临时对象”。
- 还存在明显的 Vulkan/Unity 驱动侧缓存与内部分配压力。

### 3. 去掉 C# RT 池后，部分路径的 tracked RT 峰值确实下降

可信短回归目录：

- [C:/Users/hc/AppData/Local/Temp/YanQi/AIImage/AIImage_ReproSuiteStress_20260602_173309](/C:/Users/hc/AppData/Local/Temp/YanQi/AIImage/AIImage_ReproSuiteStress_20260602_173309)

可观察到：

- `ESRGAN` tracked RT 峰值从之前几百 MB 高位明显下降到了约 `168.6MB`
- `YoloSeg` / `Matting` 的 `post_destroy` tracked `live_buffers/live_rts` 已能回到 `0/0`

## 当前遗留问题

### A. 当前工作树还不能正式交付

原因：

- `GFPGAN`
- `CodeFormer`

这两条路径在后续回归中又出现了新的异常，且异常都沿着共享的人脸检测链传播。

### B. 当前主要 blocker：`NcnnFaceRegionGenerator` 清理阶段异常

重点文件：

- [Assets/Scripts/NcnnFaceRegionGenerator.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnFaceRegionGenerator.cs)

重点代码位置：

- [NcnnFaceRegionGenerator.cs:442](/E:/Projects/AIImage/Assets/Scripts/NcnnFaceRegionGenerator.cs:442)

当前 `finally`：

```csharp
finally
{
    if (letterbox != null)
        DestroyObjectSafe(letterbox);
    if (inputPack4 != null)
        _repro?.ReturnTempArray(inputPack4);
    _repro?.ClearTempPool();
}
```

现象：

- `CodeFormer` / `GFPGAN` 在某些 suite 回归里报：
  - `Object reference not set to an instance of an object.`
- 栈最终折叠到：
  - `NcnnFaceRegionGenerator.GenerateAsync`
  - `NcnnFaceRegionGenerator.EnsureLoaded`
  - 日志显示异常最终被归因到 `NcnnFaceRegionGenerator.cs:449`

高概率原因：

- 主体推理并未失败
- `finally` 中清理时二次归还 / 二次清池 / 清理时机竞争，导致异常覆盖了原本成功结果

下一位应优先做的事：

1. 给 `NcnnFaceRegionGenerator.GenerateAsync` 的 `finally` 加逐项 `try/catch + log`
2. 分别确认：
   - 是 `ReturnTempArray(inputPack4)` 在炸
   - 还是 `ClearTempPool()` 在炸
3. 不要先大范围 suite，先单独恢复：
   - `RunCodeFormerDebugBatch`
   - `RunGfpganDebugBatch`

### C. `NcnnRepro` 老路径仍有 `RenderTexture` 生命周期未完全收干净

重点文件：

- [Assets/Scripts/NcnnCompute/NcnnRepro.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro.cs)

现象：

- 即使去掉了 C# RT 池，`ESRGAN` / `GFPGAN` 这类基于 `NcnnRepro` 的路径，tracked `live_rts` 仍可能残留非 0。
- 例如短回归里：
  - `ESRGAN post_destroy live_rts=40`
  - `GFPGAN post_destroy live_rts=14`

说明：

- 这些不是 C# RT 池造成的唯一问题
- `NcnnRepro` 内部仍有一部分 RT 生命周期和 tracker 释放点没有完全闭合

需要继续核对：

- `ForwardPack4(...)`
- `InferCore(...)`
- `InferCoreLegacy(...)`
- 所有 `ExtractTexture(...)` 后的所有权转移
- `GFPGAN` runner 自己创建/返还的中间 RT 链

### D. `Vulkan - Suboptimal memory type used for buffer/image because of low memory` 仍未彻底解决

这轮已做尝试：

1. 去掉 C# RT 池
2. 限制 buffer 池只保留中小块
3. runner 结束时主动 `ClearTempPool()`

结论：

- 有改善，但还不足以宣告解决
- 真正的 Vulkan 压力还需要在“所有路径都稳定跑通”之后再做一轮 pool-on / pool-off 对照

## 当前哪些日志可信，哪些不可信

### 可信

- [Logs/codeformer-bufferpool-check.log](/E:/Projects/AIImage/Logs/codeformer-bufferpool-check.log)
- [Logs/codeformer-02-after-poolfix.log](/E:/Projects/AIImage/Logs/codeformer-02-after-poolfix.log)
- [C:/Users/hc/AppData/Local/Temp/YanQi/AIImage/AIImage_CodeFormerRepro2_20260602_162349/codeformer_gpu_resources.txt](/C:/Users/hc/AppData/Local/Temp/YanQi/AIImage/AIImage_CodeFormerRepro2_20260602_162349/codeformer_gpu_resources.txt)
- [C:/Users/hc/AppData/Local/Temp/YanQi/AIImage/AIImage_ReproSuiteStress_20260602_170638/suite_summary.txt](/C:/Users/hc/AppData/Local/Temp/YanQi/AIImage/AIImage_ReproSuiteStress_20260602_170638/suite_summary.txt)

### 不要再直接引用作最终结论

- [Logs/gfpgan-02-after-poolfix.log](/E:/Projects/AIImage/Logs/gfpgan-02-after-poolfix.log)

原因：

- 这份日志来自一轮并行/时序不干净的 batch 调用，结果只有 Unity 启动头，没有有效推理结论。

## 下一位建议调试顺序

### 第 1 步：先修共享人脸检测链

优先文件：

- [Assets/Scripts/NcnnFaceRegionGenerator.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnFaceRegionGenerator.cs)

建议做法：

1. 在 `finally` 中拆开：
   - `ReturnTempArray(inputPack4)`
   - `ClearTempPool()`
2. 各自单独 `try/catch`
3. 把异常日志写清楚到 `Debug.LogError`
4. 跑下面两条命令确认哪一步炸

### 第 2 步：先让这两条单图 batch 恢复

```powershell
$env:AIIMAGE_DEBUG_INPUT='E:\Projects\AIImage\ref\CodeFormer-ncnn-main\data\02.png'
& 'C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe' `
  -batchmode `
  -projectPath 'E:\Projects\AIImage' `
  -executeMethod NcnnDebugRunner.RunCodeFormerDebugBatch `
  -logFile 'E:\Projects\AIImage\Logs\codeformer-02-rerun.log'
```

```powershell
$env:AIIMAGE_DEBUG_INPUT='E:\Projects\AIImage\ref\CodeFormer-ncnn-main\data\02.png'
& 'C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe' `
  -batchmode `
  -projectPath 'E:\Projects\AIImage' `
  -executeMethod NcnnDebugRunner.RunGfpganDebugBatch `
  -logFile 'E:\Projects\AIImage\Logs\gfpgan-02-rerun.log'
```

验收标准：

- 日志里有：
  - `RunCodeFormerDebugBatch done`
  - `RunGfpganDebugBatch done`
- 不应再出现：
  - `NullReferenceException`
  - `Object reference not set to an instance of an object.`

### 第 3 步：再跑整套 suite

```powershell
$env:AIIMAGE_DEBUG_INPUT='E:\Projects\AIImage\ref\CodeFormer-ncnn-main\data\02.png'
$env:AIIMAGE_STRESS_COUNT='5'
$env:AIIMAGE_REPRO_TEMP_POOL='true'
& 'C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe' `
  -batchmode `
  -projectPath 'E:\Projects\AIImage' `
  -executeMethod NcnnDebugRunner.RunReproSuiteStressBatch `
  -logFile 'E:\Projects\AIImage\Logs\repro-suite-02-rerun.log'
```

重点看：

- `suite_summary.txt`
- 各 runner 的 `*_gpu_resources.txt`
- 是否仍出现：
  - `Vulkan - Suboptimal memory type used for buffer because of low memory`
  - `Vulkan - Suboptimal memory type used for image because of low memory`

### 第 4 步：最后再做 pool on/off 对照

等所有路径都能稳定跑通后，再做：

```powershell
$env:AIIMAGE_REPRO_TEMP_POOL='true'
```

和

```powershell
$env:AIIMAGE_REPRO_TEMP_POOL='false'
```

两轮对照，比较：

- `gfx_mb`
- `peak_buffers_mb`
- `peak_rts_mb`
- Vulkan low-memory warning 次数

## 本轮主要修改文件

- [Assets/Scripts/NcnnCompute/NcnnTempComputeBufferPool.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnTempComputeBufferPool.cs)
- [Assets/Scripts/NcnnCompute/NcnnGpuResourceTracker.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnGpuResourceTracker.cs)
- [Assets/Scripts/NcnnCompute/NcnnTensorBuffer.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnTensorBuffer.cs)
- [Assets/Scripts/NcnnCompute/NcnnRepro.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro.cs)
- [Assets/Scripts/NcnnCompute/NcnnRepro2.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro2.cs)
- [Assets/Scripts/NcnnCompute/NcnnRepro3.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro3.cs)
- [Assets/Scripts/NcnnCompute/NcnnRepro4.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnCompute/NcnnRepro4.cs)
- [Assets/Scripts/RealEsrganNcnnReproRunner.cs](/E:/Projects/AIImage/Assets/Scripts/RealEsrganNcnnReproRunner.cs)
- [Assets/Scripts/GfpganNcnnReproRunner.cs](/E:/Projects/AIImage/Assets/Scripts/GfpganNcnnReproRunner.cs)
- [Assets/Scripts/YoloSegNcnnReproRunner.cs](/E:/Projects/AIImage/Assets/Scripts/YoloSegNcnnReproRunner.cs)
- [Assets/Scripts/MatterNcnnReproRunner.cs](/E:/Projects/AIImage/Assets/Scripts/MatterNcnnReproRunner.cs)
- [Assets/Scripts/NcnnFaceRegionGenerator.cs](/E:/Projects/AIImage/Assets/Scripts/NcnnFaceRegionGenerator.cs)
- [Assets/Scripts/CodeFormerNcnnReproRunner2.cs](/E:/Projects/AIImage/Assets/Scripts/CodeFormerNcnnReproRunner2.cs)
- [Assets/Editor/NcnnDebugRunner.cs](/E:/Projects/AIImage/Assets/Editor/NcnnDebugRunner.cs)

## 一句话交接结论

最初那条 `ComputeBuffer` 池 NRE 已经针对性修掉；`CodeFormer` 单图 batch 一度恢复成功；当前新的主要问题已经收敛到 `GFPGAN/CodeFormer` 共用的 `NcnnFaceRegionGenerator` 清理阶段异常，以及 `NcnnRepro` 老路径 RT 生命周期仍未完全闭环。下一位应先把共享人脸检测链修稳，再回到整套 Vulkan 压力回归。
