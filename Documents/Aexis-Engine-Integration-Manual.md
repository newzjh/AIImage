# Aexis 端侧推理引擎接入手册

## 1. 定位与发布边界

Aexis 的 UPM 包名和根命名空间为 `com.aexis` / `Aexis`。AIImage 是使用 Aexis 的例子工程，不能成为引擎 Runtime 的程序集、命名空间或资源路径依赖。发布时只分发一个 Unity Package：`Packages/com.aexis`。

当前实现自带 ONNX 解析、NCNN `.param/.bin` 读取、模型图加载、Pack4 纹理 GPU 推理、计算着色器和形状/索引执行模块。Runtime 不引用 Unity Sentis、ncnn、ONNX Runtime、MNN、UniTask 或原生推理插件。样例和 Editor 的动态 JSON 工具使用命名空间隔离的 `Aexis.Samples.Json` 源码副本；它们包含嵌套 token 遍历和诊断报告，不能以 Unity `JsonUtility` 等价替换。该副本源自 MIT 许可的 Json.NET 13.0.2，来源、固定提交、归档校验值和改写记录位于 `Samples~/AIImageApplicationExample/ThirdParty/AexisSampleJson/UPSTREAM.md`，不会与宿主工程的 Json.NET 冲突。MNN 未来增加时应落在 `Aexis.Mnn` asmdef 内，仍然由同一个 `com.aexis` 包交付。

兼容范围为 Unity `2022.3` 至 `6000.3`；发布验证分别使用 2022.3.9f1、2023.2.20f1c1 和 6000.3.2f1。最低 Package Manager 声明版本为 Unity `2022.3`。

## 2. 安装

嵌入式安装在宿主工程的 `Packages/manifest.json` 中添加：

```json
{
  "dependencies": {
    "com.aexis": "file:../AIImage/Packages/com.aexis"
  }
}
```

正式发布可改为 registry 或 Git URL。不要把 `Runtime` 目录拷贝进 `Assets`，否则计算 shader、asmdef 和包路径约束会失效。

## 3. 目录与程序集

| 路径/程序集 | 责任 |
| --- | --- |
| `Runtime/Core` / `Aexis` | 公共推理会话、张量、精度、量化契约 |
| `Runtime/Async` / `Aexis.Async` | BCL `Task` 逐帧调度 |
| `Runtime/Onnx` / `Aexis.Onnx` | ONNX protobuf 图读取与执行规划 |
| `Runtime/Ncnn` / `Aexis.Ncnn` | NCNN 图、权重、算子和 Pack4 纹理执行 |
| `Runtime/Execution` / `Aexis.Execution` | ONNX 形状/索引类 GPU 执行 |
| `Runtime/Resources/Aexis` | 由包自身加载的 compute shaders |
| `Editor` / `Aexis.Editor` | 仅 Editor 工具 |
| `Samples~` | 可选导入的脚本、示例模型和安装器 |
| `Tests/Editor` | Package 边界和规划测试 |

Runtime 下允许多个 asmdef，Unity 2022.3 至 Unity 6.3 都支持一个 UPM 包内的多程序集。它们是编译边界，不是多个 Package；用户只导入 `com.aexis`。

## 4. 核心接口

### 4.1 `Aexis`

- `IInferenceSession`：公共会话身份、后端类型、状态和 `Dispose` 契约。
- `IInferenceTensor`：公开 `TensorDescriptor` 和后端资源 ID；不直接暴露 ComputeBuffer。
- `TensorDescriptor`：保存逻辑 shape、物理 storage shape、layout、数据类型和调试名。shape 必须为正数。
- `ModelManifest`：模型精度、INT8/INT4 权重量化及校准契约。
- `InferenceContractException`：模型或执行契约无效时抛出。

### 4.2 `Aexis.Async`

`AexisAsync.YieldFrame()` 返回 BCL `Task`。Aexis 公共 API 不出现 UniTask；宿主可以自行用 UniTask、Task 或协程封装它。这样不会与宿主项目已有的 UniTask 包、GUID 或 asmdef 发生冲突。

### 4.3 `Aexis.Onnx`

```csharp
var model = OnnxModelReader.Read(onnxBytes);
Debug.Log($"{model.graph.name}: {model.graph.nodes.Count} nodes");
```

- `OnnxModelReader.Read(string|byte[])`：读取 ONNX `ModelProto`、图节点、输入/输出、initializer 和属性。
- `OnnxD3Importer.Import(...)`：生成 D3 导入元数据。
- `OnnxExecutionAdapter.TryAdapt(...)`：将受支持的 Shape/Size/NonZero/Compress/GatherND/Scatter/TopK/OneHot 类节点转换为执行节点。
- `OnnxExecutionShapePlanner.Validate(...)`：验证动态长度、容量和冲突策略。

ONNX reader 是图读取和 lowering 的入口，并不等同于“无条件执行任意 ONNX op”。应用须在模型导入阶段验证实际算子覆盖范围。

### 4.4 `Aexis.Ncnn`

```csharp
using System.IO;
using Aexis.Ncnn;

var session = NcnnInferenceSessionFactory.Create(new NcnnOps());
using var weights = new NcnnBinReader(new MemoryStream(binBytes, writable: false));
await session.LoadModelAsync(paramText, weights, progress =>
    Debug.Log($"{progress.stage} {progress.progress01:P0}"));

// input 必须是匹配模型输入的 Pack4 RenderTexture。
var output = session.ForwardPack4(input, inputPacks, "data");
// 使用 output 后，在组件销毁或切换模型时释放。
session.Release();
```

| API | 作用 |
| --- | --- |
| `NcnnOps` | 创建 package-owned compute 操作门面，加载包内 compute shader |
| `NcnnInferenceSessionFactory.Create` | 创建图会话，可自动/显式应用 manifest |
| `NcnnParamParser.Parse` | 解析 `.param` 文本用于检查或合并 |
| `NcnnBinReader` | 从 `Stream` 读取 `.bin`，模型加载完成后释放 |
| `NcnnGraphSession.LoadModel` | 同步加载 |
| `NcnnGraphSession.LoadModelAsync` | 可取消、逐帧让出的异步加载，返回 `Task` |
| `NcnnGraphSession.ForwardPack4` | 执行 Pack4 RenderTexture 推理 |
| `NcnnGraphSession.Release` | 释放会话持有的 GPU 资源，允许重复调用 |

生产路径必须保持 Pack4 RenderTexture 和 CommandBuffer 兼容的纹理流。不要因为某一层缺失而把普通推理回退到临时 ComputeBuffer；应补齐纹理实现或在严格模式下抛出可定位错误。对同时需要纹理和固定 Buffer 输入的 CommandBuffer 图，使用 `ForwardPack4WithFixedInputs`：固定 Buffer 只能在图边界由 GPU dispatch 上传为纹理（Embed token 使用精确 RFloat LinearMat），进入第一层后不允许保留或回退为 ComputeBuffer activation。

## 5. 完整应用样例

Package Manager 导入唯一的 **AIImage Main2 Application Example** 后，执行菜单：

`Aexis/Examples/Install Main2 Application StreamingAssets`

它会把 `Samples~/AIImageApplicationExample/StreamingAssets` 完整复制到 `Assets/StreamingAssets`。这是 Unity Player 会打入包体的标准位置；运行时不会使用 `AssetDatabase`。随后打开 `Scenes/Main2.unity`，或使用 `Aexis/Examples/Open Main2 Application Scene`。

| 样例组件 | API | 默认用途 |
| --- | --- | --- |
| `AexisNcnnModelLoadRunner` | `LoadAsync`、`SelectModel`、`Release`、`LoadProgressChanged` | 读取任意 NCNN param/bin 对、创建会话、报告加载进度 |
| `AexisOnnxInspectionRunner` | `InspectAsync`、`ModelInspected` | 跨平台读取 ONNX 并输出图摘要 |
| `AexisSampleModelCatalog` | `ClipImageEncoder`、`CodeFormerEncoder`、`DeepFillV2`、`Matting`、`RealEsrgan`、`YoloV8Seg` | 预设模型默认路径 |
| `Aexis.Samples.Runners.ClipNcnnReproRunner` | `ProcessAsync(Texture2D, CancellationToken)` | MobileCLIP 图像分类；样例标签缓存避免依赖未分发的文本权重 |
| `Aexis.Samples.Runners.CodeFormerNcnnReproRunner2` | `ProcessAsync(Texture2D, CancellationToken)` | 人脸检测、对齐、修复和融合；使用样例 detector/encoder/generator |
| `Aexis.Samples.Runners.DeepFillV2Runner` | `ProcessAsync(Texture, Texture, CancellationToken)` | 遮罩修补，支持 ONNX-direct 或 NCNN |
| `Aexis.Samples.Runners.MatterNcnnReproRunner` | `ProcessAsync(Texture2D, CancellationToken)` | 抠图和 alpha 合成 |
| `Aexis.Samples.Runners.RealEsrganNcnnReproRunner` | `ProcessAsync(Texture2D, CancellationToken)` | 切片超分 |
| `Aexis.Samples.Runners.YoloSegNcnnReproRunner` | `ProcessAsync(Texture2D, CancellationToken)` | 分割检测、遮罩和叠加层 |

`AexisSampleStreamingAssets.ReadBytesAsync` 通过 `UnityWebRequest` 读取 `StreamingAssets`，因此可处理 Android 等不能直接 `File.ReadAllBytes` 的平台。所有 path 都相对 `Application.streamingAssetsPath`；默认模型目录为 `Clip`、`CodeFormer`、`DeepFileV2`、`Matting`、`RealESRGAN` 和 `Yolo`。

## 6. 现有 AIImage Runner 对应关系

| AIImage Runner | 包内可复用部分 | 默认模型是否携带 |
| --- | --- | --- |
| Clip | 完整 `ClipNcnnReproRunner`、`MobileClipSimpleTokenizer` | 是 |
| CodeFormer | 完整 `CodeFormerNcnnReproRunner2`、`NcnnFaceRegionGenerator`、`NcnnFaceRegionPaster` | 是 |
| DeepFillV2 | 完整 `DeepFillV2Runner`、ONNX 读取和 catalog | 是 |
| Matting | 完整 `MatterNcnnReproRunner` | 是 |
| RealESRGAN | 完整 `RealEsrganNcnnReproRunner` | 是 |
| YOLO Segmentation | 完整 `YoloSegNcnnReproRunner` 和 sample-owned `ImageProcessing.compute` | 是 |
| GFPGAN | 仅外部配置，模型不携带 | 否 |
| Stable Diffusion | 仅外部配置，模型不携带 | 否 |
| SD Inpainting | 仅外部配置，模型不携带 | 否 |
| MONAI/VISTA | 仅外部配置，模型不携带 | 否 |
| QWEN | 仅外部配置，模型不携带 | 否 |

迁入的 Runner 已位于 `Aexis.Samples.Runners` 并使用隔离的 `Aexis.Samples.Async`；它们不再编译依赖 Aexis Runtime 之外的 UniTask。`AIImage Main2 Application Example` 是唯一的应用样例，包含 Main2、MainView2、DesignView、LibraryView、所有 Runner（含 GFPGAN、Stable Diffusion、SD Inpainting、MONAI/VISTA、QWEN）及 Editor 测试/批处理代码。它带有 Clip、CodeFormer、DeepFillV2、Matting、RealESRGAN 和 YOLO 的默认模型；GFPGAN、Stable Diffusion、SD Inpainting、MONAI/VISTA、QWEN 的模型权重、外部 exe、私有 golden 和业务数据不随该 Sample 发布。样例将所需的 UniTask 和 SharpZip 源码隔离为 `Aexis.Samples.Async` 与 `Aexis.Samples.SharpZipLib`，不会与宿主工程的同名库冲突。SharpZip 仍被 `StandardImageIO` 和 MONAI 压缩输入读取路径调用，不能按未使用代码删除。

## 7. 模型与许可证

样例当前携带 Clip、CodeFormer、DeepFillV2、Matting、RealESRGAN、YOLO 的默认模型文件，共约 433 MB。GFPGAN、Stable Diffusion、SD Inpainting、MONAI/VISTA、QWEN 的权重不随包分发。

Aexis 源码使用 MIT，不自动改变模型权重许可证。每一个发布模型必须在 release tag 记录上游 URL、固定 revision/哈希、原始许可证、版权、转换步骤和再分发许可。无法完整核查时，发布归档必须移除对应模型，只保留 Runner 和路径配置。`RealESRGAN/LICENSE` 会随其样例文件复制；其余模型不能凭名称推断许可证。

## 8. 发布验证

1. 运行 `dotnet build AIImage.sln -v minimal -m:1`。
2. 用 Unity 2022.3.9f1、2023.2.20f1c1、6000.3.2f1 分别无界面编译验证工程。
3. 每个版本新建默认空工程，分别验证 `file:` UPM 安装和 `Aexis/Release/Export Complete UnityPackage` 导出的 `.unitypackage` 导入编译。UnityPackage 导入后必须恢复 `Packages/com.aexis`，并在该包的 `Samples~/AIImageApplicationExample` 保留完整 Main2 样例；确认没有 UniTask、AIImage、Sentis、ONNX Runtime 或 ncnn Runtime 依赖。
4. 在空工程导入 AIImage Main2 Application Example，运行 StreamingAssets 安装器后再次编译，确认 Main2 与六类携带模型的 runner 可发现。
5. 安装 Unity Test Framework 并设置 `AEXIS_INCLUDE_EDITOR_TESTS` 后，确认样例附带的 Editor NUnit 测试可被编译和发现；默认导入不应因测试框架缺失而失败。
6. 检查 package archive 不包含 GFPGAN、Stable Diffusion、SD Inpainting、MONAI/VISTA、QWEN 模型和未声明许可的二进制文件；Json.NET 仅可作为带许可证、来源、固定提交和校验值记录的 `Aexis.Samples.Json` 源码副本存在，禁止复制 `Newtonsoft.Json.dll` 到 `Assets` 或声明同名 UPM 依赖。
# Player Build Resource Staging

When a Player build includes the Main2 application sample, `AexisApplicationExamplePlayerBuildPreprocessor` stages every missing package-default sample resource into `Assets/StreamingAssets` before Unity collects Player data. This includes NCNN `.param` and `.bin` files, ONNX files, tokenizer files, and inference manifests. Existing project files are not overwritten, so large externally supplied GFPGAN, Stable Diffusion, MONAI/VISTA, and QWEN model files remain intact and are included by the normal Unity StreamingAssets build flow. Set `AEXIS_SKIP_SAMPLE_STREAMING_ASSETS_STAGING=1` only for a specialized build pipeline that intentionally replaces the whole StreamingAssets tree.

## P1 model contract additions

`AexisModelAsset` is now the package importer output for `.onnx`, `.param`, and `.aexis` files. It stores source bytes, the versioned Aexis binary graph, optional NCNN weights, and the stable import diagnostic report. `AexisNcnnBinaryParam` and `AexisModelArchive` are the offline/prepack equivalents; archives must be rebuilt whenever a graph schema, custom layer declaration, or weight payload changes.

Low-precision manifests may declare BF16, calibrated Pack4 INT8 activation plans, per-layer mixed precision, and a precision gate. Calibrated signed/unsigned INT8 plans directly configure Pack4 Conv/DWConv/Gemm/InnerProduct arithmetic on both immediate and CommandBuffer paths; mixed plans select each of those layers' FP16 or FP32/BF16 physical activation storage. Treat the gate as a required release check: compare the candidate output against the recorded FP32 baseline with `AexisPrecisionGateEvaluator`, then reject a variant that exceeds its recorded error/cosine limits.

Custom layers must be registered through `AexisCustomLayerRegistry` with a versioned schema and shader kernel id. The built-in P1 visual family (GridSample, DeformableConv2D, Fold, Flip, GLU, Einsum, Diag, SPP, ROI, Proposal, DetectionOutput, and YOLO output) uses the same schema/ABI preflight as its Pack4 dispatch. An unsupported profile is a terminal error and must not be worked around with a ComputeBuffer readback or fallback. BF16 remains texture-native through FP32 storage; Pack4 `Cast` provides deterministic BF16 rounding.
