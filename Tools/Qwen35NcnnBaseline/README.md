# Qwen3.5 0.8B ncnn 外部金标基线

本目录把 `Documents/Qwen3.5 多模态大语言模型转换部署到 ncnn 框架.pdf` 和
`ref/ncnn_llm-main` 固化为三层基线：

1. 山东大学镜像中的原始 ncnn 模型及字节级校验。
2. NumPy FP32 的 `ShortConv`、`GatedDeltaRule`、视觉 patch/RoPE 数值金标。
3. 未进入 Unity 产品的外部 `llm_ncnn_run` 端到端图文金标。

最终产品实现只能使用 Unity C# 和 ComputeShader。外部 C++ CLI 仅用于产生金标，不能作为
Unity 原生插件、P/Invoke 后端或移动端 fallback。

## 已验证结果

2026-07-18 在 Windows x64 CPU 上执行等价命令：

```text
./llm_ncnn_run --model ./assets/qwen3.5_0.8b --image test.jpg
```

Python 验收器向交互式 stdin 输入固定 OCR 提示并在 130.38 秒后得到退出码 0。输出命中：

- `仍未忘跟你約定`
- `決心忘記我便記不起`
- `剪影的你輪廓太好看`
- `還記得當天旅館的門牌`

完整输出、命令、图片/可执行文件 SHA-256 和检查项位于：

- `reports/reference_cli_validation.json`
- `reports/reference_cli_validation.stdout.txt`
- `reports/reference_cli_validation.stderr.txt`

当前参考工程原先默认注入 `random/add` 演示工具，会让该 OCR 命令错误地调用 `add`。基线已把
演示工具改为只有显式传入 `--builtin-tools` 才启用，因此文档原命令可直接产生正确结果。

## 模型下载

唯一允许的模型基址：

<https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3.5_0.8b/>

默认仅下载约 6.3 MiB 的配置、tokenizer 和 param：

```powershell
python download_model.py
```

下载全部 14 个文件、约 3.18 GiB 权重：

```powershell
python download_model.py --with-weights
python download_model.py --with-weights --verify-only
```

下载器支持 `.part` 断点续传，只接受固定镜像 URL，并按镜像目录公布的精确字节数验收。

主要权重直链：

- [Decoder 1.9 GiB](https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3.5_0.8b/qwen3.5_decoder.ncnn.bin)
- [共享 Token Embedding / LM Head 970 MiB](https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3.5_0.8b/qwen3.5_embed_token.ncnn.bin)
- [Vision Encoder 372 MiB](https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3.5_0.8b/qwen3.5_vision_encoder.ncnn.bin)
- [Vision Patch Embed 4.5 MiB](https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3.5_0.8b/qwen3.5_vision_embed_patch.ncnn.bin)
- [Vision Position Embed 6.8 MiB](https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3.5_0.8b/qwen3.5_vision_embed_pos.ncnn.bin)
- [model.json](https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3.5_0.8b/model.json)
- [vocab.txt](https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3.5_0.8b/vocab.txt)
- [merges.txt](https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3.5_0.8b/merges.txt)

## 数值基线

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

测试覆盖 prefill/单 token decode 等价、cache 连续性、空 cache、FP32 有限值与确定性、
`ShortConv` 的 ncnn `kernel_size` cache 和数学最小 `kernel_size - 1` cache、视觉 patch 顺序及
二维 RoPE。`ncnn_custom_layers.py` 将同一 NumPy 实现注册到 ncnn Python binding。

可选文本 CPU runner：

```powershell
python run_ncnn_baseline.py --model-dir _models/qwen3.5_0.8b --max-new-tokens 16
```

它以正确性为目的，Python GDR 不用于性能评估。

## 参考 CLI 构建和复验

准备 Tencent/ncnn master 源码目录后执行：

```powershell
python build_reference_cli.py --ncnn-source E:\path\to\ncnn
python validate_reference_cli.py --executable _build/reference_cli/Release/llm_ncnn_run.exe
```

CMake 只注册模型显式使用的 ncnn 层及 `Flatten/Packing/Padding/Softmax` 隐式依赖，固定 SSE2
CPU 路径，避免构建机 ISA 污染金标。模型目录可连接到参考工程：

```powershell
New-Item -ItemType Junction `
  ref\ncnn_llm-main\assets\qwen3.5_0.8b `
  Tools\Qwen35NcnnBaseline\_models\qwen3.5_0.8b
```

## 逐层比对

```powershell
python inspect_model.py _models/qwen3.5_0.8b `
  --output reports/qwen35_0_8b_contract.json --strict
python write_compare_manifest.py _models/qwen3.5_0.8b `
  --output reports/qwen35_0_8b_compare_manifest.json
```

`contract` 固化 869 层 decoder、1181 blobs 和 48 进/48 出 cache；`compare_manifest` 为六个网络
中的每一层列出 bottoms、tops、稳定 checkpoint 名和容差。后续 reference/Unity dump 必须同时
写逻辑 shape、`elempack` 和解包后的 FP32 值，再按 manifest 比较，不能用最终文本相似度替代
逐层数值验收。

完整 Unity 落地约束见 `UNITY_IMPLEMENTATION_PLAN.md`。

