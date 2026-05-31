# CodeFormer ncnn + YOLOv7 人脸检出与回贴对齐排查总结

更新时间：2026-05-31

## 1. 目标

这份文档用于交接当前 `CodeFormer ncnn + yolov7` 多人脸链路的排查结果，重点覆盖：

- 目前已经对官方 `ref/CodeFormer-ncnn-main` 对齐了什么
- 这次对话里具体修了什么
- 现在还缺什么，下一步应该从哪里继续
- 如何用命令行静默调试、对比 Unity 复刻与官方 `ncnn + yolov7`
- 调试时用哪个 Python、如何安装和调用 `python-ncnn`

本文只聚焦：

- `ncnn + yolov7` 人脸检出
- 人脸对齐输入
- 人脸回贴

不把 `CodeFormer` 模型内部人脸还原质量本身当作本轮主问题；那个需要在 detector/alignment/paste 与官方一致后再单独追。

---

## 2. 本轮开始前的工程状态

在本轮对话开始前，工程里已经存在这些基础设施：

- Unity detector：`Assets/Scripts/NcnnFaceRegionGenerator.cs`
- Unity whole-image runner：`Assets/Scripts/CodeFormerNcnnReproRunner2.cs`
- Unity batch debug 入口：`Assets/Editor/NcnnDebugRunner.cs`
- 官方参考：
  - detector / align：`ref/CodeFormer-ncnn-main/src/face.cpp`
  - paste：`ref/CodeFormer-ncnn-main/src/pipeline.cpp`
  - whole-image入口：`ref/CodeFormer-ncnn-main/src/demo.cpp`

当时的总体判断是：

- 单人脸已经和官方很接近
- 多人脸还有明显差异
- 曾经怀疑点包括：
  - detector 结果
  - landmark 语义
  - 对齐矩阵
  - CodeFormer runner
  - 回贴/融合

---

## 3. 官方基线

### 3.1 官方 detector 参数

官方 `ref/CodeFormer-ncnn-main/src/face.cpp` 中：

- `prob_threshold = 0.5`
- `nms_threshold = 0.65`

这一点已经确认，不能把 Unity 默认改成 `0.35` 作为最终方案。`0.35` 只适合诊断。

### 3.2 官方多人脸 `02.png`

本轮用 `python-ncnn` 直接加载同一份 `yolov7-lite-e.param/bin` 跑 `ref/CodeFormer-ncnn-main/data/02.png`，结果表明：

- 官方风格 detector 在 `prob=0.5` 下不是 3 张脸，而是 4 张脸
- 结果大致是：
  - `proposal_count = 12`
  - `picked_count = 4`

四张脸的分数大致在：

- `0.866305`
- `0.857025`
- `0.850059`
- `0.843160`

因此，`02.png` 这张图里“官方只处理 3 张脸”的假设是错误的。

---

## 4. 这次对话里排查出的关键事实

### 4.1 修复前 Unity detector 的表现

修复前，Unity detector 在 `02.png`、`prob=0.5` 下表现为：

- `proposal_count = 8`
- `picked_count = 3`

而且左侧大脸没有进入 `picked`，其弱 proposal 大约只有：

- `score = 0.362831`

这也是为什么把阈值临时降到 `0.35` 时会变成 4 张脸。

### 4.2 不是简单阈值问题，根因是 detector 输入上下翻转

本轮最关键的结论：

- 修复前 Unity detector 的结果，和“把原图先垂直翻转，再喂给官方 ncnn detector”高度一致

具体现象：

- 修复前 Unity：
  - `picked_count = 3`
  - 分数大约 `0.653 / 0.636 / 0.615`
- 用 `python-ncnn` 跑“垂直翻转后的 02.png”：
  - `picked_count = 3`
  - 分数也落在同一量级

因此，问题不是：

- `prob=0.5` 过高
- `nms=0.65` 错了
- detector 完全没检出左脸

真正问题是：

- `NcnnFaceRegionGenerator` 喂给 detector 的 letterbox 图像方向与官方不一致
- 导致 detector 实际在看“上下翻转版输入”

### 4.3 修复后 detector 已经和官方对上

修复后，在 Unity 里重新跑 `02.png`，使用官方默认：

- `prob_threshold = 0.5`
- `nms_threshold = 0.65`

得到：

- `proposal_count = 12`
- `picked_count = 4`

四张脸分数大致变为：

- `0.867393`
- `0.857614`
- `0.850531`
- `0.843337`

这已经和 `python-ncnn` 的官方风格结果基本一致。

结论：

- `ncnn + yolov7` 人脸检出这一段，目前已经从“修阈值才有 4 张脸”推进到“在官方 `0.5` 下就有 4 张脸”

---

## 5. 这次对话里实际改了什么

### 5.1 `NcnnFaceRegionGenerator.cs`

文件：

- `Assets/Scripts/NcnnFaceRegionGenerator.cs`

本轮核心修复在这里。

#### 5.1.1 修复 detector letterbox 输入方向

位置大致在：

- `BuildLetterbox(...)`
- 约 `470` 行附近

做法：

- 构建 letterbox 时把写入目标纹理的行顺序上下翻过来
- 目的不是“修显示”，而是让送进 detector 的有效方向和官方 `ncnn + OpenCV` 一致

修复前症状：

- detector 实际等价于看到了“垂直翻转版输入”

#### 5.1.2 修复 detector 解码后的 y 坐标系

位置大致在：

- `DecodeYoloV7LiteE(...)`
- 约 `620` 行附近

做法：

- 先按官方 top-origin 方式计算 `topY0 / topY1 / topLy`
- 再在输出到当前 Unity 下游前，把结果翻回工程当前使用的 bottom-origin 约定

这样做的原因：

- detector 内部尽量贴官方
- 又不强行重写当前下游所有依赖 bottom-origin 的逻辑

#### 5.1.3 调试开关与 debug 输出增强

还补了两类辅助能力：

- `preferTexturePathForFaceDetector`
  - 用于切换 detector 走 texture path / buffer path
- `AppendProposalSummary(...)`
  - 把 `top_proposal[...]`、`picked[...]` 的 `rect / score / landmarks` 打进 `ncnn_face_debug.txt`

这些改动用于排查，不是最终业务逻辑核心。

### 5.2 `NcnnDebugRunner.cs`

文件：

- `Assets/Editor/NcnnDebugRunner.cs`

本轮补了命令行调试开关，方便静默比较不同 detector 配置。

新增环境变量：

- `AIIMAGE_DEBUG_INPUT`
- `AIIMAGE_FACE_BUFFER_PATH`
- `AIIMAGE_FACE_PROB_THRESHOLD`
- `AIIMAGE_FACE_NMS_THRESHOLD`

用途：

- 指定调试输入图
- 切换 detector texture/buffer path
- 临时覆盖 detector 阈值做诊断

### 5.3 `CodeFormerNcnnReproRunner2.cs`

文件：

- `Assets/Scripts/CodeFormerNcnnReproRunner2.cs`

本轮也对 runner 做过一些辅助修补：

- 增加 bottom-origin 五点模板
- 增加鲁棒相似变换
- 增加坏 landmark 时的 box-based fallback

这些改动的意义是：

- 让多人脸排查不至于因为明显错误 crop 完全卡死
- 更容易判断问题是在 detector、alignment、还是 generator

但要强调：

- 这部分目前**还不是**官方完全一致方案
- 只是帮助我们继续逼近官方

---

## 6. 这轮已经确认“不是问题”的点

### 6.1 不是 detector texture/buffer path 分叉导致的左脸消失

通过 `AIIMAGE_FACE_BUFFER_PATH` 做过对照后，landmark 与 proposal 没发生本质变化。

结论：

- 当前“左脸丢失”的主因不是 texture/buffer path 分叉
- 主因是 detector 输入方向错了

### 6.2 不是单纯 letterbox 插值误差导致

本轮还对比过：

- Unity `BuildLetterbox`
- OpenCV / 官方风格 letterbox

在修正方向问题前后，纯插值数值差的均值很小，远不足以单独解释：

- 左脸从官方 `0.85+` 掉到 Unity `0.36`

所以主因不是“插值公式微小差异”，而是方向。

---

## 7. 现在仍然还缺什么

虽然 detector 已经基本对上官方，但**多人脸整条链路还没有完全对齐官方**。

### 7.1 回贴仍未完全按官方 `pipeline.cpp` 复刻

官方回贴逻辑在：

- `ref/CodeFormer-ncnn-main/src/pipeline.cpp`

官方流程是：

1. 用 `trans_matrix_inv`
2. `tx += 1`, `ty += 1`
3. `warpAffine(restored_face -> bg_upsample)`
4. 生成 `512x512` 的全白 mask
5. 同样 `warpAffine(mask -> bg_upsample)`
6. 先做 `4x4 ellipse erode`
7. 再根据 `total_face_area` 计算：
   - `w_edge = sqrt(area) / 20`
   - `erosion_radius = w_edge * 2`
8. 做中心区域腐蚀
9. `GaussianBlur`
10. 按软 mask alpha blend 回原图

当前 Unity 的 `PasteAlignedFaceInPlace(...)` 仍然是自定义逻辑，不是这条官方路径。

这意味着：

- detector 已经对上
- 但 paste 还没完全对上

### 7.2 `Face::AlignFace` / 仿射矩阵仍未完全用官方语义收敛

官方对齐逻辑在：

- `ref/CodeFormer-ncnn-main/src/face.cpp`

核心是：

- `estimateAffinePartial2D(object.pts, face_template, cv::LMEDS)`
- `warpAffine(..., Size(512, 512), border=(135,133,132))`
- `invertAffineTransform(...)`
- `affine_matrix_inv *= 2`

当前 Unity runner 虽然已经做了很多辅助修复，但还没有完全落到这条官方语义上。

### 7.3 CodeFormer 还原质量本身仍未完全等于官方

在 detector 修正后：

- 左脸已经进入处理
- 4 张脸都能进后续链路

但 `17_full_output.png` 仍然和官方 `02_pro.png` 有明显差异，尤其是中间几张脸。

这部分优先怀疑：

- align / affine matrix 语义
- paste / blend
- `CodeFormerNcnnReproRunner2` 内部复刻路径

而不是 detector 本身。

---

## 8. 命令行如何调试

### 8.1 Python 版本

本机本轮实际使用的是：

- `C:\Python314\python.exe`
- 版本：`3.14.2`

PowerShell 下直接敲 `python` 即可。

### 8.2 临时安装 `python-ncnn`

本轮没有改项目依赖，而是把 `python-ncnn` 安装到了工作区临时目录：

```powershell
python -m pip install --target E:\Projects\AIImage\.dbg\pydeps ncnn==1.0.20260526
```

后续 ad-hoc Python 脚本需要先：

```python
import sys
sys.path.insert(0, r'E:\Projects\AIImage\.dbg\pydeps')
import ncnn
```

### 8.3 跑 Unity detector debug

```powershell
$env:AIIMAGE_DEBUG_INPUT='E:\Projects\AIImage\ref\CodeFormer-ncnn-main\data\02.png'
& 'C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe' `
  -batchmode `
  -projectPath 'E:\Projects\AIImage' `
  -executeMethod NcnnDebugRunner.RunFaceDebugBatch `
  -logFile 'E:\Projects\AIImage\Logs\face-02.log'
```

输出关注：

- `Logs\face-02.log`
- `%LOCALAPPDATA%\Temp\YanQi\AIImage\AIImage_NcnnFaceRegion_*`

其中最重要的是：

- `ncnn_face_debug.txt`
- `ncnn_face_landmarks.png`

### 8.4 跑 Unity CodeFormer 全流程 debug

```powershell
$env:AIIMAGE_DEBUG_INPUT='E:\Projects\AIImage\ref\CodeFormer-ncnn-main\data\02.png'
& 'C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe' `
  -batchmode `
  -projectPath 'E:\Projects\AIImage' `
  -executeMethod NcnnDebugRunner.RunCodeFormerDebugBatch `
  -logFile 'E:\Projects\AIImage\Logs\codeformer-02.log'
```

输出关注：

- `Logs\codeformer-02.log`
- `%LOCALAPPDATA%\Temp\YanQi\AIImage\AIImage_CodeFormerRepro2_*`

重点文件：

- `00_face512.png`
- `16_out_rgb.png`
- `17_full_output.png`

### 8.5 临时覆盖 detector 阈值

只用于诊断，不要当最终默认：

```powershell
$env:AIIMAGE_FACE_PROB_THRESHOLD='0.35'
```

### 8.6 临时切 detector 为 buffer path

只用于诊断：

```powershell
$env:AIIMAGE_FACE_BUFFER_PATH='1'
```

说明：

- `1 / true / buffer`：强制 detector 走 buffer path
- `0 / false / texture`：走 texture path

### 8.7 用 `python-ncnn` 跑官方风格 detector 对照

本轮使用的是 PowerShell here-string 直接喂给 Python：

```powershell
@'
import sys
sys.path.insert(0, r'E:\Projects\AIImage\.dbg\pydeps')
import ncnn
# 这里写 ad-hoc 对照脚本
'@ | python -
```

建议对照内容：

- `stride_8 / stride_16 / stride_32` 首批 raw 值
- `proposal_count`
- `picked_count`
- `picked[i].score / rect / pts`

---

## 9. 本轮最重要的结论

### 已经对上的

- 官方 detector 参数：`prob=0.5`, `nms=0.65`
- `02.png` 官方 detector 实际是 4 张脸
- Unity detector 现在也能在官方 `0.5` 下得到 4 张脸
- Unity detector 的 proposal 数和分数区间已经基本对上官方 `ncnn + yolov7`

### 还没完全对上的

- `Face::AlignFace` 到 Unity runner 的仿射矩阵语义
- `pipeline.cpp` 的官方回贴 / 软 mask / blur 融合
- CodeFormer 多人脸最终质量与官方 `02_pro.png` 的一致性

### 不要再误判的点

- 不要再把 `0.35` 当最终 detector 阈值
- 不要再认为 `02.png` 官方只处理 3 张脸
- 当前多人脸剩余差异已经不是“左脸没检出”，而是“后续 align / paste / CodeFormer 路径还没完全官方一致”

---

## 10. 下一步建议

最值得继续做的顺序：

1. 把 `CodeFormerNcnnReproRunner2` 的对齐矩阵收回到官方 `Face::AlignFace` 语义
2. 按官方 `pipeline.cpp` 重做 paste-back
3. 再逐脸比较：
   - `00_face512.png`
   - `16_out_rgb.png`
   - `17_full_output.png`
4. 最后才继续追 `CodeFormer` 模型内部人脸还原质量差异

如果只做一件事，优先做第 2 步：**把回贴改成官方 `warpAffine + erode + blur + alpha blend` 路径**。

