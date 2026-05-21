## 目标
- 将 FaceMask 从“仅靠肤色/边缘/纹理启发式”升级为“可选的人脸检测 + 人脸区域切割 +（可选）语义分割/关键点约束”的组合方案，提高遮罩稳定性（避免手臂/肤色背景误入）并便于后续共用。

## 现状（工程内）
- FaceMask 生成：由 FaceMask.compute + FaceMaskGenerator.cs 负责，当前输入为整图 Texture2D，输出为整图 mask（RHalf Texture2D），在左中 ListView 切图时生成并缓存为“当前图/男脸/女脸”三张。
- FaceMask 用途：GPUSharpen.compute 的 NoiseSuppression / FaceStruct / FaceBlend 等 kernel 通过 _FaceMaskIn 消费。

## 推荐集成位置（阶段）
- 触发时机：MainView 左中 ListView 切换图片选中后立即生成（当前已做），男/女参考图设置后立即生成（当前已做）。
- 生成阶段拆分建议（FaceMaskGenerator 内部）：
  1) 人脸检测（CPU/GPU 皆可）：得到 1~N 个人脸框/置信度/朝向
  2) ROI 切割（CPU 或 GPU blit）：将最大/最可信的人脸框裁剪成 ROI（可加 margin）
  3) ROI 内精细遮罩（GPU compute）：在 ROI 内跑分割/启发式/关键点约束，输出 ROI mask
  4) ROI mask 回填到整图 mask：仅 ROI 区域有效，其余区域为 0
  5) 可选后处理：孔洞填充、边界羽化、时域/多尺度一致性（单图可忽略）

## 可选技术路线
### 路线 A：轻量人脸检测 + 现有启发式遮罩（最快落地）
- 检测：BlazeFace / MediaPipe FaceDetection（移动端成熟），或 Unity Barracuda + 轻量 ONNX 检测器。
- 遮罩：保留当前 FaceMask.compute 的肤色/边缘/亮度门控逻辑，但只在 ROI 内运行。
- 价值：大幅降低“手臂/背景肤色”被算进 mask 的概率，因为 mask 仅在 ROI 区域允许生成。

### 路线 B：人脸关键点（FaceMesh）约束的遮罩（更稳）
- 关键点：MediaPipe FaceMesh（468 点）或同类模型输出关键点。
- 生成：
  - 用关键点生成“人脸凸包/面部椭圆/五官排除区（眼睛/嘴巴可选）”
  - 与 ROI 内启发式 mask 相乘，或直接用关键点形状作为先验再做羽化
- 价值：遮罩的中心/半径更符合真实脸型，并且在光照/肤色变化下更稳。

### 路线 C：语义分割（Face parsing）直接得到皮肤/头发/五官等类别（质量最好）
- 模型：人脸解析（face parsing）网络，输出皮肤/头发/五官/背景等多类 mask。
- 策略：
  - FaceMask = skin（可加少量脸颊区域）-（眼睛/嘴巴/眉毛可选）
  - 需要时提供多路 mask（skinMask / hairMask / featureMask）供不同后处理阶段使用
- 集成：放到 FaceMaskGenerator 的“ROI 内精细遮罩”步骤，输出 ROI mask。

## 工程落地建议（最小侵入）
- 新增“检测/关键点/分割”的 C# 模块：建议挂在 FaceMaskGenerator 同 GameObject 上，由 FaceMaskGenerator 调用。
- 只改 FaceMaskGenerator 的输入/流程，不改 GPUSharpenRunner 的增强链路：
  - FaceMaskGenerator 仍输出整图 mask Texture2D 给 GPUSharpenRunner
  - GPUSharpenRunner 继续只消费 _FaceMaskIn
- 运行时约束：
  - 需要可取消（CancellationToken）
  - ROI + 分割结果必须能回填为整图 mask（与现有 compute 接口兼容）

## 数据与调试
- 现有调试开关复用：勾选时在切图触发 faceMask 生成，导出 raw、d1~d5、最终 mask。
- 建议追加（后续可选）：
  - 导出检测框可视化（画框到一张 PNG）
  - 导出 ROI 裁剪图与 ROI mask

