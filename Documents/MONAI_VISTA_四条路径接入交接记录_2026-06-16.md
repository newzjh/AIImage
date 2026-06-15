# MONAI / VISTA 四条路径接入交接记录

日期：2026-06-16

## 当前状态

- `wholeBrainSeg_Large_UNEST_segmentation` 的脑室 Python 基线已稳定，人工复核请看 `ventricles_labelmap_refined_original.nii.gz`。
- Unity 侧 `pack4_rt` / `command_buffer_rt` 已对齐 wholeBrain 脑室基线，推理期临时 `ComputeBuffer` 已清零。
- `vista3d` 的 skull Python 基线已稳定，Unity 侧 `pack4_rt` 已对齐；`command_buffer_rt` 走同一批处理入口切换。
- 当前所有验证都按 `10` 分钟 batch 超时和既有内存守卫跑，优先保留 `pack4_rt`，不要回退到 `compute_buffer`。

## 通用入口

- Unity batch：`E:\Projects\AIImage\Tools\MonaiToNCNN\RunMonaiUnityDebug.bat`
- Unity exe：`C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe`
- Unity method：`NcnnDebugRunner.RunMonaiDebugBatch`
- 常用安全默认值：`AIIMAGE_MONAI_CLEAR_TEMP_POOL_EACH_PATCH=1`、`AIIMAGE_MONAI_TEMP_POOL_CLEAR_INTERVAL=1`、`AIIMAGE_MONAI_YIELD_INTERVAL=1`、`AIIMAGE_MONAI_MANAGED_CLEANUP_INTERVAL=1`、`AIIMAGE_MONAI_RESOURCE_SNAPSHOT_INTERVAL=1`、`AIIMAGE_MONAI_ABORT_PRIVATE_MEMORY_MB=8192`

## 接入清单

| 项目 | Python 基线 | Unity repro | 模型路径 | 结果文件 |
| --- | --- | --- | --- | --- |
| `wholeBrainSeg_Large_UNEST_segmentation` 脑室切割 | `E:\Projects\AIImage\Tools\MonaiToNCNN\MonaiNcnnBaseline.py` + `E:\Projects\AIImage\Tools\MonaiToNCNN\RefineWholeBrainVentricles.py` | `AIIMAGE_MONAI_PATCH_INPUT_MODE=pack4_rt` 或 `command_buffer_rt`，`AIIMAGE_MONAI_USE_COMMAND_BUFFER=0/1` | `E:\Projects\AIImage\Tools\MonaiToNCNN\bundle_cache\wholeBrainSeg_Large_UNEST_segmentation`，导出模型在 `E:\Projects\AIImage\Tools\MonaiToNCNN\outputs\wholeBrainSeg_Large_UNEST_segmentation\wholeBrainSeg_Large_UNEST_segmentation.param` / `.bin` / `.sim.pnnx.param` | Python 参考：`E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\wholebrain_mri_ventricles_refined\mri_custom_ventricles_mni305_refined\ventricles_labelmap_refined_original.nii.gz`；Unity 参考：`E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\wholebrain_mri_ventricles_unity_pack4_rt_mni305_align6_reviewcompare_ok\summary.txt` 和同目录下 `label_subsets\ventricles_labelmap_refined_original.nii.gz` |
| `wholeBrainSeg_Large_UNEST_segmentation` Unity pack4 / commandbuffer | 同上 | `E:\Projects\AIImage\Tools\MonaiToNCNN\RunMonaiUnityDebug.bat`，baseline manifest 指向 wholeBrain 的 `baseline_manifest.json` | 同上 | `pack4_rt` 看 `E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\wholebrain_mri_ventricles_unity_pack4_rt_mni305_align6_reviewcompare_ok\summary.txt`；`command_buffer_rt` 看 `E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\wholebrain_mri_ventricles_cmdrt_mni305_tempbufguard2\summary.txt` |
| `vista3d` skull 切割 | `E:\Projects\AIImage\Tools\MonaiToNCNN\Vista3D\Vista3DBaseline.py` | `AIIMAGE_MONAI_PATCH_INPUT_MODE=pack4_rt` 或 `command_buffer_rt`，label prompt 固定 `120`，label name `skull` | `E:\Projects\AIImage\Tools\MonaiToNCNN\bundle_cache\model-zoo-dev\models\vista3d`，导出模型在 `E:\Projects\AIImage\Tools\MonaiToNCNN\outputs\vista3d_ct_custom_skull\vista3d_fixed_ct_custom_skull.param` / `.bin` / `.pnnx.param` / `.pnnx.bin` | Python 参考：`E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\vista3d_ct_skull_python_baseline\ct_custom_skull\labelmap_restored.nii.gz`，subset 参考：`E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\vista3d_ct_skull_python_baseline\ct_custom_skull\label_subsets\skull_mask_restored.nii.gz` |
| `vista3d` skull Unity pack4 / commandbuffer | 同上 | `E:\Projects\AIImage\Tools\MonaiToNCNN\RunMonaiUnityDebug.bat`，baseline manifest 指向 `vista3d_ct_skull_python_baseline\ct_custom_skull\baseline_manifest.json`，`AIIMAGE_MONAI_USE_COMMAND_BUFFER=1` 即切 `command_buffer_rt` | 同上 | `pack4_rt` 看 `E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\vista3d_ct_skull_unity_pack4_rt_prompt_rt_finalcheck\summary.txt`；`command_buffer_rt` 代码路径已支持，但当前仓库里未看到单独归档的 `manual_test` 结果目录，接入其他工程时需要按同入口重新跑并另存输出目录 |

## 调试方式

1. 先跑 Python 基线，确认 `baseline_manifest.json`、`summary.txt`、`labelmap_restored.nii.gz` / `*_labelmap_refined_original.nii.gz` 正常。
2. Unity 先跑 `pack4_rt`，只看 `summary.txt`、`baseline_compare.json`、`resource_stats.json`、`layer_runtime_profile.tsv`。
3. 需要对照时再切 `command_buffer_rt`，不要改回 `compute_buffer`。
4. 如果有 OOM / 卡死风险，先保留 `probe_only`、`max_patches=1`、安全守卫，再放大全量。

## 推荐落点

- wholeBrain Python 初始基线目录：`E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\wholebrain_mri_ventricles_python_baseline_mni305_ov0_fix\mri_custom_ventricles_mni305_ov0_fix`
- wholeBrain Python refined 目录：`E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\wholebrain_mri_ventricles_refined\mri_custom_ventricles_mni305_refined`
- wholeBrain Unity `pack4_rt`：`E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\wholebrain_mri_ventricles_unity_pack4_rt_mni305_align6_reviewcompare_ok`
- wholeBrain Unity `command_buffer_rt`：`E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\wholebrain_mri_ventricles_cmdrt_mni305_tempbufguard2`
- vista3d Python baseline：`E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\vista3d_ct_skull_python_baseline\ct_custom_skull`
- vista3d Unity `pack4_rt`：`E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\vista3d_ct_skull_unity_pack4_rt_prompt_rt_finalcheck`

## 最小切换方式

- wholeBrain 切 `pack4_rt`：`AIIMAGE_MONAI_BASELINE_MANIFEST=<wholebrain baseline_manifest.json>`，`AIIMAGE_MONAI_PATCH_INPUT_MODE=pack4_rt`
- wholeBrain 切 `command_buffer_rt`：`AIIMAGE_MONAI_BASELINE_MANIFEST=<wholebrain baseline_manifest.json>`，`AIIMAGE_MONAI_PATCH_INPUT_MODE=command_buffer_rt`，`AIIMAGE_MONAI_USE_COMMAND_BUFFER=1`
- vista3d 切 `pack4_rt`：`AIIMAGE_MONAI_BASELINE_MANIFEST=<vista baseline_manifest.json>`，`AIIMAGE_MONAI_PATCH_INPUT_MODE=pack4_rt`
- vista3d 切 `command_buffer_rt`：`AIIMAGE_MONAI_BASELINE_MANIFEST=<vista baseline_manifest.json>`，`AIIMAGE_MONAI_PATCH_INPUT_MODE=command_buffer_rt`，`AIIMAGE_MONAI_USE_COMMAND_BUFFER=1`

## 结果约定

- wholeBrain 以 `ventricles_labelmap_refined_original.nii.gz` 作为人工复核基线。
- vista3d 以 `labelmap_restored.nii.gz` 和 `label_subsets/skull_mask_restored.nii.gz` 作为基线结果。
- Unity 结果优先对齐 Python 基线，不追求 DICOM 输出，`nii.gz` / `nrrd` 都可以。
