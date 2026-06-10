---
name: aiimage-unity-batch
description: Run and troubleshoot AIImage MONAI or VISTA Unity workflows that should use the repository's MONAI-specific batchmode conventions, external batch or exe launch patterns, MONAI debug environment variables, and timeout defaults. Use when Codex works in the AIImage repository on MONAI bundle export, MONAI Unity reproduction, whole-brain alignment, VISTA migration, or similar MONAI / VISTA tasks. Do not use this skill for unrelated AIImage runners such as CodeFormer, CLIP, GFPGAN, Matting, YOLO, Stable Diffusion, or generic repro tasks unless the user explicitly asks to reuse the MONAI batch path.
---

# AIImage Unity Batch

Use this skill only for AIImage MONAI or VISTA work that launches Unity or repo-owned external executables.

## Scope Guard

- Apply this skill to MONAI bundle workflows, MONAI Unity reproduction, whole-brain segmentation alignment, VISTA migration, and similar tasks.
- Do not use `E:\Projects\AIImage\Tools\MonaiToNCNN\RunMonaiUnityDebug.bat` for unrelated runners.
- Explicitly exclude these common non-MONAI runners unless the user asks otherwise:
  - `RunCodeFormerDebugBatch`
  - `RunClipDebugBatch`
  - `RunClipDirectoryDebugBatch`
  - `RunGfpganDebugBatch`
  - `RunMattingDebugBatch`
  - `RunYoloSegDebugBatch`
  - `RunYoloAndInpaintingDebugBatch`
  - `RunStableDiffusionDebugBatch`
- For those non-MONAI runners, use their own `NcnnDebugRunner` execute method or the task-specific workflow instead of the MONAI batch file.

## Unity Defaults

- Use Unity executable `C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe` unless the user explicitly overrides it.
- Prefer silent batch runs over interactive launches when validating code or reproductions.
- Default Unity batch entry for MONAI / VISTA debug work is:
  `E:\Projects\AIImage\Tools\MonaiToNCNN\RunMonaiUnityDebug.bat`
- Preferred direct Unity pattern is:
  `-batchmode -quit -projectPath E:\Projects\AIImage -executeMethod NcnnDebugRunner.RunMonaiDebugBatch`
- Treat `10` minutes as the default Unity batch timeout for MONAI / VISTA validation unless the task clearly requires a longer run.

## External Execution Rules

- Prefer PowerShell-native absolute-path invocation for repo-owned tools.
- For `.bat` files, prefer `cmd /c "<absolute-bat-path>" ...` when argument passing is simpler that way.
- Capture exit code and summarize the important log path or failure reason after execution.
- If Unity or another helper could run long, set an explicit timeout instead of leaving the process unbounded.

## MONAI Safety Defaults

- Keep MONAI sliding-window safety knobs enabled by default.
- Prefer these env values unless the task explicitly needs another value:
  - `AIIMAGE_MONAI_CLEAR_TEMP_POOL_EACH_PATCH=1`
  - `AIIMAGE_MONAI_TEMP_POOL_CLEAR_INTERVAL=1`
  - `AIIMAGE_MONAI_YIELD_INTERVAL=1`
  - `AIIMAGE_MONAI_MANAGED_CLEANUP_INTERVAL=1`
  - `AIIMAGE_MONAI_RESOURCE_SNAPSHOT_INTERVAL=1`
  - `AIIMAGE_MONAI_ABORT_PRIVATE_MEMORY_MB=8192`
  - `AIIMAGE_BATCH_TIMEOUT_MINUTES=10`
- If the user reports freezes, OOM, driver resets, or machine reboot risk, prefer `probe_only`, reduced patch count, or lower-memory validation before full runs.

## AIImage Anchors

- Unity batch orchestrator:
  `E:\Projects\AIImage\Assets\Editor\NcnnDebugRunner.cs`
- MONAI runtime and sliding-window guardrails:
  `E:\Projects\AIImage\Assets\Scripts\MONAINcnnReproRunner.cs`
- Preferred helper batch file:
  `E:\Projects\AIImage\Tools\MonaiToNCNN\RunMonaiUnityDebug.bat`

## Whole-Brain Alignment Context

- Latest validated full compare baseline:
  `E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\wholeBrainSeg_Large_UNEST_segmentation_unity\chris_t1_run46_full_compare_attnfix2_safe\summary.txt`
- Validated guard-abort evidence:
  `E:\Projects\AIImage\Tools\MonaiToNCNN\manual_test\wholeBrainSeg_Large_UNEST_segmentation_unity\chris_t1_run47_probe_guard_abort_check\runtime_debug.log`
- When continuing whole-brain work, assume the main task is pack4-path convergence and safe validation, not first-time model bring-up.
