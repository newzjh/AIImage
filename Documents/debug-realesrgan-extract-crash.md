[OPEN] Real-ESRGAN native: extract output failed + second-run editor crash

## Session
- sessionId: realesrgan-extract-crash
- scope: RealEsrganNcnnNativeRunner + realesrgan_unity.cpp

## Symptoms
- Realesrgan_ProcessRgba() returns "extract output failed"
- Second attempt may crash Unity Editor (suspected native resource leak / invalid Vulkan state)

## Hypotheses
- H1: Vulkan path is not actually used for extractor (wrong input type / allocator / command usage), leading to extract failure.
- H2: Model blob names differ from "data"/"output" for the selected model, causing extract failure.
- H3: Tile/prepadding crop math produces out-of-range accesses, corrupting memory and crashing next run.
- H4: VkAllocator acquire/reclaim not balanced on early returns, leaking or double-freeing, causing second-run crash.
- H5: GPU instance lifecycle (create/destroy) is mismanaged across context creation/destruction in Unity domain reload.

## Instrumentation plan
- Return detailed native error with tile index, in/out sizes, input/extract return codes.
- Report run parameters + native error back to a debug server from C# (no Console logging).

## Evidence (pre-fix)
- Unity first-run crash happens during `Realesrgan_ProcessRgba` call.
- Latest trace shows `w=2816,h=2112,runFactor=4` and `tileSize=0`, which implies a single huge tile; downloading float output for that tile can exceed 1GB and crash the editor.

## Fix (attempt)
- Treat `tileSize<=0` as auto and clamp tile to a safe range (default 256, max 512) in native.

## Run checklist
- Start debug server: python3 ... --session realesrgan-extract-crash --outdir .dbg --clean --idle 1200
- In Unity: click Real-ESRGAN(ncnn原生) once; capture whether crash happens on 2nd run

