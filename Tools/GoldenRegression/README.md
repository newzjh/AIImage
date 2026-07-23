# Inference Golden Regression

`RunGoldenRegression.ps1` compares tensor observations against JSON golden data and writes one JSON report plus one Markdown report. The tool is independent of Unity packages and has no production runtime dependency.

Run the included fixture suite:

```powershell
powershell -ExecutionPolicy Bypass -File Tools\GoldenRegression\RunGoldenRegression.ps1
```

The reports are written to `output/golden-regression/golden-report.json` and `output/golden-regression/golden-report.md`.

For a Unity Debug/Oracle execution, provide the directory exported by `NcnnGoldenDebugOracleReadback.WriteObservationBundle`:

```powershell
powershell -ExecutionPolicy Bypass -File Tools\GoldenRegression\RunGoldenRegression.ps1 -ObservationRoot output\golden-observations
```

`NcnnGoldenDebugOracleReadback` is compiled only under `Assets/Editor`. It requires the requested blob to have been included in `NcnnRepro.Infer(..., pinnedNames)` and uses the existing texture-aware `GetExistingTextureData` path. It does not materialize an intermediate `ComputeBuffer`.

The `--inject-perturbation` option exists for the regression test gate. Its format is `case_id:node/blob:index:delta`; it makes the report fail at that exact node/blob. For example:

```powershell
powershell -ExecutionPolicy Bypass -File Tools\GoldenRegression\RunGoldenRegression.ps1 -InjectPerturbation 'layer.sigmoid.pack4:sigmoid_0/sigmoid_out:2:0.1'
```

Manifest policy:

- `single_layer` cases cover Pack4 tail channels and numerical thresholds.
- `subgraph` cases pin and compare named intermediate blobs.
- `model` cases declare CLIP, Matting, YOLO, and wholeBrain probes. The wholeBrain manifest contains no patient fixture, output, or path; a private probe must supply observations through `-ObservationRoot`.
- Tensor reports always include logical shape, storage shape, layout, dtype, absolute error, and relative error.
