@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.sd15_inpaint_ncnn_venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "REQ_FILE=%SCRIPT_DIR%sd15inpainting2ncnnExporter.requirements.txt"
set "SCRIPT_FILE=%SCRIPT_DIR%export_sd15_inpaint_to_ncnn.py"

if not exist "%PYTHON_EXE%" (
    echo [sd15inpainting2ncnnExporter] Creating Python 3.10 virtual environment...
    py -3.10 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [sd15inpainting2ncnnExporter] Failed to create Python 3.10 virtual environment.
        exit /b 1
    )
)

echo [sd15inpainting2ncnnExporter] Ensuring tool dependencies...
"%PYTHON_EXE%" -c "import torch, diffusers, transformers, accelerate, safetensors, onnx, onnxruntime, onnxsim, optimum, pnnx"
if errorlevel 1 "%PYTHON_EXE%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 exit /b 1

echo [sd15inpainting2ncnnExporter] Running exporter...
"%PYTHON_EXE%" "%SCRIPT_FILE%" %*
exit /b %errorlevel%
