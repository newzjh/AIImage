@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.monai_to_ncnn_venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "REQ_FILE=%SCRIPT_DIR%MonaiToNCNN.requirements.txt"
set "SCRIPT_FILE=%SCRIPT_DIR%MonaiToNCNN.py"

if not exist "%PYTHON_EXE%" (
    echo [MonaiToNCNN] Creating virtual environment with Python 3.10...
    py -3.10 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [MonaiToNCNN] Failed to create Python 3.10 virtual environment.
        exit /b 1
    )
)

echo [MonaiToNCNN] Ensuring tool dependencies...
"%PYTHON_EXE%" -c "import monai, onnx, onnxruntime, onnxscript, onnxsim, nibabel, gdown, fire, huggingface_hub, pnnx"
if errorlevel 1 "%PYTHON_EXE%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 exit /b 1

echo [MonaiToNCNN] Running converter...
"%PYTHON_EXE%" "%SCRIPT_FILE%" %*
exit /b %errorlevel%
