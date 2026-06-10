@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.monai_to_ncnn_venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "REQ_FILE=%SCRIPT_DIR%MonaiToNCNN.requirements.txt"
set "SCRIPT_FILE=%SCRIPT_DIR%Vista3D\Vista3DFixedPromptExport.py"
set "TIMEOUT_HELPER=%SCRIPT_DIR%Vista3D\Invoke-WithTimeout.ps1"

if not exist "%PYTHON_EXE%" (
    echo [RunVista3DExport] Creating Python 3.10 virtual environment...
    py -3.10 -m venv "%VENV_DIR%"
    if errorlevel 1 exit /b 1
)

echo [RunVista3DExport] Ensuring tool dependencies...
"%PYTHON_EXE%" -c "import monai, onnx, onnxsim, nibabel, pnnx, requests"
if errorlevel 1 "%PYTHON_EXE%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 exit /b 1

if "%AIIMAGE_VISTA_LABEL_PROMPT%"=="" set "AIIMAGE_VISTA_LABEL_PROMPT=115"
if "%AIIMAGE_VISTA_LABEL_NAME%"=="" set "AIIMAGE_VISTA_LABEL_NAME=heart"
if "%AIIMAGE_VISTA_CASE_TAG%"=="" set "AIIMAGE_VISTA_CASE_TAG=ct_philips_heart"
if "%AIIMAGE_VISTA_INPUT_SHAPE%"=="" set "AIIMAGE_VISTA_INPUT_SHAPE=1,1,128,128,128"
if "%AIIMAGE_VISTA_EXPORT_OUTPUT_DIR%"=="" set "AIIMAGE_VISTA_EXPORT_OUTPUT_DIR=%SCRIPT_DIR%outputs\vista3d_ct_philips_heart"
if "%AIIMAGE_BATCH_TIMEOUT_MINUTES%"=="" set "AIIMAGE_BATCH_TIMEOUT_MINUTES=10"
set /a AIIMAGE_BATCH_TIMEOUT_SECONDS=%AIIMAGE_BATCH_TIMEOUT_MINUTES%*60

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "& { & '%TIMEOUT_HELPER%' -FilePath '%PYTHON_EXE%' -ArgumentList @('%SCRIPT_FILE%','--label-prompt','%AIIMAGE_VISTA_LABEL_PROMPT%','--label-name','%AIIMAGE_VISTA_LABEL_NAME%','--case-tag','%AIIMAGE_VISTA_CASE_TAG%','--input-shape','%AIIMAGE_VISTA_INPUT_SHAPE%','--output-dir','%AIIMAGE_VISTA_EXPORT_OUTPUT_DIR%') -TimeoutSeconds %AIIMAGE_BATCH_TIMEOUT_SECONDS% -WorkingDirectory '%SCRIPT_DIR%' }"

exit /b %errorlevel%
