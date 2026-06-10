@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.monai_to_ncnn_venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "REQ_FILE=%SCRIPT_DIR%MonaiToNCNN.requirements.txt"
set "SCRIPT_FILE=%SCRIPT_DIR%Vista3D\Vista3DBaseline.py"
set "TIMEOUT_HELPER=%SCRIPT_DIR%Vista3D\Invoke-WithTimeout.ps1"

if not exist "%PYTHON_EXE%" (
    echo [RunVista3DBaseline] Creating Python 3.10 virtual environment...
    py -3.10 -m venv "%VENV_DIR%"
    if errorlevel 1 exit /b 1
)

echo [RunVista3DBaseline] Ensuring tool dependencies...
"%PYTHON_EXE%" -c "import monai, nibabel, torch, requests"
if errorlevel 1 "%PYTHON_EXE%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 exit /b 1

if "%AIIMAGE_VISTA_LABEL_PROMPT%"=="" set "AIIMAGE_VISTA_LABEL_PROMPT=115"
if "%AIIMAGE_VISTA_LABEL_NAME%"=="" set "AIIMAGE_VISTA_LABEL_NAME=heart"
if "%AIIMAGE_VISTA_CASE_NAME%"=="" set "AIIMAGE_VISTA_CASE_NAME=ct_philips_heart"
if "%AIIMAGE_VISTA_INPUT_PATH%"=="" set "AIIMAGE_VISTA_INPUT_PATH=E:\Projects\CTData\sliceexampledata2\CT_Philips\CT_Philips.nii.gz"
if "%AIIMAGE_VISTA_BASELINE_OUTPUT_DIR%"=="" set "AIIMAGE_VISTA_BASELINE_OUTPUT_DIR=%SCRIPT_DIR%manual_test\vista3d_ct_philips_heart_baseline"
if "%AIIMAGE_VISTA_EXPORT_MANIFEST%"=="" set "AIIMAGE_VISTA_EXPORT_MANIFEST=%SCRIPT_DIR%outputs\vista3d_ct_philips_heart\manifest.json"
if "%AIIMAGE_BATCH_TIMEOUT_MINUTES%"=="" set "AIIMAGE_BATCH_TIMEOUT_MINUTES=10"
set /a AIIMAGE_BATCH_TIMEOUT_SECONDS=%AIIMAGE_BATCH_TIMEOUT_MINUTES%*60

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "& { & '%TIMEOUT_HELPER%' -FilePath '%PYTHON_EXE%' -ArgumentList @('%SCRIPT_FILE%','--input','%AIIMAGE_VISTA_INPUT_PATH%','--label-prompt','%AIIMAGE_VISTA_LABEL_PROMPT%','--label-name','%AIIMAGE_VISTA_LABEL_NAME%','--case-name','%AIIMAGE_VISTA_CASE_NAME%','--output-dir','%AIIMAGE_VISTA_BASELINE_OUTPUT_DIR%','--ncnn-manifest','%AIIMAGE_VISTA_EXPORT_MANIFEST%') -TimeoutSeconds %AIIMAGE_BATCH_TIMEOUT_SECONDS% -WorkingDirectory '%SCRIPT_DIR%' }"

exit /b %errorlevel%
