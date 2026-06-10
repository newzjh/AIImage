@echo off
setlocal
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%.monai_to_ncnn_venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    echo Python venv not found: %PYTHON_EXE%
    exit /b 1
)
"%PYTHON_EXE%" "%SCRIPT_DIR%MonaiNcnnBaseline.py" %*
endlocal
