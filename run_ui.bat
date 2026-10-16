@echo off
echo =====================================================================
echo  KHOI DONG VINMEC AI MEDICAL HUB - GIAO DIEN DIEU KHIEN (LAUNCHER UI)
echo =====================================================================
set "PYTHON_EXE=D:\annaconda\envs\handsign\python.exe"

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" vinmec_demo_app.py
) else (
    python vinmec_demo_app.py
)

