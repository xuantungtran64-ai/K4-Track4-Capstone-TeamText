@echo off
echo =====================================================================
echo  KHOI DONG VINMEC TELE-REHABILITATION DEMO - OPTION 2A (PLAN B)
echo =====================================================================
set "PYTHON_EXE=D:\annaconda\envs\handsign\python.exe"

set "RAW_ARG=%*"
:: Tu dong loai bo prefix path\to\ neu nguoi dung go theo vi du
set "CLEAN_ARG=%RAW_ARG:path\to\=%"
set "CLEAN_ARG=%CLEAN_ARG:path/to/=%"

if exist "%PYTHON_EXE%" (
    echo [MOI TRUONG] Su dung Python Conda Handsign - Python 3.10
    "%PYTHON_EXE%" vinmec_rehab_demo.py %CLEAN_ARG%
) else (
    echo [MOI TRUONG] Su dung Python mac dinh
    python vinmec_rehab_demo.py %CLEAN_ARG%
)

