@echo off
echo =====================================================================
echo  CHAY BO KIEM THU TU DONG TOAN DIEN - VINMEC FALL DETECTION
echo  Tac gia: Tran Xuan Tung - MSSV: 2A202602787
echo =====================================================================
set "PYTHON_EXE=D:\annaconda\envs\handsign\python.exe"

if exist "%PYTHON_EXE%" (
    echo [MOI TRUONG] Su dung Python Conda Handsign
    "%PYTHON_EXE%" test_fall_logic.py
) else (
    echo [MOI TRUONG] Su dung Python mac dinh
    python test_fall_logic.py
)

pause

