@echo off
echo =====================================================================
echo  KHOI DONG GIAO DIEN STREAMLIT - VINMEC AI MEDICAL HUB
echo =====================================================================
set "PYTHON_EXE=D:\annaconda\envs\handsign\python.exe"

if exist "%PYTHON_EXE%" (
    echo [MOI TRUONG] Su dung Python Conda Handsign
    "%PYTHON_EXE%" -m streamlit run app_streamlit.py
) else (
    echo [MOI TRUONG] Su dung Python mac dinh
    python -m streamlit run app_streamlit.py
)

