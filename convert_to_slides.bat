@echo off
REM Change to working directory
cd /d C:\Users\Edvard\Documents\GitHub_BAR\matrix-form-finding\Lectures

REM Activate conda environment
call conda activate CEE6501_jupyter

REM Check for input argument
if "%~1"=="" (
    echo Please provide the notebook filename without .ipynb extension
    exit /b 1
)

REM Run nbconvert
jupyter nbconvert --to slides "%~1.ipynb" --post serve

cd ..