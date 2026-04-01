@echo off
REM Install Palm Counter — CPU (Windows)
REM Requires Miniconda/Anaconda: https://docs.conda.io/en/latest/miniconda.html

echo Creating conda environment "palmcounting" (CPU) ...
call conda env create -f environment.yml --force
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: conda env create failed.
    pause & exit /b 1
)
echo.
echo Done! Activate with:   conda activate palmcounting
echo Start server with:     uvicorn app.main:app --reload
pause
