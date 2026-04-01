@echo off
REM Install Palm Counter — CUDA (Windows)
REM Requires:
REM   - Miniconda/Anaconda
REM   - NVIDIA CUDA Toolkit 11.8 or 12.x
REM   - cuDNN matching your CUDA version

echo Creating conda environment "palmcounting-cuda" (CUDA) ...
call conda env create -f environment-cuda.yml --force
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: conda env create failed. Ensure CUDA Toolkit is installed.
    pause & exit /b 1
)
echo.
echo Done! Activate with:   conda activate palmcounting-cuda
echo Start server with:     uvicorn app.main:app --reload
pause
