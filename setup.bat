@echo off
REM Setup script for solar-iot-optimization (Windows)
REM Detects NVIDIA GPU and installs the correct PyTorch version.

echo === Solar IoT Optimization — Environment Setup ===

REM Create venv if it doesn't exist
if not exist ".venv" (
    echo Creating Python 3.11 virtual environment...
    python -m venv .venv
)

REM Activate
call .venv\Scripts\activate.bat
python --version

REM Install main dependencies first
echo.
echo Installing main dependencies...
pip install -r requirements.txt

REM Install foundation model packages
echo.
echo Installing Chronos-2...
pip install "chronos-forecasting[training]>=2.0"

echo.
echo Installing TTM-R2 (IBM Granite)...
pip install granite-tsfm

echo.
echo Installing Moirai 2.0 (uni2ts) from GitHub...
pip install "uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git"

echo.
echo Installing TimesFM 2.5 from GitHub...
pip install git+https://github.com/google-research/timesfm.git

REM Install PyTorch LAST to prevent other packages from overwriting with CPU-only version
echo.
echo Detecting hardware...
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo NVIDIA GPU detected — installing PyTorch with CUDA...
    echo Trying stable release first, falling back to nightly for newer GPUs (RTX 50 series)...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall || pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
) else (
    echo No NVIDIA GPU detected — installing CPU-only PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --force-reinstall
)

REM Verify
echo.
echo === Verification ===
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo.
echo === Setup complete! ===
echo Run the pipeline with: python src/run_all.py
echo Or open the notebook: jupyter notebook notebooks/pipeline.ipynb
pause
