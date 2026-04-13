#!/bin/bash
# Setup script for solar-iot-optimization
# Detects platform and installs the correct PyTorch version automatically.

set -e

echo "=== Solar IoT Optimization — Environment Setup ==="

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python 3.11 virtual environment..."
    python3.11 -m venv .venv
fi

# Activate
source .venv/bin/activate
echo "Python: $(python --version)"

# Install main dependencies first
echo ""
echo "Installing main dependencies..."
pip install -r requirements.txt

# Install foundation model packages
echo ""
echo "Installing Chronos-2..."
pip install "chronos-forecasting[training]>=2.0" || echo "WARNING: Chronos install failed"

echo ""
echo "Installing TTM-R2 (IBM Granite)..."
pip install granite-tsfm || echo "WARNING: TTM-R2 install failed"

echo ""
echo "Installing Moirai 2.0 (uni2ts) from GitHub..."
pip install "uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git" || echo "WARNING: Moirai install failed — it will be skipped during evaluation."

echo ""
echo "Installing TimesFM 2.5 from GitHub..."
pip install git+https://github.com/google-research/timesfm.git || echo "WARNING: TimesFM install failed — it will be skipped during evaluation."

# Install PyTorch LAST to prevent other packages from overwriting with CPU-only version
echo ""
echo "Detecting hardware..."
if command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected — installing PyTorch with CUDA..."
    echo "Trying stable release first, falling back to nightly for newer GPUs (RTX 50 series)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall || \
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
elif python -c "import platform; assert platform.system()=='Darwin'" 2>/dev/null; then
    echo "macOS detected — installing PyTorch with MPS support..."
    pip install torch torchvision --force-reinstall
else
    echo "No GPU detected — installing CPU-only PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --force-reinstall
fi

# Verify
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA: {torch.version.cuda}')
elif torch.backends.mps.is_available():
    print('GPU: Apple MPS')
else:
    print('GPU: None (CPU only)')
"

echo ""
echo "=== Setup complete! ==="
echo "Run the pipeline with: make all"
echo "Or open the notebook: jupyter notebook notebooks/pipeline.ipynb"
