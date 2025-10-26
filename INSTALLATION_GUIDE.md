# whisper_trt Installation Guide for NVIDIA Jetson

This guide documents the complete installation process for `whisper_trt` on NVIDIA Jetson platforms running JetPack 6.0, including all challenges encountered and their solutions.

## Platform Specifications

- **Device**: NVIDIA Jetson (Tegra ARM64 architecture)
- **JetPack Version**: 6.0 (L4T R36.2/R36.3)
- **CUDA**: 12.6
- **cuDNN**: 9.3.0
- **TensorRT**: 10.3.0 (pre-compiled for Python 3.10)
- **Python**: 3.10.12 (managed via pyenv)

## Challenges Encountered

### 1. Dependency Confusion: `whisper` vs `openai-whisper`

**Problem**: The initial installation attempt used the wrong package. There are two packages with similar names:
- `whisper` - A generic package without the required `load_model` function
- `openai-whisper` - The correct package containing OpenAI's Whisper implementation

**Solution**: Installed the correct package:
```bash
pip install openai-whisper
```

### 2. Python Version Compatibility

**Problem**: TensorRT 10.3.0 was compiled specifically for Python 3.10, but the initial environment used Python 3.11.

**Solution**: Created a Python 3.10.12 virtual environment using pyenv:
```bash
pyenv install 3.10.12
pyenv virtualenv 3.10.12 whisper_trt
pyenv activate whisper_trt
```

### 3. PyTorch CUDA Support for ARM64

**Problem**: PyPI does not provide official PyTorch wheels with CUDA support for ARM64 architecture. The default `pip install torch` only installs CPU-only versions.

**First Attempt**: Downloaded PyTorch 2.2.0 from NVIDIA's China repository:
- URL: `https://developer.download.nvidia.cn/compute/redist/jp/v60/pytorch/`
- **Failed**: PyTorch 2.2.0 requires cuDNN 8.x, but JetPack 6.0 ships with cuDNN 9.3.0
- Error: `libcudnn.so.8: version 'libcudnn.so.8' not found`

**Final Solution**: Downloaded PyTorch 2.3.0 built specifically for JetPack 6.0 with CUDA 12.4 support:
- URL: `https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl`
- This version is compatible with cuDNN 9.x

### 4. NumPy Compatibility

**Problem**: NumPy 2.x caused compatibility issues with pre-compiled binaries (ONNX Runtime, TensorRT).

**Solution**: Downgraded to NumPy 1.26.4:
```bash
pip install "numpy<2.0"
```

### 5. TensorRT Python Bindings

**Problem**: TensorRT Python bindings are installed system-wide in `/usr/lib/python3.10/dist-packages/` but not accessible from the virtual environment.

**Solution**: Created symbolic links from system TensorRT to the virtual environment:
```bash
ln -s /usr/lib/python3.10/dist-packages/tensorrt* ~/.pyenv/versions/whisper_trt/lib/python3.10/site-packages/
```

### 6. ONNX Runtime with TensorRT Support

**Problem**: The standard `onnxruntime-gpu` from PyPI doesn't include TensorRT execution provider for ARM64.

**Solution**: Used a custom-compiled version with TensorRT and CUDA support:
- Location: `~/Downloads/onnxruntime_gpu-1.22.2-cp310-cp310-linux_aarch64.whl`
- This wheel was compiled with both `TensorrtExecutionProvider` and `CUDAExecutionProvider`

### 7. torch2trt CUDA_HOME Configuration

**Problem**: torch2trt couldn't locate CUDA installation during compilation.

**Solution**: Set `CUDA_HOME` environment variable before installation:
```bash
export CUDA_HOME=/usr/local/cuda
```

Alternatively, added to pyenv activate script for persistence.

### 8. Missing ONNX Graph Surgeon

**Problem**: Running the transcription example failed with:
```
ModuleNotFoundError: No module named 'onnx_graphsurgeon'
```

**Solution**: This is a required dependency for torch2trt that wasn't automatically installed:
```bash
pip install onnx-graphsurgeon
```

### 9. OpenAI Whisper Version Compatibility

**Problem**: The latest version of openai-whisper (20250625) has compatibility issues with PyTorch 2.3.0, causing errors with `scaled_dot_product_attention`.

**Solution**: Downgrade to a stable version of openai-whisper:
```bash
pip install openai-whisper==20231117
```

## Complete Installation Steps

### Step 1: Set Up Python Environment

```bash
# Create virtual environment
pyenv virtualenv system whisper_trt

# Activate environment
pyenv activate whisper_trt
```

### Step 2: Install Core Dependencies

```bash
# Install openai-whisper with a stable version (not the latest)
pip install openai-whisper==20231117

# Install psutil
pip install psutil

# Downgrade NumPy for compatibility
pip install "numpy<2.0"

# Install ONNX Graph Surgeon (required for torch2trt)
pip install onnx-graphsurgeon
```

### Step 3: Install ONNX Runtime with TensorRT Support

```bash
# Install custom-compiled ONNX Runtime wheel
pip install ~/Downloads/onnxruntime_gpu-1.22.2-cp310-cp310-linux_aarch64.whl
```

**Note**: This wheel must be compiled with TensorRT and CUDA support. Verify with:
```python
import onnxruntime
print(onnxruntime.get_available_providers())
# Should include: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Step 4: Link TensorRT Python Bindings

```bash
# Create symbolic links to system TensorRT installation
ln -s /usr/lib/python3.10/dist-packages/tensorrt* ~/.pyenv/versions/whisper_trt/lib/python3.10/site-packages/
```

**Verify TensorRT installation**:
```python
import tensorrt
print(tensorrt.__version__)  # Should output: 10.3.0
```

### Step 5: Install PyTorch with CUDA Support

Download PyTorch 2.3.0 for JetPack 6.0:
```bash
# Download the wheel (manually or via wget)
# URL: https://developer.download.nvidia.cn/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl
# Save to: ~/Downloads/torch-2.3.0-cp310-cp310-linux_aarch64.whl

# Install the wheel
pip install ~/Downloads/torch-2.3.0-cp310-cp310-linux_aarch64.whl
```

**Verify PyTorch installation**:
```python
import torch
print('PyTorch:', torch.__version__)          # 2.3.0
print('CUDA available:', torch.cuda.is_available())  # True
print('CUDA version:', torch.version.cuda)    # 12.4
```

### Step 6: Install torch2trt

```bash
# Set CUDA_HOME environment variable
export CUDA_HOME=/usr/local/cuda

# Clone and install torch2trt
cd /tmp
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
pyenv local whisper_trt
python setup.py install
```

**Optional**: Add CUDA_HOME to pyenv activate script for persistence:
```bash
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.pyenv/versions/whisper_trt/bin/activate
```

### Step 7: Install whisper_trt

```bash
# Navigate to whisper_trt directory
cd /home/oscarklee/dev/whisper_trt

# Install in development mode
python setup.py develop
```

**Verify installation**:
```python
import whisper_trt
print('whisper_trt version:', whisper_trt.__version__)  # 0.0.1
```

## Testing the Installation

### First Run - Building TensorRT Engines

**IMPORTANT**: The first time you run a transcription, it will take **5-15 minutes** to complete. This is because whisper_trt needs to:

1. Download the Whisper model weights from OpenAI (if not already cached)
2. Load the model in PyTorch
3. **Convert the audio encoder to TensorRT format** (this is the slowest step)
4. **Convert the text decoder to TensorRT format**
5. Save the optimized TensorRT engines to cache (`~/.cache/whisper_trt/`)

**What to expect on first run**:
- The process may appear "frozen" or "hung" - this is normal
- No progress output by default (use `verbose=True` in code to see TensorRT build logs)
- The Jetson device will be under heavy load (high CPU/GPU usage)
- Fans may spin up due to increased temperature

**Subsequent runs** will be much faster (typically 2-5 seconds to load the cached engines) because the TensorRT engines are already built and cached.

### Running the Example

```bash
# Navigate to examples directory
cd /home/oscarklee/dev/whisper_trt/examples

# Run transcription with TensorRT backend
# Note: First run will take 5-15 minutes!
python transcribe.py tiny.en speech.wav --backend whisper_trt

# For verbose output showing TensorRT build progress:
# Modify the script to use: load_trt_model(args.model, verbose=True)
```

Available models: `tiny.en`, `base.en`, `small.en`

Available backends:
- `whisper` - Original OpenAI Whisper (CPU/CUDA)
- `whisper_trt` - TensorRT-optimized version (fastest on Jetson)
- `faster_whisper` - Alternative optimized implementation

## Complete Dependency List

After successful installation, the environment includes:

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10.12 | Managed via pyenv |
| openai-whisper | 20231117 | Stable version (not latest) |
| torch | 2.3.0 | CUDA 12.4 support for JetPack 6.0 |
| onnxruntime-gpu | 1.22.2 | Custom build with TensorRT support |
| tensorrt | 10.3.0 | System installation (symlinked) |
| torch2trt | 0.4.0 | PyTorch to TensorRT converter |
| numpy | 1.26.4 | Downgraded for compatibility |
| onnx-graphsurgeon | latest | Required for torch2trt |
| whisper_trt | 0.0.1 | Installed in development mode |

## Key File Locations

- **Virtual Environment**: `~/.pyenv/versions/whisper_trt/`
- **whisper_trt Source**: `/home/oscarklee/dev/whisper_trt/`
- **PyTorch Wheel**: `~/Downloads/torch-2.3.0-cp310-cp310-linux_aarch64.whl`
- **ONNX Runtime Wheel**: `~/Downloads/onnxruntime_gpu-1.22.2-cp310-cp310-linux_aarch64.whl`
- **TensorRT System Path**: `/usr/lib/python3.10/dist-packages/tensorrt*`
- **CUDA Installation**: `/usr/local/cuda`
- **TensorRT Engine Cache**: `~/.cache/whisper_trt/` (created after first run)

## Troubleshooting

### Import Error: No module named 'whisper'
**Solution**: Install `openai-whisper`, not `whisper`:
```bash
pip uninstall whisper
pip install openai-whisper==20231117
```

### ModuleNotFoundError: No module named 'onnx_graphsurgeon'
**Solution**: Install the missing dependency:
```bash
pip install onnx-graphsurgeon
```

### Whisper version incompatibility with PyTorch 2.3.0
**Error**: `TypeError` related to `scaled_dot_product_attention` and `is_causal` parameter

**Solution**: Use a stable version of openai-whisper:
```bash
pip install openai-whisper==20231117
```

### First run appears frozen or hung
**This is normal!** The first transcription takes 5-15 minutes while TensorRT engines are being built. The process is CPU/GPU intensive but will complete. Subsequent runs will be much faster (2-5 seconds).

**Solution**: Be patient and let it complete. You can monitor progress by:
- Checking GPU/CPU usage with `tegrastats` or `jtop`
- Modifying the code to use `load_trt_model(model_name, verbose=True)` to see build logs

### ImportError: libcudnn.so.8 not found
**Solution**: Use PyTorch 2.3.0 (compatible with cuDNN 9.x), not PyTorch 2.2.0 (requires cuDNN 8.x).

### TensorRT not found in Python
**Solution**: Create symbolic links from system TensorRT to your virtual environment.

### torch2trt compilation fails
**Solution**: Ensure `CUDA_HOME` is set:
```bash
export CUDA_HOME=/usr/local/cuda
```

### ONNX Runtime missing TensorRT provider
**Solution**: Use the custom-compiled wheel with TensorRT support, not the standard PyPI version.

## Performance Tips

1. **First Run**: The first transcription takes 5-15 minutes as TensorRT engines are built and cached. This only happens once per model.
2. **Subsequent Runs**: After the first run, loading is very fast (2-5 seconds) as engines are loaded from cache.
3. **Model Cache**: TensorRT models are cached in `~/.cache/whisper_trt/` - don't delete this directory!
4. **Model Selection**: Start with `tiny.en` for testing, then upgrade to `base.en` or `small.en` for better accuracy.
5. **Backend Comparison**: Use `profile_backend.py` to compare performance between backends.
6. **Memory Usage**: Larger models (small.en) require more GPU memory. Monitor with `tegrastats` or `jtop`.

## Additional Resources

- **PyTorch for Jetson**: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
- **NVIDIA China PyTorch Repository**: https://developer.download.nvidia.cn/compute/redist/jp/v60/pytorch/
- **whisper_trt Repository**: https://github.com/NVIDIA-AI-IOT/whisper_trt
- **torch2trt Repository**: https://github.com/NVIDIA-AI-IOT/torch2trt

## Credits

This installation was completed after resolving multiple dependency conflicts specific to the NVIDIA Jetson platform running JetPack 6.0 with cuDNN 9.3.0. Special thanks to the NVIDIA forums community for providing access to the PyTorch 2.3.0 wheels compatible with the latest JetPack release.
