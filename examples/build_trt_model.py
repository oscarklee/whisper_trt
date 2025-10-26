#!/usr/bin/env python3
"""
Build TensorRT model with memory optimizations for Jetson devices
Implements sequential building strategy to minimize peak memory usage
"""

import gc
import torch
import sys
import os
import psutil

def get_gpu_memory_info():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        return mem_allocated, mem_reserved
    return 0, 0

def aggressive_cleanup():
    """Aggressively free memory"""
    gc.collect()
    gc.collect()  # Call twice for better cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def print_memory_status(stage=""):
    """Print current memory status"""
    mem_alloc, mem_reserved = get_gpu_memory_info()
    cpu_percent = psutil.virtual_memory().percent
    print(f"[{stage}] GPU Memory: {mem_alloc:.1f}MB allocated, {mem_reserved:.1f}MB reserved | CPU RAM: {cpu_percent:.1f}% used")

if len(sys.argv) < 2:
    print("Usage: python build_trt_model.py <model_name> [language] [--low-memory]")
    print("Example: python build_trt_model.py tiny es")
    print("         python build_trt_model.py base en --low-memory")
    print("         python build_trt_model.py tiny.en")
    print("\nAvailable models:")
    print("  - tiny.en, base.en, small.en (English only)")
    print("  - tiny, base, small (Multilingual)")
    print("\nOptions:")
    print("  --low-memory: Reduce TRT workspace size for devices with limited memory")
    sys.exit(1)

model_name = sys.argv[1]
language = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else "es"
low_memory_mode = '--low-memory' in sys.argv

print(f"╔══════════════════════════════════════════════════════════╗")
print(f"║  TensorRT Model Builder for Jetson                       ║")
print(f"╚══════════════════════════════════════════════════════════╝")
print(f"\nModel: {model_name}")
if not model_name.endswith('.en'):
    print(f"Language: {language}")
if low_memory_mode:
    print(f"Mode: LOW MEMORY (reduced workspace size)")
else:
    print(f"Mode: NORMAL (use --low-memory for constrained devices)")
print(f"\nEstimated build time: 10-30 minutes")
print(f"\n⚠️  IMPORTANT:")
print(f"  - Close other applications to free GPU memory")
print(f"  - Monitor with: sudo tegrastats")
print(f"  - Use --low-memory flag if you encounter OOM errors")
print(f"\n{'='*60}\n")

# Initial cleanup
print_memory_status("INITIAL")
aggressive_cleanup()
print_memory_status("AFTER CLEANUP")

# Check if we have enough memory before starting
mem_alloc, mem_reserved = get_gpu_memory_info()
gpu_free_mb = 0
if torch.cuda.is_available():
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    gpu_free_mb = gpu_total - mem_reserved

# Memory requirements (approximate, in MB)
memory_requirements = {
    'tiny': {'encoder': 500, 'decoder': 800, 'min_free': 1500},
    'tiny.en': {'encoder': 500, 'decoder': 800, 'min_free': 1500},
    'base': {'encoder': 800, 'decoder': 1500, 'min_free': 2500},
    'base.en': {'encoder': 800, 'decoder': 1500, 'min_free': 2500},
    'small': {'encoder': 1200, 'decoder': 2500, 'min_free': 4000},
    'small.en': {'encoder': 1200, 'decoder': 2500, 'min_free': 4000},
}

# Get base model name without language suffix
base_model = model_name.split('.')[0] if '.' in model_name else model_name
if model_name.endswith('.en'):
    base_model = model_name

if base_model in memory_requirements:
    required = memory_requirements[base_model]['min_free']
    if gpu_free_mb < required:
        print(f"\n❌ INSUFFICIENT MEMORY")
        print(f"{'='*60}")
        print(f"Model '{model_name}' requires: ~{required:.0f} MB free GPU memory")
        print(f"You currently have:          ~{gpu_free_mb:.0f} MB free GPU memory")
        print(f"\n💡 RECOMMENDATIONS:")
        
        if gpu_free_mb >= memory_requirements['tiny']['min_free']:
            print(f"  ✓ Try 'tiny' or 'tiny.en' model instead")
            print(f"    python build_trt_model.py tiny {language} --low-memory")
        elif gpu_free_mb >= memory_requirements['base']['min_free']:
            print(f"  ✓ Try 'base' or 'base.en' model instead")
            print(f"    python build_trt_model.py base {language} --low-memory")
        else:
            print(f"  1. Reboot device: sudo reboot")
            print(f"  2. Close all applications")
            print(f"  3. Use emergency build script:")
            print(f"     python build_trt_model_emergency.py tiny {language}")
        
        print(f"\n  Or free more memory:")
        print(f"  - Close desktop: sudo systemctl stop gdm3")
        print(f"  - Check usage: nvidia-smi")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        print(f"✓ Memory check passed: {gpu_free_mb:.0f}MB available, {required:.0f}MB required\n")

# Import after cleanup to avoid loading unnecessary modules
from whisper_trt.model import WhisperTRTBuilder
from whisper_trt.cache import get_cache_dir, make_cache_dir
from whisper import load_model
import tensorrt
from dataclasses import asdict
from whisper_trt.__version__ import __version__

print(f"\n{'='*60}")
print("STAGE 1: Loading Whisper model configuration")
print(f"{'='*60}\n")

# Configure builder
WhisperTRTBuilder.model = model_name
WhisperTRTBuilder.fp16_mode = True
WhisperTRTBuilder.verbose = True

# Adjust workspace size based on mode AND available memory
if low_memory_mode:
    # Reduce workspace from 1GB to 512MB for memory-constrained devices
    WhisperTRTBuilder.max_workspace_size = 1 << 29  # 512 MB
    print(f"⚙️  TensorRT workspace size: 512 MB (low memory mode)")
elif gpu_free_mb < 3000:
    # Auto-reduce workspace if low on memory
    WhisperTRTBuilder.max_workspace_size = 1 << 29  # 512 MB
    print(f"⚙️  TensorRT workspace size: 512 MB (auto-adjusted for available memory)")
else:
    WhisperTRTBuilder.max_workspace_size = 1 << 30  # 1 GB
    print(f"⚙️  TensorRT workspace size: 1024 MB (normal mode)")

# Get model dimensions
model_tmp = load_model(model_name)
dims = asdict(model_tmp.dims)
del model_tmp
aggressive_cleanup()
print_memory_status("CONFIG LOADED")

# Build encoder first
print(f"\n{'='*60}")
print("STAGE 2: Building Audio Encoder Engine")
print(f"{'='*60}\n")
print("This is usually the faster part (~5-10 min)...")
print_memory_status("BEFORE ENCODER BUILD")

audio_encoder_engine = WhisperTRTBuilder.build_audio_encoder_engine()
audio_encoder_state = audio_encoder_engine.state_dict()
audio_encoder_extra = WhisperTRTBuilder.get_audio_encoder_extra_state()

# Free encoder engine immediately after getting state dict
del audio_encoder_engine
aggressive_cleanup()
print_memory_status("ENCODER BUILT")
print("✓ Audio encoder built successfully\n")

# Build decoder with maximum memory available
print(f"{'='*60}")
print("STAGE 3: Building Text Decoder Engine")
print(f"{'='*60}\n")
print("This is the memory-intensive part (~10-20 min)...")
print("💡 If this fails, retry with --low-memory flag")
print_memory_status("BEFORE DECODER BUILD")

try:
    text_decoder_engine = WhisperTRTBuilder.build_text_decoder_engine()
    text_decoder_state = text_decoder_engine.state_dict()
    text_decoder_extra = WhisperTRTBuilder.get_text_decoder_extra_state()
    
    # Free decoder engine
    del text_decoder_engine
    aggressive_cleanup()
    print_memory_status("DECODER BUILT")
    print("✓ Text decoder built successfully\n")
    
except Exception as e:
    print(f"\n❌ ERROR building text decoder:")
    print(f"   {str(e)}\n")
    print("TROUBLESHOOTING:")
    print("  1. Run with --low-memory flag")
    print("  2. Close all other applications")
    print("  3. Reboot the device to clear memory")
    print("  4. Try a smaller model (tiny instead of base/small)")
    print_memory_status("AFTER ERROR")
    sys.exit(1)

# Save checkpoint
print(f"{'='*60}")
print("STAGE 4: Saving Model")
print(f"{'='*60}\n")

cache_dir = get_cache_dir()
make_cache_dir()
output_path = os.path.join(cache_dir, f"{model_name}_{language}.pth")

checkpoint = {
    "whisper_trt_version": __version__,
    "dims": dims,
    "text_decoder_engine": text_decoder_state,
    "text_decoder_extra_state": text_decoder_extra,
    "audio_encoder_engine": audio_encoder_state,
    "audio_encoder_extra_state": audio_encoder_extra
}

torch.save(checkpoint, output_path)
print(f"✓ Model saved to: {output_path}\n")

# Final cleanup
aggressive_cleanup()
print_memory_status("FINAL")

# Success message
print(f"\n{'='*60}")
print("✅ BUILD COMPLETED SUCCESSFULLY!")
print(f"{'='*60}\n")
print(f"Model: {model_name}")
print(f"Location: {output_path}")
print(f"\nYou can now use it with:")
print(f"  python live_transcription_usb.py {model_name} --language {language} --device 26")
print(f"  python transcribe.py audio.wav --model {model_name} --language {language}")
print()
