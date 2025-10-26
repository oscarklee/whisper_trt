#!/usr/bin/env python3
"""
Check available system memory before building TensorRT models
Provides recommendations based on available resources
"""

import torch
import psutil
import subprocess
import sys

def get_jetson_model():
    """Detect Jetson model if available"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip('\x00')
            return model
    except:
        return "Unknown (not a Jetson or unable to detect)"

def get_gpu_memory():
    """Get GPU memory info"""
    if not torch.cuda.is_available():
        return None, None, None
    
    # Get total GPU memory
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
    
    # Try to get free memory
    try:
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        free = total - reserved
        return total, free, allocated
    except:
        return total, None, None

def get_cpu_memory():
    """Get CPU RAM info in MB"""
    mem = psutil.virtual_memory()
    return mem.total / 1024**2, mem.available / 1024**2, mem.percent

def recommend_model_and_settings(gpu_free_mb, cpu_free_mb):
    """Provide recommendations based on available memory"""
    recommendations = []
    buildable_models = []
    
    # Memory requirements for each model
    requirements = {
        'tiny': 1500,
        'base': 2500,
        'small': 4000,
        'medium': 6000,
        'large': 10000,
    }
    
    # Model recommendations based on available memory
    if gpu_free_mb:
        for model, required in requirements.items():
            if gpu_free_mb >= required:
                buildable_models.append(model)
        
        if gpu_free_mb >= 4000:
            recommendations.append(f"✓ You can build: {', '.join(buildable_models)} models")
            low_mem_flag = ""
        elif gpu_free_mb >= 2500:
            recommendations.append(f"✓ You can build: {', '.join(buildable_models)}")
            recommendations.append("⚠ For small model, you need ~4 GB free")
            low_mem_flag = ""
        elif gpu_free_mb >= 1500:
            recommendations.append(f"✓ You can build: {', '.join(buildable_models)}")
            recommendations.append("⚠ Use --low-memory flag for better stability")
            recommendations.append("⚠ base/small models may fail - prefer tiny")
            low_mem_flag = "--low-memory"
        else:
            recommendations.append("❌ Insufficient GPU memory for building")
            recommendations.append("❌ Need at least 1.5 GB free")
            recommendations.append("💡 Close applications and reboot device")
            low_mem_flag = "emergency"
    else:
        recommendations.append("❌ CUDA not available")
        low_mem_flag = ""
    
    # CPU memory warning
    if cpu_free_mb < 1000:
        recommendations.append("⚠ Low CPU RAM - close other applications")
    
    return recommendations, low_mem_flag, buildable_models

def main():
    print("="*70)
    print("  Whisper TRT - Memory Check")
    print("="*70)
    
    # System info
    jetson_model = get_jetson_model()
    print(f"\n🖥️  Device: {jetson_model}")
    
    # GPU Memory
    print(f"\n{'GPU Memory:':<20}")
    gpu_total, gpu_free, gpu_allocated = get_gpu_memory()
    
    if gpu_total:
        print(f"  Total:      {gpu_total:>10.0f} MB")
        if gpu_free:
            print(f"  Free:       {gpu_free:>10.0f} MB")
            print(f"  Allocated:  {gpu_allocated:>10.0f} MB")
        
        # Color code based on available memory
        if gpu_free and gpu_free >= 2000:
            status = "✓ Good"
        elif gpu_free and gpu_free >= 1500:
            status = "⚠ Marginal"
        else:
            status = "❌ Low"
        print(f"  Status:     {status}")
    else:
        print("  ❌ CUDA not available")
    
    # CPU Memory
    print(f"\n{'CPU Memory (RAM):':<20}")
    cpu_total, cpu_free, cpu_percent = get_cpu_memory()
    print(f"  Total:      {cpu_total:>10.0f} MB")
    print(f"  Free:       {cpu_free:>10.0f} MB")
    print(f"  Used:       {cpu_percent:>9.1f} %")
    
    if cpu_percent > 80:
        print(f"  Status:     ❌ High usage")
    elif cpu_percent > 60:
        print(f"  Status:     ⚠ Moderate usage")
    else:
        print(f"  Status:     ✓ Good")
    
    # Recommendations
    if gpu_free:
        recommendations, low_mem_flag, buildable = recommend_model_and_settings(gpu_free, cpu_free)
        
        print(f"\n{'Recommendations:':<20}")
        for rec in recommendations:
            print(f"  {rec}")
        
        if buildable:
            print(f"\n{'Example build commands:':<20}")
            
            # Show command for each buildable model
            if 'tiny' in buildable:
                flag = "--low-memory" if low_mem_flag == "--low-memory" else ""
                print(f"  # Tiny model (fastest, least accurate):")
                print(f"  python build_trt_model.py tiny en {flag}".strip())
            
            if 'base' in buildable:
                flag = "--low-memory" if low_mem_flag == "--low-memory" else ""
                print(f"\n  # Base model (balanced):")
                print(f"  python build_trt_model.py base es {flag}".strip())
            
            if 'small' in buildable:
                print(f"\n  # Small model (better accuracy, slower):")
                print(f"  python build_trt_model.py small es")
        
        # Special handling for very low memory
        if low_mem_flag == "emergency":
            print(f"\n{'Emergency mode required:':<20}")
            print(f"  python build_trt_model_emergency.py tiny en")
        
        # Tips to free memory
        if gpu_free < 2000 or low_mem_flag in ["--low-memory", "emergency"]:
            print(f"\n⚠️  TIPS TO FREE MEMORY:")
            print(f"  1. Close all other applications")
            print(f"  2. Reboot device: sudo reboot")
            if gpu_free < 2000:
                print(f"  3. Stop desktop: sudo systemctl stop gdm3")
            print(f"  4. Check GPU usage: nvidia-smi")
            print(f"  5. Check processes: sudo tegrastats")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
