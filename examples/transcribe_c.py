# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import os
import argparse
from whisper_trt import load_trt_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=["tiny.en", "base.en", "small.en", "tiny", "base", "small"])
    parser.add_argument("audio", type=str)
    parser.add_argument("--backend", type=str, choices=["whisper", "whisper_trt", "faster_whisper"], default="whisper_trt")
    parser.add_argument("--language", type=str, default="es", help="Language for transcription (es, en, fr, de, etc.) - only for multilingual models")
    args = parser.parse_args()

    print(f"[DEBUG] Backend: {args.backend}")
    print(f"[DEBUG] Model: {args.model}")
    print(f"[DEBUG] Audio: {args.audio}")

    if args.backend == "whisper":

        from whisper import load_model

        print("[DEBUG] Loading whisper model...")
        model = load_model(args.model)
        print("[DEBUG] Model loaded successfully")

        print("[DEBUG] Starting transcription...")
        # Add language parameter for multilingual models
        if not args.model.endswith('.en'):
            result = model.transcribe(args.audio, language=args.language)
        else:
            result = model.transcribe(args.audio)
        
    elif args.backend == "whisper_trt":

        from whisper_trt import load_trt_model

        print("[DEBUG] Loading TensorRT model...")
        print("[DEBUG] WARNING: First run will take 5-15 minutes to build TensorRT engines!")
        print("[DEBUG] Building/loading with verbose output enabled...")
        import time
        start_time = time.time()
        # Pass language parameter for multilingual models
        if not args.model.endswith('.en'):
            model = load_trt_model(args.model, verbose=True, language=args.language)
        else:
            model = load_trt_model(args.model, verbose=True)
        elapsed = time.time() - start_time
        print(f"[DEBUG] TensorRT model loaded successfully in {elapsed:.2f} seconds")

        print("[DEBUG] Starting transcription...")
        result = model.transcribe(args.audio)
        print("[DEBUG] Transcription completed")

    elif args.backend == "faster_whisper":

        from faster_whisper import WhisperModel

        print("[DEBUG] Loading faster_whisper model...")
        model = WhisperModel(args.model)
        print("[DEBUG] Model loaded successfully")

        print("[DEBUG] Starting transcription...")
        # Add language parameter for multilingual models
        if not args.model.endswith('.en'):
            segs, info = model.transcribe(args.audio, language=args.language)
        else:
            segs, info = model.transcribe(args.audio)

        result = {"text": ""}
        for seg in segs:
            result["text"] += seg.text

    print("[DEBUG] Final result:")
    print(result["text"])
