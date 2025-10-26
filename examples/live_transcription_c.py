# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Modified live transcription script for generic USB microphones
Works with any audio input device, not just ReSpeaker
"""

import numpy as np
import time
import pyaudio
from multiprocessing import Process, Queue, Event
from collections import deque
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional
from whisper_trt.vad import load_vad
from whisper_trt.model import MODEL_FILENAMES
from whisper_trt.cache import get_cache_dir


def find_audio_device_index(device_name_filter: str = None):
    """Find audio device by name. Returns first input device if filter is None."""
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get("deviceCount")

    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        
        # Check if it's an input device
        if device_info.get("maxInputChannels") > 0:
            if device_name_filter is None:
                return i
            if device_name_filter.lower() in device_info.get("name").lower():
                return i
    
    return None


@contextmanager
def get_audio_stream(
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        bitwidth: int = 2,
        chunk_size: int = 1536
    ):
    """Generic audio stream context manager"""
    
    p = pyaudio.PyAudio()

    if device_index is None:
        device_index = find_audio_device_index(device_name_filter='pulse')

    if device_index is None:
        raise RuntimeError("Could not find any audio input device.")
    
    # Get device info to verify
    device_info = p.get_device_info_by_host_api_device_index(0, device_index)
    print(f"Using audio device: {device_info.get('name')}")
    print(f"Channels: {channels}, Sample Rate: {sample_rate}")

    stream = p.open(
        rate=sample_rate,
        format=p.get_format_from_width(bitwidth),
        channels=channels,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk_size
    )

    try:
        yield stream
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def audio_numpy_from_bytes(audio_bytes: bytes):
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio


def audio_numpy_slice_channel(audio_numpy: np.ndarray, channel_index: int, 
                      num_channels: int = 1):
    if num_channels == 1:
        return audio_numpy
    return audio_numpy[channel_index::num_channels]


def audio_numpy_normalize(audio_numpy: np.ndarray):
    return audio_numpy.astype(np.float32) / 32768.0


@dataclass
class AudioChunk:
    audio_raw: bytes
    audio_numpy: np.ndarray
    audio_numpy_normalized: np.ndarray
    voice_prob: float | None = None


@dataclass
class AudioSegment:
    chunks: list


class Microphone(Process):

    def __init__(self, 
                 output_queue: Queue, 
                 chunk_size: int = 1536, 
                 device_index: int | None = None,
                 use_channel: int = 0, 
                 num_channels: int = 1,
                 sample_rate: int = 16000):
        super().__init__()
        self.output_queue = output_queue
        self.chunk_size = chunk_size
        self.use_channel = use_channel
        self.num_channels = num_channels
        self.device_index = device_index
        self.sample_rate = sample_rate

    def run(self):
        with get_audio_stream(
            sample_rate=self.sample_rate, 
            device_index=self.device_index, 
            channels=self.num_channels,
            chunk_size=self.chunk_size
        ) as stream:
            while True:
                audio_raw = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_numpy = audio_numpy_from_bytes(audio_raw)
                
                # Handle multi-channel by creating array per channel
                if self.num_channels > 1:
                    audio_numpy = np.stack([
                        audio_numpy_slice_channel(audio_numpy, i, self.num_channels) 
                        for i in range(self.num_channels)
                    ])
                else:
                    audio_numpy = audio_numpy.reshape(1, -1)
                    
                audio_numpy_normalized = audio_numpy_normalize(audio_numpy)

                audio = AudioChunk(
                    audio_raw=audio_raw,
                    audio_numpy=audio_numpy,
                    audio_numpy_normalized=audio_numpy_normalized
                )

                self.output_queue.put(audio)


class VAD(Process):

    def __init__(self,
            input_queue: Queue, 
            output_queue: Queue,
            sample_rate: int = 16000,
            use_channel: int = 0,
            speech_threshold: float = 0.5,
            max_filter_window: int = 1,
            ready_flag = None,
            speech_start_flag = None,
            speech_end_flag = None):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.sample_rate = sample_rate
        self.use_channel = use_channel
        self.speech_threshold = speech_threshold
        self.max_filter_window = max_filter_window
        self.ready_flag = ready_flag
        self.speech_start_flag = speech_start_flag
        self.speech_end_flag = speech_end_flag

    def run(self):

        vad = load_vad()
        
        # warmup run
        print("[VAD] Warming up...")
        vad(np.zeros(1536, dtype=np.float32), sr=self.sample_rate)
        print("[VAD] Ready!")

        max_filter_window = deque(maxlen=self.max_filter_window)

        speech_chunks = []

        prev_is_voice = False

        if self.ready_flag is not None:
            self.ready_flag.set()

        while True:
            
            audio_chunk = self.input_queue.get()

            voice_prob = float(vad(audio_chunk.audio_numpy_normalized[self.use_channel], sr=self.sample_rate).flatten()[0])

            chunk = AudioChunk(
                audio_raw=audio_chunk.audio_raw,
                audio_numpy=audio_chunk.audio_numpy,
                audio_numpy_normalized=audio_chunk.audio_numpy_normalized,
                voice_prob=voice_prob
            )

            max_filter_window.append(chunk)

            is_voice = any(c.voice_prob > self.speech_threshold for c in max_filter_window)
            
            if is_voice > prev_is_voice:
                speech_chunks = [chunk for chunk in max_filter_window]
                # start voice
                speech_chunks.append(chunk)
                if self.speech_start_flag is not None:
                    self.speech_start_flag.set()
            elif is_voice < prev_is_voice:
                # end voice
                segment = AudioSegment(chunks=speech_chunks)
                self.output_queue.put(segment)
                if self.speech_end_flag is not None:
                    self.speech_end_flag.set()
            elif is_voice:
                # continue voice
                speech_chunks.append(chunk)

            prev_is_voice = is_voice



class ASR(Process):

    def __init__(self, model: str, backend: str, input_queue, use_channel: int = 0, ready_flag = None, verbose: bool = False, language: str = "es"):
        super().__init__()
        self.model = model
        self.input_queue = input_queue
        self.use_channel = use_channel
        self.ready_flag = ready_flag
        self.backend = backend
        self.verbose = verbose
        self.language = language

    def run(self):
        
        print(f"[ASR] Loading {self.backend} model '{self.model}'...")
        if self.language:
            print(f"[ASR] Language: {self.language}")
        
        if self.backend == "whisper_trt":
            from whisper_trt import load_trt_model
            import os
            cache_path = os.path.join(get_cache_dir(), MODEL_FILENAMES[self.model])
            
            if os.path.exists(cache_path):
                print(f"[ASR] TensorRT engines found in cache: {cache_path}")
                print(f"[ASR] Loading from cache (should be fast)...")
            else:
                print(f"[ASR] ⚠️  WARNING: First time building TensorRT engines!")
                print(f"[ASR] This will take 5-15 minutes. Please be patient...")
                print(f"[ASR] The engines will be cached at: {cache_path}")
                print(f"[ASR] Subsequent runs will be much faster!")
                if self.verbose:
                    print(f"[ASR] Verbose mode enabled - you will see TensorRT build logs")
                
            model = load_trt_model(self.model, verbose=self.verbose, language=self.language)
        elif self.backend == "whisper":
            from whisper import load_model
            model = load_model(self.model)
        elif self.backend == "faster_whisper":
            from faster_whisper import WhisperModel
            class FasterWhisperWrapper:
                def __init__(self, model):
                    self.model = model
                def transcribe(self, audio):
                    segs, info = self.model.transcribe(audio)
                    text = "".join([seg.text for seg in segs])
                    return {"text": text}
                
            model = FasterWhisperWrapper(WhisperModel(self.model))

        # warmup
        print("[ASR] Warming up model...")
        model.transcribe(np.zeros(16000, dtype=np.float32))
        print("[ASR] Ready!")

        if self.ready_flag is not None:
            self.ready_flag.set()

        while True:

            speech_segment = self.input_queue.get()

            t0 = time.perf_counter_ns()
            audio = np.concatenate([chunk.audio_numpy_normalized[self.use_channel] for chunk in speech_segment.chunks])

            result = model.transcribe(audio)
            text = result['text']

            t1 = time.perf_counter_ns()

            print(f"\n{'='*80}")
            print(f"Text: {text}")
            print(f"Transcription Time: {(t1 - t0) / 1e9:.3f}s")
            print(f"Audio Duration: {len(audio) / 16000:.3f}s")
            print(f"{'='*80}\n")


class StartEndMonitor(Process):

    def __init__(self, start_flag: Event, end_flag):
        super().__init__()
        self.start_flag = start_flag
        self.end_flag = end_flag

    def run(self):
        while True:
            self.start_flag.wait()
            self.start_flag.clear()
            print(f"🎤 Speech started...")
            self.end_flag.wait()
            self.end_flag.clear()
            print(f"🔇 Speech ended. Processing...")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Live audio transcription with Whisper")
    parser.add_argument("model", type=str, help="Model name (tiny.en, base.en, small.en, tiny, base, small)")
    parser.add_argument("--backend", type=str, default="whisper_trt", 
                       choices=["whisper", "whisper_trt", "faster_whisper"],
                       help="Backend to use for transcription")
    parser.add_argument("--device", type=int, default=None,
                       help="Audio device index (see list_audio_devices.py)")
    parser.add_argument("--channels", type=int, default=1,
                       help="Number of audio channels (1 for mono, 2 for stereo)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Voice activity detection threshold (0.0-1.0)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output (shows TensorRT build logs)")
    parser.add_argument("--language", type=str, default="es",
                       help="Language for transcription (es, en, fr, de, etc.) - only for multilingual models")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("Live Transcription Started")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Language: {args.language}")
    print(f"Device: {args.device if args.device else 'auto-detect'}")
    print(f"Channels: {args.channels}")
    print(f"VAD Threshold: {args.threshold}")
    print(f"Verbose: {args.verbose}")
    print("="*80 + "\n")

    audio_chunks = Queue()
    speech_segments = Queue()
    vad_ready = Event()
    asr_ready = Event()
    speech_start = Event()
    speech_end = Event()

    asr = ASR(args.model, args.backend, speech_segments, ready_flag=asr_ready, verbose=args.verbose, language=args.language)
    vad = VAD(audio_chunks, speech_segments, max_filter_window=5, 
             speech_threshold=args.threshold,
             ready_flag=vad_ready, speech_start_flag=speech_start, 
             speech_end_flag=speech_end)
    mic = Microphone(audio_chunks, device_index=args.device, num_channels=args.channels)
    mon = StartEndMonitor(speech_start, speech_end)

    print("Starting processes...")
    vad.start()
    asr.start()
    mon.start()

    print("Waiting for VAD and ASR to be ready...")
    vad_ready.wait()
    asr_ready.wait()

    print("\n🎙️  Microphone active - start speaking!\n")
    mic.start()

    try:
        mic.join()
        vad.join()
        asr.join()
        mon.join()
    except KeyboardInterrupt:
        print("\n\nStopping...")
