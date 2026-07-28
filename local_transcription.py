"""Used for testing purposes. This script can be run directly to transcribe a local audio file using the backend modules."""

import sys
from pathlib import Path
import os
import ctypes

# --- Import NVIDIA Libraries ---
try:
    import nvidia.cublas.lib
    import nvidia.cudnn
    
    # Costruisci i percorsi assoluti ai file .so installati da uv
    cublas_so = os.path.join(nvidia.cublas.lib.__path__[0], "libcublas.so.12")
    cudnn_so = os.path.join(nvidia.cudnn.__path__[0], "lib", "libcudnn.so.9")
    
    # Carica le librerie direttamente in RAM per renderle visibili a CTranslate2
    ctypes.CDLL(cublas_so, mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(cudnn_so, mode=ctypes.RTLD_GLOBAL)
except Exception as e:
    print(f"NVIDIA libs pre-load failed (puoi ignorare se usi CPU o Docker): {e}")

# ------------------------

from backend.modules.preprocessing.audio_processor import AudioProcessor
from backend.modules.transcription.whisper_engine import TranscriptionEngine

def transcribe(file_path: str) -> None:
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    raw_bytes = path.read_bytes()
    content_type = f"audio/{path.suffix.lstrip('.')}"  # e.g., audio/wav, audio/mp3

    print (f"Transcribing {file_path} (Content-Type: {content_type})...")
    print (f"Size: {len(raw_bytes) / 1024:.1f} KB")
    print ("-" * 60)

    # Preprocessing audio
    print ("[1/2] Preprocessing audio...")
    audio_processor = AudioProcessor()
    processed_audio = audio_processor.process(
        raw_bytes=raw_bytes,
        content_type=content_type,
        filename=path.name,
    )
    print(f"  Duration: {processed_audio.duration:.2f}s")
    print(f"  Sample rate: {processed_audio.sample_rate} Hz")
    print(f"  Format: {processed_audio.original_format}")
    print()

    # Transcription
    print ("[2/2] Transcribing audio...")
    engine = TranscriptionEngine()
    result = engine.transcribe(
        audio_data=processed_audio.data,
        sr=processed_audio.sample_rate,
        audio_duration = processed_audio.duration,
    )
    engine.close()

    print(f"  Language: {result.language}")
    print(f"  Segments: {len(result.segments)}")
    print()
    print("=" * 60)

    output_path = path.with_suffix(".txt")
    output_path.write_text("\n".join(seg.text for seg in result.segments), encoding="utf-8")

    print(f"TRANSCRIPTION COMPLETED! File saved to: {output_path}")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python local_transcription.py <audio_file>")
        sys.exit(1)

    transcribe(sys.argv[1])