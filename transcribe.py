#!/usr/bin/env python3
"""Transcribe a WAV file to text using Whisper.

Usage:
    python transcribe.py input.wav output.txt
    python transcribe.py input.wav output.txt --model medium
    python transcribe.py input.wav output.txt --chunk-minutes 10
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
import whisper


# Whisper works best with chunks under 30 minutes.
DEFAULT_CHUNK_MINUTES = 20
OVERLAP_SECONDS = 5  # overlap between chunks to avoid cutting mid-word
PAUSE_THRESHOLD = 1.5  # seconds of silence between segments to insert a line break


def segments_to_text(segments):
    """Join segment texts, inserting a blank line where there's a pause."""
    lines = []
    current_paragraph = []

    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        if not text:
            continue

        if i > 0 and current_paragraph:
            gap = seg["start"] - segments[i - 1]["end"]
            if gap >= PAUSE_THRESHOLD:
                lines.append(" ".join(current_paragraph))
                lines.append("")  # blank line for the pause
                current_paragraph = []

        current_paragraph.append(text)

    if current_paragraph:
        lines.append(" ".join(current_paragraph))

    return "\n".join(lines)


def get_audio_info(path):
    """Return (samplerate, total_frames, duration_seconds)."""
    info = sf.info(path)
    return info.samplerate, info.frames, info.duration


def transcribe_file(input_path, output_path, model_name="base", chunk_minutes=DEFAULT_CHUNK_MINUTES):
    if not os.path.isfile(input_path):
        sys.exit(f"Input file not found: {input_path}")

    samplerate, total_frames, duration = get_audio_info(input_path)
    chunk_seconds = chunk_minutes * 60

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print("Model loaded.")
    print(f"Input: {input_path}  ({duration:.1f}s, {samplerate} Hz)")

    if duration <= chunk_seconds:
        # Small file — transcribe directly
        print("Transcribing...")
        result = model.transcribe(input_path, language="en", fp16=False)
        text = segments_to_text(result["segments"])
    else:
        # Large file — split into overlapping chunks
        chunk_frames = int(chunk_seconds * samplerate)
        overlap_frames = int(OVERLAP_SECONDS * samplerate)
        num_chunks = 1 + (total_frames - chunk_frames) // (chunk_frames - overlap_frames)
        if (total_frames - chunk_frames) % (chunk_frames - overlap_frames) > 0:
            num_chunks += 1

        print(f"Large file detected ({duration:.1f}s). Splitting into {num_chunks} chunks of ~{chunk_minutes} min each.")

        segments = []
        offset = 0
        chunk_idx = 0

        while offset < total_frames:
            chunk_idx += 1
            end = min(offset + chunk_frames, total_frames)
            frames_to_read = end - offset

            data, _ = sf.read(input_path, start=offset, stop=end, dtype="float32")
            # If stereo, convert to mono
            if data.ndim > 1:
                data = data.mean(axis=1)

            # Write chunk to a temp file
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()

            try:
                sf.write(tmp_path, data, samplerate, subtype="PCM_16")
                print(f"  Chunk {chunk_idx}/{num_chunks} ({frames_to_read / samplerate:.1f}s)...")
                result = model.transcribe(tmp_path, language="en", fp16=False)
                chunk_text = segments_to_text(result["segments"])
                segments.append(chunk_text)
            finally:
                os.unlink(tmp_path)

            offset += chunk_frames - overlap_frames

        text = "\n\n".join(segments)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"Transcription saved to: {output_path}")
    print(f"Length: {len(text)} characters")


def main():
    parser = argparse.ArgumentParser(description="Transcribe a WAV file to text using Whisper.")
    parser.add_argument("input", help="Path to the input WAV file")
    parser.add_argument("output", help="Path to the output text file")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--chunk-minutes", type=int, default=DEFAULT_CHUNK_MINUTES,
                        help=f"Max chunk length in minutes for large files (default: {DEFAULT_CHUNK_MINUTES})")
    args = parser.parse_args()

    transcribe_file(args.input, args.output, model_name=args.model, chunk_minutes=args.chunk_minutes)


if __name__ == "__main__":
    main()
