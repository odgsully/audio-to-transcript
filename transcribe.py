#!/usr/bin/env python3
"""Audio-to-Transcript Tool — Transcribes m4a/mp3 files to markdown."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_EXTENSIONS = {".m4a", ".mp3"}
OUTPUT_DIR = Path(__file__).parent / "outputs"


def transcribe_api(audio_path: str) -> str:
    """Transcribe using OpenAI Whisper API."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )

    return response.text


def transcribe_local(audio_path: str, model_name: str = "base") -> str:
    """Transcribe using local Whisper model."""
    import whisper

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print("Transcribing (this may take a while)...")
    result = model.transcribe(audio_path)

    return result["text"]


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe m4a/mp3 audio files to markdown."
    )
    parser.add_argument("audio_file", help="Path to .m4a or .mp3 file")
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="Transcription backend: 'api' (OpenAI Whisper API) or 'local' (default: api)",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for local mode (default: base)",
    )
    args = parser.parse_args()

    # Validate input file
    audio_path = Path(args.audio_file).resolve()
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"Error: Unsupported file type '{audio_path.suffix}'. Use .m4a or .mp3.")
        sys.exit(1)

    # Prepare output
    OUTPUT_DIR.mkdir(exist_ok=True)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    date_display = now.strftime("%B %-d, %Y")
    stem = audio_path.stem
    output_filename = f"transcript_{date_str}_{stem}.md"
    output_path = OUTPUT_DIR / output_filename

    # Transcribe
    mode_label = "OpenAI Whisper API" if args.mode == "api" else f"Local Whisper ({args.model})"
    print(f"Transcribing '{audio_path.name}' using {mode_label}...")

    if args.mode == "api":
        transcript = transcribe_api(str(audio_path))
    else:
        transcript = transcribe_local(str(audio_path), args.model)

    # Write markdown output
    markdown = f"""# Transcript — {date_display}

**Source:** {audio_path.name}
**Date:** {date_str}
**Mode:** {mode_label}

---

{transcript}
"""

    output_path.write_text(markdown)
    print(f"Transcript saved to: {output_path}")


if __name__ == "__main__":
    main()
