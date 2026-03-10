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


def transcribe_whisperx(
    audio_path: str, model_name: str = "base", num_speakers: int | None = None
) -> dict:
    """Transcribe with speaker diarization using WhisperX.

    Returns dict with keys: segments, language, num_speakers, duration.
    """
    import gc

    try:
        import whisperx
    except ImportError:
        print("Error: whisperx not installed.")
        print("  pip install -r requirements-whisperx.txt")
        sys.exit(1)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not set. Add your HuggingFace token to .env")
        print("  Get a token at: https://huggingface.co/settings/tokens")
        print("  Then accept terms at:")
        print("    https://huggingface.co/pyannote/segmentation-3.0")
        print("    https://huggingface.co/pyannote/speaker-diarization-3.1")
        sys.exit(1)

    device = "cpu"
    compute_type = "int8"

    # Step 1: Transcribe
    print(f"Loading WhisperX model '{model_name}'...")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    print("Loading audio...")
    audio = whisperx.load_audio(audio_path)

    print("Transcribing...")
    result = model.transcribe(audio, batch_size=4)
    language = result.get("language", "en")

    # Free transcription model memory before loading alignment model
    del model
    gc.collect()

    # Step 2: Align for word-level timestamps
    print("Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False,
    )

    del model_a
    gc.collect()

    # Step 3: Diarize (speaker identification)
    print("Diarizing speakers...")
    from whisperx.diarize import DiarizationPipeline

    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        token=hf_token, device=device,
    )
    diarize_kwargs = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    diarize_segments = diarize_model(audio, **diarize_kwargs)

    # Step 4: Assign speakers to segments
    print("Assigning speakers to segments...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Calculate duration (whisperx.load_audio returns 16kHz mono)
    duration_seconds = len(audio) / 16000

    speakers = {seg.get("speaker") for seg in result["segments"] if "speaker" in seg}

    return {
        "segments": result["segments"],
        "language": language,
        "num_speakers": len(speakers),
        "duration": duration_seconds,
    }


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_diarized_transcript(segments: list[dict]) -> str:
    """Merge adjacent same-speaker segments and format as diarized text."""
    if not segments:
        return "(No speech detected)"

    merged = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        end = seg.get("end", 0)

        if not text:
            continue

        if merged and merged[-1]["speaker"] == speaker:
            merged[-1]["text"] += " " + text
            merged[-1]["end"] = end
        else:
            merged.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
            })

    lines = []
    for block in merged:
        ts = format_timestamp(block["start"])
        lines.append(f'**{block["speaker"]}** ({ts})')
        lines.append(block["text"])
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe m4a/mp3 audio files to markdown."
    )
    parser.add_argument("audio_file", help="Path to .m4a or .mp3 file")
    parser.add_argument(
        "--mode",
        choices=["api", "local", "whisperx"],
        default="api",
        help="Transcription backend: 'api' (OpenAI), 'local' (Whisper), 'whisperx' (diarized)",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for local/whisperx mode (default: base)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Number of speakers for whisperx diarization (auto-detects if omitted)",
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
    if args.mode == "whisperx":
        mode_label = f"WhisperX ({args.model})"
    elif args.mode == "api":
        mode_label = "OpenAI Whisper API"
    else:
        mode_label = f"Local Whisper ({args.model})"
    print(f"Transcribing '{audio_path.name}' using {mode_label}...")

    if args.mode == "api":
        transcript = transcribe_api(str(audio_path))
    elif args.mode == "local":
        transcript = transcribe_local(str(audio_path), args.model)
    elif args.mode == "whisperx":
        wx_result = transcribe_whisperx(
            str(audio_path), args.model, args.num_speakers
        )

    # Write markdown output
    if args.mode == "whisperx":
        formatted_body = format_diarized_transcript(wx_result["segments"])
        duration_str = format_timestamp(wx_result["duration"])
        markdown = f"""# Transcript — {date_display}

**Source:** {audio_path.name}
**Date:** {date_str}
**Duration:** {duration_str}
**Speakers:** {wx_result["num_speakers"]}
**Mode:** {mode_label}

---

{formatted_body}
"""
    else:
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
