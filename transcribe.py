import os
import json
import pandas as pd
import torch
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import assign_word_speakers

# Configuration
AUDIO_FILE = "audio.wav"
LANGUAGE = "ru"
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
OUTPUT_DIR = "output"
MODEL_DIR = "models"


def save_json(data, path):
    def convert(o):
        if hasattr(o, "start") and hasattr(o, "end"):
            return {"start": float(o.start), "end": float(o.end)}
        if isinstance(o, (pd.Timestamp, pd.Timedelta)):
            return str(o)
        raise TypeError(f"Type {type(o).__name__} not serializable")

    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=convert)


def load_json_as_dataframe(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to load JSON from {path}. File may be corrupted.")
    return pd.DataFrame(data)


def main():
    load_dotenv()
    hf_auth_token = os.getenv("HF_AUTH_TOKEN")
    if not hf_auth_token:
        raise RuntimeError("Hugging Face token not found. Set HF_AUTH_TOKEN in .env file.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load model and audio
    print(f"======== Loading WhisperX model ========")
    asr_model = whisperx.load_model("large-v3", device=device, compute_type=compute_type, download_root=MODEL_DIR, language=LANGUAGE)
    print(f"======== Loading audio ========")
    audio = whisperx.load_audio(AUDIO_FILE)

    # ASR
    asr_path = os.path.join(OUTPUT_DIR, "asr_result.json")
    if os.path.exists(asr_path):
        print(f"======== Loading ASR result from {asr_path} ========")
        with open(asr_path, "r", encoding="utf-8") as f:
            asr_result = json.load(f)
    else:
        print(f"======== Running ASR ========")
        asr_result = asr_model.transcribe(audio, language=LANGUAGE)
        print(f"======== Saving ASR result to {asr_path} ========")
        save_json(asr_result, asr_path)

    # Alignment
    align_path = os.path.join(OUTPUT_DIR, "alignment_result.json")
    if os.path.exists(align_path):
        print(f"======== Loading alignment result ========")
        with open(align_path, "r", encoding="utf-8") as f:
            alignment = json.load(f)
    else:
        print(f"======== Aligning segments ========")
        align_model, align_meta = whisperx.load_align_model(language_code=LANGUAGE, device=device)
        alignment = whisperx.align(
            asr_result["segments"], align_model, align_meta, audio, device,
            return_char_alignments=False
        )
        print(f"======== Saving align result to {align_path} ========")
        save_json(alignment, align_path)

    # Speaker diarization
    diar_path = os.path.join(OUTPUT_DIR, "diarization_result.json")
    if os.path.exists(diar_path):
        print(f"======== Loading diarization result ========")
        diar_df = load_json_as_dataframe(diar_path)
    else:
        print(f"======== Running diarization ========")
        pipeline = whisperx.DiarizationPipeline(use_auth_token=hf_auth_token, device=device)
        diar = pipeline(audio)
        print(f"======== Saving diarization result to {diar_path} ========")
        save_json(diar, diar_path)
        diar_df = pd.DataFrame(diar)

    # Combine ASR & diarization
    print(f"======== Assigning speakers to words ========")
    if isinstance(alignment, dict) and "segments" in alignment:
        segments = alignment
    elif isinstance(alignment, list):
        segments = {"segments": alignment}
    else:
        raise TypeError(f"Unexpected type for alignment: {type(alignment)}")

    result = assign_word_speakers(diar_df, segments)

    print(f"Result type: {type(result)}")
    if isinstance(result, list) and result:
        print(f"First 2 items: {result[:2]}")

    # Save to Markdown
    print(f"======== Saving transcript to Markdown ========")
    md_path = os.path.join(OUTPUT_DIR, "transcript.md")
    with open(md_path, "w", encoding="utf-8") as f:
        for seg in result.get("segments", []):
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "").strip() or " ".join(w.get("word", "") for w in seg.get("words", []))
            f.write(f"**[{speaker}]**: {text}\n\n")

    print(f"======== Done! Transcript saved to {md_path} ========")


if __name__ == "__main__":
    main()
