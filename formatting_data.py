"""
python formatting_data.py
"""

import csv
import json
import os

DOMAINS = ["roads", "content", "restaurants", "stations"]
AUDIO_ROOT = os.path.join("L2-KPNS", "data", "audio")


def list_speakers() -> list:
    return sorted(
        d for d in os.listdir(AUDIO_ROOT)
        if os.path.isdir(os.path.join(AUDIO_ROOT, d))
    )


def build_jsonl(domain: str, speaker: str) -> None:
    csv_path = os.path.join("L2-KPNS", "metadata", "recording_targets", f"{domain}_200.csv")
    audio_dir = os.path.join("L2-KPNS", "data", "audio", speaker)
    save_jsonl = os.path.join("L2-KPNS-jsonl", speaker, f"{domain}_{speaker}.jsonl")

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    results = []
    missing = []
    for row in rows:
        audio_path = f"{row['entity_id']}_{speaker}.wav"
        if not os.path.exists(os.path.join(audio_dir, audio_path)):
            missing.append(audio_path)

        results.append({"audio_path": audio_path, "answer": row["korean"]})

    os.makedirs(os.path.dirname(save_jsonl) or ".", exist_ok=True)
    with open(save_jsonl, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} entries to {save_jsonl}")
    if missing:
        print(f"⚠️ Missing audio files ({len(missing)}): {missing}")


if __name__ == "__main__":
    for speaker in list_speakers():
        for domain in DOMAINS:
            build_jsonl(domain, speaker)
