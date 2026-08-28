"""
Run every speaker x domain x (asr_adaptation off / on x ref_text) combination in one go
and save the summary (ASR-Hit, Hit@1/5/10, WER) as csv.

python -m src.run_all_experiments                        # all of SPEAKERS
python -m src.run_all_experiments --speakers P001,P002   # a subset (when splitting across jobs)
python -m src.run_all_experiments --merge-only           # merge per-speaker summaries, no inference
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import sys
from typing import List, Optional

import openpyxl
from tqdm import tqdm

import compute_metrics

# infer_qwen2.5_omni_batch.py has a "." in its filename, so it cannot be pulled in with a
# plain import statement -- load it by path via importlib. dataclass consults sys.modules when
# resolving forward refs, so the module must be registered there before exec_module.
_spec = importlib.util.spec_from_file_location(
    "infer_qwen2_5_omni_batch",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer_qwen2.5_omni_batch.py"),
)
infer_mod = importlib.util.module_from_spec(_spec)
sys.modules["infer_qwen2_5_omni_batch"] = infer_mod
_spec.loader.exec_module(infer_mod)

# ══════════════════════════════════════════════════════════════════════════
# ⚙️  Experiment settings -- this block is the only place to edit when changing conditions
# ══════════════════════════════════════════════════════════════════════════

# 📂 Input data
AUDIO_ROOT = "L2-KPNS/data/audio"
DOMAINS = ["roads", "content"] # "restaurants", "stations"]
BATCH_SIZE = 8


# 🎙️  SPEAKERS -- speakers used in the experiment
def list_speakers() -> List[str]:
    return sorted(
        d for d in os.listdir(AUDIO_ROOT)
        if os.path.isdir(os.path.join(AUDIO_ROOT, d))
    )

SPEAKERS = [
    'P001', 'P002', 'P003', 'P004', 'P005',
    'P009', 'P010', 'P011', 'P012',
    'P014', 'P015',
    'P017',
    'P019', 'P020',
]
# SPEAKERS = list_speakers()   # 👈 swap in this line to use every speaker in the audio directory


# 📝 REF_TEXTS -- reference texts used for asr_adaptation
def load_ref_text_to_entity_id(WAKEWORD_CSV) -> dict:
    mapping = {}
    with open(WAKEWORD_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mapping[row["korean"].strip().rstrip(".")] = row["entity_id"]
    return mapping

WAKEWORD_CSV = "L2-KPNS/metadata/recording_targets/wakeword_candidates.csv"

# Instead of hardcoding, use every candidate in wakeword_candidates.csv, in csv order
REF_TEXT_TO_ENTITY_ID = load_ref_text_to_entity_id(WAKEWORD_CSV)
REF_TEXTS = list(REF_TEXT_TO_ENTITY_ID)

# 👇 The older approach of picking ref_texts by hand.
# REF_TEXTS = [
#     # "안녕 안드로이드",
#     # "부엌에서 된장찌개를 끓였어",
#     "원오대길",
#     "세개의달",
#     # "남해술상",
#     # "걸포북변",
#     # "Hello Android",
#     # "The hogs were fed chopped corn and garbage",
#     "Android",
#     "안드로이드",
#     "안녕 안드로이드",
#     "안녕 갤럭시",
#     "안녕 루미나",
#     "안녕 스마트 미러",
#     "안녕 비서야",
#     "Hello Android",
#     "Hello Galaxy",
#     "Hello Lumina",
#     "Hello Smart Mirror",
#     "Hello assistant"
# ]


# 💾 Where results are written
SUMMARY_CSV = "L2-KPNS-jsonl/results/summary_all.csv"
SUMMARY_XLSX = "L2-KPNS-jsonl/results/summary_all.xlsx"
# SUMMARY_CSV = "L2-KPNS-jsonl/results/summary_android_eng.csv"
# SUMMARY_XLSX = "L2-KPNS-jsonl/results/summary_android_eng.xlsx"

SUMMARY_FIELDS = [
    "speaker", "domain", "asr_adaptation", "ref_aud_tag", "ref_text", "num_rows",
    "WER", "PED", "ASR-Hit", "Hit@1",
]

# ══════════════════════════════════════════════════════════════════════════
# End of settings -- everything below is the run logic
# ══════════════════════════════════════════════════════════════════════════


def entity_id_suffix(entity_id: str) -> str:
    """Extract just the suffix (rd) from an entity_id (e.g. pr0021_rd)."""
    return entity_id.rsplit("_", 1)[-1]


def resolve_ref_audio_path(entity_id: str, speaker: str) -> Optional[str]:
    """Use the speaker's own recording if it exists, otherwise None."""
    speaker_path = f"L2-KPNS/data/audio/{speaker}/{entity_id}_{speaker}.wav"
    if os.path.exists(speaker_path):
        return speaker_path
    print(f"  [error] {speaker_path} 없음")
    return None


def build_entity_dict_ipa(domain: str) -> dict:
    lexicon_path = f"L2-KPNS/metadata/recording_targets/{domain}_200.csv"
    with open(lexicon_path, encoding="utf-8-sig") as f:
        entity_dict = {row["entity_id"]: row["korean"] for row in csv.DictReader(f)}
    entity_names = list(dict.fromkeys(entity_dict.values()))  # dedupe, preserve order
    return {e: infer_mod.epi.transliterate(e) for e in entity_names}


def run_one_config(
    asr: "infer_mod.Qwen25OmniASRPipeline",
    speaker: str,
    domain: str,
    data: List[dict],
    entity_dict_ipa: dict,
    asr_ref_audio: Optional[object],
    entity_id: str,
    ref_aud_tag: str,
    ref_text: str,
    asr_adaptation: bool,
) -> Optional[str]:
    """Run one combination (speaker/domain/adaptation/ref_text) in batches of BATCH_SIZE and
    return the path of the result jsonl.

    If any audio file is missing, the whole combination is skipped (returns None).
    If save_jsonl already exists, ASR inference is skipped and metrics are recomputed from the
    existing results and appended to the summary."""
    save_jsonl_tag = f"_{entity_id}" if asr_adaptation else ""
    save_jsonl = (
        f"L2-KPNS-jsonl/results/{speaker}/"
        f"{domain}_asr{'_ada' if asr_adaptation else ''}{save_jsonl_tag}.jsonl"
    )

    if os.path.exists(save_jsonl):
        print(f"  [skip] {save_jsonl} 이미 존재, ASR 추론 skip 후 summary 만 갱신")
        results = infer_mod.load_jsonl(save_jsonl)
    else:
        results = []
        pbar = tqdm(total=len(data), desc=f"{speaker}/{domain}", leave=False)
        for batch_rows in infer_mod.chunked(data, BATCH_SIZE):
            batch_audio_paths = [
                f"L2-KPNS/data/audio/{speaker}/" + row["audio_path"] for row in batch_rows
            ]
            missing = next((p for p in batch_audio_paths if not os.path.exists(p)), None)
            if missing is not None:
                pbar.close()
                print(f"  [error] {missing} 없음, 이 조합({speaker}/{domain}) 전체 skip")
                return None
            batch_audio = [infer_mod.load_audio_16k(p) for p in batch_audio_paths]

            batch_out = asr.run_asr_batch(
                audios=batch_audio, ref_audio=asr_ref_audio, reference_text=ref_text,
            )

            for row, out in zip(batch_rows, batch_out):
                entity_list_text = [e[0] for e in infer_mod.retrieve_top_k(
                    utterance=out,
                    entity_ipa=entity_dict_ipa,
                    k=10,
                )]

                result = dict(row)
                result["asr_result"] = out
                result["retr_entities"] = entity_list_text
                results.append(result)
            pbar.update(len(batch_rows))
        pbar.close()

        os.makedirs(os.path.dirname(save_jsonl) or ".", exist_ok=True)
        with open(save_jsonl, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = compute_metrics.compute_metrics(results)
    write_summary_row(
        speaker, domain, asr_adaptation, ref_aud_tag, ref_text, len(results), metrics,
    )
    return save_jsonl


def write_summary_row(
    speaker, domain, asr_adaptation, ref_aud_tag, ref_text, num_rows, metrics,
):
    row = {
        "speaker": speaker,
        "domain": domain,
        "asr_adaptation": asr_adaptation,
        "ref_aud_tag": ref_aud_tag,
        "ref_text": ref_text,
        "num_rows": num_rows,
        "WER": metrics["wer"][0],
        "PED": metrics["phoneme_editdistance"][0],
        "ASR-Hit": metrics["recall@asr"][0],
        "Hit@1": metrics["recall@1"][0],
    }
    with open(speaker_summary_csv(speaker), "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writerow(row)


def speaker_summary_csv(speaker: str) -> str:
    """Write the summary to a per-speaker file. Each GPU worker owns a different speaker, so the
    files never collide. (Several workers appending to the single SUMMARY_CSV was what caused
    duplicated and interleaved rows.)"""
    return f"{os.path.dirname(SUMMARY_CSV)}/{speaker}/summary.csv"


def init_speaker_summary_csv(speaker: str) -> None:
    """Create the speaker summary file from scratch (re-running a speaker discards its old rows)."""
    path = speaker_summary_csv(speaker)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writeheader()


def merge_speaker_summaries() -> None:
    """Merge the per-speaker summary files into SUMMARY_CSV (rewritten from scratch rather than
    appended, so duplicates are impossible)."""
    os.makedirs(os.path.dirname(SUMMARY_CSV) or ".", exist_ok=True)
    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for speaker in SPEAKERS:
            path = speaker_summary_csv(speaker)
            if not os.path.exists(path):
                print(f"  [warn] {path} 없음, 요약에서 제외")
                continue
            with open(path, encoding="utf-8", newline="") as f:
                writer.writerows(csv.DictReader(f))


def append_speaker_average_rows() -> None:
    """Compute a speaker-average row per (domain, asr_adaptation, ref_aud_tag, ref_text) and
    append it to the summary csv."""
    with open(SUMMARY_CSV, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["speaker"] != "avg"]  # do not feed averages back into the average

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["domain"], row["asr_adaptation"], row["ref_aud_tag"], row["ref_text"])
        groups.setdefault(key, []).append(row)

    avg_rows = []
    for (domain, asr_adaptation, ref_aud_tag, ref_text) in sorted(groups):
        group_rows = groups[(domain, asr_adaptation, ref_aud_tag, ref_text)]
        n = len(group_rows)
        avg_rows.append({
            "speaker": "avg",
            "domain": domain,
            "asr_adaptation": asr_adaptation,
            "ref_aud_tag": ref_aud_tag,
            "ref_text": ref_text,
            "num_rows": sum(int(r["num_rows"]) for r in group_rows),
            "WER": round(sum(float(r["WER"]) for r in group_rows) / n, 2),
            "PED": round(sum(float(r["PED"]) for r in group_rows) / n, 2),
            "ASR-Hit": round(sum(float(r["ASR-Hit"]) for r in group_rows) / n, 2),
            "Hit@1": round(sum(float(r["Hit@1"]) for r in group_rows) / n, 2),
        })

    with open(SUMMARY_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writerows(avg_rows)


def convert_summary_csv_to_xlsx() -> None:
    """After every process has finished, convert the final summary_all.csv to summary_all.xlsx."""
    wb = openpyxl.Workbook()
    ws = wb.active
    with open(SUMMARY_CSV, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            ws.append(row)
    wb.save(SUMMARY_XLSX)


def gpu_worker(gpu_id: int, speaker_queue: "mp.Queue[str]") -> None:
    """One worker per GPU. Speakers are pulled off the queue one at a time and run sequentially,
    so even with more speakers than GPUs each GPU keeps exactly one process running."""
    asr = infer_mod.Qwen25OmniASRPipeline(infer_mod.ASRConfig(device=f"cuda:{gpu_id}"))
    ref_text_to_entity_id = load_ref_text_to_entity_id(WAKEWORD_CSV)

    while True:
        try:
            speaker = speaker_queue.get_nowait()
        except queue.Empty:
            break
        run_for_speaker(asr, ref_text_to_entity_id, speaker, gpu_id)


def run_for_speaker(
    asr: "infer_mod.Qwen25OmniASRPipeline",
    ref_text_to_entity_id: dict,
    speaker: str,
    gpu_id: int,
) -> None:
    """Run every domain x ref_text combination for a single speaker, sequentially."""
    init_speaker_summary_csv(speaker)

    # Load and cache the audio matching each ref_text once per speaker.
    ref_audio_by_text = {}
    for ref_text in REF_TEXTS:
        entity_id = ref_text_to_entity_id[ref_text]
        ref_audio_path = resolve_ref_audio_path(entity_id, speaker)
        if ref_audio_path is None:
            continue
        ref_audio_by_text[ref_text] = infer_mod.load_audio_16k(ref_audio_path)

    for domain in DOMAINS:
        data_jsonl = f"L2-KPNS-jsonl/{speaker}/{domain}_{speaker}.jsonl"
        data = infer_mod.load_jsonl(data_jsonl)
        entity_dict_ipa = build_entity_dict_ipa(domain)
        print(f"===== [GPU {gpu_id}] speaker={speaker} domain={domain} ({len(data)} rows) =====")

        print(f"[GPU {gpu_id}] -- asr_adaptation=off --")
        run_one_config(
            asr, speaker, domain, data, entity_dict_ipa,
            asr_ref_audio=None, entity_id="", ref_aud_tag="", ref_text="", asr_adaptation=False,
        )

        for ref_text in REF_TEXTS:
            if ref_text not in ref_audio_by_text:
                print(f"[GPU {gpu_id}] -- skip asr_adaptation=on ref_text='{ref_text}' (ref audio 없음) --")
                continue
            print(f"[GPU {gpu_id}] -- asr_adaptation=on ref_text='{ref_text}' --")
            entity_id = ref_text_to_entity_id[ref_text]
            run_one_config(
                asr, speaker, domain, data, entity_dict_ipa,
                asr_ref_audio=ref_audio_by_text[ref_text],
                entity_id=entity_id,
                ref_aud_tag=entity_id_suffix(entity_id),
                ref_text=ref_text, asr_adaptation=True,
            )


def finalize_summary() -> None:
    merge_speaker_summaries()
    append_speaker_average_rows()
    convert_summary_csv_to_xlsx()
    print(f"Saved summary to {SUMMARY_XLSX}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers", help="Comma-separated speaker list. Defaults to all of SPEAKERS. "
                                           "When splitting the run across jobs, make sure the lists do not overlap.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip inference and just merge the per-speaker summaries into the summary csv/xlsx.")
    args = parser.parse_args()

    if args.merge_only:
        finalize_summary()
        sys.exit(0)

    mp.set_start_method("spawn", force=True)

    speakers = [s.strip() for s in args.speakers.split(",")] if args.speakers else SPEAKERS
    num_gpus = infer_mod.torch.cuda.device_count() or 1
    print(f"Detected {num_gpus} GPU(s); processing {len(speakers)} speaker(s) {speakers} "
          f"x {len(REF_TEXTS)} ref_text(s) via a queue.")

    speaker_queue: "mp.Queue[str]" = mp.Queue()
    for speaker in speakers:
        speaker_queue.put(speaker)

    procs = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=gpu_worker, args=(gpu_id, speaker_queue), name=f"gpu-{gpu_id}")
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    if args.speakers:
        # Another job may still be running the remaining speakers, so do not merge here.
        print("--speakers 로 일부만 실행했습니다. 모든 잡이 끝난 뒤 "
              "`python run_all_experiments.py --merge-only` 를 실행하세요.")
    else:
        finalize_summary()
