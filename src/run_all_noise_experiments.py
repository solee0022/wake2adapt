"""
Run every speaker x domain x (asr_adaptation off / on x ref_text) x noise combination --
including the noise conditions (noise set x SNR x shift) -- in one go and save the summary
(ASR-Hit, Hit@1, WER, PED) as csv.

Noise setup (paper setting):
  For each main utterance one background utterance is sampled at random, cropped/tiled to the
  length of the main utterance, and mixed in at the given SNR.
  shift(%) is how far after the start of the main utterance the background begins, as a
  fraction of the main length: 0% = fully overlapping, 50% = only the second half overlaps,
  100% = no overlap, appended after the main utterance.

python -m src.run_all_noise_experiments                              # all of SPEAKERS x NOISE_CONDITIONS
python -m src.run_all_noise_experiments --speakers P001,P002         # a subset of speakers (splitting across jobs)
python -m src.run_all_noise_experiments --snrs 5,10 --shifts 0       # a subset of noise conditions
python -m src.run_all_noise_experiments --noise-sets Ksponspeech     # a subset of noise sets
python -m src.run_all_noise_experiments --include-clean              # also run the noise-free baseline
python -m src.run_all_noise_experiments --dump-mix                   # save the first mixture wav per combination (to listen to)
python -m src.run_all_noise_experiments --merge-only                 # merge per-speaker summaries, no inference
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import queue
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import openpyxl
import soundfile as sf
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
BATCH_SIZE = 128


# 🔊 NOISE -- speech mixed in as the background utterance
NOISE_ROOT = "noise_audios"
NOISE_SETS = ["Ksponspeech", "Librispeech"]   # subdirectory names under NOISE_ROOT
SNRS_DB = [1, 5, 10, 20, 50]                  # main : background power ratio (dB)
SHIFTS_PCT = [0, 50, 100]                     # background start offset (% of the main length)
NOISE_SEED = 1234                             # fixed so background selection is reproducible
NOISE_ON_REF_AUDIO = False                    # keep the 1-shot reference audio clean by default
INCLUDE_CLEAN = True                         # whether to also run the noise-free baseline (--include-clean)


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
# REF_TEXTS = list(REF_TEXT_TO_ENTITY_ID)

# 👇 The older approach of picking ref_texts by hand. Uncomment the list below to run a subset.
REF_TEXTS = [
    "원오대길",
    "세개의달",
    "Android",
    "안드로이드",
    # "안녕 안드로이드",
]


# 💾 Where results are written (under results_noise, kept apart from the clean experiment)
SUMMARY_CSV = "L2-KPNS-jsonl/results_noise/summary_all.csv"
SUMMARY_XLSX = "L2-KPNS-jsonl/results_noise/summary_all.xlsx"
RESULT_ROOT = "L2-KPNS-jsonl/results_noise"
MIX_DUMP_ROOT = "L2-KPNS-jsonl/results_noise/_mix_samples"   # where --dump-mix writes

SUMMARY_FIELDS = [
    "speaker", "domain", "noise_set", "snr_db", "shift_pct",
    "asr_adaptation", "ref_aud_tag", "ref_text", "num_rows",
    "WER", "PED", "ASR-Hit", "Hit@1",
]

# ══════════════════════════════════════════════════════════════════════════
# End of settings -- everything below is the run logic
# ══════════════════════════════════════════════════════════════════════════

DUMP_MIX = False   # turned on by --dump-mix (saves only the first mixture per combination as wav)

# A tuple describing one noise condition: (noise_set, snr_db, shift_pct)
# An empty noise_set means clean (no noise applied).
NoiseCond = Tuple[str, Optional[int], Optional[int]]

CLEAN_COND: NoiseCond = ("", None, None)


def build_noise_conditions(
    noise_sets: List[str], snrs: List[int], shifts: List[int], include_clean: bool,
) -> List[NoiseCond]:
    conds: List[NoiseCond] = [CLEAN_COND] if include_clean else []
    for noise_set in noise_sets:
        for snr in snrs:
            for shift in shifts:
                conds.append((noise_set, snr, shift))
    return conds


def noise_tag(cond: NoiseCond) -> str:
    """Noise condition tag used in filenames and logs."""
    noise_set, snr, shift = cond
    if not noise_set:
        return "clean"
    return f"{noise_set}_snr{snr}_sh{shift}"


# ──────────────────────────────── noise mixing ────────────────────────────────

_NOISE_BANK: dict = {}   # per-process (= per GPU worker) cache: noise_set -> [(path, audio), ...]


def load_noise_bank(noise_set: str) -> List[Tuple[str, np.ndarray]]:
    """Read the wavs of a noise set as 16k mono once and cache them."""
    if noise_set in _NOISE_BANK:
        return _NOISE_BANK[noise_set]

    noise_dir = os.path.join(NOISE_ROOT, noise_set)
    paths = sorted(
        os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.lower().endswith(".wav")
    )
    if not paths:
        raise FileNotFoundError(f"{noise_dir} 에 wav 파일이 없습니다.")
    bank = [(p, infer_mod.load_audio_16k(p)) for p in paths]
    _NOISE_BANK[noise_set] = bank
    print(f"  [noise] {noise_set}: {len(bank)} 개 background utterance 로드")
    return bank


def utterance_rng(*key_parts: str) -> random.Random:
    """A per-utterance reproducible RNG.
    Python's hash() is seeded differently in each process, so the seed is derived from md5.
    snr/shift are deliberately left out of the key, so a given utterance uses the same
    background under every noise condition."""
    key = "|".join((str(NOISE_SEED), *key_parts))
    seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def fit_to_length(bg: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    """Fit the background to the main length n: crop (from a random offset if longer) or
    tile (repeat if shorter)."""
    if len(bg) >= n:
        start = rng.randrange(0, len(bg) - n + 1)
        return bg[start:start + n]
    reps = math.ceil(n / len(bg))
    return np.tile(bg, reps)[:n]


def mix_with_snr(main: np.ndarray, bg: np.ndarray, snr_db: int, shift_pct: int) -> np.ndarray:
    """Mix main and background at the given SNR.
    shift_pct is where the background starts (as a % of the main length); at 100% it does not
    overlap at all but follows the main utterance, so the output is twice the main length."""
    n = len(main)
    offset = int(round(n * shift_pct / 100.0))

    p_main = float(np.mean(main.astype(np.float64) ** 2))
    p_bg = float(np.mean(bg.astype(np.float64) ** 2))
    if p_main <= 0 or p_bg <= 0:   # silence: nothing to mix
        return main.astype(np.float32)
    scale = math.sqrt(p_main / (p_bg * 10 ** (snr_db / 10.0)))

    mixed = np.zeros(max(n, offset + len(bg)), dtype=np.float32)
    mixed[:n] += main
    mixed[offset:offset + len(bg)] += (bg * scale).astype(np.float32)

    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:   # scaling everything by the same factor leaves the SNR unchanged
        mixed /= peak
    return mixed


def apply_noise(
    audio: np.ndarray, cond: NoiseCond, *key_parts: str,
) -> Tuple[np.ndarray, str]:
    """Mix a background utterance into audio according to cond and return
    (mixed, path of the background used)."""
    noise_set, snr_db, shift_pct = cond
    if not noise_set:
        return audio, ""
    bank = load_noise_bank(noise_set)
    rng = utterance_rng(noise_set, *key_parts)
    bg_path, bg_audio = bank[rng.randrange(len(bank))]
    bg = fit_to_length(bg_audio, len(audio), rng)
    return mix_with_snr(audio, bg, snr_db, shift_pct), bg_path


def dump_mix_sample(mixed: np.ndarray, cond: NoiseCond, speaker: str, domain: str, name: str) -> None:
    """Save one sample as wav so the mixing can be checked by ear."""
    out_dir = os.path.join(MIX_DUMP_ROOT, speaker)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{domain}_{noise_tag(cond)}_{name}")
    sf.write(out_path, mixed, infer_mod.TARGET_SR)
    print(f"  [dump] {out_path}")


# ──────────────────────────────── Experiment run ────────────────────────────────


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
    cond: NoiseCond,
) -> Optional[str]:
    """Run one combination (speaker/domain/noise/adaptation/ref_text) in batches of BATCH_SIZE
    and return the path of the result jsonl.

    If any audio file is missing, the whole combination is skipped (returns None).
    If save_jsonl already exists, ASR inference is skipped and metrics are recomputed from the
    existing results and appended to the summary."""
    noise_set, snr_db, shift_pct = cond
    save_jsonl_tag = f"_{entity_id}" if asr_adaptation else ""
    save_jsonl = (
        f"{RESULT_ROOT}/{speaker}/"
        f"{domain}_asr{'_ada' if asr_adaptation else ''}{save_jsonl_tag}_{noise_tag(cond)}.jsonl"
    )

    if os.path.exists(save_jsonl):
        print(f"  [skip] {save_jsonl} 이미 존재, ASR 추론 skip 후 summary 만 갱신")
        results = infer_mod.load_jsonl(save_jsonl)
    else:
        results = []
        dumped = False
        pbar = tqdm(total=len(data), desc=f"{speaker}/{domain}/{noise_tag(cond)}", leave=False)
        for batch_rows in infer_mod.chunked(data, BATCH_SIZE):
            batch_audio_paths = [
                f"L2-KPNS/data/audio/{speaker}/" + row["audio_path"] for row in batch_rows
            ]
            missing = next((p for p in batch_audio_paths if not os.path.exists(p)), None)
            if missing is not None:
                pbar.close()
                print(f"  [error] {missing} 없음, 이 조합({speaker}/{domain}) 전체 skip")
                return None

            batch_audio, batch_bg_path = [], []
            for row, path in zip(batch_rows, batch_audio_paths):
                # The background is sampled deterministically from (speaker, domain, audio_path),
                # so the same utterance keeps the same background across SNR/shift/ref_text.
                mixed, bg_path = apply_noise(
                    infer_mod.load_audio_16k(path), cond, speaker, domain, row["audio_path"],
                )
                if DUMP_MIX and not dumped and bg_path:
                    dump_mix_sample(mixed, cond, speaker, domain, os.path.basename(row["audio_path"]))
                    dumped = True
                batch_audio.append(mixed)
                batch_bg_path.append(bg_path)

            batch_out = asr.run_asr_batch(
                audios=batch_audio, ref_audio=asr_ref_audio, reference_text=ref_text,
            )

            for row, out, bg_path in zip(batch_rows, batch_out, batch_bg_path):
                entity_list_text = [e[0] for e in infer_mod.retrieve_top_k(
                    utterance=out,
                    entity_ipa=entity_dict_ipa,
                    k=10,
                )]

                result = dict(row)
                result["asr_result"] = out
                result["retr_entities"] = entity_list_text
                result["noise_set"] = noise_set
                result["snr_db"] = snr_db
                result["shift_pct"] = shift_pct
                result["noise_audio_path"] = bg_path
                results.append(result)
            pbar.update(len(batch_rows))
        pbar.close()

        os.makedirs(os.path.dirname(save_jsonl) or ".", exist_ok=True)
        with open(save_jsonl, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = compute_metrics.compute_metrics(results)
    write_summary_row(
        speaker, domain, cond, asr_adaptation, ref_aud_tag, ref_text, len(results), metrics,
    )
    return save_jsonl


def write_summary_row(
    speaker, domain, cond, asr_adaptation, ref_aud_tag, ref_text, num_rows, metrics,
):
    noise_set, snr_db, shift_pct = cond
    row = {
        "speaker": speaker,
        "domain": domain,
        "noise_set": noise_set or "clean",
        "snr_db": "" if snr_db is None else snr_db,
        "shift_pct": "" if shift_pct is None else shift_pct,
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


AVG_KEY_FIELDS = ["domain", "noise_set", "snr_db", "shift_pct", "asr_adaptation", "ref_aud_tag", "ref_text"]


def append_speaker_average_rows() -> None:
    """Compute a speaker-average row per (domain, noise condition, asr_adaptation, ref_aud_tag,
    ref_text) and append it to the summary csv."""
    with open(SUMMARY_CSV, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["speaker"] != "avg"]  # do not feed averages back into the average

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in AVG_KEY_FIELDS)
        groups.setdefault(key, []).append(row)

    avg_rows = []
    for key in sorted(groups):
        group_rows = groups[key]
        n = len(group_rows)
        avg_row = {"speaker": "avg", **dict(zip(AVG_KEY_FIELDS, key))}
        avg_row.update({
            "num_rows": sum(int(r["num_rows"]) for r in group_rows),
            "WER": round(sum(float(r["WER"]) for r in group_rows) / n, 2),
            "PED": round(sum(float(r["PED"]) for r in group_rows) / n, 2),
            "ASR-Hit": round(sum(float(r["ASR-Hit"]) for r in group_rows) / n, 2),
            "Hit@1": round(sum(float(r["Hit@1"]) for r in group_rows) / n, 2),
        })
        avg_rows.append(avg_row)

    with open(SUMMARY_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writerows(avg_rows)


def convert_summary_csv_to_xlsx() -> None:
    """After every process has finished, convert the final summary csv to xlsx."""
    wb = openpyxl.Workbook()
    ws = wb.active
    with open(SUMMARY_CSV, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            ws.append(row)
    wb.save(SUMMARY_XLSX)


def gpu_worker(gpu_id: int, speaker_queue: "mp.Queue[str]", conds: List[NoiseCond], dump_mix: bool) -> None:
    """One worker per GPU. Speakers are pulled off the queue one at a time and run sequentially,
    so even with more speakers than GPUs each GPU keeps exactly one process running."""
    global DUMP_MIX
    DUMP_MIX = dump_mix   # globals are not inherited by spawn-started children, so set it again here
    asr = infer_mod.Qwen25OmniASRPipeline(infer_mod.ASRConfig(device=f"cuda:{gpu_id}"))
    ref_text_to_entity_id = load_ref_text_to_entity_id(WAKEWORD_CSV)

    while True:
        try:
            speaker = speaker_queue.get_nowait()
        except queue.Empty:
            break
        run_for_speaker(asr, ref_text_to_entity_id, speaker, gpu_id, conds)


def run_for_speaker(
    asr: "infer_mod.Qwen25OmniASRPipeline",
    ref_text_to_entity_id: dict,
    speaker: str,
    gpu_id: int,
    conds: List[NoiseCond],
) -> None:
    """Run every domain x noise condition x ref_text combination for a single speaker, sequentially."""
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

        for cond in conds:
            print(f"[GPU {gpu_id}] === noise={noise_tag(cond)} ===")

            print(f"[GPU {gpu_id}] -- asr_adaptation=off --")
            run_one_config(
                asr, speaker, domain, data, entity_dict_ipa,
                asr_ref_audio=None, entity_id="", ref_aud_tag="", ref_text="",
                asr_adaptation=False, cond=cond,
            )

            for ref_text in REF_TEXTS:
                if ref_text not in ref_audio_by_text:
                    print(f"[GPU {gpu_id}] -- skip asr_adaptation=on ref_text='{ref_text}' (ref audio 없음) --")
                    continue
                print(f"[GPU {gpu_id}] -- asr_adaptation=on ref_text='{ref_text}' --")
                entity_id = ref_text_to_entity_id[ref_text]
                # The 1-shot reference audio is corrupted with the same condition only when
                # NOISE_ON_REF_AUDIO is True.
                ref_audio = ref_audio_by_text[ref_text]
                if NOISE_ON_REF_AUDIO:
                    ref_audio, _ = apply_noise(ref_audio, cond, speaker, "ref", entity_id)
                run_one_config(
                    asr, speaker, domain, data, entity_dict_ipa,
                    asr_ref_audio=ref_audio,
                    entity_id=entity_id,
                    ref_aud_tag=entity_id_suffix(entity_id),
                    ref_text=ref_text, asr_adaptation=True, cond=cond,
                )


def finalize_summary() -> None:
    merge_speaker_summaries()
    append_speaker_average_rows()
    convert_summary_csv_to_xlsx()
    print(f"Saved summary to {SUMMARY_XLSX}")


def parse_int_list(value: Optional[str], default: List[int]) -> List[int]:
    return [int(v) for v in value.split(",") if v.strip()] if value else default


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers", help="Comma-separated speaker list. Defaults to all of SPEAKERS. "
                                           "When splitting the run across jobs, make sure the lists do not overlap.")
    parser.add_argument("--noise-sets", help=f"Comma-separated noise sets. Defaults to {NOISE_SETS}")
    parser.add_argument("--snrs", help=f"Comma-separated SNRs (dB). Defaults to {SNRS_DB}")
    parser.add_argument("--shifts", help=f"Comma-separated shifts (%%). Defaults to {SHIFTS_PCT}")
    parser.add_argument("--include-clean", action="store_true",
                        help="Also run the noise-free baseline condition.")
    parser.add_argument("--dump-mix", action="store_true",
                        help="Save the first mixture per combination as wav so the mixing can be checked by ear.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip inference and just merge the per-speaker summaries into the summary csv/xlsx.")
    args = parser.parse_args()

    if args.merge_only:
        finalize_summary()
        sys.exit(0)

    noise_sets = [s.strip() for s in args.noise_sets.split(",")] if args.noise_sets else NOISE_SETS
    conds = build_noise_conditions(
        noise_sets,
        parse_int_list(args.snrs, SNRS_DB),
        parse_int_list(args.shifts, SHIFTS_PCT),
        args.include_clean or INCLUDE_CLEAN,
    )
    DUMP_MIX = args.dump_mix

    mp.set_start_method("spawn", force=True)

    speakers = [s.strip() for s in args.speakers.split(",")] if args.speakers else SPEAKERS
    num_gpus = infer_mod.torch.cuda.device_count() or 1
    print(f"Detected {num_gpus} GPU(s); processing {len(speakers)} speaker(s) {speakers} "
          f"x {len(DOMAINS)} domain(s) x {len(conds)} noise cond(s) "
          f"x (1 + {len(REF_TEXTS)}) ref_text(s) via a queue.")
    print("noise conditions: " + ", ".join(noise_tag(c) for c in conds))

    speaker_queue: "mp.Queue[str]" = mp.Queue()
    for speaker in speakers:
        speaker_queue.put(speaker)

    procs = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=gpu_worker, args=(gpu_id, speaker_queue, conds, args.dump_mix),
                       name=f"gpu-{gpu_id}")
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    if args.speakers:
        # Another job may still be running the remaining speakers, so do not merge here.
        print("--speakers 로 일부만 실행했습니다. 모든 잡이 끝난 뒤 "
              "`python run_all_noise_experiments.py --merge-only` 를 실행하세요.")
    else:
        finalize_summary()
