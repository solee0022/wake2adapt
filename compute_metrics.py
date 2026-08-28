"""
Compute recall@ASR, recall@1/5/10 and WER over ASR results (jsonl).

python compute_metrics.py --jsonl_path L2-KPNS-jsonl/results/P001/roads_asr_ada.jsonl
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import epitran
import panphon.distance

epi = epitran.Epitran("kor-Hang")
dst = panphon.distance.Distance()


def normalized_phoneme_editdistance(dist: float, utterance_ipa: str, entity_ipa: str) -> float:
    return 1 - (dist / max(len(utterance_ipa), len(entity_ipa)))


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def word_edit_distance(ref_words: List[str], hyp_words: List[str]) -> int:
    """Minimum number of edits (S+D+I) needed to turn ref_words into hyp_words."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def compute_wer(rows: List[dict]) -> Tuple[float, int, int]:
    """corpus-level WER = sum(edit distance) / sum(reference word count)."""
    total_edits = 0
    total_ref_words = 0
    for row in rows:
        ref_words = row["answer"].split()
        hyp_words = row["asr_result"].split()
        total_edits += word_edit_distance(ref_words, hyp_words)
        total_ref_words += len(ref_words)
    wer = total_edits / total_ref_words if total_ref_words else 0.0
    return wer, total_edits, total_ref_words


def compute_recall_at_k(rows: List[dict], k: int) -> Tuple[float, int, int]:
    hits = sum(1 for row in rows if row["answer"] in row["retr_entities"][:k])
    return (hits / len(rows) if rows else 0.0), hits, len(rows)


def compute_recall_at_asr(rows: List[dict]) -> Tuple[float, int, int]:
    hits = sum(1 for row in rows if row["answer"] == row["asr_result"])
    return (hits / len(rows) if rows else 0.0), hits, len(rows)


def compute_phoneme_editdistance(rows: List[dict]) -> Tuple[float, float, int]:
    """Mean phoneme-level (IPA) normalized edit distance between answer and asr_result."""
    total_score = 0.0
    for row in rows:
        answer_ipa = epi.transliterate(row["answer"])
        asr_ipa = epi.transliterate(row["asr_result"])
        dist = dst.levenshtein_distance(asr_ipa, answer_ipa)
        total_score += normalized_phoneme_editdistance(dist, asr_ipa, answer_ipa)
    avg_score = total_score / len(rows) if rows else 0.0
    return avg_score, total_score, len(rows)


def compute_metrics(rows: List[dict]) -> Dict[str, Tuple[float, int, int]]:
    return {
        "recall@asr": compute_recall_at_asr(rows),
        "recall@1": compute_recall_at_k(rows, 1),
        "recall@5": compute_recall_at_k(rows, 5),
        "recall@10": compute_recall_at_k(rows, 10),
        "wer": compute_wer(rows),
        "phoneme_editdistance": compute_phoneme_editdistance(rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    args = parser.parse_args()

    data = load_jsonl(args.jsonl_path)
    print(f"Loaded {len(data)} rows from {args.jsonl_path}")

    metrics = compute_metrics(data)
    for name, (value, num, denom) in metrics.items():
        if name in ("wer", "phoneme_editdistance"):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value:.4f} ({num}/{denom})")
