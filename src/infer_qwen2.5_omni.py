"""
# ASR (zero-shot) + phonetic edit distance retrieval
python infer_qwen2.5_omni.py \
    --speaker P001 \
    --domain roads

# ASR adaptation (1-shot) + phonetic edit distance retrieval
python infer_qwen2.5_omni.py \
    --speaker P001 \
    --domain roads \
    --ref_aud RD16245_P001.wav \
    --ref_text "상곡안길" \
    --asr_adaptation
"""

from __future__ import annotations

import argparse
import io
import json
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
import torch
import epitran
import panphon.distance
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.utils import logging as hf_logging
from qwen_omni_utils import process_mm_info

hf_logging.set_verbosity_error()

epi = epitran.Epitran("kor-Hang")
dst = panphon.distance.Distance()

TARGET_SR = 16000
TRANSCRIPTION_PREFIX = "Transcription: "


def normalized_phoneme_editdistance(dist: float, utterance_ipa: str, entity_ipa: str) -> float:
    return 1 - (dist / max(len(utterance_ipa), len(entity_ipa)))


def retrieve_top_k(
    utterance: str,
    entity_ipa: Dict[str, str],
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Retrieve the top-k entities by phoneme-level edit distance against the ASR result."""
    results = []
    utterance_ipa = epi.transliterate(utterance)
    for entity, ipa in entity_ipa.items():
        dist = dst.levenshtein_distance(utterance_ipa, ipa)
        score = round(normalized_phoneme_editdistance(dist, utterance_ipa, ipa), 2)
        results.append((entity, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


def load_audio_16k(path: str) -> np.ndarray:
    """Read a wav and normalize it to mono float32 at 16 kHz."""
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


@dataclass
class ASRConfig:
    model_path: str = "Qwen2.5-Omni-7B"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    language: str = "Korean"
    max_new_tokens: int = 256
    use_audio_in_video: bool = False


SYSTEM_TEXT = "You are Qwen, a virtual human developed by the Qwen Team."


class Qwen25OmniASRPipeline:
    """ASR using Qwen2.5-Omni."""

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.config.model_path,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device,
            attn_implementation="sdpa",
        )
        self.model.eval()
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.config.model_path)

        print("===== CUDA CHECK IN SCRIPT =====")
        print("torch version:", torch.__version__)
        print("torch cuda:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        print("device count:", torch.cuda.device_count())
        print("config device:", self.config.device)
        print("config dtype:", self.config.torch_dtype)
        print("================================")

    @staticmethod
    def _normalize_audio_input(audio):
        if isinstance(audio, io.BytesIO):
            audio.seek(0)
            wav, sr = sf.read(audio)
            return wav, sr
        return audio

    def _build_messages(self, audio, ref_audio=None, reference_text: str = "") -> list:
        """Build the transcription messages: zero-shot when ref_audio is None,
        1-shot (asr_adaptation) otherwise."""
        if ref_audio is not None:
            intro = (
                "I'll provide an example of speech from a non-native Korean speaker, "
                "followed by the correct transcription. Then I'll give you a new audio "
                "from the same speaker to transcribe into Korean."
            )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
                {"role": "user", "content": [{"type": "text", "text": intro}]},
                {"role": "assistant", "content": [{"type": "text", "text":
                    "I understand. I'll listen to the example and use it to accurately "
                    "transcribe the final audio into Korean."}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": ref_audio},
                    {"type": "text", "text": "Transcribe this audio:"},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text":
                    f"{TRANSCRIPTION_PREFIX}{reference_text}"}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text":
                        "Please transcribe this audio from the same speaker into Korean:"},
                ]},
            ]
        else:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text":
                        "Transcribe the given audio file into Korean.\n"
                        "Output only the transcription without any additional explanation."},
                ]},
            ]
        return messages

    @torch.inference_mode()
    def _transcribe_from_messages(self, messages: list, max_new_tokens: int) -> str:
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = process_mm_info(
            messages,
            use_audio_in_video=self.config.use_audio_in_video,
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            sampling_rate=TARGET_SR,
            use_audio_in_video=self.config.use_audio_in_video,
        )
        inputs = inputs.to(self.model.device).to(self.config.torch_dtype)

        generated_ids = self.model.generate(
            **inputs,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_audio_in_video=self.config.use_audio_in_video,
            return_audio=False,
        )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return output

    def run_asr(self, audio, ref_audio=None, reference_text: str = "") -> str:
        """자유 전사. ref_audio 제공 시 asr_adaptation(1-shot) 적용."""
        audio = self._normalize_audio_input(audio)
        if ref_audio is not None:
            ref_audio = self._normalize_audio_input(ref_audio)
        messages = self._build_messages(audio, ref_audio=ref_audio, reference_text=reference_text)
        result = self._transcribe_from_messages(messages, self.config.max_new_tokens)
        if result.startswith(TRANSCRIPTION_PREFIX.strip()):
            result = result[len(TRANSCRIPTION_PREFIX.strip()):].lstrip(": ").strip()
        if reference_text and result.startswith(reference_text):
            result = result[len(reference_text):].strip()
        return result


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ref_aud_suffix(ref_aud: str, speaker: str) -> str:
    """Strip the speaker suffix from the ref_aud filename and use the last "_"-separated
    token as the tag (e.g. pr0081_pr_P001.wav -> pr, RD16245_P001.wav -> RD16245)."""
    base = os.path.splitext(ref_aud)[0].removesuffix(f"_{speaker}")
    return base.rsplit("_", 1)[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True,
                        choices=["roads", "content", "restaurants", "stations"])
    parser.add_argument("--ref_aud", type=str, default=None,
                        help="1-shot reference audio filename (relative to L2-KPNS/data/audio/{speaker}/)")
    parser.add_argument("--ref_text", type=str, default="", help="1-shot reference text")
    parser.add_argument("--asr_adaptation", action="store_true",
                        help="Apply the 1-shot reference to ASR")
    args = parser.parse_args()

    if args.asr_adaptation and not args.ref_aud:
        parser.error("--ref_aud is required when --asr_adaptation is used.")

    data_jsonl = f"L2-KPNS-jsonl/{args.speaker}/{args.domain}_{args.speaker}.jsonl"
    ref_aud_tag = "_" + ref_aud_suffix(args.ref_aud, args.speaker) if args.asr_adaptation else ""
    ref_text_tag = "_" + "".join(args.ref_text.split()) if args.asr_adaptation else ""
    save_jsonl = f"L2-KPNS-jsonl/results/{args.speaker}/{args.domain}_asr{'_ada' if args.asr_adaptation else ''}{ref_aud_tag}{ref_text_tag}.jsonl"

    asr = Qwen25OmniASRPipeline()
    data = load_jsonl(data_jsonl)
    print(f"Loaded {len(data)} rows from {data_jsonl}")

    lexicon_path = f"L2-KPNS/metadata/recording_targets/{args.domain}_200.csv"
    with open(lexicon_path, encoding="utf-8-sig") as f:
        entity_dict = {row["entity_id"]: row["korean"] for row in csv.DictReader(f)}
    entity_names = list(dict.fromkeys(entity_dict.values()))  # dedupe, preserve order
    entity_dict_ipa = {e: epi.transliterate(e) for e in entity_names}
    print(f"Loaded {len(entity_names)} candidate entities from lexicon.")

    ref_aud_path = f"L2-KPNS/data/audio/{args.speaker}/{args.ref_aud}" if args.ref_aud else None
    ref_audio = load_audio_16k(ref_aud_path) if ref_aud_path else None
    asr_ref_audio = ref_audio if args.asr_adaptation else None

    results = []
    for i, row in enumerate(data):
        audio_path = f"L2-KPNS/data/audio/{args.speaker}/" + row["audio_path"]
        test_audio = load_audio_16k(audio_path)

        out = asr.run_asr(audio=test_audio, ref_audio=asr_ref_audio, reference_text=args.ref_text)

        entity_list_text = [e[0] for e in retrieve_top_k(
            utterance=out,
            entity_ipa=entity_dict_ipa,
            k=10,
        )]

        result = dict(row)
        result["asr_result"] = out
        result["retr_entities"] = entity_list_text
        results.append(result)
        print(f"[{i + 1}/{len(data)}] asr_result='{out}' -> retr_entities={entity_list_text}")

    os.makedirs(os.path.dirname(save_jsonl) or ".", exist_ok=True)
    with open(save_jsonl, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved results to {save_jsonl}")
