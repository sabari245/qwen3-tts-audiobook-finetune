#!/usr/bin/env python
"""Narrate a text file with the fine-tuned narrator voice (Qwen3-TTS 1.7B).

    python narrate.py input.txt                      # split into chunks, batched, -> input.wav
    python narrate.py input.txt --limit 20           # quick speed test on the first 20 chunks
    python narrate.py input.txt --mode full          # whole text in one generation (risky beyond a few minutes)

Split mode cuts the cleaned text into sentence-aligned chunks of at most --max-chars (the model was
trained on 3-18 s clips, so ~300 chars keeps it in-distribution), generates them in length-sorted
batches, re-generates chunks whose duration is implausible for the narrator's pace, and joins
them with short pauses (longer between paragraphs).
"""
import argparse
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

HERE = Path(__file__).parent
DEFAULT_MODEL = HERE / "workdir/models/narrator"
NARRATOR_WPS = 2.42  # measured on the training narration
GEN = dict(do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.05)

# ---------------------------------------------------------------- text cleanup

CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "«": '"', "»": '"',
    "…": "...", " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "‌": "", "‍": "", "﻿": "", "­": "",
    "‐": "-", "‑": "-", "−": "-",
    "&": " and ", "%": " percent", "+": " plus ", "=": " equals ", "@": " at ",
    "×": " times ", "°": " degrees", "€": " euros", "£": " pounds",
}


def clean_text(raw: str) -> list[str]:
    """Return cleaned paragraphs."""
    t = unicodedata.normalize("NFKC", raw)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    for k, v in CHAR_MAP.items():
        t = t.replace(k, v)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)                    # URLs
    t = re.sub(r"<[^>]+>", " ", t)                                   # HTML tags
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)                      # markdown images
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)                   # markdown links -> text
    t = re.sub(r"^\s{0,3}(#{1,6}|>+|[-*+]\s|\d+[.)]\s)\s*", "", t, flags=re.M)  # headings, quotes, bullets
    t = re.sub(r"^\s*([-*_=~]\s*){3,}$", "", t, flags=re.M)          # horizontal rules / scene breaks
    t = re.sub(r"(\*{1,3}|_{1,3}|~~)(\S.*?\S|\S)\1", r"\2", t)       # *emphasis*, __bold__, ~~strike~~
    t = re.sub(r"\s*[–—―]+\s*", ", ", t)              # en/em dashes -> spoken pause
    t = re.sub(r"\s+-{2,}\s*|\s*-{2,}\s+", ", ", t)                  # ascii -- dashes
    t = re.sub(r"\b0+(\d)", r"\1", t)                                # "Chapter 060" -> "Chapter 60"
    t = re.sub(r"\.{4,}", "...", t)
    t = re.sub(r"([!?]){2,}", r"\1", t)
    t = re.sub(r"[\[\]{}<>|\\^*_#~]", " ", t)                        # leftover markup symbols
    # drop emoji / symbols / control chars, keep letters, digits, punctuation and spaces
    t = "".join(ch for ch in t if ch == "\n" or unicodedata.category(ch)[0] in "LNPZ" or ch in "$'\"")
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r",\s*([,.;:!?])", r"\1", t)                          # ", ." from dash replacement
    t = re.sub(r"^[,;:\s]+", "", t, flags=re.M)
    t = re.sub(r"[ \t]+", " ", t)
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", t)]
    out = []
    for p in paras:
        if not re.search(r"[A-Za-z0-9]", p):
            continue
        if not re.search(r"[.!?\"')]$", p):
            p += "."                                                  # headings / titles get a full stop
        out.append(p)
    return out


def split_paragraph(p: str, max_chars: int) -> list[str]:
    # sentence = shortest run ending in .!? (plus closing quotes) that is followed by a capitalised start
    sents = [s.strip() for s in re.findall(r".+?(?:[.!?]+[\"')]*(?=\s+[\"'(]?[A-Z0-9])|$)", p) if s.strip()]
    pieces = []
    for s in sents:
        while len(s) > max_chars:  # very long sentence: break at the last comma/semicolon before the limit
            cut = max(s.rfind(c, 0, max_chars) for c in (", ", "; ", ": "))
            cut = cut + 1 if cut > max_chars // 3 else s.rfind(" ", 0, max_chars)
            pieces.append(s[:cut].strip())
            s = s[cut:].strip()
        pieces.append(s)
    chunks, cur = [], ""
    for s in pieces:
        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- generation

def plausible(text: str, wav: np.ndarray, sr: int) -> bool:
    words = len(text.split())
    dur = len(wav) / sr
    expected = words / NARRATOR_WPS
    return dur > 0.3 and 0.55 * expected - 0.5 <= dur <= 1.7 * expected + 1.5


def generate_batch(tts, texts, speaker, max_new_tokens):
    wavs, sr = tts.generate_custom_voice(text=texts, speaker=speaker, language="English",
                                         max_new_tokens=max_new_tokens, **GEN)
    return [np.asarray(w, dtype=np.float32) for w in wavs], sr


def run_split(tts, chunks, speaker, batch_size, retries, pbar):
    """chunks: list of (paragraph_idx, text). Returns list of wavs in input order."""
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i][1]), reverse=True)
    results: dict[int, np.ndarray] = {}
    best: dict[int, np.ndarray] = {}
    sr = 24000
    pending = order
    for attempt in range(retries + 1):
        failed = []
        i = 0
        while i < len(pending):
            idx = pending[i:i + batch_size]
            texts = [chunks[k][1] for k in idx]
            longest_words = max(len(t.split()) for t in texts)
            max_new = int(longest_words / NARRATOR_WPS * 12.5 * 2.0) + 60  # 12.5 codec frames/s, 2x headroom
            try:
                wavs, sr = generate_batch(tts, texts, speaker, max_new)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                tqdm.write(f"CUDA OOM -> batch size {batch_size}")
                continue
            for k, w in zip(idx, wavs):
                expected = len(chunks[k][1].split()) / NARRATOR_WPS
                if k not in best or abs(len(w) / sr - expected) < abs(len(best[k]) / sr - expected):
                    best[k] = w
                if plausible(chunks[k][1], w, sr) or attempt == retries:
                    results[k] = best[k]
                    pbar.update(1)
                else:
                    failed.append(k)
            i += len(idx)
        if not failed:
            break
        tqdm.write(f"retrying {len(failed)} chunk(s) with implausible duration (attempt {attempt + 2})")
        pending = failed
    return [results[k] for k in range(len(chunks))], sr, batch_size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="output wav (default: <input>.wav)")
    ap.add_argument("--mode", choices=["split", "full"], default="split")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--speaker", default="narrator")
    ap.add_argument("--batch-size", type=int, default=8, help="chunks generated in parallel (halved automatically on OOM)")
    ap.add_argument("--max-chars", type=int, default=300, help="max characters per chunk in split mode")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, help="only narrate the first N chunks (speed test)")
    ap.add_argument("--sentence-pause", type=float, default=0.25)
    ap.add_argument("--paragraph-pause", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-clean", action="store_true", help="also write the cleaned text next to the output")
    args = ap.parse_args()

    out = args.output or args.input.with_suffix(".wav")
    paras = clean_text(args.input.read_text(encoding="utf-8", errors="replace"))
    if args.save_clean:
        out.with_suffix(".clean.txt").write_text("\n\n".join(paras) + "\n", encoding="utf-8")

    from qwen_tts import Qwen3TTSModel
    torch.manual_seed(args.seed)
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(str(args.model), device_map="cuda:0", dtype=torch.bfloat16,
                                        attn_implementation="sdpa")
    print(f"model loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    if args.mode == "full":
        text = " ".join(paras)
        words = len(text.split())
        print(f"full mode: {words} words (~{words / NARRATOR_WPS / 60:.1f} min of audio) in a single generation. "
              "The model was trained on <=18 s clips; very long single generations may drift, skip or stop early.")
        max_new = int(words / NARRATOR_WPS * 12.5 * 1.5) + 100
        with tqdm(total=1, desc="generating", unit="text") as pbar:
            wavs, sr = generate_batch(tts, [text], args.speaker, max_new)
            pbar.update(1)
        audio = wavs[0]
    else:
        chunks = [(pi, c) for pi, p in enumerate(paras) for c in split_paragraph(p, args.max_chars)]
        if args.limit:
            chunks = chunks[:args.limit]
        words = sum(len(c.split()) for _, c in chunks)
        print(f"{len(paras)} paragraphs -> {len(chunks)} chunks, {words} words "
              f"(~{words / NARRATOR_WPS / 60:.1f} min of audio), batch size {args.batch_size}")
        with tqdm(total=len(chunks), desc="narrating", unit="chunk") as pbar:
            wavs, sr, bs = run_split(tts, chunks, args.speaker, args.batch_size, args.retries, pbar)
        pieces = []
        for n, ((pi, _), w) in enumerate(zip(chunks, wavs)):
            pieces.append(w)
            if n + 1 < len(chunks):
                gap = args.paragraph_pause if chunks[n + 1][0] != pi else args.sentence_pause
                pieces.append(np.zeros(int(gap * sr), dtype=np.float32))
        audio = np.concatenate(pieces)

    peak = np.abs(audio).max()
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    sf.write(out, audio, sr)
    gen_s, audio_s = time.time() - t0, len(audio) / sr
    print(f"wrote {out}  audio {audio_s / 60:.1f} min  generated in {gen_s / 60:.1f} min  "
          f"(real-time factor {gen_s / audio_s:.2f}, {audio_s / gen_s:.1f}x real time)")


if __name__ == "__main__":
    main()
