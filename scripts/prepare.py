"""Step 2: re-transcribe clips with Whisper, filter, pick a reference clip, extract codec tokens.

    python scripts/prepare.py --clips workdir/clips --out workdir/prepared

- Whisper large-v3 text replaces the caption text (better punctuation, casing and proper nouns).
- Clips are dropped when caption and Whisper text disagree (WER > --wer-max, usually a bad cut)
  or when the quietest 5% of the clip is louder than --floor-max-db (background noise/music).
- One clean 8-12 s clip becomes the fixed reference for every sample (recommended upstream).
- Writes train_with_codes.jsonl / val_with_codes.jsonl for scripts/train.py. Paths inside the
  jsonl files are relative to --out, so the folder can be moved between machines.
"""
import argparse
import gc
import json
import os
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch

TOKENIZER_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
WHISPER_ID = "openai/whisper-large-v3"


def floor_db(a):
    n = len(a) // 240
    db = 20 * np.log10(np.sqrt((a[: n * 240].reshape(n, 240) ** 2).mean(1)) + 1e-9)
    return float(np.percentile(db, 5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, default=Path("workdir/clips"))
    ap.add_argument("--out", type=Path, default=Path("workdir/prepared"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--whisper-model", default=WHISPER_ID)
    ap.add_argument("--asr-batch-size", type=int, default=32, help="use 4 on a 12 GB GPU")
    ap.add_argument("--codec-batch-size", type=int, default=32, help="use 4 on a 12 GB GPU")
    ap.add_argument("--wer-max", type=float, default=0.25)
    ap.add_argument("--floor-max-db", type=float, default=-45.0)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--language", default="English")
    ap.add_argument("--limit", type=int, help="only use the first N clips (smoke test)")
    args = ap.parse_args()

    from transformers import pipeline
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.clips / "segments.jsonl")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} clips")

    # 1) Whisper re-transcription
    asr = pipeline("automatic-speech-recognition", model=args.whisper_model, torch_dtype=torch.float16, device=args.device)
    norm = BasicTextNormalizer()
    B = args.asr_batch_size
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        auds = []
        for r in chunk:
            a, sr = sf.read(args.clips / r["audio"], dtype="float32")
            r["floor_db"] = floor_db(a)
            auds.append(librosa.resample(a, orig_sr=sr, target_sr=16000))
        outs = asr(auds, batch_size=B, generate_kwargs={"language": args.language.lower(), "task": "transcribe"})
        for r, o in zip(chunk, outs):
            r["whisper"] = o["text"].strip()
            ref, hyp = norm(r["text"]), norm(r["whisper"])
            r["wer"] = jiwer.wer(ref, hyp) if ref.strip() else 1.0
        print(f"asr {min(i + B, len(rows))}/{len(rows)}", flush=True)
    del asr
    gc.collect()
    torch.cuda.empty_cache()

    kept = [r for r in rows if r["wer"] <= args.wer_max and r["floor_db"] <= args.floor_max_db and r["whisper"]]
    print(f"kept {len(kept)}/{len(rows)} ({sum(r['dur'] for r in kept) / 3600:.2f} h); "
          f"dropped wer>{args.wer_max}: {sum(r['wer'] > args.wer_max for r in rows)}, "
          f"floor>{args.floor_max_db} dB: {sum(r['floor_db'] > args.floor_max_db for r in rows)}")
    with open(args.out / "asr_all.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2) one fixed reference clip: clean, 8-12 s, best caption/Whisper agreement
    cands = sorted((r for r in kept if 8 <= r["dur"] <= 12), key=lambda r: (r["wer"], r["floor_db"]))
    ref = (cands or sorted(kept, key=lambda r: (r["wer"], r["floor_db"])))[0]
    a, sr = sf.read(args.clips / ref["audio"], dtype="float32")
    if sr != 24000:
        a, sr = librosa.resample(a, orig_sr=sr, target_sr=24000), 24000
    sf.write(args.out / "ref.wav", a, sr)
    (args.out / "ref.txt").write_text(ref["whisper"])
    print("reference clip:", ref["audio"], "|", ref["whisper"])

    # 3) train/val split with Whisper text; paths relative to --out
    rel_clips = os.path.relpath(args.clips.resolve(), args.out.resolve())
    rng = np.random.default_rng(0)
    val_ids = set(rng.permutation(len(kept))[:args.n_val].tolist())

    def item(r):
        return {"audio": f"{rel_clips}/{r['audio']}", "text": r["whisper"], "ref_audio": "ref.wav",
                "language": args.language}

    splits = {"train": [item(r) for k, r in enumerate(kept) if k not in val_ids],
              "val": [item(r) for k, r in enumerate(kept) if k in val_ids]}

    # 4) 12 Hz codec tokens
    from qwen_tts import Qwen3TTSTokenizer
    tok = Qwen3TTSTokenizer.from_pretrained(TOKENIZER_ID, device_map=args.device)
    for name, items in splits.items():
        for i in range(0, len(items), args.codec_batch_size):
            b = items[i:i + args.codec_batch_size]
            enc = tok.encode([str(args.out / x["audio"]) for x in b])
            for x, c in zip(b, enc.audio_codes):
                x["audio_codes"] = c.cpu().tolist()
        with open(args.out / f"{name}_with_codes.jsonl", "w") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"{name}: {len(items)} items -> {args.out / f'{name}_with_codes.jsonl'}")


if __name__ == "__main__":
    main()
