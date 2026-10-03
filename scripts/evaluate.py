"""Step 4: compare checkpoints against the zero-shot base model and the real narrator.

    python scripts/evaluate.py --checkpoints workdir/runs/run1/checkpoint-epoch-{1,2,3}

For each system: held-out sentences (never trained on) + a long passage generated several times.
Reports word error rate (Whisper large-v3), speaker similarity to real held-out clips (Qwen3-TTS
speaker encoder, cosine) and speaking rate in words/s next to the real narrator's rate. All audio
is written to --out so you can listen; <system>_val0.wav pairs with the real clip printed at the end.
Pick the checkpoint with low long-passage worst-case WER and a pace close to the narrator.
"""
import argparse
import json
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch

import train as T

LONG_PASSAGE = (
    "When the storm finally broke over Ithaca, Penelope stood alone at the high window of the palace. "
    "Twenty years had passed since her husband sailed for Troy, and in all that time the sea had given her nothing "
    "but rumors. Some said Odysseus had drowned off the coast of Thrace. Others swore they had seen him feasting in "
    "the halls of a sorceress, his ships long rotted on a distant shore. But Penelope did not believe them. Every "
    "night she unraveled the shroud she wove by day, and every morning she told the suitors the same thing: not yet. "
    "Below her, in the great hall, their laughter rose like smoke. They drank her husband's wine and slaughtered his "
    "cattle, and they waited for her to choose. And far away, on an island at the edge of the world, a beggar in rags "
    "opened his eyes, and remembered his name."
)
GEN = dict(do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.05, max_new_tokens=4096)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("workdir/prepared"))
    ap.add_argument("--base-model", default=T.BASE_MODEL, help="zero-shot voice-clone baseline ('' to skip)")
    ap.add_argument("--out", type=Path, default=Path("workdir/eval/latest"))
    ap.add_argument("--n-val", type=int, default=15)
    ap.add_argument("--n-long", type=int, default=3)
    ap.add_argument("--long-text", type=Path, help="text file to use instead of the built-in passage")
    ap.add_argument("--speaker", default="narrator")
    ap.add_argument("--language", default="English")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    from transformers import pipeline
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    args.out.mkdir(parents=True, exist_ok=True)
    long_text = args.long_text.read_text().strip() if args.long_text else LONG_PASSAGE
    val = T.load_jsonl(args.data / "val_with_codes.jsonl")
    val_texts = [v["text"] for v in val[:args.n_val]]
    real_clips = [v["audio"] for v in val[args.n_val:args.n_val + 20]] or [v["audio"] for v in val]
    norm = BasicTextNormalizer()
    load = lambda p: Qwen3TTSModel.from_pretrained(str(p), device_map=args.device, dtype=torch.bfloat16, attn_implementation="sdpa")

    def generate_all(fn):
        res = [("val", t, *fn(t)) for t in val_texts]
        res += [("long", long_text, *fn(long_text)) for _ in range(args.n_long)]
        return [(k, t, w[0], sr) for k, t, w, sr in res]

    outputs = {}
    ref_wav, ref_txt = str(args.data / "ref.wav"), (args.data / "ref.txt").read_text()
    if args.base_model:
        base = load(T.resolve_model(args.base_model))
        outputs["base"] = generate_all(lambda t: base.generate_voice_clone(
            text=t, language=args.language, ref_audio=ref_wav, ref_text=ref_txt, **GEN))
        print("generated base", flush=True)
        del base
        torch.cuda.empty_cache()
    for ck in args.checkpoints:
        m = load(ck)
        name = f"{ck.parent.name}-{ck.name.replace('checkpoint-', '')}"
        outputs[name] = generate_all(lambda t: m.generate_custom_voice(
            text=t, speaker=args.speaker, language=args.language, **GEN))
        print("generated", name, flush=True)
        del m
        torch.cuda.empty_cache()

    # speaker encoder from the base checkpoint family (same for every system)
    enc_model = load(args.checkpoints[0])
    enc = enc_model.model.speaker_encoder

    def spk_emb(wav, sr):
        wav = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=24000)
        mel = mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000,
                              hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
        with torch.no_grad():
            e = enc(mel.to(args.device, torch.bfloat16)).float()[0]
        return e / e.norm()

    real = torch.stack([spk_emb(*sf.read(p, dtype="float32")) for p in real_clips]).mean(0)
    real = real / real.norm()
    del enc_model, enc
    torch.cuda.empty_cache()

    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device=args.device)
    report = {}
    for name, res in outputs.items():
        rows = []
        for i, (kind, text, w, sr) in enumerate(res):
            sf.write(args.out / f"{name}_{kind}{i}.wav", w, sr)
            hyp = asr({"raw": librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000), "sampling_rate": 16000},
                      return_timestamps=True, generate_kwargs={"language": args.language.lower(), "task": "transcribe"})["text"]
            rows.append({"kind": kind, "wer": jiwer.wer(norm(text), norm(hyp)), "sim": float(spk_emb(w, sr) @ real),
                         "wps": len(text.split()) / (len(w) / sr)})
        agg = {}
        for kind in ("val", "long"):
            r = [x for x in rows if x["kind"] == kind]
            agg[kind] = {k: round(float(np.mean([x[k] for x in r])), 3) for k in ("wer", "sim", "wps")}
            agg[kind]["worst_wer"] = round(max(x["wer"] for x in r), 3)
        report[name] = agg
    real_wps = float(np.mean([len(v["text"].split()) / sf.info(v["audio"]).duration for v in val]))
    report["real_narrator_wps"] = round(real_wps, 3)
    json.dump(report, open(args.out / "report.json", "w"), indent=1)

    print(f"\n{'system':<24}{'val WER':>9}{'long WER':>10}{'worst':>8}{'sim':>8}{'wps val/long':>16}")
    for name, a in report.items():
        if name == "real_narrator_wps":
            continue
        print(f"{name:<24}{a['val']['wer']:>9.3f}{a['long']['wer']:>10.3f}{a['long']['worst_wer']:>8.3f}"
              f"{a['val']['sim']:>8.3f}{a['val']['wps']:>8.2f}/{a['long']['wps']:.2f}")
    print(f"real narrator pace: {real_wps:.2f} words/s")
    print(f"listen: {args.out}/<system>_val0.wav vs the real recording {val[0]['audio']}")


if __name__ == "__main__":
    main()
