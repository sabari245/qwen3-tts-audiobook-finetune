"""Step 1: cut long-form narration into TTS training clips using word-timed WebVTT captions.

    python scripts/segment.py --raw-dir workdir/raw --out workdir/clips

Expects pairs `<name>.<mp3|wav|m4a|flac|opus|webm>` + `<name>*.vtt` in --raw-dir (YouTube
auto-captions, e.g. from `yt-dlp --write-auto-subs --sub-format vtt`). Writes 24 kHz mono FLAC
clips to <out>/flac/ and <out>/segments.jsonl ({"audio": "flac/x.flac", "text", "start", "dur"}).

- Words come from the timed cue lines (`word<ts><c> word</c>...`).
- Any bracketed event ([music], [laughter], [screaming], ...) is treated as non-narration;
  clips overlapping such a region (plus a margin) are dropped.
- Sentences are grouped into clips of MIN_DUR..MAX_DUR seconds; clip boundaries are placed
  at the lowest-energy point between the last word of one clip and the first of the next.
- Audio with no caption words is never used: it may hold music or effects without a tag. Clip
  edges stay within MAX_LEAD/MAX_TAIL of a captioned word and clips with a >MAX_GAP silence
  between words are dropped.
"""
import argparse
import json
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000
MIN_DUR, MAX_DUR, TARGET_DUR = 3.0, 18.0, 10.0
EVENT_MARGIN = 1.0
TS = r"(\d+):(\d+):(\d+)\.(\d+)"


def ts2s(s):
    h, m, sec, ms = re.match(TS, s).groups()
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def parse_vtt(path):
    words, events = [], []
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        if "-->" in lines[i]:
            a, b = lines[i].split("-->")
            start, end = ts2s(a.strip()), ts2s(b.strip().split()[0])
            body = []
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                body.append(lines[i])
                i += 1
            for ln in body:
                for ev in re.findall(r"\[[^\]]+\]", ln):
                    events.append((start, end, ev))
            # Rolling captions: 10 ms cues only repeat finished text; in real cues the last
            # line holds the new words (word-timed with <c>, or a single untimed word).
            if end - start > 0.02 and body:
                ln = body[-1]
                if ln.strip():
                    # first token has no explicit timestamp: it starts at the cue start
                    first, *rest = re.split(r"<(\d+:\d+:\d+\.\d+)>", ln)
                    toks = [(start, first)]
                    for k in range(0, len(rest), 2):
                        toks.append((ts2s(rest[k]), rest[k + 1]))
                    for t, w in toks:
                        w = re.sub(r"</?c>", "", w).strip()
                        if w and not w.startswith("["):
                            words.append([t, w])
        i += 1
    # dedupe events
    events = sorted(set(events))
    return words, events


def load_audio(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def frame_rms(audio, hop=240):  # 10 ms frames
    n = len(audio) // hop
    fr = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    k = np.ones(5) / 5  # 50 ms smoothing
    return np.convolve(rms, k, mode="same")


AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".opus", ".webm", ".ogg")


def find_pairs(raw_dir: Path):
    pairs = []
    for a in sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS):
        vtts = sorted(raw_dir.glob(f"{a.stem}*.vtt"))
        if vtts:
            pairs.append((a, vtts[0]))
        else:
            print(f"skip {a.name}: no matching .vtt")
    return pairs


def segment_one(audio_path: Path, vtt_path: Path, out_dir: Path):
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", audio_path.stem)
    (out_dir / "flac").mkdir(parents=True, exist_ok=True)
    (out_dir / "events").mkdir(parents=True, exist_ok=True)
    words, events = parse_vtt(vtt_path)
    audio = load_audio(audio_path)
    rms = frame_rms(audio)
    total = len(audio) / SR
    print(f"{stem}: {len(words)} words, {len(events)} events, {total/3600:.2f} h")

    # sentences: split after terminal punctuation
    sents, cur = [], []
    for t, w in words:
        cur.append((t, w))
        if re.search(r"[.!?][\"')\]]*$", w):
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)

    # group sentences into clips
    clips, cur = [], []
    for k, s in enumerate(sents):
        nxt_start = sents[k + 1][0][0] if k + 1 < len(sents) else total
        if cur:
            dur_if_add = nxt_start - cur[0][0][0]
            if dur_if_add > MAX_DUR or (s[0][0] - cur[0][0][0]) >= TARGET_DUR:
                clips.append(cur)
                cur = []
        cur.append(s)
    if cur:
        clips.append(cur)

    def quietest(lo_t, hi_t):
        lo, hi = int(lo_t * 100), int(hi_t * 100)
        if hi - lo < 3:
            return (lo_t + hi_t) / 2
        return (lo + int(np.argmin(rms[lo:hi]))) / 100

    # Audio without caption words may hold untagged music, so clip edges stay close to
    # captioned words: at most MAX_TAIL after the last word starts, MAX_LEAD before the first.
    MAX_TAIL, MAX_LEAD, MAX_GAP = 1.2, 0.5, 3.0
    ev_iv = [(a - EVENT_MARGIN, b + EVENT_MARGIN) for a, b, _ in events]
    rows, kept_s, drop = [], 0.0, {"event": 0, "dur": 0, "rate": 0, "gap": 0}
    for ci, c in enumerate(clips):
        ws = [w for s in c for w in s]
        prev_last = clips[ci - 1][-1][-1][0] if ci > 0 else 0.0
        next_first = clips[ci + 1][0][0][0] if ci + 1 < len(clips) else total
        first, last = ws[0][0], ws[-1][0]
        a = quietest(max(prev_last + 0.15, first - MAX_LEAD), first)
        b = quietest(last + 0.15, min(next_first, last + MAX_TAIL))
        text = " ".join(w for _, w in ws)
        text = re.sub(r"\s+", " ", text).strip()
        if any(a < e2 and b > e1 for e1, e2 in ev_iv):
            drop["event"] += 1
            continue
        if any(t2 - t1 > MAX_GAP for (t1, _), (t2, _) in zip(ws, ws[1:])):
            drop["gap"] += 1
            continue
        seg = audio[int(a * SR): int(b * SR)]
        # trim leading/trailing silence relative to the clip's peak, keep 120 ms pad
        f = frame_rms(seg)
        thr = f.max() * 10 ** (-35 / 20)
        idx = np.where(f > thr)[0]
        if len(idx) == 0:
            continue
        s0 = max(0, idx[0] * 240 - int(0.12 * SR))
        s1 = min(len(seg), (idx[-1] + 1) * 240 + int(0.12 * SR))
        seg = seg[s0:s1]
        dur = len(seg) / SR
        if not (MIN_DUR <= dur <= MAX_DUR + 2):
            drop["dur"] += 1
            continue
        cps = len(text) / dur  # chars/sec sanity check against bad alignment
        if not (6 <= cps <= 25):
            drop["rate"] += 1
            continue
        peak = np.abs(seg).max()
        if peak > 0:
            seg = seg * (0.95 / peak) if peak > 0.95 else seg
        name = f"{stem}_{len(rows):05d}.flac"
        sf.write(out_dir / "flac" / name, seg, SR, subtype="PCM_16")
        rows.append({"audio": f"flac/{name}", "text": text, "source": audio_path.name,
                     "start": round(a + s0 / SR, 3), "dur": round(dur, 3)})
        kept_s += dur

    json.dump({"events": events}, open(out_dir / "events" / f"{stem}.json", "w"), indent=1)
    print(f"{stem}: kept {len(rows)} clips, {kept_s/3600:.2f} h; dropped {drop}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("workdir/raw"))
    ap.add_argument("--out", type=Path, default=Path("workdir/clips"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    pairs = find_pairs(args.raw_dir)
    if not pairs:
        raise SystemExit(f"no audio + .vtt pairs found in {args.raw_dir}")
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(segment_one, [a for a, _ in pairs], [v for _, v in pairs], [args.out] * len(pairs)))
    rows = [r for rs in results for r in rs]
    with open(args.out / "segments.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"total: {len(rows)} clips, {sum(r['dur'] for r in rows) / 3600:.2f} h -> {args.out / 'segments.jsonl'}")


if __name__ == "__main__":
    main()
