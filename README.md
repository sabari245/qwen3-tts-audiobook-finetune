# qwen3-tts-audiobook-finetune

Fine-tune **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 12Hz 1.7B** on a few hours of a single narrator's
audiobook recordings, then narrate whole books in that voice: accent, pacing, pauses and reading style included.

- **Input:** long-form narration audio plus word-timed captions (`.vtt`, e.g. YouTube auto-captions).
- **Output:** a `custom_voice` checkpoint plus `narrate.py`, which turns any `.txt` into a full-length audiobook WAV.
- **Cost of the reference run:** ~8.5 h of narration; **12.8 min** of A100-80GB training on [Modal](https://modal.com); **about $4 total**, including data prep, probing and evaluation.
- **Inference speed:** ~**5× faster than real time** on a 12 GB RTX 3060 (batched).
- **Fixes three bugs in the upstream fine-tuning script.** Without them, the base model starts at a validation loss of ~10.3 instead of ~1.4 (see [below](#the-three-upstream-fine-tuning-bugs)).

> No audio, transcripts, or model weights are included in this repository. Bring your own data, and only use
> recordings you have the rights and the speaker's consent to use.

---

## Results (reference run)

There are 15 held-out sentences that were never trained on, plus one ~130-word passage generated 3 times.
Metrics:
- **WER**: Whisper large-v3 word error rate on the generated audio.
- **sim**: cosine similarity to the real narrator's speaker embedding.
- **pace**: words per second. The real narrator speaks at **2.42 w/s**.

| system | val WER | long WER (mean / worst) | sim | pace val / long |
|---|---|---|---|---|
| base model, zero-shot clone from a 10 s reference | 0.013 | 0.000 / 0.000 | 0.986 | 2.69 / **3.11** |
| fine-tuned, checkpoint-epoch-1 | 0.018 | 0.013 / 0.013 | 0.991 | 2.40 / 2.60 |
| fine-tuned, checkpoint-epoch-2 | 0.013 | **0.233 / 0.673** | 0.993 | 2.50 / 4.55 |
| **fine-tuned, checkpoint-epoch-3 (shipped)** | **0.004** | 0.013 / 0.020 | **0.994** | **2.41 / 2.56** |

Zero-shot cloning already sounds like the speaker, but it reads ~30% too fast and sounds flat. The fine-tune matches the
narrator's pace and delivery. Note the epoch-2 checkpoint: it has a fine sentence-level score, but on one of its three
long-passage runs it rushed and skipped words. Always evaluate on long passages (see
[choosing a checkpoint](#choosing-a-checkpoint)).

---

## Pipeline

```
raw/*.mp3 + *.vtt ──segment.py──▶ clips/ (3–18 s FLAC, music & non-speech removed)
                     (CPU, local)
clips/ ──prepare.py──▶ prepared/ (Whisper-verified text, ref.wav, 12 Hz codec tokens)
                       (GPU ≥ 16 GB)
prepared/ ──probe.py──▶ sanity check of the loss before spending money on training
prepared/ ──train.py──▶ runs/<run>/checkpoint-epoch-N   (custom_voice checkpoints)
checkpoints ──evaluate.py──▶ WER / similarity / pace report + WAVs to listen to
checkpoint + book.txt ──narrate.py──▶ book.wav
```

| step | script | what it does |
|---|---|---|
| 1 | `scripts/segment.py` | Parses rolling YouTube-style VTT captions into word timings. Cuts at the quietest point near sentence ends into 3–18 s clips. Drops anything near `[music]`, `[applause]` and other bracketed events, and any stretch with no caption words. |
| 2 | `scripts/prepare.py` | Re-transcribes every clip with Whisper large-v3, which gives better punctuation and casing than auto-captions. Drops clips where caption and Whisper text disagree (WER > 0.25, usually a bad cut) or where the noise floor is above −45 dB (background music). Picks one clean 8–12 s reference clip, splits train/val and extracts 12 Hz codec tokens. |
| 3 | `scripts/probe.py` | Computes the validation loss of the *untrained* base model under the upstream recipe and under the fixed recipe. If "fixed" isn't ~1.4, stop and look before training. |
| 4 | `scripts/train.py` | Full fine-tune with the three fixes, using fp32 master weights and bf16 autocast, warmup and cosine LR. Writes one checkpoint per epoch that loads directly as a `custom_voice` model. |
| 5 | `scripts/evaluate.py` | Compares the base zero-shot clone with your checkpoints on WER, speaker similarity and pace, and saves every generated WAV. |
| 6 | `narrate.py` | Cleans the text, splits it into sentence-sized chunks, synthesises them in batches with retries for bad generations, and stitches the result with natural pauses. |

### How music and non-speech are removed

The rule assumes clean narration: **if there is caption text, there is speech and no music; if there is no caption
text, that span is thrown away.** Concretely, in `segment.py`:

- Bracketed caption events (`[music]`, `[laughter]`, `[applause]`, …) and 1 s on either side of them are excluded.
- A clip never spans more than 3 s with no caption words (`MAX_GAP`).
- A clip starts at most 0.5 s before its first word and ends at most 1.2 s after its last word (`MAX_LEAD` / `MAX_TAIL`),
  so music that comes right before or after speech is not pulled in.
- Edges are trimmed at −35 dB with 120 ms padding. `prepare.py` then removes anything that still has a raised noise floor.

If your recordings have music *under* the voice, these rules are not enough. Run a source separator such as Demucs
before step 1.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/), Python 3.12, an NVIDIA GPU with a CUDA 12.8-capable driver, and `ffmpeg`.
Every dependency is pinned with `==` in `pyproject.toml` and fully locked in `uv.lock`. qwen-tts is pinned to
upstream commit `022e286`.

```bash
git clone https://github.com/sabari245/qwen3-tts-audiobook-finetune
cd qwen3-tts-audiobook-finetune
uv sync                  # local GPU route
uv sync --group modal    # also installs the Modal client
```

`sox` is optional; qwen-tts prints a warning when it is missing, which is harmless. flash-attn is not needed, since everything uses
PyTorch SDPA.

### Data layout

Everything lives under `workdir/`, which is git-ignored:

```
workdir/
  raw/        your recordings: book1.mp3 + book1.en.vtt, book2.m4a + book2.vtt, ...
  clips/      step 1 output
  prepared/   step 2 output
  runs/       checkpoints
  eval/       evaluation WAVs + report.json
  models/narrator/   the checkpoint you pick (narrate.py default)
```

Each audio file needs a captions file whose name starts with the same stem (`book1.mp3` ↔ `book1*.vtt`). The captions
must be **word-timed**: YouTube auto-captions have `<00:00:01.234><c> word</c>` tags. For your own uploads,
`yt-dlp --write-auto-subs --sub-langs en --sub-format vtt <url>` downloads them next to the audio. Sentence-level captions
also work but give coarser cuts.

**How much data?** The reference run used ~8.5 h, which became roughly 2.3k clips. 1–2 h is likely enough to pick up a voice, but
long-form *style* benefits from more.

---

## Route A: Modal (what the reference run used)

Step 1 runs locally on CPU, and the GPU steps run on Modal. Authenticate once with `modal token new`. Your token
lives in `~/.modal.toml`; never commit it.

```bash
# 1. segment locally (CPU)
uv run python scripts/segment.py --raw-dir workdir/raw --out workdir/clips

# 2. upload clips (volume name can be changed via AUDIOBOOK_TTS_VOLUME)
modal volume create audiobook-tts
modal volume put audiobook-tts workdir/clips /clips

# 3. GPU steps
uv run modal run modal_app.py::prepare                              # L40S
uv run modal run modal_app.py::probe                                # A100-80GB, ~2 min
uv run modal run --detach modal_app.py::train --run-name run1       # A100-80GB, ~2.5 min/epoch for 8.5 h of data
uv run modal app logs <app-id>                                      # follow a detached run
uv run modal run modal_app.py::evaluate --run-name run1 --epochs 1,2,3   # L40S

# 4. download the evaluation and the checkpoint you pick
modal volume get audiobook-tts eval/run1 workdir/eval/
modal volume get audiobook-tts runs/run1/checkpoint-epoch-3 workdir/models/
mv workdir/models/checkpoint-epoch-3 workdir/models/narrator
```

Each checkpoint is ~4.3 GB (bf16 talker plus the speech tokenizer). Delete runs you don't need with
`modal volume rm -r audiobook-tts runs/<run>`.

**Cost of the reference run:** $4.03 in total on Modal, according to `modal billing report`. That covers data prep, several probe runs,
a discarded first training run, the final 5-epoch run (12.8 min on A100-80GB) and evaluation. Modal list prices
at the time were A100-80GB ≈ $2.50/h and L40S ≈ $1.95/h. A single clean pass through the pipeline should cost well under that.

Check what you have spent with `modal billing report --for today`.

## Route B: your own GPU

```bash
uv run python scripts/segment.py --raw-dir workdir/raw --out workdir/clips
uv run python scripts/prepare.py --clips workdir/clips --out workdir/prepared   # 24 GB+; on 12 GB add --asr-batch-size 4 --codec-batch-size 4
uv run python scripts/probe.py   --data workdir/prepared
uv run python scripts/train.py   --data workdir/prepared --run run1
uv run python scripts/evaluate.py --checkpoints workdir/runs/run1/checkpoint-epoch-{1,2,3}
cp -r workdir/runs/run1/checkpoint-epoch-3 workdir/models/narrator
```

Step 2 runs Whisper large-v3 over every clip and is slow on a consumer GPU; with the small batch sizes it will also
keep the card busy for a long time, so consider running it on Modal even if you train locally.

VRAM guide for training (full fine-tune with AdamW):

| model | `--precision` | weights + optimizer | fits on |
|---|---|---|---|
| 1.7B | `fp32-master` (default, used for the results above) | ~16 B/param ≈ 27 GB + activations | 40 GB+ (A100/H100, A6000 48 GB) |
| 1.7B | `bf16` | ~8 B/param ≈ 14 GB + activations | 24 GB (RTX 3090/4090) with `--batch-size 2 --grad-accum 8` |
| 0.6B | `bf16` | ≈ 5 GB + activations | 12–16 GB with `--batch-size 2 --grad-accum 8` |

`bf16` keeps the weights and optimizer state in bf16, which is the upstream behaviour. Small updates can get rounded away, so
you may need a higher `--lr` (try 2e-5) and more epochs. Only the `fp32-master` 1.7B configuration produced the
results above. The smaller configurations are estimates; check `peak-mem` in the training log and adjust.

For a quick check that your setup works before a long run:
```bash
uv run python scripts/train.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --precision bf16 \
    --batch-size 2 --grad-accum 1 --max-steps 10 --no-save --run smoke
```

---

## Narrating a book

```bash
uv run python narrate.py book.txt -o book.wav                    # uses workdir/models/narrator
uv run python narrate.py book.txt --limit 10                     # first 10 chunks only, to check quickly
uv run python narrate.py book.txt --batch-size 4                 # if you run out of VRAM
uv run python narrate.py book.txt --save-clean                   # also writes the cleaned text next to the output
```

What it does:
- **Text cleanup.** Normalises smart quotes, ellipses and dashes. Strips URLs, HTML, markdown, emoji and stray symbols.
  `Chapter 060` becomes `Chapter 60`, and headings get a full stop so they are read as headings.
- **Chunking.** Splits into sentences and packs them into chunks of ≤ 300 characters (`--max-chars`). Very long
  sentences are broken at commas. `--mode full` sends the whole text in one generation. It is not recommended:
  the model drifts on very long outputs.
- **Batching.** Chunks are sorted by length and generated in batches (`--batch-size 8` fits in 12 GB with the 1.7B model).
  When a batch runs out of memory it is halved and retried automatically.
- **Retries.** Each chunk's duration is checked against the narrator's pace (2.42 words/s). A chunk that is
  implausibly short (skipped words) or long (babbling) is regenerated up to `--retries` times.
- **Joins.** Chunks are joined with 0.25 s of silence inside a paragraph and 0.7 s between paragraphs (`--sentence-pause`,
  `--paragraph-pause`).

Measured: **5.3× real time** on an RTX 3060 12 GB at batch 8. A 10-hour book takes about 2 hours.

To use a different narrator's pace in the retry check, change `NARRATOR_WPS` in `narrate.py`. `evaluate.py` prints
the real narrator's pace.

---

## The three upstream fine-tuning bugs

Upstream `finetuning/sft_12hz.py` (at commit `022e286`) trains the model on inputs that differ from what `generate()` feeds
it at inference. The quickest symptom: the validation loss of the **untouched base model** is ~10.3, about as high as
random guessing over 2048 codes would give. A model that already speaks fluently should score far better on its
own tokens. `scripts/probe.py` measures this on your data (1.7B, 40 held-out clips):

| variant | codec-0 loss | code-predictor loss |
|---|---|---|
| upstream | 10.33 | 9.23 |
| fixed (all three) | **1.39** | **5.47** |

1. **Missing text projection.** `generate()` passes text embeddings through `talker.text_projection`, but upstream
   training adds the raw embeddings.
2. **Off-by-one in the code predictor's conditioning.** At inference, the code predictor for frame *t* sees the
   talker hidden state that *produced* codec-0 of frame *t*. That is `codec_mask[:, 1:]`, not `codec_mask[:, :-1]`.
3. **Labels shifted twice.** The labels are already aligned with the logits, then HF's `ForCausalLMLoss` shifts them
   again. The fix computes plain cross-entropy instead.

Each fix can be switched off in `scripts/train.py` (`PROJECT_TEXT`, `SUB_MASK_SHIFT`, `ALIGNED_LOSS`), and `probe.py`
compares them side by side.

## Choosing a checkpoint

The validation loss is a poor guide here. In the reference run it was lowest after the first epoch (0.81) and rose
afterwards (1.15 at epoch 3). The epoch-3 checkpoint is still the best one to listen to, and the epoch-2 checkpoint
broke on long text. Teacher-forced loss on short clips doesn't measure how the model behaves when generating
a 60-second passage on its own. So:

- Run `evaluate.py` on several epochs and look at the **worst** long-passage WER, not the mean.
- Compare the pace to the real narrator's. A checkpoint that rushes usually also skips words.
- Listen to `<system>_val0.wav` next to the real clip that `evaluate.py` prints at the end.

Hyperparameters used: lr 1e-5, batch 8 × grad-accum 2, 5 epochs, 5% warmup then cosine decay, weight decay 0.01,
grad clip 1.0, loss = codec-0 + 0.3 × code-predictor. A first run at lr 2e-5 gave worse results and was discarded.

---

## Repository layout

```
narrate.py            text file → audiobook WAV
modal_app.py          Modal wrappers for prepare / probe / train / evaluate
scripts/segment.py    step 1  captions → clean clips (CPU)
scripts/prepare.py    step 2  Whisper filter, reference clip, codec tokens
scripts/probe.py      sanity check of the training loss
scripts/train.py      step 3  fixed fine-tuning script
scripts/evaluate.py   step 4  WER / similarity / pace comparison
scripts/dataset.py    vendored upstream TTSDataset
pyproject.toml, uv.lock   pinned environment (torch 2.11 + CUDA 12.8)
```

## Ethics

A fine-tuned voice model is a voice clone. Train only on recordings of a narrator who has agreed to it, and only use
recordings you have the rights to. Don't use the result to impersonate anyone. Label generated audio as synthetic
when you publish it.

## Acknowledgements and license

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Alibaba Qwen team (Apache-2.0). `scripts/dataset.py` is vendored
  from it and `scripts/train.py` is derived from its fine-tuning script; see `NOTICE`.
- [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) for transcript verification.
- [Modal](https://modal.com) for on-demand GPUs.

Code in this repository is released under the Apache License 2.0. The model weights are downloaded from Hugging Face and
are covered by their own licenses.
