# SPDX-License-Identifier: Apache-2.0
# Derived from QwenLM/Qwen3-TTS finetuning/sft_12hz.py, Copyright 2026 The Alibaba Qwen team.
"""Step 3: single-speaker full fine-tune of Qwen3-TTS-12Hz (1.7B or 0.6B Base).

    python scripts/train.py --data workdir/prepared --run run1                       # 1.7B, needs ~40+ GB
    python scripts/train.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --precision bf16 \\
        --batch-size 2 --grad-accum 8 --run small                                    # fits 12-16 GB

Adapted from Qwen3-TTS/finetuning/sft_12hz.py (Apache-2.0, commit 022e286). Changes vs upstream:

  1. Text embeddings go through talker.text_projection, exactly as generate() does. Upstream
     feeds raw text embeddings, so the model is trained on inputs it never sees at inference.
  2. The code predictor is conditioned on the talker hidden state *before* each frame
     (codec_mask[:, 1:]), matching generate()'s past_hidden. Upstream uses codec_mask[:, :-1].
  3. Both losses are computed as plain cross-entropy on aligned logits. Upstream passes already
     shifted labels into HF ForCausalLMLoss, which shifts them a second time.
  Measured on the base 1.7B model before any training: val codec-0 loss 10.34 -> 1.41,
  code-predictor loss 9.23 -> 5.47 (run scripts/probe.py to reproduce on your data).

  Also: fp32 master weights + bf16 autocast by default (pure-bf16 AdamW silently drops small
  updates), SDPA attention (no flash-attn build), warmup + cosine LR, held-out validation loss
  per epoch, bf16 checkpoints that load directly with Qwen3TTSModel.from_pretrained.
"""
import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_cosine_schedule_with_warmup

BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Switches for the three fixes above; scripts/probe.py flips them to compare against upstream.
PROJECT_TEXT = True
SUB_MASK_SHIFT = 1  # 1 -> codec_mask[:, 1:] (matches generate), 0 -> codec_mask[:, :-1] (upstream)
ALIGNED_LOSS = True


def resolve_model(path_or_id: str) -> str:
    if os.path.isdir(path_or_id):
        return path_or_id
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_id)


def load_jsonl(path: Path):
    """Load items; relative audio paths are resolved against the jsonl's folder."""
    items = []
    for line in open(path):
        x = json.loads(line)
        for k in ("audio", "ref_audio"):
            if k in x and not os.path.isabs(x[k]):
                x[k] = str((path.parent / x[k]).resolve())
        items.append(x)
    return items


def compute_loss(model, batch):
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]

    speaker_embedding = model.speaker_encoder(batch["ref_mels"].to(model.device, model.dtype)).detach()

    input_text_embedding = model.talker.model.text_embedding(input_ids[:, :, 0])
    if PROJECT_TEXT:
        input_text_embedding = model.talker.text_projection(input_text_embedding)
    input_text_embedding = input_text_embedding * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_ids[:, :, 1]) * codec_embedding_mask
    input_codec_embedding[:, 6, :] = speaker_embedding
    input_embeddings = input_text_embedding + input_codec_embedding
    for i in range(1, 16):
        emb = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        input_embeddings = input_embeddings + emb * codec_mask.unsqueeze(-1)

    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=codec_0_labels[:, 1:],
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[0][-1]
    sub_mask = codec_mask[:, 1:] if SUB_MASK_SHIFT else codec_mask[:, :-1]
    talker_hidden_states = hidden_states[sub_mask]
    talker_codec_ids = codec_ids[codec_mask]
    sub_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
    talker_loss = outputs.loss
    if ALIGNED_LOSS:
        logits = outputs.logits
        talker_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), codec_0_labels[:, 1:].reshape(-1), ignore_index=-100)
        sub_talker_loss = F.cross_entropy(sub_logits.reshape(-1, sub_logits.shape[-1]).float(), talker_codec_ids[:, 1:].reshape(-1))
    return talker_loss + 0.3 * sub_talker_loss, talker_loss, sub_talker_loss, speaker_embedding


def save_checkpoint(accelerator, model, init_path, out_dir, speaker_name, spk_emb):
    # copy everything except the top-level talker weights (rewritten below); keep speech_tokenizer/ intact
    shutil.copytree(init_path, out_dir, dirs_exist_ok=True,
                    ignore=lambda d, names: [n for n in names if os.path.samefile(d, init_path) and n.startswith("model") and ".safetensors" in n])
    with open(os.path.join(init_path, "config.json")) as f:
        cfg = json.load(f)
    cfg["tts_model_type"] = "custom_voice"
    cfg.setdefault("talker_config", {})
    cfg["talker_config"]["spk_id"] = {speaker_name: 3000}
    cfg["talker_config"]["spk_is_dialect"] = {speaker_name: False}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    sd = accelerator.unwrap_model(model).state_dict()
    sd = {k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in sd.items() if not k.startswith("speaker_encoder")}
    sd["talker.model.codec_embedding.weight"][3000] = spk_emb.to(torch.bfloat16).cpu()
    save_file(sd, os.path.join(out_dir, "model.safetensors"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("workdir/prepared"), help="folder written by prepare.py")
    p.add_argument("--model", default=BASE_MODEL, help="HF id or local dir of a Qwen3-TTS-12Hz *-Base model")
    p.add_argument("--runs-dir", type=Path, default=Path("workdir/runs"))
    p.add_argument("--run", default="run1")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--speaker-name", default="narrator")
    p.add_argument("--precision", choices=["fp32-master", "bf16"], default="fp32-master",
                   help="fp32-master: fp32 weights + bf16 autocast (best, ~16 B/param). "
                        "bf16: bf16 weights and optimizer (upstream behaviour, ~8 B/param, for small GPUs)")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--max-steps", type=int, help="stop after N batches (smoke test)")
    p.add_argument("--no-save", action="store_true", help="skip writing checkpoints (smoke test)")
    args = p.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    init_path = resolve_model(args.model)
    out_dir = args.runs_dir / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_master = args.precision == "fp32-master"
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16" if fp32_master else "no")

    qwen3tts = Qwen3TTSModel.from_pretrained(init_path, dtype=torch.float32 if fp32_master else torch.bfloat16,
                                             attn_implementation="sdpa")
    config = AutoConfig.from_pretrained(init_path)
    model = qwen3tts.model
    model.speaker_encoder.requires_grad_(False)

    train_ds = TTSDataset(load_jsonl(args.data / "train_with_codes.jsonl"), qwen3tts.processor, config)
    val_ds = TTSDataset(load_jsonl(args.data / "val_with_codes.jsonl"), qwen3tts.processor, config)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate_fn, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=val_ds.collate_fn, num_workers=min(2, args.num_workers))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(train_dl) / args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(model, optimizer, train_dl, val_dl, scheduler)

    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                with accelerator.autocast():
                    _, l0, _, _ = compute_loss(model, batch)
                tot += l0.item()
                n += 1
        model.train()
        return tot / max(n, 1)

    history = []
    v = evaluate()
    accelerator.print(f"[init] val codec0 loss {v:.4f}  (expect ~1.4 for the 1.7B base; ~10 means a broken pipeline)")
    history.append({"epoch": -1, "val_loss": v})
    spk_emb = None
    t0 = time.time()
    seen = 0
    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss, l0, l1, emb = compute_loss(model, batch)
                if spk_emb is None:
                    spk_emb = emb[0].detach().float()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % 20 == 0:
                el = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
                accelerator.print(f"ep {epoch} step {step}/{len(train_dl)} loss {loss.item():.4f} (c0 {l0.item():.4f} "
                                  f"sub {l1.item():.4f}) lr {scheduler.get_last_lr()[0]:.2e} elapsed {el / 60:.1f}m "
                                  f"peak-mem {mem:.1f}GiB", flush=True)
            seen += 1
            if args.max_steps and seen >= args.max_steps:
                break
        v = evaluate()
        history.append({"epoch": epoch, "val_loss": v, "elapsed_min": (time.time() - t0) / 60})
        accelerator.print(f"[epoch {epoch}] val codec0 loss {v:.4f}", flush=True)
        if accelerator.is_main_process:
            if not args.no_save:
                save_checkpoint(accelerator, model, init_path, str(out_dir / f"checkpoint-epoch-{epoch}"),
                                args.speaker_name, spk_emb)
            json.dump(history, open(out_dir / "history.json", "w"), indent=1)
        if args.max_steps and seen >= args.max_steps:
            break


if __name__ == "__main__":
    main()
