"""Sanity check before training: validation loss of the *untouched* base model under the upstream
training recipe vs. the fixed one (see scripts/train.py). A pretrained model should already
predict its own codec tokens well, so the fixed variant must start low.

    python scripts/probe.py --data workdir/prepared

Reference numbers (1.7B base, 40 held-out clips):
    upstream                 codec0 10.33  sub 9.23
    fixed (all three)        codec0  1.41  sub 5.47
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig

import train as T
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

VARIANTS = [  # (name, PROJECT_TEXT, SUB_MASK_SHIFT, ALIGNED_LOSS)
    ("upstream", False, 0, False),
    ("+aligned loss", False, 0, True),
    ("+text projection", True, 0, True),
    ("fixed (all three)", True, 1, True),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("workdir/prepared"))
    ap.add_argument("--model", default=T.BASE_MODEL)
    ap.add_argument("--n", type=int, default=40, help="number of validation clips")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    path = T.resolve_model(args.model)
    m = Qwen3TTSModel.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=args.device)
    items = T.load_jsonl(args.data / "val_with_codes.jsonl")[:args.n]
    ds = TTSDataset(items, m.processor, AutoConfig.from_pretrained(path))
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=ds.collate_fn)
    model = m.model.eval()
    for name, proj, shift, aligned in VARIANTS:
        T.PROJECT_TEXT, T.SUB_MASK_SHIFT, T.ALIGNED_LOSS = proj, shift, aligned
        l0s, l1s = [], []
        with torch.no_grad():
            for b in dl:
                b = {k: v.to(args.device) for k, v in b.items()}
                _, l0, l1, _ = T.compute_loss(model, b)
                l0s.append(l0.item())
                l1s.append(l1.item())
        print(f"{name:<20} codec0 {sum(l0s) / len(l0s):6.3f}  sub {sum(l1s) / len(l1s):6.3f}", flush=True)


if __name__ == "__main__":
    main()
