"""Run the GPU steps on Modal (https://modal.com). This is the route used for the published results:
~13 min of A100-80GB training, about $4 in total including data prep, probing and evaluation.

    # one-time: upload clips produced locally by scripts/segment.py
    modal volume create audiobook-tts
    modal volume put audiobook-tts workdir/clips /clips

    modal run modal_app.py::prepare                     # Whisper filter + codec tokens  (L40S, ~10 min)
    modal run modal_app.py::probe                       # sanity check: expect codec0 ~1.4 for "fixed"
    modal run --detach modal_app.py::train --run-name run1  # 1.7B full fine-tune           (A100-80GB)
    modal run modal_app.py::evaluate --run-name run1 --epochs 1,2,3

    modal volume get audiobook-tts runs/run1/checkpoint-epoch-3 workdir/models/narrator
    modal volume get audiobook-tts eval/run1 workdir/eval/run1

Set AUDIOBOOK_TTS_VOLUME to use a different volume name. All paths below are relative to the volume.
"""
import os
import subprocess

import modal

VOL = "/vol"
vol = modal.Volume.from_name(os.environ.get("AUDIOBOOK_TTS_VOLUME", "audiobook-tts"), create_if_missing=True)
here = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "sox", "libsox-dev", "git")
    .uv_sync(uv_project_dir=here, frozen=True)
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(os.path.join(here, "scripts"), "/opt/scripts")
)
app = modal.App("audiobook-tts", image=image)


def run(script, *args):
    cmd = ["python", f"/opt/scripts/{script}", *map(str, args)]
    print("+", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, cwd="/opt/scripts")
    finally:
        vol.commit()


@app.function(gpu="L40S", volumes={VOL: vol}, timeout=3600, cpu=8, memory=32768)
def prepare(clips: str = "clips", out: str = "prepared", wer_max: float = 0.25, floor_max_db: float = -45.0,
            n_val: int = 40, language: str = "English"):
    run("prepare.py", "--clips", f"{VOL}/{clips}", "--out", f"{VOL}/{out}", "--wer-max", wer_max,
        "--floor-max-db", floor_max_db, "--n-val", n_val, "--language", language)


@app.function(gpu="A100-80GB", volumes={VOL: vol}, timeout=1200)
def probe(data: str = "prepared", model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", n: int = 40):
    run("probe.py", "--data", f"{VOL}/{data}", "--model", model, "--n", n)


@app.function(gpu="A100-80GB", volumes={VOL: vol}, timeout=3 * 3600, cpu=8, memory=65536)
def train(data: str = "prepared", run_name: str = "run1", model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
          epochs: int = 5, lr: float = 1e-5, batch_size: int = 8, grad_accum: int = 2, max_steps: int = 0,
          no_save: bool = False):
    extra = (["--max-steps", max_steps] if max_steps else []) + (["--no-save"] if no_save else [])
    run("train.py", "--data", f"{VOL}/{data}", "--runs-dir", f"{VOL}/runs", "--run", run_name, "--model", model,
        "--epochs", epochs, "--lr", lr, "--batch-size", batch_size, "--grad-accum", grad_accum, *extra)


@app.function(gpu="L40S", volumes={VOL: vol}, timeout=3600)
def evaluate(run_name: str = "run1", epochs: str = "1,2,3", data: str = "prepared", n_val: int = 15, n_long: int = 3):
    cks = [f"{VOL}/runs/{run_name}/checkpoint-epoch-{e}" for e in epochs.split(",")]
    run("evaluate.py", "--checkpoints", *cks, "--data", f"{VOL}/{data}", "--out", f"{VOL}/eval/{run_name}",
        "--n-val", n_val, "--n-long", n_long)
