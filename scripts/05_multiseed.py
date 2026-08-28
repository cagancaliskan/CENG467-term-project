"""Multi-seed retraining driver (reviewer comments R1 and R2).

R1: "Only a single training run was conducted, and no confidence intervals or
statistical significance tests were reported."
R2: "...the paper claims that the systems are equivalent, that teacher selection
has little effect, and that LoRA rank 8 provides the best trade-off."

Retrains every TRAINED system in Table I across three seeds, plus LoRA ranks 16
and 32 so the rank-8 Pareto claim can be tested rather than asserted.

Design decision, fixed in PREREGISTRATION.md section 5: only the TRAINING seed
varies -- LoRA initialisation and batch shuffling.  The training subset is held
constant, so `prepare_synthetic` is always called with seed 42 and its output is
reused across seeds.  The measured spread is therefore optimisation noise, not
data resampling.  Getting this wrong would silently conflate the two.

Idempotent: any run whose adapter already exists (locally or on Drive) is
skipped, so a disconnected Colab session can simply be re-run.

Usage (Colab)::

    python scripts/05_multiseed.py --group core  --seeds 1337,2024 \
        --drive /content/drive/MyDrive/ceng467_termproject/CENG467-term-project
    python scripts/05_multiseed.py --group lora  --seeds 1337,2024 --drive ...
    python scripts/05_multiseed.py --group infer --seeds 1337,2024 --drive ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# run_name -> (teacher, prompt_variant, size, lora_rank, group)
RUNS = {
    "B2_human_n10000_r8":  ("human",     "concise", 10000, 8,  "core"),
    "S_gpt_n10000_r8":     ("openai",    "concise", 10000, 8,  "core"),
    "S_claude_n10000_r8":  ("anthropic", "concise", 10000, 8,  "core"),
    "S_gpt_n10000_r16":    ("openai",    "concise", 10000, 16, "lora"),
    "S_gpt_n10000_r32":    ("openai",    "concise", 10000, 32, "lora"),
}
# Only the core systems are evaluated out of domain (Table II).
OOD_RUNS = {"B2_human_n10000_r8", "S_gpt_n10000_r8", "S_claude_n10000_r8"}

PREP_SEED = 42  # NEVER varied -- see module docstring


def sh(cmd: str) -> int:
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True).returncode
    print(f"  -> rc={rc} in {time.time()-t0:.0f}s", flush=True)
    return rc


def need(path: Path, what: str) -> None:
    if not path.exists():
        sys.exit(f"MISSING {what}: {path}\nCopy it from Drive before running this script.")


def ensure_data(drive: Path | None) -> None:
    """Make sure data/raw and data/synthetic exist locally, pulling from Drive."""
    # --- synthetic teacher cache ---------------------------------------
    # Prefer the consolidated JSONLs: four files instead of ~22k, which is the
    # difference between seconds and half an hour when the source is Drive.
    syn = Path("data/synthetic")
    syn.mkdir(parents=True, exist_ok=True)
    if not list(syn.glob("*.jsonl")):
        copied = False
        if drive is not None:
            for j in sorted((drive / "data/synthetic").glob("*.jsonl")):
                shutil.copy2(j, syn / j.name)
                print(f"copied {j.name}", flush=True)
                copied = True
        if not copied:
            if drive is None:
                sys.exit("data/synthetic is empty and no --drive given.")
            src = drive / "data/synthetic"
            need(src, "data/synthetic")
            print("No consolidated JSONL found; copying the per-article cache "
                  "from Drive. This is slow (~22k files).", flush=True)
            print("Run scripts/consolidate_synthetic.py afterwards so the next "
                  "session does not pay this cost again.", flush=True)
            shutil.copytree(src, syn, dirs_exist_ok=True)

    # --- raw corpora -----------------------------------------------------
    raw = Path("data/raw")
    if not (raw.exists() and any(raw.rglob("*.jsonl"))):
        if drive is None:
            sys.exit("data/raw is empty and no --drive given.")
        need(drive / "data/raw", "data/raw")
        print("copying data/raw from Drive", flush=True)
        shutil.copytree(drive / "data/raw", raw, dirs_exist_ok=True)

    test = Path("data/raw/mlsum_tr/test.jsonl")
    if not test.exists():
        # Fall back to reconstructing the test set from an archived prediction
        # file, which carries {id, article, reference} for exactly the 2,000
        # scored articles.  Safer than re-downloading MLSUM and hoping the
        # sampling matches.
        for cand in ("outputs/predictions/v1_raw/S_gpt.jsonl",
                     "outputs/predictions/clean/S_gpt.jsonl"):
            p = Path(cand)
            if p.exists():
                print(f"reconstructing {test} from {p}", flush=True)
                test.parent.mkdir(parents=True, exist_ok=True)
                with open(p, encoding="utf-8") as fi, open(test, "w", encoding="utf-8") as fo:
                    for line in fi:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        fo.write(json.dumps({"id": r["id"], "article": r["article"],
                                             "reference": r.get("reference")},
                                            ensure_ascii=False) + "\n")
                break
    need(test, "MLSUM-TR test split")


def prep_dir(teacher: str, prompt: str, size: int) -> Path:
    return Path(f"data/processed/{teacher}_{prompt}_n{size}_seed{PREP_SEED}")


def ensure_prepared(teacher: str, prompt: str, size: int) -> Path:
    out = prep_dir(teacher, prompt, size)
    if (out / "train.jsonl").exists() and (out / "validation.jsonl").exists():
        print(f"[skip prep] {out}")
        return out
    cmd = (f"python -m src.data.prepare_synthetic --teacher {teacher} "
           f"--prompt-variant {prompt} --size {size} --out-dir {out} --seed {PREP_SEED}")
    if sh(cmd) != 0:
        sys.exit(f"prepare_synthetic failed for {teacher}/{prompt}/{size}")
    return out


def sync(local: Path, drive: Path | None, rel: str) -> None:
    if drive is None or not local.exists():
        return
    dst = drive / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if local.is_dir():
        shutil.copytree(local, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(local, dst)
    print(f"  synced -> {dst}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed retraining + inference.")
    ap.add_argument("--group", required=True, choices=["core", "lora", "infer", "all"])
    ap.add_argument("--seeds", default="1337,2024")
    ap.add_argument("--drive", default=None, help="Drive repo root, for checkpoint/prediction sync.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    drive = Path(args.drive) if args.drive else None
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not args.dry_run:
        ensure_data(drive)

    train_groups = {"core": ["core"], "lora": ["lora"], "all": ["core", "lora"], "infer": []}[args.group]
    planned = [(n, c) for n, c in RUNS.items() if c[4] in train_groups]

    # ---------------- training ----------------
    for run_name, (teacher, prompt, size, rank, _grp) in planned:
        proc = ensure_prepared(teacher, prompt, size) if not args.dry_run else prep_dir(teacher, prompt, size)
        for seed in seeds:
            tag = f"{run_name}_s{seed}"
            ckpt = Path(f"outputs/checkpoints/{tag}")
            final = ckpt / "final"
            if (final / "adapter_config.json").exists():
                print(f"[skip train] {tag} (local adapter exists)")
                continue
            if drive and (drive / f"outputs/checkpoints/{tag}/final/adapter_config.json").exists():
                print(f"[restore]    {tag} from Drive")
                if not args.dry_run:
                    shutil.copytree(drive / f"outputs/checkpoints/{tag}", ckpt, dirs_exist_ok=True)
                continue
            cmd = (f"python -m src.student.train "
                   f"--train-file {proc}/train.jsonl --val-file {proc}/validation.jsonl "
                   f"--output-dir {ckpt} --lora-rank {rank} --num-epochs {args.epochs} --seed {seed}")
            print(f"\n{'='*70}\nTRAIN {tag}  (rank={rank}, seed={seed})\n{'='*70}")
            if args.dry_run:
                print(f"$ {cmd}")
                continue
            if sh(cmd) != 0:
                sys.exit(f"training failed: {tag}")
            # keep only the final adapter; step checkpoints are large and useless here
            for d in ckpt.glob("checkpoint-*"):
                shutil.rmtree(d, ignore_errors=True)
            sync(ckpt, drive, f"outputs/checkpoints/{tag}")

    # ---------------- inference ----------------
    if args.group in ("infer", "all"):
        jobs = []
        for run_name in RUNS:
            for seed in seeds:
                tag = f"{run_name}_s{seed}"
                final = Path(f"outputs/checkpoints/{tag}/final")
                if not (final / "adapter_config.json").exists():
                    print(f"[skip infer] {tag} (no adapter)")
                    continue
                jobs.append((tag, final, "data/raw/mlsum_tr/test.jsonl", f"{tag}.jsonl"))
                if run_name in OOD_RUNS and Path("data/raw/trnews/test.jsonl").exists():
                    jobs.append((tag, final, "data/raw/trnews/test.jsonl", f"ood_{tag}.jsonl"))

        for tag, final, inp, outname in jobs:
            out = Path(f"outputs/predictions/multiseed/{outname}")
            if out.exists():
                print(f"[skip infer] {outname} (exists)")
                continue
            cmd = (f"python -m src.student.infer --model-path {final} "
                   f"--input {inp} --out {out}")
            print(f"\n{'='*70}\nINFER {outname}\n{'='*70}")
            if args.dry_run:
                print(f"$ {cmd}")
                continue
            if sh(cmd) != 0:
                sys.exit(f"inference failed: {outname}")
            sync(out, drive, f"outputs/predictions/multiseed/{outname}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
