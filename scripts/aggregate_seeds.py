#!/usr/bin/env python3
"""
Aggregate results across 3 seeds for the compendium experiments.
Produces mean ± std for all metrics, updates the generalization table.
Run after all 3 seeds complete:
  outputs/compendium_seed1/best_model.pt
  outputs/compendium_seed2/best_model.pt
  outputs/compendium_seed3/best_model.pt
"""
import numpy as np
import torch
import json
import sys
from pathlib import Path
sys.path.insert(0, '.')

from src.realistic_model import RealisticMicrobiomeModel
from src.sparsity_loss import compute_target_sparsity_from_data, compute_target_prevalence_from_data

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS  = [1, 2, 3]
SEED_DIRS = [Path(f'outputs/compendium_seed{s}') for s in SEEDS]
OUTPUT_DIR = Path('outputs/compendium_multiseed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate(model, val_data, device):
    model.eval()
    with torch.no_grad():
        gen = model.generate(len(val_data), device=device).detach().cpu().numpy()
    s = gen.sum(axis=1, keepdims=True)
    gen = gen / np.where(s == 0, 1, s)
    real = val_data
    eps  = 1e-10

    diff     = real.mean(0) - gen.mean(0)
    mfd      = float(np.dot(diff, diff))
    alpha_r  = float(np.mean(-np.sum(np.clip(real,eps,1)*np.log(np.clip(real,eps,1)),axis=1)))
    alpha_g  = float(np.mean(-np.sum(np.clip(gen, eps,1)*np.log(np.clip(gen, eps,1)),axis=1)))
    spar_r   = float(np.mean(real == 0))
    spar_g   = float(np.mean(gen  == 0))
    pr, pg   = (real > 0).mean(0), (gen > 0).mean(0)
    pc       = float(np.corrcoef(pr, pg)[0, 1]) if pg.std() > 0 else 0.0

    return {
        'mfd':           round(mfd,   4),
        'alpha_real':    round(alpha_r,4),
        'alpha_gen':     round(alpha_g,4),
        'sparsity_real': round(spar_r, 4),
        'sparsity_gen':  round(spar_g, 4),
        'prev_corr':     round(pc,     4),
    }

print("Loading compendium data...")
d          = np.load('data/compendium/non_western_40k.npz', allow_pickle=True)
comps      = d['compositions']
np.random.seed(42)
idx        = np.random.permutation(len(comps))
n_val      = int(len(comps) * 0.2)
val_data   = comps[idx[:n_val]]
train_data = comps[idx[n_val:]]
num_taxa   = comps.shape[1]
print(f"  Val: {val_data.shape}")

train_t = torch.tensor(train_data, dtype=torch.float32)
ts, _   = compute_target_sparsity_from_data(train_t)
tp      = compute_target_prevalence_from_data(train_t)

# ── Evaluate each seed ────────────────────────────────────────────────────────
all_results = []
for seed, seed_dir in zip(SEEDS, SEED_DIRS):
    best = seed_dir / 'best_model.pt'
    final = seed_dir / 'model.pt'
    ckpt = best if best.exists() else (final if final.exists() else None)

    if ckpt is None:
        print(f"  Seed {seed}: NOT FOUND — skipping")
        continue

    print(f"  Seed {seed}: loading {ckpt}...")
    model = RealisticMicrobiomeModel(
        num_taxa=num_taxa, embedding_dim=64, hidden_dim=256,
        target_sparsity=float(ts), taxon_prevalences=tp,
        lambda_kl=0.1, lambda_sparse=1.0, lambda_alpha=0.1,
        lambda_beta=0.1, lambda_coex=0.1, lambda_rare=1.0
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, weights_only=False))
    r = evaluate(model, val_data, DEVICE)
    r['seed'] = seed
    all_results.append(r)
    print(f"    MFD={r['mfd']:.4f}  sparsity={r['sparsity_gen']:.4f}  "
          f"alpha={r['alpha_gen']:.4f}  pc={r['prev_corr']:.4f}")

if len(all_results) == 0:
    print("No seed results found. Are the seeds still training?")
    sys.exit(1)

# ── Aggregate ─────────────────────────────────────────────────────────────────
metrics = ['mfd', 'alpha_gen', 'sparsity_gen', 'prev_corr',
           'alpha_real', 'sparsity_real']

summary = {}
for m in metrics:
    vals = [r[m] for r in all_results]
    summary[m] = {
        'mean':   round(float(np.mean(vals)), 4),
        'std':    round(float(np.std(vals)),  4),
        'min':    round(float(np.min(vals)),  4),
        'max':    round(float(np.max(vals)),  4),
        'values': vals,
    }

print("\n" + "="*60)
print(f"MULTI-SEED RESULTS ({len(all_results)} seeds)")
print("="*60)
print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-"*60)
for m in metrics:
    s = summary[m]
    print(f"{m:<20} {s['mean']:>10.4f} {s['std']:>10.4f} "
          f"{s['min']:>10.4f} {s['max']:>10.4f}")

# ── LaTeX table row ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LATEX TABLE ROW (for Table 7 in manuscript):")
print("="*60)
mfd   = summary['mfd']
spar  = summary['sparsity_gen']
alpha = summary['alpha_gen']
pc    = summary['prev_corr']
print(f"Non-Western Compendium & 46,763 & "
      f"${mfd['mean']:.4f} \\pm {mfd['std']:.4f}$ & "
      f"{summary['alpha_real']['mean']:.3f} & "
      f"${alpha['mean']:.3f} \\pm {alpha['std']:.3f}$ & "
      f"{summary['sparsity_real']['mean']:.3f} & "
      f"${spar['mean']:.3f} \\pm {spar['std']:.3f}$ \\\\")

print(f"\nPrevalence correlation: ${pc['mean']:.4f} \\pm {pc['std']:.4f}$")

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    'n_seeds':     len(all_results),
    'seeds':       SEEDS[:len(all_results)],
    'per_seed':    all_results,
    'summary':     summary,
}
with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR}/results.json")
