#!/usr/bin/env python3
"""
Generate all Section 5.7 figures for the non-Western compendium.
Clean publication style: no suptitles, proper axis labels, consistent palette.
Usage: python3 scripts/generate_compendium_figures_v2.py
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm as sci_norm
import json
import sys
from pathlib import Path
sys.path.insert(0, '.')

from src.realistic_model import RealisticMicrobiomeModel
from src.sparsity_loss import compute_target_sparsity_from_data, compute_target_prevalence_from_data
from src.baselines import BaselineVAE, BaselineGAN

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path('outputs/figures_compendium_v2')
OUT.mkdir(parents=True, exist_ok=True)

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          10,
    'axes.titlesize':     10,
    'axes.titleweight':   'bold',
    'axes.labelsize':     9,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.8,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'legend.fontsize':    8,
    'legend.frameon':     False,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    'ours':      '#2166AC',   # blue
    'vae':       '#D6604D',   # red-orange
    'gan':       '#F4A582',   # light orange
    'copula':    '#762A83',   # purple
    'dirichlet': '#1B7837',   # green
    'real':      '#404040',   # dark grey
}
METHOD_ORDER = ['ours', 'vae', 'gan', 'copula', 'dirichlet']
METHOD_LABELS = {
    'ours':      'Ours',
    'vae':       'VAE',
    'gan':       'GAN',
    'copula':    'Copula',
    'dirichlet': 'Dirichlet',
    'real':      'Real',
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
d = np.load('data/compendium/non_western_40k.npz', allow_pickle=True)
comps = d['compositions']
idx = np.random.permutation(len(comps))
n_val = int(len(comps) * 0.2)
val_data   = comps[idx[:n_val]]
train_data = comps[idx[n_val:]]
num_taxa   = comps.shape[1]

# ── Load our model ────────────────────────────────────────────────────────────
print("Loading models...")
train_t = torch.tensor(train_data, dtype=torch.float32)
ts, _   = compute_target_sparsity_from_data(train_t)
tp      = compute_target_prevalence_from_data(train_t)

model = RealisticMicrobiomeModel(
    num_taxa=num_taxa, embedding_dim=64, hidden_dim=256,
    target_sparsity=float(ts), taxon_prevalences=tp,
    lambda_kl=0.1, lambda_sparse=1.0, lambda_alpha=0.1,
    lambda_beta=0.1, lambda_coex=0.1, lambda_rare=1.0
).to(DEVICE)
model.load_state_dict(torch.load(
    'outputs/realistic_compendium_v4/best_model.pt', weights_only=False))
model.eval()
with torch.no_grad():
    gen_ours = model.generate(min(2000, len(val_data)), device=DEVICE).detach().cpu().numpy()
s = gen_ours.sum(axis=1, keepdims=True)
gen_ours = gen_ours / np.where(s == 0, 1, s)

# ── Load baselines ────────────────────────────────────────────────────────────
vae = BaselineVAE(num_taxa=num_taxa, latent_dim=64, hidden_dims=[512, 256, 128]).to(DEVICE)
vae.load_state_dict(torch.load('outputs/baselines_compendium/vae.pt', weights_only=False))
vae.eval()
with torch.no_grad():
    gen_vae = vae.sample(2000, device=DEVICE).cpu().numpy()

gan = BaselineGAN(num_taxa=num_taxa, latent_dim=128, hidden_dim=512).to(DEVICE)
gan.load_state_dict(torch.load('outputs/baselines_compendium/gan.pt', weights_only=False))
gan.eval()
with torch.no_grad():
    gen_gan = gan.generate(2000, device=DEVICE).cpu().numpy()

alpha_dir = np.load('outputs/baselines_compendium/dirichlet_alpha.npy')
gen_dirichlet = np.random.dirichlet(alpha_dir, size=2000)

mp   = np.load('outputs/baselines_compendium/copula_marginals.npy')
corr = np.load('outputs/baselines_compendium/copula_corr.npy')
try:
    L = np.linalg.cholesky(corr)
except:
    ev, evec = np.linalg.eigh(corr)
    L = np.linalg.cholesky(evec @ np.diag(np.maximum(ev, 1e-8)) @ evec.T)
z_cop  = np.random.randn(2000, num_taxa) @ L.T
u_cop  = np.clip(sci_norm.cdf(z_cop), 1e-6, 1 - 1e-6)
clr_g  = np.array([sci_norm.ppf(u_cop[:, i]) * mp[i, 1] + mp[i, 0] for i in range(num_taxa)]).T
gen_raw = np.exp(clr_g)
gen_copula = gen_raw / gen_raw.sum(axis=1, keepdims=True)
gen_copula[:, (train_data > 0).mean(0) < 0.01] = 0.0
s2 = gen_copula.sum(axis=1, keepdims=True)
gen_copula = gen_copula / np.where(s2 == 0, 1, s2)

real_sub = val_data[:2000]
eps = 1e-10

GEN = {
    'ours': gen_ours, 'vae': gen_vae, 'gan': gen_gan,
    'copula': gen_copula, 'dirichlet': gen_dirichlet,
}

def metrics(gen, real):
    s = gen.sum(axis=1, keepdims=True)
    gen = gen / np.where(s == 0, 1, s)
    mfd      = float(np.dot(real.mean(0) - gen.mean(0), real.mean(0) - gen.mean(0)))
    alpha_g  = float(np.mean(-np.sum(np.clip(gen,eps,1)*np.log(np.clip(gen,eps,1)),axis=1)))
    sparsity = float(np.mean(gen == 0))
    pr, pg   = (real > 0).mean(0), (gen > 0).mean(0)
    pc       = float(np.corrcoef(pr, pg)[0, 1]) if pg.std() > 0 else 0.0
    return {'mfd': mfd, 'alpha': alpha_g, 'sparsity': sparsity, 'prev_corr': pc}

M = {k: metrics(v, real_sub) for k, v in GEN.items()}
M['real'] = {
    'mfd': 0.0,
    'alpha': float(np.mean(-np.sum(np.clip(real_sub,eps,1)*np.log(np.clip(real_sub,eps,1)),axis=1))),
    'sparsity': float(np.mean(real_sub == 0)),
    'prev_corr': 1.0,
}
real_sparsity = M['real']['sparsity']
real_alpha    = M['real']['alpha']

print("Metrics:")
for k, v in M.items():
    print(f"  {k:12s}  MFD={v['mfd']:.4f}  sparsity={v['sparsity']:.3f}  alpha={v['alpha']:.3f}  pc={v['prev_corr']:.3f}")


# ════════════════════════════════════════════════════════════════════
# FIG 1 — Sparsity Analysis  (6-panel)
# ════════════════════════════════════════════════════════════════════
print("\nFig 1: Sparsity Analysis...")
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

# (A) Sparsity bar chart
ax = axes[0, 0]
methods_bar = ['real'] + METHOD_ORDER
vals = [M[m]['sparsity'] for m in methods_bar]
bars = ax.bar(range(len(methods_bar)), vals,
              color=[C.get(m, '#888') for m in methods_bar],
              width=0.6, edgecolor='white', linewidth=0.5)
ax.axhline(real_sparsity, color=C['real'], linestyle='--', linewidth=1.0, alpha=0.6)
ax.set_xticks(range(len(methods_bar)))
ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods_bar], rotation=30, ha='right')
ax.set_ylabel('Fraction of zeros')
ax.set_ylim(0, 1.1)
ax.set_title('(A) Sparsity by method')
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7)

# (B) Non-zero abundance distribution
ax = axes[0, 1]
real_nz = real_sub[real_sub > 0].flatten()
our_nz  = gen_ours[gen_ours > 0].flatten()
ax.hist(np.log10(real_nz + eps), bins=40, alpha=0.6, color=C['real'],
        label='Real', density=True)
ax.hist(np.log10(our_nz + eps),  bins=40, alpha=0.6, color=C['ours'],
        label='Ours', density=True)
ax.set_xlabel('log₁₀(abundance)')
ax.set_ylabel('Density')
ax.set_title('(B) Non-zero abundance distribution')
ax.legend()

# (C) Per-taxon prevalence scatter
ax = axes[0, 2]
pr_taxa = (real_sub > 0).mean(0)
pg_taxa = (gen_ours > 0).mean(0)
ax.scatter(pr_taxa, pg_taxa, alpha=0.25, s=5, color=C['ours'], rasterized=True)
lim = max(pr_taxa.max(), pg_taxa.max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Real prevalence')
ax.set_ylabel('Generated prevalence')
ax.set_title(f'(C) Prevalence correlation (r={M["ours"]["prev_corr"]:.3f})')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)

# (D) Per-sample sparsity distribution
ax = axes[1, 0]
rs = (real_sub == 0).mean(1)
os = (gen_ours == 0).mean(1)
ax.hist(rs, bins=30, alpha=0.6, color=C['real'],
        label=f'Real (μ={rs.mean():.3f})', density=True)
ax.hist(os, bins=30, alpha=0.6, color=C['ours'],
        label=f'Ours (μ={os.mean():.3f})', density=True)
ax.set_xlabel('Per-sample sparsity')
ax.set_ylabel('Density')
ax.set_title('(D) Sample sparsity distribution')
ax.legend()

# (E) Alpha diversity distribution
ax = axes[1, 1]
ra = -np.sum(np.clip(real_sub,eps,1)*np.log(np.clip(real_sub,eps,1)), axis=1)
oa = -np.sum(np.clip(gen_ours,eps,1)*np.log(np.clip(gen_ours,eps,1)), axis=1)
ax.hist(ra, bins=35, alpha=0.6, color=C['real'],
        label=f'Real (μ={ra.mean():.2f})', density=True)
ax.hist(oa, bins=35, alpha=0.6, color=C['ours'],
        label=f'Ours (μ={oa.mean():.2f})', density=True)
ax.set_xlabel('Shannon entropy')
ax.set_ylabel('Density')
ax.set_title('(E) Alpha diversity distribution')
ax.legend()

# (F) Prevalence deviation bar
ax = axes[1, 2]
pr_taxa_f = (real_sub > 0).mean(0)
devs = [np.abs((GEN[m] > 0).mean(0) - pr_taxa_f).mean() for m in METHOD_ORDER]
bars2 = ax.bar(range(len(METHOD_ORDER)), devs,
               color=[C[m] for m in METHOD_ORDER],
               width=0.6, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(METHOD_ORDER)))
ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=30, ha='right')
ax.set_ylabel('Mean |Δ prevalence|')
ax.set_title('(F) Prevalence deviation from real')
for bar, val in zip(bars2, devs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout(pad=0.8)
plt.savefig(OUT / 'fig_sparsity_comparison.png')
plt.close()
print("  Saved fig_sparsity_comparison.png")


# ════════════════════════════════════════════════════════════════════
# FIG 2 — Method Comparison  (3-panel)
# ════════════════════════════════════════════════════════════════════
print("Fig 2: Method Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(11, 4))

# (A) MFD
ax = axes[0]
mfds = [M[m]['mfd'] for m in METHOD_ORDER]
bars = ax.bar(range(len(METHOD_ORDER)), mfds,
              color=[C[m] for m in METHOD_ORDER],
              width=0.6, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(METHOD_ORDER)))
ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=30, ha='right')
ax.set_ylabel('MFD (lower = better)')
ax.set_title('(A) Microbiome Fréchet Distance')
for bar, val in zip(bars, mfds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7)

# (B) Alpha diversity
ax = axes[1]
alphas = [M[m]['alpha'] for m in METHOD_ORDER]
bars2 = ax.bar(range(len(METHOD_ORDER)), alphas,
               color=[C[m] for m in METHOD_ORDER],
               width=0.6, edgecolor='white', linewidth=0.5)
ax.axhline(real_alpha, color=C['real'], linestyle='--', linewidth=1.0, alpha=0.6,
           label=f'Real ({real_alpha:.2f})')
ax.set_xticks(range(len(METHOD_ORDER)))
ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=30, ha='right')
ax.set_ylabel('Shannon entropy')
ax.set_title('(B) Alpha diversity')
ax.legend(loc='upper right')

# (C) Sparsity deviation
ax = axes[2]
sdev = [abs(M[m]['sparsity'] - real_sparsity) for m in METHOD_ORDER]
bars3 = ax.bar(range(len(METHOD_ORDER)), sdev,
               color=[C[m] for m in METHOD_ORDER],
               width=0.6, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(METHOD_ORDER)))
ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=30, ha='right')
ax.set_ylabel('|Sparsity(gen) − Sparsity(real)|')
ax.set_title('(C) Sparsity deviation')
for bar, val in zip(bars3, sdev):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout(pad=0.8)
plt.savefig(OUT / 'fig_method_comparison.png')
plt.close()
print("  Saved fig_method_comparison.png")


# ════════════════════════════════════════════════════════════════════
# FIG 3 — Generalization Summary  (2-panel)
# ════════════════════════════════════════════════════════════════════
print("Fig 3: Generalization Summary...")
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

datasets      = ['AGP\n(4,827)', 'Compendium\n(46,763)']
mfd_vals      = [0.0094, M['ours']['mfd']]
sparsity_real = [0.656,  real_sparsity]
sparsity_gen  = [0.590,  M['ours']['sparsity']]
x = np.arange(len(datasets))
w = 0.35

ax = axes[0]
bars_m = ax.bar(x, mfd_vals, width=0.5, color=C['ours'],
                edgecolor='white', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylabel('MFD')
ax.set_title('(A) MFD across datasets')
for bar, val in zip(bars_m, mfd_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax = axes[1]
b1 = ax.bar(x - w/2, sparsity_real, width=w, color=C['real'],
            label='Real', edgecolor='white', linewidth=0.5)
b2 = ax.bar(x + w/2, sparsity_gen,  width=w, color=C['ours'],
            label='Generated', edgecolor='white', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylabel('Sparsity (fraction of zeros)')
ax.set_ylim(0, 1.1)
ax.set_title('(B) Sparsity: real vs generated')
ax.legend()
for bar, val in zip(list(b1) + list(b2), sparsity_real + sparsity_gen):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout(pad=0.8)
plt.savefig(OUT / 'fig_generalization_summary.png')
plt.close()
print("  Saved fig_generalization_summary.png")


# ════════════════════════════════════════════════════════════════════
# FIG 4 — Ablation Heatmap  (2-panel horizontal bar)
# ════════════════════════════════════════════════════════════════════
print("Fig 4: Ablation Heatmap...")
with open('outputs/ablations_realistic/results.json') as f:
    abl_c = json.load(f)
with open('outputs/ablations_realistic_agp/results.json') as f:
    abl_a = json.load(f)

cfg_keys   = ['1_full_model','2_no_sparsity_loss','3_no_diversity_loss',
              '4_no_kl','5_no_coexclusion','6_no_rare_taxa','7_base_model']
cfg_labels = ['Full model','w/o Sparsity loss','w/o Diversity loss',
              'w/o KL','w/o Co-exclusion','w/o Rare taxa','Base model']

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, abl, title in [
    (axes[0], abl_c, 'Non-Western Compendium (sparsity = 0.898)'),
    (axes[1], abl_a, 'AGP (sparsity = 0.646)'),
]:
    base = abl['1_full_model']['mfd']
    mfds = [abl[k]['mfd'] for k in cfg_keys]
    deltas = [m - base for m in mfds]
    colors = [C['ours']] + ['#D6604D' if d > 0 else '#1B7837' for d in deltas[1:]]
    y = range(len(cfg_keys))
    bars = ax.barh(list(y), mfds, color=colors, edgecolor='white',
                   linewidth=0.5, height=0.6)
    ax.axvline(base, color='#404040', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(cfg_labels[::-1] if False else cfg_labels, fontsize=8)
    ax.set_xlabel('MFD (lower = better)')
    ax.set_title(title, fontsize=9)
    for bar, val, delta in zip(bars, mfds, deltas):
        label = f'{val:.4f}' if abs(delta) < 1e-6 else f'{val:.4f} ({delta:+.4f})'
        ax.text(val + max(mfds) * 0.01,
                bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=7)

plt.tight_layout(pad=0.8)
plt.savefig(OUT / 'fig_ablation_heatmap.png')
plt.close()
print("  Saved fig_ablation_heatmap.png")


# ════════════════════════════════════════════════════════════════════
# FIG 5 — Alpha Diversity Analysis  (3-panel)
# ════════════════════════════════════════════════════════════════════
print("Fig 5: Alpha Diversity Analysis...")

# Load AGP model for comparison
agp_data  = np.load('data/agp_processed.npz', allow_pickle=True)
agp_comps = agp_data['compositions']
agp_idx   = np.random.permutation(len(agp_comps))
agp_val   = agp_comps[agp_idx[:int(len(agp_comps)*0.2)]]
agp_train = agp_comps[agp_idx[int(len(agp_comps)*0.2):]]
agp_t     = torch.tensor(agp_train, dtype=torch.float32)
agp_ts, _ = compute_target_sparsity_from_data(agp_t)
agp_tp    = compute_target_prevalence_from_data(agp_t)

agp_model = RealisticMicrobiomeModel(
    num_taxa=agp_comps.shape[1], embedding_dim=64, hidden_dim=256,
    target_sparsity=float(agp_ts), taxon_prevalences=agp_tp,
    lambda_kl=0.1, lambda_sparse=1.0, lambda_alpha=0.1,
    lambda_beta=0.1, lambda_coex=0.1, lambda_rare=1.0
).to(DEVICE)
agp_path = next((p for p in [
    'outputs/realistic_agp_v5/model.pt',
    'outputs/realistic_agp_v4/model.pt',
    'outputs/realistic_agp/model.pt',
] if Path(p).exists()), None)
if agp_path:
    agp_model.load_state_dict(torch.load(agp_path, weights_only=False))
    agp_model.eval()
    with torch.no_grad():
        gen_agp = agp_model.generate(min(1000, len(agp_val)), device=DEVICE).detach().cpu().numpy()
    s_agp = gen_agp.sum(axis=1, keepdims=True)
    gen_agp = gen_agp / np.where(s_agp == 0, 1, s_agp)
else:
    gen_agp = None

def alpha(x):
    return -np.sum(np.clip(x, eps, 1) * np.log(np.clip(x, eps, 1)), axis=1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

# (A) Boxplot comparison
ax = axes[0]
data_boxes, labels_boxes = [], []
data_boxes.append(alpha(agp_val[:1000]));  labels_boxes.append('AGP\nReal')
if gen_agp is not None:
    data_boxes.append(alpha(gen_agp));     labels_boxes.append('AGP\nGen')
data_boxes.append(alpha(real_sub[:1000])); labels_boxes.append('Compendium\nReal')
data_boxes.append(alpha(gen_ours[:1000])); labels_boxes.append('Compendium\nGen')

colors_box = [C['real'], C['ours'], C['real'], C['ours']]
bp = ax.boxplot(data_boxes, labels=labels_boxes, patch_artist=True,
                medianprops=dict(color='white', linewidth=1.5),
                flierprops=dict(marker='o', markersize=2, alpha=0.3),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8))
for patch, col in zip(bp['boxes'], colors_box):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
ax.set_ylabel('Shannon entropy')
ax.set_title('(A) Alpha diversity: real vs generated')
ax.grid(axis='y', linewidth=0.5, alpha=0.4)

# (B) Dominant taxon distribution
ax = axes[1]
real_max = real_sub[:1000].max(axis=1)
our_max  = gen_ours[:1000].max(axis=1)
ax.hist(real_max, bins=30, alpha=0.6, color=C['real'],
        label=f'Real (μ={real_max.mean():.3f})', density=True)
ax.hist(our_max,  bins=30, alpha=0.6, color=C['ours'],
        label=f'Ours (μ={our_max.mean():.3f})', density=True)
ax.set_xlabel('Max taxon abundance per sample')
ax.set_ylabel('Density')
ax.set_title('(B) Dominant taxon distribution')
ax.legend()

# (C) Taxa richness
ax = axes[2]
real_rich = (real_sub[:1000] > 0).sum(axis=1)
our_rich  = (gen_ours[:1000] > 0).sum(axis=1)
ax.hist(real_rich, bins=30, alpha=0.6, color=C['real'],
        label=f'Real (μ={real_rich.mean():.1f})', density=True)
ax.hist(our_rich,  bins=30, alpha=0.6, color=C['ours'],
        label=f'Ours (μ={our_rich.mean():.1f})', density=True)
ax.set_xlabel('Number of present taxa per sample')
ax.set_ylabel('Density')
ax.set_title('(C) Taxa richness distribution')
ax.legend()

plt.tight_layout(pad=0.8)
plt.savefig(OUT / 'fig_alpha_diversity_analysis.png')
plt.close()
print("  Saved fig_alpha_diversity_analysis.png")

# ── Save metrics ──────────────────────────────────────────────────────────────
with open(OUT / 'metrics_summary.json', 'w') as f:
    json.dump({'compendium_metrics': {k: {m: round(v, 4) for m, v in M[k].items()}
                                      for k in M},
               'real_sparsity': round(real_sparsity, 4),
               'real_alpha':    round(real_alpha, 4)}, f, indent=2)

print(f"\nAll figures saved to {OUT}/")
print("\nSUMMARY:")
print(f"{'Method':<12} {'MFD':>8} {'Sparsity':>10} {'Alpha':>8} {'PrevCorr':>10}")
print('-' * 52)
for m in ['real'] + METHOD_ORDER:
    r = M[m]
    print(f"{METHOD_LABELS[m]:<12} {r['mfd']:>8.4f} {r['sparsity']:>10.4f} {r['alpha']:>8.4f} {r['prev_corr']:>10.4f}")
