#!/usr/bin/env python3
"""
Generate all figures for Section 3.10 (Generalization to Diverse Populations).
Requires:
  - outputs/realistic_compendium_v3/model.pt   (our model)
  - outputs/baselines_compendium/results.json  (baseline results)
  - data/compendium/non_western_40k.npz
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json
import sys
from pathlib import Path
sys.path.insert(0, '.')

from src.realistic_model import RealisticMicrobiomeModel
from src.sparsity_loss import compute_target_sparsity_from_data, compute_target_prevalence_from_data
from src.baselines import BaselineVAE, BaselineGAN

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FIGURES_DIR = Path('outputs/figures_compendium')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    'ours':      '#1F4E79',
    'vae':       '#E67E22',
    'gan':       '#E74C3C',
    'copula':    '#8E44AD',
    'dirichlet': '#27AE60',
    'real':      '#2C3E50',
}
LABELS = {
    'ours': 'Ours (Sparsity-Preserving)',
    'vae': 'VAE',
    'gan': 'GAN',
    'copula': 'Copula',
    'dirichlet': 'Dirichlet',
    'real': 'Real Data',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
d = np.load('data/compendium/non_western_40k.npz', allow_pickle=True)
comps = d['compositions']
np.random.seed(SEED)
idx = np.random.permutation(len(comps))
n_val = int(len(comps) * 0.2)
val_data  = comps[idx[:n_val]]
train_data = comps[idx[n_val:]]
num_taxa = comps.shape[1]

# ── Load our model ────────────────────────────────────────────────────────────
print("Loading our model...")
train_t = torch.tensor(train_data, dtype=torch.float32)
target_sparsity, _ = compute_target_sparsity_from_data(train_t)
taxon_prevalences = compute_target_prevalence_from_data(train_t)

model = RealisticMicrobiomeModel(
    num_taxa=num_taxa, embedding_dim=64, hidden_dim=256,
    target_sparsity=float(target_sparsity),
    taxon_prevalences=taxon_prevalences,
    lambda_kl=0.1, lambda_sparse=1.0,
    lambda_alpha=0.1, lambda_beta=0.1,
    lambda_coex=0.1, lambda_rare=1.0
).to(DEVICE)
model.load_state_dict(torch.load('outputs/realistic_compendium_v4/best_model.pt', weights_only=False))
model.eval()
with torch.no_grad():
    gen_ours = model.generate(min(2000, len(val_data)), device=DEVICE).detach().cpu().numpy()
s = gen_ours.sum(axis=1, keepdims=True)
gen_ours = gen_ours / np.where(s==0, 1, s)

# ── Load baselines ────────────────────────────────────────────────────────────
print("Loading baselines...")
vae = BaselineVAE(num_taxa=num_taxa, latent_dim=64, hidden_dims=[512,256,128]).to(DEVICE)
vae.load_state_dict(torch.load('outputs/baselines_compendium/vae.pt', weights_only=False))
vae.eval()
with torch.no_grad():
    gen_vae = vae.sample(2000, device=DEVICE).cpu().numpy()

gan = BaselineGAN(num_taxa=num_taxa, latent_dim=128, hidden_dim=512).to(DEVICE)
gan.load_state_dict(torch.load('outputs/baselines_compendium/gan.pt', weights_only=False))
gan.eval()
with torch.no_grad():
    gen_gan = gan.generate(2000, device=DEVICE).cpu().numpy()

# Copula and Dirichlet
alpha = np.load('outputs/baselines_compendium/dirichlet_alpha.npy')
gen_dirichlet = np.random.dirichlet(alpha, size=2000)

from scipy.stats import norm as sci_norm
marginal_params = np.load('outputs/baselines_compendium/copula_marginals.npy')
corr = np.load('outputs/baselines_compendium/copula_corr.npy')
try:
    L = np.linalg.cholesky(corr)
except:
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-8)
    L = np.linalg.cholesky(eigvecs @ np.diag(eigvals) @ eigvecs.T)
z_cop = np.random.randn(2000, num_taxa) @ L.T
u_cop = np.clip(sci_norm.cdf(z_cop), 1e-6, 1-1e-6)
clr_gen = np.zeros_like(u_cop)
for i in range(num_taxa):
    mu_i, sigma_i = marginal_params[i]
    clr_gen[:,i] = sci_norm.ppf(u_cop[:,i]) * sigma_i + mu_i
gen_raw = np.exp(clr_gen)
gen_copula = gen_raw / gen_raw.sum(axis=1, keepdims=True)
prevalences = (train_data > 0).mean(0)
gen_copula[:, prevalences < 0.01] = 0.0
s2 = gen_copula.sum(axis=1, keepdims=True)
gen_copula = gen_copula / np.where(s2==0, 1, s2)

real_sub = val_data[:2000]
eps = 1e-10

all_generated = {
    'ours': gen_ours, 'vae': gen_vae, 'gan': gen_gan,
    'copula': gen_copula, 'dirichlet': gen_dirichlet
}

def get_metrics(gen, real):
    s = gen.sum(axis=1, keepdims=True)
    gen = gen / np.where(s==0, 1, s)
    diff = real.mean(0) - gen.mean(0)
    mfd = float(np.dot(diff, diff))
    alpha_gen = float(np.mean(-np.sum(np.clip(gen,eps,1)*np.log(np.clip(gen,eps,1)),axis=1)))
    sparsity = float(np.mean(gen==0))
    prev_real = (real>0).mean(0)
    prev_gen  = (gen>0).mean(0)
    corr_val = float(np.corrcoef(prev_real, prev_gen)[0,1]) if prev_gen.std() > 0 else 0.0
    return {'mfd': mfd, 'alpha': alpha_gen, 'sparsity': sparsity, 'prev_corr': corr_val}

metrics = {k: get_metrics(v, real_sub) for k, v in all_generated.items()}
metrics['real'] = {
    'mfd': 0.0,
    'alpha': float(np.mean(-np.sum(np.clip(real_sub,eps,1)*np.log(np.clip(real_sub,eps,1)),axis=1))),
    'sparsity': float(np.mean(real_sub==0)),
    'prev_corr': 1.0
}
print("Metrics computed:")
for k, v in metrics.items():
    print(f"  {k}: MFD={v['mfd']:.4f}, Sparsity={v['sparsity']:.3f}, Alpha={v['alpha']:.3f}, PrevCorr={v['prev_corr']:.3f}")

# ════════════════════════════════════════════════════════════════════════
# FIGURE 1: Sparsity Comparison (equivalent to paper Figure 4)
# ════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 1: Sparsity Comparison...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Sparsity Analysis — Non-Western Compendium', fontsize=14, fontweight='bold', y=1.01)

methods_order = ['real', 'ours', 'vae', 'gan', 'copula', 'dirichlet']

# A) Sparsity bar chart
ax = axes[0, 0]
sparsity_vals = [metrics[m]['sparsity'] for m in methods_order]
bars = ax.bar([LABELS[m] for m in methods_order], sparsity_vals,
              color=[COLORS[m] for m in methods_order], alpha=0.85, edgecolor='white', linewidth=0.5)
ax.axhline(metrics['real']['sparsity'], color=COLORS['real'], linestyle='--', linewidth=2, label='Real target')
ax.set_ylabel('Sparsity (Fraction of Zeros)')
ax.set_title('A) Sparsity Levels Across Methods')
ax.set_ylim(0, 1.05)
ax.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, sparsity_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# B) Non-zero abundance distribution
ax = axes[0, 1]
real_nz = real_sub[real_sub > 0]
our_nz  = gen_ours[gen_ours > 0]
ax.hist(np.log10(real_nz + 1e-10), bins=50, alpha=0.6, color=COLORS['real'], label='Real', density=True)
ax.hist(np.log10(our_nz + 1e-10),  bins=50, alpha=0.6, color=COLORS['ours'], label='Ours', density=True)
ax.set_xlabel('log10(Abundance)')
ax.set_ylabel('Density')
ax.set_title('B) Non-zero Abundance Distributions')
ax.legend()

# C) Per-taxon prevalence correlation (ours vs real)
ax = axes[0, 2]
prev_real_taxa = (real_sub > 0).mean(0)
prev_ours_taxa = (gen_ours > 0).mean(0)
ax.scatter(prev_real_taxa, prev_ours_taxa, alpha=0.3, s=8, color=COLORS['ours'])
max_val = max(prev_real_taxa.max(), prev_ours_taxa.max())
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect correlation')
corr_val = metrics['ours']['prev_corr']
ax.set_xlabel('Real Prevalence')
ax.set_ylabel('Generated Prevalence')
ax.set_title(f'C) Prevalence Correlation (r={corr_val:.3f})')
ax.legend(fontsize=9)

# D) Sample sparsity distribution
ax = axes[1, 0]
real_sp_per_sample = (real_sub == 0).mean(1)
our_sp_per_sample  = (gen_ours == 0).mean(1)
ax.hist(real_sp_per_sample, bins=30, alpha=0.6, color=COLORS['real'], label=f'Real (mean={real_sp_per_sample.mean():.3f})', density=True)
ax.hist(our_sp_per_sample,  bins=30, alpha=0.6, color=COLORS['ours'], label=f'Ours (mean={our_sp_per_sample.mean():.3f})', density=True)
ax.set_xlabel('Per-sample Sparsity')
ax.set_ylabel('Density')
ax.set_title('D) Sample Sparsity Distribution')
ax.legend()

# E) Alpha diversity comparison
ax = axes[1, 1]
real_alpha = -np.sum(np.clip(real_sub,eps,1)*np.log(np.clip(real_sub,eps,1)),axis=1)
our_alpha  = -np.sum(np.clip(gen_ours,eps,1)*np.log(np.clip(gen_ours,eps,1)),axis=1)
ax.hist(real_alpha, bins=40, alpha=0.6, color=COLORS['real'], label=f'Real (mean={real_alpha.mean():.2f})', density=True)
ax.hist(our_alpha,  bins=40, alpha=0.6, color=COLORS['ours'], label=f'Ours (mean={our_alpha.mean():.2f})', density=True)
ax.set_xlabel('Shannon Entropy (Alpha Diversity)')
ax.set_ylabel('Density')
ax.set_title('E) Alpha Diversity Distribution')
ax.legend()

# F) Prevalence deviation across all methods
ax = axes[1, 2]
prev_real_taxa = (real_sub > 0).mean(0)
deviations = []
method_labels = []
for m in ['ours', 'vae', 'gan', 'copula', 'dirichlet']:
    gen_m = all_generated[m]
    prev_m = (gen_m > 0).mean(0)
    dev = np.abs(prev_m - prev_real_taxa).mean()
    deviations.append(dev)
    method_labels.append(LABELS[m])
bars2 = ax.bar(method_labels, deviations,
               color=[COLORS[m] for m in ['ours','vae','gan','copula','dirichlet']],
               alpha=0.85, edgecolor='white')
ax.set_ylabel('Mean |Δ Prevalence|')
ax.set_title('F) Prevalence Deviation from Real')
ax.tick_params(axis='x', rotation=30)
for bar, val in zip(bars2, deviations):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_sparsity_comparison.png')
plt.close()
print("  Saved fig_sparsity_comparison.png")

# ════════════════════════════════════════════════════════════════════════
# FIGURE 2: Method Comparison (equivalent to paper Figure 5)
# ════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Method Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Generation Quality — Non-Western Compendium', fontsize=14, fontweight='bold')

methods_plot = ['ours', 'vae', 'gan', 'copula', 'dirichlet']

# A) MFD
ax = axes[0]
mfd_vals = [metrics[m]['mfd'] for m in methods_plot]
bars = ax.bar([LABELS[m] for m in methods_plot], mfd_vals,
              color=[COLORS[m] for m in methods_plot], alpha=0.85, edgecolor='white')
ax.set_ylabel('MFD (lower is better)')
ax.set_title('A) Microbiome Fréchet Distance')
ax.tick_params(axis='x', rotation=35)
for bar, val in zip(bars, mfd_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=8)

# B) Alpha diversity
ax = axes[1]
alpha_real_val = metrics['real']['alpha']
alpha_vals = [metrics[m]['alpha'] for m in methods_plot]
bars2 = ax.bar([LABELS[m] for m in methods_plot], alpha_vals,
               color=[COLORS[m] for m in methods_plot], alpha=0.85, edgecolor='white')
ax.axhline(alpha_real_val, color=COLORS['real'], linestyle='--', linewidth=2, label=f'Real ({alpha_real_val:.2f})')
ax.set_ylabel('Shannon Entropy')
ax.set_title('B) Alpha Diversity')
ax.tick_params(axis='x', rotation=35)
ax.legend(fontsize=9)

# C) Sparsity deviation
ax = axes[2]
sparsity_real_val = metrics['real']['sparsity']
sparsity_dev = [abs(metrics[m]['sparsity'] - sparsity_real_val) for m in methods_plot]
bars3 = ax.bar([LABELS[m] for m in methods_plot], sparsity_dev,
               color=[COLORS[m] for m in methods_plot], alpha=0.85, edgecolor='white')
ax.set_ylabel('|Sparsity Generated - Sparsity Real|')
ax.set_title('C) Sparsity Deviation')
ax.tick_params(axis='x', rotation=35)
for bar, val in zip(bars3, sparsity_dev):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_method_comparison.png')
plt.close()
print("  Saved fig_method_comparison.png")

# ════════════════════════════════════════════════════════════════════════
# FIGURE 3: Ablation Heatmap (cross-dataset)
# ════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Ablation Heatmap...")
with open('outputs/ablations_realistic/results.json') as f:
    abl_compendium = json.load(f)
with open('outputs/ablations_realistic_agp/results.json') as f:
    abl_agp = json.load(f)

config_names = {
    '1_full_model': 'Full Model',
    '2_no_sparsity_loss': 'w/o Sparsity Loss',
    '3_no_diversity_loss': 'w/o Diversity Loss',
    '4_no_kl': 'w/o KL',
    '5_no_coexclusion': 'w/o Co-exclusion',
    '6_no_rare_taxa': 'w/o Rare Taxa Loss',
    '7_base_model': 'Base Model',
}
configs = list(config_names.keys())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Ablation Study — MFD Across Datasets', fontsize=14, fontweight='bold')

for ax, (abl_data, dataset_name) in zip(axes, [
    (abl_compendium, 'Non-Western Compendium (sparsity=0.898)'),
    (abl_agp, 'AGP (sparsity=0.646)')
]):
    base_mfd = abl_data['1_full_model']['mfd']
    mfds = [abl_data[c]['mfd'] for c in configs]
    deltas = [m - base_mfd for m in mfds]
    colors_abl = ['#2ECC71' if d <= 0 else '#E74C3C' for d in deltas]
    colors_abl[0] = '#1F4E79'
    bars = ax.barh([config_names[c] for c in configs], mfds,
                   color=colors_abl, alpha=0.85, edgecolor='white')
    ax.axvline(base_mfd, color='#1F4E79', linestyle='--', linewidth=1.5)
    ax.set_xlabel('MFD (lower is better)')
    ax.set_title(dataset_name)
    for bar, val, delta in zip(bars, mfds, deltas):
        label = f'{val:.4f}' if delta == 0 else f'{val:.4f} ({delta:+.4f})'
        ax.text(val + 0.0001, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_ablation_heatmap.png')
plt.close()
print("  Saved fig_ablation_heatmap.png")

# ════════════════════════════════════════════════════════════════════════
# FIGURE 4: Alpha Diversity Analysis
# ════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: Alpha Diversity Analysis...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Alpha Diversity Analysis — Compendium vs AGP', fontsize=14, fontweight='bold')

# Load AGP for comparison
agp_data = np.load('data/agp_processed.npz', allow_pickle=True)
agp_comps = agp_data['compositions']
np.random.seed(SEED)
agp_idx = np.random.permutation(len(agp_comps))
agp_val = agp_comps[agp_idx[:int(len(agp_comps)*0.2)]]
agp_train = agp_comps[agp_idx[int(len(agp_comps)*0.2):]]

# Load AGP model
agp_train_t = torch.tensor(agp_train, dtype=torch.float32)
agp_ts, _ = compute_target_sparsity_from_data(agp_train_t)
agp_tp = compute_target_prevalence_from_data(agp_train_t)
agp_model = RealisticMicrobiomeModel(
    num_taxa=agp_comps.shape[1], embedding_dim=64, hidden_dim=256,
    target_sparsity=float(agp_ts), taxon_prevalences=agp_tp,
    lambda_kl=0.1, lambda_sparse=1.0, lambda_alpha=0.1, lambda_beta=0.1,
    lambda_coex=0.1, lambda_rare=1.0
).to(DEVICE)
# Use best available AGP model
agp_model_path = 'outputs/realistic_agp_v5/model.pt' if Path('outputs/realistic_agp_v5/model.pt').exists() else 'outputs/realistic_agp/model.pt'
agp_model.load_state_dict(torch.load(agp_model_path, weights_only=False))
agp_model.eval()
with torch.no_grad():
    gen_agp = agp_model.generate(min(1000, len(agp_val)), device=DEVICE).detach().cpu().numpy()
s_agp = gen_agp.sum(axis=1, keepdims=True)
gen_agp = gen_agp / np.where(s_agp==0, 1, s_agp)

def alpha_div(x):
    return -np.sum(np.clip(x,eps,1)*np.log(np.clip(x,eps,1)), axis=1)

# A) Alpha diversity comparison across datasets
ax = axes[0]
ax.boxplot([alpha_div(agp_val[:1000]), alpha_div(gen_agp),
            alpha_div(real_sub[:1000]), alpha_div(gen_ours[:1000])],
           labels=['AGP\nReal', 'AGP\nGen', 'Compendium\nReal', 'Compendium\nGen'],
           patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7),
           medianprops=dict(color='navy', linewidth=2))
ax.set_ylabel('Shannon Entropy')
ax.set_title('A) Alpha Diversity: Real vs Generated')
ax.grid(axis='y', alpha=0.3)

# B) Max taxon abundance (dominance)
ax = axes[1]
real_max = real_sub[:1000].max(axis=1)
our_max  = gen_ours[:1000].max(axis=1)
ax.hist(real_max, bins=30, alpha=0.6, color=COLORS['real'],
        label=f'Real (mean={real_max.mean():.3f})', density=True)
ax.hist(our_max, bins=30, alpha=0.6, color=COLORS['ours'],
        label=f'Ours (mean={our_max.mean():.3f})', density=True)
ax.set_xlabel('Max Taxon Abundance per Sample')
ax.set_ylabel('Density')
ax.set_title('B) Dominant Taxon Distribution')
ax.legend()

# C) Nonzero taxa per sample
ax = axes[2]
real_nz_count = (real_sub[:1000] > 0).sum(axis=1)
our_nz_count  = (gen_ours[:1000] > 0).sum(axis=1)
ax.hist(real_nz_count, bins=30, alpha=0.6, color=COLORS['real'],
        label=f'Real (mean={real_nz_count.mean():.1f})', density=True)
ax.hist(our_nz_count,  bins=30, alpha=0.6, color=COLORS['ours'],
        label=f'Ours (mean={our_nz_count.mean():.1f})', density=True)
ax.set_xlabel('Number of Present Taxa per Sample')
ax.set_ylabel('Density')
ax.set_title('C) Taxa Richness Distribution')
ax.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_alpha_diversity_analysis.png')
plt.close()
print("  Saved fig_alpha_diversity_analysis.png")

# ════════════════════════════════════════════════════════════════════════
# FIGURE 5: Generalization Results Summary
# ════════════════════════════════════════════════════════════════════════
print("Generating Figure 5: Generalization Summary...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Generalization to Non-Western Populations', fontsize=14, fontweight='bold')

datasets = ['AGP', 'Compendium']
mfd_vals_ds = [0.0094, metrics['ours']['mfd']]
sparsity_real_ds = [0.656, metrics['real']['sparsity']]
sparsity_gen_ds  = [0.590, metrics['ours']['sparsity']]

x = np.arange(len(datasets))
width = 0.35

ax = axes[0]
bars1 = ax.bar(x - width/2, mfd_vals_ds, width, label='MFD', color=COLORS['ours'], alpha=0.85)
ax.set_ylabel('MFD (lower is better)')
ax.set_title('A) Generalization: MFD Across Datasets')
ax.set_xticks(x)
ax.set_xticklabels([f'{d}\n({n:,} samples)' for d, n in zip(datasets, [4827, 46763])])
for bar, val in zip(bars1, mfd_vals_ds):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, max(mfd_vals_ds)*1.3)

ax = axes[1]
bars2 = ax.bar(x - width/2, sparsity_real_ds, width, label='Real', color=COLORS['real'], alpha=0.85)
bars3 = ax.bar(x + width/2, sparsity_gen_ds,  width, label='Generated', color=COLORS['ours'], alpha=0.85)
ax.set_ylabel('Sparsity')
ax.set_title('B) Sparsity: Real vs Generated')
ax.set_xticks(x)
ax.set_xticklabels([f'{d}\n({n:,} samples)' for d, n in zip(datasets, [4827, 46763])])
ax.legend()
ax.set_ylim(0, 1.1)
for bars_g, vals in [(bars2, sparsity_real_ds), (bars3, sparsity_gen_ds)]:
    for bar, val in zip(bars_g, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_generalization_summary.png')
plt.close()
print("  Saved fig_generalization_summary.png")

# ── Save metrics summary ──────────────────────────────────────────────────────
summary = {
    'compendium_metrics': {k: {m: round(v,4) for m,v in metrics[k].items()} for k in metrics},
    'real_sparsity': round(float(np.mean(real_sub==0)), 4),
    'real_alpha': round(float(np.mean(alpha_div(real_sub))), 4),
}
with open(FIGURES_DIR / 'metrics_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nAll figures saved to {FIGURES_DIR}")
print("\nFINAL METRICS SUMMARY:")
print(f"{'Method':<25} {'MFD':>8} {'Sparsity':>10} {'Alpha':>8} {'PrevCorr':>10}")
print("-"*65)
for m in ['real','ours','vae','gan','copula','dirichlet']:
    r = metrics[m]
    print(f"{LABELS[m]:<25} {r['mfd']:>8.4f} {r['sparsity']:>10.4f} {r['alpha']:>8.4f} {r['prev_corr']:>10.4f}")
