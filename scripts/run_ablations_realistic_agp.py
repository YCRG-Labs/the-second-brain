#!/usr/bin/env python3
import numpy as np, torch, json, sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, '.')
from src.realistic_model import RealisticMicrobiomeModel
from src.microbiome_datasets import load_npz_dataset
from src.sparsity_loss import compute_target_sparsity_from_data, compute_target_prevalence_from_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 128
SEED = 42

def evaluate(model, test_data, n=500):
    model.eval()
    with torch.no_grad():
        gen = model.generate(n, device=DEVICE).cpu().numpy()
    s = gen.sum(axis=1, keepdims=True)
    gen = gen / np.where(s==0, 1, s)
    real = test_data[:n]
    eps = 1e-10
    diff = real.mean(0) - gen.mean(0)
    mfd = float(np.dot(diff, diff))
    ar = float(np.mean(-np.sum(np.clip(real,eps,1)*np.log(np.clip(real,eps,1)),axis=1)))
    ag = float(np.mean(-np.sum(np.clip(gen,eps,1)*np.log(np.clip(gen,eps,1)),axis=1)))
    model.train()
    return {"mfd": round(mfd,4), "alpha_real": round(ar,4), "alpha_gen": round(ag,4),
            "sparsity_real": round(float(np.mean(real==0)),4),
            "sparsity_gen": round(float(np.mean(gen==0)),4)}

def run_ablation(name, train_data, test_data, kwargs):
    print("\n" + "="*50)
    print("Ablation:", name)
    print("="*50)
    model = RealisticMicrobiomeModel(num_taxa=train_data.shape[1], **kwargs).to(DEVICE)
    loader = DataLoader(TensorDataset(torch.tensor(train_data, dtype=torch.float32)),
                        batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            loss = model.compute_loss(batch)["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} Loss: {total/len(loader):.4f}")
    metrics = evaluate(model, test_data)
    print(f"  MFD={metrics['mfd']}, Alpha_gen={metrics['alpha_gen']}, Sparsity_gen={metrics['sparsity_gen']}")
    return metrics

def main():
    ds = load_npz_dataset("data/agp_processed.npz")
    comps = ds.compositions
    np.random.seed(SEED)
    idx = np.random.permutation(len(comps))
    n_val = int(len(comps)*0.2)
    test_data = comps[idx[:n_val]]
    train_data = comps[idx[n_val:]]
    print(f"Train: {train_data.shape}, Sparsity: {np.mean(train_data==0):.3f}")
    train_t = torch.tensor(train_data, dtype=torch.float32)
    target_sparsity, _ = compute_target_sparsity_from_data(train_t)
    taxon_prevalences = compute_target_prevalence_from_data(train_t)
    base = {"embedding_dim": 64, "hidden_dim": 256,
            "target_sparsity": float(target_sparsity),
            "taxon_prevalences": taxon_prevalences,
            "lambda_kl": 0.1, "lambda_sparse": 1.0,
            "lambda_alpha": 0.1, "lambda_beta": 0.1,
            "lambda_coex": 0.1, "lambda_rare": 1.0}
    ablations = {
        "1_full_model":        {**base},
        "2_no_sparsity_loss":  {**base, "lambda_sparse": 0.0},
        "3_no_diversity_loss": {**base, "lambda_alpha": 0.0, "lambda_beta": 0.0},
        "4_no_kl":             {**base, "lambda_kl": 0.0},
        "5_no_coexclusion":    {**base, "lambda_coex": 0.0},
        "6_no_rare_taxa":      {**base, "lambda_rare": 0.0},
        "7_base_model":        {**base, "lambda_sparse": 0.0, "lambda_alpha": 0.0,
                                "lambda_beta": 0.0, "lambda_coex": 0.0, "lambda_rare": 0.0},
    }
    results = {}
    for name, kwargs in ablations.items():
        results[name] = run_ablation(name, train_data, test_data, kwargs)
    Path("outputs/ablations_realistic_agp").mkdir(parents=True, exist_ok=True)
    with open("outputs/ablations_realistic_agp/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*65)
    print("ABLATION SUMMARY - RealisticMicrobiomeModel")
    print("="*65)
    print(f"{'Config':<30} {'MFD':>8} {'Alpha_gen':>10} {'Sparsity_gen':>13}")
    print("-"*65)
    base_mfd = results["1_full_model"]["mfd"]
    for name, r in results.items():
        delta = r["mfd"] - base_mfd
        marker = "" if name == "1_full_model" else f"({delta:+.4f})"
        print(f"{name:<30} {r['mfd']:>8.4f} {r['alpha_gen']:>10.4f} {r['sparsity_gen']:>13.4f}  {marker}")

if __name__ == "__main__":
    main()
