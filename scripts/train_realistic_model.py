#!/usr/bin/env python3
import argparse, numpy as np, torch, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training import train_on_dataset, RealisticTrainingConfig
from src.microbiome_datasets import load_npz_dataset, create_train_val_split

def evaluate_model(model, test_data, n_samples=1000):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        generated = model.generate(n_samples, device=device).cpu().numpy()
    row_sums = generated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    generated = generated / row_sums
    real = test_data[:n_samples]
    eps = 1e-10
    diff = real.mean(0) - generated.mean(0)
    mfd = float(np.dot(diff, diff))
    ar = float(np.mean(-np.sum(np.clip(real,eps,1)*np.log(np.clip(real,eps,1)),axis=1)))
    ag = float(np.mean(-np.sum(np.clip(generated,eps,1)*np.log(np.clip(generated,eps,1)),axis=1)))
    model.train()
    return {"mfd": round(mfd,4), "alpha_real": round(ar,4), "alpha_gen": round(ag,4),
            "sparsity_real": round(float(np.mean(real==0)),4), "sparsity_gen": round(float(np.mean(generated==0)),4)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="agp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="outputs/realistic_model")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*60)
    print("RealisticMicrobiomeModel Training")
    print("="*60)
    print("Dataset:", args.dataset)
    print("Epochs:", args.epochs)
    print("Device:", device)

    if args.dataset.endswith(".npz"):
        ds = load_npz_dataset(args.dataset)
        compositions = ds.compositions
        np.random.seed(42)
        idx = np.random.permutation(len(compositions))
        n_val = int(len(compositions) * 0.2)
        val_data = compositions[idx[:n_val]]
        train_data = compositions[idx[n_val:]]
        num_taxa = compositions.shape[1]
        print("Train:", train_data.shape, "Val:", val_data.shape)
        print("Sparsity:", round(float(np.mean(train_data==0)),3))
        from src.realistic_model import RealisticMicrobiomeModel
        from torch.utils.data import TensorDataset, DataLoader
        train_tensor = torch.tensor(train_data, dtype=torch.float32)
        model = RealisticMicrobiomeModel.from_real_data(
            real_data=train_tensor,
            lambda_alpha=2.0,
            lambda_beta=2.0,
            lambda_sparse=1.0
        ).to(device)
        print("Parameters:", sum(p.numel() for p in model.parameters()))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        loader = DataLoader(TensorDataset(train_tensor), batch_size=args.batch_size, shuffle=True)
        losses = []
        real_mean = torch.tensor(train_data.mean(0), dtype=torch.float32, device=device)
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                base_loss = model.compute_loss(batch)["total_loss"]
                # Invariance regularization: penalize deviation from real taxon rank ordering
                output = model(batch)
                gen_mean = output["reconstruction"].mean(0)
                # Spearman-like rank correlation loss
                real_ranks = real_mean.argsort().argsort().float()
                gen_ranks = gen_mean.argsort().argsort().float()
                rank_loss = torch.nn.functional.mse_loss(
                    gen_ranks / len(gen_ranks),
                    real_ranks / len(real_ranks)
                )
                loss = base_loss + 0.5 * rank_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            avg = epoch_loss / len(loader)
            losses.append(avg)
            if (epoch+1) % 10 == 0:
                lr = scheduler.get_last_lr()[0]
                print("Epoch", epoch+1, "/", args.epochs, "Loss:", round(avg,4), "LR:", round(lr,6))
    else:
        data_path = "data/american_gut" if args.dataset == "agp" else None
        dname = "american_gut" if args.dataset == "agp" else args.dataset
        config = RealisticTrainingConfig(dataset=dname, num_epochs=args.epochs,
                                         batch_size=args.batch_size, num_taxa=500)
        model, result = train_on_dataset(dname, config=config, use_real_data=True,
                                         data_path=data_path, verbose=True)
        losses = getattr(result, "train_losses", [])
        ds2 = load_npz_dataset(None) if False else None
        from src.microbiome_datasets import load_dataset
        ds2 = load_dataset(dname, use_real_data=True, data_dir=data_path)
        _, val_ds = create_train_val_split(ds2, val_fraction=0.2, seed=42)
        val_data = val_ds.compositions

    print("Evaluating...")
    metrics = evaluate_model(model, val_data)
    for k, v in metrics.items(): print(" ", k, v)
    torch.save(model.state_dict(), output_dir / "model.pt")
    json.dump({"dataset": args.dataset, "epochs": args.epochs, "metrics": metrics, "losses": losses},
              open(output_dir / "results.json", "w"), indent=2)
    print("Saved to", output_dir)
    print("DONE")

if __name__ == "__main__":
    main()