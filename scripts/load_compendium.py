import gzip
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.microbiome_datasets import MicrobiomeDatasetPreprocessor

DATA_DIR    = Path("data/compendium")
META_PATH   = DATA_DIR / "sample_metadata.tsv"
TABLE_PATH  = DATA_DIR / "taxonomic_table.csv.gz"
OUTPUT_PATH = DATA_DIR / "non_western_processed.npz"
N_SUBSAMPLE = 15000
EXCLUDE_ISO = {"US","CA","GB","DK","FI","DE","NL","SE","AU","NZ","unknown"}
RANDOM_SEED = 42

def main():
    print("="*60)
    print("Human Microbiome Compendium - Non-Western Cohort Loader")
    print("="*60)

    print("\n[1/5] Loading metadata...")
    meta = pd.read_csv(META_PATH, sep="\t")
    print(f"      Total samples: {len(meta):,}")
    filtered = meta[~meta["iso"].isin(EXCLUDE_ISO)].copy()
    print(f"      After filter: {len(filtered):,}")
    if N_SUBSAMPLE and len(filtered) > N_SUBSAMPLE:
        filtered = filtered.sample(n=N_SUBSAMPLE, random_state=RANDOM_SEED)
        print(f"      Subsampled to: {N_SUBSAMPLE:,}")
    target_srr = set(filtered["srr"].astype(str))
    print(f"      Countries: {filtered['iso'].nunique()}")
    print(filtered["region"].value_counts().to_string())

    print("\n[2/5] Loading taxonomic table...")
    with gzip.open(TABLE_PATH, "rt") as f:
        tax = pd.read_csv(f, index_col=0)
    print(f"      Full shape: {tax.shape}")

    # sample column format is PROJECT_SRR — extract just the SRR part
    tax["srr"] = tax["sample"].str.split("_").str[-1]
    tax = tax.set_index("srr")
    tax = tax.drop(columns=["sample"], errors="ignore")

    available = set(tax.index.astype(str)) & target_srr
    print(f"      Matching samples: {len(available):,}")

    if len(available) == 0:
        print("ERROR: Still no matches after fixing ID format.")
        print("Tax SRR sample:", list(tax.index[:5]))
        print("Meta SRR sample:", list(filtered["srr"].head()))
        sys.exit(1)

    print("\n[3/5] Filtering...")
    tax_f = tax.loc[tax.index.isin(available)]
    print(f"      Shape: {tax_f.shape}")
    counts = tax_f.values.astype(np.float32)
    print(f"      Raw sparsity: {np.mean(counts==0):.3f}")

    print("\n[4/5] Preprocessing...")
    pre = MicrobiomeDatasetPreprocessor(min_prevalence=0.01, min_abundance=1e-5, max_taxa=500)
    ds = pre.process(counts=counts, taxa_names=list(tax_f.columns), sample_ids=list(tax_f.index))
    print(f"      Samples: {ds.compositions.shape[0]:,}, Taxa: {ds.compositions.shape[1]:,}")
    print(f"      Sparsity:  {ds.stats.mean_sparsity:.3f}")
    print(f"      Alpha div: {ds.stats.alpha_diversity_mean:.3f}")
    print(f"      Beta div:  {ds.stats.beta_diversity_mean:.3f}")

    print(f"\n[5/5] Saving to {OUTPUT_PATH}...")
    np.savez_compressed(OUTPUT_PATH,
        compositions=ds.compositions,
        taxa_names=np.array(ds.taxa_names),
        sample_ids=np.array(ds.sample_ids),
        mean_sparsity=ds.stats.mean_sparsity,
        taxon_prevalences=ds.stats.taxon_prevalences,
        alpha_diversity_mean=ds.stats.alpha_diversity_mean,
        beta_diversity_mean=ds.stats.beta_diversity_mean)

    print("\n" + "="*60)
    print("SUCCESS")
    print(f"compositions shape: {ds.compositions.shape}")

if __name__ == "__main__":
    main()
