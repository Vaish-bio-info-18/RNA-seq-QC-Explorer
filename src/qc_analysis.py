
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_counts(file_path):
    """Load and validate a gene-by-sample count matrix."""
    counts = pd.read_csv(file_path, index_col=0)

    if counts.empty:
        raise ValueError("The input matrix is empty.")

    if counts.index.has_duplicates:
        raise ValueError("Duplicate gene identifiers were found.")

    if counts.columns.has_duplicates:
        raise ValueError("Duplicate sample names were found.")

    counts = counts.apply(pd.to_numeric, errors="raise")

    if counts.isna().any().any():
        raise ValueError("Missing values were found.")

    if not np.isfinite(counts.to_numpy()).all():
        raise ValueError("Non-finite values were found.")

    if (counts < 0).any().any():
        raise ValueError("Negative expression values were found.")

    if (counts.sum(axis=0) == 0).any():
        raise ValueError("At least one sample has zero total counts.")

    return counts


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/qc_analysis.py data/demo_counts.csv")
        sys.exit(1)

    counts = load_counts(sys.argv[1])

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log-transform counts for exploratory visualization.
    log_counts = np.log2(counts + 1)

    # 1. Expression distribution
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=log_counts, orient="v")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("log2(count + 1)")
    plt.title("RNA-seq Expression Distributions")
    plt.tight_layout()
    plt.savefig(output_dir / "expression_distributions.png", dpi=300)
    plt.close()

    # 2. Sample correlation
    correlations = log_counts.corr(method="pearson")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        square=True
    )
    plt.title("Sample-to-Sample Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "sample_correlation.png", dpi=300)
    plt.close()

    # 3. PCA
    # Samples are observations; genes are features.
    sample_matrix = log_counts.T

    if sample_matrix.shape[0] < 2:
        raise ValueError("At least two samples are required for PCA.")

    if sample_matrix.shape[1] < 2:
        raise ValueError("At least two genes are required for PCA.")

    # Remove constant genes before scaling.
    variable = sample_matrix.var(axis=0) > 0
    sample_matrix = sample_matrix.loc[:, variable]

    if sample_matrix.shape[1] < 2:
        raise ValueError("At least two variable genes are required for PCA.")

    scaled = StandardScaler().fit_transform(sample_matrix)
    coordinates = PCA(n_components=2).fit_transform(scaled)

    pca_df = pd.DataFrame(
        coordinates,
        index=sample_matrix.index,
        columns=["PC1", "PC2"]
    )

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        s=100
    )

    for sample, row in pca_df.iterrows():
        plt.annotate(sample, (row["PC1"], row["PC2"]), xytext=(5, 5),
                     textcoords="offset points")

    plt.title("PCA of RNA-seq Samples")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_samples.png", dpi=300)
    plt.close()

    # Save a simple QC summary.
    summary = pd.DataFrame({
        "sample": counts.columns,
        "total_counts": counts.sum(axis=0).values,
        "detected_genes": (counts > 0).sum(axis=0).values,
        "mean_count": counts.mean(axis=0).values
    })

    summary.to_csv("results/qc_summary.csv", index=False)

    print(f"Samples analysed: {counts.shape[1]}")
    print(f"Genes analysed: {counts.shape[0]}")
    print(f"Figures saved to: {output_dir}")
    print("QC summary saved to results/qc_summary.csv")


if __name__ == "__main__":
    main()
