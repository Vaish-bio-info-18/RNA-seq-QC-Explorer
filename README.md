# RNA-seq QC Explorer

A lightweight Python workflow for exploratory quality control and visualization of RNA-seq gene-expression count matrices.

## Features

- Validates the input count matrix
- Checks missing values and duplicate gene identifiers
- Generates a PCA plot of samples
- Visualizes sample-to-sample correlations
- Compares expression distributions across samples

## Requirements

Python 3.10 or later is recommended.

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

Prepare a CSV file with genes as rows and samples as columns. The first column should contain gene identifiers, and the remaining columns should contain numeric expression counts.

Run:

```bash
python src/qc_analysis.py data/demo_counts.csv
```

Figures will be saved to `results/figures/`.

## Important note

This tool provides exploratory QC visualizations. PCA and correlation patterns do not independently establish differential expression, biological significance, or diagnostic performance.
