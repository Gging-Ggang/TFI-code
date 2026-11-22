# Code Instructions

**Clean-Relative Separation and Matched-Control Ordering in Transformation-Based Feature Scoring of Nightshade**

## Library installation

```bash
pip install -r requirements.txt
```

## Dataset download

From the repository root:

```bash
hf download GgingGgang/TFI-dataset \
  --repo-type dataset \
  --include "data/**" \
  --local-dir .
```

The scripts use `data/` under the repository root. For another location:

```bash
export TFI_DATA_ROOT=/path/to/TFI-dataset/data
```

The download contains the 3,000-source DomainNet pool, NS, GN, and GN-star at strengths 0.02 and 0.04, and the auxiliary datasets. Calculations exclude the five uniformly zero-RGB clipart sources and use 2,995 DomainNet sources.

When using the downloaded images, start from Step 3.

## 1. Data sampling

`001_Sampling.py` samples 500 images per domain with seed 0, center-crops, and resizes to 512 x 512.

Input directory:

```text
data/DomainNet/
```

Output directory:

```text
data/D_domainnet_clean/
```

Run:

```bash
python 001_Sampling.py
```

DomainNet: https://ai.bu.edu/DomainNet/

## 2. Perturbation generation

### 2-1. Nightshade

`Nightshade/gen_poison.py` uses the following target names:

| Source domain | Target name |
| --- | --- |
| clipart | infograph |
| infograph | clipart |
| painting | natural |
| quickdraw | sketch |
| real | painting |
| sketch | quickdraw |

Example for clipart at strength 0.04:

```bash
python Nightshade/gen_poison.py \
  --directory data/D_domainnet_clean/clipart \
  --target_name infograph \
  --outdir data/D_domainnet_nightshade_linf_004/clipart \
  --eps 0.04 \
  --constraint linf \
  --steps 500
```

For strength 0.02, use `--eps 0.02` and the output suffix `002`.

### 2-2. Gaussian noise

Set the strength and input/output directories in each script for 0.02 or 0.04.

```bash
python 002_GN.py
python 002_GN-star.py
```

## 3. Image transformations

`003_apply_image_transformations.py` applies 14 transformations to the DomainNet conditions and JPEG Q85 to the auxiliary datasets.

Input directory: `data/`

Output directory: `data/Transformation/`

```bash
python 003_apply_image_transformations.py
```

## 4. Feature extraction

`004_feature_extraction.py` extracts DomainNet features using six encoders and auxiliary-dataset features using SD VAE.

Input directories: `data/`, `data/Transformation/`

Output directory: `result/features/`

Outputs:

- `feature_metadata.parquet`
- `features.float16.npy` for each encoder

```bash
python 004_feature_extraction.py
```

## 5. TFI calculation

`005_calculate_tfi.py` calculates the 14 TFI scores.

Input directory: `result/features/`

Output directory: `result/tfi/`

Outputs: `tfi.csv`, `tfi.float32.npy`

```bash
python 005_calculate_tfi.py
```

## 7. Perturbation magnitudes

`007_calculate_perturbation_magnitude.py` calculates L-infinity and LPIPS magnitudes against the clean images.

Input: `result/features/feature_metadata.parquet` and image files.

Output: `result/perturbation/perturbation_per_image.csv`

```bash
python 007_calculate_perturbation_magnitude.py
```

## 8. Direct response baselines

`008_calculate_baseline_screening_scores.py` calculates pixel RMSE, pixel MAE, LPIPS response, frequency RMSE, and TFI under JPEG Q85 for clean, NS, and GN-star.

Input directories: `data/`, `data/Transformation/`, `result/tfi/`

Output directory: `result/baseline/`

```bash
python 008_calculate_baseline_screening_scores.py
```

## 10. Benign-processing TFI

`010_screening_metrics_with_benign.py` calculates reference-probe TFI after each of the 14 clean-image processing operations.

Input: `result/features/` and processed image files.

Output: `result/tfi/010_benign_tfi_sd_vae_jpeg85.csv`

```bash
python 010_screening_metrics_with_benign.py
```

## 12. Feature-response decomposition

`012_decomposition_analysis.py` calculates A, C, gamma, and squared TFI for NS and GN-star across 84 probes at both strengths.

Input directory: `result/features/`

Output: `result/analysis/012_decomposition_per_sample.csv`

```bash
python 012_decomposition_analysis.py
```

## 13. Paper calculations

`013_paper_analysis.py` calculates the main-paper and supplementary statistics from the measurements above.

Input directories:

- `result/tfi/`
- `result/baseline/`
- `result/analysis/`
- `result/perturbation/`

Output directory: `result/paper/`

```bash
python 013_paper_analysis.py
```

| Results | Output CSV |
| --- | --- |
| Probe AUROC, median shift, KS, Wasserstein | `013_probe_metrics.csv`, `013_score_distributions.csv` |
| Figure 1 | `013_fig1_scores.csv` |
| Figure 2 | `013_fig2_auroc.csv` |
| Figure 3 | `013_fig3_alignment_auroc.csv` |
| Table 2 | `013_reference_metrics.csv` |
| Table 3 | `013_baseline_auroc.csv` |
| Table 4 | `013_probe_metrics.csv` |
| Table 5 | `013_response_correlations.csv` |
| Table 6; domain results | `013_domain_metrics.csv` |
| Table 7; auxiliary results | `013_auxiliary_metrics.csv`, `013_auxiliary_distributions.csv` |
| Perturbation magnitudes | `013_perturbation_magnitudes.csv`, `013_auxiliary_magnitudes.csv` |
| Clean-relative/direct associations | `013_clean_direct_correlations.csv` |
| Direct-order counts | `013_direct_order_counts.csv` |
| Paired-response medians | `013_probe_response_medians.csv` |
| Paired ordering | `013_paired_ordering.csv`, `013_paired_ordering_summary.csv` |
| Source/probe sign counts | `013_source_sign_offsets.csv`, `013_sign_distributions.csv`, `013_joint_sign_distributions.csv` |
| Partial Spearman correlation | `013_partial_spearman.csv` |
| Benign-processing results | `013_benign_metrics.csv` |

For existing measurements in another location:

```bash
python 013_paper_analysis.py \
  --tfi-csv /path/to/tfi.csv \
  --responses /path/to/per_image_features_002.csv.gz \
  --responses /path/to/per_image_features_004.csv.gz \
  --baseline-csv /path/to/reference_direct_baseline_scores.csv.gz \
  --benign-csv /path/to/benign_tfi.csv.gz \
  --perturbation-csv /path/to/perturbation_per_image.csv
```

## References

[1] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. "Moment Matching for Multi-Source Domain Adaptation." ICCV 2019. https://ai.bu.edu/DomainNet/

[2] Shawn-Shan/nightshade-release. https://github.com/Shawn-Shan/nightshade-release

## License

Except for `Nightshade/`, the source code is licensed under the MIT License; see `LICENSE`.

Files under `Nightshade/` are adapted from the official Nightshade release and remain under GPL-3.0; see `Nightshade/LICENSE` and `Nightshade/NOTICE`.
