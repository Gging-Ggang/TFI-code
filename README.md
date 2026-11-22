# Code Instructions (Supplementary Materials)

**Clean-Relative Separation and Matched-Control Ordering in Transformation-Based Feature Scoring of Nightshade**

## Library installation

For image processing and feature extraction:

```bash
pip install -r requirements.txt
```

For analysis of existing measurement CSVs and feature caches without model installation:

```bash
pip install -r requirements-analysis.txt
```

The image-processing and encoder stages require the environment in `requirements.txt`. The statistical analysis in `013_paper_analysis.py` uses NumPy, pandas, and SciPy; `012_decomposition_analysis.py` also reads Parquet metadata.

## Dataset download

The Hugging Face repository provides the full 3,000-source DomainNet image pool, both strengths (0.02 and 0.04) of Nightshade, nominal-budget Gaussian noise, and LPIPS-matched Gaussian noise, together with the auxiliary image-protection cohorts.

```bash
hf download GgingGgang/TFI-dataset \
  --repo-type dataset \
  --include "data/**" \
  --local-dir .
```

- Dataset: https://huggingface.co/datasets/GgingGgang/TFI-dataset
- Scripts use `data/` under the repository root by default.
- Set `TFI_DATA_ROOT` when the images are stored elsewhere:

```bash
export TFI_DATA_ROOT=/path/to/TFI-dataset/data
```

PowerShell:

```powershell
$env:TFI_DATA_ROOT = 'D:\path\to\TFI-dataset\data'
```

Use the downloaded images directly. The optional sampling and generation utilities are not prerequisites for evaluating the supplied images.

The paper excludes the same five clean clipart sources with uniformly zero RGB pixel values from every modification and strength. `paper_config.py` contains the explicit identities and validates the exact final cohort: 495 clipart images and 500 images from each other domain, totaling **2,995 sources**. A matching row count alone is not accepted as proof of cohort agreement.

The older CSVs under the HF repository's `supplementary_data/` describe historical mixture-screening experiments. They are **not inputs to the current paper analysis**. The `NS_lpips` image condition is likewise not part of the controlled benchmark analyzed here.

## 1. Optional sampling and perturbation generation

`001_Sampling.py` samples 500 images per DomainNet domain with seed 0, converts to RGB, center-crops, and resizes to 512 × 512.

- DomainNet: https://ai.bu.edu/DomainNet/
- Input: `data/DomainNet/`
- Output: `data/D_domainnet_clean/`

The generation utilities remain available:

| Condition | Code |
| --- | --- |
| Nightshade | `Nightshade/gen_poison.py` |
| Nominal-budget Gaussian noise | `002_GN.py` |
| LPIPS-matched Gaussian noise | `002_GN-star.py` |

Nightshade's source and target domains are clipart–infograph, real–painting, and sketch–quickdraw; painting uses the target name `natural`. The official implementation supports `--eps 0.02` and `--eps 0.04`. Gaussian generation scripts retain their original configuration blocks and seeds. When regenerating rather than downloading, configure each Gaussian run for its corresponding strength and NS directory. Do not rerun generation over the downloaded images as an analysis step.

Nightshade code is adapted from https://github.com/Shawn-Shan/nightshade-release and remains licensed under GPL-3.0; see `Nightshade/LICENSE` and `Nightshade/NOTICE`.

## 2. Current full-paper workflow

Run these commands from the repository root:

```bash
python 003_apply_image_transformations.py
python 004_feature_extraction.py
python 005_calculate_tfi.py
python 007_calculate_perturbation_magnitude.py
python 008_calculate_baseline_screening_scores.py
python 010_screening_metrics_with_benign.py --measurements-only
python 012_decomposition_analysis.py
python 013_paper_analysis.py
python 014_verify_paper_results.py
```

The existing numbered, standalone-script layout is retained. Current paper analysis does **not** use poison prevalence, inspection budgets, candidate mixtures, P@5, or trained detectors.

### 2.1 Transformations and features

`003_apply_image_transformations.py` processes clean images and all six DomainNet modification/strength combinations.

- JPEG quality: 95, 85, 75, 50.
- Gaussian blur sigma: 0.5, 1.0, 2.0, 3.0.
- Downscale–upscale ratio: 0.75, 0.6, 0.5, 0.4, 0.3, 0.25.

Auxiliary image-protection datasets need only original images and JPEG Q85.

`004_feature_extraction.py` writes row-aligned feature arrays and `feature_metadata.parquet` under `result/features/` for SD VAE, ResNet-50, ConvNeXt-Tiny, DINOv2 ViT-B/14, CLIP ViT-L/14-336, and ViT-B/16-384. Encoder/preprocessing settings remain in the script. The evaluation SD VAE is `stabilityai/sd-vae-ft-mse`, distinct from Nightshade's attack encoder.

`005_calculate_tfi.py` writes `result/tfi/tfi.csv` and its row-aligned `tfi.float32.npy`. The array retains the existing layout of 14 TFI columns followed by 14 angular-distance columns. Unmeasured auxiliary cells outside JPEG Q85 are explicit NaNs, not fabricated scores. Missing DomainNet measurements and missing auxiliary JPEG Q85 measurements are errors. Only the reference SD VAE–JPEG Q85 values enter auxiliary paper results.

### 2.2 Perturbation magnitudes and direct-response measures

`007_calculate_perturbation_magnitude.py` measures modifications against their corresponding clean images, with the paper's source exclusions and auxiliary cohorts kept separate.

`008_calculate_baseline_screening_scores.py` retains its filename and score definitions but now measures **JPEG Q85 at both strengths** for the current Table 3. Its measures are pixel RMSE, pixel MAE, AlexNet LPIPS, frequency RMSE, and SD-VAE TFI.

TFI comes from the existing 005 feature-score array, using its full 14-transformation column order. It is not recomputed with a different encoder.

### 2.3 Benign-processing measurements

`010_screening_metrics_with_benign.py --measurements-only` writes:

```text
result/tfi/010_benign_tfi_sd_vae_jpeg85.csv
```

For each of the same 14 processing operations P, the value is:

```text
||f(P(x)) - f(JPEG_Q85(P(x)))||_2
```

This is not the distance from untouched clean x to P(x). Existing model/cache-consistency checks are retained. No mixture-screening experiments are needed for the paper.

Measurements-only is also the default without a flag. Historical mixture screening requires the explicit --legacy-screening option.

### 2.4 Source-paired response measurements

`012_decomposition_analysis.py` reads the feature cache and writes:

```text
result/analysis/012_decomposition_per_sample.csv
```

The file contains both strengths, all 84 probes, and NS/GNstar source responses. It records A, C, gamma, clean/modified TFI, their squared scores, and numerical identity checks.

```text
c = f(x) - f(T(x))
d = [f(x_m) - f(x)] - [f(T(x_m)) - f(T(x))]
A = ||d||^2
C = 2 <c, d>
gamma = <c, d> / (||c|| ||d||)
||c + d||^2 = ||c||^2 + A + C
```

Reversing both response vectors gives the same A, C, and gamma. Undefined alignment is not replaced by a fabricated angle. The script checks source identities and agreement with 005 TFI values before publishing the CSV.

### 2.5 Full-distribution analysis

`013_paper_analysis.py` calculates:

- Clean-relative NS/GN/GNstar AUROC, KS, median shifts, and score distributions.
- Direct NS–GN and NS–GNstar AUROC, reversal counts, and probe rankings.
- The four direct-response baseline AUROCs.
- Source-paired NS-minus-GNstar differences in A, C, gamma, and squared TFI.
- Probe-level medians, normalization, raw/normalized Spearman correlations, within-group correlations, source-level sign offsets, joint signs, and partial Spearman correlations.
- Domain-wise, auxiliary-modification, and benign-processing results.

AUROC uses **all cross-distribution pairs**, with half credit for exact ties. Source-paired ordering is reported separately.

For response summaries, the code calculates each source difference first and then takes its median. It does not subtract independently aggregated condition medians. The normalization denominator is the **median of squared clean TFI** for that probe, not the square of its median.

Partial Spearman uses average ranks, OLS residualization with an intercept, and Pearson correlation of the two residual vectors.

Outputs are written only under `result/paper/`. Input files are read without being changed. `013_run_manifest.json` records input paths, sizes, hashes, cohort settings, and analysis definitions.

## 3. Reuse existing measurements

The analysis CLI accepts explicit file locations without requiring a particular research-workspace directory layout:

```bash
python 013_paper_analysis.py \
  --tfi-csv /path/to/tfi.csv \
  --responses /path/to/per_image_features_002.csv.gz \
  --responses /path/to/per_image_features_004.csv.gz \
  --baseline-csv /path/to/reference_direct_baseline_scores.csv.gz \
  --benign-csv /path/to/benign_tfi.csv.gz \
  --perturbation-csv /path/to/perturbation_per_image.csv \
  --output-root result/paper
```

Input contracts:

| Input | Required contents |
| --- | --- |
| TFI CSV | `domain`, `data_index`, `feature_extractor`, `poisoning`, `poisoning_factor`, and 14 `ld_*` columns; include the auxiliary cohorts for full-paper mode |
| Response CSV(s) | `strength`, source identity, probe, `modification` (NS/GNstar), `clean_tfi`, `modified_tfi`, `clean_s2`, `modified_s2`, `A`, `C`, `gamma` |
| Optional baseline CSV | Source identity, `strength`, `condition` (clean/NS/GNstar), and the five direct-response score columns |
| Default baseline files | The existing 008 row manifest, column manifest, and score NPY under `result/baseline/` |
| Benign CSV | Source identity, `benign_transformation`, `benign_factor`, and `latent_distance` or `tfi_processed_clean` |
| Perturbation CSV | Source identity, modification/strength, `linf`, and `lpips`, including auxiliary modifications |

`--responses` may be supplied once for a combined file or repeated for separate strength files. The analysis uses saved measurements as supplied; it does not silently replace them with newly inferred values.

`--scores-only` produces the core TFI-probe results without the response, baseline, magnitude, or auxiliary analyses. It is explicitly marked as a partial analysis in its manifest and is **not** a full-paper reproduction.

## 4. Paper-to-output mapping

| Paper item | Output under `result/paper/` |
| --- | --- |
| Figure 1 | `013_fig1_scores.csv` |
| Figure 2; Supplementary S6–S7 | `013_fig2_auroc.csv`, `013_probe_metrics.csv` |
| Figure 3; Supplementary paired-median summaries | `013_fig3_alignment_auroc.csv`, `013_probe_response_medians.csv` |
| Table 2 | `013_reference_metrics.csv` |
| Table 3 | `013_baseline_auroc.csv` |
| Table 4 | `013_probe_metrics.csv`, JPEG Q85, strength 0.04 |
| Table 5; Supplementary response correlations | `013_response_correlations.csv` |
| Table 6; Supplementary domain tables | `013_domain_metrics.csv` |
| Table 7 | `013_auxiliary_metrics.csv` |
| Supplementary perturbation/auxiliary magnitudes | `013_perturbation_magnitudes.csv`, `013_auxiliary_magnitudes.csv` |
| Supplementary S4–S5 and S10 distributions | `013_score_distributions.csv` |
| Clean-relative/direct rank associations | `013_clean_direct_correlations.csv` |
| Control-relative counts and probe selection | `013_direct_order_counts.csv`, `013_probe_rankings.csv`, `013_top3_overlap.csv` |
| Paired ordering versus AUROC | `013_paired_ordering.csv` |
| Source-level sign offsets | `013_source_sign_offsets.csv` |
| Probe signs and joint signs | `013_sign_distributions.csv`, `013_joint_sign_distributions.csv` |
| Partial Spearman | `013_partial_spearman.csv` |
| Auxiliary score distributions | `013_auxiliary_distributions.csv` |
| Benign-processing tables | `013_benign_metrics.csv` |

The scripts generate numerical inputs, not new figure designs or manuscript prose.

## 5. Verification

```bash
python -m unittest discover -s tests -v
python 014_verify_paper_results.py
```

Unit tests cover exact AUROC ties, ordering direction, paired versus full-distribution statistics, partial Spearman with tied ranks, exact source cohorts, paired-median aggregation, the squared-score identity, and explicitly unmeasured auxiliary cache cells.

The paper-specific verifier checks cohort/probe counts and the reported reversal counts, five high-clean-separation reversals, the rounded frequency-RMSE AUROC, partial Spearman values, source sign counts, and top-three probe overlap. These expected findings are **verification checks**, not inputs to the statistical calculations.

Use its `--reference-probes`, `--reference-baselines`, `--reference-medians`, `--reference-correlations`, `--reference-partial`, `--reference-domains`, `--reference-auxiliary`, `--reference-benign`, `--reference-paired`, `--reference-source-signs`, and `--reference-joint-signs` options to compare independent saved tables. The report is `result/paper/014_verification.json`.

## 6. Historical utilities

`006_ks_wasserstein.py`, `008_baseline_screening_metrics.py`, `009_screening_metrics.py`, and `011_transformation_selectivity.py` retain the earlier screening workflow for historical compatibility. They are **not part of the current paper workflow**. Current KS statistics and all current AUROCs come from 013. The old mixture branch in 010 is likewise not required.

## References

[1] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. "Moment Matching for Multi-Source Domain Adaptation." ICCV 2019. https://ai.bu.edu/DomainNet/

[2] Shawn-Shan/nightshade-release. Official Nightshade research code release. https://github.com/Shawn-Shan/nightshade-release

## License

Except for `Nightshade/`, the original code in this repository is licensed under the MIT License; see `LICENSE`. Files under `Nightshade/` remain under GPL-3.0; see `Nightshade/LICENSE` and `Nightshade/NOTICE`.
