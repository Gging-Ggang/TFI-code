# Code Instructions (Supplementary Materials)

**Prioritizing Candidate Images for Nightshade Inspection via Transformation-Induced Feature Instability**

## Library installation

```bash
pip install -r requirements.txt
```

## Dataset download

From the repository root, download only the data directory from Hugging Face:

```bash
hf download GgingGgang/TFI-dataset \
  --repo-type dataset \
  --include "data/**" \
  --local-dir .
```

The scripts use `data/` under the repository root by default. If the dataset is downloaded elsewhere, set `TFI_DATA_ROOT` to the downloaded `data` directory:

```bash
export TFI_DATA_ROOT=/path/to/TFI-dataset/data
```

The Hugging Face dataset already contains the clean, Nightshade, Gaussian, and LPIPS-matched Gaussian directories used by the analysis. When using that download, start from Step 3. Steps 1 and 2 are only needed to regenerate those directories; Step 1 expects the original DomainNet archives under `data/DomainNet/`.

## 1. Data sampling

Because the full DomainNet dataset is too large to submit with this supplementary package, we provide 10 sampled images from each DomainNet domain: `clipart`, `infograph`, `painting`, `quickdraw`, `real`, and `sketch`. This representative subset is much smaller than the full benchmark, so some evaluation settings, including poison prevalence ratios, are adjusted for the included data, and results may differ from the paper's full-scale results.

The samples were taken from the cleaned version of DomainNet, which is recommended by the dataset providers:

- DomainNet dataset page: https://ai.bu.edu/DomainNet/
- Cleaned data archives used:

| Domain | Data archive |
| --- | --- |
| Clipart | [clipart.zip](https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip) |
| Infograph | [infograph.zip](https://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip) |
| Painting | [painting.zip](https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip) |
| Quickdraw | [quickdraw.zip](https://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip) |
| Real | [real.zip](https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip) |
| Sketch | [sketch.zip](https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip) |

The sampling script is provided in `001_Sampling.py` and is intended for use when the original DomainNet archives are available locally. It recursively collects image files from each domain folder, sorts the file list for deterministic sampling, samples images using a fixed random seed, center-crops each image to a square, resizes it to 512 x 512, and saves the result as PNG. The number of sampled images per domain is controlled by `SAMPLES_PER_DOMAIN` in `001_Sampling.py`.

Expected input directory:

```text
data/DomainNet/
```

Expected output directory:

```text
data/D_domainnet_clean/
```

Run:

```bash
python 001_Sampling.py
```

## 2. Perturbation generation

### 2-1. Nightshade ($\textrm{NS}$)

The code in `Nightshade/` is adapted from the official Nightshade research code release, which is licensed under GPL-3.0. We used the necessary components from the official implementation with minor modifications for PNG folder input, PNG output, deterministic input ordering, and command-line perturbation settings. See `Nightshade/LICENSE` and `Nightshade/NOTICE`.

The six DomainNet domains are paired with the following target concepts.

| Source domain | Target concept |
| --- | --- |
| `clipart` | `infograph` |
| `infograph` | `clipart` |
| `painting` | `natural` |
| `quickdraw` | `sketch` |
| `real` | `painting` |
| `sketch` | `quickdraw` |

Run the following commands for the `linf` setting with epsilon `0.04`.

```bash
python Nightshade/gen_poison.py \
  --directory data/D_domainnet_clean/clipart \
  --target_name infograph \
  --outdir data/D_domainnet_nightshade_linf_004/clipart \
  --eps 0.04 \
  --constraint linf \
  --steps 500

python Nightshade/gen_poison.py \
  --directory data/D_domainnet_clean/infograph \
  --target_name clipart \
  --outdir data/D_domainnet_nightshade_linf_004/infograph \
  --eps 0.04 \
  --constraint linf \
  --steps 500

python Nightshade/gen_poison.py \
  --directory data/D_domainnet_clean/painting \
  --target_name natural \
  --outdir data/D_domainnet_nightshade_linf_004/painting \
  --eps 0.04 \
  --constraint linf \
  --steps 500

python Nightshade/gen_poison.py \
  --directory data/D_domainnet_clean/quickdraw \
  --target_name sketch \
  --outdir data/D_domainnet_nightshade_linf_004/quickdraw \
  --eps 0.04 \
  --constraint linf \
  --steps 500

python Nightshade/gen_poison.py \
  --directory data/D_domainnet_clean/real \
  --target_name painting \
  --outdir data/D_domainnet_nightshade_linf_004/real \
  --eps 0.04 \
  --constraint linf \
  --steps 500

python Nightshade/gen_poison.py \
  --directory data/D_domainnet_clean/sketch \
  --target_name quickdraw \
  --outdir data/D_domainnet_nightshade_linf_004/sketch \
  --eps 0.04 \
  --constraint linf \
  --steps 500
```

### 2-2. Gaussian Noise ($\textrm{GN}$)

`002_GN.py` creates Gaussian-noise baselines with fixed L-infinity bounds.

```bash
python 002_GN.py
```


### 2-3. LPIPS-Matched Gaussian ($\textrm{GN}^\star$)

`002_GN-star.py` creates per-image Gaussian-noise baselines whose LPIPS values are matched to the corresponding NS images.

```bash
python 002_GN-star.py
```


## 3. Image transformations

`003_apply_image_transformations.py` applies image transformations to the clean, NS, GN, and GN* datasets.

Input datasets:

- `data/D_domainnet_clean`
- `data/D_domainnet_nightshade_linf_004`
- `data/D_domainnet_gaussian_linf_004`
- `data/D_domainnet_gaussian_lpips_matched_ns_linf_004`

Transformations:

- Downscaling and upscaling: factors `0.75`, `0.6`, `0.5`, `0.4`, `0.3`, `0.25`
- Gaussian blur: sigma `0.5`, `1.0`, `2.0`, `3.0`
- JPEG compression: quality `95`, `85`, `75`, `50`

Output directory:

```text
data/Transformation/
```

Run:

```bash
python 003_apply_image_transformations.py
```


## 4. Feature extraction

`004_feature_extraction.py` extracts encoder features from the original and transformed images.

Input roots:

- `data/`
- `data/Transformation/`

Output directory:

```text
result/features/
```

Outputs:

- `feature_metadata.parquet`
- `features.float16.npy` for each enabled encoder

Run:

```bash
python 004_feature_extraction.py
```


## 5. TFI calculation

`005_calculate_tfi.py` calculates transformation-induced feature instability from the extracted features.

Input directory:

```text
result/features/
```

Output directory:

```text
result/tfi/
```

Outputs:

- `tfi.csv`
- `tfi.float32.npy`

Run:

```bash
python 005_calculate_tfi.py
```


## 6. KS and Wasserstein statistics

`006_ks_wasserstein.py` compares clean and perturbed TFI distributions using KS statistics and Wasserstein distance.

Input directory:

```text
result/tfi/
```

Output directory:

```text
result/ks_wasserstein/
```

Run:

```bash
python 006_ks_wasserstein.py
```


## 7. Perturbation magnitude

`007_calculate_perturbation_magnitude.py` calculates pixel, SSIM, and LPIPS perturbation magnitudes against the clean images.

Input roots:

- `data/`
- `result/features/feature_metadata.parquet`

Output directory:

```text
result/perturbation/
```

Run:

```bash
python 007_calculate_perturbation_magnitude.py
```


## 8. Baseline screening

`008_calculate_baseline_screening_scores.py` builds baseline screening score matrices from image transformations and SD-VAE TFI values.
`008_baseline_screening_metrics.py` evaluates the scores under domain-stratified, pooled-random, and fair-subset settings.

Because the submitted sample contains only 10 images per domain, `008_baseline_screening_metrics.py` uses `RATIOS = (0.10, 0.20)`. To change the screening ratios for a larger dataset, edit the `RATIOS` constant near the top of `008_baseline_screening_metrics.py`.

Input roots:

- `data/`
- `data/Transformation/`
- `result/tfi/`
- `result/baseline/`

Output directories:

```text
result/baseline/
result/baseline/screening/
```

Run:

```bash
python 008_calculate_baseline_screening_scores.py
python 008_baseline_screening_metrics.py
```


## 9. Screening metrics

`009_screening_metrics.py` evaluates TFI-based screening performance.

Input directory:

```text
result/tfi/
```

Output directory:

```text
result/screening/
```

Run:

```bash
python 009_screening_metrics.py
```


## 10. Screening metrics with benign transformations

`010_screening_metrics_with_benign.py` evaluates screening when benign transformations are mixed with Nightshade samples.

Input roots:

- `result/tfi/`
- `result/features/`

Output directory:

```text
result/screening/
```

Run:

```bash
python 010_screening_metrics_with_benign.py
```


## 11. Transformation selectivity

`011_transformation_selectivity.py` summarizes transformation selectivity using TFI, KS/Wasserstein statistics, and screening metrics.

Input roots:

- `result/tfi/`
- `result/ks_wasserstein/`
- `result/screening/`

Output directory:

```text
result/analysis/
```

Run:

```bash
python 011_transformation_selectivity.py
```


## 12. Probe-dependent selectivity decomposition

`012_decomposition_analysis.py` decomposes the squared TFI of a perturbed image into an interaction magnitude and a directional alignment term. For clean transformation response $c$, pre-transformation perturbation $p$, post-transformation perturbation $q$, and $\delta=p-q$, it calculates

```text
||c + delta||^2 = ||c||^2 + A + B
A = ||delta||^2
B = 2 <c, delta>
gamma = <c, delta> / (||c|| ||delta||)
```

The script source-pairs Nightshade and LPIPS-matched Gaussian images, summarizes their per-image decomposition differences for all 84 combinations of six feature extractors and 14 transformations, and relates those differences to the P@5 margin from `009_screening_metrics.py`. It also reports Spearman correlations between the NS-minus-GN-star gamma difference and the P@5 margin over all probes and within each transformation family.

The submitted 60-image sample uses `POISON_RATIO = 0.10`, matching the available setting in `009_screening_metrics.py`. For the full-data paper setting, change `POISON_RATIO` to `0.05` near the top of `012_decomposition_analysis.py` after generating the corresponding 5% screening results. Results from the submitted sample can differ from the full-data values reported in the paper.

Input roots:

- `result/features/`
- `result/tfi/`
- `result/screening/`

Output directory:

```text
result/analysis/
```

Outputs:

- `012_decomposition_per_sample.csv`
- `012_decomposition_ns_vs_gnstar_summary.csv`
- `012_decomposition_representative_probes.csv`
- `012_decomposition_correlations.csv`
- `012_decomposition_sign_counts.csv`

Run:

```bash
python 012_decomposition_analysis.py
```


## References

[1] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. "Moment Matching for Multi-Source Domain Adaptation." ICCV 2019. DomainNet dataset: https://ai.bu.edu/DomainNet/

[2] Shawn-Shan/nightshade-release. Official Nightshade research code release from University of Chicago. https://github.com/Shawn-Shan/nightshade-release

## License

Except for the `Nightshade/` directory, the original source code in this repository is licensed under the MIT License. See `LICENSE`.

The files under `Nightshade/` are adapted from the official Nightshade release and are licensed under GPL-3.0. See `Nightshade/LICENSE` and `Nightshade/NOTICE`.
