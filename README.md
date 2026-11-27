# class-site-design

Code for two studies on multi-site medical AI:

1. **Data source identity can inflate the pooled AUC of chest radiograph AI models** (paper 1).
2. **Before training: a graph-based design check for multi-site medical AI studies** (paper 2).

When a disease is collected at different rates by different hospitals, scanners or datasets, a model can learn the
source instead of the disease. Both studies represent a multi-site design as a bipartite graph linking each class to
the sites where it was observed. From case counts alone, this graph tells whether disease and site can be separated,
how precisely, how much of a pooled AUC the site alone can achieve, which departures from the additive site model can
be detected, and where additional samples would be most useful.

## Quick start: check a design from counts

`src/designcheck.py` needs only a CSV of case counts per class and site (no images, no model):

```bash
cd src
python designcheck.py ../examples/nih_chexpert_counts.csv --positive effusion
python designcheck.py ../examples/ham10000_counts.csv --positive melanoma --budget 2000 --step 100
```

It reports connectivity (estimability), the effective sample size `1/R_eff` and the design efficiency for every pair
of classes, the number of independent cycles, and, for two classes, the share of within-site positive–negative pairs,
the pooled AUC achievable by the site alone, and a greedy allocation of additional samples.

## Repository structure

```
class-site-design/
├── README.md
├── LICENSE
├── requirements.txt
├── examples/                  case counts of the NIH + CheXpert and HAM10000 designs
├── data_preparation/
│   ├── prepare_datasets.py    builds one .npz file per dataset (cohort selection and representations)
│   └── csdprep/               dataset adapters (NIH ChestX-ray14, CheXpert, HAM10000, PTB-XL, ABIDE) and feature extraction
└── src/
    ├── config.py              data paths (set with environment variables)
    ├── designcheck.py         stand-alone design check
    ├── *.py                   analysis scripts (see the tables below)
    ├── *.json, *.jsonl, *.csv results of the runs reported in the papers
    └── scores_e2e_es/         per-image test scores of the fine-tuned models (used by e2e_decompose.py)
```

## Installation

```bash
conda create -n csd python=3.11
conda activate csd
pip install -r requirements.txt
```

The experiments were run with Python 3.11, PyTorch 2.2.2 and torchvision 0.17.2 on an Apple M3 Pro (MPS); CUDA GPUs
are used automatically when available.

## Data

The datasets are public but cannot be redistributed. Obtain them from their distributors under their data-use
agreements and place them in one folder (`CSD_DATA_ROOT`) with the following sub-folders:

| Sub-folder | Dataset | Source |
| --- | --- | --- |
| `NIH ChestX/` | NIH ChestX-ray14 (images and `Data_Entry_2017.csv`) | NIH Clinical Center |
| `chexpert/` | CheXpert-v1.0-small (`train.csv`, `valid.csv`, images) | Stanford AIMI |
| `HAM10000/` | HAM10000 (images and `HAM10000_metadata`) | Harvard Dataverse |
| `PTB-XL/` | PTB-XL (`ptbxl_database.csv`, `scp_statements.csv`, records) | PhysioNet |
| `ABIDE/` | ABIDE I and II preprocessed slices and phenotypic files | ABIDE / public copy |
| `WILDS/camelyon17_v1.0/` | Camelyon17 metadata (`metadata.csv`) | WILDS |

Then set the paths and prepare the datasets:

```bash
export CSD_DATA_ROOT=/path/to/raw/datasets
export CSD_DATASETS_DIR=/path/to/prepared            # where the .npz files are written and read
python data_preparation/prepare_datasets.py --root "$CSD_DATA_ROOT" --out "$CSD_DATASETS_DIR"
cd src
python cohort_funnel.py          # verifies the inclusion rules and reports the selection funnel
python cache_cxr.py              # 224 x 224 radiograph cache for end-to-end training (about 2 GB)
```

The inclusion rules (Supplemental Table S1 of paper 2) are: NIH ChestX-ray14, effusion (other findings allowed) versus
“No Finding”; CheXpert, frontal views, pleural effusion positive versus “No Finding” with uncertain effusion labels
excluded; whole patients drawn at random with a fixed seed until 20,000 images per dataset; HAM10000, melanoma versus
nevus; PTB-XL, records whose diagnostic superclass is exactly MI or exactly NORM; ABIDE, autism versus typical control.

## Reproducing paper 1 (data source identity and pooled AUC)

All commands are run from `src/`. Running times are for the hardware above.

| Result | Command | Output |
| --- | --- | --- |
| Selection funnel (Fig S1), demographics (Table 1) | `python cohort_funnel.py`; `python paper1_ci_demographics.py` | `cohort_funnel.md`, `paper1_demographics.txt` |
| Pooled-AUC decomposition with 95% CIs, frozen features (Table 2, Fig 2) | `python paper1_ci_demographics.py` (10–20 min) | `paper1_ci_results.csv`, `paper1_ci_per_replicate.csv` |
| ComBat at inference uses no test label (Methods) | `python audit_p1.py combat` | printed |
| Fine-tuned models, natural pool (Table 2) | `python e2e_cxr.py --exp natural --reps 10`; `python e2e_arbn.py --exp natural --reps 10 --methods 'AR-BN'`; `python e2e_decompose.py` | `results_e2e_es_natural.jsonl`, `scores_e2e_es/` |
| Confounded training, fine-tuned (Table 3, Fig 3) | `python e2e_cxr.py --exp constructed --reps 10`; `python e2e_arbn.py --exp constructed --reps 10` | `results_e2e_es_constructed.jsonl` |
| Frozen ResNet-18 features of the same images | `python p2_experiments.py r18` | `cxr_r18_frozen.npy` (not distributed) |
| Frozen versus fine-tuned, paired tests (Table 3) | `python frozen_vs_e2e.py` (about 20 min) | printed |
| Anchor-referenced CORAL versus ComBat, frozen features | `python coral_vs_combat_10.py` (about 15 min) | printed |
| Site-only AUC versus prevalence difference (Fig S2) | analytic; `designcheck.py` computes it for any design | — |

The end-to-end runs take several hours each; the scripts resume after an interruption.

## Reproducing paper 2 (design check)

| Result | Command | Output |
| --- | --- | --- |
| Resistance law, nonlinear site effects (Table 2) | `python graph_resistance.py` | `graph_resistance_results.json` |
| Variance bound on random designs (Table 2) | `python lower_bound_check.py` | printed |
| Misspecification bias and cycle test in simulation (Table 2) | `python bias_cycles.py` | `bias_cycles_results.json` |
| Site-swapped classification, bridge size (Table 2) | `python feasibility.py`; `python sweeps.py` | `feasibility_results.json`, `sweeps_results.json` |
| Real-image stress tests, frozen ResNet-50 (Table 3) | `python semireal_ham.py`; `python semireal_cxr.py`; `python fond_baseline.py ham`; `python fond_baseline.py cxr`; `python corrected_stats.py` | `semireal_*_results.json`, `fond_*_results.json` |
| Design check of six public sources (Table 4, Fig 2) | `python natural_design_audit.py`; `python audit_p1.py patients`; `python camelyon_units.py` | printed |
| Cycle test with patient-level bootstrap (Table 5) | `python p2_experiments.py cyclecal` | printed |
| Allocation, planning and held-out validation (Table 6, Fig 3) | `python allocation.py`; `python p2_experiments.py alloc` | printed |

## Notes

- Data and data-derived files (images, image caches, embeddings, `.npz` files) are excluded from the repository
  (`.gitignore`) because the datasets cannot be redistributed.
- `fond_baseline.py` reproduces the FOND objective from the authors’ public implementation (Kaai et al., IEEE TPAMI
  2025); please cite their work when using it.
- The analysis scripts import each other (`audit_p1.py`, `semireal_cxr.py` and `p2_experiments.py` provide shared
  functions) and must be run from `src/`.

## Citation

If you use this code, please cite the corresponding paper(s). Citation details will be added on publication.

## License

MIT License (see `LICENSE`).
