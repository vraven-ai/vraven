# VRAVEN Cross-Domain Dataset Suite

This directory defines the download and preparation pipeline for the dataset
suite selected for the joint-certification paper. Raw and prepared datasets are
kept out of Git; the tracked manifests record the files, hashes and protocols
from the reference preparation run. The scientific focus is the behaviour of
different neural-network families under VRAVEN, not large-scale dataset
engineering.

No models are trained by the dataset downloader.

## Selected datasets

| # | Folder | Dataset | Domain | Modality | Primary task | Approximate scale | Candidate model families |
|---:|---|---|---|---|---|---:|---|
| 1 | `kronodroid/` | KronoDroid | Cybersecurity / Android malware | High-dimensional tabular | Binary and malware-family classification | Local combined CSV, 84 MB | MLP, 1-D CNN, residual MLP, tabular Transformer |
| 2 | `fashion_mnist/` | Fashion-MNIST | Computer vision / apparel | 28×28 greyscale images | 10-class classification | 70,000 images, about 30 MB compressed | CNN, ViT, MLP |
| 3 | `uci_har/` | Human Activity Recognition Using Smartphones | Wearable sensing | Multivariate time series and engineered features | 6-class classification | 10,299 windows, about 58 MB compressed | 1-D CNN, GRU, Transformer, MLP |
| 4 | `sms_spam/` | SMS Spam Collection | Natural language / communications | Short text | Binary classification | 5,574 messages, about 0.2 MB | Text CNN, GRU, compact Transformer |
| 5 | `wdbc/` | Breast Cancer Wisconsin (Diagnostic) | Medical diagnostics | Tabular cytology features | Binary classification | 569 samples, about 50 KB | MLP, residual MLP, tabular Transformer |
| 6 | `eeg_eye_state/` | EEG Eye State | Neurophysiology | Sequential EEG measurements | Binary classification | 14,980 measurements, about 1.6 MB | 1-D CNN, GRU, Transformer |
| 7 | `ai4i_predictive_maintenance/` | AI4I 2020 Predictive Maintenance | Industrial systems | Multivariate tabular/time-ordered records | Failure classification | 10,000 samples, about 0.5 MB | MLP, residual MLP, tabular Transformer |
| 8 | `dry_bean/` | Dry Bean | Agriculture / computer-vision-derived morphology | Tabular shape features | 7-class classification | 13,611 samples, about 4.5 MB | MLP, residual MLP, tabular Transformer |
| 9 | `bank_marketing/` | Bank Marketing | Finance / customer behaviour | Mixed tabular | Binary classification | Up to 45,211 samples, about 1 MB | MLP, entity-embedding network, tabular Transformer |
| 10 | `wine_quality/` | Wine Quality | Food science / chemistry | Tabular physicochemical features | Ordinal or multiclass quality prediction | 6,497 samples, about 0.1 MB | MLP, ordinal network, tabular Transformer |

The suite covers nine application areas and five useful input structures: image, text, tabular, sequential/time-series and high-dimensional cybersecurity features.

## Directory convention

```text
DATASETS/
├── README.md
├── download_datasets.py
├── DOWNLOAD_MANIFEST.json       # generated after download
├── prepare_training_data.py
├── TRAINING_READY_MANIFEST.json # generated after format conversion
└── <dataset>/
    ├── raw/                     # immutable downloaded/extracted source data
    └── training_ready/          # canonical CSV/NPZ files consumed by loaders
```

Future preprocessing must go into `<dataset>/processed/<protocol-version>/`. Raw files must not be edited in place. Train/validation/test splits must be written as sample identifiers, not by rewriting the raw dataset.

## Download

From the VRAVEN repository root:

```bash
python3 DATASETS/download_datasets.py
```

The script downloads from the official UCI or Fashion-MNIST source, safely extracts archives, includes the existing local KronoDroid CSV by hard link when possible, and writes SHA-256 provenance for every file. It performs no preprocessing and no training.

## Prepare model-ready formats

```bash
python3 DATASETS/prepare_training_data.py
```

This creates only the canonical files needed by model loaders: CSV for tabular/text datasets and compressed NumPy tensors for image and sensor datasets. It performs no fitted preprocessing, feature selection or training. The source archives remain under `raw/` because raw evidence and model-ready data serve different reproducibility purposes.

## Scientific use rules

- Freeze preprocessing, label mapping and splits before final experiments.
- Use held-out calibration data for VRAVEN thresholds and protected-coordinate selection.
- Preserve natural groups: subject-wise splits for UCI HAR, chronological splits where temporal claims are made, and temporal splits for KronoDroid concept-drift studies.
- Treat WDBC as a diagnostic benchmark only; do not imply clinical validation.
- State that AI4I is synthetic and do not generalise its results to real machinery without external validation.
- Wine Quality can be treated as ordinal/multiclass only after the label protocol is declared in advance.
- Confirm the provenance and redistribution licence of the existing combined KronoDroid CSV before publishing it with an artefact.

## Sources and licences

| Dataset | Authoritative source | Licence recorded by source | Citation identifier |
|---|---|---|---|
| KronoDroid | [Computers & Security article](https://doi.org/10.1016/j.cose.2021.102399) | Existing local combined file; redistribution provenance must be confirmed | DOI `10.1016/j.cose.2021.102399` |
| Fashion-MNIST | [Official Zalando Research repository](https://github.com/zalandoresearch/fashion-mnist) | MIT | arXiv `1708.07747` |
| UCI HAR | [UCI dataset 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | CC BY 4.0 | DOI `10.24432/C54S4K` |
| SMS Spam | [UCI dataset 228](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | CC BY 4.0 | DOI `10.24432/C5CC84` |
| WDBC | [UCI dataset 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | CC BY 4.0 | DOI `10.24432/C5DW2B` |
| EEG Eye State | [UCI dataset 264](https://archive.ics.uci.edu/dataset/264/eeg+eye+state) | CC BY 4.0 | DOI `10.24432/C57G7J` |
| AI4I 2020 | [UCI dataset 601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | CC BY 4.0 | DOI `10.24432/C5HS5C` |
| Dry Bean | [UCI dataset 602](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | CC BY 4.0 | DOI `10.24432/C50S4B` |
| Bank Marketing | [UCI dataset 222](https://archive.ics.uci.edu/dataset/222/bank+marketing) | CC BY 4.0 | DOI `10.24432/C5K306` |
| Wine Quality | [UCI dataset 186](https://archive.ics.uci.edu/dataset/186/wine+quality) | CC BY 4.0 | DOI `10.24432/C56S3T` |
