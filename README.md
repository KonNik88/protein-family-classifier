# ESM Protein Family Classifier 🧬

Classify proteins into functional **families** (kinases, receptors, hydrolases, transporters, etc.)
using **pretrained ESM-1b embeddings** and compact ML models.

The goal of this repo is to provide a **clean, reproducible, and interpretable** pipeline that can run on a
single-GPU workstation and still look good on GitHub / in a portfolio.

<p align="left">
  <!-- Core tech -->
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="https://github.com/facebookresearch/esm"><img alt="ESM" src="https://img.shields.io/badge/ESM-1b-3C7EBB"></a>
  <a href="https://mlflow.org/"><img alt="MLflow" src="https://img.shields.io/badge/MLflow-tracking%20enabled-0194E2?logo=mlflow&logoColor=white"></a>
  <a href="https://optuna.org/"><img alt="Optuna" src="https://img.shields.io/badge/Optuna-hyperparam%20search-7F52FF"></a>
  <a href="https://jupyter.org/"><img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white"></a>
  <!-- Meta -->
  <a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

---

## 🔎 Project at a glance

> From raw UniProt sequences to interpretable, structure-aware protein family classification.

- **Input:** amino-acid sequence (UniProt, 50–1000 aa).
- **Representation:** frozen **ESM-1b** embeddings (1280-d per protein).
- **Models:** Logistic Regression, RandomForest, MLP, CatBoost.
- **Tracking:** hyperparameter search with **Optuna**, experiments logged to **MLflow**.
- **Interpretability:**
  - **UMAP** projection of the embedding space,
  - **SHAP** explanations on top of ESM embeddings,
  - one **3D structure case study** per family via **py3Dmol** (real PDB/CIF).

Conceptually, the pipeline looks like this:

```text
UniProt sequence
      ↓
  ESM-1b encoder  →  1280-d embedding
      ↓
  ML classifier (LogReg / RF / MLP / CatBoost)
      ↓
Prediction + UMAP + SHAP + 3D structure (py3Dmol)
```

---

## Project Overview

- **Task:** multi-class classification of protein **functional family** from amino-acid sequence.
- **Embeddings:** `esm1b_t33_650M_UR50S` (ESM-1b). Token representations are mean-pooled into a fixed-size
  **1280-d vector** per protein.
- **Dataset:** 4264 curated proteins from UniProt (10 functional classes). Sequences are limited to 50–1000 aa
  to satisfy ESM-1b constraints.
- **Models:** light ML baselines on top of frozen ESM embeddings:
  - Logistic Regression
  - RandomForest
  - small MLP
  - CatBoost
- **Interpretability:** UMAP projections of the embedding space + SHAP on top of the final classifier.
- **3D structures:** optional **py3Dmol** case studies for one protein per family, using real PDB entries.
- **Experiment tracking:** all model runs logged with **MLflow**.

> Everything is designed to fit comfortably on a single GPU (e.g. RTX 2070) and be easy to re-run.

---

## Repository Structure

Actual layout used in this project:

```bash
ESM/
├─ data/
│  ├─ raw/                # raw UniProt exports / CSV
│  ├─ processed/          # cleaned metadata / filtered tables
│  └─ structures/         # optional PDB/CIF files for 3D case studies
│
├─ artifacts/
│  ├─ embeddings/         # cached ESM-1b embeddings (.npy, .csv)
│  ├─ models/             # trained models (joblib / CatBoost .cbm)
│  └─ figures/            # UMAP plots, confusion matrices, SHAP plots
│
├─ logs/
│  └─ mlruns/             # MLflow tracking data
│
├─ notebooks/
│  ├─ 01_build_dataset.ipynb            # build curated dataset from UniProt
│  ├─ 02_eda_and_fetch.ipynb            # EDA, label/family checks
│  ├─ 03_esm_embeddings.ipynb           # computing & caching ESM-1b embeddings
│  ├─ 04_train_and_eval.ipynb           # model training, Optuna, MLflow logging
│  └─ 05_interpret_and_visualize.ipynb  # UMAP, SHAP, 3D py3Dmol case studies
│
├─ src/
│  ├─ __init__.py
│  ├─ config.py              # paths, constants, class list, MLflow setup
│  ├─ data_uniprot.py        # download / load UniProt data
│  ├─ eda_utils.py           # helper plots for EDA (families, lengths, etc.)
│  ├─ esm_embed.py           # ESM-1b loading + sequence → embedding
│  ├─ train.py               # training helpers (metrics, Optuna loops, logging)
│  ├─ interpret.py           # SHAP utilities on top of sklearn / CatBoost models
│  └─ viz.py                 # UMAP / PCA / t-SNE and confusion matrix plotting
│
├─ README.md
├─ requirements.txt
├─ .gitignore
└─ LICENSE
```

All absolute paths in the notebooks are defined via `src/config.py` and can be adapted to your machine.

---

## Installation

You can use either `conda` or pure `pip`. A minimal example with conda:

```bash
# 1) Create and activate environment
conda create -n esm_env python=3.10 -y
conda activate esm_env

# 2) Install dependencies
pip install -r requirements.txt
```

> PyTorch wheels for CUDA may depend on your system. If needed, install PyTorch separately
> using the command recommended on https://pytorch.org/get-started/locally/

At first run, ESM weights will be automatically downloaded to your Torch cache directory.

---

## 🔁 How to reproduce the results

Below is the exact flow used in this repo.  
You can either reuse the precomputed embeddings or recompute everything from scratch.

### 1. Clone the repo

```bash
git clone https://github.com/KonNik88/protein-family-classifier.git
cd protein-family-classifier
```

### 2. Create and activate environment

Using conda (recommended):

```bash
conda create -n esm_env python=3.10 -y
conda activate esm_env

pip install -r requirements.txt
```

> If needed, install a CUDA-compatible PyTorch build separately following the instructions from https://pytorch.org/

### 3. (Option A) Use precomputed embeddings

If you already have:

- `artifacts/embeddings/esm1b_embeddings_small_maxlen1000.npy`
- `artifacts/embeddings/metadata_small_maxlen1000.csv`

you can skip directly to **Step 5**.

### 3. (Option B) Rebuild dataset from UniProt (optional)

Open and run:

- `notebooks/01_build_dataset.ipynb` – assemble & clean the protein dataset.
- `notebooks/02_eda_and_fetch.ipynb` – sanity checks, distributions, family balance.

This will populate `data/processed/` with the curated metadata.

### 4. Compute ESM-1b embeddings

Run:

- `notebooks/03_esm_embeddings.ipynb`

This notebook:

- loads sequences and metadata,
- runs ESM-1b once over all proteins,
- saves embeddings to `artifacts/embeddings/esm1b_embeddings_small_maxlen1000.npy`,
- saves aligned metadata to `artifacts/embeddings/metadata_small_maxlen1000.csv`.

### 5. Train and evaluate models

Run:

- `notebooks/04_train_and_eval.ipynb`

This notebook:

- performs a stratified train/val/test split,
- tunes hyperparameters with Optuna for:
  - Logistic Regression
  - RandomForest
  - MLP
  - CatBoost
- logs all runs and metrics to **MLflow**,
- saves the best models to `artifacts/models/`.

To inspect MLflow logs:

```bash
mlflow ui --backend-store-uri file://$(pwd)/logs/mlruns
```

and open the printed URL in your browser.

### 6. Interpret and visualize

Run:

- `notebooks/05_interpret_and_visualize.ipynb`

This notebook:

- builds **UMAP** projections of the ESM embedding space,
- analyzes the final classifier with **SHAP**,
- selects one representative protein per family,
- fetches real PDB/CIF structures via PDBe/RCSB,
- renders 3D folds for each family with **py3Dmol**.

At the end you get:

- a global picture of how ESM organizes protein families in latent space,
- class-wise and per-sample explanations,
- a structural view (3D) for each functional family.

---

## Dataset

The working dataset after preprocessing:

- **N = 4264** proteins
- **10 functional families:**

  - kinase (500)
  - transporter (499)
  - ligase (495)
  - chaperone (490)
  - transcription (484)
  - hydrolase (445)
  - ion_channel (420)
  - receptor (418)
  - protease (356)
  - dna_binding (157)

- Sequence length filtered to **50–1000 aa**.
- Metadata stored in `artifacts/embeddings/metadata_small_maxlen1000.csv`
  and aligned with the embeddings in `esm1b_embeddings_small_maxlen1000.npy`.

The raw UniProt exports are not committed to Git (see `.gitignore`). Instead, the processed metadata
and embeddings are treated as **reproducible artifacts**.

---

## Modeling & Results

We train several models on frozen ESM-1b embeddings (1280-d per protein) with a stratified
train/val/test split (70/15/15). Hyperparameters are tuned with **Optuna**, and all runs are logged to **MLflow**.

### Models

- **Logistic Regression** (multinomial, `lbfgs`, tuned `C`)
- **RandomForestClassifier** (tuned depth, estimators, etc.)
- **MLPClassifier** (1–2 hidden layers, tuned size & learning rate)
- **CatBoostClassifier** (GPU-ready gradient boosting)

### Test set performance

Approximate test metrics (macro-averaged):

| Model              | Test accuracy | Test F1 (macro) |
|--------------------|---------------|------------------|
| LogisticRegression | **0.80**      | **0.77**         |
| MLP (2-layer)      | 0.79          | 0.77             |
| RandomForest       | 0.79          | 0.74             |
| CatBoost           | 0.78          | 0.73             |

Key observations:

- ESM-1b embeddings are so informative that a **simple linear model (LogReg) generalizes best**.
- More complex models (RF, CatBoost, MLP) tend to overfit train/val and do **not** improve test F1.
- The hardest class is **dna_binding** (smallest support and high intra-class variability).

---

## Interpretability & Visualization

All interpretability lives in `05_interpret_and_visualize.ipynb` and `src/interpret.py` / `src/viz.py`.

### 1. UMAP

- We embed all 4264 protein embeddings into 2D with **UMAP**.
- Plots are colored by:
  - functional family
  - dataset split (train / val / test)

This reveals clear clusters for most families (kinase, ligase, chaperone, transcription) and
more diffuse regions for heterogeneous groups (hydrolase, protease, dna_binding).

### 2. SHAP on ESM embeddings

- We build a SHAP explainer for the final Logistic Regression model.
- Global importance: mean |SHAP| across samples and classes highlights the most influential
  directions in the ESM-1b embedding space.
- Per-class analysis: SHAP summary plots for selected classes (e.g. kinase vs dna_binding)
  show different latent patterns used by the classifier.

### 3. Case studies

- One representative protein per family is selected from the test set.
- For each case we inspect:
  - model prediction and probability vector,
  - position in the UMAP space,
  - SHAP profile.

This links **global clusters** to individual examples.

---

## 3D Structural Case Studies (py3Dmol)

For each functional family, we automatically map UniProt IDs to PDB entries (via PDBe API) and
download one representative structure (PDB or CIF). Examples include:

- chaperone → 8SHG
- dna_binding → 6FY5
- hydrolase → 4IGD
- ion_channel → 9VEC
- kinase → 3SMS
- ligase → 4XTV
- protease → 4KSK
- receptor → 3H8N
- transcription → 7EGB
- transporter → 8HBV

These structures are rendered in notebooks using **py3Dmol**:

```python
import py3Dmol
view = py3Dmol.view(query="pdb:3SMS", width=600, height=400)
view.setStyle({"cartoon": {"color": "spectrum"}})
view.zoomTo()
view.show()
```

This provides a qualitative link between **model predictions**, **ESM embedding geometry**, and
**real 3D folds** of proteins from each family.

---

## Reproducibility

- All data splits and models use fixed random seeds.
- ESM embeddings are cached in `artifacts/embeddings/` and reused across runs.
- Training/validation/test metrics and hyperparameters are logged to **MLflow** under
  experiments such as `esm_protein_family_baselines`.

To export your environment:

```bash
conda env export --no-builds > environment.yml
```

---

## Roadmap / Possible Extensions

- [ ] Token-level attribution (per-residue importance) using per-token ESM representations.
- [ ] Sequence motif discovery for top-contributing proteins in each family.
- [ ] Simple API or Streamlit app for interactive exploration.
- [ ] Additional foundation models (e.g. ESM-2, ProtT5) for comparison.

---

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## Acknowledgements

- [ESM (Meta FAIR)](https://github.com/facebookresearch/esm) for pretrained protein language models.
- [UniProt](https://www.uniprot.org/), [Pfam](https://pfam.xfam.org/), [InterPro](https://www.ebi.ac.uk/interpro/)
  for protein annotations.
- [PDBe](https://www.ebi.ac.uk/pdbe/) and [RCSB PDB](https://www.rcsb.org/) for structural data.
