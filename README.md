# Protein Family Classifier (ESM)

Classify proteins into functional **families/domains** (e.g., kinases, receptors, hydrolases, transporters) using **pretrained ESM embeddings** and compact ML models. The project highlights a clean, reproducible pipeline with **UMAP visualizations**, **interpretable predictions (SHAP/LIME)**, and optional **3D structure rendering**.

<p align="left">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="https://github.com/facebookresearch/esm"><img alt="ESM" src="https://img.shields.io/badge/ESM-1b/2-3C7EBB"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

---

## Project Scope

- **Task:** multi-class classification of protein **family/domain** from amino-acid sequence.
- **Embeddings:** `esm1b_t33_650M_UR50S` (ESM-1b). Average-pooled token representations → fixed-size vector (1280-d).
- **Models:** light baselines (**LogReg**, **RandomForest**, **CatBoost**, small **MLP**).  
- **Interpretability:** **SHAP** / **LIME** on top of ESM embeddings.  
- **Visualization:** **UMAP** / t-SNE of the embedding space; **py3Dmol/nglview** (optional) for structure.  
- **Reproducibility:** conda env, seeds, MLflow runs.

> This repo is designed to be practical, resource-friendly, and portfolio-ready on a single-GPU workstation (e.g., RTX 2070).

---

## Repository Structure

```
protein-family-classifier/
├─ data/
│  ├─ raw/                # FASTA/CSV as downloaded
│  └─ processed/          # cleaned splits, labels
├─ notebooks/
│  ├─ 01_eda_and_fetch.ipynb
│  ├─ 02_esm_embeddings.ipynb
│  ├─ 03_train_and_eval.ipynb
│  └─ 04_interpret_and_visualize.ipynb
├─ src/
│  ├─ data_io.py          # load/save FASTA/CSV, gget helpers
│  ├─ embedding.py        # ESM loading + sequence → vector
│  ├─ modeling.py         # train/eval utilities
│  ├─ interpret.py        # SHAP/LIME helpers
│  └─ viz.py              # UMAP, confusion matrix, 3D hooks
├─ artifacts/
│  ├─ embeddings/         # cached .npy/.parquet
│  ├─ models/             # saved models
│  └─ figures/            # plots
├─ app/
│  └─ streamlit_app.py    # optional demo
├─ environment.yml
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ LICENSE
```

---

## Quickstart

```bash
# 1) Create/activate env (or reuse your existing esm_env)
conda create -n esm_env python=3.10 -y
conda activate esm_env

# 2) Install core deps (mixed conda+pip for stability)
conda install numpy pandas matplotlib seaborn scikit-learn umap-learn biopython plotly tqdm -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fair-esm shap lime mlflow optuna py3Dmol gseapy gget

# 3) Run notebooks (recommended flow)
jupyter notebook  # open notebooks/ in your browser
```

> ESM weights are auto-downloaded to `~/.cache/torch/hub/checkpoints/` at first run.

---

## Data

You can work with a compact curated CSV such as:
```
uniprot_id,sequence,label
P31749,MSDVEG...,kinase
Q9Y6K9,MNKHLL...,receptor
...
```
- **Labels:** functional families/domains (e.g., kinase, receptor, hydrolase, transporter, ligase, TF, peptidase…).  
- **Sources:** UniProtKB/Swiss-Prot + Pfam/InterPro annotations.  
- **Helpers:** `gget` can be used to fetch sequences/metadata programmatically.

> Place raw FASTA/CSV in `data/raw/`. Not committed if large/private (see `.gitignore`).

---

## Modeling

- Start with simple baselines (LogReg/LinearSVM).  
- Add **RandomForest**/**CatBoost**/**MLP** on top of 1280-d ESM embeddings.  
- Evaluate with **macro-F1**, **balanced accuracy**, confusion matrix, **per-class report**.  
- Log all experiments to **MLflow** (`mlruns/`).

---

## Interpretability

- **SHAP**: global feature importance on embedding dimensions; local explanations for selected proteins.  
- **LIME**: local surrogate for top predictions.  
- (Optional) Map token-level saliency by using per-residue representations instead of pooled vector.

---

## 3D Visualization (optional)

- Use **py3Dmol** to render PDB/AlphaFold models by ID:  
  ```python
  import py3Dmol
  v = py3Dmol.view(query='pdb:1BNA')
  v.setStyle({'cartoon': {'color': 'spectrum'}}); v.zoomTo(); v.show()
  ```
- For notebook widgets, ensure classic Jupyter Notebook (not JupyterLab), or configure lab widgets accordingly.

---

## Reproducibility

- Fix seeds (`numpy`, `torch`, model seeds).  
- Save ESM embeddings to disk and reuse them across runs.  
- Export environment:
  ```bash
  conda env export --no-builds > environment.yml
  ```

---

## Roadmap

- [ ] Initial dataset + labels (10–20 families)
- [ ] ESM embedding cache
- [ ] Baseline models + metrics
- [ ] UMAP plots + confusion matrices
- [ ] SHAP/LIME notebooks
- [ ] (Optional) Streamlit demo app
- [ ] README polish + GitHub release

---

## Tags

`bioinformatics` `esm` `protein-classification` `protein-family` `machine-learning` `pytorch` `embedding` `umap` `shap` `lime` `mlflow`

---

## License

MIT — see `LICENSE`.

---

## Acknowledgments

- [ESM (Meta AI)](https://github.com/facebookresearch/esm)
- UniProt / Pfam / InterPro for annotations

