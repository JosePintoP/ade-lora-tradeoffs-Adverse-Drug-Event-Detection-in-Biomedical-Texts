# Full Fine-Tuning vs. LoRA for Adverse Drug Event Detection

Source code for the paper *"Performance–Efficiency Trade-Offs of Full Fine-Tuning and Low-Rank
Adaptation for Adverse Drug Event Detection in Biomedical Texts"* by Rosa L. Figueroa Iturrieta,
José I. Pinto, Miguel E. Figueroa Toro and Christopher A. Flores.

BERT (`bert-base-cased`) and BioBERT (`dmis-lab/biobert-base-cased-v1.1`) are adapted with full
fine-tuning and with LoRA on ADE-Corpus V2, for two tasks:

- sentence classification (does the sentence report an adverse drug event?);
- named entity recognition (one BIO tagger for drugs and one for adverse effects).

Predictive performance, training time and peak GPU memory are compared over six seeds (42–47).

## Repository structure

```
scripts/    pipeline (run from the repository root)
docs/       SHA-256 hashes of the processed splits used in the paper
results/    aggregated results reported in the paper (no corpus text)
data/       created by the scripts, not versioned
outputs/    created by the scripts, not versioned
```

| Script | Role |
|---|---|
| `02_preparar_datos.py` | Downloads ADE-Corpus V2, builds the BIO tags and the PMID-grouped splits |
| `03_exploracion.py` | Verifies the splits and writes descriptive statistics; shared helpers |
| `04_baseline_supervisado.py` | Full fine-tuning; shared training loop |
| `05_comparaciones_errores.py` | Statistical tests and error analysis; shared helpers |
| `06_lora_peft.py` | LoRA learning-rate selection and LoRA training |
| `07_paper_results.py` | Tables III–V and Figs. 3–6 of the paper |
| `make_table1.py` | Table I (corpus statistics by split) |
| `wordpiece_stats.py` | Sentence lengths and truncation counts |

## Data

The corpus is **not redistributed** here. Script 02 downloads `DRUG-AE.rel` and `ADE-NEG.txt` from
<https://github.com/trunghlt/AdverseDrugReaction> into `data/raw/`, and never modifies them. If you
use the corpus, cite its authors (Gurulingappa et al.; see the reference list of the paper).

## Requirements

- Python 3.11 (the paper used 3.11.17); exact package versions are listed in `requirements.txt`
- An NVIDIA GPU with CUDA. The paper used a GeForce RTX 2050 (4 GB) with 8 GB of RAM.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
# 1) install PyTorch with CUDA 12.6 support, as used in the paper
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
# 2) install the remaining dependencies
pip install -r requirements.txt
```

## Reproducing the paper

All commands are run from the repository root.

### 1. Data

```bash
python scripts/02_preparar_datos.py --skip bc5cdr cwlc
python scripts/03_exploracion.py --skip bc5cdr cwlc --no-baselines --expected-manifest docs/expected_manifest_ade.json
python scripts/make_table1.py
python scripts/wordpiece_stats.py data/processed/ade bert-base-cased --max-length 128
```

The second command stops with an error if your splits differ from those of the paper
(train/validation/test: 14,148 / 1,572 / 5,240 sentences).

### 2. Full fine-tuning

```bash
python scripts/04_baseline_supervisado.py --corpora ade --model-en bert-base-cased --seeds 42 43 44 45 46 47
python scripts/04_baseline_supervisado.py --corpora ade --model-en dmis-lab/biobert-base-cased-v1.1 --seeds 42 43 44 45 46 47
```

### 3. LoRA: learning-rate selection (validation set, seed 42)

The grid must be passed explicitly; the default of the script has fewer values.

```bash
python scripts/06_lora_peft.py --stage grid --corpora ade --model-en bert-base-cased --lr-grid 1e-4 3e-4 5e-4 1e-3 3e-3
python scripts/06_lora_peft.py --stage grid --corpora ade --model-en dmis-lab/biobert-base-cased-v1.1 --lr-grid 1e-4 3e-4 5e-4 1e-3 3e-3
```

### 4. LoRA: final runs

```bash
python scripts/06_lora_peft.py --stage final --corpora ade --model-en bert-base-cased --lr-grid 1e-4 3e-4 5e-4 1e-3 3e-3 --seeds 42 43 44 45 46 47
python scripts/06_lora_peft.py --stage final --corpora ade --model-en dmis-lab/biobert-base-cased-v1.1 --lr-grid 1e-4 3e-4 5e-4 1e-3 3e-3 --seeds 42 43 44 45 46 47
```

Finished runs are cached and skipped if a command is repeated.

### 5. Tables and figures

```bash
python scripts/07_paper_results.py
```

| Paper item | Output |
|---|---|
| Table I | `outputs/tables/table1_ade.csv` / `.tex` |
| LoRA learning rates (Table II) | `outputs/tables/peft_lr_selection.csv` |
| Tables III, IV and V | `outputs/tables/paper/` |
| Figs. 3–6 | `outputs/figures/paper/` |

Figures 1 and 2 are diagrams that were drawn manually.

## Notes on reproducibility

- Tables III and IV and Figs. 3–6 correspond to one representative run (seed 42). Table V uses the
  six seeds.
- Seeds and data splits are fixed, but GPU kernels are not bit-wise deterministic, so metrics can
  differ slightly between executions.
- Training time and peak memory depend on the hardware. Measure them with no other process using
  the GPU.
- The LoRA learning rate is selected on the validation set with one seed; the test set is not used
  for that choice.
- Some scripts also support other corpora (BC5CDR, Chilean Waiting List) that are **not** part of
  this paper. The options `--skip bc5cdr cwlc` and `--corpora ade` leave them out.
- Parts of the code comments and console messages are in Spanish.

## Citation

The paper is under review; the full reference will be added upon publication. Until then, see
`CITATION.cff`.

## Funding

ANID BECAS/DOCTORADO NACIONAL, Grant 21240959; Universidad de Concepción, VRID Project
2026001612INV; Universidad del Bío-Bío, FAPEI Project FP2629439.
