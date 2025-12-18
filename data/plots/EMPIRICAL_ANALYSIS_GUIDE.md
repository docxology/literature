# Empirical Analysis & Verifiable Elements Guide

## Overview

The enhanced code analysis script (`analyze_code_repositories.py`) can identify empirically verifiable elements in research repositories, including:

1. **Data inputs** - What data is used
2. **Analysis methods** - How data is processed
3. **Evaluation metrics** - How results are measured
4. **Reproducibility elements** - What makes results verifiable

---

## What Can Be Detected

### 1. Data Inputs

#### Data Files
The script detects data files by extension:
- **CSV/TSV**: `.csv`, `.tsv` - Tabular data
- **JSON**: `.json` - Structured data
- **NumPy**: `.npy`, `.npz` - Numerical arrays
- **HDF5**: `.h5`, `.hdf5` - Hierarchical data format
- **Pickle**: `.pkl`, `.pickle` - Python serialized objects
- **MATLAB**: `.mat` - MATLAB data files
- **Parquet/Feather/Arrow**: Columnar data formats

**What this tells us:**
- Types of data used in research
- Data volume (file counts)
- Data formats preferred

#### Data Directories
Detects common data directory names:
- `data/`, `dataset/`, `datasets/`
- `input/`, `inputs/`
- `raw_data/`, `processed_data/`
- `results/`, `outputs/`

**What this tells us:**
- Data organization patterns
- Whether data is included in repository
- Data processing pipeline structure

#### Dataset References
Finds references to external datasets:
- URLs to datasets (Zenodo, Kaggle, Google Drive)
- Dataset names in code
- HuggingFace datasets
- Standard datasets (MNIST, CIFAR, etc.)

**What this tells us:**
- Where data comes from
- Whether data is publicly available
- Data accessibility for reproduction

#### Data Loading Patterns
Detects how data is loaded in code:
- `pandas.read_csv()` - CSV loading
- `np.load()` - NumPy array loading
- `torch.load()` - PyTorch tensors
- `datasets.load_dataset()` - HuggingFace datasets
- `gym.make()` - RL environments
- `sklearn.datasets` - Scikit-learn datasets

**What this tells us:**
- Data processing workflows
- Tool preferences for data handling
- Integration patterns

---

### 2. Analysis Methods

#### Code Patterns
The script analyzes code to identify:
- **Data preprocessing**: Cleaning, normalization, feature engineering
- **Model training**: Training loops, optimization
- **Inference**: Prediction, generation
- **Visualization**: Plotting, result presentation

#### Tool Usage
Identifies which tools are actually used:
- Deep learning frameworks (PyTorch, TensorFlow, JAX)
- Scientific computing (NumPy, SciPy)
- ML libraries (scikit-learn)
- Domain-specific tools (pymdp, ROS)

**What this tells us:**
- Actual implementation vs. paper claims
- Tool integration patterns
- Code complexity

---

### 3. Evaluation Metrics

The script detects evaluation metrics mentioned in code:

#### Classification Metrics
- `accuracy`, `precision`, `recall`, `f1_score`
- `confusion_matrix`, `roc_auc`

#### Regression Metrics
- `rmse`, `mae`, `r2_score`

#### Information Theory Metrics
- `free_energy`, `variational_free_energy`
- `elbo` (Evidence Lower Bound)
- `kl_divergence`, `entropy`
- `perplexity`

#### NLP Metrics
- `bleu`, `rouge`, `meteor`

**What this tells us:**
- How results are measured
- Whether metrics match paper claims
- Evaluation rigor

---

### 4. Reproducibility Elements

#### Seed Setting
Detects random seed initialization:
- `random.seed()`
- `np.random.seed()`
- `torch.manual_seed()`
- `tf.random.set_seed()`

**What this tells us:**
- Whether results are reproducible
- Randomness control

#### Environment Files
Detects environment specification:
- `requirements.txt` - Python dependencies
- `environment.yml` / `conda.yml` - Conda environment
- `Pipfile` - Pipenv
- `Dockerfile` - Docker containerization

**What this tells us:**
- Dependency management
- Environment reproducibility
- Setup complexity

#### Configuration Files
Finds configuration files:
- `config.yaml` / `config.yml`
- `config.json`
- `config.toml`

**What this tells us:**
- Hyperparameter management
- Experiment configuration
- Reproducibility support

#### Experiment Tracking
Detects experiment tracking tools:
- **Weights & Biases (wandb)**: Experiment logging
- **TensorBoard**: Training visualization
- **MLflow**: ML lifecycle management
- **Comet**: Experiment tracking

**What this tells us:**
- Experiment management practices
- Result tracking
- Reproducibility infrastructure

---

## Empirical Verification Checklist

### ✅ Can Verify:

1. **Data Availability**
   - ✅ Data files present in repository
   - ✅ External dataset URLs accessible
   - ✅ Data loading code functional

2. **Code Functionality**
   - ✅ Dependencies specified
   - ✅ Environment reproducible
   - ✅ Code structure clear

3. **Results Reproducibility**
   - ✅ Random seeds set
   - ✅ Hyperparameters documented
   - ✅ Configuration files present

4. **Evaluation Rigor**
   - ✅ Metrics implemented
   - ✅ Evaluation code present
   - ✅ Results files generated

### ⚠️ Limitations:

1. **Data Size**
   - Large datasets may not be in repository
   - May require external download
   - Storage limitations

2. **Code Completeness**
   - Some code may be missing
   - Preprocessing steps may be undocumented
   - Results may not match paper exactly

3. **Environment Differences**
   - Hardware differences (GPU, CPU)
   - OS differences
   - Library version mismatches

---

## Usage Example

```bash
# Analyze repositories for empirical elements
python3 scripts/analyze_code_repositories.py --limit 20 --output data/plots/empirical_analysis

# Check the generated report
cat data/plots/empirical_analysis/code_analysis_summary.md

# View full analysis data
cat data/plots/empirical_analysis/code_analysis_full.json | jq '.[0].data_analysis'
```

---

## What to Look For

### High Reproducibility
- ✅ Seed setting present
- ✅ Environment files (requirements.txt, conda.yml)
- ✅ Configuration files
- ✅ Data files included or clearly referenced
- ✅ Evaluation metrics implemented
- ✅ Experiment tracking (wandb, tensorboard)

### Medium Reproducibility
- ⚠️ Some seeds set, but not all
- ⚠️ Dependencies listed, but no environment file
- ⚠️ Data referenced, but not included
- ⚠️ Metrics mentioned, but not clearly implemented

### Low Reproducibility
- ❌ No seed setting
- ❌ No environment specification
- ❌ No configuration files
- ❌ Data not accessible
- ❌ Metrics not implemented

---

## Research Insights

### Data Patterns
- **What data formats are common?** (CSV, JSON, NumPy arrays)
- **Are datasets included or external?** (affects accessibility)
- **What data loading patterns are used?** (reveals workflows)

### Analysis Patterns
- **How is data preprocessed?** (code patterns)
- **What tools are actually used?** (vs. paper claims)
- **How complex is the code?** (file counts, structure)

### Evaluation Patterns
- **What metrics are used?** (classification, regression, information theory)
- **Are metrics properly implemented?** (code vs. paper)
- **Are results saved?** (results files present)

### Reproducibility Patterns
- **How many repos set seeds?** (reproducibility rate)
- **What environment management is used?** (Docker, Conda, requirements.txt)
- **Are experiments tracked?** (wandb, tensorboard usage)

---

## Next Steps

1. **Run analysis on all repositories**
   ```bash
   python3 scripts/analyze_code_repositories.py --limit 100
   ```

2. **Compare empirical elements across papers**
   - Which papers have highest reproducibility?
   - What patterns emerge?
   - Are there domain differences?

3. **Validate paper claims**
   - Do code implementations match paper descriptions?
   - Are metrics correctly implemented?
   - Can results be reproduced?

4. **Identify best practices**
   - What makes a repository highly reproducible?
   - What tools support reproducibility?
   - What patterns should be encouraged?

---

## Output Files

### `code_analysis_summary.md`
Human-readable summary with:
- Data file counts
- Data loading patterns
- Evaluation metrics
- Reproducibility elements
- Experiment tracking tools

### `code_analysis_full.json`
Complete analysis data with:
- Detailed file counts
- All dataset references
- All data loading patterns
- Complete metric lists
- Full reproducibility information

---

## Questions This Analysis Can Answer

1. **Data Questions**
   - What types of data are used?
   - Where does data come from?
   - Is data accessible?

2. **Analysis Questions**
   - How is data processed?
   - What tools are actually used?
   - How complex is the implementation?

3. **Evaluation Questions**
   - What metrics are used?
   - Are metrics correctly implemented?
   - Are results saved?

4. **Reproducibility Questions**
   - Can results be reproduced?
   - What's needed to reproduce?
   - Are best practices followed?

---

## Summary

The enhanced code analysis provides **empirically verifiable insights** into:

- ✅ **Data inputs** - What data is used and where it comes from
- ✅ **Analysis methods** - How data is processed and analyzed
- ✅ **Evaluation metrics** - How results are measured
- ✅ **Reproducibility** - What makes results verifiable

This enables **evidence-based assessment** of research reproducibility and implementation quality.

