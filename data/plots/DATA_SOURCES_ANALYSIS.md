# Data Sources Analysis - What Can Be Detected

## Yes! The script can find where data comes from

The enhanced `analyze_code_repositories.py` script now includes comprehensive data source detection that identifies:

---

## 1. Data Source URLs

### Detected Sources:

#### **Zenodo** (Research data repository)
- Example: `https://zenodo.org/record/123456`
- Common for: Published research datasets, supplementary materials

#### **Kaggle** (Data science platform)
- Example: `https://www.kaggle.com/datasets/...`
- Common for: Competition datasets, public datasets

#### **Google Drive** (Cloud storage)
- Example: `https://drive.google.com/file/d/...`
- Common for: Shared datasets, large files

#### **GitHub/GitLab** (Code repositories)
- Example: `https://github.com/user/dataset-repo`
- Common for: Code + data repositories, versioned datasets

#### **UCI ML Repository** (Classic ML datasets)
- Example: `https://archive.ics.uci.edu/ml/datasets/...`
- Common for: Standard benchmark datasets

#### **HuggingFace** (ML model & dataset hub)
- Example: `https://huggingface.co/datasets/...`
- Common for: NLP datasets, preprocessed data

#### **Other Sources**
- Dropbox, Figshare, Data.gov, OpenData portals
- Custom data hosting services
- Research institution data repositories

---

## 2. Dataset Names

The script identifies dataset names mentioned in code:
- **Standard datasets**: MNIST, CIFAR, ImageNet, COCO
- **Domain-specific**: Custom dataset names
- **File references**: Local data file names

**Example detection:**
- `"MNIST"` → Standard ML dataset
- `"data/SNLS80mV.mat"` → Local data file
- `"my_custom_dataset"` → Custom dataset

---

## 3. Data Citations (DOIs)

Detects Digital Object Identifiers (DOIs) for datasets:
- Format: `DOI: 10.1234/example.doi`
- Indicates: Published, citable datasets
- Provides: Permanent reference to data source

---

## 4. Download Scripts

Identifies scripts that download data:
- Files named: `download_data.py`, `fetch_data.sh`, `get_data.py`
- Contains: `wget`, `curl`, `urllib`, `requests` calls
- Shows: How data is obtained programmatically

---

## 5. README Mentions

Extracts data source information from README files:
- Data sections in README
- Dataset descriptions
- Download instructions
- Data citation information

---

## 6. Data Directory README Files

Finds README files in data directories:
- `data/README.md`
- `datasets/README.md`
- Often contains: Data source information, download instructions

---

## What This Tells Us

### Data Accessibility
- ✅ **Public datasets**: URLs to Zenodo, Kaggle, UCI → Easy to access
- ⚠️ **Private/shared**: Google Drive, Dropbox → May require access
- ❌ **Missing**: No URLs found → Data may not be accessible

### Data Provenance
- ✅ **DOI citations**: Published, citable datasets
- ✅ **Standard datasets**: Well-known benchmarks (MNIST, CIFAR)
- ⚠️ **Custom datasets**: May be harder to obtain

### Reproducibility
- ✅ **Download scripts**: Automated data acquisition
- ✅ **README instructions**: Clear data access instructions
- ❌ **Missing info**: Data source unclear

---

## Example Output

From the analysis, you'll see sections like:

```markdown
### Data Sources & Origins

#### Data Source URLs

**Total data source URLs found:** 3

**GitHub/GitLab** (2 URLs):
- https://github.com/biaslab/IWAI2020-onlinesysid/issues
- https://github.com/biaslab/ForneyLab.jl

**Other** (1 URLs):
- https://iwaiworkshop.github.io/

#### Dataset Names Referenced

Found 1 unique dataset names:
- data/SNLS80mV.mat

#### Data Citations (DOIs)

No data citations (DOIs) detected.

#### Data Download Scripts

No download scripts detected.
```

---

## How to Use This Information

### 1. **Assess Data Accessibility**
```bash
# Check which repos have accessible data sources
python3 scripts/analyze_code_repositories.py --limit 50
# Look for "Data Source URLs" section in output
```

### 2. **Find Reproducible Research**
- Repos with: URLs + Download scripts + README instructions = **Highly reproducible**
- Repos with: Only URLs = **Moderately reproducible**
- Repos with: No data source info = **Low reproducibility**

### 3. **Identify Data Patterns**
- Which data sources are most common?
- Are datasets publicly available?
- Do papers cite their data sources?

### 4. **Validate Paper Claims**
- Does code match paper's data description?
- Are data sources accessible?
- Can data be downloaded?

---

## Limitations

### What May Not Be Detected:
1. **Embedded data**: Data included directly in code (base64, hardcoded)
2. **API data**: Data fetched from APIs without clear URLs
3. **Generated data**: Data created during runtime
4. **Private data**: Data sources not mentioned in public code

### What Requires Manual Checking:
1. **URL accessibility**: URLs may be broken or require authentication
2. **Data completeness**: URLs may point to partial data
3. **Data versions**: May not detect which version of dataset is used

---

## Best Practices Detected

### ✅ High Quality Repositories:
- Clear data source URLs in README
- Download scripts for automated access
- DOI citations for datasets
- Data directory README files

### ⚠️ Medium Quality:
- URLs mentioned in code comments
- Dataset names but no URLs
- Partial documentation

### ❌ Low Quality:
- No data source information
- Data files present but no source info
- Unclear data provenance

---

## Summary

**Yes, the script can find where data comes from!** It detects:

1. ✅ **Data source URLs** (Zenodo, Kaggle, Google Drive, GitHub, etc.)
2. ✅ **Dataset names** (MNIST, CIFAR, custom datasets)
3. ✅ **Data citations** (DOIs)
4. ✅ **Download scripts** (automated data acquisition)
5. ✅ **README mentions** (documentation)
6. ✅ **Data directory READMEs** (data-specific docs)

This enables **empirical verification** of:
- Data accessibility
- Data provenance
- Reproducibility potential
- Paper claims validation

---

## Next Steps

1. **Run full analysis**:
   ```bash
   python3 scripts/analyze_code_repositories.py --limit 100
   ```

2. **Analyze data source patterns**:
   - Which sources are most common?
   - Are datasets publicly accessible?
   - Do papers cite their data?

3. **Assess reproducibility**:
   - How many repos have accessible data?
   - What's the data accessibility rate?
   - Are there domain differences?

4. **Validate claims**:
   - Compare paper data descriptions with code
   - Check if data sources match claims
   - Identify missing or inaccessible data

