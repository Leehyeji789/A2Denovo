# A2Denovo

A hybrid *de novo* variant (DNV) detection framework for PacBio HiFi long-read trio sequencing data.

## Overview

A2Denovo combines two complementary approaches for accurate DNV detection:

- **Assembly-based discovery**: Identifies candidate DNVs from pangenome or assembly-based variant calls, capturing variants that may be missed by traditional alignment-based methods
- **Alignment-based refinement**: Extracts read-level features from BAM alignments and applies machine learning classification to distinguish true DNVs from artifacts

This hybrid approach leverages the strengths of both paradigms—the sensitivity of assembly-based methods and the interpretability of alignment-based features—to achieve high-confidence DNV calls.

### Pipeline Steps

1. **DNV Candidate Extraction** (`extract_dnv_candidates.py`): Extract Mendelian error variants from trio VCF using Hail
2. **Feature Extraction** (`dnv_feature_builder.py`): Extract read-level and genomic context features from candidate DNV sites
3. **Model Training** (`a2denovo_train.py`): Train a logistic regression classifier with L1 feature selection and probability calibration
4. **Prediction** (`a2denovo_predict.py`): Apply the trained model to predict DNVs in new samples

## Installation

### Requirements

- Python 3.8+
- pysam
- pandas
- numpy
- scikit-learn
- tqdm
- hail (for Step 0 only)

```bash
# Clone repository
git clone https://github.com/your-username/A2Denovo.git
cd A2Denovo

# Install dependencies
pip install -r requirements.txt

# For Step 0 (DNV candidate extraction), also install Hail
pip install hail
```

### Resource Files

Download the following reference files and place them in the `resources/` directory.

#### Reference Genome
- **GRCh38 FASTA**: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/

#### Repeat & Difficult Region Annotations
- **RepeatMasker**: UCSC Table Browser (https://genome.ucsc.edu/cgi-bin/hgTables), track: RepeatMasker
- **Low-complexity regions (LCR)**: https://github.com/lh3/varcmp/raw/master/scripts/LCR-hs38.bed.gz (Li, 2014)
- **Tandem Repeat catalog**: https://zenodo.org/records/13178746 (Porubsky et al., 2025)

#### GIAB Genome Stratifications (v3.6)
Download from: https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/genome-stratifications/v3.6/GRCh38@all/
- `GRCh38_AllTandemRepeatsandHomopolymers_slop5.bed.gz` - Homopolymers and tandem repeats
- `GRCh38_segdups.bed.gz` - Segmental duplications
- `GRCh38_alldifficultregions.bed.gz` - All difficult regions combined

#### Population Frequency (Optional)
- **gnomAD v3.1 Hail Table**: https://gnomad.broadinstitute.org/downloads

## Usage

### Step 0: Extract DNV Candidates (Hail)

Extract candidate de novo variants from trio VCF based on Mendelian inheritance patterns.

```bash
python scripts/extract_dnv_candidates.py \
    --vcf trio.vcf.gz \
    --pedigree family.ped \
    --output candidates.tsv.gz \
    --gnomad /path/to/gnomad.ht \
    --reference GRCh38
```

Key parameters:
- `--vcf`: Input VCF file (bgzipped)
- `--pedigree`: Pedigree file in PED format
- `--gnomad`: gnomAD Hail Table for AF filtering (optional)
- `--af-threshold`: gnomAD AF threshold (default: 0.001)

The script filters for:
- Child heterozygous genotype
- Both parents homozygous reference (or missing)
- gnomAD allele frequency < 0.1%

## Input File Formats

### 1. Candidate DNV File (for Feature Extraction)

Output from Step 0 or user-provided TSV file. **Required columns:**

| Column | Description | Example |
|--------|-------------|---------|
| `chrom` | Chromosome | chr1 |
| `pos` | Position (1-based) | 12345 |
| `ref` | Reference allele | A |
| `alt` | Alternate allele | G |
| `id` | Sample ID (child) | NA12879 |
| `internal_AC` | Allele count in cohort | 1 |

Example:
```
chrom	pos	ref	alt	variant_ID	id	internal_AC
chr1	12345	A	G	chr1:12345:A:G	NA12879	1
chr1	67890	C	T	chr1:67890:C:T	NA12879	1
chr2	11111	G	GA	chr2:11111:G:GA	NA12882	1
```

### 2. True Variants File (for Model Training)

Validated true de novo variants for training the classifier. The file must contain variant identifiers that can be matched to candidates.

**Option A - Using `variant_ID` column:**
```
variant_ID	sample
chr1:12345:A:G	NA12879
chr1:67890:C:T	NA12879
chr2:11111:G:GA	NA12882
```

**Option B - Using `locus_38`, `ref`, `alt` columns:**
```
locus_38	ref	alt	sample	variant_type
chr1:12345	A	G	NA12879	SNV
chr1:67890	C	T	NA12879	SNV
chr2:11111	G	GA	NA12882	INS
```

The training script automatically creates `variant_ID` from `locus_38:ref:alt` if `variant_ID` column is not present.

### 3. Sample List File (for Training)

Tab-separated file with sample information. **Required columns:**

| Column | Description |
|--------|-------------|
| `subject_id` | Sample ID |
| `ROLE` | Role in trio: `child`, `father`, or `mother` |

Example:
```
subject_id	ROLE	bam_path
NA12879	child	/path/to/NA12879.bam
NA12877	father	/path/to/NA12877.bam
NA12878	mother	/path/to/NA12878.bam
NA12882	child	/path/to/NA12882.bam
NA12877	father	/path/to/NA12877.bam
NA12878	mother	/path/to/NA12878.bam
```

### 4. Trio Sample List File (for Batch Feature Extraction)

Tab-separated file for `run_feature_builder.sh`:

```
child_id	child_bam	father_id	father_bam	mother_id	mother_bam
NA12879	/path/to/NA12879.bam	NA12877	/path/to/NA12877.bam	NA12878	/path/to/NA12878.bam
NA12882	/path/to/NA12882.bam	NA12877	/path/to/NA12877.bam	NA12878	/path/to/NA12878.bam
```

### 5. Pedigree File (for Step 0)

Standard PED format:
```
family_id	sample_id	father_id	mother_id	sex	phenotype
FAM01	NA12879	NA12877	NA12878	1	0
FAM01	NA12877	0	0	1	0
FAM01	NA12878	0	0	2	0
```

## Step-by-Step Usage

### Step 1: Feature Extraction

```bash
python scripts/dnv_feature_builder.py \
    --variants-txt candidates.tsv \
    --bam-child child.bam \
    --bam-father father.bam \
    --bam-mother mother.bam \
    --fasta reference.fasta \
    --repeatmasker-cats repeatmasker.bed \
    --bed segdup=segdups.bed \
    --bed lcr=lcr.bed \
    --bed tr=tandem_repeats.bed \
    --bed homopolymer=homopolymers.bed \
    --bed GIABdifficult=difficult_regions.bed \
    --win 50 \
    --radius 1000 \
    -o output_features.tsv
```

Or use the batch script for multiple samples:
```bash
bash scripts/run_feature_builder.sh
```

### Step 2: Model Training

```bash
python scripts/a2denovo_train.py \
    --feature-dir ./features/ \
    --sample-list sample_list.txt \
    --true-variants true_dnvs.tsv \
    --output-dir ./model/ \
    --recall-target 0.90
```

Key parameters:
- `--feature-dir`: Directory containing feature TSV files
- `--sample-list`: Sample information file
- `--true-variants`: File with validated true DNV IDs
- `--recall-target`: Target recall for threshold selection (default: 0.90)

### Step 3: Prediction

```bash
python scripts/a2denovo_predict.py \
    --model-bundle ./model/final_model_bundle.pkl \
    --feature-dir ./features/ \
    --output-dir ./predictions/ \
    --filter-ac1
```

Key parameters:
- `--model-bundle`: Path to trained model bundle
- `--feature-dir`: Directory with feature files to predict
- `--filter-ac1`: Filter to AC=1 variants only (recommended)
- `--no-read-filters`: Disable post-prediction filters (not recommended)

**Note**: Post-prediction read-level filters are applied by default. These filters ensure high-confidence DNV calls by requiring proper allele support patterns.

### Using the Pre-trained Model

A pre-trained model is included in the `models/` directory. You can skip the training step and directly run prediction:

```bash
python scripts/a2denovo_predict.py \
    --model-bundle ./models/final_model_bundle_250929.pkl \
    --feature-dir ./features/ \
    --output-dir ./predictions/ \
    --filter-ac1
```

This model was trained on 7 trios from Korean families using PacBio HiFi Revio sequencing (~30x coverage) and can be used for general DNV prediction.

## Output Files

### Training Outputs
- `final_model_bundle.pkl`: Trained model with calibrator and threshold
- `logo_fold_results.csv`: Leave-one-group-out cross-validation results
- `calibration_results.npz`: Calibration data
- `feature_importance.csv`: Feature importance rankings

### Prediction Outputs
- `*_combined.csv.gz`: All predictions with both model and filtered labels
  - `pred_label_model`: Raw model prediction (before post-filtering)
  - `pred_label_filtered`: After post-filtering
  - `pred_dnv`: Final DNV call (same as filtered)
- `*_final_dnv_calls.tsv`: Final high-confidence DNV calls only
- `*_stats.txt`: Summary statistics
- `per_sample/`: Individual sample predictions (if `--save-per-sample`)

## Method Details

### Feature Categories

1. **Read-level features**
   - Allelic depth (total and alternate)
   - Mapping quality
   - Edit distance
   - Read quality scores
   - Polymerase passes
   - Signal-to-noise ratios

2. **Genomic context features**
   - GC content (±50bp window)
   - Shannon entropy
   - Overlap with repeat regions
   - Distance to nearest Mendelian error

### Model Architecture

- L1-penalized logistic regression for feature selection
- L2-regularized logistic regression for classification
- Platt scaling for probability calibration
- Leave-one-group-out cross-validation (group = trio)

### Post-prediction Filtering

Variants passing the probability threshold are further filtered:
- No alternate reads in parents
- Total depth < 1,000 in child and parents
- ≥5 reads depth in each parent
- ≥1 alternate read in child
- Allele balance 0.2–0.8 in child

## License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

- ✅ **Permitted**: Academic research, personal use, educational purposes, non-profit organizations
- ❌ **Not permitted**: Commercial use without separate licensing agreement

For commercial licensing inquiries, please contact the authors.

See [LICENSE](LICENSE) for full terms.

## Patent Notice

This software may be subject to pending patent applications. The hybrid assembly-alignment approach for de novo variant detection described herein may be covered by intellectual property protections.

## Contact

For questions or issues, please open a GitHub issue.
