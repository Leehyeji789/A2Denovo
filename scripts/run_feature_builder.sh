#!/usr/bin/env bash
#===============================================================================
# A2Denovo Feature Extraction Pipeline
#
# This script extracts features from candidate DNV sites for multiple samples
# in parallel. It requires:
#   - A sample list file with trio information
#   - Candidate DNV file (gzipped TSV)
#   - Reference files (FASTA, BED annotations)
#
# Usage:
#   bash run_feature_builder.sh
#
# Before running, edit the "CONFIGURATION" section below.
#===============================================================================
set -euo pipefail

#===============================================================================
# CONFIGURATION - Edit these paths
#===============================================================================

# Sample list file (TSV with columns: child_id, child_bam, father_id, father_bam, mother_id, mother_bam)
SAMPLE_LIST="./resources/sample_list.txt"

# Candidate DNV file (gzipped TSV with columns: chrom, pos, ref, alt, ..., id)
# The 'id' column should contain the child sample ID
DNV_FILE="./data/candidates.tsv.gz"

# Reference files
FASTA="./resources/GRCh38.fasta"
RMSK="./resources/hg38_repeatmasker.bed"

# Region annotation BED files
BED_SEGDUP="./resources/GRCh38_segdups.bed"
BED_LCR="./resources/GRCh38_lcr.bed"
BED_TR="./resources/GRCh38_tandem_repeats.bed"
BED_HP="./resources/GRCh38_homopolymers.bed"
BED_DIFF="./resources/GRCh38_difficult_regions.bed"

# Feature extraction parameters
WIN=50          # Window size for GC content (±bp)
RADIUS=1000     # Radius for neighbor variant counting (±bp)

# Output directories
OUTDIR="./features"
LOGDIR="./logs"
TMPDIR_BASE="./tmp"

# Parallelization
MAX_JOBS=4      # Maximum parallel jobs
BAM_THREADS=6   # Threads per BAM processing
WORKERS=3       # Workers per sample

# Version string for output files
VERSION="v1"

#===============================================================================
# SETUP
#===============================================================================
mkdir -p "$OUTDIR" "$LOGDIR"
TMPDIR=$(mktemp -d "${TMPDIR_BASE}/a2denovo.XXXXXX")
RUN_DATE=$(date +%Y%m%d)

# Cleanup on exit
trap "rm -rf $TMPDIR" EXIT

echo "[INFO $(date)] A2Denovo Feature Extraction Pipeline"
echo "[INFO $(date)] Output directory: $OUTDIR"
echo "[INFO $(date)] Max parallel jobs: $MAX_JOBS"

#===============================================================================
# STEP 1: Split DNV file by sample
#===============================================================================
echo "[INFO $(date)] Splitting DNV file by sample..."

# Extract header
header=$(pigz -dc "$DNV_FILE" | head -n1 || true)

# Determine id column index (assumes 'id' column exists)
id_col=$(echo "$header" | tr '\t' '\n' | grep -n "^id$" | cut -d: -f1)
if [[ -z "$id_col" ]]; then
    echo "[ERROR] Cannot find 'id' column in DNV file"
    exit 1
fi

# Split by sample ID
pigz -dc "$DNV_FILE" | tail -n +2 | \
    awk -F'\t' -v col="$id_col" -v out="$TMPDIR" '{print > out"/"$col}'

# Add header to each split file
for f in "$TMPDIR"/*; do
    [[ -f "$f" ]] || continue
    { echo "$header"; cat "$f"; } > "${f}.tmp" && mv "${f}.tmp" "$f"
done

n_samples=$(ls "$TMPDIR" 2>/dev/null | wc -l || echo 0)
echo "[INFO $(date)] Split done: $n_samples per-sample files"

#===============================================================================
# STEP 2: Define processing function
#===============================================================================
process_one() {
    local SAMPLE=$1
    local BAM_CHILD=$2
    local BAM_FATHER=$3
    local BAM_MOTHER=$4
    
    local VAR_FILE="$TMPDIR/${SAMPLE}"
    
    # Skip if no variants for this sample
    if [[ ! -f "$VAR_FILE" ]]; then
        echo "[WARN] No DNV candidates for $SAMPLE, skipping"
        return 0
    fi
    
    local LOG="$LOGDIR/${SAMPLE}.${VERSION}.${RUN_DATE}.log"
    local OUT="$OUTDIR/${SAMPLE}.win${WIN}_radius${RADIUS}.features.${VERSION}.${RUN_DATE}.tsv"
    
    {
        echo "=== $(date) START $SAMPLE ==="
        
        # Build command
        local CMD="python $(dirname "$0")/dnv_feature_builder.py"
        CMD="$CMD --variants-txt $VAR_FILE"
        CMD="$CMD --bam-child $BAM_CHILD"
        
        [[ -n "$BAM_FATHER" ]] && CMD="$CMD --bam-father $BAM_FATHER"
        [[ -n "$BAM_MOTHER" ]] && CMD="$CMD --bam-mother $BAM_MOTHER"
        
        CMD="$CMD --fasta $FASTA"
        
        [[ -n "$RMSK" && -f "$RMSK" ]] && CMD="$CMD --repeatmasker-cats $RMSK"
        
        [[ -n "$BED_SEGDUP" && -f "$BED_SEGDUP" ]] && CMD="$CMD --bed segdup=$BED_SEGDUP"
        [[ -n "$BED_LCR" && -f "$BED_LCR" ]] && CMD="$CMD --bed lcr=$BED_LCR"
        [[ -n "$BED_TR" && -f "$BED_TR" ]] && CMD="$CMD --bed tr=$BED_TR"
        [[ -n "$BED_HP" && -f "$BED_HP" ]] && CMD="$CMD --bed homopolymer=$BED_HP"
        [[ -n "$BED_DIFF" && -f "$BED_DIFF" ]] && CMD="$CMD --bed GIABdifficult=$BED_DIFF"
        
        CMD="$CMD --win $WIN --radius $RADIUS"
        CMD="$CMD --bam-threads $BAM_THREADS"
        CMD="$CMD --worker $WORKERS"
        CMD="$CMD -o $OUT"
        
        /usr/bin/time -v $CMD
        
        echo "=== $(date) END $SAMPLE ==="
    } 2>&1 | tee "$LOG"
}

# Export function and variables
export -f process_one
export TMPDIR FASTA RMSK
export BED_SEGDUP BED_LCR BED_TR BED_HP BED_DIFF
export WIN RADIUS OUTDIR LOGDIR VERSION RUN_DATE
export BAM_THREADS WORKERS

#===============================================================================
# STEP 3: Process samples in parallel
#===============================================================================
echo "[INFO $(date)] Starting parallel processing (max $MAX_JOBS jobs)..."

while read -r child_id child_bam father_id father_bam mother_id mother_bam; do
    # Skip empty lines and comments
    [[ -z "$child_id" || "$child_id" == \#* ]] && continue
    
    # Launch job in background
    process_one "$child_id" "$child_bam" "$father_bam" "$mother_bam" &
    
    # Limit concurrent jobs
    while [[ $(jobs -rp | wc -l) -ge $MAX_JOBS ]]; do
        sleep 1
    done
    
done < "$SAMPLE_LIST"

# Wait for all jobs to complete
wait

#===============================================================================
# CLEANUP
#===============================================================================
echo "[INFO $(date)] Pipeline finished successfully!"
echo "[INFO $(date)] Features saved to: $OUTDIR"
echo "[INFO $(date)] Logs saved to: $LOGDIR"
