#!/usr/bin/env python3
"""
A2Denovo Feature Builder

Extracts read-level and genomic context features from candidate DNV sites
for use in the A2Denovo classification model.

Features include:
- Read depth and allelic support
- Mapping quality and edit distance
- PacBio-specific quality metrics
- GC content and sequence entropy
- Overlap with repetitive/difficult regions
- Nearby Mendelian error density

Usage:
    python dnv_feature_builder.py \
        --variants-txt candidates.tsv \
        --bam-child child.bam \
        --bam-father father.bam \
        --bam-mother mother.bam \
        --fasta reference.fasta \
        --repeatmasker-cats repeatmasker.bed \
        --bed segdup=segdups.bed \
        --bed lcr=lcr.bed \
        -o output_features.tsv
"""

import os
import sys
import math
import argparse
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pysam
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ==============================================================================
# Utility Functions
# ==============================================================================

def parse_bed_arg(arg_str):
    """Parse bed argument in format 'name=path'."""
    if '=' in arg_str:
        name, path = arg_str.split('=', 1)
        return name.strip(), path.strip()
    return os.path.basename(arg_str).replace('.bed', ''), arg_str


def load_bed_as_set(bed_path):
    """Load BED file as set of (chrom, pos) tuples."""
    regions = set()
    if not bed_path or not os.path.exists(bed_path):
        return regions
    
    import gzip
    opener = gzip.open if bed_path.endswith('.gz') else open
    
    with opener(bed_path, 'rt') as f:
        for line in f:
            if line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            for pos in range(start, end):
                regions.add((chrom, pos))
    
    return regions


def load_bed_intervals(bed_path):
    """Load BED file as list of (chrom, start, end) intervals."""
    intervals = []
    if not bed_path or not os.path.exists(bed_path):
        return intervals
    
    import gzip
    opener = gzip.open if bed_path.endswith('.gz') else open
    
    with opener(bed_path, 'rt') as f:
        for line in f:
            if line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            intervals.append((parts[0], int(parts[1]), int(parts[2])))
    
    return intervals


def check_overlap_interval(chrom, pos, intervals):
    """Check if position overlaps any interval (binary search would be better for large files)."""
    for c, s, e in intervals:
        if c == chrom and s <= pos < e:
            return 1
    return 0


def load_repeatmasker(rmsk_path):
    """
    Load RepeatMasker annotations.
    Expected columns: chrom, start, end, repeat_class
    """
    if not rmsk_path or not os.path.exists(rmsk_path):
        return {}
    
    rmsk_dict = defaultdict(list)
    
    import gzip
    opener = gzip.open if rmsk_path.endswith('.gz') else open
    
    with opener(rmsk_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            chrom, start, end, rep_class = parts[0], int(parts[1]), int(parts[2]), parts[3]
            rmsk_dict[(chrom, start, end)] = rep_class
    
    return rmsk_dict


def get_repeat_class(chrom, pos, rmsk_dict):
    """Get repeat class for a position."""
    for (c, s, e), rep_class in rmsk_dict.items():
        if c == chrom and s <= pos < e:
            return rep_class
    return "None"


def compute_gc_content(seq):
    """Compute GC content of sequence."""
    if not seq:
        return 0.0
    seq = seq.upper()
    gc = sum(1 for b in seq if b in 'GC')
    return gc / len(seq)


def compute_shannon_entropy(seq):
    """
    Compute Shannon entropy of sequence.
    H = -sum(p * log2(p)) for each nucleotide.
    """
    if not seq:
        return 0.0
    seq = seq.upper()
    counts = defaultdict(int)
    for b in seq:
        if b in 'ACGT':
            counts[b] += 1
    
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


# ==============================================================================
# Read Feature Extraction
# ==============================================================================

def get_read_features_at_site(bam, chrom, pos, ref, alt, is_snv=True):
    """
    Extract read-level features at a variant site.
    
    Returns dict with:
    - depth_total: Total read depth
    - depth_alt: Alternate allele depth
    - depth_ref: Reference allele depth
    - mean_mapq: Mean mapping quality
    - mean_baseq: Mean base quality at position
    - mean_edit_dist: Mean edit distance (NM tag)
    - allele_balance: Alt / Total ratio
    - Additional PacBio metrics if available
    """
    features = {
        'depth_total': 0,
        'depth_alt': 0,
        'depth_ref': 0,
        'mean_mapq': 0.0,
        'mean_baseq': 0.0,
        'mean_edit_dist': 0.0,
        'allele_balance': 0.0,
        'mean_rq': 0.0,        # Read quality
        'mean_ec': 0.0,        # Effective coverage
        'mean_np': 0.0,        # Number of passes
    }
    
    try:
        # pysam uses 0-based coordinates
        pileup = bam.pileup(chrom, pos - 1, pos, truncate=True, min_base_quality=0)
    except Exception:
        return features
    
    mapqs = []
    baseqs = []
    edit_dists = []
    rqs = []
    ecs = []
    nps = []
    alt_count = 0
    ref_count = 0
    total_count = 0
    
    for pileup_col in pileup:
        if pileup_col.pos != pos - 1:  # 0-based
            continue
        
        for read in pileup_col.pileups:
            if read.is_del or read.is_refskip:
                continue
            
            aln = read.alignment
            total_count += 1
            mapqs.append(aln.mapping_quality)
            
            # Get base at position
            query_pos = read.query_position
            if query_pos is not None:
                base = aln.query_sequence[query_pos]
                qual = aln.query_qualities[query_pos] if aln.query_qualities else 0
                baseqs.append(qual)
                
                if is_snv:
                    if base.upper() == alt.upper():
                        alt_count += 1
                    elif base.upper() == ref.upper():
                        ref_count += 1
                else:
                    # For indels, more complex logic needed
                    # Simplified: check if read supports alt
                    if len(alt) > len(ref):  # Insertion
                        alt_count += 1 if read.indel > 0 else 0
                    else:  # Deletion
                        alt_count += 1 if read.indel < 0 else 0
                    ref_count = total_count - alt_count
            
            # Edit distance
            if aln.has_tag('NM'):
                edit_dists.append(aln.get_tag('NM'))
            
            # PacBio-specific tags
            if aln.has_tag('rq'):
                rqs.append(aln.get_tag('rq'))
            if aln.has_tag('ec'):
                ecs.append(aln.get_tag('ec'))
            if aln.has_tag('np'):
                nps.append(aln.get_tag('np'))
    
    features['depth_total'] = total_count
    features['depth_alt'] = alt_count
    features['depth_ref'] = ref_count
    features['mean_mapq'] = np.mean(mapqs) if mapqs else 0.0
    features['mean_baseq'] = np.mean(baseqs) if baseqs else 0.0
    features['mean_edit_dist'] = np.mean(edit_dists) if edit_dists else 0.0
    features['allele_balance'] = alt_count / total_count if total_count > 0 else 0.0
    features['mean_rq'] = np.mean(rqs) if rqs else 0.0
    features['mean_ec'] = np.mean(ecs) if ecs else 0.0
    features['mean_np'] = np.mean(nps) if nps else 0.0
    
    return features


def get_alt_read_features(bam, chrom, pos, ref, alt, is_snv=True):
    """
    Extract features from alternate-supporting reads only.
    """
    features = {
        'mean_mapq_alt': 0.0,
        'mean_baseq_alt': 0.0,
        'mean_edit_dist_alt': 0.0,
        'mean_rq_alt': 0.0,
        'mean_np_alt': 0.0,
    }
    
    try:
        pileup = bam.pileup(chrom, pos - 1, pos, truncate=True, min_base_quality=0)
    except Exception:
        return features
    
    mapqs = []
    baseqs = []
    edit_dists = []
    rqs = []
    nps = []
    
    for pileup_col in pileup:
        if pileup_col.pos != pos - 1:
            continue
        
        for read in pileup_col.pileups:
            if read.is_del or read.is_refskip:
                continue
            
            aln = read.alignment
            query_pos = read.query_position
            
            if query_pos is None:
                continue
            
            base = aln.query_sequence[query_pos]
            
            # Check if this read supports alt
            is_alt_read = False
            if is_snv:
                is_alt_read = base.upper() == alt.upper()
            else:
                if len(alt) > len(ref):
                    is_alt_read = read.indel > 0
                else:
                    is_alt_read = read.indel < 0
            
            if not is_alt_read:
                continue
            
            mapqs.append(aln.mapping_quality)
            qual = aln.query_qualities[query_pos] if aln.query_qualities else 0
            baseqs.append(qual)
            
            if aln.has_tag('NM'):
                edit_dists.append(aln.get_tag('NM'))
            if aln.has_tag('rq'):
                rqs.append(aln.get_tag('rq'))
            if aln.has_tag('np'):
                nps.append(aln.get_tag('np'))
    
    features['mean_mapq_alt'] = np.mean(mapqs) if mapqs else 0.0
    features['mean_baseq_alt'] = np.mean(baseqs) if baseqs else 0.0
    features['mean_edit_dist_alt'] = np.mean(edit_dists) if edit_dists else 0.0
    features['mean_rq_alt'] = np.mean(rqs) if rqs else 0.0
    features['mean_np_alt'] = np.mean(nps) if nps else 0.0
    
    return features


# ==============================================================================
# Main Feature Extraction
# ==============================================================================

def extract_features_for_variant(variant_row, bam_child, bam_father, bam_mother,
                                  fasta, rmsk_dict, bed_intervals, window, radius,
                                  all_variants_df):
    """
    Extract all features for a single variant.
    
    Parameters:
    -----------
    variant_row : dict
        Variant information (chrom, pos, ref, alt, id, etc.)
    bam_child : pysam.AlignmentFile
        Child BAM file
    bam_father : pysam.AlignmentFile or None
        Father BAM file
    bam_mother : pysam.AlignmentFile or None
        Mother BAM file
    fasta : pysam.FastaFile
        Reference FASTA
    rmsk_dict : dict
        RepeatMasker annotations
    bed_intervals : dict
        BED interval annotations by name
    window : int
        Window size for GC/entropy calculation
    radius : int
        Radius for neighbor variant counting
    all_variants_df : pd.DataFrame
        All variants for neighbor counting
    
    Returns:
    --------
    dict : Feature dictionary
    """
    chrom = str(variant_row['chrom'])
    pos = int(variant_row['pos'])
    ref = str(variant_row['ref'])
    alt = str(variant_row['alt'])
    
    is_snv = len(ref) == 1 and len(alt) == 1
    variant_type = 'SNV' if is_snv else ('INS' if len(alt) > len(ref) else 'DEL')
    variant_len = abs(len(alt) - len(ref))
    
    features = {
        'chrom': chrom,
        'pos': pos,
        'ref': ref,
        'alt': alt,
        'variant_ID': f"{chrom}:{pos}:{ref}:{alt}",
        'variant_type': variant_type,
        'variant_len': variant_len,
    }
    
    # Copy ID columns
    for col in ['id', 'internal_AC']:
        if col in variant_row:
            features[col] = variant_row[col]
    
    # ----- Child read features -----
    if bam_child:
        child_feats = get_read_features_at_site(bam_child, chrom, pos, ref, alt, is_snv)
        for k, v in child_feats.items():
            features[k] = v
        
        child_alt_feats = get_alt_read_features(bam_child, chrom, pos, ref, alt, is_snv)
        for k, v in child_alt_feats.items():
            features[k] = v
    
    # ----- Father read features -----
    if bam_father:
        father_feats = get_read_features_at_site(bam_father, chrom, pos, ref, alt, is_snv)
        for k, v in father_feats.items():
            features[f'{k}_father'] = v
    
    # ----- Mother read features -----
    if bam_mother:
        mother_feats = get_read_features_at_site(bam_mother, chrom, pos, ref, alt, is_snv)
        for k, v in mother_feats.items():
            features[f'{k}_mother'] = v
    
    # ----- Genomic context -----
    if fasta:
        try:
            # Get sequence around variant
            start = max(0, pos - window - 1)
            end = pos + window
            seq = fasta.fetch(chrom, start, end)
            
            features['gc_content'] = compute_gc_content(seq)
            features['shannon_entropy'] = compute_shannon_entropy(seq)
        except Exception:
            features['gc_content'] = 0.0
            features['shannon_entropy'] = 0.0
    
    # ----- Repeat annotations -----
    if rmsk_dict:
        rep_class = get_repeat_class(chrom, pos, rmsk_dict)
        features['repeat_class'] = rep_class
        features['in_repeat'] = 1 if rep_class != "None" else 0
    
    # ----- BED region overlaps -----
    for bed_name, intervals in bed_intervals.items():
        features[f'in_{bed_name}'] = check_overlap_interval(chrom, pos, intervals)
    
    # ----- Neighbor variant density -----
    if all_variants_df is not None and len(all_variants_df) > 0:
        same_chrom = all_variants_df[all_variants_df['chrom'] == chrom]
        in_radius = same_chrom[
            (same_chrom['pos'] >= pos - radius) & 
            (same_chrom['pos'] <= pos + radius) &
            (same_chrom['pos'] != pos)
        ]
        features['n_neighbors'] = len(in_radius)
        
        if len(in_radius) > 0:
            distances = np.abs(in_radius['pos'].values - pos)
            features['min_neighbor_dist'] = int(distances.min())
        else:
            features['min_neighbor_dist'] = radius + 1
    else:
        features['n_neighbors'] = 0
        features['min_neighbor_dist'] = radius + 1
    
    return features


def process_variant_batch(batch_args):
    """Process a batch of variants (for parallel execution)."""
    (variants_batch, bam_child_path, bam_father_path, bam_mother_path,
     fasta_path, rmsk_dict, bed_intervals, window, radius, all_variants_df) = batch_args
    
    # Open files for this process
    bam_child = pysam.AlignmentFile(bam_child_path, 'rb') if bam_child_path else None
    bam_father = pysam.AlignmentFile(bam_father_path, 'rb') if bam_father_path else None
    bam_mother = pysam.AlignmentFile(bam_mother_path, 'rb') if bam_mother_path else None
    fasta = pysam.FastaFile(fasta_path) if fasta_path else None
    
    results = []
    for _, row in variants_batch.iterrows():
        feat = extract_features_for_variant(
            row.to_dict(), bam_child, bam_father, bam_mother,
            fasta, rmsk_dict, bed_intervals, window, radius, all_variants_df
        )
        results.append(feat)
    
    # Close files
    if bam_child:
        bam_child.close()
    if bam_father:
        bam_father.close()
    if bam_mother:
        bam_mother.close()
    if fasta:
        fasta.close()
    
    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="A2Denovo Feature Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--variants-txt", required=True,
                        help="Input variants TSV (chrom, pos, ref, alt, id, ...)")
    parser.add_argument("--bam-child", required=True,
                        help="Child BAM file")
    parser.add_argument("-o", "--output", required=True,
                        help="Output TSV file")
    
    # Optional BAM files
    parser.add_argument("--bam-father", default=None,
                        help="Father BAM file")
    parser.add_argument("--bam-mother", default=None,
                        help="Mother BAM file")
    
    # Reference files
    parser.add_argument("--fasta", default=None,
                        help="Reference FASTA file")
    parser.add_argument("--repeatmasker-cats", default=None,
                        help="RepeatMasker BED file")
    
    # BED annotations (can specify multiple)
    parser.add_argument("--bed", action="append", default=[],
                        help="BED annotation in format 'name=path' (can repeat)")
    
    # Parameters
    parser.add_argument("--win", type=int, default=50,
                        help="Window size for GC/entropy (default: 50)")
    parser.add_argument("--radius", type=int, default=1000,
                        help="Radius for neighbor counting (default: 1000)")
    
    # Performance
    parser.add_argument("--bam-threads", type=int, default=4,
                        help="BAM reading threads (default: 4)")
    parser.add_argument("--worker", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Variants per batch (default: 100)")
    
    args = parser.parse_args()
    
    # Load variants
    print(f"Loading variants from {args.variants_txt}...")
    sep = '\t' if args.variants_txt.endswith('.tsv') else ','
    variants_df = pd.read_csv(args.variants_txt, sep=sep)
    print(f"  Loaded {len(variants_df):,} variants")
    
    # Required columns
    required_cols = ['chrom', 'pos', 'ref', 'alt']
    for col in required_cols:
        if col not in variants_df.columns:
            sys.exit(f"Error: Missing required column '{col}'")
    
    # Load reference files
    rmsk_dict = {}
    if args.repeatmasker_cats:
        print(f"Loading RepeatMasker annotations...")
        rmsk_dict = load_repeatmasker(args.repeatmasker_cats)
        print(f"  Loaded {len(rmsk_dict):,} repeat regions")
    
    # Load BED annotations
    bed_intervals = {}
    for bed_arg in args.bed:
        name, path = parse_bed_arg(bed_arg)
        print(f"Loading BED: {name} from {path}...")
        bed_intervals[name] = load_bed_intervals(path)
        print(f"  Loaded {len(bed_intervals[name]):,} intervals")
    
    # Process variants
    print(f"\nExtracting features...")
    print(f"  Window: ±{args.win}bp")
    print(f"  Neighbor radius: ±{args.radius}bp")
    print(f"  Workers: {args.worker}")
    
    if args.worker == 1:
        # Single-threaded processing
        bam_child = pysam.AlignmentFile(args.bam_child, 'rb', threads=args.bam_threads)
        bam_father = pysam.AlignmentFile(args.bam_father, 'rb', threads=args.bam_threads) if args.bam_father else None
        bam_mother = pysam.AlignmentFile(args.bam_mother, 'rb', threads=args.bam_threads) if args.bam_mother else None
        fasta = pysam.FastaFile(args.fasta) if args.fasta else None
        
        results = []
        for _, row in tqdm(variants_df.iterrows(), total=len(variants_df), desc="Processing"):
            feat = extract_features_for_variant(
                row.to_dict(), bam_child, bam_father, bam_mother,
                fasta, rmsk_dict, bed_intervals, args.win, args.radius, variants_df
            )
            results.append(feat)
        
        bam_child.close()
        if bam_father:
            bam_father.close()
        if bam_mother:
            bam_mother.close()
        if fasta:
            fasta.close()
    
    else:
        # Parallel processing
        n_batches = (len(variants_df) + args.batch_size - 1) // args.batch_size
        batches = np.array_split(variants_df, n_batches)
        
        batch_args = [
            (batch, args.bam_child, args.bam_father, args.bam_mother,
             args.fasta, rmsk_dict, bed_intervals, args.win, args.radius, variants_df)
            for batch in batches
        ]
        
        results = []
        with ProcessPoolExecutor(max_workers=args.worker) as executor:
            futures = [executor.submit(process_variant_batch, ba) for ba in batch_args]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Batches"):
                results.extend(future.result())
    
    # Save results
    print(f"\nSaving {len(results):,} feature rows to {args.output}...")
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, sep='\t' if args.output.endswith('.tsv') else ',', index=False)
    
    print("Done!")


if __name__ == "__main__":
    main()
