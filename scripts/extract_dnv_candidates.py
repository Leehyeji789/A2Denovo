#!/usr/bin/env python3
"""
A2Denovo Step 0: Extract DNV Candidates

Extract candidate de novo variants from trio VCF using Hail.
This script performs:
1. VCF formatting (split multi-allelic, normalize)
2. Trio matrix construction
3. Mendelian error filtering (child het, parents hom-ref or missing)
4. Population frequency filtering (gnomAD AF < 0.001)

Requirements:
- Hail (pip install hail)
- Input VCF with trio samples
- Pedigree file (.ped format)
- gnomAD Hail Table (optional, for AF filtering)

Usage:
    python extract_dnv_candidates.py \
        --vcf input.vcf.gz \
        --pedigree trio.ped \
        --output candidates.tsv.gz \
        --gnomad gnomad.ht \
        --reference GRCh38
"""

import argparse
import os
import sys

try:
    import hail as hl
except ImportError:
    print("Error: Hail is required. Install with: pip install hail")
    sys.exit(1)


def log(msg):
    """Print with flush."""
    print(f"[A2Denovo] {msg}", flush=True)


def init_hail(log_path=None, driver_memory='16g'):
    """Initialize Hail with custom settings."""
    if log_path is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"./hail_{timestamp}.log"
    
    hl.init(
        spark_conf={'spark.driver.memory': driver_memory},
        log=log_path,
        quiet=True
    )
    log(f"Hail initialized. Log: {log_path}")


def format_vcf(mt):
    """
    Format VCF: split multi-allelic sites and normalize variants.
    """
    log("Splitting multi-allelic sites...")
    mt = hl.split_multi_hts(mt)
    
    log("Normalizing variant representation...")
    mt = mt.annotate_rows(min_rep=hl.min_rep(mt.locus, mt.alleles))
    mt = mt.key_rows_by()
    mt = mt.drop('locus', 'alleles')
    mt = mt.annotate_rows(
        locus=mt.min_rep.locus,
        alleles=mt.min_rep.alleles
    )
    mt = mt.key_rows_by('locus', 'alleles')
    
    log("Annotating variant types...")
    mt = mt.annotate_rows(
        variant_type=hl.case()
        .when(hl.is_snp(mt.alleles[0], mt.alleles[1]), "SNV")
        .when(hl.is_indel(mt.alleles[0], mt.alleles[1]), "INDEL")
        .when(hl.is_complex(mt.alleles[0], mt.alleles[1]), "COMPLEX")
        .default("OTHER")
    )
    
    return mt


def create_trio_matrix(mt, pedigree_path):
    """
    Create trio matrix from pedigree.
    """
    log(f"Reading pedigree: {pedigree_path}")
    pedigree = hl.Pedigree.read(pedigree_path)
    
    log("Creating trio matrix...")
    trio_mt = hl.trio_matrix(mt, pedigree, complete_trios=True)
    
    log("Phasing by transmission...")
    trio_mt = hl.experimental.phase_trio_matrix_by_transmission(trio_mt)
    
    return trio_mt


def filter_mendelian_errors(trio_mt):
    """
    Filter for Mendelian error variants (candidate DNVs).
    
    Keeps variants where:
    - Child is heterozygous AND
    - Both parents are hom-ref, OR one/both parents have missing GT
    """
    log("Filtering Mendelian error variants...")
    
    dn = trio_mt.filter_entries(
        # Child het, both parents hom-ref
        (trio_mt.proband_entry.GT.is_het() & 
         trio_mt.father_entry.GT.is_hom_ref() & 
         trio_mt.mother_entry.GT.is_hom_ref()) |
        # Child het, father missing, mother hom-ref
        (trio_mt.proband_entry.GT.is_het() & 
         hl.is_missing(trio_mt.father_entry.GT) & 
         trio_mt.mother_entry.GT.is_hom_ref()) |
        # Child het, father hom-ref, mother missing
        (trio_mt.proband_entry.GT.is_het() & 
         trio_mt.father_entry.GT.is_hom_ref() & 
         hl.is_missing(trio_mt.mother_entry.GT)) |
        # Child het, both parents missing
        (trio_mt.proband_entry.GT.is_het() & 
         hl.is_missing(trio_mt.father_entry.GT) & 
         hl.is_missing(trio_mt.mother_entry.GT))
    )
    
    return dn


def filter_by_gnomad(mt, gnomad_path, af_threshold=0.001):
    """
    Filter variants by gnomAD allele frequency.
    """
    if gnomad_path is None:
        log("Skipping gnomAD filtering (no path provided)")
        return mt
    
    log(f"Loading gnomAD: {gnomad_path}")
    gnomad = hl.read_table(gnomad_path)
    
    log(f"Filtering by gnomAD AF < {af_threshold}...")
    mt = mt.annotate_rows(gnomAD_AF=gnomad[mt.locus, mt.alleles].AF)
    mt = mt.filter_rows(
        (mt.gnomAD_AF < af_threshold) | ~hl.is_defined(mt.gnomAD_AF)
    )
    
    return mt


def export_candidates(trio_mt, output_path):
    """
    Export candidate DNVs to TSV.
    """
    log("Preparing output table...")
    
    # Run variant QC to get allele counts
    trio_mt = hl.variant_qc(trio_mt)
    
    # Annotate with required fields
    trio_mt = trio_mt.annotate_rows(
        internal_AC=trio_mt.variant_qc.AC[1],
        chrom=trio_mt.locus.contig,
        pos=trio_mt.locus.position,
        ref=trio_mt.alleles[0],
        alt=trio_mt.alleles[1],
        variant_ID=hl.str(trio_mt.locus.contig) + ":" + 
                   hl.str(trio_mt.locus.position) + ":" +
                   trio_mt.alleles[0] + ":" + trio_mt.alleles[1]
    )
    
    # Convert to table
    entries = trio_mt.entries().key_by()
    
    # Select output columns
    output = entries.select(
        'chrom', 'pos', 'ref', 'alt', 'variant_ID', 'id', 'internal_AC'
    )
    
    log(f"Exporting to: {output_path}")
    output.export(output_path)
    
    # Count results
    n_variants = output.count()
    log(f"Exported {n_variants:,} candidate DNV entries")
    
    return n_variants


def main():
    parser = argparse.ArgumentParser(
        description="A2Denovo Step 0: Extract DNV Candidates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python extract_dnv_candidates.py \\
        --vcf trio.vcf.gz \\
        --pedigree family.ped \\
        --output candidates.tsv.gz \\
        --gnomad /path/to/gnomad.ht

Output format (TSV):
    chrom, pos, ref, alt, variant_ID, id, internal_AC
        """
    )
    
    # Required arguments
    parser.add_argument("--vcf", required=True,
                        help="Input VCF file (bgzipped)")
    parser.add_argument("--pedigree", required=True,
                        help="Pedigree file (.ped format)")
    parser.add_argument("--output", required=True,
                        help="Output TSV file (will be gzipped)")
    
    # Optional arguments
    parser.add_argument("--gnomad", default=None,
                        help="gnomAD Hail Table path for AF filtering")
    parser.add_argument("--af-threshold", type=float, default=0.001,
                        help="gnomAD AF threshold (default: 0.001)")
    parser.add_argument("--reference", default="GRCh38",
                        choices=["GRCh37", "GRCh38"],
                        help="Reference genome (default: GRCh38)")
    parser.add_argument("--exclude-samples", default=None,
                        help="File with sample IDs to exclude (one per line)")
    parser.add_argument("--temp-dir", default=None,
                        help="Temporary directory for intermediate files")
    parser.add_argument("--driver-memory", default="16g",
                        help="Spark driver memory (default: 16g)")
    parser.add_argument("--log", default=None,
                        help="Hail log file path")
    
    args = parser.parse_args()
    
    # Initialize Hail
    init_hail(log_path=args.log, driver_memory=args.driver_memory)
    
    # Step 1: Import VCF
    log(f"Importing VCF: {args.vcf}")
    mt = hl.import_vcf(
        args.vcf,
        force_bgz=True,
        reference_genome=args.reference
    )
    
    n_variants, n_samples = mt.count()
    log(f"Loaded {n_variants:,} variants, {n_samples} samples")
    
    # Step 2: Exclude samples if specified
    if args.exclude_samples:
        log(f"Loading samples to exclude: {args.exclude_samples}")
        exclude_table = hl.import_table(
            args.exclude_samples, 
            no_header=True
        ).key_by('f0')
        mt = mt.filter_cols(~hl.is_defined(exclude_table[mt.s]))
        n_samples_after = mt.count_cols()
        log(f"Samples after exclusion: {n_samples_after}")
    
    # Step 3: Format VCF
    mt = format_vcf(mt)
    
    # Step 4: Variant QC
    log("Running variant QC...")
    mt = hl.variant_qc(mt)
    
    # Step 5: Save formatted MT (optional checkpoint)
    if args.temp_dir:
        formatted_path = os.path.join(args.temp_dir, "formatted.mt")
        log(f"Checkpointing formatted MT: {formatted_path}")
        mt = mt.checkpoint(formatted_path, overwrite=True)
    
    # Step 6: Create trio matrix
    trio_mt = create_trio_matrix(mt, args.pedigree)
    
    # Step 7: Filter Mendelian errors
    dn_mt = filter_mendelian_errors(trio_mt)
    
    # Step 8: Filter by gnomAD
    dn_mt = filter_by_gnomad(dn_mt, args.gnomad, args.af_threshold)
    
    # Step 9: Export candidates
    n_exported = export_candidates(dn_mt, args.output)
    
    log("=" * 50)
    log("DNV candidate extraction complete!")
    log(f"Output: {args.output}")
    log(f"Total candidates: {n_exported:,}")
    log("=" * 50)
    
    # Stop Hail
    hl.stop()


if __name__ == "__main__":
    main()
