#!/usr/bin/env python3
"""
A2Denovo Prediction Script

Apply trained model to predict de novo variants in new samples.

Usage:
    python a2denovo_predict.py \
        --model-bundle ./model/final_model_bundle.pkl \
        --feature-dir ./features/ \
        --output-dir ./predictions/
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from glob import glob


def log(msg):
    """Print with flush."""
    print(msg, flush=True)


def load_model_bundle(path):
    """Load trained model bundle."""
    log(f"🔹 Loading model bundle: {path}")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    
    log(f"  ✓ Features: {len(bundle['base_features'])}")
    log(f"  ✓ Threshold: {bundle['final_threshold']:.4f}")
    
    return bundle


def find_feature_files(feature_dir, pattern="*.tsv"):
    """Find feature files in directory."""
    log(f"🔹 Searching for feature files: {feature_dir}")
    
    search_path = os.path.join(feature_dir, pattern)
    files = sorted(glob(search_path))
    
    if not files:
        raise FileNotFoundError(f"No files found: {search_path}")
    
    log(f"  ✓ Found {len(files)} files")
    for f in files[:5]:
        log(f"    - {os.path.basename(f)}")
    if len(files) > 5:
        log(f"    ... and {len(files) - 5} more")
    
    return files


def load_and_combine_features(files, add_sample_id=True):
    """Load and combine feature files."""
    log(f"🔹 Loading {len(files)} feature files...")
    
    dfs = []
    for file_path in files:
        try:
            # Detect separator
            sep = "\t" if file_path.endswith(".tsv") else ","
            df = pd.read_csv(file_path, sep=sep)
            
            if add_sample_id and 'sample_id' not in df.columns:
                sample_id = os.path.basename(file_path).split(".")[0]
                df['sample_id'] = sample_id
            
            dfs.append(df)
        except Exception as e:
            log(f"  ⚠️ Error loading {os.path.basename(file_path)}: {e}")
    
    if not dfs:
        raise ValueError("No feature files loaded successfully")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.columns = combined.columns.astype(str).str.strip()
    
    log(f"  ✓ Combined: {len(combined):,} variants from {len(dfs)} files")
    return combined


def filter_variants(df, ac_col="internal_AC", filter_ac1=True):
    """Filter variants by allele count."""
    if not filter_ac1:
        log(f"🔹 AC filtering disabled")
        return df
    
    log(f"🔹 Filtering variants with {ac_col}=1...")
    
    if ac_col not in df.columns:
        log(f"  ⚠️ Column '{ac_col}' not found, skipping filter")
        return df
    
    mask = pd.to_numeric(df[ac_col], errors='coerce').fillna(-1).astype(int) == 1
    filtered = df.loc[mask].copy()
    
    log(f"  ✓ Filtered: {len(filtered):,} / {len(df):,} variants")
    return filtered


def prepare_features(df, feat_cols):
    """Prepare feature matrix for prediction."""
    log("🔹 Preparing features...")
    
    # Add missing columns
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        log(f"  - Adding {len(missing)} missing features (filled with 0)")
        for c in missing:
            df[c] = 0.0
    
    X = df[feat_cols].copy()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float64)
    
    # Replace non-finite values
    arr = X.to_numpy()
    bad = ~np.isfinite(arr)
    if bad.any():
        log(f"  - Replaced {bad.sum():,} non-finite values")
        arr[bad] = 0.0
        X.iloc[:, :] = arr
    
    log(f"  ✓ Shape: {X.shape}")
    return X


def predict_raw_scores(pipe, X):
    """Get raw scores from pipeline."""
    if hasattr(pipe, "decision_function"):
        s = pipe.decision_function(X)
    elif hasattr(pipe, "predict_proba"):
        s = pipe.predict_proba(X)[:, 1]
    else:
        s = pipe.predict(X)
        s = np.where(np.asarray(s) == 1, 3.0, -3.0)
    
    s = np.asarray(s)
    if s.ndim == 2:
        s = s[:, -1]
    return s.ravel()


def make_predictions(bundle, X):
    """Run predictions with calibration."""
    log("🔹 Running predictions...")
    
    pipe = bundle["final_pipe"]
    calibrator = bundle["calibrator"]
    threshold = bundle["final_threshold"]
    
    raw_scores = predict_raw_scores(pipe, X)
    
    # Handle non-finite scores
    bad_raw = ~np.isfinite(raw_scores)
    if bad_raw.any():
        log(f"  - Fixed {bad_raw.sum()} non-finite raw scores")
        raw_scores = np.where(np.isfinite(raw_scores), raw_scores, 0.0)
    
    # Calibrate
    prob_calib = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
    
    bad_prob = ~np.isfinite(prob_calib)
    if bad_prob.any():
        log(f"  - Fixed {bad_prob.sum()} non-finite calibrated probs")
        prob_calib = np.where(np.isfinite(prob_calib), prob_calib, 0.5)
    
    pred_label = (prob_calib >= threshold).astype(np.int8)
    
    log(f"  ✓ Predicted de novo: {(pred_label == 1).sum():,} / {len(pred_label):,}")
    log(f"  ✓ Predicted artifact: {(pred_label == 0).sum():,} / {len(pred_label):,}")
    
    return raw_scores, prob_calib, pred_label


def apply_read_filters(df, pred_label, verbose=True):
    """
    Apply additional read-level filters to predictions.
    
    Post-prediction filters (from Methods):
    - No alternate-supporting reads in either parent
    - Total read depth < 1,000 reads in child and each parent
    - At least 5 reads of total depth in each parent
    - At least 1 alternate-supporting read in child
    - Child allele balance between 0.2 and 0.8
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature dataframe with read-level columns
    pred_label : np.ndarray
        Initial prediction labels (1=DNV, 0=artifact)
    verbose : bool
        Print filter statistics
    
    Returns:
    --------
    np.ndarray : Filtered prediction labels
    """
    if verbose:
        log("🔹 Applying post-prediction read-level filters...")
    
    mask = pred_label == 1
    n_before = mask.sum()
    
    if n_before == 0:
        if verbose:
            log("  ✓ No positive predictions to filter")
        return pred_label
    
    # Column name mappings (feature_builder output → filter requirement)
    # Try multiple possible column names for compatibility
    col_mappings = {
        'depth_alt_father': ['depth_alt_father', 'alt_depth_father', 'AD_father'],
        'depth_alt_mother': ['depth_alt_mother', 'alt_depth_mother', 'AD_mother'],
        'depth_total': ['depth_total', 'total_depth', 'DP'],
        'depth_total_father': ['depth_total_father', 'total_depth_father', 'DP_father'],
        'depth_total_mother': ['depth_total_mother', 'total_depth_mother', 'DP_mother'],
        'depth_alt': ['depth_alt', 'alt_depth', 'AD'],
        'allele_balance': ['allele_balance', 'AB', 'VAF'],
    }
    
    # Find available columns
    available_cols = {}
    missing_cols = []
    
    for target, candidates in col_mappings.items():
        found = False
        for col in candidates:
            if col in df.columns:
                available_cols[target] = col
                found = True
                break
        if not found:
            missing_cols.append(target)
    
    if missing_cols:
        if verbose:
            log(f"  ⚠️ Missing columns for filtering: {missing_cols}")
            log(f"  ⚠️ Skipping post-prediction filters (returning model predictions only)")
        return pred_label
    
    # Build filter conditions
    filter_conditions = []
    filter_stats = {}
    
    # 1. No alternate reads in parents
    father_alt_col = available_cols['depth_alt_father']
    mother_alt_col = available_cols['depth_alt_mother']
    
    cond_no_parent_alt = (
        (df[father_alt_col].fillna(0) == 0) & 
        (df[mother_alt_col].fillna(0) == 0)
    )
    filter_conditions.append(cond_no_parent_alt)
    filter_stats['no_parent_alt'] = cond_no_parent_alt[mask].sum()
    
    # 2. Total depth < 1000 in child and parents
    depth_child_col = available_cols['depth_total']
    depth_father_col = available_cols['depth_total_father']
    depth_mother_col = available_cols['depth_total_mother']
    
    cond_depth_limit = (
        (df[depth_child_col].fillna(0) < 1000) &
        (df[depth_father_col].fillna(0) < 1000) &
        (df[depth_mother_col].fillna(0) < 1000)
    )
    filter_conditions.append(cond_depth_limit)
    filter_stats['depth_<1000'] = cond_depth_limit[mask].sum()
    
    # 3. At least 5 reads in each parent
    cond_parent_depth = (
        (df[depth_father_col].fillna(0) >= 5) &
        (df[depth_mother_col].fillna(0) >= 5)
    )
    filter_conditions.append(cond_parent_depth)
    filter_stats['parent_depth_>=5'] = cond_parent_depth[mask].sum()
    
    # 4. At least 1 alternate read in child
    alt_child_col = available_cols['depth_alt']
    cond_child_alt = df[alt_child_col].fillna(0) >= 1
    filter_conditions.append(cond_child_alt)
    filter_stats['child_alt_>=1'] = cond_child_alt[mask].sum()
    
    # 5. Allele balance 0.2-0.8 in child
    ab_col = available_cols['allele_balance']
    cond_ab = (
        (df[ab_col].fillna(0) >= 0.2) &
        (df[ab_col].fillna(1) <= 0.8)
    )
    filter_conditions.append(cond_ab)
    filter_stats['AB_0.2-0.8'] = cond_ab[mask].sum()
    
    # Combine all filters
    combined_filter = filter_conditions[0]
    for cond in filter_conditions[1:]:
        combined_filter = combined_filter & cond
    
    # Apply to predictions
    filtered_pred = np.where(mask & combined_filter.values, 1, 0).astype(np.int8)
    n_after = (filtered_pred == 1).sum()
    
    if verbose:
        log(f"  Filter statistics (among {n_before} predicted DNVs):")
        for fname, count in filter_stats.items():
            log(f"    - {fname}: {count:,} pass")
        log(f"  ✓ Final: {n_after:,} / {n_before:,} passed all filters")
    
    return filtered_pred


def save_results(df, raw_scores, prob_calib, pred_label_model, pred_label_filtered, 
                 bundle, output_dir, prefix, filter_ac1, save_per_sample=False):
    """
    Save prediction results.
    
    Saves both model predictions and post-filtered predictions.
    """
    log(f"🔹 Saving results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    id_cols = bundle.get("id_cols", [])
    reserved = {"raw_score", "prob_calib", "pred_label", "pred_label_filtered", "pred_dnv"}
    id_cols_clean = [c for c in id_cols if c not in reserved and c in df.columns]
    
    if 'sample_id' in df.columns and 'sample_id' not in id_cols_clean:
        id_cols_clean = ['sample_id'] + id_cols_clean
    
    # Build prediction DataFrame with both labels
    pred_df = pd.concat([
        df[id_cols_clean].reset_index(drop=True),
        pd.DataFrame({
            "raw_score": raw_scores.astype(np.float64),
            "prob_calib": prob_calib.astype(np.float64),
            "pred_label_model": pred_label_model.astype(np.int8),      # Model prediction only
            "pred_label_filtered": pred_label_filtered.astype(np.int8), # After post-filtering
            "pred_dnv": pred_label_filtered.astype(np.int8),            # Final call (same as filtered)
        })
    ], axis=1)
    
    ac_suffix = ".AC1" if filter_ac1 else ""
    
    # Save combined predictions
    pred_path = os.path.join(output_dir, f"{prefix}_combined{ac_suffix}.csv.gz")
    pred_df.to_csv(pred_path, index=False, compression='gzip')
    log(f"  ✓ Combined predictions: {pred_path}")
    
    # Save final DNV calls only (filtered)
    dnv_df = pred_df[pred_df['pred_dnv'] == 1]
    if len(dnv_df) > 0:
        dnv_path = os.path.join(output_dir, f"{prefix}_final_dnv_calls{ac_suffix}.tsv")
        dnv_df.to_csv(dnv_path, sep='\t', index=False)
        log(f"  ✓ Final DNV calls: {dnv_path} ({len(dnv_df):,} variants)")
    
    # Save statistics
    stats = {
        "total_variants": len(pred_df),
        "model_predicted_denovo": int((pred_label_model == 1).sum()),
        "filtered_predicted_denovo": int((pred_label_filtered == 1).sum()),
        "removed_by_filter": int((pred_label_model == 1).sum() - (pred_label_filtered == 1).sum()),
        "predicted_artifact": int((pred_label_filtered == 0).sum()),
        "mean_prob_calib": float(prob_calib.mean()),
        "median_prob_calib": float(np.median(prob_calib)),
        "threshold_used": float(bundle["final_threshold"]),
        "ac1_filtered": filter_ac1,
    }
    
    if 'sample_id' in df.columns:
        stats['num_samples'] = df['sample_id'].nunique()
    
    stats_path = os.path.join(output_dir, f"{prefix}_stats.txt")
    with open(stats_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("A2Denovo Prediction Statistics\n")
        f.write("=" * 50 + "\n\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    log(f"  ✓ Statistics: {stats_path}")
    
    # Save per-sample results
    if save_per_sample and 'sample_id' in df.columns:
        sample_dir = os.path.join(output_dir, "per_sample")
        os.makedirs(sample_dir, exist_ok=True)
        
        for sample_id in df['sample_id'].unique():
            mask = df['sample_id'] == sample_id
            sample_pred = pred_df[mask.values]
            sample_path = os.path.join(sample_dir, f"{sample_id}{ac_suffix}.csv.gz")
            sample_pred.to_csv(sample_path, index=False, compression='gzip')
        
        log(f"  ✓ Per-sample results: {sample_dir}")
    
    return pred_df


def main():
    parser = argparse.ArgumentParser(
        description="A2Denovo Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--model-bundle", required=True,
                        help="Path to trained model bundle (.pkl)")
    parser.add_argument("--feature-dir", required=True,
                        help="Directory containing feature files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for predictions")
    
    # Optional arguments
    parser.add_argument("--file-pattern", default="*.tsv",
                        help="File pattern to match (default: *.tsv)")
    parser.add_argument("--output-prefix", default="predictions",
                        help="Output file prefix")
    parser.add_argument("--filter-ac1", action="store_true",
                        help="Filter to AC=1 variants only")
    parser.add_argument("--no-read-filters", action="store_true",
                        help="Disable post-prediction read-level filters (not recommended)")
    parser.add_argument("--save-per-sample", action="store_true",
                        help="Save individual sample predictions")
    
    args = parser.parse_args()
    
    # Load model
    bundle = load_model_bundle(args.model_bundle)
    
    # Find and load features
    files = find_feature_files(args.feature_dir, args.file_pattern)
    df = load_and_combine_features(files)
    
    # Filter variants
    df_filtered = filter_variants(df, filter_ac1=args.filter_ac1)
    
    if df_filtered.empty:
        log("⚠️ No variants to predict after filtering")
        sys.exit(1)
    
    # Prepare features
    feat_cols = bundle["base_features"]
    X = prepare_features(df_filtered, feat_cols)
    
    # Make predictions
    raw_scores, prob_calib, pred_label_model = make_predictions(bundle, X)
    
    # Apply post-prediction read-level filters (default: enabled)
    if args.no_read_filters:
        log("⚠️ Post-prediction read filters DISABLED (--no-read-filters)")
        pred_label_filtered = pred_label_model.copy()
    else:
        pred_label_filtered = apply_read_filters(df_filtered, pred_label_model)
    
    # Save results
    save_results(
        df_filtered, raw_scores, prob_calib, pred_label_model, pred_label_filtered,
        bundle, args.output_dir, args.output_prefix, args.filter_ac1, args.save_per_sample
    )
    
    log("✅ Prediction complete!")


if __name__ == "__main__":
    main()
