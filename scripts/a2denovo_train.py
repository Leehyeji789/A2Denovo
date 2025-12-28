#!/usr/bin/env python3
"""
A2Denovo Model Training Script

Trains a logistic regression classifier with L1 feature selection and 
probability calibration for de novo variant detection.

Usage:
    python a2denovo_train.py \
        --feature-dir ./features/ \
        --sample-list sample_list.txt \
        --true-variants true_dnvs.tsv \
        --output-dir ./model/
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedGroupKFold, 
    RandomizedSearchCV, 
    LeaveOneGroupOut
)
from sklearn.metrics import (
    average_precision_score, 
    precision_recall_curve, 
    roc_auc_score
)
from sklearn.base import clone


# ====================
# Configuration
# ====================
ID_COLS = [
    'chrom', 'pos', 'ref', 'alt', 'variant_ID', 'id', 'internal_AC',
    'variant_type'
]

EXCLUDE_FROM_FEATURES = [
    'homopolymer_max_len_incl_varpos',
    'homopolymer_max_len_excl_varpos',
    'homopolymer_run_len',
    'is_homopolymer_ge6',
    'mean_bc_hamm_dist_mother_alt',
    'mean_bc_hamm_dist_father_alt',
    'mean_bc_hamm_dist_alt',
    'mean_bc_hamm_dist_mother_all',
    'mean_bc_hamm_dist_father_all',
    'mean_bc_hamm_dist_all',
    'zm_div_alt',
    'zm_div_father_alt',
    'zm_div_mother_alt',
    'zm_div_all',
    'zm_div_father_all',
    'zm_div_mother_all'
]


# ====================
# Utility Functions
# ====================
def log(msg):
    """Print with flush."""
    print(msg, flush=True)


def pick_threshold_for_recall(y_true, y_score, recall_min=0.90):
    """
    Pick threshold that achieves at least recall_min while maximizing precision.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    idx_all = np.where(rec >= recall_min)[0]
    
    if idx_all.size == 0:
        return 0.5, 0.0, 0.0
    
    idx_use = idx_all[idx_all > 0]
    if idx_use.size == 0:
        chosen_thr = float(thr.min()) if thr.size else 0.0
        return chosen_thr, float(prec[0]), float(rec[0])
    
    j = idx_use[np.argmax(prec[idx_use])]
    chosen_thr = float(thr[j - 1])
    return chosen_thr, float(prec[j]), float(rec[j])


def evaluate_binary(y_true, y_prob, recall_target=0.90):
    """Compute evaluation metrics."""
    ap = average_precision_score(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob) if np.unique(y_true).size > 1 else np.nan
    thr, p_at_r, r_at_thr = pick_threshold_for_recall(y_true, y_prob, recall_min=recall_target)
    return {
        "PR_AUC": ap, 
        "AUROC": auc, 
        "thr": thr, 
        "Prec@thr": p_at_r, 
        "Recall@thr": r_at_thr
    }


def _score(est, X):
    """Get prediction scores from estimator."""
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X)


def get_selected_feature_names(fitted_pipe, input_cols):
    """Get names of features selected by the pipeline."""
    vt = fitted_pipe.named_steps['vt']
    mask_vt = vt.get_support(indices=False)
    cols_after_vt = [c for c, m in zip(input_cols, mask_vt) if m]
    
    if 'l1sel' in fitted_pipe.named_steps:
        mask_l1 = fitted_pipe.named_steps['l1sel'].get_support(indices=False)
        final_cols = [c for c, m in zip(cols_after_vt, mask_l1) if m]
        return final_cols
    return cols_after_vt


def per_child_undersample(df, label_col, group_col, neg_per_pos=5, 
                          min_neg=1, pos_label=1, random_state=42, replace=False):
    """Undersample negatives per child to handle class imbalance."""
    rng = np.random.default_rng(random_state)
    keep_idx = []
    
    for gid, sub in df.groupby(group_col):
        idx = sub.index.values
        y = sub[label_col].values
        pos_idx = idx[y == pos_label]
        neg_idx = idx[y != pos_label]
        keep_idx.extend(pos_idx.tolist())
        
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        if n_neg > 0:
            target_neg = min(
                max(min_neg, neg_per_pos * max(1, n_pos)),
                n_neg if not replace else max(1, neg_per_pos * max(1, n_pos))
            )
            choose = rng.choice(neg_idx, size=target_neg, replace=replace)
            keep_idx.extend(choose.tolist())
    
    out = df.loc[keep_idx].copy()
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def make_train_frame(train_part_df, mode, label_col, group_col,
                     neg_pos_ratio=5, boot_rounds=10, seed=42):
    """Create training frame with optional sampling."""
    if mode == "class_weight":
        return train_part_df.copy()
    
    if mode == "undersample_simple":
        return per_child_undersample(
            train_part_df, label_col, group_col,
            neg_per_pos=neg_pos_ratio, min_neg=1,
            pos_label=1, random_state=seed, replace=False
        )
    
    if mode == "undersample_bootstrap":
        bags = []
        for b in range(boot_rounds):
            bags.append(per_child_undersample(
                train_part_df, label_col, group_col,
                neg_per_pos=neg_pos_ratio, min_neg=1,
                pos_label=1, random_state=seed + b, replace=True
            ))
        return pd.concat(bags, axis=0, ignore_index=True)
    
    raise ValueError(f"Unknown sampling mode: {mode}")


def get_oof_raw_score(estimator, train_part, feat_cols, label_col, group_col,
                      n_splits=5, random_state=42, sampling_mode="class_weight",
                      neg_pos_ratio=5, boot_rounds=10):
    """Generate out-of-fold raw scores for calibration."""
    n_splits = min(n_splits, max(2, train_part[group_col].nunique() // 2))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_raw = np.zeros(len(train_part), dtype=float)
    
    y_true = train_part[label_col].values
    groups = train_part[group_col].values
    
    for tr, va in cv.split(train_part[feat_cols], y_true, groups):
        fold_train = train_part.iloc[tr].copy()
        fold_val = train_part.iloc[va].copy()
        
        fold_train = make_train_frame(
            fold_train, mode=sampling_mode,
            label_col=label_col, group_col=group_col,
            neg_pos_ratio=neg_pos_ratio, boot_rounds=boot_rounds, seed=random_state
        )
        
        est = clone(estimator)
        est.fit(fold_train[feat_cols], fold_train[label_col])
        
        if hasattr(est, "decision_function"):
            oof_raw[va] = est.decision_function(fold_val[feat_cols])
        else:
            oof_raw[va] = est.predict_proba(fold_val[feat_cols])[:, 1]
    
    return oof_raw


# ====================
# Main Training
# ====================
def train_model(args):
    """Main training function."""
    
    # Configuration
    LABEL_COL = "label"
    GROUP_COL = "group_id"
    CHILD_ID_COL = "id"
    VAR_THRESH = 1e-5
    SEED = args.seed
    RECALL_TARGET = args.recall_target
    SAMPLING_MODE = args.sampling_mode
    NEG_POS_RATIO = args.neg_pos_ratio
    BOOT_ROUNDS = args.boot_rounds
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ====================
    # 1) Load Data
    # ====================
    log("🔹 Loading sample info...")
    sample_df = pd.read_csv(args.sample_list, sep="\t")
    child_ids = sample_df.loc[sample_df["ROLE"] == "child", "subject_id"].tolist()
    
    log("🔹 Loading feature TSVs...")
    feat_list = []
    for sid in tqdm(child_ids, desc="Loading features"):
        # Try different file patterns
        patterns = [
            f"{sid}.*.features.*.tsv",
            f"{sid}_features.tsv",
            f"{sid}.tsv"
        ]
        
        found = False
        for pattern in patterns:
            import glob
            matches = glob.glob(os.path.join(args.feature_dir, pattern.replace("*", "*")))
            if not matches:
                # Try exact pattern
                matches = glob.glob(os.path.join(args.feature_dir, f"{sid}*.tsv"))
            
            if matches:
                feat_list.append(pd.read_csv(matches[0], sep="\t" if matches[0].endswith('.tsv') else ","))
                found = True
                break
        
        if not found:
            log(f"  WARNING: No feature file found for {sid}")
    
    if not feat_list:
        sys.exit("❌ No feature files found!")
    
    ft = pd.concat(feat_list, ignore_index=True)
    log(f"  Loaded {len(ft):,} total variants from {len(feat_list)} samples")
    
    # ====================
    # 2) Preprocessing
    # ====================
    log("🔹 Preprocessing...")
    all_exclude = set(ID_COLS + EXCLUDE_FROM_FEATURES)
    feat_cols_all = [c for c in ft.columns if c not in all_exclude]
    
    # Remove features with >40% missing
    na_ratio = ft[feat_cols_all].isna().mean()
    keep_cols = na_ratio[na_ratio <= 0.40].index.tolist()
    
    df_tmp = ft[keep_cols].copy()
    
    # Handle categorical columns
    cat_cols = df_tmp.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols):
        df_tmp = pd.get_dummies(df_tmp, columns=cat_cols, dummy_na=True)
    
    # Median imputation
    num_cols = df_tmp.select_dtypes(include=[np.number]).columns
    df_tmp[num_cols] = df_tmp[num_cols].fillna(df_tmp[num_cols].median(numeric_only=True))
    
    # Combine with ID columns
    id_cols_present = [c for c in ID_COLS if c in ft.columns]
    proc = pd.concat([ft[id_cols_present].reset_index(drop=True), df_tmp.reset_index(drop=True)], axis=1)
    
    # ====================
    # 3) Labeling
    # ====================
    log("🔹 Labeling & grouping...")
    
    true_ids_df = pd.read_csv(args.true_variants, sep="\t")
    
    # Create variant_ID if needed
    if "variant_ID" not in true_ids_df.columns:
        if "locus_38" in true_ids_df.columns:
            true_ids_df["variant_ID"] = (
                true_ids_df["locus_38"].astype(str) + ":" +
                true_ids_df["ref"].astype(str) + ":" +
                true_ids_df["alt"].astype(str)
            )
        elif "ID" in true_ids_df.columns:
            true_ids_df["variant_ID"] = true_ids_df["ID"]
        else:
            sys.exit("❌ Cannot determine variant IDs from true variants file")
    
    true_ids = true_ids_df["variant_ID"].tolist()
    
    # Define labels: True = validated DNV with AC=1, False = AC>=2
    is_true = (proc["variant_ID"].isin(true_ids)) & (proc.get("internal_AC", 0) == 1)
    is_false = (proc.get("internal_AC", 0) >= 2)
    
    proc[GROUP_COL] = proc[CHILD_ID_COL]
    train_df = proc[is_true | is_false].copy()
    train_df[LABEL_COL] = np.where(is_true[train_df.index], 1, 0)
    
    if train_df[GROUP_COL].isna().any():
        train_df = train_df.dropna(subset=[GROUP_COL]).copy()
    
    # Get feature columns
    FEATURE_EXCLUDE = set(id_cols_present + [LABEL_COL, GROUP_COL, "id", "group_id"])
    feat_cols_model = [c for c in train_df.columns if c not in FEATURE_EXCLUDE]
    
    # Verify numeric
    non_num = train_df[feat_cols_model].select_dtypes(exclude=[np.number]).columns.tolist()
    assert len(non_num) == 0, f"Non-numeric features: {non_num}"
    
    log(f"  Training set: {len(train_df):,} variants ({train_df[LABEL_COL].sum()} true, {(train_df[LABEL_COL]==0).sum()} false)")
    log(f"  Features: {len(feat_cols_model)}")
    
    # ====================
    # 4) LOGO Cross-Validation
    # ====================
    log("🔹 Starting LOGO cross-validation...")
    
    # Handle excluded samples
    exclude_from_test = args.exclude_from_test.split(",") if args.exclude_from_test else []
    train_df_for_logo = train_df[~train_df[GROUP_COL].isin(exclude_from_test)].copy()
    train_df_excluded = train_df[train_df[GROUP_COL].isin(exclude_from_test)].copy()
    
    log(f"  LOGO samples: {train_df_for_logo[GROUP_COL].nunique()}")
    if len(train_df_excluded) > 0:
        log(f"  Excluded from test: {train_df_excluded[GROUP_COL].nunique()} samples")
    
    # Pipeline and hyperparameters
    param_distributions = {
        'l1sel__estimator__C': [0.1, 0.5, 1, 2, 5, 10],
        'clf__C': [0.3, 1, 3, 10],
    }
    
    base_pipe = Pipeline([
        ('vt', VarianceThreshold(threshold=VAR_THRESH)),
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('l1sel', SelectFromModel(
            LogisticRegression(penalty='l1', solver='saga', max_iter=5000, tol=1e-2, random_state=SEED),
            threshold=0.0, importance_getter='coef_'
        )),
        ('clf', LogisticRegression(penalty='l2', solver='liblinear', max_iter=5000, tol=1e-2, random_state=SEED))
    ])
    
    outer = LeaveOneGroupOut()
    outer_results = []
    best_params_all = []
    logo_true_all = []
    logo_prob_all = []
    
    for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(
        train_df_for_logo[feat_cols_model],
        train_df_for_logo[LABEL_COL],
        train_df_for_logo[GROUP_COL]
    )):
        outer_train_logo = train_df_for_logo.iloc[tr_idx].copy()
        outer_test = train_df_for_logo.iloc[te_idx].copy()
        
        # Add excluded samples to training
        outer_train = pd.concat([outer_train_logo, train_df_excluded], axis=0, ignore_index=True)
        
        test_group = outer_test[GROUP_COL].unique()[0]
        log(f"  Fold {fold_idx}: testing on {test_group}")
        
        n_groups = outer_train[GROUP_COL].nunique()
        n_inner = max(2, min(5, n_groups))
        inner_cv = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=SEED)
        
        train_bal = make_train_frame(
            outer_train, mode=SAMPLING_MODE, label_col=LABEL_COL, group_col=GROUP_COL,
            neg_pos_ratio=NEG_POS_RATIO, boot_rounds=BOOT_ROUNDS, seed=SEED
        )
        
        rand = RandomizedSearchCV(
            base_pipe, param_distributions, n_iter=12, scoring="average_precision",
            cv=inner_cv, n_jobs=-1, refit=True, random_state=SEED, verbose=0
        )
        
        rand.fit(train_bal[feat_cols_model], train_bal[LABEL_COL], groups=train_bal[GROUP_COL])
        best = rand.best_estimator_
        best_params_all.append(rand.best_params_)
        
        prob = _score(best, outer_test[feat_cols_model])
        metrics = evaluate_binary(outer_test[LABEL_COL].values, prob, RECALL_TARGET)
        metrics['fold'] = fold_idx
        metrics['test_sample'] = test_group
        metrics['n_test'] = len(outer_test)
        metrics['n_true'] = int(outer_test[LABEL_COL].sum())
        metrics['n_false'] = int((outer_test[LABEL_COL] == 0).sum())
        
        outer_results.append(metrics)
        logo_true_all.append(outer_test[LABEL_COL].values)
        logo_prob_all.append(prob)
    
    # Summary
    outer_df = pd.DataFrame(outer_results)
    
    print("\n" + "=" * 80)
    print("=== LOGO Cross-Validation Results ===")
    print("=" * 80)
    col_order = ['fold', 'test_sample', 'n_test', 'n_true', 'n_false',
                 'PR_AUC', 'AUROC', 'thr', 'Prec@thr', 'Recall@thr']
    print(outer_df[col_order].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("=== Summary (mean ± std) ===")
    print("=" * 80)
    summary_cols = ['PR_AUC', 'AUROC', 'thr', 'Prec@thr', 'Recall@thr']
    print(outer_df[summary_cols].agg(['mean', 'std']).T.to_string())
    print("=" * 80 + "\n")
    
    # ====================
    # 5) Final Model Training
    # ====================
    log("🔹 Training final model...")
    
    best_params_majority = dict(Counter([tuple(sorted(d.items())) for d in best_params_all]).most_common(1)[0][0])
    
    final_train = make_train_frame(
        train_df, mode=SAMPLING_MODE, label_col=LABEL_COL, group_col=GROUP_COL,
        neg_pos_ratio=NEG_POS_RATIO, boot_rounds=BOOT_ROUNDS, seed=SEED
    )
    
    final_pipe = Pipeline([
        ('vt', VarianceThreshold(threshold=VAR_THRESH)),
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('l1sel', SelectFromModel(
            LogisticRegression(penalty='l1', solver='saga', max_iter=5000, tol=1e-2, random_state=SEED),
            threshold=0.0, importance_getter='coef_'
        )),
        ('clf', LogisticRegression(penalty='l2', solver='liblinear', max_iter=5000, tol=1e-2, random_state=SEED))
    ])
    final_pipe.set_params(**best_params_majority)
    final_pipe.fit(final_train[feat_cols_model], final_train[LABEL_COL])
    
    # ====================
    # 6) Probability Calibration
    # ====================
    log("🔹 Training probability calibrator...")
    
    oof_raw = get_oof_raw_score(
        final_pipe, train_df, feat_cols_model, LABEL_COL, GROUP_COL,
        n_splits=min(5, max(2, train_df[GROUP_COL].nunique() // 2)),
        random_state=SEED, sampling_mode=SAMPLING_MODE,
        neg_pos_ratio=NEG_POS_RATIO, boot_rounds=BOOT_ROUNDS
    )
    
    y_true_all = train_df[LABEL_COL].values
    
    # Raw threshold
    thr_raw, p_at_r_raw, r_at_thr_raw = pick_threshold_for_recall(
        y_true_all, oof_raw, recall_min=RECALL_TARGET
    )
    log(f"  [RAW] @R≥{RECALL_TARGET}: thr={thr_raw:.4f} | Prec={p_at_r_raw:.3f}, Recall={r_at_thr_raw:.3f}")
    
    # Sigmoid (Platt) calibration
    sig_calibrator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED)
    sig_calibrator.fit(oof_raw.reshape(-1, 1), y_true_all)
    
    calib_oof_prob = sig_calibrator.predict_proba(oof_raw.reshape(-1, 1))[:, 1]
    thr_final, p_at_r, r_at_thr = pick_threshold_for_recall(
        y_true_all, calib_oof_prob, recall_min=RECALL_TARGET
    )
    log(f"  [CALIBRATED] @R≥{RECALL_TARGET}: thr={thr_final:.4f} | Prec={p_at_r:.3f}, Recall={r_at_thr:.3f}")
    
    # ====================
    # 7) Save Artifacts
    # ====================
    log("🔹 Saving model artifacts...")
    
    # LOGO results
    y_logo_all = np.concatenate(logo_true_all)
    p_logo_all = np.concatenate(logo_prob_all)
    
    np.savez(
        os.path.join(args.output_dir, "logo_results.npz"),
        y_logo_all=y_logo_all,
        p_logo_raw=p_logo_all
    )
    
    # Calibration results
    np.savez(
        os.path.join(args.output_dir, "calibration_results.npz"),
        thr_final=thr_final,
        p_at_r=p_at_r,
        r_at_thr=r_at_thr,
        calib_oof_prob=calib_oof_prob,
        y_true_all=y_true_all,
        oof_raw=oof_raw,
        thr_raw=thr_raw
    )
    
    # Model bundle
    artifacts = {
        "final_pipe": final_pipe,
        "base_features": feat_cols_model,
        "calibrator": sig_calibrator,
        "calibration": {"type": "sigmoid_platt", "input": "raw_score"},
        "final_threshold": float(thr_final),
        "recall_target": RECALL_TARGET,
        "id_cols": id_cols_present,
        "label_col": LABEL_COL,
        "group_col": GROUP_COL,
        "child_id_col_in_feat": CHILD_ID_COL,
        "best_params_majority": best_params_majority,
        "sampling_mode": SAMPLING_MODE,
        "var_threshold": VAR_THRESH,
    }
    
    with open(os.path.join(args.output_dir, "final_model_bundle.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
    
    # LOGO fold results
    outer_df.to_csv(os.path.join(args.output_dir, "logo_fold_results.csv"), index=False)
    
    log("✅ Training complete!")
    log(f"   Model saved to: {os.path.join(args.output_dir, 'final_model_bundle.pkl')}")


def main():
    parser = argparse.ArgumentParser(
        description="A2Denovo Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--feature-dir", required=True,
                        help="Directory containing feature TSV files")
    parser.add_argument("--sample-list", required=True,
                        help="Sample list TSV (columns: subject_id, ROLE)")
    parser.add_argument("--true-variants", required=True,
                        help="True DNV list TSV")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for model artifacts")
    
    # Optional arguments
    parser.add_argument("--recall-target", type=float, default=0.90,
                        help="Target recall for threshold selection (default: 0.90)")
    parser.add_argument("--sampling-mode", default="class_weight",
                        choices=["class_weight", "undersample_simple", "undersample_bootstrap"],
                        help="Sampling strategy for class imbalance")
    parser.add_argument("--neg-pos-ratio", type=int, default=5,
                        help="Negative to positive ratio for undersampling")
    parser.add_argument("--boot-rounds", type=int, default=10,
                        help="Bootstrap rounds for undersample_bootstrap mode")
    parser.add_argument("--exclude-from-test", default="",
                        help="Comma-separated sample IDs to exclude from LOGO testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
