"""
OffScroll Preliminary Model Fitting — Task #239
Ada (Algorithm Architect), IRAS

Fits technical proficiency and style models on batch-001 graded spreads.
Generates diagnostics, plots, and the preliminary report data.
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(OUT_DIR, "..", "features", "features.csv")
GRADES_PATH = os.path.join(OUT_DIR, "..", "grades", "batch-001.csv")

# ── Load and join ──────────────────────────────────────────────────

df = pd.read_csv(FEATURES_PATH)
grades = pd.read_csv(GRADES_PATH)

# Features CSV already has technical_grade and style_grade merged in,
# plus the split column. Verify consistency with grades CSV.
assert set(df["spread_id"]) == set(grades["spread_id"]), "Spread IDs don't match"

# Drop NA columns (rendered-output features not available)
na_cols = [c for c in df.columns if df[c].isna().all() or (df[c] == "NA").all()]
print(f"Dropping NA columns: {na_cols}")
df = df.drop(columns=na_cols)

# Drop non-feature columns
id_cols = ["spread_id", "split", "technical_grade", "style_grade",
           "spread_type", "page_role"]  # categorical strings
target_tech = df["technical_grade"].values
target_style = df["style_grade"].values
split = df["split"].values

# One-hot encode categoricals
df_enc = pd.get_dummies(df.drop(columns=["spread_id", "split",
                                          "technical_grade", "style_grade"]),
                        columns=["spread_type", "page_role"],
                        drop_first=True)

feature_names = list(df_enc.columns)
X = df_enc.values.astype(float)

train_mask = split == "train"
val_mask = split == "val"

X_train, X_val = X[train_mask], X[val_mask]
y_tech_train, y_tech_val = target_tech[train_mask], target_tech[val_mask]
y_style_train, y_style_val = target_style[train_mask], target_style[val_mask]

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Features: {X_train.shape[1]}")
print(f"Tech grades — train mean: {y_tech_train.mean():.2f}, val mean: {y_tech_val.mean():.2f}")
print(f"Style grades — train mean: {y_style_train.mean():.2f}, val mean: {y_style_val.mean():.2f}")

# ── Model fitting ──────────────────────────────────────────────────

results = {}

for target_name, y_train, y_val in [
    ("technical", y_tech_train, y_tech_val),
    ("style", y_style_train, y_style_val),
]:
    results[target_name] = {}

    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    pred_train_r = ridge.predict(X_train)
    pred_val_r = ridge.predict(X_val)

    results[target_name]["ridge"] = {
        "r2_train": r2_score(y_train, pred_train_r),
        "r2_val": r2_score(y_val, pred_val_r),
        "mae_train": mean_absolute_error(y_train, pred_train_r),
        "mae_val": mean_absolute_error(y_val, pred_val_r),
        "pred_train": pred_train_r,
        "pred_val": pred_val_r,
        "coefs": dict(zip(feature_names, ridge.coef_)),
    }

    # Gradient-boosted tree (conservative hyperparams for n=40)
    gbt = GradientBoostingRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.1,
        min_samples_leaf=5, subsample=0.8, random_state=42
    )
    gbt.fit(X_train, y_train)
    pred_train_g = gbt.predict(X_train)
    pred_val_g = gbt.predict(X_val)

    results[target_name]["gbt"] = {
        "r2_train": r2_score(y_train, pred_train_g),
        "r2_val": r2_score(y_val, pred_val_g),
        "mae_train": mean_absolute_error(y_train, pred_train_g),
        "mae_val": mean_absolute_error(y_val, pred_val_g),
        "pred_train": pred_train_g,
        "pred_val": pred_val_g,
        "importances": dict(zip(feature_names, gbt.feature_importances_)),
    }

    print(f"\n{'='*60}")
    print(f"  {target_name.upper()} MODEL RESULTS")
    print(f"{'='*60}")
    for model_name in ["ridge", "gbt"]:
        r = results[target_name][model_name]
        print(f"\n  {model_name.upper()}:")
        print(f"    R² train: {r['r2_train']:.3f}  |  R² val: {r['r2_val']:.3f}")
        print(f"    MAE train: {r['mae_train']:.2f}  |  MAE val: {r['mae_val']:.2f}")

# ── Feature importances ───────────────────────────────────────────

print(f"\n{'='*60}")
print("  FEATURE IMPORTANCES (GBT)")
print(f"{'='*60}")

for target_name in ["technical", "style"]:
    imp = results[target_name]["gbt"]["importances"]
    sorted_imp = sorted(imp.items(), key=lambda x: -abs(x[1]))
    print(f"\n  {target_name.upper()} — Top 10:")
    for fname, fval in sorted_imp[:10]:
        print(f"    {fname:30s} {fval:.4f}")

# ── Ridge coefficients (top features) ────────────────────────────

print(f"\n{'='*60}")
print("  RIDGE COEFFICIENTS (Top 10 by |coef|)")
print(f"{'='*60}")

for target_name in ["technical", "style"]:
    coefs = results[target_name]["ridge"]["coefs"]
    sorted_coefs = sorted(coefs.items(), key=lambda x: -abs(x[1]))
    print(f"\n  {target_name.upper()}:")
    for fname, fval in sorted_coefs[:10]:
        print(f"    {fname:30s} {fval:+.4f}")

# ── Diagnostic plots ──────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for row, (target_name, y_tr, y_v) in enumerate([
    ("technical", y_tech_train, y_tech_val),
    ("style", y_style_train, y_style_val),
]):
    for col, model_name in enumerate(["ridge", "gbt"]):
        ax = axes[row, col]
        r = results[target_name][model_name]

        ax.scatter(y_tr, r["pred_train"], alpha=0.6, label="train", c="steelblue", s=40)
        ax.scatter(y_v, r["pred_val"], alpha=0.8, label="val", c="tomato", s=60, marker="^")

        # Perfect prediction line
        lims = [0.5, 7]
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual Grade")
        ax.set_ylabel("Predicted Grade")
        ax.set_title(f"{target_name.title()} — {model_name.upper()}\n"
                     f"R²={r['r2_val']:.3f}, MAE={r['mae_val']:.2f} (val)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_aspect("equal")

plt.suptitle("Preliminary Model Fitting: Predicted vs Actual Grades\n"
             "Batch 001 (50 spreads, 40 train / 10 val)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pred_vs_actual.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: pred_vs_actual.png")

# ── Residual analysis (style model) ──────────────────────────────

# Use GBT style model for residual analysis
style_gbt_pred_all = np.concatenate([
    results["style"]["gbt"]["pred_train"],
    results["style"]["gbt"]["pred_val"]
])
y_style_all = np.concatenate([y_style_train, y_style_val])
spread_ids_all = np.concatenate([
    df["spread_id"].values[train_mask],
    df["spread_id"].values[val_mask]
])

residuals = np.abs(y_style_all - style_gbt_pred_all)
resid_order = np.argsort(-residuals)

print(f"\n{'='*60}")
print("  TOP 10 HIGHEST-RESIDUAL SPREADS (Style GBT)")
print(f"{'='*60}")
print(f"  {'Spread ID':15s} {'Actual':>6s} {'Predicted':>9s} {'|Residual|':>10s}")
for i in resid_order[:10]:
    print(f"  {spread_ids_all[i]:15s} {y_style_all[i]:6.0f} {style_gbt_pred_all[i]:9.2f} {residuals[i]:10.2f}")

# ── Feature correlation matrix ────────────────────────────────────

# Use numeric features only
numeric_features = df_enc.columns.tolist()
corr_matrix = df_enc.corr()

fig2, ax2 = plt.subplots(1, 1, figsize=(14, 12))
im = ax2.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax2.set_xticks(range(len(numeric_features)))
ax2.set_yticks(range(len(numeric_features)))
ax2.set_xticklabels(numeric_features, rotation=90, fontsize=7)
ax2.set_yticklabels(numeric_features, fontsize=7)
plt.colorbar(im, ax=ax2, fraction=0.046)
ax2.set_title("Feature Correlation Matrix (Batch 001)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_correlation.png"), dpi=150, bbox_inches="tight")
print(f"Saved: feature_correlation.png")

# Print highly correlated pairs (|r| > 0.85)
print(f"\n{'='*60}")
print("  HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.85)")
print(f"{'='*60}")
for i in range(len(numeric_features)):
    for j in range(i+1, len(numeric_features)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.85:
            print(f"  {numeric_features[i]:30s} × {numeric_features[j]:30s}  r={r:.3f}")

# ── Bootstrap confidence intervals on R² ──────────────────────────

print(f"\n{'='*60}")
print("  BOOTSTRAP CONFIDENCE INTERVALS ON R² (1000 iterations)")
print(f"{'='*60}")

np.random.seed(42)

for target_name, y_full in [("technical", target_tech), ("style", target_style)]:
    # Bootstrap on the full dataset (small n, so use all 50)
    r2_boot = []
    for _ in range(1000):
        idx = resample(np.arange(len(y_full)), random_state=None)
        oob = np.setdiff1d(np.arange(len(y_full)), idx)
        if len(oob) < 3:
            continue
        model = Ridge(alpha=1.0)
        model.fit(X[idx], y_full[idx])
        pred_oob = model.predict(X[oob])
        r2 = r2_score(y_full[oob], pred_oob)
        r2_boot.append(r2)

    r2_boot = np.array(r2_boot)
    lo, hi = np.percentile(r2_boot, [2.5, 97.5])
    print(f"\n  {target_name.upper()} (Ridge, OOB bootstrap):")
    print(f"    Median R²: {np.median(r2_boot):.3f}")
    print(f"    95% CI: [{lo:.3f}, {hi:.3f}]")
    print(f"    Fraction R² < 0: {(r2_boot < 0).mean():.1%}")

# ── Learning curve analysis ───────────────────────────────────────

print(f"\n{'='*60}")
print("  LEARNING CURVE ANALYSIS (Ridge, 5-fold cross-val style)")
print(f"{'='*60}")

train_sizes = [15, 20, 25, 30, 35, 40]
for target_name, y_full in [("technical", target_tech), ("style", target_style)]:
    print(f"\n  {target_name.upper()}:")
    print(f"  {'n_train':>8s} {'mean_R²_val':>12s} {'std_R²_val':>12s}")
    lc_means = []
    lc_stds = []
    for n_tr in train_sizes:
        r2s = []
        for _ in range(200):
            perm = np.random.permutation(len(y_full))
            tr_idx = perm[:n_tr]
            val_idx = perm[n_tr:]
            if len(val_idx) < 3:
                continue
            m = Ridge(alpha=1.0)
            m.fit(X[tr_idx], y_full[tr_idx])
            p = m.predict(X[val_idx])
            r2s.append(r2_score(y_full[val_idx], p))
        r2s = np.array(r2s)
        lc_means.append(np.mean(r2s))
        lc_stds.append(np.std(r2s))
        print(f"  {n_tr:>8d} {np.mean(r2s):>12.3f} {np.std(r2s):>12.3f}")

# ── Save all numeric results to JSON ──────────────────────────────

summary = {}
for target_name in ["technical", "style"]:
    summary[target_name] = {}
    for model_name in ["ridge", "gbt"]:
        r = results[target_name][model_name]
        summary[target_name][model_name] = {
            "r2_train": round(r["r2_train"], 4),
            "r2_val": round(r["r2_val"], 4),
            "mae_train": round(r["mae_train"], 3),
            "mae_val": round(r["mae_val"], 3),
        }
        if "importances" in r:
            top_imp = sorted(r["importances"].items(), key=lambda x: -x[1])[:10]
            summary[target_name][model_name]["top_features"] = [
                {"feature": f, "importance": round(v, 4)} for f, v in top_imp
            ]
        if "coefs" in r:
            top_coefs = sorted(r["coefs"].items(), key=lambda x: -abs(x[1]))[:10]
            summary[target_name][model_name]["top_features"] = [
                {"feature": f, "coef": round(v, 4)} for f, v in top_coefs
            ]

with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved: results_summary.json")
print("\nDone.")
