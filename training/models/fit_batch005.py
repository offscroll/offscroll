"""
Re-run go/no-go gate at n=200 with the expanded visual-hierarchy
feature set (features-006.csv).
Belle — Task #378 (follow-up to #344 / #346).

Gate criteria (same as #344):
  Val R² > 0.5   → GO
  Val R² 0.4-0.5 → QUALIFIED GO (deploy with caveats)
  Val R² < 0.4   → NO-GO (need new features)

Reports 5-fold CV R² and bootstrap CI for both technical and
style models. Comparison to #344 baseline. Same train/val split
seed 42 over lex-sorted IDs so the baselines line up.
"""

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

BASE = "/home/modus/offscroll/training"
features_path = f"{BASE}/features/features-006.csv"

df = pd.read_csv(features_path)
print(f"Loaded {len(df)} spreads from features-006.csv")

# ─── New visual-hierarchy feature health ───
print("\n" + "=" * 60)
print("NEW VISUAL-HIERARCHY FEATURE HEALTH")
print("=" * 60)
new_features = [
    "h_distinct_font_sizes", "h_size_std_chars", "h_max_size_to_body",
    "h_distinct_weights", "h_bold_char_frac", "h_italic_char_frac",
    "h_sans_char_frac", "h_block_area_max_to_median", "h_block_area_cv",
    "h_gap_cv", "h_max_gap_to_median", "h_headline_count",
    "h_headline_area_frac", "h_pull_quote_count",
]
for col in new_features:
    na = df[col].isna().sum()
    nz = (df[col] != 0).sum()
    print(
        f"  {col:30s}: {na} NA, {nz}/{len(df)} non-zero, "
        f"range=[{df[col].min():.4f}, {df[col].max():.4f}]"
    )

# ─── Original features ───
print("\nOriginal d-features (carried forward from features-005):")
prior_d = ["d2_orphans", "d2_widows", "d4_col_balance", "d6_dead_space"]
for col in prior_d:
    na = df[col].isna().sum()
    nz = (df[col] != 0).sum()
    print(
        f"  {col:30s}: {na} NA, {nz}/{len(df)} non-zero, "
        f"range=[{df[col].min():.4f}, {df[col].max():.4f}]"
    )

# ─── Define feature columns ───
binary_cols = ["is_solo", "is_front", "is_terminal"]
numeric_feature_cols = [
    "n_pages_in_spread", "edition_page_count", "page_position_frac",
    "edition_item_count", "edition_word_count_total", "edition_word_count_mean",
    "edition_brief_frac", "edition_standard_frac", "edition_template_entropy",
    "edition_image_count_total", "edition_source_count", "edition_section_count",
    "est_item_count", "est_items_per_page", "est_word_count",
    "est_words_per_page", "est_word_count_mean", "est_brief_count",
    "est_standard_count", "est_image_count", "est_source_count",
    "d3_image_fraction", "d5_fill_fraction", "d7_template_entropy",
    "d8_word_count_cv", "anchor_strength",
    "d2_orphans", "d2_widows", "d4_col_balance", "d6_dead_space",
    # New visual-hierarchy features:
    *new_features,
]
all_feature_cols = binary_cols + numeric_feature_cols
df[all_feature_cols] = df[all_feature_cols].fillna(0)

print(f"\nTotal raw features: {len(all_feature_cols)} "
      f"({len(new_features)} new)")

# ─── Redundancy removal (same rules as #344) ───
print("\n" + "=" * 60)
print("REDUNDANCY REMOVAL")
print("=" * 60)

feat_corr = df[all_feature_cols].corr()
print("\n--- Highly correlated feature pairs (|r| > 0.85) ---")
high_corr_pairs = []
for i in range(len(all_feature_cols)):
    for j in range(i + 1, len(all_feature_cols)):
        r = feat_corr.iloc[i, j]
        if abs(r) > 0.85:
            print(f"  {all_feature_cols[i]} <-> {all_feature_cols[j]}: r={r:.3f}")
            high_corr_pairs.append((all_feature_cols[i], all_feature_cols[j], float(r)))

# Same drop-list as #344, plus we let the new features compete
to_drop = [
    "is_front", "is_terminal", "n_pages_in_spread",
    "edition_standard_frac",
    "est_word_count", "est_item_count", "est_standard_count",
    "d8_word_count_cv", "est_word_count_mean",
    "edition_word_count_total", "edition_word_count_mean",
    "edition_item_count", "edition_image_count_total",
    "edition_source_count", "edition_section_count",
]
# Auto-drop new features that are >0.85 correlated with each other.
auto_drops = []
new_set = set(new_features)
for a, b, r in high_corr_pairs:
    if a in new_set and b in new_set and a not in to_drop and b not in to_drop:
        # drop the second of the pair (alphabetical tiebreak via sort)
        drop = sorted([a, b])[1]
        if drop not in auto_drops:
            auto_drops.append(drop)
if auto_drops:
    print(f"\nAuto-dropping new features with |r|>0.85: {auto_drops}")
    to_drop.extend(auto_drops)

reduced_cols = [c for c in all_feature_cols if c not in to_drop]
print(f"\nDropped {len(to_drop)} redundant features. Remaining: {len(reduced_cols)}")
print("Reduced features:", reduced_cols)

# ─── Split ───
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()
print(f"\nTrain: {len(train_df)}, Validation: {len(val_df)}")

X_train = train_df[reduced_cols].values
X_val = val_df[reduced_cols].values
y_tech_train = train_df["technical_grade"].values
y_tech_val = val_df["technical_grade"].values
y_style_train = train_df["style_grade"].values
y_style_val = val_df["style_grade"].values

print("\n--- Score distributions ---")
print(f"Technical: {df['technical_grade'].value_counts().sort_index().to_dict()}")
print(f"Style:     {df['style_grade'].value_counts().sort_index().to_dict()}")

# =================================================================
# 1. TECHNICAL — alpha sweep + ridge + gbt
# =================================================================
print("\n" + "=" * 60)
print("1. TECHNICAL PROFICIENCY MODEL")
print("=" * 60)

alpha_sweep_tech = {}
for alpha in [0.01, 0.1, 1.0, 10.0]:
    r = Ridge(alpha=alpha).fit(X_train, y_tech_train)
    tr = r2_score(y_tech_train, r.predict(X_train))
    va = r2_score(y_tech_val, r.predict(X_val))
    print(f"  Ridge alpha={alpha}: Train R²={tr:.4f}, Val R²={va:.4f}")
    alpha_sweep_tech[alpha] = {"train_r2": float(tr), "val_r2": float(va)}

ridge_tech01 = Ridge(alpha=0.1).fit(X_train, y_tech_train)
pred_tt01 = ridge_tech01.predict(X_train)
pred_tv01 = ridge_tech01.predict(X_val)
r2_tech_ridge01_train = r2_score(y_tech_train, pred_tt01)
r2_tech_ridge01_val = r2_score(y_tech_val, pred_tv01)
mae_tech_ridge01_train = mean_absolute_error(y_tech_train, pred_tt01)
mae_tech_ridge01_val = mean_absolute_error(y_tech_val, pred_tv01)
print(f"\nRidge (alpha=0.1): Train R²={r2_tech_ridge01_train:.4f} MAE={mae_tech_ridge01_train:.4f} | "
      f"Val R²={r2_tech_ridge01_val:.4f} MAE={mae_tech_ridge01_val:.4f}")

ridge_tech10 = Ridge(alpha=10.0).fit(X_train, y_tech_train)
pred_tt = ridge_tech10.predict(X_train)
pred_tv = ridge_tech10.predict(X_val)
r2_tech_ridge_train = r2_score(y_tech_train, pred_tt)
r2_tech_ridge_val = r2_score(y_tech_val, pred_tv)
mae_tech_ridge_train = mean_absolute_error(y_tech_train, pred_tt)
mae_tech_ridge_val = mean_absolute_error(y_tech_val, pred_tv)
print(f"Ridge (alpha=10): Train R²={r2_tech_ridge_train:.4f} MAE={mae_tech_ridge_train:.4f} | "
      f"Val R²={r2_tech_ridge_val:.4f} MAE={mae_tech_ridge_val:.4f}")

feat_std = np.std(X_train, axis=0); feat_std[feat_std == 0] = 1
ridge_imp_tech = np.abs(ridge_tech01.coef_ * feat_std)
ridge_imp_idx = np.argsort(ridge_imp_tech)[::-1]
print("\n  Top features (Ridge alpha=0.1, standardized |coef|):")
ridge_top_tech = []
for i in ridge_imp_idx:
    print(f"    {reduced_cols[i]:30s} {ridge_imp_tech[i]:.4f} (coef={ridge_tech01.coef_[i]:+.4f})")
    ridge_top_tech.append({
        "feature": reduced_cols[i],
        "importance": float(ridge_imp_tech[i]),
        "coef": float(ridge_tech01.coef_[i]),
    })

gbt_tech = GradientBoostingRegressor(
    n_estimators=50, max_depth=2, learning_rate=0.05,
    min_samples_leaf=5, random_state=42, subsample=0.8,
)
gbt_tech.fit(X_train, y_tech_train)
pred_gtt = gbt_tech.predict(X_train); pred_gtv = gbt_tech.predict(X_val)
r2_tech_gbt_train = r2_score(y_tech_train, pred_gtt)
r2_tech_gbt_val = r2_score(y_tech_val, pred_gtv)
mae_tech_gbt_train = mean_absolute_error(y_tech_train, pred_gtt)
mae_tech_gbt_val = mean_absolute_error(y_tech_val, pred_gtv)
print(f"\nGBT (n=50, depth=2): Train R²={r2_tech_gbt_train:.4f} MAE={mae_tech_gbt_train:.4f} | "
      f"Val R²={r2_tech_gbt_val:.4f} MAE={mae_tech_gbt_val:.4f}")

print("\n  Feature importances (GBT):")
gbt_imp_idx = np.argsort(gbt_tech.feature_importances_)[::-1]
gbt_top_tech = []
for i in gbt_imp_idx:
    if gbt_tech.feature_importances_[i] > 0.001:
        print(f"    {reduced_cols[i]:30s} {gbt_tech.feature_importances_[i]:.4f}")
        gbt_top_tech.append({
            "feature": reduced_cols[i],
            "importance": float(gbt_tech.feature_importances_[i]),
        })

# =================================================================
# 2. STYLE
# =================================================================
print("\n" + "=" * 60)
print("2. STYLE MODEL")
print("=" * 60)

alpha_sweep_style = {}
for alpha in [0.01, 0.1, 1.0, 10.0]:
    r = Ridge(alpha=alpha).fit(X_train, y_style_train)
    tr = r2_score(y_style_train, r.predict(X_train))
    va = r2_score(y_style_val, r.predict(X_val))
    print(f"  Ridge alpha={alpha}: Train R²={tr:.4f}, Val R²={va:.4f}")
    alpha_sweep_style[alpha] = {"train_r2": float(tr), "val_r2": float(va)}

ridge_style01 = Ridge(alpha=0.1).fit(X_train, y_style_train)
pred_st01 = ridge_style01.predict(X_train); pred_sv01 = ridge_style01.predict(X_val)
r2_style_ridge01_train = r2_score(y_style_train, pred_st01)
r2_style_ridge01_val = r2_score(y_style_val, pred_sv01)
mae_style_ridge01_train = mean_absolute_error(y_style_train, pred_st01)
mae_style_ridge01_val = mean_absolute_error(y_style_val, pred_sv01)
print(f"\nRidge (alpha=0.1): Train R²={r2_style_ridge01_train:.4f} MAE={mae_style_ridge01_train:.4f} | "
      f"Val R²={r2_style_ridge01_val:.4f} MAE={mae_style_ridge01_val:.4f}")

ridge_style10 = Ridge(alpha=10.0).fit(X_train, y_style_train)
pred_st = ridge_style10.predict(X_train); pred_sv = ridge_style10.predict(X_val)
r2_style_ridge_train = r2_score(y_style_train, pred_st)
r2_style_ridge_val = r2_score(y_style_val, pred_sv)
mae_style_ridge_train = mean_absolute_error(y_style_train, pred_st)
mae_style_ridge_val = mean_absolute_error(y_style_val, pred_sv)
print(f"Ridge (alpha=10): Train R²={r2_style_ridge_train:.4f} MAE={mae_style_ridge_train:.4f} | "
      f"Val R²={r2_style_ridge_val:.4f} MAE={mae_style_ridge_val:.4f}")

ridge_imp_style = np.abs(ridge_style01.coef_ * feat_std)
ridge_imp_idx_s = np.argsort(ridge_imp_style)[::-1]
print("\n  Top features (Ridge alpha=0.1, standardized |coef|):")
ridge_top_style = []
for i in ridge_imp_idx_s:
    print(f"    {reduced_cols[i]:30s} {ridge_imp_style[i]:.4f} (coef={ridge_style01.coef_[i]:+.4f})")
    ridge_top_style.append({
        "feature": reduced_cols[i],
        "importance": float(ridge_imp_style[i]),
        "coef": float(ridge_style01.coef_[i]),
    })

gbt_style = GradientBoostingRegressor(
    n_estimators=50, max_depth=2, learning_rate=0.05,
    min_samples_leaf=5, random_state=42, subsample=0.8,
)
gbt_style.fit(X_train, y_style_train)
pred_gst = gbt_style.predict(X_train); pred_gsv = gbt_style.predict(X_val)
r2_style_gbt_train = r2_score(y_style_train, pred_gst)
r2_style_gbt_val = r2_score(y_style_val, pred_gsv)
mae_style_gbt_train = mean_absolute_error(y_style_train, pred_gst)
mae_style_gbt_val = mean_absolute_error(y_style_val, pred_gsv)
print(f"\nGBT: Train R²={r2_style_gbt_train:.4f} MAE={mae_style_gbt_train:.4f} | "
      f"Val R²={r2_style_gbt_val:.4f} MAE={mae_style_gbt_val:.4f}")

print("\n  Feature importances (GBT):")
gbt_imp_idx_s = np.argsort(gbt_style.feature_importances_)[::-1]
gbt_top_style = []
for i in gbt_imp_idx_s:
    if gbt_style.feature_importances_[i] > 0.001:
        print(f"    {reduced_cols[i]:30s} {gbt_style.feature_importances_[i]:.4f}")
        gbt_top_style.append({
            "feature": reduced_cols[i],
            "importance": float(gbt_style.feature_importances_[i]),
        })

# Top residuals
style_resid = np.abs(y_style_val - pred_gsv)
resid_order = np.argsort(style_resid)[::-1]
print("\n  Top residual spreads (style, GBT, val):")
top_resids = []
for idx in resid_order[:10]:
    sid = val_df.iloc[idx]["spread_id"]
    a = y_style_val[idx]; p = pred_gsv[idx]; r = style_resid[idx]
    print(f"    {sid}: actual={a}, predicted={p:.2f}, residual={r:.2f}")
    top_resids.append({"spread_id": sid, "actual": float(a),
                       "predicted": float(p), "residual": float(r)})

tech_resid = np.abs(y_tech_val - pred_gtv)
tech_resid_order = np.argsort(tech_resid)[::-1]
print("\n  Top residual spreads (technical, GBT, val):")
top_resids_tech = []
for idx in tech_resid_order[:10]:
    sid = val_df.iloc[idx]["spread_id"]
    a = y_tech_val[idx]; p = pred_gtv[idx]; r = tech_resid[idx]
    print(f"    {sid}: actual={a}, predicted={p:.2f}, residual={r:.2f}")
    top_resids_tech.append({"spread_id": sid, "actual": float(a),
                            "predicted": float(p), "residual": float(r)})

# =================================================================
# 3. DIAGNOSTICS
# =================================================================
print("\n" + "=" * 60)
print("3. DIAGNOSTICS")
print("=" * 60)

# Pred vs actual scatter
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax = axes[0, 0]
ax.scatter(y_tech_train, pred_tt01, alpha=0.5, label="Train")
ax.scatter(y_tech_val, pred_tv01, alpha=0.8, marker="x", s=80, label="Val")
ax.plot([1, 7], [1, 7], "k--", alpha=0.3)
ax.set_xlabel("Actual Technical"); ax.set_ylabel("Predicted")
ax.set_title(f"Technical — Ridge α=0.1 (Val R²={r2_tech_ridge01_val:.3f})"); ax.legend()

ax = axes[0, 1]
ax.scatter(y_tech_train, pred_gtt, alpha=0.5, label="Train")
ax.scatter(y_tech_val, pred_gtv, alpha=0.8, marker="x", s=80, label="Val")
ax.plot([1, 7], [1, 7], "k--", alpha=0.3)
ax.set_xlabel("Actual Technical"); ax.set_ylabel("Predicted")
ax.set_title(f"Technical — GBT (Val R²={r2_tech_gbt_val:.3f})"); ax.legend()

ax = axes[1, 0]
ax.scatter(y_style_train, pred_st01, alpha=0.5, label="Train")
ax.scatter(y_style_val, pred_sv01, alpha=0.8, marker="x", s=80, label="Val")
ax.plot([1, 7], [1, 7], "k--", alpha=0.3)
ax.set_xlabel("Actual Style"); ax.set_ylabel("Predicted")
ax.set_title(f"Style — Ridge α=0.1 (Val R²={r2_style_ridge01_val:.3f})"); ax.legend()

ax = axes[1, 1]
ax.scatter(y_style_train, pred_gst, alpha=0.5, label="Train")
ax.scatter(y_style_val, pred_gsv, alpha=0.8, marker="x", s=80, label="Val")
ax.plot([1, 7], [1, 7], "k--", alpha=0.3)
ax.set_xlabel("Actual Style"); ax.set_ylabel("Predicted")
ax.set_title(f"Style — GBT (Val R²={r2_style_gbt_val:.3f})"); ax.legend()

plt.tight_layout()
plt.savefig(f"{BASE}/models/pred_vs_actual_005.png", dpi=150)
plt.close()
print("Saved: pred_vs_actual_005.png")

# Correlation matrix
feat_corr_red = pd.DataFrame(X_train, columns=reduced_cols).corr()
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(feat_corr_red.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(reduced_cols))); ax.set_yticks(range(len(reduced_cols)))
ax.set_xticklabels(reduced_cols, rotation=90, fontsize=7)
ax.set_yticklabels(reduced_cols, fontsize=7)
plt.colorbar(im); plt.title("Feature Correlation (reduced, batch-005 / features-006)")
plt.tight_layout()
plt.savefig(f"{BASE}/models/feature_correlation_005.png", dpi=150)
plt.close()
print("Saved: feature_correlation_005.png")

# Remaining high correlations
print("\n--- Remaining high correlations (|r| > 0.85) ---")
remaining_hc = []
for i in range(len(reduced_cols)):
    for j in range(i + 1, len(reduced_cols)):
        r = feat_corr_red.iloc[i, j]
        if abs(r) > 0.85:
            print(f"  {reduced_cols[i]} <-> {reduced_cols[j]}: r={r:.3f}")
            remaining_hc.append({"a": reduced_cols[i], "b": reduced_cols[j], "r": float(r)})
if not remaining_hc:
    print("  None — multicollinearity resolved")

# Bimodal separation — include new visual features
print("\n--- Bimodal separation (T>=5 vs T<=3) ---")
high = df[df["technical_grade"] >= 5]
low = df[df["technical_grade"] <= 3]
print(f"High: {len(high)}, Low: {len(low)}, Middle (T=4): {(df['technical_grade']==4).sum()}")

sep_features = [
    "d5_fill_fraction", "d6_dead_space", "d4_col_balance",
    "d2_orphans", "d2_widows",
    "est_items_per_page", "est_words_per_page",
    "anchor_strength", "is_solo", "est_brief_count",
    "page_position_frac",
] + new_features
print(f"  {'Feature':30s} {'High':>10s} {'Low':>10s} {'Diff':>10s} {'t-stat':>8s} {'p':>10s}")
sep_results = []
for f in sep_features:
    if f in df.columns:
        h = high[f].values; l = low[f].values
        if np.std(h) > 0 or np.std(l) > 0:
            t, p = stats.ttest_ind(h, l, equal_var=False)
            sig = "*" if p < 0.05 else ""
            print(f"  {f:30s} {h.mean():10.3f} {l.mean():10.3f} "
                  f"{h.mean()-l.mean():10.3f} {t:7.2f}{sig} {p:10.4g}")
            sep_results.append({
                "feature": f, "high_mean": float(h.mean()), "low_mean": float(l.mean()),
                "diff": float(h.mean() - l.mean()), "t": float(t), "p": float(p),
            })

# Classification accuracy (GBT, threshold = 4)
mask_h_v = y_tech_val >= 5; mask_l_v = y_tech_val <= 3
hi_corr_v = (pred_gtv[mask_h_v] > 4).mean() if mask_h_v.sum() else None
lo_corr_v = (pred_gtv[mask_l_v] < 4).mean() if mask_l_v.sum() else None
print(f"\n  GBT val-only: T>=5 predicted >4: "
      f"{hi_corr_v*100 if hi_corr_v is not None else 'n/a':.1f}% ({mask_h_v.sum()})")
print(f"  GBT val-only: T<=3 predicted <4: "
      f"{lo_corr_v*100 if lo_corr_v is not None else 'n/a':.1f}% ({mask_l_v.sum()})")

all_pred = np.concatenate([pred_gtt, pred_gtv])
all_actual = np.concatenate([y_tech_train, y_tech_val])
mh = all_actual >= 5; ml = all_actual <= 3
hi_corr_all = (all_pred[mh] > 4).mean()
lo_corr_all = (all_pred[ml] < 4).mean()
print(f"  GBT all data: T>=5 predicted >4: {hi_corr_all*100:.1f}%")
print(f"  GBT all data: T<=3 predicted <4: {lo_corr_all*100:.1f}%")

# Bootstrap CIs at alpha=0.1
print("\n--- Bootstrap CIs (Ridge alpha=0.1, 1000 iters) ---")
np.random.seed(42)
X_all = df[reduced_cols].values
y_t_all = df["technical_grade"].values
y_s_all = df["style_grade"].values

boot_t01, boot_s01 = [], []
for _ in range(1000):
    idx = np.random.choice(len(X_all), size=len(X_all), replace=True)
    oob = np.array([i for i in range(len(X_all)) if i not in set(idx)])
    if len(oob) < 5:
        continue
    rb = Ridge(alpha=0.1).fit(X_all[idx], y_t_all[idx])
    if np.var(y_t_all[oob]) > 0:
        boot_t01.append(r2_score(y_t_all[oob], rb.predict(X_all[oob])))
    rbs = Ridge(alpha=0.1).fit(X_all[idx], y_s_all[idx])
    if np.var(y_s_all[oob]) > 0:
        boot_s01.append(r2_score(y_s_all[oob], rbs.predict(X_all[oob])))

boot_t01 = np.array(boot_t01); boot_s01 = np.array(boot_s01)
print(f"  Tech (alpha=0.1):  median={np.median(boot_t01):.3f}, "
      f"95% CI=[{np.percentile(boot_t01,2.5):.3f}, {np.percentile(boot_t01,97.5):.3f}]")
print(f"  Style (alpha=0.1): median={np.median(boot_s01):.3f}, "
      f"95% CI=[{np.percentile(boot_s01,2.5):.3f}, {np.percentile(boot_s01,97.5):.3f}]")

# Bootstrap at alpha=10 for comparability
np.random.seed(42)
boot_t, boot_s = [], []
for _ in range(1000):
    idx = np.random.choice(len(X_all), size=len(X_all), replace=True)
    oob = np.array([i for i in range(len(X_all)) if i not in set(idx)])
    if len(oob) < 5:
        continue
    rb = Ridge(alpha=10).fit(X_all[idx], y_t_all[idx])
    if np.var(y_t_all[oob]) > 0:
        boot_t.append(r2_score(y_t_all[oob], rb.predict(X_all[oob])))
    rbs = Ridge(alpha=10).fit(X_all[idx], y_s_all[idx])
    if np.var(y_s_all[oob]) > 0:
        boot_s.append(r2_score(y_s_all[oob], rbs.predict(X_all[oob])))

boot_t = np.array(boot_t); boot_s = np.array(boot_s)
print(f"  Tech (alpha=10):   median={np.median(boot_t):.3f}, "
      f"95% CI=[{np.percentile(boot_t,2.5):.3f}, {np.percentile(boot_t,97.5):.3f}]")
print(f"  Style (alpha=10):  median={np.median(boot_s):.3f}, "
      f"95% CI=[{np.percentile(boot_s,2.5):.3f}, {np.percentile(boot_s,97.5):.3f}]")

# 5-fold CV at full n=200
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_t01, cv_scores_t10, cv_scores_s01, cv_scores_s10 = [], [], [], []
for tr_idx, te_idx in kf.split(X_all):
    rcv = Ridge(alpha=0.1).fit(X_all[tr_idx], y_t_all[tr_idx])
    cv_scores_t01.append(r2_score(y_t_all[te_idx], rcv.predict(X_all[te_idx])))
    rcv = Ridge(alpha=10).fit(X_all[tr_idx], y_t_all[tr_idx])
    cv_scores_t10.append(r2_score(y_t_all[te_idx], rcv.predict(X_all[te_idx])))
    rcv = Ridge(alpha=0.1).fit(X_all[tr_idx], y_s_all[tr_idx])
    cv_scores_s01.append(r2_score(y_s_all[te_idx], rcv.predict(X_all[te_idx])))
    rcv = Ridge(alpha=10).fit(X_all[tr_idx], y_s_all[tr_idx])
    cv_scores_s10.append(r2_score(y_s_all[te_idx], rcv.predict(X_all[te_idx])))
cv_t01_m = float(np.mean(cv_scores_t01)); cv_t01_s = float(np.std(cv_scores_t01))
cv_t10_m = float(np.mean(cv_scores_t10)); cv_t10_s = float(np.std(cv_scores_t10))
cv_s01_m = float(np.mean(cv_scores_s01)); cv_s01_s = float(np.std(cv_scores_s01))
cv_s10_m = float(np.mean(cv_scores_s10)); cv_s10_s = float(np.std(cv_scores_s10))
print(f"\n  5-fold CV @ n=200 — Tech alpha=0.1:  {cv_t01_m:.3f} +/- {cv_t01_s:.3f}")
print(f"  5-fold CV @ n=200 — Tech alpha=10:   {cv_t10_m:.3f} +/- {cv_t10_s:.3f}")
print(f"  5-fold CV @ n=200 — Style alpha=0.1: {cv_s01_m:.3f} +/- {cv_s01_s:.3f}")
print(f"  5-fold CV @ n=200 — Style alpha=10:  {cv_s10_m:.3f} +/- {cv_s10_s:.3f}")

# ───────── Ablation: structural-only vs. structural + visual ─────────
print("\n--- Ablation: structural-only baseline (drop new h_* features) ---")
struct_only = [c for c in reduced_cols if not c.startswith("h_")]
X_all_s = df[struct_only].values

cv_s_t01, cv_s_s01 = [], []
for tr_idx, te_idx in kf.split(X_all_s):
    rcv = Ridge(alpha=0.1).fit(X_all_s[tr_idx], y_t_all[tr_idx])
    cv_s_t01.append(r2_score(y_t_all[te_idx], rcv.predict(X_all_s[te_idx])))
    rcv = Ridge(alpha=0.1).fit(X_all_s[tr_idx], y_s_all[tr_idx])
    cv_s_s01.append(r2_score(y_s_all[te_idx], rcv.predict(X_all_s[te_idx])))
cv_s_t01_m = float(np.mean(cv_s_t01)); cv_s_s01_m = float(np.mean(cv_s_s01))
cv_s_t01_s = float(np.std(cv_s_t01)); cv_s_s01_s = float(np.std(cv_s_s01))
print(f"  Structural-only ({len(struct_only)} features): "
      f"Tech CV={cv_s_t01_m:.3f}+/-{cv_s_t01_s:.3f}, "
      f"Style CV={cv_s_s01_m:.3f}+/-{cv_s_s01_s:.3f}")
print(f"  +Visual ({len(reduced_cols)} features):       "
      f"Tech CV={cv_t01_m:.3f}+/-{cv_t01_s:.3f}, "
      f"Style CV={cv_s01_m:.3f}+/-{cv_s01_s:.3f}")
print(f"  Δ from visual hierarchy:           "
      f"Tech +{cv_t01_m - cv_s_t01_m:+.3f}, Style +{cv_s01_m - cv_s_s01_m:+.3f}")

# ───────── Visual-only baseline ─────────
visual_only = [c for c in reduced_cols if c.startswith("h_")]
X_all_v = df[visual_only].values
cv_v_t01, cv_v_s01 = [], []
for tr_idx, te_idx in kf.split(X_all_v):
    rcv = Ridge(alpha=0.1).fit(X_all_v[tr_idx], y_t_all[tr_idx])
    cv_v_t01.append(r2_score(y_t_all[te_idx], rcv.predict(X_all_v[te_idx])))
    rcv = Ridge(alpha=0.1).fit(X_all_v[tr_idx], y_s_all[tr_idx])
    cv_v_s01.append(r2_score(y_s_all[te_idx], rcv.predict(X_all_v[te_idx])))
cv_v_t01_m = float(np.mean(cv_v_t01))
cv_v_s01_m = float(np.mean(cv_v_s01))
print(f"  Visual-only ({len(visual_only)} features): "
      f"Tech CV={cv_v_t01_m:.3f}, Style CV={cv_v_s01_m:.3f}")

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    "task": "#378 (n=200 + visual-hierarchy features)",
    "feature_version": "features-006",
    "n_total": len(df),
    "n_train": len(train_df),
    "n_val": len(val_df),
    "n_features_full": len(all_feature_cols),
    "n_features_reduced": len(reduced_cols),
    "n_new_visual_features": len(new_features),
    "n_visual_features_after_redundancy": len(visual_only),
    "reduced_features": reduced_cols,
    "new_visual_features": new_features,
    "auto_dropped_visual_features": auto_drops,
    "remaining_high_corr": remaining_hc,
    "alpha_sweep_tech": alpha_sweep_tech,
    "alpha_sweep_style": alpha_sweep_style,
    "technical_model": {
        "ridge_alpha10": {
            "train_r2": float(r2_tech_ridge_train), "val_r2": float(r2_tech_ridge_val),
            "train_mae": float(mae_tech_ridge_train), "val_mae": float(mae_tech_ridge_val),
        },
        "ridge_alpha01": {
            "train_r2": float(r2_tech_ridge01_train), "val_r2": float(r2_tech_ridge01_val),
            "train_mae": float(mae_tech_ridge01_train), "val_mae": float(mae_tech_ridge01_val),
        },
        "gbt": {
            "train_r2": float(r2_tech_gbt_train), "val_r2": float(r2_tech_gbt_val),
            "train_mae": float(mae_tech_gbt_train), "val_mae": float(mae_tech_gbt_val),
        },
        "ridge_top_features": ridge_top_tech[:15],
        "gbt_top_features": gbt_top_tech[:15],
        "top_residuals_val": top_resids_tech,
    },
    "style_model": {
        "ridge_alpha10": {
            "train_r2": float(r2_style_ridge_train), "val_r2": float(r2_style_ridge_val),
            "train_mae": float(mae_style_ridge_train), "val_mae": float(mae_style_ridge_val),
        },
        "ridge_alpha01": {
            "train_r2": float(r2_style_ridge01_train), "val_r2": float(r2_style_ridge01_val),
            "train_mae": float(mae_style_ridge01_train), "val_mae": float(mae_style_ridge01_val),
        },
        "gbt": {
            "train_r2": float(r2_style_gbt_train), "val_r2": float(r2_style_gbt_val),
            "train_mae": float(mae_style_gbt_train), "val_mae": float(mae_style_gbt_val),
        },
        "ridge_top_features": ridge_top_style[:15],
        "gbt_top_features": gbt_top_style[:15],
        "top_residuals_val": top_resids,
    },
    "bootstrap_1000_alpha01": {
        "technical": {
            "median": float(np.median(boot_t01)),
            "ci_low": float(np.percentile(boot_t01, 2.5)),
            "ci_high": float(np.percentile(boot_t01, 97.5)),
        },
        "style": {
            "median": float(np.median(boot_s01)),
            "ci_low": float(np.percentile(boot_s01, 2.5)),
            "ci_high": float(np.percentile(boot_s01, 97.5)),
        },
    },
    "bootstrap_1000_alpha10": {
        "technical": {
            "median": float(np.median(boot_t)),
            "ci_low": float(np.percentile(boot_t, 2.5)),
            "ci_high": float(np.percentile(boot_t, 97.5)),
        },
        "style": {
            "median": float(np.median(boot_s)),
            "ci_low": float(np.percentile(boot_s, 2.5)),
            "ci_high": float(np.percentile(boot_s, 97.5)),
        },
    },
    "cv_5fold_n200": {
        "tech_alpha01":  {"mean": cv_t01_m, "std": cv_t01_s},
        "tech_alpha10":  {"mean": cv_t10_m, "std": cv_t10_s},
        "style_alpha01": {"mean": cv_s01_m, "std": cv_s01_s},
        "style_alpha10": {"mean": cv_s10_m, "std": cv_s10_s},
    },
    "ablation_5fold_alpha01": {
        "structural_only": {
            "n_features": len(struct_only),
            "tech_mean": cv_s_t01_m, "tech_std": cv_s_t01_s,
            "style_mean": cv_s_s01_m, "style_std": cv_s_s01_s,
        },
        "visual_only": {
            "n_features": len(visual_only),
            "tech_mean": cv_v_t01_m,
            "style_mean": cv_v_s01_m,
        },
        "combined": {
            "n_features": len(reduced_cols),
            "tech_mean": cv_t01_m, "tech_std": cv_t01_s,
            "style_mean": cv_s01_m, "style_std": cv_s01_s,
        },
    },
    "bimodal_separation": sep_results,
    "classification_accuracy": {
        "val_only_high_correct": float(hi_corr_v) if hi_corr_v is not None else None,
        "val_only_low_correct": float(lo_corr_v) if lo_corr_v is not None else None,
        "all_high_correct": float(hi_corr_all),
        "all_low_correct": float(lo_corr_all),
    },
}

with open(f"{BASE}/models/results_batch005.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved: results_batch005.json")
print("\nDone.")
