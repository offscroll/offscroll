"""
Re-run go/no-go gate with corrected features (batch-002, 50 spreads).
Ada — Task #330 (re-run of #311 after feature fixes #329)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# --- Load and join data ---
features_path = '/home/modus/repos/ada/offscroll/training/features/features-002.csv'
grades_path = '/home/modus/repos/ada/offscroll/training/grades/batch-002.csv'

feat_df = pd.read_csv(features_path)
grades_df = pd.read_csv(grades_path)

df = feat_df.copy()

# Verify grades match
merged_check = df.merge(grades_df[['spread_id', 'technical', 'style']], on='spread_id')
assert (merged_check['technical_grade'] == merged_check['technical']).all()
assert (merged_check['style_grade'] == merged_check['style']).all()
print(f"Grade consistency check passed. {len(df)} spreads total.")

# --- Verify feature fixes (#329) ---
print("\n" + "="*60)
print("FEATURE FIX VERIFICATION")
print("="*60)

previously_na = ['d2_orphans', 'd2_widows', 'd4_col_balance', 'd6_dead_space']
for col in previously_na:
    na_count = df[col].isna().sum()
    non_zero = (df[col] != 0).sum()
    print(f"  {col}: {na_count} NA, {non_zero}/{len(df)} non-zero, "
          f"range=[{df[col].min():.4f}, {df[col].max():.4f}]")

print(f"\n  d5_fill_fraction: range=[{df['d5_fill_fraction'].min():.4f}, "
      f"{df['d5_fill_fraction'].max():.4f}]")
assert df['d5_fill_fraction'].max() <= 1.0, "fill_fraction still > 1!"
assert df['d5_fill_fraction'].min() >= 0.0, "fill_fraction < 0!"
print("  d5_fill_fraction in [0, 1] — PASS")

# --- Define feature columns ---
meta_cols = ['spread_id', 'split', 'technical_grade', 'style_grade',
             'spread_type', 'page_role', 'left_page', 'right_page']
binary_cols = ['is_solo', 'is_front', 'is_terminal']

numeric_feature_cols = [
    'n_pages_in_spread', 'edition_page_count', 'page_position_frac',
    'edition_item_count', 'edition_word_count_total', 'edition_word_count_mean',
    'edition_brief_frac', 'edition_standard_frac', 'edition_template_entropy',
    'edition_image_count_total', 'edition_source_count', 'edition_section_count',
    'est_item_count', 'est_items_per_page', 'est_word_count',
    'est_words_per_page', 'est_word_count_mean', 'est_brief_count',
    'est_standard_count', 'est_image_count', 'est_source_count',
    'd3_image_fraction', 'd5_fill_fraction', 'd7_template_entropy',
    'd8_word_count_cv', 'anchor_strength',
    'd2_orphans', 'd2_widows', 'd4_col_balance', 'd6_dead_space'
]

all_feature_cols = binary_cols + numeric_feature_cols

# Handle NAs
print("\n--- Missing values ---")
na_counts = df[all_feature_cols].isna().sum()
na_cols = na_counts[na_counts > 0]
print(na_cols if len(na_cols) > 0 else "No missing values in numeric features")
df[all_feature_cols] = df[all_feature_cols].fillna(0)

# --- Correlation analysis and redundancy removal ---
print("\n" + "="*60)
print("REDUNDANCY REMOVAL")
print("="*60)

feat_corr = df[all_feature_cols].corr()
print("\n--- Highly correlated feature pairs (|r| > 0.85) ---")
drop_candidates = set()
for i in range(len(all_feature_cols)):
    for j in range(i+1, len(all_feature_cols)):
        r = feat_corr.iloc[i, j]
        if abs(r) > 0.85:
            print(f"  {all_feature_cols[i]} <-> {all_feature_cols[j]}: r={r:.3f}")

# Remove redundant features based on correlation and domain knowledge
# Rules:
#   - Between perfectly correlated pairs, keep the more interpretable one
#   - is_solo, is_front, is_terminal, n_pages_in_spread are all confounded (solo=1page=front)
#     Keep is_solo (captures the structural distinction)
#   - edition_brief_frac + edition_standard_frac sum to ~1: keep edition_brief_frac
#   - est_word_count ~ est_words_per_page (correlated via n_pages): keep est_words_per_page
#   - est_item_count ~ est_standard_count: keep est_items_per_page
#   - d8_word_count_cv ~ anchor_strength (anchor = 1 + d8 + others): keep anchor_strength
#   - Remove edition-level aggregates that don't vary within-edition
#     (edition_word_count_total, edition_word_count_mean, edition_item_count, etc.)
#     These are per-edition constants — they help predict "which edition" not "which spread"
#   - Keep the diagnostic features (d2-d8) and spread-level estimates

to_drop = [
    'is_front',           # redundant with is_solo
    'is_terminal',        # near-redundant with n_pages (terminal = special page count)
    'n_pages_in_spread',  # captured by is_solo
    'edition_standard_frac',  # = 1 - edition_brief_frac
    'est_word_count',     # ~ est_words_per_page * n_pages
    'est_item_count',     # ~ est_items_per_page * n_pages
    'est_standard_count', # ~ est_item_count (r=0.93)
    'd8_word_count_cv',   # subsumed by anchor_strength
    'est_word_count_mean', # ~ est_words_per_page for 2-page spreads
    # Edition-level constants (same for all spreads in same edition)
    'edition_word_count_total',
    'edition_word_count_mean',
    'edition_item_count',
    'edition_image_count_total',
    'edition_source_count',
    'edition_section_count',
]

reduced_cols = [c for c in all_feature_cols if c not in to_drop]
print(f"\nDropped {len(to_drop)} redundant features. Remaining: {len(reduced_cols)}")
for c in reduced_cols:
    print(f"  {c}")

# --- Split ---
train_df = df[df['split'] == 'train'].copy()
val_df = df[df['split'] == 'val'].copy()
print(f"\nTrain: {len(train_df)}, Validation: {len(val_df)}")

X_train_full = train_df[all_feature_cols].values
X_val_full = val_df[all_feature_cols].values
X_train = train_df[reduced_cols].values
X_val = val_df[reduced_cols].values
y_tech_train = train_df['technical_grade'].values
y_tech_val = val_df['technical_grade'].values
y_style_train = train_df['style_grade'].values
y_style_val = val_df['style_grade'].values

# --- Score distribution ---
print("\n--- Score distributions ---")
print(f"Technical grades: {sorted(df['technical_grade'].unique())}")
print(f"  Distribution: {df['technical_grade'].value_counts().sort_index().to_dict()}")
print(f"Style grades: {sorted(df['style_grade'].unique())}")
print(f"  Distribution: {df['style_grade'].value_counts().sort_index().to_dict()}")

# ============================================================
# 1. TECHNICAL PROFICIENCY MODEL (reduced features)
# ============================================================
print("\n" + "="*60)
print("1. TECHNICAL PROFICIENCY MODEL (reduced features)")
print("="*60)

# Ridge regression
for alpha in [0.1, 1.0, 10.0]:
    ridge_tech = Ridge(alpha=alpha)
    ridge_tech.fit(X_train, y_tech_train)
    r2_t = r2_score(y_tech_train, ridge_tech.predict(X_train))
    r2_v = r2_score(y_tech_val, ridge_tech.predict(X_val))
    print(f"  Ridge alpha={alpha}: Train R²={r2_t:.4f}, Val R²={r2_v:.4f}")

# Use alpha=10 (stronger regularization for small n)
ridge_tech = Ridge(alpha=10.0)
ridge_tech.fit(X_train, y_tech_train)
pred_tech_ridge_train = ridge_tech.predict(X_train)
pred_tech_ridge_val = ridge_tech.predict(X_val)

r2_tech_ridge_train = r2_score(y_tech_train, pred_tech_ridge_train)
r2_tech_ridge_val = r2_score(y_tech_val, pred_tech_ridge_val)
mae_tech_ridge_train = mean_absolute_error(y_tech_train, pred_tech_ridge_train)
mae_tech_ridge_val = mean_absolute_error(y_tech_val, pred_tech_ridge_val)

print(f"\nRidge (alpha=10):")
print(f"  Train — R²: {r2_tech_ridge_train:.4f}, MAE: {mae_tech_ridge_train:.4f}")
print(f"  Val   — R²: {r2_tech_ridge_val:.4f}, MAE: {mae_tech_ridge_val:.4f}")

# Feature importances (standardized coefficients)
feat_std = np.std(X_train, axis=0)
feat_std[feat_std == 0] = 1
ridge_importance_tech = np.abs(ridge_tech.coef_ * feat_std)
ridge_imp_idx = np.argsort(ridge_importance_tech)[::-1]

print(f"\n  Top features (Ridge, standardized |coef|):")
for i in ridge_imp_idx[:len(reduced_cols)]:
    print(f"    {reduced_cols[i]:30s} {ridge_importance_tech[i]:.4f} (coef={ridge_tech.coef_[i]:+.4f})")

# GBT with reduced depth/estimators for small n
gbt_tech = GradientBoostingRegressor(
    n_estimators=50, max_depth=2, learning_rate=0.05,
    min_samples_leaf=5, random_state=42, subsample=0.8
)
gbt_tech.fit(X_train, y_tech_train)
pred_tech_gbt_train = gbt_tech.predict(X_train)
pred_tech_gbt_val = gbt_tech.predict(X_val)

r2_tech_gbt_train = r2_score(y_tech_train, pred_tech_gbt_train)
r2_tech_gbt_val = r2_score(y_tech_val, pred_tech_gbt_val)
mae_tech_gbt_train = mean_absolute_error(y_tech_train, pred_tech_gbt_train)
mae_tech_gbt_val = mean_absolute_error(y_tech_val, pred_tech_gbt_val)

print(f"\nGradient Boosted Trees (n=50, depth=2, lr=0.05):")
print(f"  Train — R²: {r2_tech_gbt_train:.4f}, MAE: {mae_tech_gbt_train:.4f}")
print(f"  Val   — R²: {r2_tech_gbt_val:.4f}, MAE: {mae_tech_gbt_val:.4f}")

print(f"\n  Feature importances (GBT):")
gbt_imp_idx = np.argsort(gbt_tech.feature_importances_)[::-1]
for i in gbt_imp_idx[:len(reduced_cols)]:
    if gbt_tech.feature_importances_[i] > 0.001:
        print(f"    {reduced_cols[i]:30s} {gbt_tech.feature_importances_[i]:.4f}")

# ============================================================
# 2. STYLE MODEL (reduced features)
# ============================================================
print("\n" + "="*60)
print("2. STYLE MODEL (reduced features)")
print("="*60)

for alpha in [0.1, 1.0, 10.0]:
    ridge_s = Ridge(alpha=alpha)
    ridge_s.fit(X_train, y_style_train)
    r2_t = r2_score(y_style_train, ridge_s.predict(X_train))
    r2_v = r2_score(y_style_val, ridge_s.predict(X_val))
    print(f"  Ridge alpha={alpha}: Train R²={r2_t:.4f}, Val R²={r2_v:.4f}")

ridge_style = Ridge(alpha=10.0)
ridge_style.fit(X_train, y_style_train)
pred_style_ridge_train = ridge_style.predict(X_train)
pred_style_ridge_val = ridge_style.predict(X_val)

r2_style_ridge_train = r2_score(y_style_train, pred_style_ridge_train)
r2_style_ridge_val = r2_score(y_style_val, pred_style_ridge_val)
mae_style_ridge_train = mean_absolute_error(y_style_train, pred_style_ridge_train)
mae_style_ridge_val = mean_absolute_error(y_style_val, pred_style_ridge_val)

print(f"\nRidge (alpha=10):")
print(f"  Train — R²: {r2_style_ridge_train:.4f}, MAE: {mae_style_ridge_train:.4f}")
print(f"  Val   — R²: {r2_style_ridge_val:.4f}, MAE: {mae_style_ridge_val:.4f}")

ridge_importance_style = np.abs(ridge_style.coef_ * feat_std)
ridge_imp_idx_style = np.argsort(ridge_importance_style)[::-1]
print(f"\n  Top features (Ridge, standardized |coef|):")
for i in ridge_imp_idx_style[:len(reduced_cols)]:
    print(f"    {reduced_cols[i]:30s} {ridge_importance_style[i]:.4f} (coef={ridge_style.coef_[i]:+.4f})")

gbt_style = GradientBoostingRegressor(
    n_estimators=50, max_depth=2, learning_rate=0.05,
    min_samples_leaf=5, random_state=42, subsample=0.8
)
gbt_style.fit(X_train, y_style_train)
pred_style_gbt_train = gbt_style.predict(X_train)
pred_style_gbt_val = gbt_style.predict(X_val)

r2_style_gbt_train = r2_score(y_style_train, pred_style_gbt_train)
r2_style_gbt_val = r2_score(y_style_val, pred_style_gbt_val)
mae_style_gbt_train = mean_absolute_error(y_style_train, pred_style_gbt_train)
mae_style_gbt_val = mean_absolute_error(y_style_val, pred_style_gbt_val)

print(f"\nGradient Boosted Trees:")
print(f"  Train — R²: {r2_style_gbt_train:.4f}, MAE: {mae_style_gbt_train:.4f}")
print(f"  Val   — R²: {r2_style_gbt_val:.4f}, MAE: {mae_style_gbt_val:.4f}")

print(f"\n  Feature importances (GBT):")
gbt_imp_idx_style = np.argsort(gbt_style.feature_importances_)[::-1]
for i in gbt_imp_idx_style[:len(reduced_cols)]:
    if gbt_style.feature_importances_[i] > 0.001:
        print(f"    {reduced_cols[i]:30s} {gbt_style.feature_importances_[i]:.4f}")

# Residuals
style_resid_val = np.abs(y_style_val - pred_style_gbt_val)
resid_order = np.argsort(style_resid_val)[::-1]
print("\n  Top highest-residual spreads (style, GBT, validation):")
for idx in resid_order[:min(10, len(resid_order))]:
    sid = val_df.iloc[idx]['spread_id']
    actual = y_style_val[idx]
    predicted = pred_style_gbt_val[idx]
    print(f"    {sid}: actual={actual}, predicted={predicted:.2f}, residual={style_resid_val[idx]:.2f}")

# ============================================================
# 3. DIAGNOSTIC ANALYSIS
# ============================================================
print("\n" + "="*60)
print("3. DIAGNOSTIC ANALYSIS")
print("="*60)

# --- Predicted vs Actual scatter plots ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.scatter(y_tech_train, pred_tech_ridge_train, alpha=0.6, label='Train')
ax.scatter(y_tech_val, pred_tech_ridge_val, alpha=0.8, marker='x', s=80, label='Val')
ax.plot([1, 7], [1, 7], 'k--', alpha=0.3)
ax.set_xlabel('Actual Technical Grade')
ax.set_ylabel('Predicted')
ax.set_title(f'Technical — Ridge (Val R²={r2_tech_ridge_val:.3f})')
ax.legend()

ax = axes[0, 1]
ax.scatter(y_tech_train, pred_tech_gbt_train, alpha=0.6, label='Train')
ax.scatter(y_tech_val, pred_tech_gbt_val, alpha=0.8, marker='x', s=80, label='Val')
ax.plot([1, 7], [1, 7], 'k--', alpha=0.3)
ax.set_xlabel('Actual Technical Grade')
ax.set_ylabel('Predicted')
ax.set_title(f'Technical — GBT (Val R²={r2_tech_gbt_val:.3f})')
ax.legend()

ax = axes[1, 0]
ax.scatter(y_style_train, pred_style_ridge_train, alpha=0.6, label='Train')
ax.scatter(y_style_val, pred_style_ridge_val, alpha=0.8, marker='x', s=80, label='Val')
ax.plot([1, 7], [1, 7], 'k--', alpha=0.3)
ax.set_xlabel('Actual Style Grade')
ax.set_ylabel('Predicted')
ax.set_title(f'Style — Ridge (Val R²={r2_style_ridge_val:.3f})')
ax.legend()

ax = axes[1, 1]
ax.scatter(y_style_train, pred_style_gbt_train, alpha=0.6, label='Train')
ax.scatter(y_style_val, pred_style_gbt_val, alpha=0.8, marker='x', s=80, label='Val')
ax.plot([1, 7], [1, 7], 'k--', alpha=0.3)
ax.set_xlabel('Actual Style Grade')
ax.set_ylabel('Predicted')
ax.set_title(f'Style — GBT (Val R²={r2_style_gbt_val:.3f})')
ax.legend()

plt.tight_layout()
plt.savefig('/home/modus/repos/ada/offscroll/training/models/pred_vs_actual_002.png', dpi=150)
plt.close()
print("Saved: pred_vs_actual_002.png")

# --- Feature correlation matrix (reduced) ---
feat_corr_reduced = pd.DataFrame(X_train, columns=reduced_cols).corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(feat_corr_reduced.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(reduced_cols)))
ax.set_yticks(range(len(reduced_cols)))
ax.set_xticklabels(reduced_cols, rotation=90, fontsize=8)
ax.set_yticklabels(reduced_cols, fontsize=8)
plt.colorbar(im)
plt.title('Feature Correlation Matrix (reduced, batch-002 v2)')
plt.tight_layout()
plt.savefig('/home/modus/repos/ada/offscroll/training/models/feature_correlation_002.png', dpi=150)
plt.close()
print("Saved: feature_correlation_002.png")

# Remaining high correlations
print("\n--- Remaining high correlations (|r| > 0.85) ---")
high_corr_count = 0
for i in range(len(reduced_cols)):
    for j in range(i+1, len(reduced_cols)):
        r = feat_corr_reduced.iloc[i, j]
        if abs(r) > 0.85:
            print(f"  {reduced_cols[i]} <-> {reduced_cols[j]}: r={r:.3f}")
            high_corr_count += 1
if high_corr_count == 0:
    print("  None — multicollinearity resolved")

# --- Bimodal separation check ---
print("\n--- Bimodal separation check ---")
cluster_high = df[df['technical_grade'] >= 5]
cluster_low = df[df['technical_grade'] <= 3]
print(f"High cluster (T>=5): {len(cluster_high)} spreads")
print(f"Low cluster (T<=3): {len(cluster_low)} spreads")
print(f"Middle (T=4): {len(df[df['technical_grade'] == 4])} spreads")

sep_features = ['d5_fill_fraction', 'd6_dead_space', 'd4_col_balance',
                'd2_orphans', 'd2_widows',
                'est_items_per_page', 'est_words_per_page',
                'anchor_strength', 'is_solo']
print(f"\n  Feature means by cluster:")
print(f"  {'Feature':30s} {'High(T>=5)':>10s} {'Low(T<=3)':>10s} {'Diff':>10s} {'t-stat':>8s}")

from scipy import stats
for f in sep_features:
    if f in df.columns:
        h_vals = cluster_high[f].values
        l_vals = cluster_low[f].values
        h = h_vals.mean()
        l = l_vals.mean()
        # Welch's t-test
        if np.std(h_vals) > 0 or np.std(l_vals) > 0:
            t_stat, p_val = stats.ttest_ind(h_vals, l_vals, equal_var=False)
            sig = '*' if p_val < 0.05 else ''
            print(f"  {f:30s} {h:10.3f} {l:10.3f} {h-l:10.3f} {t_stat:7.2f}{sig}")
        else:
            print(f"  {f:30s} {h:10.3f} {l:10.3f} {h-l:10.3f}     n/a")

# Classification accuracy on validation only
pred_tech_gbt_val_arr = pred_tech_gbt_val
mask_high_val = y_tech_val >= 5
mask_low_val = y_tech_val <= 3
if mask_high_val.sum() > 0:
    high_correct = (pred_tech_gbt_val_arr[mask_high_val] > 4).mean()
    print(f"\n  GBT val-only: {high_correct*100:.1f}% of T>=5 spreads predicted > 4.0 ({mask_high_val.sum()} spreads)")
if mask_low_val.sum() > 0:
    low_correct = (pred_tech_gbt_val_arr[mask_low_val] < 4).mean()
    print(f"  GBT val-only: {low_correct*100:.1f}% of T<=3 spreads predicted < 4.0 ({mask_low_val.sum()} spreads)")

# Also on full data for reference
all_pred_tech = np.concatenate([pred_tech_gbt_train, pred_tech_gbt_val])
all_actual_tech = np.concatenate([y_tech_train, y_tech_val])
mask_high_all = all_actual_tech >= 5
mask_low_all = all_actual_tech <= 3
if mask_high_all.sum() > 0:
    high_correct_all = (all_pred_tech[mask_high_all] > 4).mean()
    print(f"  GBT all data: {high_correct_all*100:.1f}% of T>=5 predicted > 4.0")
if mask_low_all.sum() > 0:
    low_correct_all = (all_pred_tech[mask_low_all] < 4).mean()
    print(f"  GBT all data: {low_correct_all*100:.1f}% of T<=3 predicted < 4.0")

# --- Bootstrap confidence intervals ---
print("\n--- Bootstrap confidence intervals (Ridge alpha=10, 1000 iterations) ---")
np.random.seed(42)
n_boot = 1000

X_all = df[reduced_cols].values
y_tech_all = df['technical_grade'].values
y_style_all = df['style_grade'].values

boot_r2_tech = []
boot_r2_style = []
for _ in range(n_boot):
    idx = np.random.choice(len(X_all), size=len(X_all), replace=True)
    oob = np.array([i for i in range(len(X_all)) if i not in idx])
    if len(oob) < 5:
        continue
    ridge_b = Ridge(alpha=10.0)
    ridge_b.fit(X_all[idx], y_tech_all[idx])
    pred_b = ridge_b.predict(X_all[oob])
    var_actual = np.var(y_tech_all[oob])
    if var_actual > 0:
        boot_r2_tech.append(r2_score(y_tech_all[oob], pred_b))

    ridge_bs = Ridge(alpha=10.0)
    ridge_bs.fit(X_all[idx], y_style_all[idx])
    pred_bs = ridge_bs.predict(X_all[oob])
    var_actual_s = np.var(y_style_all[oob])
    if var_actual_s > 0:
        boot_r2_style.append(r2_score(y_style_all[oob], pred_bs))

boot_r2_tech = np.array(boot_r2_tech)
boot_r2_style = np.array(boot_r2_style)

print(f"  Technical R² (OOB): median={np.median(boot_r2_tech):.3f}, "
      f"95% CI=[{np.percentile(boot_r2_tech, 2.5):.3f}, {np.percentile(boot_r2_tech, 97.5):.3f}]")
print(f"  Style R² (OOB): median={np.median(boot_r2_style):.3f}, "
      f"95% CI=[{np.percentile(boot_r2_style, 2.5):.3f}, {np.percentile(boot_r2_style, 97.5):.3f}]")

# --- Learning curve ---
print("\n--- Learning curve (Ridge alpha=10, reduced features) ---")
fractions = [0.3, 0.5, 0.7, 0.85, 1.0]
lc_results = []
for frac in fractions:
    n = int(len(X_train) * frac)
    if n < 5:
        continue
    scores = []
    for _ in range(50):
        idx = np.random.choice(len(X_train), size=n, replace=False)
        ridge_lc = Ridge(alpha=10.0)
        ridge_lc.fit(X_train[idx], y_tech_train[idx])
        pred_lc = ridge_lc.predict(X_val)
        scores.append(r2_score(y_tech_val, pred_lc))
    lc_results.append((n, np.mean(scores), np.std(scores)))
    print(f"  n={n:3d}: Val R² = {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    'task': '#330 (re-run of #311)',
    'feature_version': 'features-002 corrected (#329)',
    'n_features_full': len(all_feature_cols),
    'n_features_reduced': len(reduced_cols),
    'reduced_features': reduced_cols,
    'technical_model': {
        'ridge': {'train_r2': r2_tech_ridge_train, 'val_r2': r2_tech_ridge_val,
                  'train_mae': mae_tech_ridge_train, 'val_mae': mae_tech_ridge_val,
                  'alpha': 10.0},
        'gbt': {'train_r2': r2_tech_gbt_train, 'val_r2': r2_tech_gbt_val,
                'train_mae': mae_tech_gbt_train, 'val_mae': mae_tech_gbt_val}
    },
    'style_model': {
        'ridge': {'train_r2': r2_style_ridge_train, 'val_r2': r2_style_ridge_val,
                  'train_mae': mae_style_ridge_train, 'val_mae': mae_style_ridge_val,
                  'alpha': 10.0},
        'gbt': {'train_r2': r2_style_gbt_train, 'val_r2': r2_style_gbt_val,
                'train_mae': mae_style_gbt_train, 'val_mae': mae_style_gbt_val}
    },
    'bootstrap': {
        'technical_r2_median': float(np.median(boot_r2_tech)),
        'technical_r2_ci_low': float(np.percentile(boot_r2_tech, 2.5)),
        'technical_r2_ci_high': float(np.percentile(boot_r2_tech, 97.5)),
        'style_r2_median': float(np.median(boot_r2_style)),
        'style_r2_ci_low': float(np.percentile(boot_r2_style, 2.5)),
        'style_r2_ci_high': float(np.percentile(boot_r2_style, 97.5)),
    },
    'n_train': len(train_df),
    'n_val': len(val_df),
}

with open('/home/modus/repos/ada/offscroll/training/models/results_batch002.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved: results_batch002.json")
print("\nDone.")
