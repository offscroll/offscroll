"""
Re-run go/no-go gate at n=150 checkpoint (features-003.csv).
Ada — Task #342 (re-run of #330 with 3x data).
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

BASE = '/home/modus/offscroll/training'
features_path = f'{BASE}/features/features-003.csv'

df = pd.read_csv(features_path)
print(f"Loaded {len(df)} spreads from features-003.csv")

# --- Verify feature health ---
print("\n" + "="*60)
print("FEATURE HEALTH CHECK")
print("="*60)

previously_na = ['d2_orphans', 'd2_widows', 'd4_col_balance', 'd6_dead_space']
for col in previously_na:
    na_count = df[col].isna().sum()
    non_zero = (df[col] != 0).sum()
    print(f"  {col}: {na_count} NA, {non_zero}/{len(df)} non-zero, "
          f"range=[{df[col].min():.4f}, {df[col].max():.4f}]")

print(f"\n  d5_fill_fraction: range=[{df['d5_fill_fraction'].min():.4f}, "
      f"{df['d5_fill_fraction'].max():.4f}]")
assert df['d5_fill_fraction'].max() <= 1.0
assert df['d5_fill_fraction'].min() >= 0.0
print("  d5_fill_fraction in [0, 1] — PASS")

# --- Define feature columns ---
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

# Handle NAs (defensive)
df[all_feature_cols] = df[all_feature_cols].fillna(0)

# --- Redundancy removal (same rules as #330) ---
print("\n" + "="*60)
print("REDUNDANCY REMOVAL (same rules as #330)")
print("="*60)

feat_corr = df[all_feature_cols].corr()
print("\n--- Highly correlated feature pairs (|r| > 0.85) ---")
for i in range(len(all_feature_cols)):
    for j in range(i+1, len(all_feature_cols)):
        r = feat_corr.iloc[i, j]
        if abs(r) > 0.85:
            print(f"  {all_feature_cols[i]} <-> {all_feature_cols[j]}: r={r:.3f}")

to_drop = [
    'is_front', 'is_terminal', 'n_pages_in_spread',
    'edition_standard_frac',
    'est_word_count', 'est_item_count', 'est_standard_count',
    'd8_word_count_cv', 'est_word_count_mean',
    'edition_word_count_total', 'edition_word_count_mean',
    'edition_item_count', 'edition_image_count_total',
    'edition_source_count', 'edition_section_count',
]
reduced_cols = [c for c in all_feature_cols if c not in to_drop]
print(f"\nDropped {len(to_drop)} redundant features. Remaining: {len(reduced_cols)}")

# --- Split ---
train_df = df[df['split'] == 'train'].copy()
val_df = df[df['split'] == 'val'].copy()
print(f"\nTrain: {len(train_df)}, Validation: {len(val_df)}")

X_train = train_df[reduced_cols].values
X_val = val_df[reduced_cols].values
y_tech_train = train_df['technical_grade'].values
y_tech_val = val_df['technical_grade'].values
y_style_train = train_df['style_grade'].values
y_style_val = val_df['style_grade'].values

# --- Score distributions ---
print("\n--- Score distributions ---")
print(f"Technical: {df['technical_grade'].value_counts().sort_index().to_dict()}")
print(f"Style:     {df['style_grade'].value_counts().sort_index().to_dict()}")

# ============================================================
# 1. TECHNICAL MODEL
# ============================================================
print("\n" + "="*60)
print("1. TECHNICAL PROFICIENCY MODEL")
print("="*60)

for alpha in [0.1, 1.0, 10.0]:
    r = Ridge(alpha=alpha).fit(X_train, y_tech_train)
    print(f"  Ridge alpha={alpha}: Train R²={r2_score(y_tech_train, r.predict(X_train)):.4f}, "
          f"Val R²={r2_score(y_tech_val, r.predict(X_val)):.4f}")

ridge_tech = Ridge(alpha=10.0)
ridge_tech.fit(X_train, y_tech_train)
pred_tt = ridge_tech.predict(X_train)
pred_tv = ridge_tech.predict(X_val)
r2_tech_ridge_train = r2_score(y_tech_train, pred_tt)
r2_tech_ridge_val = r2_score(y_tech_val, pred_tv)
mae_tech_ridge_train = mean_absolute_error(y_tech_train, pred_tt)
mae_tech_ridge_val = mean_absolute_error(y_tech_val, pred_tv)
print(f"\nRidge (alpha=10): Train R²={r2_tech_ridge_train:.4f} MAE={mae_tech_ridge_train:.4f} | "
      f"Val R²={r2_tech_ridge_val:.4f} MAE={mae_tech_ridge_val:.4f}")

feat_std = np.std(X_train, axis=0); feat_std[feat_std == 0] = 1
ridge_imp_tech = np.abs(ridge_tech.coef_ * feat_std)
ridge_imp_idx = np.argsort(ridge_imp_tech)[::-1]
print("\n  Top features (Ridge, standardized |coef|):")
ridge_top_tech = []
for i in ridge_imp_idx:
    print(f"    {reduced_cols[i]:30s} {ridge_imp_tech[i]:.4f} (coef={ridge_tech.coef_[i]:+.4f})")
    ridge_top_tech.append({'feature': reduced_cols[i], 'importance': float(ridge_imp_tech[i]),
                           'coef': float(ridge_tech.coef_[i])})

gbt_tech = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05,
                                     min_samples_leaf=5, random_state=42, subsample=0.8)
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
        gbt_top_tech.append({'feature': reduced_cols[i], 'importance': float(gbt_tech.feature_importances_[i])})

# ============================================================
# 2. STYLE MODEL
# ============================================================
print("\n" + "="*60)
print("2. STYLE MODEL")
print("="*60)

for alpha in [0.1, 1.0, 10.0]:
    r = Ridge(alpha=alpha).fit(X_train, y_style_train)
    print(f"  Ridge alpha={alpha}: Train R²={r2_score(y_style_train, r.predict(X_train)):.4f}, "
          f"Val R²={r2_score(y_style_val, r.predict(X_val)):.4f}")

ridge_style = Ridge(alpha=10.0)
ridge_style.fit(X_train, y_style_train)
pred_st = ridge_style.predict(X_train); pred_sv = ridge_style.predict(X_val)
r2_style_ridge_train = r2_score(y_style_train, pred_st)
r2_style_ridge_val = r2_score(y_style_val, pred_sv)
mae_style_ridge_train = mean_absolute_error(y_style_train, pred_st)
mae_style_ridge_val = mean_absolute_error(y_style_val, pred_sv)
print(f"\nRidge (alpha=10): Train R²={r2_style_ridge_train:.4f} MAE={mae_style_ridge_train:.4f} | "
      f"Val R²={r2_style_ridge_val:.4f} MAE={mae_style_ridge_val:.4f}")

ridge_imp_style = np.abs(ridge_style.coef_ * feat_std)
ridge_imp_idx_s = np.argsort(ridge_imp_style)[::-1]
print("\n  Top features (Ridge, standardized |coef|):")
ridge_top_style = []
for i in ridge_imp_idx_s:
    print(f"    {reduced_cols[i]:30s} {ridge_imp_style[i]:.4f} (coef={ridge_style.coef_[i]:+.4f})")
    ridge_top_style.append({'feature': reduced_cols[i], 'importance': float(ridge_imp_style[i]),
                            'coef': float(ridge_style.coef_[i])})

gbt_style = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05,
                                      min_samples_leaf=5, random_state=42, subsample=0.8)
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
        gbt_top_style.append({'feature': reduced_cols[i], 'importance': float(gbt_style.feature_importances_[i])})

# Highest residuals
style_resid = np.abs(y_style_val - pred_gsv)
resid_order = np.argsort(style_resid)[::-1]
print("\n  Top residual spreads (style, GBT, val):")
top_resids = []
for idx in resid_order[:10]:
    sid = val_df.iloc[idx]['spread_id']
    a = y_style_val[idx]; p = pred_gsv[idx]; r = style_resid[idx]
    print(f"    {sid}: actual={a}, predicted={p:.2f}, residual={r:.2f}")
    top_resids.append({'spread_id': sid, 'actual': float(a), 'predicted': float(p), 'residual': float(r)})

# ============================================================
# 3. DIAGNOSTICS — plots, bimodal, bootstrap, learning curve
# ============================================================
print("\n" + "="*60)
print("3. DIAGNOSTICS")
print("="*60)

# Pred vs actual scatter (4 panels)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax = axes[0, 0]
ax.scatter(y_tech_train, pred_tt, alpha=0.5, label='Train')
ax.scatter(y_tech_val, pred_tv, alpha=0.8, marker='x', s=80, label='Val')
ax.plot([1, 7], [1, 7], 'k--', alpha=0.3)
ax.set_xlabel('Actual Technical'); ax.set_ylabel('Predicted')
ax.set_title(f'Technical — Ridge (Val R²={r2_tech_ridge_val:.3f})'); ax.legend()

ax = axes[0, 1]
ax.scatter(y_tech_train, pred_gtt, alpha=0.5, label='Train')
ax.scatter(y_tech_val, pred_gtv, alpha=0.8, marker='x', s=80, label='Val')
ax.plot([1, 7], [1, 7], 'k--', alpha=0.3)
ax.set_xlabel('Actual Technical'); ax.set_ylabel('Predicted')
ax.set_title(f'Technical — GBT (Val R²={r2_tech_gbt_val:.3f})'); ax.legend()

ax = axes[1, 0]
ax.scatter(y_style_train, pred_st, alpha=0.5, label='Train')
ax.scatter(y_style_val, pred_sv, alpha=0.8, marker='x', s=80, label='Val')
ax.plot([1, 7], [1, 7], 'k--', alpha=0.3)
ax.set_xlabel('Actual Style'); ax.set_ylabel('Predicted')
ax.set_title(f'Style — Ridge (Val R²={r2_style_ridge_val:.3f})'); ax.legend()

ax = axes[1, 1]
ax.scatter(y_style_train, pred_gst, alpha=0.5, label='Train')
ax.scatter(y_style_val, pred_gsv, alpha=0.8, marker='x', s=80, label='Val')
ax.plot([1, 7], [1, 7], 'k--', alpha=0.3)
ax.set_xlabel('Actual Style'); ax.set_ylabel('Predicted')
ax.set_title(f'Style — GBT (Val R²={r2_style_gbt_val:.3f})'); ax.legend()

plt.tight_layout()
plt.savefig(f'{BASE}/models/pred_vs_actual_003.png', dpi=150)
plt.close()
print("Saved: pred_vs_actual_003.png")

# Correlation matrix
feat_corr_red = pd.DataFrame(X_train, columns=reduced_cols).corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(feat_corr_red.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(reduced_cols))); ax.set_yticks(range(len(reduced_cols)))
ax.set_xticklabels(reduced_cols, rotation=90, fontsize=8)
ax.set_yticklabels(reduced_cols, fontsize=8)
plt.colorbar(im); plt.title('Feature Correlation (reduced, batch-003)')
plt.tight_layout()
plt.savefig(f'{BASE}/models/feature_correlation_003.png', dpi=150)
plt.close()
print("Saved: feature_correlation_003.png")

# Remaining high correlations
print("\n--- Remaining high correlations (|r| > 0.85) ---")
remaining_hc = []
for i in range(len(reduced_cols)):
    for j in range(i+1, len(reduced_cols)):
        r = feat_corr_red.iloc[i, j]
        if abs(r) > 0.85:
            print(f"  {reduced_cols[i]} <-> {reduced_cols[j]}: r={r:.3f}")
            remaining_hc.append({'a': reduced_cols[i], 'b': reduced_cols[j], 'r': float(r)})
if not remaining_hc:
    print("  None — multicollinearity resolved")

# Bimodal separation
print("\n--- Bimodal separation (T>=5 vs T<=3) ---")
high = df[df['technical_grade'] >= 5]
low = df[df['technical_grade'] <= 3]
print(f"High: {len(high)}, Low: {len(low)}, Middle (T=4): {(df['technical_grade']==4).sum()}")

from scipy import stats
sep_features = ['d5_fill_fraction', 'd6_dead_space', 'd4_col_balance',
                'd2_orphans', 'd2_widows',
                'est_items_per_page', 'est_words_per_page',
                'anchor_strength', 'is_solo', 'est_brief_count',
                'page_position_frac']
print(f"  {'Feature':30s} {'High':>10s} {'Low':>10s} {'Diff':>10s} {'t-stat':>8s} {'p':>10s}")
sep_results = []
for f in sep_features:
    if f in df.columns:
        h = high[f].values; l = low[f].values
        if np.std(h) > 0 or np.std(l) > 0:
            t, p = stats.ttest_ind(h, l, equal_var=False)
            sig = '*' if p < 0.05 else ''
            print(f"  {f:30s} {h.mean():10.3f} {l.mean():10.3f} {h.mean()-l.mean():10.3f} {t:7.2f}{sig} {p:10.4g}")
            sep_results.append({'feature': f, 'high_mean': float(h.mean()), 'low_mean': float(l.mean()),
                                'diff': float(h.mean()-l.mean()), 't': float(t), 'p': float(p)})

# Classification accuracy
mask_h_v = y_tech_val >= 5; mask_l_v = y_tech_val <= 3
hi_corr_v = (pred_gtv[mask_h_v] > 4).mean() if mask_h_v.sum() else None
lo_corr_v = (pred_gtv[mask_l_v] < 4).mean() if mask_l_v.sum() else None
print(f"\n  GBT val-only: T>=5 predicted >4: {hi_corr_v*100 if hi_corr_v is not None else 'n/a':.1f}% ({mask_h_v.sum()})")
print(f"  GBT val-only: T<=3 predicted <4: {lo_corr_v*100 if lo_corr_v is not None else 'n/a':.1f}% ({mask_l_v.sum()})")

all_pred = np.concatenate([pred_gtt, pred_gtv])
all_actual = np.concatenate([y_tech_train, y_tech_val])
mh = all_actual >= 5; ml = all_actual <= 3
hi_corr_all = (all_pred[mh] > 4).mean()
lo_corr_all = (all_pred[ml] < 4).mean()
print(f"  GBT all data: T>=5 predicted >4: {hi_corr_all*100:.1f}%")
print(f"  GBT all data: T<=3 predicted <4: {lo_corr_all*100:.1f}%")

# Bootstrap CIs
print("\n--- Bootstrap CIs (Ridge alpha=10, 1000 iters) ---")
np.random.seed(42)
X_all = df[reduced_cols].values
y_t_all = df['technical_grade'].values
y_s_all = df['style_grade'].values

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
print(f"  Tech: median={np.median(boot_t):.3f}, 95% CI=[{np.percentile(boot_t,2.5):.3f}, {np.percentile(boot_t,97.5):.3f}]")
print(f"  Style: median={np.median(boot_s):.3f}, 95% CI=[{np.percentile(boot_s,2.5):.3f}, {np.percentile(boot_s,97.5):.3f}]")

# Learning curve — fixed val=30, vary n_train ∈ {40,60,80,100,120}
# Plus full-data 5-fold CV point at n=150 (each fold trains on 120, tests on 30)
print("\n--- Learning curve (Ridge alpha=10) ---")
np.random.seed(42)
n_targets = [40, 60, 80, 100, 120]
lc_results = []
for n in n_targets:
    if n > len(X_train):
        continue
    scores = []
    for _ in range(50):
        idx = np.random.choice(len(X_train), size=n, replace=False)
        rl = Ridge(alpha=10).fit(X_train[idx], y_tech_train[idx])
        scores.append(r2_score(y_tech_val, rl.predict(X_val)))
    m = float(np.mean(scores)); s = float(np.std(scores))
    print(f"  n_train={n:3d}: Val R² = {m:.3f} +/- {s:.3f}")
    lc_results.append({'n_train': n, 'val_r2_mean': m, 'val_r2_std': s})

# n_total=150 via 5-fold CV (train≈120, test≈30 per fold) for an apples-to-apples plateau check
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for tr_idx, te_idx in kf.split(X_all):
    rcv = Ridge(alpha=10).fit(X_all[tr_idx], y_t_all[tr_idx])
    cv_scores.append(r2_score(y_t_all[te_idx], rcv.predict(X_all[te_idx])))
cv_m = float(np.mean(cv_scores)); cv_s = float(np.std(cv_scores))
print(f"  n_total=150 (5-fold CV, ~120 train): Val R² = {cv_m:.3f} +/- {cv_s:.3f}")
lc_results.append({'n_train': 120, 'val_r2_mean': cv_m, 'val_r2_std': cv_s, 'method': '5-fold CV on full 150'})

# Style learning curve too
print("\n--- Learning curve — STYLE (Ridge alpha=10) ---")
np.random.seed(42)
lc_results_style = []
for n in n_targets:
    if n > len(X_train):
        continue
    scores = []
    for _ in range(50):
        idx = np.random.choice(len(X_train), size=n, replace=False)
        rl = Ridge(alpha=10).fit(X_train[idx], y_style_train[idx])
        scores.append(r2_score(y_style_val, rl.predict(X_val)))
    m = float(np.mean(scores)); s = float(np.std(scores))
    print(f"  n_train={n:3d}: Val R² = {m:.3f} +/- {s:.3f}")
    lc_results_style.append({'n_train': n, 'val_r2_mean': m, 'val_r2_std': s})

# Plot learning curve
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ns = [r['n_train'] for r in lc_results if 'method' not in r]
ms = [r['val_r2_mean'] for r in lc_results if 'method' not in r]
ss = [r['val_r2_std'] for r in lc_results if 'method' not in r]
ax.errorbar(ns, ms, yerr=ss, marker='o', label='Technical (n_val=30 fixed)', linewidth=2)
ax.scatter([120], [cv_m], marker='*', s=200, color='red', label=f'Tech 5-fold CV @ n=150 ({cv_m:.3f})', zorder=5)

ns_s = [r['n_train'] for r in lc_results_style]
ms_s = [r['val_r2_mean'] for r in lc_results_style]
ss_s = [r['val_r2_std'] for r in lc_results_style]
ax.errorbar(ns_s, ms_s, yerr=ss_s, marker='s', label='Style (n_val=30 fixed)', linewidth=2)

# Reference: #330 (n=50) endpoint at n_train=40
ax.scatter([40], [0.32], marker='D', s=120, color='black',
           label='#330 endpoint (n_train=40, R²=0.32)', zorder=5)

ax.axhline(0.5, color='green', linestyle=':', alpha=0.5, label='GO threshold (R²=0.5)')
ax.axhline(0.3, color='orange', linestyle=':', alpha=0.5, label='Continue threshold (R²=0.3)')
ax.set_xlabel('Training samples (n_train)')
ax.set_ylabel('Validation R²')
ax.set_title('Learning Curve — n=150 checkpoint (#342)')
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{BASE}/models/learning_curve_003.png', dpi=150)
plt.close()
print("Saved: learning_curve_003.png")

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    'task': '#342 (n=150 checkpoint, follow-up of #330)',
    'feature_version': 'features-003',
    'n_total': len(df),
    'n_train': len(train_df),
    'n_val': len(val_df),
    'n_features_full': len(all_feature_cols),
    'n_features_reduced': len(reduced_cols),
    'reduced_features': reduced_cols,
    'remaining_high_corr': remaining_hc,
    'technical_model': {
        'ridge': {'train_r2': float(r2_tech_ridge_train), 'val_r2': float(r2_tech_ridge_val),
                  'train_mae': float(mae_tech_ridge_train), 'val_mae': float(mae_tech_ridge_val),
                  'alpha': 10.0},
        'gbt': {'train_r2': float(r2_tech_gbt_train), 'val_r2': float(r2_tech_gbt_val),
                'train_mae': float(mae_tech_gbt_train), 'val_mae': float(mae_tech_gbt_val)},
        'ridge_top_features': ridge_top_tech[:10],
        'gbt_top_features': gbt_top_tech[:10],
    },
    'style_model': {
        'ridge': {'train_r2': float(r2_style_ridge_train), 'val_r2': float(r2_style_ridge_val),
                  'train_mae': float(mae_style_ridge_train), 'val_mae': float(mae_style_ridge_val),
                  'alpha': 10.0},
        'gbt': {'train_r2': float(r2_style_gbt_train), 'val_r2': float(r2_style_gbt_val),
                'train_mae': float(mae_style_gbt_train), 'val_mae': float(mae_style_gbt_val)},
        'ridge_top_features': ridge_top_style[:10],
        'gbt_top_features': gbt_top_style[:10],
        'top_residuals_val': top_resids,
    },
    'bootstrap_1000': {
        'technical': {'median': float(np.median(boot_t)),
                      'ci_low': float(np.percentile(boot_t, 2.5)),
                      'ci_high': float(np.percentile(boot_t, 97.5))},
        'style': {'median': float(np.median(boot_s)),
                  'ci_low': float(np.percentile(boot_s, 2.5)),
                  'ci_high': float(np.percentile(boot_s, 97.5))},
    },
    'learning_curve_technical': lc_results,
    'learning_curve_style': lc_results_style,
    'cv_5fold_tech_n150': {'mean': cv_m, 'std': cv_s},
    'bimodal_separation': sep_results,
    'classification_accuracy': {
        'val_only_high_correct': float(hi_corr_v) if hi_corr_v is not None else None,
        'val_only_low_correct': float(lo_corr_v) if lo_corr_v is not None else None,
        'all_high_correct': float(hi_corr_all),
        'all_low_correct': float(lo_corr_all),
    },
}

with open(f'{BASE}/models/results_batch003.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("Saved: results_batch003.json")
print("\nDone.")
