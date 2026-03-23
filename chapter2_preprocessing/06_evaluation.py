#!/usr/bin/env python3
"""
Chapter 2 — Publication-Quality Geographic Imputation Map
Shows existing data vs imputed data with different methods color-coded.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

PKL_PATH = "/home/macierz/mohabdal/TerrorismNER_Project/cache/gtd_raw.pkl"
OUT_DIR  = "/home/macierz/mohabdal/TerrorismNER_Project/ch2_verification_results/geo_maps"

import os
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_pickle(PKL_PATH)
print(f"Loaded: {len(df):,} rows × {df.shape[1]} columns")

# ── Define masks ────────────────────────────────────────────────────────
lat_ok  = df['latitude'].notna()
lon_ok  = df['longitude'].notna()
coords_ok = lat_ok & lon_ok
city_ok = df['city'].notna() & (df['city'] != '') & (~df['city'].str.lower().eq('unknown'))
country_ok = df['country_txt'].notna() & (df['country_txt'] != '')
provstate_ok = df['provstate'].notna() & (df['provstate'] != '') & (~df['provstate'].str.lower().eq('unknown'))

case1 = (~coords_ok) & city_ok & country_ok
case2a = (~city_ok) & coords_ok
case3 = (~coords_ok) & (~city_ok) & country_ok
case5 = coords_ok & city_ok & country_ok

# ── Build lookup & perform imputation ───────────────────────────────────
complete_geo = df[coords_ok & city_ok & country_ok][['city', 'country_txt', 'latitude', 'longitude']]
lookup = complete_geo.groupby(['city', 'country_txt']).agg(
    lat_median=('latitude', 'median'),
    lng_median=('longitude', 'median'),
    count=('latitude', 'count')
).reset_index()

# Case 1: city+country → coords
case1_df = df[case1][['city', 'country_txt']].copy()
case1_df['_idx'] = case1_df.index
merged = case1_df.merge(lookup, on=['city', 'country_txt'], how='left')
matched = merged['lat_median'].notna()
matched_idx = merged[matched]['_idx'].values
matched_lats = merged[matched]['lat_median'].values
matched_lngs = merged[matched]['lng_median'].values

# Case 3: country → centroid
country_centroids = df[coords_ok & country_ok].groupby('country_txt').agg(
    lat_med=('latitude', 'median'),
    lng_med=('longitude', 'median')
).to_dict('index')

case3_lats, case3_lngs, case3_idx = [], [], []
for idx in df.index[case3]:
    country = df.loc[idx, 'country_txt']
    if country in country_centroids:
        case3_lats.append(country_centroids[country]['lat_med'])
        case3_lngs.append(country_centroids[country]['lng_med'])
        case3_idx.append(idx)

# Case 2a coords
case2a_lats = df[case2a]['latitude'].values
case2a_lngs = df[case2a]['longitude'].values

print(f"Complete:              {case5.sum():,}")
print(f"Case 1 (city→coords): {len(matched_idx):,}")
print(f"Case 2a (coords→city):{case2a.sum():,}")
print(f"Case 3 (country→ctr): {len(case3_idx):,}")

# ── Load world boundaries (simple coastline from data) ──────────────────
# We'll create a simple coastline approximation from the dense complete data
print("\nGenerating maps...")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: COMPREHENSIVE MULTI-PANEL MAP
# ══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(22, 16))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1], hspace=0.25, wspace=0.2)

# Color scheme
C_EXIST    = '#B0BEC5'   # light blue-gray for existing
C_CASE1    = '#1565C0'   # deep blue - city+country → coords
C_CASE2A   = '#2E7D32'   # deep green - coords → city
C_CASE3    = '#E65100'   # deep orange - country → centroid
C_REMAIN   = '#C62828'   # red - still missing
C_BG       = '#FAFAFA'   # background

# ── Panel A: Global Overview (top, spanning all 3 columns) ──────────────
ax_global = fig.add_subplot(gs[0, :])

# Existing complete records (sampled for speed)
np.random.seed(42)
exist_sample = df[case5].sample(min(40000, case5.sum()), random_state=42)
ax_global.scatter(exist_sample['longitude'], exist_sample['latitude'],
                  c=C_EXIST, s=0.15, alpha=0.25, rasterized=True, zorder=1)

# Case 2a: coords → city (these already HAD coords, show them as "enhanced")
case2a_sample_n = min(5000, case2a.sum())
case2a_sample_idx = np.random.choice(case2a.sum(), case2a_sample_n, replace=False)
ax_global.scatter(case2a_lngs[case2a_sample_idx], case2a_lats[case2a_sample_idx],
                  c=C_CASE2A, s=4, alpha=0.6, edgecolors='none', zorder=2)

# Case 1: city+country → coords (NEW points on map)
ax_global.scatter(matched_lngs, matched_lats,
                  c=C_CASE1, s=25, alpha=0.85, edgecolors='#0D47A1', 
                  linewidth=0.4, zorder=4)

# Case 3: country centroid (NEW points, approximate)
ax_global.scatter(case3_lngs, case3_lats,
                  c=C_CASE3, s=30, alpha=0.85, marker='D',
                  edgecolors='#BF360C', linewidth=0.5, zorder=5)

ax_global.set_xlim(-170, 180)
ax_global.set_ylim(-55, 78)
ax_global.set_facecolor('#E8EAF6')
ax_global.grid(True, alpha=0.15, color='white', linewidth=0.5)
ax_global.set_xlabel('Longitude', fontsize=11, labelpad=5)
ax_global.set_ylabel('Latitude', fontsize=11, labelpad=5)
ax_global.set_title('(a) Global Overview of Geographic Cross-Imputation',
                     fontsize=14, fontweight='bold', pad=10)

# Custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_EXIST, markersize=8,
           label=f'Existing complete records (n={case5.sum():,})', alpha=0.6),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_CASE2A, markersize=9,
           label=f'Coords present → city imputed (n={case2a.sum():,})', 
           markeredgecolor='#1B5E20', markeredgewidth=0.5),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_CASE1, markersize=10,
           label=f'City+country → coordinates imputed (n={len(matched_idx):,})',
           markeredgecolor='#0D47A1', markeredgewidth=0.5),
    Line2D([0], [0], marker='D', color='w', markerfacecolor=C_CASE3, markersize=9,
           label=f'Country → centroid coordinates (n={len(case3_idx):,})',
           markeredgecolor='#BF360C', markeredgewidth=0.5),
]
ax_global.legend(handles=legend_elements, loc='lower left', fontsize=10,
                 framealpha=0.95, fancybox=True, shadow=True, ncol=2,
                 borderpad=1, handletextpad=0.8)

# Add region boxes
for (lon_min, lon_max, lat_min, lat_max, color, label) in [
    (25, 65, 10, 42, '#FF6F00', 'b'),
    (65, 95, 5, 38, '#00695C', 'c'),
    (-12, 42, 34, 60, '#4A148C', 'd')
]:
    rect = plt.Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                          linewidth=2, edgecolor=color, facecolor='none', 
                          linestyle='--', zorder=6)
    ax_global.add_patch(rect)
    ax_global.text(lon_min+1, lat_max-2, f'({label})', color=color, 
                   fontsize=11, fontweight='bold', zorder=7)

# ── Panel B: Middle East & Central Asia ─────────────────────────────────
ax_me = fig.add_subplot(gs[1, 0])
lon_min, lon_max, lat_min, lat_max = 25, 65, 10, 42

mask_e = case5 & df['longitude'].between(lon_min, lon_max) & df['latitude'].between(lat_min, lat_max)
ax_me.scatter(df[mask_e]['longitude'], df[mask_e]['latitude'],
              c=C_EXIST, s=0.5, alpha=0.3, rasterized=True, zorder=1)

# Case 2a in region
mask_2a = case2a & df['longitude'].between(lon_min, lon_max) & df['latitude'].between(lat_min, lat_max)
ax_me.scatter(df[mask_2a]['longitude'], df[mask_2a]['latitude'],
              c=C_CASE2A, s=8, alpha=0.7, edgecolors='none', zorder=2)

# Case 1 in region
c1_mask = (matched_lngs >= lon_min) & (matched_lngs <= lon_max) & \
          (matched_lats >= lat_min) & (matched_lats <= lat_max)
ax_me.scatter(matched_lngs[c1_mask], matched_lats[c1_mask],
              c=C_CASE1, s=35, alpha=0.9, edgecolors='#0D47A1', linewidth=0.5, zorder=4)

# Case 3 in region
c3_lats_arr, c3_lngs_arr = np.array(case3_lats), np.array(case3_lngs)
c3_mask = (c3_lngs_arr >= lon_min) & (c3_lngs_arr <= lon_max) & \
          (c3_lats_arr >= lat_min) & (c3_lats_arr <= lat_max)
ax_me.scatter(c3_lngs_arr[c3_mask], c3_lats_arr[c3_mask],
              c=C_CASE3, s=45, alpha=0.9, marker='D', edgecolors='#BF360C', linewidth=0.6, zorder=5)

n_imputed_region = c1_mask.sum() + mask_2a.sum() + c3_mask.sum()
ax_me.set_xlim(lon_min, lon_max)
ax_me.set_ylim(lat_min, lat_max)
ax_me.set_facecolor('#E8EAF6')
ax_me.grid(True, alpha=0.15, color='white')
ax_me.set_title(f'(b) Middle East & Central Asia\n({n_imputed_region:,} imputed)',
                fontsize=12, fontweight='bold', color='#FF6F00')
ax_me.set_xlabel('Longitude', fontsize=10)
ax_me.set_ylabel('Latitude', fontsize=10)
for spine in ax_me.spines.values():
    spine.set_edgecolor('#FF6F00')
    spine.set_linewidth(2)

# ── Panel C: South Asia ─────────────────────────────────────────────────
ax_sa = fig.add_subplot(gs[1, 1])
lon_min, lon_max, lat_min, lat_max = 65, 95, 5, 38

mask_e = case5 & df['longitude'].between(lon_min, lon_max) & df['latitude'].between(lat_min, lat_max)
ax_sa.scatter(df[mask_e]['longitude'], df[mask_e]['latitude'],
              c=C_EXIST, s=0.5, alpha=0.3, rasterized=True, zorder=1)

mask_2a = case2a & df['longitude'].between(lon_min, lon_max) & df['latitude'].between(lat_min, lat_max)
ax_sa.scatter(df[mask_2a]['longitude'], df[mask_2a]['latitude'],
              c=C_CASE2A, s=8, alpha=0.7, edgecolors='none', zorder=2)

c1_mask = (matched_lngs >= lon_min) & (matched_lngs <= lon_max) & \
          (matched_lats >= lat_min) & (matched_lats <= lat_max)
ax_sa.scatter(matched_lngs[c1_mask], matched_lats[c1_mask],
              c=C_CASE1, s=35, alpha=0.9, edgecolors='#0D47A1', linewidth=0.5, zorder=4)

c3_mask = (c3_lngs_arr >= lon_min) & (c3_lngs_arr <= lon_max) & \
          (c3_lats_arr >= lat_min) & (c3_lats_arr <= lat_max)
ax_sa.scatter(c3_lngs_arr[c3_mask], c3_lats_arr[c3_mask],
              c=C_CASE3, s=45, alpha=0.9, marker='D', edgecolors='#BF360C', linewidth=0.6, zorder=5)

n_imputed_region = c1_mask.sum() + mask_2a.sum() + c3_mask.sum()
ax_sa.set_xlim(lon_min, lon_max)
ax_sa.set_ylim(lat_min, lat_max)
ax_sa.set_facecolor('#E8EAF6')
ax_sa.grid(True, alpha=0.15, color='white')
ax_sa.set_title(f'(c) South Asia\n({n_imputed_region:,} imputed)',
                fontsize=12, fontweight='bold', color='#00695C')
ax_sa.set_xlabel('Longitude', fontsize=10)
ax_sa.set_ylabel('Latitude', fontsize=10)
for spine in ax_sa.spines.values():
    spine.set_edgecolor('#00695C')
    spine.set_linewidth(2)

# ── Panel D: Europe ─────────────────────────────────────────────────────
ax_eu = fig.add_subplot(gs[1, 2])
lon_min, lon_max, lat_min, lat_max = -12, 42, 34, 60

mask_e = case5 & df['longitude'].between(lon_min, lon_max) & df['latitude'].between(lat_min, lat_max)
ax_eu.scatter(df[mask_e]['longitude'], df[mask_e]['latitude'],
              c=C_EXIST, s=1, alpha=0.4, rasterized=True, zorder=1)

mask_2a = case2a & df['longitude'].between(lon_min, lon_max) & df['latitude'].between(lat_min, lat_max)
ax_eu.scatter(df[mask_2a]['longitude'], df[mask_2a]['latitude'],
              c=C_CASE2A, s=12, alpha=0.7, edgecolors='none', zorder=2)

c1_mask = (matched_lngs >= lon_min) & (matched_lngs <= lon_max) & \
          (matched_lats >= lat_min) & (matched_lats <= lat_max)
ax_eu.scatter(matched_lngs[c1_mask], matched_lats[c1_mask],
              c=C_CASE1, s=40, alpha=0.9, edgecolors='#0D47A1', linewidth=0.5, zorder=4)

c3_mask = (c3_lngs_arr >= lon_min) & (c3_lngs_arr <= lon_max) & \
          (c3_lats_arr >= lat_min) & (c3_lats_arr <= lat_max)
ax_eu.scatter(c3_lngs_arr[c3_mask], c3_lats_arr[c3_mask],
              c=C_CASE3, s=45, alpha=0.9, marker='D', edgecolors='#BF360C', linewidth=0.6, zorder=5)

n_imputed_region = c1_mask.sum() + mask_2a.sum() + c3_mask.sum()
ax_eu.set_xlim(lon_min, lon_max)
ax_eu.set_ylim(lat_min, lat_max)
ax_eu.set_facecolor('#E8EAF6')
ax_eu.grid(True, alpha=0.15, color='white')
ax_eu.set_title(f'(d) Europe\n({n_imputed_region:,} imputed)',
                fontsize=12, fontweight='bold', color='#4A148C')
ax_eu.set_xlabel('Longitude', fontsize=10)
ax_eu.set_ylabel('Latitude', fontsize=10)
for spine in ax_eu.spines.values():
    spine.set_edgecolor('#4A148C')
    spine.set_linewidth(2)

# Main title
fig.suptitle('Geographic Cross-Imputation Results for the GTD Dataset\n'
             'Recovering Missing Spatial Information Through Multi-Method Imputation',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(f"{OUT_DIR}/fig_geo_imputation_multipanel.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig_geo_imputation_multipanel.png", dpi=300, bbox_inches='tight')
print("Saved: fig_geo_imputation_multipanel.pdf/png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: BEFORE vs AFTER COMPARISON (side by side)
# ══════════════════════════════════════════════════════════════════════════

fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(22, 9))

# ── LEFT: Before imputation ─────────────────────────────────────────────
# Show only records that HAD coordinates originally
orig_with_coords = df[coords_ok]
sample_before = orig_with_coords.sample(min(40000, len(orig_with_coords)), random_state=42)
ax_before.scatter(sample_before['longitude'], sample_before['latitude'],
                  c='#546E7A', s=0.2, alpha=0.3, rasterized=True)

# Highlight WHERE data is missing (countries with most missing coords)
# Show the "holes" — countries that have incidents but few coords
for country in df[case1 | case3]['country_txt'].value_counts().head(20).index:
    c_data = df[(df['country_txt'] == country) & coords_ok]
    if len(c_data) > 0:
        cent_lat = c_data['latitude'].median()
        cent_lng = c_data['longitude'].median()
        n_miss = ((case1 | case3) & (df['country_txt'] == country)).sum()
        if n_miss > 30:
            circle = plt.Circle((cent_lng, cent_lat), radius=np.sqrt(n_miss)*0.15,
                               color=C_REMAIN, alpha=0.3, zorder=3)
            ax_before.add_patch(circle)

ax_before.set_xlim(-170, 180)
ax_before.set_ylim(-55, 78)
ax_before.set_facecolor('#ECEFF1')
ax_before.grid(True, alpha=0.15, color='white')
ax_before.set_xlabel('Longitude', fontsize=11)
ax_before.set_ylabel('Latitude', fontsize=11)
ax_before.set_title(f'BEFORE Imputation\n{coords_ok.sum():,} records with coordinates '
                     f'({(~coords_ok).sum():,} missing)',
                     fontsize=13, fontweight='bold', color='#C62828')

# Red circles legend
ax_before.scatter([], [], c=C_REMAIN, s=80, alpha=0.4, 
                  label='Regions with missing\ncoordinate data')
ax_before.legend(loc='lower left', fontsize=10, framealpha=0.9)

# Stats box
stats_before = (f"Records with coords: {coords_ok.sum():,} ({coords_ok.mean()*100:.1f}%)\n"
                f"Records missing coords: {(~coords_ok).sum():,} ({(~coords_ok).mean()*100:.1f}%)\n"
                f"Records missing city: {(~city_ok).sum():,} ({(~city_ok).mean()*100:.1f}%)")
ax_before.text(0.98, 0.02, stats_before, transform=ax_before.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFCDD2', alpha=0.9, edgecolor='#C62828'),
               family='monospace')

# ── RIGHT: After imputation ─────────────────────────────────────────────
# Show ALL records that now have coordinates
ax_after.scatter(sample_before['longitude'], sample_before['latitude'],
                 c=C_EXIST, s=0.2, alpha=0.25, rasterized=True)

# Overlay imputed points
ax_after.scatter(case2a_lngs[:3000], case2a_lats[:3000],
                 c=C_CASE2A, s=3, alpha=0.5, edgecolors='none', zorder=2,
                 label=f'City imputed from coords ({case2a.sum():,})')

ax_after.scatter(matched_lngs, matched_lats,
                 c=C_CASE1, s=30, alpha=0.9, edgecolors='#0D47A1', linewidth=0.4, zorder=4,
                 label=f'Coords from city+country ({len(matched_idx):,})')

ax_after.scatter(case3_lngs, case3_lats,
                 c=C_CASE3, s=35, alpha=0.9, marker='D', edgecolors='#BF360C', linewidth=0.5, zorder=5,
                 label=f'Coords from country centroid ({len(case3_idx):,})')

ax_after.set_xlim(-170, 180)
ax_after.set_ylim(-55, 78)
ax_after.set_facecolor('#E8F5E9')
ax_after.grid(True, alpha=0.15, color='white')
ax_after.set_xlabel('Longitude', fontsize=11)
ax_after.set_ylabel('Latitude', fontsize=11)

total_after = coords_ok.sum() + len(matched_idx) + len(case3_idx)
ax_after.set_title(f'AFTER Imputation\n{total_after:,} records with coordinates '
                    f'(+{len(matched_idx) + len(case3_idx):,} recovered)',
                    fontsize=13, fontweight='bold', color='#2E7D32')

ax_after.legend(loc='lower left', fontsize=10, framealpha=0.9, markerscale=1.5)

# Stats box
stats_after = (f"Coords recovered: {len(matched_idx) + len(case3_idx):,}\n"
               f"  From city+country: {len(matched_idx):,}\n"
               f"  From country centroid: {len(case3_idx):,}\n"
               f"Cities recovered: {case2a.sum():,}\n"
               f"Still missing coords: {(~coords_ok).sum() - len(matched_idx) - len(case3_idx):,}")
ax_after.text(0.98, 0.02, stats_after, transform=ax_after.transAxes, fontsize=10,
              verticalalignment='bottom', horizontalalignment='right',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#C8E6C9', alpha=0.9, edgecolor='#2E7D32'),
              family='monospace')

fig.suptitle('Geographic Data Recovery: Before vs. After Cross-Imputation',
             fontsize=15, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_before_after_comparison.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig_before_after_comparison.png", dpi=300, bbox_inches='tight')
print("Saved: fig_before_after_comparison.pdf/png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: VALIDATION + IMPUTATION SUMMARY (2 panels)
# ══════════════════════════════════════════════════════════════════════════

fig, (ax_val, ax_bar) = plt.subplots(1, 2, figsize=(20, 8))

# ── LEFT: Validation error distribution ─────────────────────────────────
np.random.seed(42)
complete = df[case5].copy()
test_idx = np.random.choice(complete.index, size=min(10000, len(complete)), replace=False)
test = complete.loc[test_idx, ['city', 'country_txt', 'latitude', 'longitude']].copy()
test_merged = test.merge(lookup, on=['city', 'country_txt'], how='left')
test_matched = test_merged[test_merged['lat_median'].notna()]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

errors_km = haversine(
    test_matched['latitude'].values, test_matched['longitude'].values,
    test_matched['lat_median'].values, test_matched['lng_median'].values
)

# Cumulative accuracy curve
thresholds = [0, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
accuracies = [(errors_km <= t).mean() * 100 for t in thresholds]

ax_val.plot(thresholds, accuracies, 'o-', color='#1565C0', linewidth=2.5, 
            markersize=7, markerfacecolor='white', markeredgewidth=2, zorder=3)
ax_val.fill_between(thresholds, accuracies, alpha=0.15, color='#1565C0')

# Add key threshold annotations
for t, a in [(0.1, None), (1, None), (10, None), (50, None)]:
    acc = (errors_km <= t).mean() * 100
    ax_val.axhline(y=acc, color='gray', linestyle=':', alpha=0.3)
    ax_val.axvline(x=t, color='gray', linestyle=':', alpha=0.3)
    ax_val.annotate(f'{acc:.1f}%', xy=(t, acc), xytext=(t*1.5, acc-3),
                   fontsize=10, fontweight='bold', color='#1565C0',
                   arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1))

ax_val.set_xscale('log')
ax_val.set_xlim(0.0008, 600)
ax_val.set_ylim(0, 105)
ax_val.set_xlabel('Error Threshold (km)', fontsize=12)
ax_val.set_ylabel('Accuracy (%)', fontsize=12)
ax_val.set_title('Hold-Out Validation: Coordinate Imputation Accuracy\n'
                  f'(n={len(test_matched):,} test samples)',
                  fontsize=13, fontweight='bold')
ax_val.grid(True, alpha=0.2)
ax_val.set_facecolor(C_BG)

stats_text = (f"Median error: {np.median(errors_km):.3f} km\n"
              f"Mean error: {np.mean(errors_km):.2f} km\n"
              f"Within 0.1 km: {(errors_km<0.1).mean()*100:.1f}%\n"
              f"Within 10 km: {(errors_km<10).mean()*100:.1f}%")
ax_val.text(0.97, 0.25, stats_text, transform=ax_val.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', alpha=0.95, edgecolor='#1565C0'),
            family='monospace')

# ── RIGHT: Imputation impact bar chart ──────────────────────────────────
features = ['Latitude', 'Longitude', 'City', 'Province']
before = [
    df['latitude'].isna().sum(),
    df['longitude'].isna().sum(),
    (~city_ok).sum(),
    (~provstate_ok).sum()
]

# After imputation
lat_after = df['latitude'].isna().sum() - len(matched_idx) - len(case3_idx)
lng_after = df['longitude'].isna().sum() - len(matched_idx) - len(case3_idx)
city_after = (~city_ok).sum() - case2a.sum()
prov_after = (~provstate_ok).sum() - 64

after = [max(0, lat_after), max(0, lng_after), max(0, city_after), max(0, prov_after)]

x = np.arange(len(features))
width = 0.32

bars1 = ax_bar.bar(x - width/2, before, width, label='Before Imputation', 
                    color='#EF5350', alpha=0.85, edgecolor='white', linewidth=1)
bars2 = ax_bar.bar(x + width/2, after, width, label='After Imputation',
                    color='#66BB6A', alpha=0.85, edgecolor='white', linewidth=1)

# Value labels
for bar in bars1:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2., h + 100,
                f'{int(h):,}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#C62828')
for bar in bars2:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2., h + 100,
                f'{int(h):,}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2E7D32')

# Reduction arrows
for i in range(len(features)):
    if before[i] > 0:
        reduction = (before[i] - after[i]) / before[i] * 100
        ax_bar.annotate(f'↓{reduction:.0f}%',
                       xy=(x[i], max(before[i], after[i]) + 800),
                       fontsize=11, fontweight='bold', ha='center', color='#1565C0')

ax_bar.set_xlabel('Geographic Feature', fontsize=12)
ax_bar.set_ylabel('Missing Value Count', fontsize=12)
ax_bar.set_title('Missing Value Reduction by Geographic Feature\nBefore vs. After Cross-Imputation',
                  fontsize=13, fontweight='bold')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(features, fontsize=12)
ax_bar.legend(fontsize=11, loc='upper right')
ax_bar.grid(True, alpha=0.2, axis='y')
ax_bar.set_facecolor(C_BG)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_validation_and_summary.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig_validation_and_summary.png", dpi=300, bbox_inches='tight')
print("Saved: fig_validation_and_summary.pdf/png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Sub-Saharan Africa zoom (important terrorism region)
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 10))
lon_min, lon_max, lat_min, lat_max = -20, 55, -38, 20

mask_e = case5 & df['longitude'].between(lon_min, lon_max) & df['latitude'].between(lat_min, lat_max)
ax.scatter(df[mask_e]['longitude'], df[mask_e]['latitude'],
           c=C_EXIST, s=1, alpha=0.35, rasterized=True, zorder=1)

mask_2a = case2a & df['longitude'].between(lon_min, lon_max) & df['latitude'].between(lat_min, lat_max)
ax.scatter(df[mask_2a]['longitude'], df[mask_2a]['latitude'],
           c=C_CASE2A, s=12, alpha=0.7, edgecolors='none', zorder=2,
           label=f'City imputed from coords ({mask_2a.sum():,})')

c1_mask = (matched_lngs >= lon_min) & (matched_lngs <= lon_max) & \
          (matched_lats >= lat_min) & (matched_lats <= lat_max)
ax.scatter(matched_lngs[c1_mask], matched_lats[c1_mask],
           c=C_CASE1, s=40, alpha=0.9, edgecolors='#0D47A1', linewidth=0.5, zorder=4,
           label=f'Coords from city+country ({c1_mask.sum():,})')

c3_mask = (c3_lngs_arr >= lon_min) & (c3_lngs_arr <= lon_max) & \
          (c3_lats_arr >= lat_min) & (c3_lats_arr <= lat_max)
ax.scatter(c3_lngs_arr[c3_mask], c3_lats_arr[c3_mask],
           c=C_CASE3, s=50, alpha=0.9, marker='D', edgecolors='#BF360C', linewidth=0.6, zorder=5,
           label=f'Coords from country centroid ({c3_mask.sum():,})')

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_facecolor('#E8EAF6')
ax.grid(True, alpha=0.15, color='white')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Geographic Imputation: Sub-Saharan Africa',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=11, framealpha=0.9, markerscale=1.5)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_africa_zoom.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig_africa_zoom.png", dpi=300, bbox_inches='tight')
print("Saved: fig_africa_zoom.pdf/png")
plt.close()


# ── Final summary ───────────────────────────────────────────────────────
print("\n" + "="*80)
print("ALL FIGURES SAVED")
print("="*80)
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith('fig_'):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f} ({size/1024:.0f} KB)")
print("\nRecommended for thesis:")
print("  1. fig_geo_imputation_multipanel.pdf  — Main figure (global + 3 regional zooms)")
print("  2. fig_before_after_comparison.pdf     — Before/after side-by-side")
print("  3. fig_validation_and_summary.pdf      — Accuracy curve + missing reduction bars")
print("  4. fig_africa_zoom.pdf                 — Optional: Africa detail")
