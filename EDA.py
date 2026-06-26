import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# import pandas as pd

# df = pd.read_csv('data/FeNdB_ML_dataset_long_constrained.csv')

# # Pearson correlations apply to numeric columns; ``phase`` is categorical.
# numeric_df = df.select_dtypes(include=np.number)
# pearson_corr = numeric_df.corr(method='pearson')

# print("Pearson correlation matrix:")
# print(pearson_corr.round(4).to_string())

# # Pearson correlation heatmap
# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(
#     pearson_corr,
#     annot=True,
#     fmt='.2f',
#     cmap='coolwarm',
#     vmin=-1,
#     vmax=1,
#     center=0,
#     square=True,
#     linewidths=0.5,
#     cbar_kws={'label': 'Pearson correlation'},
#     ax=ax,
# )
# ax.set_title('Pearson Correlation Heatmap')
# plt.tight_layout()
# plt.savefig('pearson_correlation_heatmap.png', dpi=150, bbox_inches='tight')
# plt.close(fig)
# print("Saved: pearson_correlation_heatmap.png")

# sns.pairplot(df, vars=['x_Nd', 'x_B', 'temperature_C', 'NP'], hue='phase', palette='tab10', plot_kws={'alpha': 0.5})
# plt.savefig('pairplot.png', dpi=150, bbox_inches='tight')


# ── Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/FeNdB_dataset_long.csv')

# ── Quick stats ─────────────────────────────────────────────────────────────
print("Shape:", df.shape)
print("\nPhase counts:\n", df['phase'].value_counts())
print("\nNP stats per phase:\n",
      df.groupby('phase')['NP'].agg(['mean','median','std','min','max']).round(4))

phases_per_ct = df.groupby(['x_Nd','x_B','temperature_C'])['phase'].count()
print("\nPhases per (comp, T):\n", phases_per_ct.describe().round(3))

# ── Plot setup ───────────────────────────────────────────────────────────────
ACCENT = ['#00e5ff','#ff4081','#69ff47','#ffab00','#e040fb',
          '#ff6d00','#1de9b6','#c6ff00','#ff1744','#d500f9',
          '#00b0ff','#76ff03','#ff3d00']
BG, AX_BG, TEXT = '#0f0f0f', '#1a1a1a', '#e0e0e0'

def style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

phase_counts = df['phase'].value_counts()

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── 1. Phase frequency ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
bars = ax1.barh(phase_counts.index[::-1], phase_counts.values[::-1], color=ACCENT[:13])
ax1.set_xlabel('Row count')
ax1.set_title('Phase Frequency', fontsize=10, fontweight='bold')
for bar, val in zip(bars, phase_counts.values[::-1]):
    ax1.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
             f'{val:,} ({val/len(df)*100:.1f}%)', va='center', fontsize=7, color=TEXT)
style_ax(ax1)

# ── 2. Composition space ─────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
comps = df[['x_Nd','x_B']].drop_duplicates()
ax2.scatter(comps['x_Nd'], comps['x_B'], s=6, alpha=0.6, color='#00e5ff')
ax2.axvline(x=2/17, color='#ff4081', linewidth=1, linestyle='--', alpha=0.8)
ax2.axhline(y=1/17, color='#ff4081', linewidth=1, linestyle='--', alpha=0.8)
ax2.text(2/17+0.01, 0.85, 'Nd₂Fe₁₄B\nstoich', color='#ff4081', fontsize=7)
ax2.set_xlabel('x_Nd'); ax2.set_ylabel('x_B')
ax2.set_title('Composition Space\n(502 unique points)', fontsize=10, fontweight='bold')
style_ax(ax2)

# ── 3. NP boxplot per phase ──────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
phase_order = df.groupby('phase')['NP'].median().sort_values(ascending=False).index
bp = ax3.boxplot([df[df['phase']==p]['NP'].values for p in phase_order],
                 tick_labels=phase_order, patch_artist=True,
                 medianprops=dict(color='white', linewidth=1.5),
                 whiskerprops=dict(color='#555'),
                 capprops=dict(color='#555'),
                 flierprops=dict(marker='.', markersize=2, alpha=0.3, color='#888'))
for patch, color in zip(bp['boxes'], ACCENT):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax3.set_ylabel('NP (phase fraction)')
ax3.set_title('NP Distribution per Phase', fontsize=10, fontweight='bold')
ax3.tick_params(axis='x', rotation=35, labelsize=7)
style_ax(ax3)

# ── 4. Phases per (comp, T) histogram ───────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(phases_per_ct.values, bins=[0.5,1.5,2.5,3.5],
         color='#00e5ff', edgecolor='white', linewidth=0.5, rwidth=0.8)
ax4.set_xlabel('# phases present')
ax4.set_ylabel('# (comp, T) combos')
ax4.set_title('Phases per\n(Composition, T)', fontsize=10, fontweight='bold')
ax4.set_xticks([1,2,3])
style_ax(ax4)

# ── 5. Phase presence vs temperature heatmap ────────────────────────────────
ax5 = fig.add_subplot(gs[2, :])
pivot = df.groupby(['temperature_C','phase']).size().unstack(fill_value=0)
pivot = pivot[phase_counts.index]
sns.heatmap(pivot.T, ax=ax5, cmap='YlOrRd', linewidths=0.3,
            cbar_kws={'label': 'row count', 'shrink': 0.6},
            xticklabels=[str(t) if i%2==0 else '' for i,t in enumerate(pivot.index)])
ax5.set_title('Phase Presence Across Temperature', fontsize=10, fontweight='bold')
ax5.set_xlabel('Temperature (°C)', color=TEXT)
ax5.set_ylabel('')
ax5.tick_params(colors=TEXT, labelsize=7)
ax5.title.set_color(TEXT)
ax5.set_facecolor(AX_BG)

fig.suptitle('ThermoLM — Fe-Nd-B EDA Overview', fontsize=14,
             fontweight='bold', color='white', y=0.98)
plt.savefig('FeNdB_EDA.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved: FeNdB_EDA.png")
