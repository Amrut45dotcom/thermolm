import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

os.makedirs('reports', exist_ok=True)

df = pd.read_csv('data/FeNdB_dataset_long.csv')
PHASES = sorted(df['phase'].unique().tolist())

BG     = '#0f0f0f'
AX_BG  = '#1a1a1a'
TEXT   = '#e0e0e0'
ACCENT = ['#00e5ff','#ff4081','#69ff47','#ffab00','#e040fb','#ff6d00']

def style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

phase_counts = df['phase'].value_counts()
phases_per_ct = df.groupby(['x_Nd', 'x_B', 'temperature_C'])['phase'].count()


fig1 = plt.figure(figsize=(18, 5))
fig1.patch.set_facecolor(BG)
gs1 = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.38)

ax = fig1.add_subplot(gs1[0, 0])
bars = ax.barh(phase_counts.index[::-1], phase_counts.values[::-1], color=ACCENT)
ax.set_xlabel('Row count')
ax.set_title('Phase Frequency', fontsize=10, fontweight='bold')
for bar, val in zip(bars, phase_counts.values[::-1]):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
            f'{val:,} ({val/len(df)*100:.1f}%)', va='center', fontsize=7, color=TEXT)
style_ax(ax)

ax = fig1.add_subplot(gs1[0, 1])
comps = df[['x_Nd', 'x_B']].drop_duplicates()
ax.scatter(comps['x_Nd'], comps['x_B'], s=6, alpha=0.6, color='#00e5ff')
ax.axvline(x=2/17, color='#ff4081', linewidth=1, linestyle='--', alpha=0.8)
ax.axhline(y=1/17, color='#ff4081', linewidth=1, linestyle='--', alpha=0.8)
ax.set_xlabel('x_Nd')
ax.set_ylabel('x_B')
ax.set_title('Composition Space', fontsize=10, fontweight='bold')
style_ax(ax)

ax = fig1.add_subplot(gs1[0, 2])
ax.hist(phases_per_ct.values, bins=[0.5, 1.5, 2.5, 3.5],
        color='#00e5ff', edgecolor='white', linewidth=0.5, rwidth=0.8)
ax.set_xlabel('# phases present')
ax.set_ylabel('# (comp, T) combos')
ax.set_title('Phases per (Composition, T)', fontsize=10, fontweight='bold')
ax.set_xticks([1, 2, 3])
style_ax(ax)

fig1.suptitle('ThermoLM Fe-Nd-B — Dataset Overview', fontsize=13,
              fontweight='bold', color='white', y=1.02)
plt.savefig('reports/eda_dataset_overview.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig1)
print("Saved: reports/eda_dataset_overview.png")


fig2, axes = plt.subplots(1, 2, figsize=(18, 5))
fig2.patch.set_facecolor(BG)

ax = axes[0]
phase_order = df.groupby('phase')['NP'].median().sort_values(ascending=False).index
bp = ax.boxplot([df[df['phase'] == p]['NP'].values for p in phase_order],
                tick_labels=phase_order, patch_artist=True,
                medianprops=dict(color='white', linewidth=1.5),
                whiskerprops=dict(color='#555'),
                capprops=dict(color='#555'),
                flierprops=dict(marker='.', markersize=2, alpha=0.3, color='#888'))
for patch, color in zip(bp['boxes'], ACCENT):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('NP (phase fraction)')
ax.set_title('NP Distribution per Phase', fontsize=10, fontweight='bold')
ax.tick_params(axis='x', rotation=35, labelsize=7)
style_ax(ax)

ax = axes[1]
pivot = df.groupby(['temperature_C', 'phase']).size().unstack(fill_value=0)
pivot = pivot[phase_counts.index]
sns.heatmap(pivot.T, ax=ax, cmap='YlOrRd', linewidths=0.3,
            cbar_kws={'label': 'row count', 'shrink': 0.6},
            xticklabels=[str(t) if i % 2 == 0 else '' for i, t in enumerate(pivot.index)])
ax.set_title('Phase Presence vs Temperature', fontsize=10, fontweight='bold')
ax.set_xlabel('Temperature (°C)', color=TEXT)
ax.set_ylabel('')
ax.tick_params(colors=TEXT, labelsize=7)
ax.title.set_color(TEXT)
ax.set_facecolor(AX_BG)

fig2.suptitle('ThermoLM Fe-Nd-B — Phase Distribution', fontsize=13,
              fontweight='bold', color='white', y=1.02)
plt.savefig('reports/eda_phase_distribution.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print("Saved: reports/eda_phase_distribution.png")


fig3, axes = plt.subplots(2, 3, figsize=(18, 10))
fig3.patch.set_facecolor(BG)
axes = axes.flatten()

for idx, (phase, color) in enumerate(zip(PHASES, ACCENT)):
    ax = axes[idx]
    phase_df = df[df['phase'] == phase]
    temp_np = phase_df.groupby('temperature_C')['NP'].agg(['mean', 'std']).reset_index()
    ax.plot(temp_np['temperature_C'], temp_np['mean'], color=color, linewidth=2)
    ax.fill_between(temp_np['temperature_C'],
                    temp_np['mean'] - temp_np['std'],
                    temp_np['mean'] + temp_np['std'],
                    alpha=0.2, color=color)
    ax.set_title(phase, fontsize=10, fontweight='bold')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('NP (mean ± std)')
    style_ax(ax)

fig3.suptitle('ThermoLM Fe-Nd-B — NP vs Temperature per Phase', fontsize=13,
              fontweight='bold', color='white', y=1.01)
plt.tight_layout()
plt.savefig('reports/eda_np_vs_temperature.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig3)
print("Saved: reports/eda_np_vs_temperature.png")


fig4, axes = plt.subplots(2, 3, figsize=(18, 10))
fig4.patch.set_facecolor(BG)
axes = axes.flatten()

for idx, (phase, color) in enumerate(zip(PHASES, ACCENT)):
    ax = axes[idx]
    phase_df = df[df['phase'] == phase][['x_Nd', 'x_B', 'temperature_C', 'NP']].copy()
    corr = phase_df.corr()['NP'].drop('NP')
    bars = ax.barh(corr.index, corr.values,
                   color=[color if v >= 0 else '#ff1744' for v in corr.values], alpha=0.8)
    ax.axvline(0, color='#555', linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_title(phase, fontsize=10, fontweight='bold')
    ax.set_xlabel('Pearson r with NP')
    style_ax(ax)

fig4.suptitle('ThermoLM Fe-Nd-B — Input Correlation with NP per Phase', fontsize=13,
              fontweight='bold', color='white', y=1.01)
plt.tight_layout()
plt.savefig('reports/eda_input_np_correlation.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig4)
print("Saved: reports/eda_input_np_correlation.png")