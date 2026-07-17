"""
correlation_analysis.py
Analyse la corrélation entre antennes calculées (features) 
et cellule_home / cellule_work
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(".")))

from features import DistanceIndex, compute_all_features

PATH_SOURCE = Path("data/raw")
PATH_DATA   = Path("data")
DAY         = 0   # on analyse un seul jour

# --- Chargement ---
print("Chargement...")
dist_matrix_raw = np.load(PATH_DATA / "dist_matrix.npy", allow_pickle=True)
cell_ids        = np.load(PATH_DATA / "cell_ids.npy", allow_pickle=True)
dist_idx        = DistanceIndex(dist_matrix_raw, cell_ids)

events = np.load(PATH_SOURCE / f"events_day_{DAY+12}.npz", allow_pickle=True)["data"].item()
meta   = np.load(PATH_SOURCE / f"meta_day_{DAY+12}.npy",   allow_pickle=True).item()

# --- Calcul features ---
print("Calcul des features...")
features = compute_all_features(events, meta, dist_idx, verbose=True)

# --- Build DataFrame ---
print("Building DataFrame...")

rows = []

for uid, f in features.items():

    sig = f.get("signature_horaire")

    if sig is None:
        continue

    row = {
        "cellule_home": f.get("cellule_home"),
        "cellule_work": f.get("cellule_work"),
    }

    for h in range(24):
        row[f"h{h:02d}"] = sig[h]

    rows.append(row)

df = pd.DataFrame(rows)

print(f"\nDataFrame: {len(df):,} users")

# --- Hourly correlations ---
hours = list(range(24))

home_rates = []
work_rates = []

home_ns = []
work_ns = []

for h in hours:

    col = f"h{h:02d}"

    # vs home
    mask_home = df[col].notna() & df["cellule_home"].notna()

    if mask_home.any():
        sub_home = df[mask_home]
        rate_home = (sub_home[col] == sub_home["cellule_home"]).mean() * 100
        n_home = len(sub_home)
    else:
        rate_home = np.nan
        n_home = 0

    home_rates.append(rate_home)
    home_ns.append(n_home)

    # vs work
    mask_work = df[col].notna() & df["cellule_work"].notna()

    if mask_work.any():
        sub_work = df[mask_work]
        rate_work = (sub_work[col] == sub_work["cellule_work"]).mean() * 100
        n_work = len(sub_work)
    else:
        rate_work = np.nan
        n_work = 0

    work_rates.append(rate_work)
    work_ns.append(n_work)

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# HOME
axes[0].plot(hours, home_rates, marker="o")
axes[0].set_title("Hourly antenna correlation with home cell")
axes[0].set_ylabel("Matching rate (%)")
axes[0].set_ylim(0, 100)
axes[0].grid(True, alpha=0.3)

# WORK
axes[1].plot(hours, work_rates, marker="o")
axes[1].set_title("Hourly antenna correlation with work cell")
axes[1].set_ylabel("Matching rate (%)")
axes[1].set_xlabel("Hour of day")
axes[1].set_ylim(0, 100)
axes[1].grid(True, alpha=0.3)

plt.xticks(hours)

plt.tight_layout()

plt.savefig("results/hourly_correlation.png", dpi=150)
plt.close()

print("\n✓ results/hourly_correlation.png")













# # --- Construire le DataFrame ---
# print("Construction du DataFrame...")
# rows = []
# for uid, f in features.items():
#     rows.append({
#         "night_cell":  f.get("antenne_nuit"),
#         "day_cell":  f.get("antenne_jour"),
#         "modal_cell": f.get("antenne_modale"),
#         "cellule_home":  f.get("cellule_home"),
#         "cellule_work":  f.get("cellule_work"),
#     })
# df = pd.DataFrame(rows)

# print(f"\nDataFrame : {len(df):,} utilisateurs")
# print(f"  cellule_home renseignée : {df['cellule_home'].notna().sum():,}")
# print(f"  cellule_work renseignée : {df['cellule_work'].notna().sum():,}")
# print(f"  antenne_nuit calculée   : {df['night_cell'].notna().sum():,}")
# print(f"  antenne_jour calculée   : {df['day_cell'].notna().sum():,}")

# # --- Calcul des taux de concordance ---
# # Pour chaque paire (feature calculée, champ meta), calculer :
# # % de fois où les deux sont identiques (parmi les cas où les deux sont renseignés)

# pairs = [
#     ("night_cell",   "cellule_home"),
#     ("night_cell",   "cellule_work"),
#     ("day_cell",   "cellule_home"),
#     ("day_cell",   "cellule_work"),
#     ("modal_cell", "cellule_home"),
#     ("modal_cell", "cellule_work"),
# ]

# print("\n=== Taux de concordance (même antenne) ===")
# results = []
# for feat, ref in pairs:
#     mask = df[feat].notna() & df[ref].notna()
#     sub  = df[mask]
#     if len(sub) == 0:
#         continue
#     match_rate = (sub[feat] == sub[ref]).mean() * 100
#     n          = len(sub)
#     results.append({
#         "feature":    feat,
#         "reference":  ref,
#         "n":          n,
#         "match_rate": match_rate,
#     })
#     print(f"  {feat:20s} vs {ref:15s} : {match_rate:5.1f}%  (n={n:,})")

# # --- Plot ---
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle(
#     "Correlation between computed features and home/work cells",
#     fontsize=13
# )

# colors = {"cellule_home": "steelblue", "cellule_work": "seagreen"}

# titles = {
#     "cellule_home": "vs home cell",
#     "cellule_work": "vs work cell"
# }

# for ax, ref in zip(axes, ["cellule_home", "cellule_work"]):
#     subset = [r for r in results if r["reference"] == ref]
#     if not subset:
#         continue

#     feats = [r["feature"] for r in subset]
#     rates = [r["match_rate"] for r in subset]
#     ns    = [r["n"] for r in subset]

#     bars = ax.barh(
#         feats,
#         rates,
#         color=colors[ref],
#         edgecolor="white",
#         height=0.5
#     )

#     ax.set_xlim(0, 100)
#     ax.set_xlabel("Matching rate (%)")
#     ax.set_title(titles[ref])

#     ax.axvline(
#         50,
#         color="gray",
#         linestyle="--",
#         alpha=0.5,
#         label="50%"
#     )

#     # annotations
#     for bar, rate, n in zip(bars, rates, ns):
#         ax.text(
#             rate + 1,
#             bar.get_y() + bar.get_height() / 2,
#             f"{rate:.1f}%  (n={n:,})",
#             va="center",
#             fontsize=10
#         )

# plt.tight_layout()
# plt.savefig("results/correlation_features.png", dpi=150)
# plt.close()

# print("\n✓ results/correlation_features.png")