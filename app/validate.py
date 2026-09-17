"""
validate.py
-----------
Validation de la qualité des chaînes reconstituées.
Produit des analyses de cohérence et des visualisations.

Lancer : python validate.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PATH_SOURCE  = Path("data/raw")
PATH_DATA    = Path("data")
PATH_RESULTS = Path("results")

N_DAYS = 15   # nombre de jours dans ton pipeline

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_all():
    print("Chargement des chaînes...")
    with open(PATH_RESULTS / "chains.json") as f:
        chains = json.load(f)

    print("Chargement des features journalières...")
    # On recharge les meta pour valider la cohérence age/sexe/comportement
    all_meta = {}
    for day in range(N_DAYS):
        real_day = day + 12
        meta = np.load(PATH_SOURCE / f"meta_day_{real_day}.npy", allow_pickle=True).item()
        all_meta[day] = meta

    cell_ids = np.load(PATH_DATA / "cell_ids.npy", allow_pickle=True)
    coords   = np.load(PATH_DATA / "coords.npy")
    coord_map = {code: coords[i] for i, code in enumerate(cell_ids)}

    print(f"  {len(chains):,} chaînes chargées\n")
    return chains, all_meta, coord_map


# ---------------------------------------------------------------------------
# 1. Distribution des longueurs et scores
# ---------------------------------------------------------------------------

def plot_distributions(chains):
    lengths = [c["length"] for c in chains]
    scores  = [c["score_mean"] for c in chains]
    scores_min = [c["score_min"] for c in chains]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Distribution of reconstituted chains", fontsize=13)

    # Longueurs
    ax = axes[0]
    cnt = Counter(lengths)
    ax.bar(cnt.keys(), cnt.values(), color="steelblue", edgecolor="white")
    ax.set_xlabel("chain length (days)")
    ax.set_ylabel("Number of chains")
    ax.set_title("Length distributions")
    ax.axvline(np.mean(lengths), color="red", linestyle="--",
               label=f"Average : {np.mean(lengths):.1f}")
    ax.legend()

    # Score moyen
    ax = axes[1]
    ax.hist(scores, bins=50, color="seagreen", edgecolor="white")
    ax.set_xlabel("Average score")
    ax.set_ylabel("Number of chains")
    ax.set_title("Average score distributions")
    ax.axvline(np.mean(scores), color="red", linestyle="--",
               label=f"Average : {np.mean(scores):.3f}")
    ax.legend()

    # Score minimum
    ax = axes[2]
    ax.hist(scores_min, bins=50, color="coral", edgecolor="white")
    ax.set_xlabel("Minimum score")
    ax.set_ylabel("Number of chains")
    ax.set_title("Minimum score by chain\n(weakest link)")
    ax.axvline(0.65, color="black", linestyle="--", label="θ = 0.65")
    ax.legend()

    plt.tight_layout()
    plt.savefig(PATH_RESULTS / "validation_distributions.png", dpi=150)
    plt.close()
    print("✓ validation_distributions.png")


# ---------------------------------------------------------------------------
# 2. Cohérence des champs statiques (age, sexe, comportement)
# ---------------------------------------------------------------------------

def check_static_coherence(chains, all_meta):
    """
    Pour chaque chaîne, vérifie que age/sexe/comportement sont cohérents
    entre tous les jours où l'utilisateur est présent.
    """
    results = {
        "age_coherent":         0,
        "age_incoherent":       0,
        "age_missing":          0,
        "sexe_coherent":        0,
        "sexe_incoherent":      0,
        "sexe_missing":         0,
        "comportement_coherent":   0,
        "comportement_incoherent": 0,
        "comportement_missing":    0,
    }

    for c in chains:
        ids_by_day = c["ids"]  # {str(day) -> id}

        ages, sexes, comps = [], [], []
        for day_str, uid in ids_by_day.items():
            day = int(day_str)
            m = all_meta[day].get(uid, {})
            if m.get("age"):
                ages.append(m["age"])
            if m.get("sexe"):
                sexes.append(m["sexe"])
            if m.get("comportement"):
                comps.append(m["comportement"])

        # Age : écart max toléré = 0 (champ statique)
        if len(ages) >= 2:
            if len(set(ages)) == 1:
                results["age_coherent"] += 1
            else:
                results["age_incoherent"] += 1
        else:
            results["age_missing"] += 1

        # Sexe
        if len(sexes) >= 2:
            if len(set(sexes)) == 1:
                results["sexe_coherent"] += 1
            else:
                results["sexe_incoherent"] += 1
        else:
            results["sexe_missing"] += 1

        # Comportement
        if len(comps) >= 2:
            if len(set(comps)) == 1:
                results["comportement_coherent"] += 1
            else:
                results["comportement_incoherent"] += 1
        else:
            results["comportement_missing"] += 1

    total = len(chains)
    print("=== Cohérence des champs statiques ===")
    for field in ["age", "sexe", "comportement"]:
        coh = results[f"{field}_coherent"]
        inc = results[f"{field}_incoherent"]
        mis = results[f"{field}_missing"]
        total_known = coh + inc
        pct = coh / total_known * 100 if total_known > 0 else 0
        print(f"  {field:15s} : {coh:,} cohérents / {total_known:,} renseignés "
              f"({pct:.1f}%)  |  {mis:,} sans donnée")

    return results


# ---------------------------------------------------------------------------
# 3. Stabilité géographique (antenne nocturne)
# ---------------------------------------------------------------------------

def check_geo_stability(chains, all_meta):
    """
    Vérifie que l'antenne home est stable sur la chaîne.
    Un utilisateur sédentaire devrait avoir la même cellule_home tous les jours.
    """
    same_home = 0
    diff_home = 0
    no_home   = 0

    for c in chains:
        homes = []
        for day_str, uid in c["ids"].items():
            day = int(day_str)
            m = all_meta[day].get(uid, {})
            h = m.get("cellule_home")
            if h:
                homes.append(h)

        if len(homes) >= 2:
            if len(set(homes)) == 1:
                same_home += 1
            else:
                diff_home += 1
        else:
            no_home += 1

    total_known = same_home + diff_home
    pct = same_home / total_known * 100 if total_known > 0 else 0
    print(f"\n=== Stabilité géographique (cellule_home) ===")
    print(f"  Stable    : {same_home:,} / {total_known:,} ({pct:.1f}%)")
    print(f"  Instable  : {diff_home:,} / {total_known:,} ({100-pct:.1f}%)")
    print(f"  Sans home : {no_home:,}")

    return same_home, diff_home, no_home


# ---------------------------------------------------------------------------
# 4. Visualisation de trajectoires individuelles sur carte
# ---------------------------------------------------------------------------

def plot_sample_trajectories(chains, all_meta, coord_map, n_samples=6):
    """
    Affiche les trajectoires géographiques de N chaînes aléatoires.
    Chaque point = centroïde des antennes visitées ce jour-là.
    """
    # Sélectionner des chaînes longues pour un rendu intéressant
    long_chains = [c for c in chains if c["length"] >= 8]
    if not long_chains:
        long_chains = chains
    sample = np.random.choice(long_chains,
                              size=min(n_samples, len(long_chains)),
                              replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Weakest link: Reconstructed user trajectories", fontsize=13)
    axes = axes.flatten()

    colors = cm.viridis(np.linspace(0, 1, N_DAYS))

    for idx, chain in enumerate(sample):
        ax = axes[idx]
        ax.set_title(f"Chain {chain['chain_id']} — {chain['length']} jours "
                     f"(average score {chain['score_mean']:.2f})")

        lats, lons, days_present = [], [], []

        for day_str, uid in sorted(chain["ids"].items(), key=lambda x: int(x[0])):
            day = int(day_str)
            m = all_meta[day].get(uid, {})

            # Utiliser cellule_home ou cellule_work comme proxy de position
            ant = m.get("cellule_home") or m.get("cellule_work")
            if ant and ant in coord_map:
                lat, lon = coord_map[ant]
                lats.append(lat)
                lons.append(lon)
                days_present.append(day)

        if len(lats) >= 2:
            # Tracer la trajectoire
            ax.plot(lons, lats, "k-", alpha=0.3, linewidth=1)
            for i, (lat, lon, day) in enumerate(zip(lats, lons, days_present)):
                ax.scatter(lon, lat, color=colors[day], s=80, zorder=5)
                ax.annotate(f"J{day}", (lon, lat),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=colors[day])
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        else:
            ax.text(0.5, 0.5, "Pas assez de\ncoordonnées",
                    ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(PATH_RESULTS / "validation_trajectoires.png", dpi=150)
    plt.close()
    print("✓ validation_trajectoires.png")


# ---------------------------------------------------------------------------
# 5. Analyse des taux de matching par jour
# ---------------------------------------------------------------------------

def plot_matching_rates():
    """
    Relit les fichiers matches_dayX_to_Y.json et affiche le taux de matching
    jour par jour — utile pour détecter des jours atypiques (week-end, etc.)
    """
    match_files = sorted(PATH_RESULTS.glob("matches.json"))
    if not match_files:
        print("  [SKIP] Pas de fichiers matches_day*.json trouvés")
        return

    days, rates, n_matches = [], [], []
    for f in match_files:
        # Extraire les numéros de jours depuis le nom de fichier
        parts = f.stem.split("_")  # matches_day12_to_13
        day_a = int(parts[1].replace("day", ""))
        with open(f) as fh:
            matches = json.load(fh)
        days.append(day_a)
        n_matches.append(len(matches))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(days, n_matches, color="steelblue", edgecolor="white")
    ax.set_xlabel("Day D (transition D → D+1)")
    ax.set_ylabel("Match number")
    ax.set_title("Number of matches per inter-day transition")
    ax.set_xticks(days)

    plt.tight_layout()
    plt.savefig(PATH_RESULTS / "validation_matching_par_jour.png", dpi=150)
    plt.close()
    print("✓ validation_matching_par_jour.png")


# ---------------------------------------------------------------------------
# 6. Résumé textuel
# ---------------------------------------------------------------------------

def print_summary(chains):
    lengths = [c["length"] for c in chains]
    scores  = [c["score_mean"] for c in chains]

    print("\n" + "="*50)
    print("RÉSUMÉ DE VALIDATION")
    print("="*50)
    print(f"  Chaînes totales       : {len(chains):,}")
    print(f"  Chaînes complètes     : {sum(1 for l in lengths if l == N_DAYS):,} "
          f"({sum(1 for l in lengths if l == N_DAYS)/len(chains)*100:.1f}%)")
    print(f"  Chaînes ≥ 7 jours     : {sum(1 for l in lengths if l >= 7):,} "
          f"({sum(1 for l in lengths if l >= 7)/len(chains)*100:.1f}%)")
    print(f"  Chaînes ≥ 10 jours    : {sum(1 for l in lengths if l >= 10):,} "
          f"({sum(1 for l in lengths if l >= 10)/len(chains)*100:.1f}%)")
    print(f"  Longueur moyenne      : {np.mean(lengths):.2f} jours")
    print(f"  Score moyen           : {np.mean(scores):.4f}")
    print(f"  Score < 0.70          : {sum(1 for s in scores if s < 0.70):,} chaînes")
    print(f"  Score ≥ 0.80          : {sum(1 for s in scores if s >= 0.80):,} chaînes")
    print("="*50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chains, all_meta, coord_map = load_all()

    print_summary(chains)
    plot_distributions(chains)
    check_static_coherence(chains, all_meta)
    check_geo_stability(chains, all_meta)
    plot_sample_trajectories(chains, all_meta, coord_map)
    plot_matching_rates()

    print(f"\nTous les graphiques sauvegardés dans {PATH_RESULTS}/")