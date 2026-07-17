"""
chain.py + main pipeline
------------------------
Assemble les matchs inter-jours en chaînes d'identifiants sur N jours.
Une chaîne = un utilisateur reconstitué sur la période.
"""

from collections import defaultdict
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Assemblage des chaînes
# ---------------------------------------------------------------------------

def build_chains(daily_matches: list[list[dict]],
                 n_days: int,
                 max_gap: int = 1) -> list[dict]:

    # Construire les index forward[jour] : id_a → (id_b, score)
    forward = []
    for matches in daily_matches:
        fwd = {}
        for m in matches:
            fwd[m["id_a"]] = (m["id_b"], m["score"])
        forward.append(fwd)

    # Un début de chaîne = id présent dans forward[jour]
    # mais PAS dans les id_b de forward[jour-1]
    # → on cherche les "têtes" jour par jour
    chains = []
    chain_id = 0
    visited: set = set()  # (jour, id) déjà intégrés dans une chaîne

    for start_day, fwd in enumerate(forward):
        # IDs qui sont des id_b au jour précédent → pas des débuts
        if start_day > 0:
            already_matched = set(
                id_b for id_b, _ in forward[start_day - 1].values()
            )
        else:
            already_matched = set()

        for start_id in fwd.keys():
            if start_id in already_matched:
                continue  # milieu de chaîne, pas un début
            if (start_day, start_id) in visited:
                continue

            # Construire la chaîne depuis ce point de départ
            chain_ids = {start_day: start_id}
            chain_scores = []
            current_id = start_id
            visited.add((start_day, start_id))

            day = start_day
            gap = 0

            while day < len(forward):
                fwd_day = forward[day]
                if current_id in fwd_day:
                    next_id, score = fwd_day[current_id]
                    chain_ids[day + 1] = next_id
                    chain_scores.append(score)
                    visited.add((day + 1, next_id))
                    current_id = next_id
                    gap = 0
                else:
                    gap += 1
                    if gap > max_gap:
                        break
                day += 1

            if len(chain_ids) >= 2:
                chains.append({
                    "chain_id":   chain_id,
                    "ids":        chain_ids,
                    "length":     len(chain_ids),
                    "score_mean": round(float(np.mean(chain_scores)), 4) if chain_scores else 0.0,
                    "score_min":  round(float(np.min(chain_scores)), 4) if chain_scores else 0.0,
                    "scores":     chain_scores,
                })
                chain_id += 1

    return chains


def chain_stats(chains: list[dict], n_days: int) -> dict:
    """Statistiques sur les chaînes reconstituées."""
    if not chains:
        return {}

    lengths = [c["length"] for c in chains]
    scores  = [c["score_mean"] for c in chains]

    complete = sum(1 for c in chains if c["length"] == n_days)

    return {
        "n_chains":          len(chains),
        "n_complete":        complete,
        "pct_complete":      round(complete / len(chains) * 100, 1),
        "length_mean":       round(float(np.mean(lengths)), 2),
        "length_median":     float(np.median(lengths)),
        "score_mean":        round(float(np.mean(scores)), 4),
        "score_min":         round(float(np.min(scores)), 4),
    }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(days, get_day_fn, dist_idx,
                 theta=0.65, max_gap=1, weights=None,
                 verbose=True) -> tuple[list[dict], list[dict]]:
    """
    Pipeline complet sur N jours.

    Paramètres :
        days        : liste ordonnée des indices de jours [0, 1, 2, ...]
        get_day_fn  : votre fonction get_pickle_day
        dist_matrix : DataFrame des distances inter-antennes
        theta       : seuil de matching (↑ = plus précis, ↓ = plus de rappel)
        max_gap     : jours manquants tolérés dans une chaîne

    Retourne :
        (chains, all_matches)
        chains      : liste des chaînes reconstituées
        all_matches : liste des matchs inter-jours (pour audit)
    """
    from features  import compute_all_features
    from matching  import match_days, build_neighbors
    
    # Construire les voisinages une fois
    if verbose:
        print("=== Construction du voisinage géographique ===")
    neighbors = build_neighbors(dist_idx, radius=2000)

    # Calcul des features jour par jour
    if verbose:
        print("=== Calcul des features ===")

    daily_features = {}
    for day in days:
        if verbose:
            print(f"\n-- Jour {day} --")
        events, meta = get_day_fn(day)
        daily_features[day] = compute_all_features(events, meta, dist_idx, verbose)
        if verbose:
            print(f"   {len(daily_features[day]):,} utilisateurs")

    # Matching inter-jours
    if verbose:
        print("\n=== Matching inter-jours ===")

    all_matches = []
    for i in range(len(days) - 1):
        day_a, day_b = days[i], days[i + 1]
        if verbose:
            print(f"\n-- Jour {day_a} → Jour {day_b} --")
        matches = match_days(
            daily_features[day_a],
            daily_features[day_b],
            dist_idx,
            neighbors,      # ← AJOUTER
            theta=theta,
            weights=weights,
            verbose=verbose,
        )
        all_matches.append(matches)

    # Assemblage des chaînes
    if verbose:
        print("\n=== Assemblage des chaînes ===")

    chains = build_chains(all_matches, n_days=len(days), max_gap=max_gap)

    if verbose:
        stats = chain_stats(chains, len(days))
        print(f"  Chaînes totales    : {stats.get('n_chains', 0):,}")
        print(f"  Chaînes complètes  : {stats.get('n_complete', 0):,}  ({stats.get('pct_complete', 0)}%)")
        print(f"  Longueur moyenne   : {stats.get('length_mean', 0):.2f} jours")
        print(f"  Score moyen        : {stats.get('score_mean', 0):.4f}")

    return chains, all_matches


# ---------------------------------------------------------------------------
# Exemple d'utilisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    import numpy as np

    # --- Vos chemins ---
    PATH_SOURCE   = Path("data/raw")
    PATH_DATA = Path("data")

    dist_matrix_raw = np.load(PATH_DATA / "dist_matrix.npy", allow_pickle=True)
    cell_ids = np.load(PATH_DATA / "cell_ids.npy", allow_pickle=True)
    
    from features import DistanceIndex
    dist_idx = DistanceIndex(dist_matrix_raw, cell_ids)

    def get_pickle_day(day):
        day += 12
        events = np.load(PATH_SOURCE / f"events_day_{day}.npz", allow_pickle=True)["data"].item()
        meta   = np.load(PATH_SOURCE / f"meta_day_{day}.npy",   allow_pickle=True).item()
        return events, meta

    # --- Lancer sur les 14 jours ---
    chains, all_matches = run_pipeline(
        days         = list(range(15)),
        get_day_fn   = get_pickle_day,
        dist_idx  = dist_idx,
        theta        = 0.65,   # à calibrer
        max_gap      = 1,
        verbose      = True,
    )

    # --- Sauvegarder les chaînes ---
    import json
    import pandas as pd

    # --- 1. chains.json : les chaînes complètes avec tous les détails ---
    # Format : liste de dicts
    # {chain_id, ids: {jour -> id}, length, score_mean, score_min, scores}
    with open("results/all_days_chains.json", "w") as f:
        json.dump(chains, f, indent=2)

    # --- 2. chains.csv : version à plat, plus facile à explorer ---
    # Une ligne par (chaîne × jour), joinable avec tes meta
    rows = []
    for c in chains:
        for day, uid in c["ids"].items():
            rows.append({
                "chain_id":   c["chain_id"],
                "day":        int(day),
                "user_id":    uid,
                "length":     c["length"],
                "score_mean": c["score_mean"],
                "score_min":  c["score_min"],
            })

    df = pd.DataFrame(rows)
    df.to_csv("results/all_days_chains.csv", index=False)

    # --- 3. matches.json : les matchs bruts inter-jours (pour audit) ---
    # Utile pour déboguer ou recalibrer theta
    with open("results/matches.json", "w") as f:
        json.dump(all_matches, f)

    print(f"\nFichiers sauvegardés dans results/")
    print(f"  chains.json  : {len(chains):,} chaînes")
    print(f"  chains.csv   : {len(df):,} lignes")
