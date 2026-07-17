"""
matching.py — version optimisée
---------------------------------
Utilise DistanceIndex + parallélisme par cluster.
"""

import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from features import DistanceIndex
from similarity import similarity_score, DEFAULT_WEIGHTS

# ---------------------------------------------------------------------------
# Voisinage géographique
# ---------------------------------------------------------------------------


def build_neighbors(dist_idx: DistanceIndex, radius: float = 2000) -> dict:
    """
    Pour chaque antenne, liste des antennes dans un rayon (mètres).
    À construire une seule fois au démarrage.
    """
    return {code: dist_idx.neighbors(code, radius) for code in dist_idx.cell_ids}


# ---------------------------------------------------------------------------
# Pré-filtre + scoring par cluster
# ---------------------------------------------------------------------------


def _score_cluster(args) -> list:
    ids_a, ids_b, features_a, features_b, dist_idx, weights, theta = args
    candidates = []
    for id_a in ids_a:
        fa = features_a[id_a]
        for id_b in ids_b:
            sc = similarity_score(fa, features_b[id_b], weights)
            if sc["score"] >= theta:
                candidates.append((id_a, id_b, sc))
    return candidates


def candidate_pairs_fast(
    features_a,
    features_b,
    dist_idx: DistanceIndex,
    neighbors,
    theta=0.65,
    weights=None,
    n_workers=None,
    verbose=True,
) -> list:
    w = weights or DEFAULT_WEIGHTS
    n_workers = n_workers or max(1, os.cpu_count() - 1)

    # Grouper B par antenne nocturne
    groups_b = {}
    for uid, fb in features_b.items():
        key = fb.get("antenne_nuit") or fb.get("antenne_modale") or "__unknown__"
        groups_b.setdefault(key, []).append(uid)

    # Construire les clusters A → candidats B (via voisinage)
    cluster_tasks = {}
    for uid_a, fa in features_a.items():
        ant_a = fa.get("antenne_nuit") or fa.get("antenne_modale") or "__unknown__"
        voisines = neighbors.get(ant_a, [ant_a])
        ids_b_candidates = list(
            {uid_b for ant_v in voisines for uid_b in groups_b.get(ant_v, [])}
        )
        if ids_b_candidates:
            if ant_a not in cluster_tasks:
                cluster_tasks[ant_a] = ([], ids_b_candidates)
            cluster_tasks[ant_a][0].append(uid_a)

    job_args = [
        (ids_a, ids_b, features_a, features_b, dist_idx, w, theta)
        for ids_a, ids_b in cluster_tasks.values()
        if ids_a and ids_b
    ]

    total_pairs = sum(len(a) * len(b) for a, b, *_ in job_args)
    if verbose:
        print(
            f"  {len(job_args)} clusters | {n_workers} workers | ~{total_pairs:,} paires"
        )

    candidates = []
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_score_cluster, args): i for i, args in enumerate(job_args)
        }
        for future in as_completed(futures):
            candidates.extend(future.result())
            done += 1
            if verbose and done % 50 == 0:
                print(f"  Clusters : {done}/{len(job_args)}", end="\r")

    if verbose:
        print(
            f"  Clusters : {len(job_args)}/{len(job_args)} | "
            f"candidats ≥ θ={theta} : {len(candidates):,}"
        )
    return candidates


# ---------------------------------------------------------------------------
# Matching greedy 1-to-1
# ---------------------------------------------------------------------------


def greedy_matching(candidates, theta=0.65) -> list:
    if not candidates:
        return []
    candidates_sorted = sorted(candidates, key=lambda x: x[2]["score"], reverse=True)
    used_a, used_b = set(), set()
    results = []
    for id_a, id_b, sc in candidates_sorted:
        if id_a not in used_a and id_b not in used_b:
            used_a.add(id_a)
            used_b.add(id_b)
            results.append({"id_a": id_a, "id_b": id_b, **sc})
    return results


def match_days(
    features_a,
    features_b,
    dist_idx: DistanceIndex,
    neighbors,
    theta=0.65,
    weights=None,
    n_workers=None,
    verbose=True,
) -> list:
    if verbose:
        print(f"Matching {len(features_a):,} × {len(features_b):,} (θ={theta})")
    candidates = candidate_pairs_fast(
        features_a,
        features_b,
        dist_idx,
        neighbors,
        theta=theta,
        weights=weights,
        n_workers=n_workers,
        verbose=verbose,
    )
    matches = greedy_matching(candidates, theta)
    if verbose:
        rate = len(matches) / max(len(features_a), 1) * 100
        print(f"  Matchs : {len(matches):,}  ({rate:.1f}%)")
    return matches


def matching_stats(matches) -> dict:
    if not matches:
        return {"n_matches": 0}
    scores = [m["score"] for m in matches]
    return {
        "n_matches": len(matches),
        "score_mean": round(float(np.mean(scores)), 4),
        "score_median": round(float(np.median(scores)), 4),
        "score_min": round(float(np.min(scores)), 4),
        "score_max": round(float(np.max(scores)), 4),
    }
