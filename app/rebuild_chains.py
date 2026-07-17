import json
import numpy as np
from pathlib import Path
from chain import build_chains, chain_stats

PATH_RESULTS = Path("results")
N_DAYS = 15

# Relire les matches depuis le disque
match_files = sorted(PATH_RESULTS.glob("matches_day*_to_*.json"))
daily_matches = []
for f in match_files:
    with open(f) as fh:
        daily_matches.append(json.load(fh))

print(f"{len(daily_matches)} transitions chargées")

# Reconstruire les chaînes
chains = build_chains(daily_matches, n_days=N_DAYS, max_gap=1)

# Stats
stats = chain_stats(chains, N_DAYS)
print(f"Chaînes totales    : {stats['n_chains']:,}")
print(f"Chaînes complètes  : {stats['n_complete']:,} ({stats['pct_complete']}%)")
print(f"Longueur moyenne   : {stats['length_mean']:.2f} jours")
print(f"Score moyen        : {stats['score_mean']:.4f}")

# Sauvegarder
with open(PATH_RESULTS / "chains.json", "w") as f:
    json.dump(chains, f, indent=2)
print("chains.json mis à jour")