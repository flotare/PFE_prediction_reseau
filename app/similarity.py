import numpy as np
import math


from pathlib import Path
import pickle


PATH_DATA = Path("data")


with open(PATH_DATA / "trans_proba.pkl", "rb") as f:
    trans_proba = pickle.load(f)

# DEFAULT_WEIGHTS = {
#     "night_cell": 0.15,
#     "day_cell": 0.10,
#     "most_common_cell": 0.2,
#     "medoid_cell": 0.15,
#     "gyration_radius": 0.05,
#     "dist_night_day": 0.15,
#     "nb_distinct_cell": 0.05,
#     "hour_signature": 0.15,
# }

DEFAULT_WEIGHTS = {
    "night_cell": 0.15,
    "day_cell": 0.15,
    "medoid_cell": 0.2,
    "MFCC": 0.3,
    "ml_score": 0.2
}

SIM_FUNCTIONS = {
    "night_cell": lambda a, b: _sim_cell(a["night_cell"], b["night_cell"]),
    "day_cell": lambda a, b: _sim_cell(a["day_cell"], b["day_cell"]),
    "medoid_cell": lambda a, b: _sim_cell(a["medoid_cell"], b["medoid_cell"]),
    "MFCC": lambda a, b: _sim_mfcc(a["MFCC"], b["MFCC"]),
    "ml_score": lambda a, b: _sim_ml_score(a["ml_score"], b["ml_score"]),
}


def _sim_cell(a, b) -> float:

    if a is None or b is None:
        return 0.5

    pab = trans_proba.get(a, {}).get(b, 0)
    pba = trans_proba.get(b, {}).get(a, 0)

    return (pab + pba) / 2

def _sim_signature(sig_a, sig_b) -> float:
    scores = [_sim_cell(a, b) for a, b in zip(sig_a, sig_b)]
    return float(np.mean(scores))


def _sim_night_day(fa, fb) -> float:
    return 0.5 * _sim_cell(
        fa.get("night_cell"), fb.get("night_cell")
    ) + 0.5 * _sim_cell(fa.get("day_cell"), fb.get("day_cell"))
    

def _sim_gyration(r1, r2):
    # cas spéciaux
    if r1 == 0 and r2 == 0:
        return 1.0
    if r1 == 0 or r2 == 0:
        return 0.0

    # cas général
    return 1 / (1 + abs(math.log(r1 / r2)))

def _sim_mfcc(a, b):

    if a is None or b is None:
        return 0.5

    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

    return (sim + 1) / 2

def _sim_ml_score(s1, s2, scale=0.05):
    if s1 is None or s2 is None:
        return 0.5

    return np.exp(-abs(s1 - s2) / scale)


def similarity_score(fa, fb, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    scores = {
        name: func(fa, fb)
        for name, func in SIM_FUNCTIONS.items()
    }
    total = sum(
        weights[name] * scores[name]
        for name in weights
    )
    scores["score"] = round(total, 4)
    return {
        k: round(v,4)
        for k,v in scores.items()
    }


def explain_score(sc):

    lines = [f"Global score : {sc['score']:.3f}"]

    for feature, weight in DEFAULT_WEIGHTS.items():
        lines.append(
            f"{feature:15s}: {sc[feature]:.3f} ({weight*100:.0f}%)"
        )

    return "\n".join(lines)
   
from pathlib import Path
import numpy as np

OUTPUT_PATH = PATH_SOURCE / "model_features"
OUTPUT_PATH.mkdir(exist_ok=True)

KEEP_FEATURES = {
    "id",
    "night_cell",
    "day_cell",
    "medoid_cell",
    "MFCC",
    "ml_score",
}


for day in range(12, 27):

    print(f"Processing day {day}")

    feats = get_pickle_features(day)

    filtered_feats = {}

    for user_id, feat in feats.items():

        hour_signature = feat.get("hour_signature")

        # On retire les utilisateurs ayant au moins un None
        if hour_signature is None or any(cell is None for cell in hour_signature):
            continue

        filtered_feats[user_id] = {
            key: feat[key]
            for key in KEEP_FEATURES
            if key in feat
        }

    np.savez_compressed(
        OUTPUT_PATH / f"features_day_{day}.npz",
        data=filtered_feats,
    )

    print(f"  -> kept {len(filtered_feats):,} users")