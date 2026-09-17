import numpy as np
from scipy.stats import entropy
from collections import Counter
from sklearn.ensemble import IsolationForest
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

from utils import get_pickle_day


FEATURE_NAMES = [
    "events",
    "median_dt",
    "mean_dt",
    "std_dt",
    "cv",
    "ratio_dt_1s",
    "ratio_dt_5s",
    "nb_0",
    "zero_ratio",
    "entropy_dt",
    "unique_dt_ratio",
    "active_hours",
]


def features(timestamps):

    ts = np.sort(timestamps)
    dt = np.diff(ts)

    if len(dt) == 0:
        return None

    counts = Counter(dt)
    p = np.array(list(counts.values()), dtype=float)
    p /= p.sum()

    return {
        "events": len(ts),
        "median_dt": np.median(dt),
        "mean_dt": np.mean(dt),
        "std_dt": np.std(dt),
        "cv": np.std(dt) / np.mean(dt) if np.mean(dt) != 0 else 0,

        "ratio_dt_1s": np.mean(dt <= 1),
        "ratio_dt_5s": np.mean(dt <= 5),

        "nb_0": np.sum(dt == 0),
        "zero_ratio": np.mean(dt == 0),

        "entropy_dt": entropy(p),

        "unique_dt_ratio": len(set(dt)) / len(dt),

        "active_hours": len(np.unique(ts // 3600)),
    }


def explainable_score(f):

    score = 0
    reasons = []

    if f["events"] > 10000:
        score += 2
        reasons.append("Très grand volume")

    if f["median_dt"] <= 1:
        score += 2
        reasons.append("Médiane dt ≤ 1s")

    if f["ratio_dt_1s"] > 0.5:
        score += 2
        reasons.append(">50% dt ≤ 1s")

    if f["zero_ratio"] > 0.1:
        score += 2
        reasons.append("Beaucoup de timestamps identiques")

    if f["active_hours"] > 20:
        score += 1
        reasons.append("Activité quasi 24h/24")

    if f["entropy_dt"] < 1.5:
        score += 1
        reasons.append("Faible entropie des intervalles")

    return score, reasons


# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

day = 12

event, _ = get_pickle_day(day)

feature_dicts = []
users = []

for user in event:

    _, times = event[user]

    f = features(times)

    if f is None:
        continue

    feature_dicts.append(f)
    users.append(user)


# ============================================================
# 2. CONSTRUCTION DE LA MATRICE X
# ============================================================

X = np.array([
    [f[name] for name in FEATURE_NAMES]
    for f in feature_dicts
])

print(f"{len(X):,} users")


# ============================================================
# 3. ENTRAÎNEMENT ISOLATION FOREST
# ============================================================

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)

model.fit(X)


# ============================================================
# 4. CALCUL DU ROBOT SCORE
# ============================================================

ml_scores = model.decision_function(X)

# Score IsolationForest :
# score faible  -> plus anormal
# score élevé   -> plus normal


# ============================================================
# 5. SCORE EXPLICABLE
# ============================================================

results = []

for i, user in enumerate(users):

    f = feature_dicts[i]

    rule_score, reasons = explainable_score(f)

    results.append({
        "user": user,
        "ml_score": ml_scores[i],
        "rule_score": rule_score,
        "reasons": reasons
    })


# Plus petit score = plus anormal
results.sort(key=lambda x: x["ml_score"])


# ============================================================
# 6. AFFICHER LES 20 PLUS ANORMAUX
# ============================================================

for r in results[:20]:

    print("\n===================")
    print("USER:", r["user"])
    print("ML score:", r["ml_score"])
    print("Rule score:", r["rule_score"])

    print("Reasons:")

    for reason in r["reasons"]:
        print("-", reason)


# ============================================================
# 7. GRAPHE DU ROBOT SCORE
# ============================================================


df = pd.DataFrame({
    "robot_score": ml_scores
})

fig = px.histogram(
    df,
    x="robot_score",
    nbins=100,
    title=f"Distribution of M2M scores — Day {day}",
    opacity=0.85
)

fig.update_layout(
    xaxis_title="M2M score",
    yaxis_title="Number of users",
    template="plotly_white"
)

fig.show()


# ============================================================
# 8. SAUVEGARDE DES USERS LES PLUS ANORMAUX
# ============================================================

bots = results

bot_ids = [b["user"] for b in bots]

with open("results/bot/detected_bots_day_12.txt", "w") as f:

    for bot_id in bot_ids:
        f.write(f"{bot_id}\n")

print("\nBot list saved to results/bot/detected_bots_day_12.txt")


# days = range(12, 27)   # jours 12 à 26 inclus

# feature_dicts = []
# users = []
# day_ids = []

# for day in days:

#     print(f"Loading day {day}")

#     event, _ = get_pickle_day(day)

#     for user in event:

#         _, times = event[user]

#         f = features(times)

#         if f is None:
#             continue

#         feature_dicts.append(f)
#         users.append(user)
#         day_ids.append(day)

# model, X = train_isolation_forest(feature_dicts)

# ml_scores = model.decision_function(X)
# pred = model.predict(X)

# results = []

# for i in range(len(users)):

#     f = feature_dicts[i]

#     rule_score, reasons = explainable_score(f)

#     results.append({
#         "day": day_ids[i],
#         "user": users[i],
#         "ml_score": ml_scores[i],
#         "prediction": pred[i],   # -1 = anomalie, 1 = normal
#         "rule_score": rule_score,
#         "reasons": reasons,
#     })

# ml_scores_dict = {
#     (r["day"], r["user"]): r["ml_score"]
#     for r in results
# }

# days = range(12, 27)

# for day in days:

#     path = PATH_SOURCE / f"features_day_{day}.npz"

#     feats = np.load(path, allow_pickle=True)["data"].item()

#     for user_id, feat_user in feats.items():

#         feat_user["ml_score"] = ml_scores_dict.get((day, user_id), None)

#     np.savez_compressed(path, data=feats)

#     print(f"Day {day} saved.")

# dt = np.diff(np.sort(times))

# print(dt.min(), dt.max())
# print(np.median(dt))
# print(np.mean(dt))


# print_features(features(times))

# # score = regularity_score(times)

# # print(f"Score : {score}")
# # if score < 0.1:
# #     print("Très probablement un robot")

# import matplotlib.pyplot as plt

# plt.hist(np.diff(np.sort(times)), bins=100, log=True)

# plt.show()