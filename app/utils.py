from geopy.distance import geodesic
from pathlib import Path
from collections import Counter

import numpy as np
import features
# import similarity

import pandas as pd

import graph

BASE_DIR = Path(__file__).resolve().parent.parent

PATH_DATA_FILE = BASE_DIR / "cd_142_dataset" / "raw_dataset"

PATH_LIST_FILE_NAME = [
    PATH_DATA_FILE / f"2014-03-{day:02d}_vektory.csv" for day in range(12, 27)
]

PATH_DATA_CELL_142 = PATH_DATA_FILE / "cells_142.csv"

PATH_SOURCE = BASE_DIR / "app" / "source"

dist_matrix = np.load(PATH_SOURCE / "dist_matrix.npy", allow_pickle=True)
cell_ids = np.load(PATH_SOURCE / "cell_ids.npy", allow_pickle=True)
coords = np.load(PATH_SOURCE / "coords.npy")

index_map = {cell_id: i for i, cell_id in enumerate(cell_ids)}

dist_idx = features.DistanceIndex(dist_matrix, cell_ids)

EMPTY_ROW_TEMPLATE = {
    "Day": None,
    "User": None,
    "Age": None,
    "Sex": None,
    "Big_nb": None,
    "Behaviour": None,
    "Home_cell": None,
    "Activity_cell": None,
    "Ant_nuit": None,
    "Ant_jour": None,
    "Gyration": None,
    "Dist_night_day": None,
    "Nb_records": None,
    "Nb_antennes": None,
}

def get_data_day(id):

    if id < 12 or id > 26:
        raise ValueError("ID must be between 12 and 26 inclusive.")

    data = []

    with open(PATH_LIST_FILE_NAME[id - 12], encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")

            # Vecteur Standard
            meta = {
                "id": to_int(parts[0]),
                "age": to_int(parts[1]),
                "sexe": parts[2],
                "Big_number": to_int(parts[3]),
                "comportement": parts[4],
                "cellule_home": parts[5],
                "cellule_work": parts[6],
                "nb_records": to_int(parts[7]),
            }

            # Vecteur records
            events_raw = parts[8:]

            events = []
            for i in range(0, len(events_raw), 2):
                if i + 1 < len(events_raw):
                    code = events_raw[i]
                    s = events_raw[i + 1]
                    events.append((code, int(s)))

            nb_moves = 0
            for i in range(1, len(events)):
                prev_cell = events[i - 1][0]
                curr_cell = events[i][0]

                if distance_between_cells(prev_cell, curr_cell) > 0:
                    nb_moves += 1

            meta["nb_moves"] = nb_moves

            data.append({"meta": meta, "events": events})

    return {user["meta"]["id"]: user for user in data}  # index by user_id

def get_pickle_day(day):
    events = np.load(PATH_SOURCE / f"events_day_{day}.npz", allow_pickle=True)[
        "data"
    ].item()
    meta = np.load(PATH_SOURCE / f"meta_day_{day}.npy", allow_pickle=True).item()
    return events, meta

def get_pickle_features(day):
    features = np.load(PATH_SOURCE / "model_features" / f"features_day_{day}.npz", allow_pickle=True)[
        "data"
    ].item()
    return features

def save_features():
    days = range(24, 27)

    for d in days:
        events, _ = get_pickle_day(d)

        features_day = {}
        
        for user_id in events:
            features_day[user_id] = features.compute_features(
                user_id,
                events,
                dist_idx
            )

        np.savez_compressed(
            PATH_SOURCE / f"features_day_{d}.npz",
            data=features_day
        )

        print(f"Jour {d} sauvegardé ({len(features_day)} utilisateurs)")
    
    
def convert_day(id):
    raw = get_data_day(id)

    events_data = {}
    meta_data = {}

    for user_id, user in raw.items():

        # META (on garde tel quel)
        meta_data[user_id] = user["meta"]

        events = user["events"]

        codes = np.fromiter((c for c, _ in events), dtype=object, count=len(events))
        times = np.fromiter((t for _, t in events), dtype=np.int32, count=len(events))

        events_data[user_id] = (codes, times)

    # Sauvegarde
    np.savez_compressed(PATH_SOURCE / f"events_day_{id}.npz", data=events_data)
    np.save(PATH_SOURCE / f"meta_day_{id}.npy", meta_data)
    


def build_all_days():
    for day in range(12, 27):  # 12 → 26 inclus
        print(f"Processing day {day}...")
        convert_day(day)

def to_int(x: str) -> int | None:
    x = x.strip().replace('"', "")
    return int(x) if x else None

def convert_lat_lon_distance_to_meter(
    point1: tuple[float, float], point2: tuple[float, float]
) -> float:
    return geodesic(point1, point2).meters

def build_trajectory(events):
    cells, times = events  # arrays

    times_out = []
    lats = []
    lons = []
    dist_raws = []
    dist_cums = []
    dist_origins = []
    cells_out = []

    prev_idx = None
    origin_idx = None
    cumulative_dist = 0.0

    for i in range(len(cells)):
        cell = cells[i]
        t = times[i]

        if cell not in index_map:
            continue

        idx = index_map[cell]
        lat, lon = coords[idx]

        if origin_idx is None:
            origin_idx = idx

        if prev_idx is None:
            dist_raw = 0.0
        else:
            dist_raw = dist_matrix[prev_idx, idx]

        cumulative_dist += dist_raw
        dist_origin = dist_matrix[origin_idx, idx]

        times_out.append(t)
        lats.append(lat)
        lons.append(lon)
        dist_raws.append(dist_raw)
        dist_cums.append(cumulative_dist)
        dist_origins.append(dist_origin)
        cells_out.append(cell)

        prev_idx = idx

    return {
        "time": np.array(times_out),
        "latitude": np.array(lats),
        "longitude": np.array(lons),
        "dist_raw": np.array(dist_raws),
        "dist_cumulative": np.array(dist_cums),
        "dist_origin": np.array(dist_origins),
        "cell": np.array(cells_out),
    }

def build_trajectory_medoid(events, dist_idx: features.DistanceIndex):
    cells, times = events

    times_out = []
    lats = []
    lons = []
    cells_out = []

    # ----------------------------------------------------------
    # filtrage trajectoire
    # ----------------------------------------------------------

    valid_cells = []
    valid_times = []

    for cell, t in zip(cells, times):

        if cell not in dist_idx._idx:
            continue

        idx = dist_idx._idx[cell]
        lat, lon = coords[idx]

        times_out.append(t)
        lats.append(lat)
        lons.append(lon)
        cells_out.append(cell)

        valid_cells.append(cell)
        valid_times.append(t)

    # ----------------------------------------------------------
    # cas vide
    # ----------------------------------------------------------

    if len(valid_cells) == 0:
        return {
            "time": np.array([]),
            "latitude": np.array([]),
            "longitude": np.array([]),
            "cell": np.array([]),
            "medoid_cell": None,
            "radius_of_gyration": None,
        }

    valid_cells = np.array(valid_cells)
    valid_times = np.array(valid_times)

    # ----------------------------------------------------------
    # réutilisation fonction features
    # ----------------------------------------------------------

    medoid_cell, rg = features._rayon_gyration_fast(valid_cells, valid_times, dist_idx)

    idx_medoid = dist_idx._idx[medoid_cell]
    lat_medoid, lon_medoid = coords[idx_medoid]

    # ----------------------------------------------------------
    # output
    # ----------------------------------------------------------

    return {
        "time": np.array(times_out),
        "latitude": np.array(lats),
        "longitude": np.array(lons),
        "cell": np.array(cells_out),
        "medoid_cell": medoid_cell,
        "radius_of_gyration": rg,
        "lat_medoid": lat_medoid,
        "lon_medoid": lon_medoid,
    }

def circle_coordinates(lat, lon, radius_m, n_points=100):
    """
    Génère un cercle géographique approximatif.
    """

    R = 6378137  # rayon Terre (m)

    angles = np.linspace(0, 2 * np.pi, n_points)

    circle_lats = []
    circle_lons = []

    lat_rad = np.radians(lat)

    for a in angles:

        dx = radius_m * np.cos(a)
        dy = radius_m * np.sin(a)

        dlat = dy / R
        dlon = dx / (R * np.cos(lat_rad))

        circle_lats.append(lat + np.degrees(dlat))
        circle_lons.append(lon + np.degrees(dlon))

    return circle_lats, circle_lons

def distance_between_cells(cell1, cell2):
    if cell1 not in index_map or cell2 not in index_map:
        return None

    return dist_matrix[index_map[cell1], index_map[cell2]]

def normalize(events, t_min=0, t_max=86400):
    codes, times = events

    order = np.argsort(times)
    codes = codes[order]
    times = times[order]

    keep = np.ones(len(codes), dtype=bool)
    keep[1:] = codes[1:] != codes[:-1]

    codes = codes[keep]
    times = times[keep]

    start_extra = times[0] > t_min
    end_extra = times[-1] < t_max

    new_size = len(times) + start_extra + end_extra

    new_codes = np.empty(new_size, dtype=object)
    new_times = np.empty(new_size, dtype=times.dtype)

    idx = 0

    if start_extra:
        new_codes[0] = codes[0]
        new_times[0] = t_min
        idx = 1

    new_codes[idx : idx + len(codes)] = codes
    new_times[idx : idx + len(times)] = times

    idx += len(codes)

    if end_extra:
        new_codes[idx] = codes[-1]
        new_times[idx] = t_max

    return new_codes, new_times

def compute_merge_on_timeline_distance(events1, events2):

    t_min = min(events1[1][0], events2[1][0])
    t_max = max(events1[1][-1], events2[1][-1])

    c1, t1 = normalize(events1, t_min=t_min, t_max=t_max)
    c2, t2 = normalize(events2, t_min=t_min, t_max=t_max)

    i = j = 0
    total = 0.0
    t = t_min

    while i < len(t1) - 1 or j < len(t2) - 1:

        next_t1 = t1[i + 1] if i + 1 < len(t1) else t_max
        next_t2 = t2[j + 1] if j + 1 < len(t2) else t_max

        t_next = next_t1 if next_t1 < next_t2 else next_t2

        dt = t_next - t

        code1 = c1[i]
        code2 = c2[j]

        total += distance_between_cells(code1, code2) * dt

        if t_next == next_t1:
            i += 1
        if t_next == next_t2:
            j += 1

        t = t_next

        if t >= t_max:
            break

    return total / (t_max - t_min)

def build_probability_transition(events1, events2):
    codes1, _ = events1
    codes2, _ = events2

    all_codes = sorted(set(codes1) | set(codes2))
    code_to_idx = {code: i for i, code in enumerate(all_codes)}
    n = len(all_codes)

    count_matrix1 = np.zeros((n, n))
    count_matrix2 = np.zeros((n, n))

    for i in range(len(codes1) - 1):
        a, b = codes1[i], codes1[i + 1]
        count_matrix1[code_to_idx[a], code_to_idx[b]] += 1

    for i in range(len(codes2) - 1):
        a, b = codes2[i], codes2[i + 1]
        count_matrix2[code_to_idx[a], code_to_idx[b]] += 1

    row_sums1 = count_matrix1.sum(axis=1, keepdims=True)
    row_sums1[row_sums1 == 0] = 1
    prob_matrix1 = count_matrix1 / row_sums1

    row_sums2 = count_matrix1.sum(axis=1, keepdims=True)
    row_sums2[row_sums2 == 0] = 1
    prob_matrix2 = count_matrix2 / row_sums2

    return prob_matrix1, prob_matrix2, all_codes

def compute_markov_like_distance(events1, events2):
    P1, P2, codes = build_probability_transition(events1, events2)
    n = len(codes)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = distance_between_cells(codes[i], codes[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    diff = np.abs(P1 - P2)

    weighted_sum = np.sum(diff * dist_matrix)

    return weighted_sum / n

def user_infos(day, user):
    e, m = get_pickle_day(day)

    if user not in m:
        return None

    JOUR_START, JOUR_END = 6 * 3600 + 30 * 60, 19 * 3600 + 50 * 60

    Ant_nuit = features._dominant_antenne(
        e[user][0], e[user][1], JOUR_START, JOUR_END, periode="nuit"
    )

    Ant_jour = features._dominant_antenne(
        e[user][0], e[user][1], JOUR_START, JOUR_END, periode="jour"
    )

    cell, radius = features._rayon_gyration_fast(e[user][0], e[user][1], dist_idx)

    radius = round(radius)

    return {
        "Day": day,
        "User": user,
        "Age": m[user]["age"],
        "Sex": m[user]["sexe"],
        "Big_nb": m[user]["Big_number"],
        "Behaviour": m[user]["comportement"],
        "Home_cell": m[user]["cellule_home"],
        "Activity_cell": m[user]["cellule_work"],
        "Ant_nuit": Ant_nuit,
        "Ant_jour": Ant_jour,
        "Gyration": (cell, radius),
        "Dist_night_day": (
            round(dist_idx.get(Ant_nuit, Ant_jour)) if Ant_jour and Ant_nuit else None
        ),
        "Nb_records": m[user]["nb_records"],
        "Nb_antennes": len(set(e[user][0])),
    }

def chain_infos(chain):

    rows = []

    if isinstance(chain[0], tuple):

        for day, user in chain:
            infos = user_infos(day, user)

            if infos:
                rows.append(infos)

    else:
        day = 12
        i = 0

        while day < 27 and i < len(chain):

            user = chain[i]
            infos = user_infos(day, user)

            if infos:
                rows.append(infos)
                i += 1
            else:
                empty = EMPTY_ROW_TEMPLATE.copy()
                empty["Day"] = day
                rows.append(empty)

            day += 1

    # ---------- Affichage tableau Excel ----------

    if not rows:
        print("Aucune donnée")
        return

    headers = list(rows[0].keys())

    # lignes
    for row in rows:

        values = []

        for h in headers:

            v = row.get(h, "NA")

            # remplace None / vide
            if v is None or v == "":
                v = "NA"

            # évite les retours ligne qui cassent Excel
            v = str(v).replace("\n", " ").replace("\t", " ")

            values.append(v)

        df = pd.DataFrame(rows)
        df.to_csv("csv/users.csv", sep=";", index=False)

def plot_chain_trajectoy_medoid(chain):
    if isinstance(chain[0], tuple):
        for day, user in chain:
            e, _ = get_pickle_day(day)
            traj = build_trajectory_medoid(e[user], dist_idx)
            fig = graph.plot_trajectory_medoid(traj, day, user)
            fig.show()
    else:
        day = 12
        i = 0
        while day < 27 and i < len(chain):
            user = chain[i]
            e, _ = get_pickle_day(day)
            if user in e:
                traj = build_trajectory_medoid(e[user], dist_idx)
                fig = graph.plot_trajectory_medoid(traj, day, user)
                fig.show()
                i += 1
                day += 1
            else:
                day += 1

def evaluate(userA, eA, userB, eB):
    fa = features.compute_features(userA, eA, dist_idx)
    fb = features.compute_features(userB, eB, dist_idx)
    score = similarity.similarity_score(fa, fb)
    return score

def evaluate_chain(chain):
    
    rows = []
    
    if isinstance(chain[0], tuple):

        for i in range(len(chain) - 1):
            eA, _ = get_pickle_day(chain[i][0])
            eB, _ = get_pickle_day(chain[i + 1][0])
            score = evaluate(chain[i][1], eA, chain[i + 1][1], eB)

            if score:
                rows.append(score)
    
    else:
        dayA = 12
        i = 0

        while dayA < 26 and i < len(chain) - 1:
            
            eA, _ = get_pickle_day(dayA)
            
            if chain[i] in eA:
                dayB = dayA + 1
                findB = False
                while dayB < 27 and not findB:
                    eB, _ = get_pickle_day(dayB)
                    if chain[i + 1] in eB:
                        findB = True
                        score = evaluate(chain[i], eA, chain[i + 1], eB)
                        if score:
                            rows.append(score)
                        i += 1
                        dayA = dayB
                    else:
                        dayB += 1
            else:
                dayA += 1
                
    # ---------- Affichage tableau Excel ----------

    if not rows:
        print("Aucune donnée")
        return

    headers = list(rows[0].keys())

    # lignes
    for row in rows:

        values = []

        for h in headers:

            v = row.get(h, "NA")

            # remplace None / vide
            if v is None or v == "":
                v = "NA"

            # évite les retours ligne qui cassent Excel
            v = str(v).replace("\n", " ").replace("\t", " ")

            values.append(v)

        df = pd.DataFrame(rows)
        df.to_csv("csv/users.csv", sep=";", index=False)
                
   
import matplotlib.pyplot as plt    
    
def plot_score(userA, dayA, dayB): 
    eA, _ = get_pickle_day(dayA)
    eB, _ = get_pickle_day(dayB)
    
    fa = features.compute_features(userA, eA, dist_idx)
    
    all_scores = {
        "score": [],
        "night_cell": [],
        "day_cell": [],
        "most_common_cell": [],
        "medoid_cell": [],
        "gyration_radius": [],
        "dist_night_day": [],
        "nb_distinct_cell": [],
        "hour_signature": [],
    }


    for userB in eB.keys():

        fb = features.compute_features(userB, eB, dist_idx)

        s = similarity.similarity_score(fa, fb)

        for key in all_scores:
            all_scores[key].append(s[key])


    for key, values in all_scores.items():
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=30)
        plt.title(key)
        plt.grid(True)
        plt.show()
        
        
def plot_gap_proportion(times_dict, gap_hours=4):

    GAP = gap_hours * 3600 + 30

    def hour(t):
        return (t // 3600) % 24

    hour_users = np.zeros(24)
    hour_with_gap = np.zeros(24)

    for user, times in times_dict.items():

        times = sorted(times)

        has_gap_in_hour = set()

        for t1, t2 in zip(times[:-1], times[1:]):
            if t2 - t1 >= GAP:

                h1, h2 = hour(t1), hour(t2)

                # couvrir toutes les heures touchées
                if h2 >= h1:
                    hours = range(h1, h2 + 1)
                else:
                    hours = list(range(h1, 24)) + list(range(0, h2 + 1))

                has_gap_in_hour.update(hours)

        for h in range(24):
            hour_users[h] += 1
            if h in has_gap_in_hour:
                hour_with_gap[h] += 1

    return hour_with_gap / np.maximum(hour_users, 1)

def plot_gap_users(times_dict):
    p = plot_gap_proportion(times_dict)

    plt.plot(range(24), p, marker='o')
    plt.xticks(range(24))
    plt.xlabel("Hour")
    plt.ylabel("Nb of users with a gap ≥ 4h")
    plt.title("Users outside cellular network coverage by hour")
    plt.grid()
    plt.show()
    
def times_dict(day):
    e, _ = get_pickle_day(day)

    times_dict = {}

    for user in e:
        _, times = e[user]
        times_dict[user] = times

    return times_dict
