from geopy.distance import geodesic
from pathlib import Path

import numpy as np

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

            data.append({"meta": meta, "events": events})

    return {user["meta"]["id"]: user for user in data}  # index by user_id

def get_pickle_day(day):
    events = np.load(PATH_SOURCE / f"events_day_{day}.npz", allow_pickle=True)["data"].item()
    meta = np.load(PATH_SOURCE / f"meta_day_{day}.npy", allow_pickle=True).item()
    return events, meta

def convert_day(id):
    raw = get_data_day(id)

    events_data = {}
    meta_data = {}

    for user_id, user in raw.items():

        # META (on garde tel quel)
        meta_data[user_id] = user["meta"]
        
        events = user['events']

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

def distance_between_cells(cell1, cell2):
    if cell1 not in index_map or cell2 not in index_map:
        return None

    return dist_matrix[index_map[cell1], index_map[cell2]]

def normalize(events):
    codes, times = events

    order = np.argsort(times)
    codes = codes[order]
    times = times[order]

    keep = np.ones(len(codes), dtype=bool)
    keep[1:] = codes[1:] != codes[:-1]

    codes = codes[keep]
    times = times[keep]

    start_extra = times[0] > 0
    end_extra = times[-1] < 86400

    new_size = len(times) + start_extra + end_extra

    new_codes = np.empty(new_size, dtype=object)
    new_times = np.empty(new_size, dtype=times.dtype)

    idx = 0

    if start_extra:
        new_codes[0] = codes[0]
        new_times[0] = 0
        idx = 1

    new_codes[idx:idx+len(codes)] = codes
    new_times[idx:idx+len(times)] = times

    idx += len(codes)

    if end_extra:
        new_codes[idx] = codes[-1]
        new_times[idx] = 86400

    return new_codes, new_times

def compute_merge_on_timeline_distance(events1, events2):

    c1, t1 = normalize(events1)
    c2, t2 = normalize(events2)

    i = j = 0
    total = 0.0
    t = 0

    while i < len(t1) - 1 or j < len(t2) - 1:

        next_t1 = t1[i + 1] if i + 1 < len(t1) else 86400
        next_t2 = t2[j + 1] if j + 1 < len(t2) else 86400

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

        if t >= 86400:
            break

    return total / 86400

# def build_histogram(events, T_end=86400):
#     hist = {}

#     # début : on suppose que la personne est déjà dans la première cellule
#     cell_prev, t_prev = events[0]

#     # cas début journée
#     if t_prev > 0:
#         hist[cell_prev] = hist.get(cell_prev, 0) + t_prev

#     for i in range(len(events) - 1):
#         cell, t = events[i]
#         cell_next, t_next = events[i + 1]

#         duration = t_next - t
#         hist[cell] = hist.get(cell, 0) + duration

#     # fin journée
#     last_cell, last_time = events[-1]
#     if last_time < T_end:
#         hist[last_cell] = hist.get(last_cell, 0) + (T_end - last_time)

#     # normalisation
#     total = sum(hist.values())
#     for c in hist:
#         hist[c] /= total

#     return hist


# import numpy as np
# import ot

# def emd_distance(histA, histB, dist_matrix):
#     a = np.array(histA)
#     b = np.array(histB)
#     return ot.emd2(a, b, dist_matrix) # essayer ot.sinkhorn2(a, b, M, reg=...)

# def build_transition(events):
#     trans = {}
#     total = 0

#     for i in range(len(events) - 1):
#         c1 = events[i][0]
#         c2 = events[i+1][0]

#         trans[(c1, c2)] = trans.get((c1, c2), 0) + 1
#         total += 1

#     for k in trans:
#         trans[k] /= total

#     return trans
