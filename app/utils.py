from geopy.distance import geodesic
from pathlib import Path

import pandas as pd

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

    return data


def to_int(x: str) -> int | None:
    x = x.strip().replace('"', "")
    return int(x) if x else None


def convert_lat_lon_distance_to_meter(
    point1: tuple[float, float], point2: tuple[float, float]
) -> float:
    return geodesic(point1, point2).meters

def build_trajectory(events):
    times = []
    lats = []
    lons = []
    dist_raws = []
    dist_cums = []
    dist_origins = []
    cells = []

    prev_idx = None
    origin_idx = None
    cumulative_dist = 0.0

    for cell, t in events:
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

        # append arrays
        times.append(t)
        lats.append(lat)
        lons.append(lon)
        dist_raws.append(dist_raw)
        dist_cums.append(cumulative_dist)
        dist_origins.append(dist_origin)
        cells.append(cell)

        prev_idx = idx

    return {
        "time": np.array(times),
        "latitude": np.array(lats),
        "longitude": np.array(lons),
        "dist_raw": np.array(dist_raws),
        "dist_cumulative": np.array(dist_cums),
        "dist_origin": np.array(dist_origins),
        "cell": np.array(cells),
    }

def distance_between_cells(cell1, cell2):
    if cell1 not in index_map or cell2 not in index_map:
        return None
    
    return dist_matrix[index_map[cell1], index_map[cell2]]

def distance_between_trajectories(trajectory1, trajectory2):
    total_distance = 0.0
    
