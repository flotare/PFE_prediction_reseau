import utils
import features
from pathlib import Path
import numpy as np
import json


PATH_DATA = Path("data")

dist_matrix = np.load(PATH_DATA / "dist_matrix.npy", allow_pickle=True)
cell_ids = np.load(PATH_DATA / "cell_ids.npy", allow_pickle=True)
coords = np.load(PATH_DATA / "coords.npy", allow_pickle=True)

with open(PATH_DATA / "transition_number.json", 'r') as file:
    trans = json.load(file)

dist_idx = features.DistanceIndex(dist_matrix, cell_ids)

max_dists = []

for cell in trans:
    max = 0
    for c in trans[cell]:
        if dist_idx.get(cell, c) > max:
            max = dist_idx.get(cell, c)
    max_dists.append(max)
    
print(max_dists, len(max_dists))
