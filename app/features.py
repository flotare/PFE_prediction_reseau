import numpy as np
from collections import Counter, defaultdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import librosa


class DistanceIndex:
    """
    Lookup O(1) de distance entre deux codes antennes.
    Utilise un dict Python pur + matrice numpy brute.
    """

    def __init__(self, dist_matrix_raw: np.ndarray, cell_ids: np.ndarray):
        self.max_dist = float(dist_matrix_raw.max())
        self._idx = {code: i for i, code in enumerate(cell_ids)}
        self._matrix = dist_matrix_raw.astype(np.float32)
        self.cell_ids = cell_ids

    def get(self, a: str, b: str) -> Optional[float]:
        if a == b:
            return 0.0
        ia = self._idx.get(a)
        ib = self._idx.get(b)
        if ia is None or ib is None:
            return None
        return float(self._matrix[ia, ib])

    def neighbors(self, a: str, radius: float) -> list:
        ia = self._idx.get(a)
        if ia is None:
            return [a]
        row = self._matrix[ia]
        return [self.cell_ids[i] for i in np.where(row <= radius)[0]]

    def norm_sim(self, d: Optional[float]) -> float:
        if d is None:
            return 0.5
        if self.max_dist == 0:
            return 1.0
        return float(1.0 - min(d / self.max_dist, 1.0))


JOUR_START, JOUR_END = 6 * 3600 + 30 * 60, 19 * 3600 + 50 * 60


def _dominant_antenne(antennes, timestamps, jour_start, jour_end, periode="jour"):

    if len(timestamps) == 0:
        return None

    duration_by_ant = {}

    next_hour = ((timestamps[-1] // 3600) + 1) * 3600

    # durée entre chaque point et le suivant
    next_ts = np.append(timestamps[1:], next_hour)

    gaps = next_ts - timestamps

    MAX_ALLOWED_GAP = 4 * 3600 + 30
    TRUNCATED_DURATION = 2 * 3600

    durations = np.where(gaps > MAX_ALLOWED_GAP, TRUNCATED_DURATION, gaps)

    for ant, ts, dur in zip(antennes, timestamps, durations):

        if periode == "jour":
            keep = (ts > jour_start) and (ts < jour_end)

        elif periode == "nuit":
            keep = (ts <= jour_start) or (ts >= jour_end)

        else:
            raise ValueError("periode doit être 'jour' ou 'nuit'")

        if keep:
            duration_by_ant[ant] = duration_by_ant.get(ant, 0) + dur

    if not duration_by_ant:
        return None

    return max(duration_by_ant, key=duration_by_ant.get)


def _signature_horaire(
    antennes,
    timestamps,
    max_allowed_gap=4 * 3600 + 30,
    truncated_duration=2 * 3600,
):
    """
    Signature horaire :
    antenne dominante pour chaque heure de la journée.
    """

    if len(timestamps) == 0:
        return [None] * 24

    # --------------------------------------------------------------
    # Reconstruction des segments temporels
    # --------------------------------------------------------------

    next_hour = ((timestamps[-1] // 3600) + 1) * 3600

    next_ts = np.append(timestamps[1:], next_hour)

    gaps = next_ts - timestamps

    effective_next_ts = np.where(
        gaps > max_allowed_gap, timestamps + truncated_duration, next_ts
    )

    # --------------------------------------------------------------
    # Accumulateur :
    # hour -> antenna -> duration
    # --------------------------------------------------------------

    hour_ant_duration = [defaultdict(float) for _ in range(24)]

    # --------------------------------------------------------------
    # Distribution des segments dans les heures
    # --------------------------------------------------------------

    for ant, start, end in zip(antennes, timestamps, effective_next_ts):

        current = start

        while current < end:

            hour = int(current // 3600)

            hour_end = min((hour + 1) * 3600, end)

            overlap = hour_end - current

            if overlap > 0:
                hour_ant_duration[hour][ant] += overlap

            current = hour_end

    # --------------------------------------------------------------
    # Antenne dominante par heure
    # --------------------------------------------------------------

    signature = []

    for h in range(24):

        if not hour_ant_duration[h]:
            signature.append(None)

        else:
            dominant = max(hour_ant_duration[h], key=hour_ant_duration[h].get)

            signature.append(dominant)

    return signature


def _rayon_gyration_fast(antennes, timestamps, dist_idx: DistanceIndex):
    """
    Rayon de gyration pondéré par le temps passé sur chaque antenne.
    Utilise un médoïde pondéré temporellement.
    """

    if len(antennes) == 0:
        return None, None

    # ------------------------------------------------------------------
    # Estimation des durées de présence
    # ------------------------------------------------------------------

    next_hour = ((timestamps[-1] // 3600) + 1) * 3600

    next_ts = np.append(timestamps[1:], next_hour)

    gaps = next_ts - timestamps

    MAX_ALLOWED_GAP = 4 * 3600 + 30
    TRUNCATED_DURATION = 2 * 3600

    durations = np.where(gaps > MAX_ALLOWED_GAP, TRUNCATED_DURATION, gaps)

    # ------------------------------------------------------------------
    # Agrégation du temps passé par antenne
    # ------------------------------------------------------------------

    duration_by_ant = {}

    for ant, dur in zip(antennes, durations):

        if ant not in dist_idx._idx:
            continue

        duration_by_ant[ant] = duration_by_ant.get(ant, 0.0) + dur

    if len(duration_by_ant) <= 1:
        center_global = antennes[0]
        return center_global, 0.0

    ants = list(duration_by_ant.keys())

    indices = np.array([dist_idx._idx[a] for a in ants], dtype=np.int32)

    weights = np.array([duration_by_ant[a] for a in ants], dtype=np.float32)

    # ------------------------------------------------------------------
    # Sous-matrice des distances
    # ------------------------------------------------------------------

    sub = dist_idx._matrix[np.ix_(indices, indices)]

    # ------------------------------------------------------------------
    # Médoïde pondéré
    #
    # maximise :
    # w_i / (Σ d(i, centre))
    # ------------------------------------------------------------------

    dist_sums = sub.sum(axis=1)

    if dist_sums[0] == 0:

        center_local_idx = int(np.argmax(weights))
        center_global_idx = indices[center_local_idx]
        center_global = dist_idx.cell_ids[center_global_idx]

        return center_global, 0.0

    weighted_sums = weights / dist_sums

    center_local_idx = int(np.argmax(weighted_sums))

    center_global_idx = indices[center_local_idx]

    center_global = dist_idx.cell_ids[center_global_idx]

    # ------------------------------------------------------------------
    # Rayon de gyration pondéré
    #
    # sqrt( Σ w_i * d² / Σ w_i )
    # ------------------------------------------------------------------

    dists = dist_idx._matrix[center_global_idx, indices]

    rg = np.sqrt(np.sum(weights * (dists**2)) / np.sum(weights))

    return center_global, float(rg)

def filter_times(times, min_gap=3):
    if len(times) == 0:
        return times

    filtered = [times[0]]

    for t in times[1:]:
        if t == filtered[-1] or t - filtered[-1] >= min_gap:
            filtered.append(t)

    return np.array(filtered)          

def period_max(times):
    if len(times) < 2:
        return 0, 0

    diffs = np.array(
        [
            times[j] - times[i]
            for i in range(len(times))
            for j in range(i + 1, len(times))
        ]
    )

    counts = Counter(diffs)
    periods = np.arange(0, diffs.max() + 1)
    values = np.array([counts.get(p, 0) for p in periods])
    values[values == 1] = 0
    
    idx_max = np.argmax(values)

    value_max = values[idx_max]
    periode_max = periods[idx_max]
        
    return periode_max, value_max

def find_nearest(times_set, target, tolerance=2):
    """Retourne la valeur la plus proche de target dans times_set, dans la tolérance."""
    candidates = [t for t in times_set if abs(t - target) <= tolerance]
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(t - target))

def filter_freq(times, periode, min_length=2):
    times_set = set(times)
    to_remove = []
    visited = set()

    for time in times:
        if time in visited:
            continue
        suite = []
        current = time
        while True:
            nearest = find_nearest(times_set, current, tolerance=2)
            if nearest is None:
                break
            else:
                suite.append(nearest)
                visited.add(nearest)
                current = nearest + periode
        if len(suite) >= min_length:
            to_remove = to_remove + suite

    if len(to_remove) > min_length:

        remaining = len(np.unique(times)) - len(np.unique(to_remove))

        if remaining >= 5:
            mask = ~np.isin(times, to_remove)
            times_filtered = times[mask]

            return times_filtered
        else:
            return None

def filter_frequences(times):
    freq = []

    periode_max, value_max = period_max(times)
        
    if value_max == 0:
        return times, 0

    ref_value_max = value_max

    times_filtered = times.copy()

    while value_max >= max(ref_value_max // 10, 2) and periode_max > 1:
        freq.append(periode_max)
        new_times_filtered = filter_freq(times_filtered, periode_max)
         
        if new_times_filtered is None:
            break
        else:
            times_filtered = new_times_filtered

        periode_max, value_max = period_max(times_filtered)
    return times_filtered, freq

def build_signal(timestamps, day_seconds=86400):

    signal = np.zeros(day_seconds)

    ts = np.array(timestamps, dtype=int)
    ts = ts[(ts >= 0) & (ts < day_seconds)]

    for t in ts:
        signal[t] += 1

    return signal

def extract_mfcc(signal, sr=1, n_mfcc=1, n_fft=1024, hop_length=128):
    """
    signal : array 1D
    sr : fréquence d'échantillonnage (1 Hz ici)
    """

    signal = np.asarray(signal, dtype=np.float32)

    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )

    return mfcc.T

def _mfcc(timestamps):
    times = filter_times(timestamps)
        
    times_filtered, freq = filter_frequences(times)
    
    signal = build_signal(times_filtered)
        
    mfcc = extract_mfcc(signal)

    return mfcc, freq


def compute_features(user_id, events, dist_idx: DistanceIndex):
    antennes, timestamps = events[user_id]

    ant_nuit = _dominant_antenne(
        antennes, timestamps, JOUR_START, JOUR_END, periode="nuit"
    )

    ant_jour = _dominant_antenne(
        antennes, timestamps, JOUR_START, JOUR_END, periode="jour"
    )
    most_common_cell = (
        Counter(antennes).most_common(1)[0][0] if len(antennes) > 0 else None
    )
    medoid_cell, gyration_radius = _rayon_gyration_fast(antennes, timestamps, dist_idx)
    
    mfcc, freq = _mfcc(timestamps)
    
    return {
        "id": user_id,
        "night_cell": ant_nuit,
        "day_cell": ant_jour,
        "most_common_cell": most_common_cell,
        "medoid_cell": medoid_cell,
        "gyration_radius": gyration_radius,
        "dist_night_day": (
            dist_idx.get(ant_nuit, ant_jour) if ant_jour and ant_nuit else None
        ),
        "nb_distinct_cell": int(len(set(antennes))),
        "hour_signature": _signature_horaire(antennes, timestamps),
        "nb_records": len(antennes),
        "MFCC": mfcc,
        "frequences": freq,
    }


def _compute_batch(args):
    uid_batch, events, dist_idx = args
    results = {}
    for uid in uid_batch:
        try:
            results[uid] = compute_features(uid, events, dist_idx)
        except Exception:
            pass
    return results


def compute_all_features(events, dist_idx: DistanceIndex, n_workers=None, verbose=True):
    all_ids = list(events.keys())
    n_workers = n_workers or max(1, os.cpu_count() - 1)
    batch_size = max(500, len(all_ids) // (n_workers * 4))
    batches = [all_ids[i : i + batch_size] for i in range(0, len(all_ids), batch_size)]

    if verbose:
        print(
            f"  {len(all_ids):,} utilisateurs | {len(batches)} batches | {n_workers} workers"
        )

    features = {}
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_compute_batch, (batch, events, dist_idx)): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_result = future.result()
            features.update(batch_result)
            done += len(batch_result)
            if verbose:
                print(f"  {done:,}/{len(all_ids):,} traités...", end="\r")

    if verbose:
        print(f"  {len(features):,} utilisateurs traités.    ")

    return features


