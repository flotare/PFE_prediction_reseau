import numpy as np
from utils import get_pickle_day
from graph import plot_signal, plot_fft, plot_dupplicate_record_repartition

from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d

def build_signal(timestamps, day_seconds=86400):

    signal = np.zeros(day_seconds)

    ts = np.array(timestamps, dtype=int)
    ts = ts[(ts >= 0) & (ts < day_seconds)]

    for t in ts:
        signal[t] += 1

    return signal

def compute_fft(signal):

    fft_vals = np.fft.rfft(signal)
    power = np.abs(fft_vals)

    power[0] = 0  # DC component

    freqs = np.fft.rfftfreq(len(signal), d=1)

    return freqs, power


def fft_features(timestamps):

    signal = build_signal(timestamps)

    fft_vals = np.fft.rfft(signal)

    power = np.abs(fft_vals)

    freqs = np.fft.rfftfreq(len(signal), d=1)

    return freqs, power


def fft_peak_features(power):

    peaks, _ = find_peaks(power, prominence=np.std(power))

    if len(peaks) == 0:
        return {"n_peaks": 0, "max_peak": 0, "mean_peak": 0, "peak_ratio": 0}

    peak_vals = power[peaks]

    return {
        "n_peaks": len(peaks),
        "max_peak": np.max(peak_vals),
        "mean_peak": np.mean(peak_vals),
        "peak_ratio": np.max(peak_vals) / np.mean(power),
    }


def fft_bot_score(peaks_info):

    score = 0
    reasons = []

    if peaks_info["n_peaks"] == 0:
        return 0, ["Pas de périodicité"]

    if peaks_info["n_peaks"] == 1:
        score += 2
        reasons.append("Une seule fréquence dominante")

    if peaks_info["peak_ratio"] > 10:
        score += 2
        reasons.append("Pic FFT dominant")

    if peaks_info["max_peak"] > 50:
        score += 1
        reasons.append("Signal très périodique")

    return score, reasons


def extract_harmonics(freqs, power):

    power = power.copy()
    power[0] = 0

    # smoothing
    smooth = gaussian_filter1d(power, sigma=3)

    # peaks
    peaks, props = find_peaks(smooth, prominence=np.std(smooth), width=5)

    peak_freqs = freqs[peaks]
    peak_powers = smooth[peaks]

    if len(peak_freqs) < 2:
        return []

    # ----------------------------
    # recherche de fondamentale
    # ----------------------------

    best_f0 = None
    best_score = -1

    candidates = peak_freqs[:10]  # candidats plausibles

    for f0 in candidates:

        score = 0

        for f in peak_freqs:

            ratio = f / f0
            error = abs(ratio - round(ratio))

            if error < 0.08:
                score += 1

        if score > best_score:
            best_score = score
            best_f0 = f0

    # ----------------------------
    # conserve vraies harmoniques
    # ----------------------------

    harmonics = []

    for f in peak_freqs:

        ratio = f / best_f0

        if abs(ratio - round(ratio)) < 0.08:
            harmonics.append(f)

    return harmonics, best_f0


def detect_frequencies(
    power, sample_rate=1.0, method="savgol", win=15, thresh_pct=0.30, min_dist=30
):
    freqs = np.fft.rfftfreq(len(power) * 2 - 2, d=1 / sample_rate)

    # Lissage
    if method == "moving":
        kernel = np.ones(win) / win
        smoothed = np.convolve(power, kernel, mode="same")
    elif method == "gaussian":
        smoothed = gaussian_filter1d(power, sigma=win / 5)
    elif method == "savgol":
        smoothed = savgol_filter(power, window_length=win, polyorder=2)

    # Détection des pics
    min_height = smoothed.max() * thresh_pct
    peaks, props = find_peaks(smoothed, height=min_height, distance=min_dist)

    return freqs[peaks], smoothed[peaks], smoothed


def merge_close_peaks(harmonics, powers, tol=0.02):
    """
    Fusionne les fréquences proches en gardant
    la plus puissante dans chaque groupe.
    """

    pairs = sorted(zip(harmonics, powers), key=lambda x: x[0])

    merged_freqs = []
    merged_powers = []

    current_group = [pairs[0]]

    for f, p in pairs[1:]:

        # même cluster fréquentiel
        if abs(f - current_group[-1][0]) < tol:
            current_group.append((f, p))

        else:
            # garde le plus puissant
            best = max(current_group, key=lambda x: x[1])

            merged_freqs.append(best[0])
            merged_powers.append(best[1])

            current_group = [(f, p)]

    # dernier groupe
    best = max(current_group, key=lambda x: x[1])

    merged_freqs.append(best[0])
    merged_powers.append(best[1])

    return np.array(merged_freqs), np.array(merged_powers)


def merge_peaks_centroid(harmonics, powers, tol=0.03):

    pairs = sorted(zip(harmonics, powers), key=lambda x: x[0])

    groups = []
    current = [pairs[0]]

    for f, p in pairs[1:]:

        if abs(f - current[-1][0]) < tol:
            current.append((f, p))
        else:
            groups.append(current)
            current = [(f, p)]

    groups.append(current)

    merged_freqs = []
    merged_powers = []

    for group in groups:

        freqs = np.array([g[0] for g in group])
        power = np.array([g[1] for g in group])

        # centroid énergétique
        centroid = np.sum(freqs * power) / np.sum(power)

        merged_freqs.append(centroid)
        merged_powers.append(np.max(power))

    return np.array(merged_freqs), np.array(merged_powers)


def bot_features(freqs, power, day_seconds=86400):
    power = power.copy()
    power[0] = 0

    # Ignore tout ce qui est en dessous de 1/3600 Hz (période > 1h, trop lent)
    power[freqs < 1 / 3600] = 0

    # Masque les fréquences journalières
    for k in range(1, 20):
        mask = np.abs(freqs - k / day_seconds) < 2e-5
        power[mask] = 0

    noise_floor = np.median(power[power > 0])

    # distance en indices correspondant à ~0.01 Hz minimum entre deux pics
    freq_resolution = freqs[1] - freqs[0]  # ~1/86400
    min_distance = int(0.01 / freq_resolution)  # ~860 indices

    peaks, _ = find_peaks(power, prominence=noise_floor * 3, distance=min_distance)

    if len(peaks) == 0:
        return {"n_harmonics": 0, "snr": 0, "regularity": 0, "harmonic_freqs": []}

    peaks = peaks[np.argsort(power[peaks])[::-1]]
    peak_freqs = freqs[peaks]
    top_freq = peak_freqs[0]

    harmonics = [
        f
        for f in peak_freqs[1:]
        if any(abs(f - k * top_freq) / top_freq < 0.05 for k in range(2, 20))
    ]

    return {
        "n_harmonics": len(harmonics),
        "harmonic_freqs": [round(f, 4) for f in harmonics],
        "snr": power[peaks[0]] / noise_floor,
        "regularity": len(harmonics) / max(len(peak_freqs) - 1, 1),
    }


def analyze_fft(timestamps):

    signal = build_signal(timestamps)

    freqs, power = compute_fft(signal)

    harmonics, best_f0 = extract_harmonics(freqs, power)

    # features = bot_features(freqs, power)

    return freqs, power, harmonics, best_f0


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

        # Construire la suite depuis ce point
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

    


from collections import Counter


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


import librosa


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


def filter_times(times, min_gap=3):
    if len(times) == 0:
        return times

    filtered = [times[0]]

    for t in times[1:]:
        if t == filtered[-1] or t - filtered[-1] >= min_gap:
            filtered.append(t)

    return np.array(filtered)


def filter_frequences(times):
    freq = []

    periode_max, value_max = period_max(times)
        
    if value_max == 0:
        return times, 0

    ref_value_max = value_max

    times_filtered = times.copy()
    
    print(f"p_max : {periode_max}, ref_v_max : {value_max}")

    while value_max >= max(ref_value_max // 10, 2) and periode_max > 1:
        freq.append(periode_max)
        new_times_filtered = filter_freq(times_filtered, periode_max)
         
        if new_times_filtered is None:
            break
        else:
            print(f"new : {np.array_equal(times_filtered, new_times_filtered)}")
            times_filtered = new_times_filtered

        periode_max, value_max = period_max(times_filtered)
        print(f"p_max : {periode_max}, value_max : {value_max}")
    print("sortit")
    return times_filtered, freq


def get_mfcc_freq(user_id, event):

    _, times = event[user_id]
    
    times = filter_times(times)
        
    times_filtered, freq = filter_frequences(times)
    
    signal = build_signal(times_filtered)
        
    mfcc = extract_mfcc(signal)

    return mfcc, freq


from scipy.spatial.distance import cosine


def evaluate_mfcc(userA, eA, userB, eB):
    mfccA, freqA = get_mfcc_freq(userA, eA)
    mfccB, freqB = get_mfcc_freq(userB, eB)
    score = 1 - cosine(np.ravel(mfccA), np.ravel(mfccB))
    return {
        "userA-userB": f"{userA}-{userB}",
        "freqA": freqA,
        "freqB": freqB,
        "score": score,
    }


import pandas as pd


def evaluate_chain(chain):

    rows = []

    if isinstance(chain[0], tuple):

        for i in range(len(chain) - 1):
            eA, _ = get_pickle_day(chain[i][0])
            eB, _ = get_pickle_day(chain[i + 1][0])
            score = evaluate_mfcc(chain[i][1], eA, chain[i + 1][1], eB)

            if score:
                rows.append(score)

    else:
        dayA = 12
        i = 0

        while dayA < 26 and i < len(chain) - 1:
            print(f"day : {dayA}, user {chain[i]} -> {chain[i+1]}")

            eA, _ = get_pickle_day(dayA)

            if chain[i] in eA:
                dayB = dayA + 1
                findB = False
                while dayB < 27 and not findB:
                    eB, _ = get_pickle_day(dayB)
                    if chain[i + 1] in eB:
                        findB = True
                        score = evaluate_mfcc(chain[i], eA, chain[i + 1], eB)
                        print(f"score : {score}")
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


# chain = [211805, 2054373, 3289518, 5181216, 7339819, 7929474, 9571993, 11206984, 13282164, 15944783, 15721737, 15875553, 17415362, 20923224, 19412322]
# chain = [555933, 3239968, 4902378, 6980330, 7354759, 10663454, 10709261, 13139752, 14658525, 15542677, 15782763, 18096367, 17149014, 21391582, 20451830]
# chain = [471290,1460012,2160772,6282029,6290888,6451376,11613093,13270961,8682862,12976180,17705900,16564763,21427857,19551794,20900318]

# chain = [101877,3745234,2786163,3624428,5859953,11138232,10433344,10955956,13282164,15944783,15008305,12633743,15685231,21264515,19312909]
# chain = [396316,3580161,6882287,6153514,7339819,7929474,10634514,12995267]

# # chain = [5500244,7308276,8456610,11749797,11923100,13676959,16595315,17698482]

# evaluate_chain(chain)

# user = 616942
# user = 127455
# user = 59011
# user = 654979
# user = 131808
# user = 79060
# user = 1769
# user = 2784


# top_n_indices = np.argsort(values)[::-1][: (86400 // periode_max)]
# top_n_periods = periods[top_n_indices]


# periods_array = top_n_periods.copy()
# periodic = []
# others = list(periods_array)

# for val in periods_array:
#     # Cherche si val a un voisin à ~600s dans la liste
#     has_neighbor = any(
#         abs(abs(val - other) - 600) <= 1
#         for other in periods_array
#         if other != val
#     )
#     if has_neighbor and val not in periodic:
#         periodic.append(val)
#         if val in others:
#             others.remove(val)

# periodic = np.array(sorted(periodic))
# others = np.array(sorted(others))

# mask = ~np.isin(periods, periodic)
# periods_filtered = periods[mask]
# values_filtered = values[mask]

# import plotly.graph_objects as go

# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=periods,
#     y=values,
#     mode='lines',
#     line=dict(color='steelblue', width=1),
#     name='occurrences'
# ))

# fig.update_layout(
#     title=f"Distribution of periods recorded in the signal, user {user}, day {day}",
#     xaxis_title='Periods (s)',
#     yaxis_title='Number of records',
# )

# fig.show()

################################### Plot signal of a user ###########################

# day = 12

# user = 50882

# e, _ = get_pickle_day(day)

# _, times = e[user]

# signal = build_signal(times)

# plot_signal(signal, user, day, len(times))

# mfccA, freqA = get_mfcc_freq(user, e)

# breakpoint()

# plot_signal(signal, user, day, len(times))

############################## plot fft time user #######################


# import matplotlib.pyplot as plt

# signal = build_signal(times)
# freqs, power = compute_fft(signal)

# mask = freqs > 0

# periods = 1 / freqs[mask]  # en secondes

# plt.figure(figsize=(12, 6))
# plt.plot(periods, power[mask])
# plt.xscale("log")
# plt.xlabel("Période (s)")
# plt.ylabel("Amplitude FFT")
# plt.grid(True)
# plt.show()

############################## plot fft user #######################

# freqs, power, harmonics, best_f0 = analyze_fft(times)

# harmonics = np.array(harmonics)

# # puissance associée
# harmonic_power = []

# for h in harmonics:
#     idx = np.argmin(np.abs(freqs - h))
#     harmonic_power.append(power[idx])

# harmonic_power = np.array(harmonic_power)

# # merge intelligent
# harmonics, harmonic_power = merge_peaks_centroid(
#     harmonics,
#     harmonic_power,
#     tol=0.035
# )

# plot_fft(freqs, power, harmonics, harmonic_power, user, day)

# print("harmonics info:", harmonics)


################### plot bode #######################


# import matplotlib.pyplot as plt
# import numpy as np

# freqs, power, peaks_info = analyze_fft(times)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
# fig.suptitle(f"Diagramme de Bode — User {user}", fontsize=13, fontweight='bold')

# # --- Magnitude (dB) ---
# power_db = 20 * np.log10(np.maximum(power[1:], 1e-6))
# freqs_plot = freqs[1:]

# ax1.semilogx(freqs_plot, power_db, color='#185FA5', linewidth=1.2)
# ax1.set_ylabel("Magnitude (dB)")
# ax1.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
# ax1.set_title("Magnitude", fontsize=11, loc='left', color='gray')

# # Marqueurs de pics
# from scipy.signal import find_peaks
# peaks, _ = find_peaks(power[1:], prominence=np.std(power[1:]))
# for p in peaks:
#     ax1.axvline(x=freqs_plot[p], color='#D85A30', linewidth=0.8, linestyle='--', alpha=0.7)

# # --- Phase (°) ---
# fft_vals = np.fft.rfft(build_signal(times))
# phase = np.angle(fft_vals[1:], deg=True)

# ax2.semilogx(freqs_plot, phase, color='#1D9E75', linewidth=1.2)
# ax2.set_ylabel("Phase (°)")
# ax2.set_xlabel("Fréquence (Hz)")
# ax2.set_ylim(-180, 180)
# ax2.set_yticks([-180, -90, 0, 90, 180])
# ax2.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
# ax2.set_title("Phase", fontsize=11, loc='left', color='gray')

# for p in peaks:
#     ax2.axvline(x=freqs_plot[p], color='#D85A30', linewidth=0.8, linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()


########################### Plot users with more than 1 records / s for day #############################

# plot_dupplicate_record_repartition(event, day)

############################ Plot MFCC ##################################################################

# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
# plt.imshow(
#     mfcc.T,
#     aspect='auto',
#     origin='lower'
# )
# plt.colorbar()
# plt.ylabel("Coefficient MFCC")
# plt.xlabel("Frame")
# plt.title(f"MFCC user {user}, day {day}")
# plt.show()


##################################### compte les utilisateurs avec des gap #####################################

# from features import _signature_horaire

# for day in range(12, 27):
#     e, _ = get_pickle_day(day)

#     tot = len(e)
#     count = 0
#     for user in e:

#         antennes, timestamps = e[user]

#         sig = _signature_horaire(
#                 antennes,
#                 timestamps,
#                 max_allowed_gap=4 * 3600 + 30,
#                 truncated_duration=2 * 3600,
#             )
        
#         if None in sig[4:21]:
#             count += 1

#     print(f"pourcentage user_to_keep day {day} : {(tot - count) / tot} : Nb of users : {tot} : Nb to keep : {tot - count}")
