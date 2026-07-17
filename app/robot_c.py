import numpy as np
from scipy.signal import find_peaks
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import get_pickle_day


# ── Signal construction ────────────────────────────────────────────────────────

def build_signal(timestamps, day_seconds=86400):
    signal = np.zeros(day_seconds)
    ts = np.array(timestamps, dtype=int)
    ts = ts[(ts >= 0) & (ts < day_seconds)]
    for t in ts:
        signal[t] += 1
    return signal


# ── FFT features ───────────────────────────────────────────────────────────────

def compute_fft(signal):
    fft_vals = np.fft.rfft(signal)
    power = np.abs(fft_vals)
    power[0] = 0  # remove DC
    freqs = np.fft.rfftfreq(len(signal), d=1)
    return freqs, power


def fft_peak_features(power):
    peaks, _ = find_peaks(power, prominence=np.std(power))

    if len(peaks) == 0:
        return {
            "n_peaks": 0,
            "max_peak": 0.0,
            "mean_peak": 0.0,
            "peak_ratio": 0.0,
            "top_peak_concentration": 0.0,
        }

    peak_vals = power[peaks]
    total_power = np.sum(power)
    top3 = np.sort(peak_vals)[::-1][:3]
    concentration = np.sum(top3) / (total_power + 1e-9)

    return {
        "n_peaks": len(peaks),
        "max_peak": float(np.max(peak_vals)),
        "mean_peak": float(np.mean(peak_vals)),
        "peak_ratio": float(np.max(peak_vals) / (np.mean(power) + 1e-9)),
        "top_peak_concentration": float(concentration),
    }


# ── Temporal features ─────────────────────────────────────────────────────────

def compute_iets(timestamps):
    """Retourne les inter-event times (IETs) triés, > 0."""
    ts = np.sort(np.array(timestamps, dtype=float))
    if len(ts) < 2:
        return np.array([])
    iets = np.diff(ts)
    return iets[iets > 0]


def temporal_features(iets, n_events):
    if len(iets) == 0:
        return {"n_events": n_events, "iet_cv": 0.0, "iet_entropy": 0.0, "burst_ratio": 0.0}

    cv = float(np.std(iets) / (np.mean(iets) + 1e-9))

    bins = np.logspace(np.log10(max(iets.min(), 1)), np.log10(iets.max() + 1), 20)
    hist, _ = np.histogram(iets, bins=bins)
    iet_ent = float(entropy(hist + 1e-9))

    burst_ratio = float(np.sum(iets <= 1) / len(iets))

    return {
        "n_events": n_events,
        "iet_cv": cv,
        "iet_entropy": iet_ent,
        "burst_ratio": burst_ratio,
    }


# ── Spectral entropy ──────────────────────────────────────────────────────────

def spectral_entropy(power):
    p = power / (np.sum(power) + 1e-9)
    return float(entropy(p + 1e-9))


# ── Composite bot score ────────────────────────────────────────────────────────

def bot_score(fft_feats, temp_feats, spec_ent, verbose=False):
    """Score de 0 à 10. Plus élevé = plus bot-like."""
    score = 0.0
    reasons = []

    if fft_feats["n_peaks"] == 1:
        score += 2.0
        reasons.append("Une seule fréquence dominante dans le spectre")

    if fft_feats["peak_ratio"] > 10:
        score += 2.0
        reasons.append(f"Pic FFT très dominant (ratio={fft_feats['peak_ratio']:.1f})")

    if fft_feats["top_peak_concentration"] > 0.5:
        score += 1.5
        reasons.append("50 %+ de la puissance concentrée sur 3 fréquences")

    if spec_ent < 5.0:
        score += 1.5
        reasons.append(f"Entropie spectrale faible ({spec_ent:.2f}) → signal régulier")

    if temp_feats["iet_cv"] < 0.3:
        score += 2.0
        reasons.append(f"Intervalles très réguliers (CV={temp_feats['iet_cv']:.2f})")
    elif temp_feats["iet_cv"] < 0.7:
        score += 0.5
        reasons.append(f"Intervalles modérément réguliers (CV={temp_feats['iet_cv']:.2f})")

    if temp_feats["burst_ratio"] < 0.05:
        score += 1.0
        reasons.append("Très peu d'activité en rafale (typique des bots réguliers)")

    score = min(score, 10.0)

    if verbose:
        label = "🤖 BOT" if score >= 4 else "👤 HUMAIN"
        print(f"\n{label}  (score={score:.1f}/10)")
        for r in reasons:
            print(f"  • {r}")

    return score, reasons


# ── Full analysis pipeline ─────────────────────────────────────────────────────

def analyze_user(timestamps, verbose=False):
    signal  = build_signal(timestamps)
    freqs, power = compute_fft(signal)
    iets    = compute_iets(timestamps)

    fft_feats  = fft_peak_features(power)
    temp_feats = temporal_features(iets, n_events=len(timestamps))
    spec_ent   = spectral_entropy(power)

    score, reasons = bot_score(fft_feats, temp_feats, spec_ent, verbose=verbose)

    return {
        "signal":     signal,
        "freqs":      freqs,
        "power":      power,
        "iets":       iets,          # ← stocké proprement ici
        "fft_features":      fft_feats,
        "temporal_features": temp_feats,
        "spectral_entropy":  spec_ent,
        "bot_score":  score,
        "reasons":    reasons,
    }


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_analysis(user_id, result):
    score = result["bot_score"]
    label = "🤖 BOT" if score >= 4 else "👤 HUMAIN"

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"User {user_id}  —  score bot : {score:.1f}/10  ({label})",
        fontsize=14, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Activité temporelle
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(result["signal"], lw=0.6, color="#378ADD")
    ax1.set_title("Activité sur la journée (domaine temporel)")
    ax1.set_xlabel("Seconde de la journée")
    ax1.set_ylabel("Événements / s")

    # 2. Spectre FFT
    ax2 = fig.add_subplot(gs[0, 2])
    freqs, power = result["freqs"], result["power"]
    ax2.plot(freqs[1:], power[1:], lw=0.8, color="#D85A30")
    ax2.set_title("Spectre FFT")
    ax2.set_xlabel("Fréquence (Hz)")
    ax2.set_ylabel("Puissance")

    # 3. Distribution des IETs
    ax3 = fig.add_subplot(gs[1, :2])
    iets = result["iets"]
    if len(iets) > 0:
        ax3.hist(iets, bins=100, color="#1D9E75", edgecolor="none", log=True)
        ax3.set_title(
            f"Distribution des inter-event times  "
            f"(CV={result['temporal_features']['iet_cv']:.2f})"
        )
        ax3.set_xlabel("Intervalle entre événements (s)")
        ax3.set_ylabel("Fréquence (log)")
    else:
        ax3.text(0.5, 0.5, "Pas assez d'événements", ha="center", va="center",
                 transform=ax3.transAxes)
        ax3.set_title("Distribution des inter-event times")

    # 4. Résumé textuel du score
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    lines = [f"Score : {score:.1f} / 10", ""]
    if result["reasons"]:
        lines += [f"• {r}" for r in result["reasons"]]
    else:
        lines.append("Aucun signal bot détecté")
    ax4.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax4.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#F1EFE8", alpha=0.8)
    )
    ax4.set_title("Raisons du score")

    plt.savefig(f"user_{user_id}_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure sauvegardée : user_{user_id}_analysis.png")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DAY     = 12
    USER_ID = 616942

    event, _ = get_pickle_day(DAY)
    _, timestamps = event[USER_ID]

    result = analyze_user(timestamps, verbose=True)
    plot_analysis(USER_ID, result)

    # ── Batch : top-10 utilisateurs suspects ──────────────────────────────────
    print("\n── Top-10 utilisateurs suspects ──")
    scores = []
    for uid, (_, ts) in event.items():
        r = analyze_user(ts)
        scores.append((uid, r["bot_score"], r["reasons"]))

    scores.sort(key=lambda x: -x[1])
    for uid, sc, reasons in scores[:10]:
        tag = "BOT" if sc >= 4 else "   "
        first = reasons[0] if reasons else "—"
        print(f"  [{tag}] User {uid:>8}  score={sc:.1f}  {first}")