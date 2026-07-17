from utils import get_pickle_day, build_signal
from graph import plot_signal

from scipy.ndimage import gaussian_filter1d

def smooth_signal(signal, sigma=100):
    """
    sigma en secondes.
    sigma=300 => lissage sur environ 5 minutes.
    """
    return gaussian_filter1d(signal.astype(float), sigma=sigma)

day = 12

event, _ = get_pickle_day(day)

user = 616942

_, times = event[user]

signal = build_signal(times)

signal_smooth = smooth_signal(signal)

plot_signal(signal_smooth, user, day, len(times))
