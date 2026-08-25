"""
Linear Frequency Cepstral Coefficients (LFCC) Feature Extraction.
LFCCs are the gold standard acoustic feature for ASVspoof speech anti-spoofing
and deepfake voice detection because they preserve high-frequency vocoder phase artifacts.
"""

import numpy as np
import scipy.fftpack as fftpack
from scipy.signal import get_window

def linear_filter_bank(
    n_filters: int = 20,
    n_fft: int = 512,
    sr: int = 16000,
    f_min: float = 0.0,
    f_max: float = 8000.0
) -> np.ndarray:
    """
    Constructs linearly spaced triangular filter banks across the spectrum.
    """
    f_max = min(f_max, sr / 2.0)
    linear_freqs = np.linspace(f_min, f_max, n_filters + 2)
    fft_bin_freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)

    filter_bank = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)

    for i in range(n_filters):
        left = linear_freqs[i]
        center = linear_freqs[i + 1]
        right = linear_freqs[i + 2]

        up_idx = np.where((fft_bin_freqs >= left) & (fft_bin_freqs <= center))[0]
        if center > left:
            filter_bank[i, up_idx] = (fft_bin_freqs[up_idx] - left) / (center - left)

        down_idx = np.where((fft_bin_freqs >= center) & (fft_bin_freqs <= right))[0]
        if right > center:
            filter_bank[i, down_idx] = (right - fft_bin_freqs[down_idx]) / (right - center)

    return filter_bank

def compute_deltas(feat: np.ndarray, width: int = 5) -> np.ndarray:
    """
    Calculates delta derivatives along the temporal axis.
    """
    n_feats, n_frames = feat.shape
    if n_frames < width:
        return np.zeros_like(feat)
    
    half_w = width // 2
    padded = np.pad(feat, ((0, 0), (half_w, half_w)), mode='edge')
    deltas = np.zeros_like(feat)
    
    denom = 2 * sum(i**2 for i in range(1, half_w + 1))
    for i in range(1, half_w + 1):
        deltas += i * (padded[:, half_w + i : half_w + i + n_frames] - padded[:, half_w - i : half_w - i + n_frames])
    return deltas / denom

def extract_lfcc(
    y: np.ndarray,
    sr: int = 16000,
    n_lfcc: int = 20,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    f_min: float = 0.0,
    f_max: float = 8000.0,
    with_deltas: bool = True
) -> np.ndarray:
    """
    Extracts LFCC + Delta + Delta-Delta matrix.
    Shape: (3 * n_lfcc, n_frames) if with_deltas else (n_lfcc, n_frames)
    """
    if len(y) < win_length:
        y = np.pad(y, (0, win_length - len(y)), mode='constant')

    window = get_window('hamming', win_length, fftbins=True)
    pad_width = (n_fft - win_length) // 2
    
    frames = []
    for start in range(0, len(y) - win_length + 1, hop_length):
        chunk = y[start : start + win_length] * window
        chunk_padded = np.pad(chunk, (pad_width, n_fft - win_length - pad_width), mode='constant')
        fft_mag = np.abs(np.fft.rfft(chunk_padded, n=n_fft))
        frames.append(fft_mag ** 2)

    if len(frames) == 0:
        return np.zeros((3 * n_lfcc if with_deltas else n_lfcc, 1), dtype=np.float32)

    spectrogram = np.array(frames).T
    fb = linear_filter_bank(n_filters=n_lfcc, n_fft=n_fft, sr=sr, f_min=f_min, f_max=f_max)
    filter_energy = np.dot(fb, spectrogram)
    filter_energy = np.maximum(filter_energy, 1e-10)
    log_energy = np.log(filter_energy)

    lfcc_base = fftpack.dct(log_energy, type=2, axis=0, norm='ortho')[:n_lfcc, :]

    if not with_deltas:
        return lfcc_base.astype(np.float32)

    delta1 = compute_deltas(lfcc_base)
    delta2 = compute_deltas(delta1)

    lfcc_full = np.concatenate([lfcc_base, delta1, delta2], axis=0)
    return lfcc_full.astype(np.float32)
