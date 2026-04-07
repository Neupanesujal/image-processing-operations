"""
core/fft_engine.py
------------------
Centralised FFT pipeline.
All frequency-domain operations in frequency_domain.py call these functions.
"""

import numpy as np


# ──────────────────────────────────────────────
# FFT / IFFT helpers
# ──────────────────────────────────────────────

def compute_fft(image):
    """
    Compute the 2-D FFT of a grayscale image.

    Returns
    -------
    fshift      : complex array, zero-frequency at centre (fftshift applied)
    magnitude   : log-scaled magnitude spectrum, normalised to [0, 255]
    """
    f       = np.fft.fft2(image.astype(np.float64))
    fshift  = np.fft.fftshift(f)
    mag_raw = np.log1p(np.abs(fshift))
    magnitude = (mag_raw / mag_raw.max() * 255).astype(np.uint8)
    return fshift, magnitude


def apply_mask_and_ifft(fshift, mask):
    """
    Multiply the shifted FFT by a mask then inverse-transform.

    Parameters
    ----------
    fshift : complex array (fftshift-ed)
    mask   : real array in [0, 1], same shape as fshift

    Returns
    -------
    filtered_magnitude : log-scaled magnitude of the filtered spectrum [0,255]
    reconstructed      : spatial-domain result, clipped to uint8 [0,255]
    """
    filtered        = fshift * mask
    mag_raw         = np.log1p(np.abs(filtered))
    filtered_magnitude = (mag_raw / (mag_raw.max() + 1e-9) * 255).astype(np.uint8)

    f_ishift        = np.fft.ifftshift(filtered)
    img_back        = np.fft.ifft2(f_ishift)
    reconstructed   = np.abs(img_back)
    reconstructed   = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return filtered_magnitude, reconstructed


# ──────────────────────────────────────────────
# Distance map (reused by all mask builders)
# ──────────────────────────────────────────────

def _distance_map(shape):
    """Return array D where D[r,c] = Euclidean distance from centre."""
    rows, cols = shape
    cr, cc     = rows // 2, cols // 2
    r = np.arange(rows) - cr
    c = np.arange(cols) - cc
    C, R       = np.meshgrid(c, r)
    return np.sqrt(R**2 + C**2)


# ──────────────────────────────────────────────
# Mask builders
# ──────────────────────────────────────────────

def build_ideal_lp_mask(shape, cutoff):
    """Ideal low-pass: 1 inside circle of radius cutoff, 0 outside."""
    D = _distance_map(shape)
    return (D <= cutoff).astype(np.float64)


def build_ideal_hp_mask(shape, cutoff):
    return 1.0 - build_ideal_lp_mask(shape, cutoff)


def build_ideal_bp_mask(shape, d_low, d_high):
    """Band-pass: pass frequencies between d_low and d_high."""
    D = _distance_map(shape)
    return ((D >= d_low) & (D <= d_high)).astype(np.float64)


def build_ideal_bs_mask(shape, d_low, d_high):
    """Band-stop (notch): reject frequencies between d_low and d_high."""
    return 1.0 - build_ideal_bp_mask(shape, d_low, d_high)


def build_gaussian_lp_mask(shape, cutoff):
    """Gaussian low-pass: H = exp(-D² / 2σ²)."""
    D = _distance_map(shape)
    return np.exp(-(D**2) / (2 * cutoff**2))


def build_gaussian_hp_mask(shape, cutoff):
    return 1.0 - build_gaussian_lp_mask(shape, cutoff)


def build_laplacian_mask(shape):
    """
    Laplacian in frequency domain: H = -4π²D²  (sharpening operator).
    Returned normalised to [0,1] for display; sign kept for processing.
    """
    D   = _distance_map(shape)
    H   = -(4 * np.pi**2) * (D**2)
    # Shift so minimum is 0, normalise to [0,1]
    H   = H - H.min()
    H   = H / (H.max() + 1e-9)
    return H


def build_butterworth_lp_mask(shape, cutoff, order):
    """Butterworth low-pass: H = 1 / (1 + (D/D₀)^(2n))."""
    D = _distance_map(shape)
    return 1.0 / (1.0 + (D / (cutoff + 1e-9)) ** (2 * order))


def build_butterworth_hp_mask(shape, cutoff, order):
    """Butterworth high-pass: H = 1 / (1 + (D₀/D)^(2n))."""
    D = _distance_map(shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = 1.0 / (1.0 + np.where(D == 0, 1e-9, cutoff / D) ** (2 * order))
    return H
