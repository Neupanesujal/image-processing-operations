"""
modules/frequency_domain.py
-----------------------------
Module 7 — Frequency Domain Processing

Filters available:
  1.  Ideal Low-Pass
  2.  Ideal High-Pass
  3.  Ideal Band-Pass
  4.  Ideal Band-Stop (Notch)
  5.  Gaussian Low-Pass
  6.  Gaussian High-Pass
  7.  Laplacian (frequency domain)
  8.  Butterworth Low-Pass   (user enters order n)
  9.  Butterworth High-Pass  (user enters order n)
  10. Ideal vs Gaussian Comparison
  11. Butterworth Order Comparison

Every filter uses the 4-panel display:
  [Original] [Mask] [Filtered Spectrum] [Reconstructed]
"""

import numpy as np
from core.fft_engine import (
    compute_fft,
    apply_mask_and_ifft,
    build_ideal_lp_mask,
    build_ideal_hp_mask,
    build_ideal_bp_mask,
    build_ideal_bs_mask,
    build_gaussian_lp_mask,
    build_gaussian_hp_mask,
    build_laplacian_mask,
    build_butterworth_lp_mask,
    build_butterworth_hp_mask,
)
from core.display_utils import (
    display_fft_result,
    display_ideal_vs_gaussian,
    display_butterworth_orders,
    plot_histogram,
)


# ──────────────────────────────────────────────
# Helper: normalise mask to [0,255] for display
# ──────────────────────────────────────────────

def _mask_display(mask):
    m = mask - mask.min()
    if m.max() > 0:
        m = m / m.max()
    return (m * 255).astype(np.uint8)


# ──────────────────────────────────────────────
# Generic single-filter pipeline
# ──────────────────────────────────────────────

def _run_filter(image, mask, filter_name):
    fshift, magnitude          = compute_fft(image)
    filtered_mag, reconstructed = apply_mask_and_ifft(fshift, mask)
    display_fft_result(
        image,
        _mask_display(mask),
        filtered_mag,
        reconstructed,
        filter_name=filter_name,
    )


# ──────────────────────────────────────────────
# Input helpers
# ──────────────────────────────────────────────

def _ask_cutoff(label="Cutoff frequency D₀"):
    while True:
        try:
            val = float(input(f"  {label} (pixels, e.g. 30): "))
            if val <= 0:
                raise ValueError
            return val
        except ValueError:
            print("  Enter a positive number.")


def _ask_order():
    while True:
        try:
            n = int(input("  Butterworth order n (e.g. 1, 2, 4): "))
            if n < 1:
                raise ValueError
            return n
        except ValueError:
            print("  Enter a positive integer.")


# ──────────────────────────────────────────────
# Module runner
# ──────────────────────────────────────────────

def run(image):
    shape = image.shape

    while True:
        print("\n── Frequency Domain Processing ────────")
        print("  1.  Ideal Low-Pass filter")
        print("  2.  Ideal High-Pass filter")
        print("  3.  Ideal Band-Pass filter")
        print("  4.  Ideal Band-Stop (Notch) filter")
        print("  5.  Gaussian Low-Pass filter")
        print("  6.  Gaussian High-Pass filter")
        print("  7.  Laplacian (frequency domain)")
        print("  8.  Butterworth Low-Pass filter")
        print("  9.  Butterworth High-Pass filter")
        print("  10. Ideal vs Gaussian Comparison")
        print("  11. Butterworth Order Comparison")
        print("  12. View FFT magnitude spectrum only")
        print("  0.  Back")
        choice = input("Select: ").strip()

        # ── 1. Ideal LP ──────────────────────────
        if choice == "1":
            d0   = _ask_cutoff()
            mask = build_ideal_lp_mask(shape, d0)
            _run_filter(image, mask, f"Ideal Low-Pass  D₀={d0}")

        # ── 2. Ideal HP ──────────────────────────
        elif choice == "2":
            d0   = _ask_cutoff()
            mask = build_ideal_hp_mask(shape, d0)
            _run_filter(image, mask, f"Ideal High-Pass  D₀={d0}")

        # ── 3. Ideal BP ──────────────────────────
        elif choice == "3":
            print("  Band-Pass: pass frequencies BETWEEN d_low and d_high")
            d_low  = _ask_cutoff("Inner cutoff d_low")
            d_high = _ask_cutoff("Outer cutoff d_high")
            if d_high <= d_low:
                print("  d_high must be > d_low."); continue
            mask = build_ideal_bp_mask(shape, d_low, d_high)
            _run_filter(image, mask, f"Ideal Band-Pass  [{d_low}, {d_high}]")

        # ── 4. Ideal BS ──────────────────────────
        elif choice == "4":
            print("  Band-Stop: REJECT frequencies BETWEEN d_low and d_high")
            d_low  = _ask_cutoff("Inner cutoff d_low")
            d_high = _ask_cutoff("Outer cutoff d_high")
            if d_high <= d_low:
                print("  d_high must be > d_low."); continue
            mask = build_ideal_bs_mask(shape, d_low, d_high)
            _run_filter(image, mask, f"Ideal Band-Stop  [{d_low}, {d_high}]")

        # ── 5. Gaussian LP ───────────────────────
        elif choice == "5":
            sigma = _ask_cutoff("Gaussian σ (cutoff)")
            mask  = build_gaussian_lp_mask(shape, sigma)
            _run_filter(image, mask, f"Gaussian Low-Pass  σ={sigma}")

        # ── 6. Gaussian HP ───────────────────────
        elif choice == "6":
            sigma = _ask_cutoff("Gaussian σ (cutoff)")
            mask  = build_gaussian_hp_mask(shape, sigma)
            _run_filter(image, mask, f"Gaussian High-Pass  σ={sigma}")

        # ── 7. Laplacian ─────────────────────────
        elif choice == "7":
            mask = build_laplacian_mask(shape)
            _run_filter(image, mask, "Laplacian (Frequency Domain)")

        # ── 8. Butterworth LP ────────────────────
        elif choice == "8":
            d0    = _ask_cutoff()
            n     = _ask_order()
            mask  = build_butterworth_lp_mask(shape, d0, n)
            _run_filter(image, mask, f"Butterworth Low-Pass  D₀={d0}, n={n}")

        # ── 9. Butterworth HP ────────────────────
        elif choice == "9":
            d0    = _ask_cutoff()
            n     = _ask_order()
            mask  = build_butterworth_hp_mask(shape, d0, n)
            _run_filter(image, mask, f"Butterworth High-Pass  D₀={d0}, n={n}")

        # ── 10. Ideal vs Gaussian comparison ─────
        elif choice == "10":
            print("\n  Compare Ideal vs Gaussian filter")
            print("  1. Low-Pass   2. High-Pass")
            sub = input("  [1/2] (default 1): ").strip()
            kind = "High-Pass" if sub == "2" else "Low-Pass"
            d0   = _ask_cutoff()

            if kind == "Low-Pass":
                ideal_mask    = build_ideal_lp_mask(shape, d0)
                gaussian_mask = build_gaussian_lp_mask(shape, d0)
            else:
                ideal_mask    = build_ideal_hp_mask(shape, d0)
                gaussian_mask = build_gaussian_hp_mask(shape, d0)

            fshift, _ = compute_fft(image)
            _, ideal_recon    = apply_mask_and_ifft(fshift, ideal_mask)
            _, gaussian_recon = apply_mask_and_ifft(fshift, gaussian_mask)

            display_ideal_vs_gaussian(
                image, ideal_recon, gaussian_recon,
                cutoff=d0, kind=kind
            )

        # ── 11. Butterworth order comparison ─────
        elif choice == "11":
            print("\n  Butterworth Order Comparison")
            print("  1. Low-Pass   2. High-Pass")
            sub  = input("  [1/2] (default 1): ").strip()
            kind = "High-Pass" if sub == "2" else "Low-Pass"
            d0   = _ask_cutoff()
            n    = _ask_order()

            fshift, _ = compute_fft(image)
            orders     = sorted(set([1, 2, n]))   # always show n=1, n=2, user's n

            results = []
            for order in orders:
                if kind == "Low-Pass":
                    mask = build_butterworth_lp_mask(shape, d0, order)
                else:
                    mask = build_butterworth_hp_mask(shape, d0, order)
                _, recon = apply_mask_and_ifft(fshift, mask)
                results.append((order, recon))

            display_butterworth_orders(image, results, cutoff=d0, kind=kind)

        # ── 12. FFT spectrum only ─────────────────
        elif choice == "12":
            fshift, magnitude = compute_fft(image)
            from core.display_utils import show_images
            show_images(
                [image, magnitude],
                ["Original", "FFT Magnitude Spectrum (log scale)"],
                cmap="gray"
            )

        elif choice == "0":
            break
        else:
            print("Invalid choice.")
