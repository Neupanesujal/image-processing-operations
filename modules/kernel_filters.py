"""
modules/kernel_filters.py
--------------------------
Module 6 — Kernel-Based Filtering
  User enters kernel size and values manually.
  scipy.ndimage.convolve applies the kernel with constant (zero) border.

  Typical use cases:
    - Gaussian smoothing
    - Laplacian sharpening
    - Sobel edge detection (Gx / Gy / combined)
    - Custom edge detection
    - Image enhancement

  Display modes for output:
    - clip   → [0,255]          for smoothing / enhancement
    - abs    → |result|         for edge maps (normalised)
    - signed → result + 128     for Laplacian / signed kernels
"""

import numpy as np
from scipy.ndimage import convolve
from core.display_utils import show_images


# ──────────────────────────────────────────────
# Preset kernels  (shown as guidance, not forced)
# ──────────────────────────────────────────────

PRESETS = {
    "gaussian_3x3": np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float64) / 16,

    "laplacian_4conn": np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
    ], dtype=np.float64),

    "laplacian_8conn": np.array([
        [ 1,  1,  1],
        [ 1, -8,  1],
        [ 1,  1,  1]
    ], dtype=np.float64),

    "sobel_x": np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64),

    "sobel_y": np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64),

    "sharpen": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float64),
}


def _print_presets():
    print("\n  Reference presets (enter values manually below):")
    for name, k in PRESETS.items():
        print(f"\n  [{name}]")
        for row in k:
            print("   ", "  ".join(f"{v:6.3f}" for v in row))
    print()


def _get_kernel_from_user():
    """
    Prompt user for kernel size then values row-by-row.
    Returns np.ndarray of shape (n, n) float64.
    """
    while True:
        try:
            n = int(input("  Kernel size (odd integer, e.g. 3, 5): "))
            if n % 2 == 0:
                print("  Must be odd."); continue
            if n < 1:
                raise ValueError
            break
        except ValueError:
            print("  Enter a positive odd integer.")

    print(f"\n  Enter {n}×{n} kernel values row by row.")
    print(f"  Each row: {n} space-separated numbers.")
    rows = []
    for i in range(n):
        while True:
            try:
                vals = list(map(float, input(f"  Row {i+1}: ").split()))
                if len(vals) != n:
                    print(f"  Need exactly {n} values."); continue
                rows.append(vals)
                break
            except ValueError:
                print("  Enter numbers only.")

    return np.array(rows, dtype=np.float64)


def _apply_kernel(image, kernel, display_mode="clip"):
    """
    Apply kernel via scipy.ndimage.convolve with constant (zero) border.

    display_mode:
      'clip'   → clip to [0,255]         (smoothing, enhancement)
      'abs'    → |result| normalised     (edge maps)
      'signed' → result + 128 clipped    (Laplacian, signed display)
    """
    result = convolve(image.astype(np.float64), kernel, mode="constant", cval=0)

    if display_mode == "abs":
        result = np.abs(result)
        if result.max() > 0:
            result = result / result.max() * 255
    elif display_mode == "signed":
        result = result + 128

    return np.clip(result, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# Module runner
# ──────────────────────────────────────────────

def run(image):
    while True:
        print("\n── Kernel-Based Filtering ─────────────")
        print("  1. Enter custom kernel")
        print("  2. Sobel edge detection (Gx, Gy, magnitude)")
        print("  p. Print preset kernel reference")
        print("  0. Back")
        choice = input("Select: ").strip().lower()

        if choice == "0":
            break

        elif choice == "p":
            _print_presets()

        elif choice == "1":
            _print_presets()
            print("  What kind of operation is this kernel for?")
            print("  1. Smoothing / Enhancement  (clip output)")
            print("  2. Edge detection           (|output|, normalised)")
            print("  3. Laplacian / Signed       (output + 128)")
            mode_map = {"1": "clip", "2": "abs", "3": "signed"}
            mode_in  = input("  Select display mode [1/2/3] (default 1): ").strip()
            display_mode = mode_map.get(mode_in, "clip")

            kernel = _get_kernel_from_user()
            result = _apply_kernel(image, kernel, display_mode)
            show_images(
                [image, result],
                ["Original", f"Kernel filtered  ({display_mode} mode)"]
            )

        elif choice == "2":
            gx  = _apply_kernel(image, PRESETS["sobel_x"], "abs")
            gy  = _apply_kernel(image, PRESETS["sobel_y"], "abs")

            mag = np.sqrt(
                convolve(image.astype(np.float64), PRESETS["sobel_x"], mode="constant", cval=0)**2 +
                convolve(image.astype(np.float64), PRESETS["sobel_y"], mode="constant", cval=0)**2
            )
            mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)

            show_images(
                [image, gx, gy, mag],
                ["Original", "Sobel Gx", "Sobel Gy", "Gradient Magnitude"]
            )

        else:
            print("Invalid choice.")
