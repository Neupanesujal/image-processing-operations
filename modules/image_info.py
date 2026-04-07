"""
modules/image_info.py
---------------------
Module 1 — Image Information
Displays metadata, grid image, histogram, cumulative histogram,
and intensity profile for a loaded grayscale image.
"""

import numpy as np
from core.display_utils import (
    show_with_grid,
    plot_histogram,
    plot_cumulative_histogram,
    plot_intensity_profile,
)


def print_metadata(image):
    """Print image statistics to the console."""
    h, w    = image.shape
    total   = h * w
    mn      = int(image.min())
    mx      = int(image.max())
    mean    = float(image.mean())
    std     = float(image.std())

    print("\n" + "═" * 40)
    print("  IMAGE METADATA")
    print("═" * 40)
    print(f"  Width          : {w} px")
    print(f"  Height         : {h} px")
    print(f"  Total pixels   : {total}")
    print(f"  Min intensity  : {mn}")
    print(f"  Max intensity  : {mx}")
    print(f"  Mean intensity : {mean:.2f}")
    print(f"  Std deviation  : {std:.2f}")
    print("═" * 40 + "\n")


def run(image):
    """Interactive menu for the Image Information module."""
    while True:
        print("\n── Image Information ──────────────────")
        print("  1. Print metadata")
        print("  2. Display image with grid overlay")
        print("  3. Plot histogram")
        print("  4. Plot cumulative histogram")
        print("  0. Back to main menu")
        choice = input("Select: ").strip()

        if choice == "1":
            print_metadata(image)

        elif choice == "2":
            try:
                interval = int(input("Grid interval (default 50): ") or 50)
            except ValueError:
                interval = 50
            show_with_grid(image, interval=interval)

        elif choice == "3":
            plot_histogram(image, title="Image Histogram")

        elif choice == "4":
            plot_cumulative_histogram(image)

        # elif choice == "5":
        #     axis = input("Profile along row or column? [row/col] (default row): ").strip().lower()
        #     if axis not in ("row", "col"):
        #         axis = "row"
        #     h, w = image.shape
        #     max_idx = (h - 1) if axis == "row" else (w - 1)
        #     try:
        #         idx = int(input(f"Index (0–{max_idx}, default middle): ") or (max_idx // 2))
        #     except ValueError:
        #         idx = max_idx // 2
        #     plot_intensity_profile(image, axis=axis, index=idx)

        elif choice == "0":
            break
        else:
            print("Invalid choice.")
