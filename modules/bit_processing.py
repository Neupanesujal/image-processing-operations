"""
modules/bit_processing.py
--------------------------
Module 3 — Bit-Level Processing
  - Bit Plane Slicing: extract and display all 8 bit planes in one grid.
"""

import numpy as np
import matplotlib.pyplot as plt
from core.display_utils import show_images


def extract_bit_plane(image, bit):
    """
    Extract a single bit plane (0 = LSB, 7 = MSB).
    Returns a binary image with values 0 or 255.
    """
    return (((image >> bit) & 1) * 255).astype(np.uint8)


def run(image):
    while True:
        print("\n── Bit-Level Processing ───────────────")
        print("  1. Bit Plane Slicing (all 8 planes)")
        print("  2. Single bit plane")
        print("  0. Back")
        choice = input("Select: ").strip()

        if choice == "1":
            planes = [extract_bit_plane(image, b) for b in range(8)]
            titles = [f"Bit Plane {b}  ({'MSB' if b==7 else 'LSB' if b==0 else ''})"
                      for b in range(8)]

            # 2×4 grid layout for all 8 planes
            fig, axes = plt.subplots(2, 4, figsize=(18, 9))
            for ax, plane, title in zip(axes.flat, planes, titles):
                ax.imshow(plane, cmap="gray", vmin=0, vmax=255)
                ax.set_title(title, fontsize=11)
                ax.axis("off")
            plt.suptitle("Bit Plane Slicing — All 8 Planes", fontsize=14)
            plt.tight_layout()
            plt.show()

        elif choice == "2":
            try:
                bit = int(input("Bit plane (0=LSB … 7=MSB): "))
                if bit < 0 or bit > 7:
                    raise ValueError
            except ValueError:
                print("Enter an integer between 0 and 7."); continue
            plane = extract_bit_plane(image, bit)
            show_images([image, plane], ["Original", f"Bit Plane {bit}"])

        elif choice == "0":
            break
        else:
            print("Invalid choice.")
