"""
modules/geometric_transforms.py
--------------------------------
Module 4 — Geometric Transformations
  1. Zooming  (bilinear interpolation)
  2. Shrinking
  3. Mirror   (horizontal / vertical / both)
  4. Cropping
"""

import numpy as np
from core.display_utils import show_images


# ──────────────────────────────────────────────
# Transform functions
# ──────────────────────────────────────────────

def zoom(image, factor):
    """
    Zoom by `factor` using bilinear interpolation.
    factor > 1 enlarges, factor < 1 shrinks (use shrink() for that).
    """
    h, w      = image.shape
    new_h     = int(h * factor)
    new_w     = int(w * factor)

    # Output pixel grid mapped back to input coordinates
    row_idx   = np.linspace(0, h - 1, new_h)
    col_idx   = np.linspace(0, w - 1, new_w)
    col_grid, row_grid = np.meshgrid(col_idx, row_idx)

    return _bilinear_interp(image, row_grid, col_grid, new_h, new_w)


def shrink(image, factor):
    """Shrink by `factor` (factor > 1 shrinks).  Uses bilinear interpolation."""
    return zoom(image, 1.0 / factor)


def _bilinear_interp(image, row_f, col_f, new_h, new_w):
    """Core bilinear interpolation on floating-point coordinate grids."""
    h, w  = image.shape
    r0    = np.clip(np.floor(row_f).astype(int), 0, h - 1)
    c0    = np.clip(np.floor(col_f).astype(int), 0, w - 1)
    r1    = np.clip(r0 + 1, 0, h - 1)
    c1    = np.clip(c0 + 1, 0, w - 1)

    dr    = row_f - r0
    dc    = col_f - c0

    img   = image.astype(np.float64)
    result = (img[r0, c0] * (1 - dr) * (1 - dc) +
              img[r0, c1] * (1 - dr) *      dc   +
              img[r1, c0] *      dr  * (1 - dc) +
              img[r1, c1] *      dr  *      dc)

    return np.clip(result, 0, 255).astype(np.uint8).reshape(new_h, new_w)


def mirror(image, direction="horizontal"):
    """
    Mirror the image.
    direction: 'horizontal'  → flip left-right
               'vertical'    → flip top-bottom
               'both'        → flip both axes
    """
    if direction == "horizontal":
        return np.fliplr(image)
    elif direction == "vertical":
        return np.flipud(image)
    elif direction == "both":
        return np.flipud(np.fliplr(image))
    else:
        raise ValueError(f"Unknown direction: {direction}")


def crop(image, x, y, width, height):
    """
    Crop a region from the image.

    Parameters
    ----------
    x, y          : top-left corner (column, row)
    width, height : size of the crop region
    """
    h_img, w_img = image.shape
    x      = max(0, min(x,     w_img - 1))
    y      = max(0, min(y,     h_img - 1))
    width  = max(1, min(width,  w_img - x))
    height = max(1, min(height, h_img - y))
    return image[y: y + height, x: x + width]


# ──────────────────────────────────────────────
# Module runner
# ──────────────────────────────────────────────

def run(image):
    while True:
        print("\n── Geometric Transformations ──────────")
        print("  1. Zooming (bilinear interpolation)")
        print("  2. Shrinking")
        print("  3. Mirror Image")
        print("  4. Cropping")
        print("  0. Back")
        choice = input("Select: ").strip()

        if choice == "1":
            try:
                factor = float(input("Zoom factor (e.g. 2.0 = 2× larger): "))
            except ValueError:
                print("Invalid input."); continue
            result = zoom(image, factor)
            show_images([image, result],
                        ["Original", f"Zoomed ×{factor}  ({result.shape[1]}×{result.shape[0]})"])

        elif choice == "2":
            try:
                factor = float(input("Shrink factor (e.g. 2.0 = half size): "))
            except ValueError:
                print("Invalid input."); continue
            result = shrink(image, factor)
            show_images([image, result],
                        ["Original", f"Shrunk ÷{factor}  ({result.shape[1]}×{result.shape[0]})"])

        elif choice == "3":
            print("  Direction: h=horizontal  v=vertical  b=both")
            d_map  = {"h": "horizontal", "v": "vertical", "b": "both"}
            d_in   = input("  Choice [h/v/b] (default h): ").strip().lower()
            direction = d_map.get(d_in, "horizontal")
            result = mirror(image, direction)
            show_images([image, result], ["Original", f"Mirror ({direction})"])

        elif choice == "4":
            h, w = image.shape
            print(f"  Image size: {w}×{h}")
            try:
                x      = int(input("  x (left edge, 0-based): "))
                y      = int(input("  y (top edge,  0-based): "))
                width  = int(input("  width  (px): "))
                height = int(input("  height (px): "))
            except ValueError:
                print("Invalid input."); continue
            result = crop(image, x, y, width, height)
            show_images([image, result],
                        ["Original", f"Cropped  {width}×{height} @ ({x},{y})"])

        elif choice == "0":
            break
        else:
            print("Invalid choice.")
