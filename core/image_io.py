"""
core/image_io.py
----------------
Handles image loading and validation.
Always returns a grayscale uint8 numpy array.
"""

import numpy as np
from skimage import io, color


def load_image(path):
    """
    Load a grayscale image from disk.

    Parameters
    ----------
    path : str  — full or relative path to the image file

    Returns
    -------
    np.ndarray  shape (H, W), dtype uint8, values in [0, 255]
    """
    image = io.imread(path)
    if image.ndim == 3:
        image = color.rgb2gray(image)           # → float [0, 1]
        image = (image * 255).astype(np.uint8)
    elif image.ndim == 2:
        image = image.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    _validate(image)
    return image


def _validate(image):
    """Basic sanity checks."""
    if image.ndim != 2:
        raise ValueError("Image must be 2D grayscale.")
    if image.dtype != np.uint8:
        raise TypeError("Image dtype must be uint8.")
    if image.size == 0:
        raise ValueError("Image is empty.")
