# Image-processing-operations

A comprehensive, modular image processing system built in Python for academic use in **Image Processing**. The system supports a wide range of operations — from basic intensity transforms to full frequency domain filtering — all through a clean terminal menu interface.

---

## 📁 Project Structure

```
image_processing_system/
├── main.py                  ← Entry point, menu controller
├── requirements.txt         ← Python dependencies
│
├── core/                    ← Shared utilities (imported by all modules)
│   ├── __init__.py
│   ├── image_io.py          ← Image loading & validation
│   ├── display_utils.py     ← All matplotlib display functions
│   └── fft_engine.py        ← FFT pipeline & frequency mask builders
│
└── modules/                 ← Processing modules (each self-contained)
    ├── __init__.py
    ├── image_info.py         ← Metadata, grid, histogram, intensity profile
    ├── intensity_transforms.py  ← Negative, gamma, contrast, histogram EQ
    ├── bit_processing.py     ← Bit plane slicing
    ├── geometric_transforms.py  ← Zoom, shrink, mirror, crop
    ├── spatial_filters.py    ← Median, max, min, midpoint, alpha-trimmed mean
    ├── kernel_filters.py     ← Custom kernel input, Sobel edge detection
    └── frequency_domain.py   ← FFT filters, Butterworth, Ideal vs Gaussian
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/image_processing_system.git
cd image_processing_system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run**
```bash
python main.py
```

---

## 🚀 Usage

On launch, a **file dialog window** opens automatically — browse and select any image file from your system (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`). The image is converted to grayscale automatically if it is in color.

You are then presented with the main menu:

```

  ┌─────────────────────────────────────────────┐
  │  MAIN MENU                                  │
  │  1.  Image Information                      │
  │  2.  Intensity Transformations              │
  │  3.  Bit-Level Processing                   │
  │  4.  Geometric Transformations              │
  │  5.  Spatial Filtering                      │
  │  6.  Kernel-Based Filtering                 │
  │  7.  Frequency Domain Processing            │
  │  r.  Reload / change image                  │
  │  0.  Exit                                   │
  └─────────────────────────────────────────────┘
```

Press `r` at any time to open the file dialog again and load a different image.

---

## 🧩 Modules

### 1. Image Information
Inspect your image before processing.

| Option | Description |
|--------|-------------|
| Print metadata | Width, height, total pixels, min/max intensity, mean, std deviation |
| Grid overlay | Image displayed with red pixel-index grid at configurable intervals |
| Histogram | Bar chart of intensity distribution |
| Cumulative histogram | CDF of pixel intensities |
| Intensity profile | Pixel intensity plotted along a selected row or column |

---

### 2. Intensity Transformations

| Operation | Details |
|-----------|---------|
| Image Negative | `s = 255 - r` |
| Binarization | User provides threshold value |
| Power-Law (Gamma) | `s = c × r^γ` — user provides γ and c |
| Contrast Stretching | Linear stretch between user-defined or auto min/max |
| Histogram Equalization | See details below |

#### Histogram Equalization — 5 Cases

| Case | What happens |
|------|--------------|
| **Original** | Equalization applied directly to the loaded image |
| **Bright** | Pixel values shifted **+80** first, then equalized |
| **Dark** | Pixel values shifted **−80** first, then equalized |
| **Low Contrast** | Image stretched to a narrow range (100–160), then equalized |
| **High Contrast** | Image amplified, then equalized |

> For each case, both the **before/after image pair** and **before/after histograms** are displayed all at once just to compare them.

---

### 3. Bit-Level Processing

Extracts and displays individual **bit planes** (0 = LSB to 7 = MSB) of the image as binary (black and white) images.

- **All 8 planes** displayed in a 2×4 grid
- **Single plane** selection with LSB/MSB label

---

### 4. Geometric Transformations

| Operation | Details |
|-----------|---------|
| Zoom | User provides factor (e.g. `2.0`). Bilinear interpolation, implemented manually |
| Shrink | User provides factor (e.g. `2.0` = half size) |
| Mirror | Horizontal, vertical, or both axes |
| Crop | User provides x, y, width, height — clamped to image bounds |

---

### 5. Spatial Filtering

All filters operate on a user-defined **odd mask size** (e.g. 3, 5, 7).

| Filter | Description |
|--------|-------------|
| Median | Replaces each pixel with the median of its neighbourhood |
| Max | Replaces with the maximum value (dilation effect) |
| Min | Replaces with the minimum value (erosion effect) |
| Midpoint | Average of max and min in the neighbourhood |
| Alpha-Trimmed Mean | Trims the lowest and highest `α/2` fraction before averaging |

---

### 6. Kernel-Based Filtering

The user **manually enters** any kernel size and values. The kernel is applied via `scipy.ndimage.convolve`.

Three display modes to choose from:

| Mode | Use for |
|------|---------|
| **Clip** | Smoothing, enhancement kernels |
| **Abs** | Edge detection (normalised absolute value) |
| **Signed** | Laplacian / kernels with negative outputs (output + 128) |

**Reference presets** are printed before input (Gaussian, Laplacian 4-conn, Laplacian 8-conn, Sobel X, Sobel Y, Sharpen) so you can copy known values directly.

A dedicated **Sobel option** automatically computes Gx, Gy, and gradient magnitude, displaying all four panels at once.

---

### 7. Frequency Domain Processing

All operations display a **4-panel result**: `[Original] [Mask] [Filtered Spectrum] [Reconstructed]`

| Option | User Input |
|--------|-----------|
| Ideal Low-Pass | Cutoff D₀ |
| Ideal High-Pass | Cutoff D₀ |
| Ideal Band-Pass | Inner and outer cutoff |
| Ideal Band-Stop (Notch) | Inner and outer cutoff |
| Gaussian Low-Pass | σ |
| Gaussian High-Pass | σ |
| Laplacian | None |
| Butterworth Low-Pass | D₀ and order n |
| Butterworth High-Pass | D₀ and order n |
| **Ideal vs Gaussian Comparison** | Cutoff — shows ringing/Gibbs effect on ideal filter |
| **Butterworth Order Comparison** | D₀ and n — always shows n=1, n=2, and your n side by side |
| FFT Spectrum only | None — shows log-scaled magnitude |

---

## 🏗️ Architecture

The system is built in **three layers** to maximise code reuse:

```
main.py
  └── modules/*
        ├── core/display_utils   ← all modules use this
        ├── core/fft_engine      ← frequency module only
        └── core/image_io        ← loaded once in main
```

**Key design rules:**
- No module imports another module — all shared logic lives in `core/` only
- `show_images()` handles 2, 3, 4, or 8 panels with one call
- `plot_histogram()` is reusable — called from any module that needs it
- All filter outputs are clipped and cast: `np.clip(result, 0, 255).astype(np.uint8)`
- Border handling uses `mode="constant", cval=0` — you see the real mathematical behaviour at edges

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, transforms |
| `scipy` | Spatial filters, convolution |
| `matplotlib` | All display and plotting |
| `scikit-image` | Image I/O and color conversion |

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🖥️ Platform Notes

This system uses `tkinter` for the file dialog. It is included with most Python installations.

If you see an error like `_tkinter not found`, install it with:
```bash
# Ubuntu / Debian
sudo apt install python3-tk

# Fedora
sudo dnf install python3-tkinter
```

If matplotlib shows a `FigureCanvasAgg is non-interactive` warning, make sure the backend in `main.py` matches what is available on your system:

```python
import matplotlib
matplotlib.use("TkAgg")   # or Qt5Agg, Qt6Agg, WXAgg
```

To check available backends on your machine:
```bash
python -c "import matplotlib.rcsetup; print(matplotlib.rcsetup.all_backends)"
```

---

## 📄 License

This project is intended for academic and educational use.