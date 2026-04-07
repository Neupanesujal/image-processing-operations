

from tkinter import Tk, filedialog
import sys
from core.image_io import load_image
from modules import (
    image_info,
    intensity_transforms,
    bit_processing,
    geometric_transforms,
    spatial_filters,
    kernel_filters,
    frequency_domain,
)



MENU = """
  ┌─────────────────────────────────────────────┐
  │  MAIN MENU                                  │
  ├─────────────────────────────────────────────┤
  │  1.  Image Information                      │
  │  2.  Intensity Transformations              │
  │  3.  Bit-Level Processing                   │
  │  4.  Geometric Transformations              │
  │  5.  Spatial Filtering                      │
  │  6.  Kernel-Based Filtering                 │
  │  7.  Frequency Domain Processing            │
  │  r.  Reload / change image                  │
  │  0.  Exit                                   │
  └─────────────────────────────────────────────┘"""

MODULE_MAP = {
    "1": image_info,
    "2": intensity_transforms,
    "3": bit_processing,
    "4": geometric_transforms,
    "5": spatial_filters,
    "6": kernel_filters,
    "7": frequency_domain,
}


def load_with_dialog():
    while True:
        try:
            root = Tk()
            root.withdraw()

            path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                    ("All files", "*.*")
                ]
            )

            # If user cancels → exit program
            if not path:
                print("  No file selected. Exiting program.")
                sys.exit()

            image = load_image(path)
            print(f"  Loaded: {path}  ({image.shape[1]}×{image.shape[0]} px)\n")
            return image

        except FileNotFoundError:
            print(f"  File not found: '{path}'. Please try again.")
        except Exception as e:
            print(f"  Could not load image: {e}. Please try another file.")


def main():

    print("  Load Image ")
    image = load_with_dialog()

    while True:
        print(MENU)
        choice = input("  Select: ").strip().lower()

        if choice == "0":
            print("\n  Goodbye!\n")
            break

        elif choice == "r":
            print("   Reload Image ")
            image = load_with_dialog()

        elif choice in MODULE_MAP:
            MODULE_MAP[choice].run(image)

        else:
            print("  Invalid choice, try again.")


if __name__ == "__main__":
    main()
