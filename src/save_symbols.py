# Save each segmented symbol as an individual image file

import cv2
import os
import sys
from preprocess import binarize
from staff import detect_staff_lines, group_into_staves, group_into_systems, remove_staff_lines
from segment import filter_symbols, find_symbols


def save_symbols(output_dir="symbols"):
    os.makedirs(output_dir, exist_ok=True)

    binary = binarize(sys.argv[1])
    staff_line_rows = detect_staff_lines(binary)
    staves = group_into_staves(staff_line_rows)
    staves = group_into_systems(staves)
    cleaned = remove_staff_lines(binary, staff_line_rows)

    candidates = find_symbols(cleaned, staves)
    symbols = filter_symbols(candidates, staves, cleaned)
    print(f"Found {len(symbols)} symbols — saving to '{output_dir}/'")

    for i, s in enumerate(symbols):
        filename = (
            f"sys{s['system_index']:02d}"
            f"_staff{s['staff_index']:02d}"
            f"_x{int(s['x']):04d}"
            f"_{i:04d}.png"
        )
        cv2.imwrite(os.path.join(output_dir, filename), s["image"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python save_symbols.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "symbols"
    save_symbols(output_dir)