# Aafi Mansuri, Terry Zhen
# Symbol segmentation for OMR pipeline
# Isolates individual musical symbols from cleaned image

import cv2
import numpy as np


def find_symbols(cleaned, staves):
    """
    Find all connected components in the cleaned image and extract
    bounding boxes and metadata for each blob.
    Returns a list of raw symbol candidates.
    """
    # invert so blobs are white on black
    inverted = cv2.bitwise_not(cleaned)

    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        cx = x + w // 2  # x center
        cy = y + h // 2  # y center

        candidates.append({
            "x": cx,
            "y": cy,
            "bbox": (x, y, w, h),
            "area": area,
            "contour": contour,
        })

    return candidates


def assign_to_staff(candidate, staves):
    """
    Assign a symbol candidate to the closest staff based on y-center.
    Returns the staff index or -1 if not close to any staff.
    """
    cy = candidate["y"]
    best_staff = -1
    best_dist = float("inf")

    for i, staff in enumerate(staves):
        top_line = staff["lines"][0]
        bottom_line = staff["lines"][-1]
        staff_center = (top_line + bottom_line) / 2

        # allow some margin above and below the staff for ledger line notes
        margin = staff["spacing"] * 3
        if top_line - margin <= cy <= bottom_line + margin:
            dist = abs(cy - staff_center)
            if dist < best_dist:
                best_dist = dist
                best_staff = i

    return best_staff


def filter_symbols(candidates, staves, cleaned):
    """
    Filter out noise, bar lines, clefs, lyrics, and other non-note symbols.
    Returns a list of filtered symbol dictionaries.
    """
    if not staves:
        return []

    avg_spacing = np.mean([s["spacing"] for s in staves])
    img_h, img_w = cleaned.shape

    symbols = []

    for cand in candidates:
        x, y, w, h = cand["bbox"]

        # discard very small blobs
        if cand["area"] < avg_spacing * 2:
            continue

        # discard very large blobs
        if cand["area"] > avg_spacing * avg_spacing * 20:
            continue

        # discard very tall thin blobs
        if h > avg_spacing * 4 and w < avg_spacing:
            continue

        # assign to a staff
        staff_idx = assign_to_staff(cand, staves)

        # discard if not close to any staff
        if staff_idx == -1:
            continue

        # discard if staff is bass clef
        if staves[staff_idx].get("clef") == "bass":
            continue

        # discard symbols at far left
        staff = staves[staff_idx]
        staff_top = staff["lines"][0]
        staff_bottom = staff["lines"][-1]

        # clefs and time signatures are typically in the first 15% of the image width
        if x < img_w * 0.12:
            continue

        # discard symbols clearly above the staff
        if y + h < staff_top - avg_spacing * 2:
            continue

        # discard symbols clearly below the staff
        if y > staff_bottom + avg_spacing * 2:
            continue

        # crop the symbol from the cleaned image
        pad = 2
        crop_x = max(0, x - pad)
        crop_y = max(0, y - pad)
        crop_w = min(img_w, x + w + pad) - crop_x
        crop_h = min(img_h, y + h + pad) - crop_y
        cropped = cleaned[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        symbols.append({
            "image": cropped,
            "x": cand["x"],
            "y": cand["y"],
            "bbox": cand["bbox"],
            "staff_index": staff_idx,
            "system_index": staves[staff_idx]["system_index"],
        })

    # sort by system index, then by x position (reading order)
    symbols.sort(key=lambda s: (s["system_index"], s["x"]))

    return symbols


def draw_debug(original_binary, symbols, staves):
    """
    Draw bounding boxes around detected symbols for visual debugging.
    Green boxes for accepted symbols, staff lines shown in blue.
    """
    debug = cv2.cvtColor(original_binary, cv2.COLOR_GRAY2BGR)

    # draw staff lines in blue
    for staff in staves:
        for line_y in staff["lines"]:
            cv2.line(debug, (0, line_y), (debug.shape[1], line_y), (255, 0, 0), 1)

    # draw symbol bounding boxes in green
    for sym in symbols:
        x, y, w, h = sym["bbox"]
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return debug


if __name__ == "__main__":
    import sys
    from preprocess import binarize
    from staff import detect_staff_lines, group_into_staves, group_into_systems, remove_staff_lines

    if len(sys.argv) < 2:
        print("Usage: python segment.py <image_path>")
        sys.exit(1)

    binary = binarize(sys.argv[1])
    staff_line_rows = detect_staff_lines(binary)
    staves = group_into_staves(staff_line_rows)
    staves = group_into_systems(staves)
    cleaned = remove_staff_lines(binary, staff_line_rows)

    # segment
    candidates = find_symbols(cleaned, staves)
    print(f"Found {len(candidates)} raw candidates")

    symbols = filter_symbols(candidates, staves, cleaned)
    print(f"Filtered to {len(symbols)} symbols")

    for i, sym in enumerate(symbols):
        print(f"  Symbol {i}: x={sym['x']}, y={sym['y']}, staff={sym['staff_index']}, system={sym['system_index']}")

    # debug visualization
    debug = draw_debug(binary, symbols, staves)
    cv2.imshow("Segmentation Debug", debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()