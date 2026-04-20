# Aafi Mansuri, Terry Zhen
# Staff detection, grouping, and removal for OMR pipeline

import cv2
import numpy as np


def detect_staff_lines(binary):
    """
    Detect staff line rows using horizontal projection.
    Count the number of black pixels (0) in each row.
    Rows with counts above a threshold are staff lines.
    Returns a list of y-coordinates where staff lines exist.
    """
    # count black pixels per row (black = 0)
    h, w = binary.shape
    row_black_counts = np.sum(binary == 0, axis=1)

    # staff lines span most of the image width
    threshold = w * 0.3
    staff_line_rows = np.where(row_black_counts > threshold)[0]

    return staff_line_rows.tolist()


def group_into_staves(staff_line_rows):
    """
    Group detected staff line rows into staves of 5 lines each.
    Lines within a staff are closely spaced, gaps between staves are larger.
    Returns a list of staves, each with line y-coords and spacing.
    """
    if not staff_line_rows:
        return []

    # merge consecutive rows into single line positions
    merged_lines = []
    current_group = [staff_line_rows[0]]

    for i in range(1, len(staff_line_rows)):
        if staff_line_rows[i] - staff_line_rows[i - 1] <= 2:
            current_group.append(staff_line_rows[i])
        else:
            # take the middle row of the group as the line position
            merged_lines.append(int(np.mean(current_group)))
            current_group = [staff_line_rows[i]]

    merged_lines.append(int(np.mean(current_group)))

    # lines within a staff have small, consistent gaps
    # gaps between staves are significantly larger
    staves = []
    current_staff = [merged_lines[0]]

    for i in range(1, len(merged_lines)):
        gap = merged_lines[i] - merged_lines[i - 1]

        if len(current_staff) == 1:
            current_staff.append(merged_lines[i])
        else:
            avg_spacing = (current_staff[-1] - current_staff[0]) / (len(current_staff) - 1)

            if gap < avg_spacing * 2 and len(current_staff) < 5:
                # gap is consistent with within-staff spacing
                current_staff.append(merged_lines[i])
            else:
                # gap is too large so it will start a new staff
                staves.append(current_staff)
                current_staff = [merged_lines[i]]

    if current_staff:
        staves.append(current_staff)

    # staff metadata
    staff_metadata = []
    for staff in staves:
        if len(staff) == 5:
            spacings = [staff[i + 1] - staff[i] for i in range(4)]
            avg_spacing = np.mean(spacings)
            staff_metadata.append({
                "lines": staff,
                "spacing": avg_spacing,
            })

    return staff_metadata


def group_into_systems(staves):
    """
    Group staves into systems based on gaps between them.
    For treble-only input, each system has one staff.
    For treble+bass, each system has two staves.
    Identifies treble vs bass: in a paired system, top staff is treble, bottom is bass.
    """
    if not staves:
        return staves

    if len(staves) <= 1:
        staves[0]["system_index"] = 0
        staves[0]["clef"] = "treble"
        return staves

    # calculate gaps between consecutive staves - bottom line to top line
    gaps = []
    for i in range(1, len(staves)):
        gap = staves[i]["lines"][0] - staves[i - 1]["lines"][-1]
        gaps.append(gap)


    # if there are two distinct gap sizes, staves are paired
    # if all gaps are roughly similar, each staff is its own system
    min_gap = min(gaps)
    max_gap = max(gaps)

    if max_gap > min_gap * 1.8:
        # use the midpoint between min and max as the threshold
        threshold = (min_gap + max_gap) / 2
    else:
        # all gaps are similar so each staff is its own system
        threshold = 0

    # group staves into systems
    systems = [[0]]
    for i in range(len(gaps)):
        if threshold > 0 and gaps[i] < threshold:
            # small gap - same system (treble+bass pair)
            systems[-1].append(i + 1)
        else:
            # large gap or no pairing - new system
            systems.append([i + 1])

    # assign system index and clef to each staff
    for sys_idx, system in enumerate(systems):
        if len(system) == 1:
            staves[system[0]]["system_index"] = sys_idx
            staves[system[0]]["clef"] = "treble"
        else:
            for j, staff_idx in enumerate(system):
                staves[staff_idx]["system_index"] = sys_idx
                staves[staff_idx]["clef"] = "treble" if j == 0 else "bass"

    return staves


def remove_staff_lines(binary, staff_line_rows):
    """
    Remove staff line pixels using conditional deletion.
    Only remove a black pixel on a staff line row if it has no
    black pixels above and below outside the staff line region.
    Handles staff lines that are multiple pixels thick.
    Returns a cleaned image.
    """
    cleaned = binary.copy()
    h, w = cleaned.shape

    # build a set of staff line rows for quick lookup
    staff_row_set = set(staff_line_rows)

    for y in staff_line_rows:
        for x in range(w):
            if cleaned[y, x] == 0:  # black pixel on staff line
                # find the first row above that is not a staff line row
                check_above = y - 1
                while check_above >= 0 and check_above in staff_row_set:
                    check_above -= 1
                has_above = check_above >= 0 and cleaned[check_above, x] == 0

                # find the first row below that is not a staff line row
                check_below = y + 1
                while check_below < h and check_below in staff_row_set:
                    check_below += 1
                has_below = check_below < h and cleaned[check_below, x] == 0

                if not has_above and not has_below:
                    cleaned[y, x] = 255  # set to white

    return cleaned


if __name__ == "__main__":
    import sys
    from preprocess import binarize

    if len(sys.argv) < 2:
        print("Usage: python staff.py <image_path>")
        sys.exit(1)

    binary = binarize(sys.argv[1])

    # detect staff lines
    staff_line_rows = detect_staff_lines(binary)
    print(f"Detected {len(staff_line_rows)} staff line rows")

    # group into staves
    staves = group_into_staves(staff_line_rows)
    print(f"Grouped into {len(staves)} staves:")

    # group into systems
    staves = group_into_systems(staves)

    print(f"Grouped into {len(staves)} staves:")
    for i, staff in enumerate(staves):
        print(f"  Staff {i}: lines={staff['lines']}, spacing={staff['spacing']:.1f}, system={staff['system_index']}, clef={staff['clef']}")


    # remove staff lines
    cleaned = remove_staff_lines(binary, staff_line_rows)

    # show results
    cv2.imshow("Original Binary", binary)
    cv2.imshow("Staff Lines Removed", cleaned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()