# Aafi Mansuri, Terry Zhen
# Pitch determination and duration mapping for OMR pipeline
# Calculates MIDI pitch from y-position and maps CNN labels to tick durations

# treble clef pitch array
# index 0 = bottom line (E4), positive = upward, negative = below staff
TREBLE_PITCHES = [
    "C4", "D4",   # below staff (ledger line region)
    "E4", "F4", "G4", "A4", "B4",  # within staff lines and spaces
    "C5", "D5", "E5", "F5",        # above staff
    "G5", "A5",                      # further above
]
TREBLE_BASE_INDEX = 2  # index of E4 (bottom line) in the array

# note name to MIDI number
NOTE_TO_MIDI = {
    "C4": 60, "D4": 62, "E4": 64, "F4": 65, "G4": 67,
    "A4": 69, "B4": 71, "C5": 72, "D5": 74, "E5": 76,
    "F5": 77, "G5": 79, "A5": 81,
}

# CNN label to tick duration (at 480 ticks per beat)
DURATION_MAP = {
    "whole_note":    1920,  # 4 beats
    "half_note":      960,  # 2 beats
    "quarter_note":   480,  # 1 beat
    "eighth_note":    240,  # 0.5 beats
    "whole_rest":    1920,
    "half_rest":      960,
    "quarter_rest":   480,
    "eighth_rest":    240,
    "block_rest":     960,  # default to half rest duration
}

# labels that are rests (no pitch)
REST_LABELS = {"whole_rest", "half_rest", "quarter_rest", "eighth_rest", "block_rest"}

# labels that are notes (have pitch)
NOTE_LABELS = {"whole_note", "half_note", "quarter_note", "eighth_note"}


def determine_pitch(symbol, staves):
    """
    Calculate MIDI pitch from a note's y-position relative to its staff lines.
    Returns the MIDI pitch number or None for rests/unknown.
    """
    label = symbol.get("label", "")

    # rests and unknown symbols have no pitch
    if label in REST_LABELS or label not in NOTE_LABELS:
        return None

    staff = staves[symbol["staff_index"]]
    bottom_line_y = staff["lines"][-1]  # largest y = bottom line (y increases downward)
    top_line_y = staff["lines"][0]
    half_spacing = staff["spacing"] / 2

    # use the note head position instead of bounding box center
    # for tall symbols (with stems), the note head is at one end
    # for short symbols (whole notes), the center is fine
    x, y, w, h = symbol["bbox"]
    bbox_top = y
    bbox_bottom = y + h

    if h > w * 1.5:
        # tall symbol (has a stem) — note head is at one end
        # check which end is closer to a staff line
        min_dist_top = min(abs(bbox_top - line) for line in staff["lines"])
        min_dist_bottom = min(abs(bbox_bottom - line) for line in staff["lines"])

        if min_dist_bottom <= min_dist_top:
            note_y = bbox_bottom
        else:
            note_y = bbox_top

        print(f"  bbox_top={bbox_top}, bbox_bottom={bbox_bottom}, dist_top={min_dist_top}, dist_bottom={min_dist_bottom}, chose={'bottom' if note_y==bbox_bottom else 'top'}, note_y={note_y}")
    else:
        # short symbol (whole note, rest) — use center
        note_y = symbol["y"]
        print(f"  short symbol, using center y={note_y}")

    position_index = round((bottom_line_y - note_y) / half_spacing)

    # map to pitch array
    pitch_array_index = TREBLE_BASE_INDEX + position_index

    if 0 <= pitch_array_index < len(TREBLE_PITCHES):
        note_name = TREBLE_PITCHES[pitch_array_index]
        return NOTE_TO_MIDI.get(note_name)
    else:
        return None


def determine_duration(symbol):
    """
    Map a CNN label to its tick duration.
    Returns tick count or None if label is unknown.
    """
    label = symbol.get("label", "")
    return DURATION_MAP.get(label)


def is_rest(symbol):
    """Check if a symbol is a rest."""
    return symbol.get("label", "") in REST_LABELS


def annotate_symbols(symbols, staves):
    """
    Add midi_pitch and ticks to each symbol dictionary.
    Returns the annotated symbol list.
    """
    annotated = []

    for sym in symbols:
        midi_pitch = determine_pitch(sym, staves)
        ticks = determine_duration(sym)

        if ticks is None:
            # unknown label, skip this symbol
            continue

        annotated.append({
            **sym,
            "midi_pitch": midi_pitch,
            "ticks": ticks,
            "is_rest": is_rest(sym),
        })

    return annotated


if __name__ == "__main__":
    import sys
    from preprocess import binarize
    from staff import detect_staff_lines, group_into_staves, group_into_systems, remove_staff_lines
    from segment import find_symbols, filter_symbols
    from classify import load_model, classify_symbols

    if len(sys.argv) < 2:
        print("Usage: python pitch.py <image_path>")
        sys.exit(1)

    # run full pipeline
    binary = binarize(sys.argv[1])
    staff_rows = detect_staff_lines(binary)
    staves = group_into_staves(staff_rows)
    staves = group_into_systems(staves)
    cleaned = remove_staff_lines(binary, staff_rows)

    candidates = find_symbols(cleaned, staves)
    symbols = filter_symbols(candidates, staves, cleaned)

    model, device = load_model("src/music_cnn.pt")
    classified = classify_symbols(symbols, model, device)

    # annotate with pitch and duration
    annotated = annotate_symbols(classified, staves)

    print(f"\nAnnotated {len(annotated)} symbols:\n")
    for sym in annotated:
        if sym["is_rest"]:
            print(f"  sys={sym['system_index']}  x={sym['x']:4d}  {sym['label']:15s}  rest  ticks={sym['ticks']}")
        else:
            # find note name from midi pitch
            note_name = "?"
            for name, num in NOTE_TO_MIDI.items():
                if num == sym["midi_pitch"]:
                    note_name = name
                    break
            print(f"  sys={sym['system_index']}  x={sym['x']:4d}  {sym['label']:15s}  {note_name:4s}  midi={sym['midi_pitch']}  ticks={sym['ticks']}")