# Aafi Mansuri, Terry Zhen
# MIDI generation for OMR pipeline
# Converts annotated symbols into a playable MIDI file

import mido


def generate_midi(annotated_symbols, output_path="output.mid", tempo_bpm=120):
    """
    Generate a MIDI file from annotated symbols.
    Symbols should already be sorted by system_index then x position.
    Each symbol should have: midi_pitch, ticks, is_rest, label
    """
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # set tempo
    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo))

    # set instrument to acoustic grand piano
    track.append(mido.Message("program_change", program=0, time=0))

    velocity = 64

    for sym in annotated_symbols:
        if sym["is_rest"]:
            # rest
            track.append(mido.Message("note_on", note=0, velocity=0, time=sym["ticks"]))
        elif sym["midi_pitch"] is not None:
            # note: play it
            pitch = sym["midi_pitch"]
            duration = sym["ticks"]

            track.append(mido.Message("note_on", note=pitch, velocity=velocity, time=0))
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=duration))
        # skip symbols with no pitch and not a rest

    # end of track
    track.append(mido.MetaMessage("end_of_track", time=0))

    mid.save(output_path)
    print(f"MIDI file saved to: {output_path}")

    return mid


if __name__ == "__main__":
    import sys
    from preprocess import binarize
    from staff import detect_staff_lines, group_into_staves, group_into_systems, remove_staff_lines
    from segment import find_symbols, filter_symbols
    from classify import load_model, classify_symbols
    from note_mapping import annotate_symbols

    if len(sys.argv) < 2:
        print("Usage: python midi_gen.py <image_path> [output.mid] [tempo_bpm]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/output.mid"
    tempo_bpm = int(sys.argv[3]) if len(sys.argv) > 3 else 120

    # run full pipeline
    print("Preprocessing...")
    binary = binarize(image_path)

    print("Detecting staff lines...")
    staff_rows = detect_staff_lines(binary)
    staves = group_into_staves(staff_rows)
    staves = group_into_systems(staves)

    print("Removing staff lines...")
    cleaned = remove_staff_lines(binary, staff_rows)

    print("Segmenting symbols...")
    candidates = find_symbols(cleaned, staves)
    symbols = filter_symbols(candidates, staves, cleaned)
    print(f"  Found {len(symbols)} symbols")

    print("Classifying symbols...")
    model, device = load_model("src/music_cnn.pt")
    classified = classify_symbols(symbols, model, device)

    print("Determining pitch and duration...")
    annotated = annotate_symbols(classified, staves)
    print(f"  Annotated {len(annotated)} symbols")

    # print summary
    for sym in annotated:
        if sym["is_rest"]:
            print(f"  {sym['label']:15s}  rest  ticks={sym['ticks']}")
        else:
            note_name = "?"
            from note_mapping import NOTE_TO_MIDI
            for name, num in NOTE_TO_MIDI.items():
                if num == sym["midi_pitch"]:
                    note_name = name
                    break
            print(f"  {sym['label']:15s}  {note_name:4s}  midi={sym['midi_pitch']}  ticks={sym['ticks']}")

    print(f"\nGenerating MIDI at {tempo_bpm} BPM...")
    generate_midi(annotated, output_path, tempo_bpm)
    print("Done!")