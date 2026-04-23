# SheetVision

An optical music recognition pipeline that converts printed sheet music images into playable MIDI files. Built as a final project for CS 5330 (Pattern Recognition and Computer Vision) at Northeastern University.

## What it does

Give it a clean image of printed sheet music and it will:
1. Detect and remove staff lines while preserving note symbols
2. Segment individual musical symbols from the cleaned image
3. Classify each symbol using a CNN (quarter note, half note, rest, etc.)
4. Determine pitch from vertical position on the staff
5. Generate a MIDI file you can listen to

## Demo

Input sheet music → pipeline → playable MIDI output

Tested on Twinkle Twinkle Little Star, Jingle Bells, and Mary Had a Little Lamb. All three produce recognizable melodies.

## Project Structure

```
sheet-to-midi/
├── README.md
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── preprocess.py           # Binarization (grayscale + Otsu threshold)
│   ├── staff.py                # Staff detection, grouping, and removal
│   ├── segment.py              # Connected components + geometric filtering
│   ├── cnn_architecture.py     # CNN model definition
│   ├── train.py                # CNN training script
│   ├── classify.py             # CNN inference on segmented symbols
│   ├── note_mapping.py         # Pitch determination and duration mapping
│   ├── midi_gen.py             # MIDI file generation
│   ├── music_cnn.pt            # Saved CNN model weights
│   └── dataset_generator.py    # Synthetic training data generation
│
├── dataset/                    # Generated synthetic training images (896 total)
│   ├── whole_note/
│   ├── half_note/
│   ├── quarter_note/
│   ├── eighth_note/
│   ├── block_rest/
│   ├── quarter_rest/
│   └── eighth_rest/    
│
├── sheets/                      # Sample sheet music images for testing
│
├── output/                     # Generated MIDI files

```

## Setup

### Requirements

- Python 3.10+
- CUDA-compatible GPU (optional, CPU works too)

### Install PyTorch

With CUDA:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

CPU only:
```bash
uv pip install torch torchvision
```

### Install remaining dependencies

```bash
uv pip install -r requirements.txt
```

### Fonts (for dataset regeneration only)

If you want to regenerate the synthetic dataset from scratch, download these fonts and place the `.otf` files in a `fonts/` directory:

- [Bravura](https://github.com/w3c/smufl/releases) (SIL Open Font License)
- [Leland](https://github.com/MuseScoreFonts/Leland/releases) (SIL Open Font License)

The pre-generated dataset is already included in the repo, so this step is optional.

## Usage

### Run the full pipeline

```bash
python src/midi_gen.py input/twinkle.png output/twinkle.mid 120
```

Arguments:
- `sheets/twinkle.png` - path to sheet music image
- `output/twinkle.mid` - output MIDI file path
- `120` - tempo in BPM (optional, defaults to 120)

### Run individual stages

```bash
# Binarization only
python src/preprocess.py input/twinkle.png

# Staff detection and removal
python src/staff.py input/twinkle.png

# Segmentation (shows debug visualization)
python src/segment.py input/twinkle.png

# Classification
python src/classify.py input/twinkle.png

# Pitch determination
python src/note_mapping.py input/twinkle.png
```

### Regenerate the dataset

```bash
python src/dataset_generator.py
```

This generates 896 images (128 per class) across 7 symbol categories using the Bravura and Leland fonts. A fixed seed ensures reproducibility.

### Train the CNN

```bash
python src/train.py
```

Trains for 10 epochs with an 80/20 train-validation split. Saves the model to `music_cnn.pt`.

## How it works

### Staff Detection
Horizontal projection profiling counts black pixels per row. Rows exceeding 30% of the image width are staff lines. Consecutive rows are merged and grouped into staves of five using gap analysis.

### Staff Line Removal
Conditional pixel deletion removes staff line pixels only if they have no black neighbors above or below (outside the staff line region). This preserves note stems that pass through staff lines.

### Segmentation
Connected components finds all symbol blobs. Geometric filters remove noise (too small), bar lines (tall and thin), clefs and time signatures (far left), chord labels (above staff), and lyrics (below staff). Each symbol is assigned to its nearest staff.

### Classification
A CNN with two double-convolution blocks (32 and 64 filters) classifies symbols into 7 categories. Trained on synthetic data rendered from music fonts with rotation, translation, and noise augmentation.

### Pitch Determination
Each note's vertical position relative to its staff lines is converted to a pitch index. The formula `(bottom_line_y - note_y) / half_spacing` maps to a treble clef pitch array starting at E4 on the bottom line.

### MIDI Generation
Symbols are sorted in reading order (by system, then x-position). Note labels map to tick durations and pitch values are converted to MIDI numbers. The mido library writes the final MIDI file.

## Scope and Limitations

The current version handles:
- Clean printed sheet music
- Treble clef only
- Single voice (no chords)
- Whole, half, quarter, and eighth notes and rests
- Assumed 4/4 time signature

Not yet supported:
- Beamed notes (eighth/sixteenth notes connected by horizontal bars)
- Bass clef (infrastructure exists but not fully integrated)
- Accidentals (sharps, flats)
- Dotted notes, ties, slurs
- Handwritten music

## Synthetic Dataset

The dataset is generated from SMuFL-compatible music fonts rather than manually labeled real symbols. Each symbol is rendered at multiple sizes, augmented with light rotation, translation, and noise, then resized to 64x64 preserving aspect ratio. The generator script (`dataset_generator.py`) uses a fixed seed so the exact same dataset can be reproduced.

7 classes: `whole_note`, `half_note`, `quarter_note`, `eighth_note`, `block_rest`, `quarter_rest`, `eighth_rest`

Whole and half rests are combined into a single `block_rest` class because they look visually identical as isolated symbols. They are distinguished later by their vertical position on the staff.

## Authors

- Aafi Mansuri
- Terry Zhen

CS 5330 Pattern Recognition and Computer Vision, Northeastern University, Spring 2026