# SheetVision
An optical music recognition pipeline that converts printed sheet music into MIDI files. Takes an image of single-voice, treble-clef melody as input, detects and removes staff lines, segments musical symbols, classifies them with a CNN trained on synthetic data, determines pitch from staff position, and outputs a playable MIDI file.


# Fonts:

Bravura: https://github.com/steinbergmedia/bravura/blob/master/redist/otf/Bravura.otf 

Leland: https://github.com/MuseScoreFonts/Leland/blob/main/Leland.otf