"""
Aafi Mansuri, Terry Zhen
Apr 2026

This script is used to preview the fonts and the symbols in the fonts.
"""

from PIL import Image, ImageDraw, ImageFont
import os

SYMBOLS = {
    # Note heads
    "whole_notehead": "\uE0A2",
    "half_notehead": "\uE0A3",
    "filled_notehead": "\uE0A4",  # used for quarter and eighth notes

    # Individual notes with stem
    "whole_note": "\uE1D2",
    "half_note_up": "\uE1D3",
    "half_note_down": "\uE1D4",
    "quarter_note_up": "\uE1D5",
    "quarter_note_down": "\uE1D6",
    "eighth_note_up": "\uE1D7",
    "eighth_note_down": "\uE1D8",

    # Rests
    "whole_rest": "\uE4E3",
    "half_rest": "\uE4E4",
    "quarter_rest": "\uE4E5",
    "eighth_rest": "\uE4E6",

    # Clefs
    "treble_clef": "\uE050",
    "bass_clef": "\uE062",
}

def render_symbol_grid(font_path, font_name, font_size=60, output_path="fonts/preview"):
    """Render all symbols from a font into a grid image for inspection."""
    os.makedirs(output_path, exist_ok=True)

    font = ImageFont.truetype(font_path, font_size)
    symbols = list(SYMBOLS.items())
    cols = 4
    rows = (len(symbols) + cols - 1) // cols

    cell_w, cell_h = 200, 200
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(grid)

    try:
        label_font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        label_font = ImageFont.load_default()

    for i, (name, codepoint) in enumerate(symbols):
        col = i % cols
        row = i // cols
        x = col * cell_w
        y = row * cell_h

        # draw cell border
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline="lightgray")

        # draw label
        draw.text((x + 5, y + 5), name, fill="gray", font=label_font)

        # render symbol centered in cell
        bbox = font.getbbox(codepoint)
        if bbox:
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
            char_x = x + (cell_w - char_w) // 2 - bbox[0]
            char_y = y + (cell_h - char_h) // 2 - bbox[1] + 10
            draw.text((char_x, char_y), codepoint, fill="black", font=font)
        else:
            draw.text((x + 5, y + 100), "NOT FOUND", fill="red", font=label_font)

    save_path = os.path.join(output_path, f"{font_name}_symbols.png")
    grid.save(save_path)
    print(f"Saved grid to {save_path}")
    grid.show()


def main():
    fonts = {
        "bravura": "fonts/Bravura.otf",
        "leland": "fonts/Leland.otf",
    }

    for font_name, font_path in fonts.items():
        if not os.path.exists(font_path):
            print(f"Font not found: {font_path}")
            continue

        print(f"\n{font_name.upper()}:")
        print("Rendering grid...")
        render_symbol_grid(font_path, font_name)


if __name__ == "__main__":
    main()