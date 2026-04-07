"""
Aafi Mansuri, Terry Zhen
Apr 2026

This script is used to generate a synthetic dataset of musical symbols from the SMuFL-compatible fonts.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# target image size for CNN input
TARGET_SIZE = (64, 64)

# font sizes to render
FONT_SIZES = [330, 390, 450, 510]

# target images per class
TARGET_PER_CLASS = 128

# fonts to use
FONTS = {
    "bravura": "../fonts/Bravura.otf",
    "leland": "../fonts/Leland.otf",
}

# symbol classes with their SMuFL codepoints
# notes with stem-up and stem-down variants are grouped under one class
SYMBOL_CLASSES = {
    "whole_note": ["\uE1D2"],
    "half_note": ["\uE1D3", "\uE1D4"],          
    "quarter_note": ["\uE1D5", "\uE1D6"],        
    "eighth_note": ["\uE1D7", "\uE1D8"],          
    "block_rest": ["\uE4E3", "\uE4E4"],            # whole rest and half rest
    "quarter_rest": ["\uE4E5"],
    "eighth_rest": ["\uE4E6"],
}

OUTPUT_DIR = "../dataset"


def render_base(font, codepoint, canvas_size=600):
    """Render a single symbol on a white canvas and crop tightly."""
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)

    bbox = font.getbbox(codepoint)
    if not bbox:
        return None

    # center the symbol on the canvas
    char_x = (canvas_size - (bbox[2] - bbox[0])) // 2 - bbox[0]
    char_y = (canvas_size - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((char_x, char_y), codepoint, fill=0, font=font)

    # crop to non-white pixels
    img_bbox = img.getbbox()
    if not img_bbox:
        return None

    return img.crop(img_bbox)


def add_padding(img, pad=10):
    """Add white padding around the image to allow room for augmentation."""
    w, h = img.size
    padded = Image.new("L", (w + 2 * pad, h + 2 * pad), 255)
    padded.paste(img, (pad, pad))
    return padded


def augment(img):
    """Apply random augmentations to a base symbol image."""
    # random rotation (-2 to +2 degrees)
    angle = random.uniform(-2, 2)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

    # random translation
    shift_x = random.randint(-2, 2)
    shift_y = random.randint(-2, 2)
    img = img.transform(
        img.size, Image.AFFINE, (1, 0, -shift_x, 0, 1, -shift_y), fillcolor=255
    )

    # light gaussian noise
    img_array = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(1, 4), img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    return img


def process_and_save(img, save_path):
    """Crop tightly, resize preserving aspect ratio, center on square canvas, and save."""
    # crop to non-white pixels
    bbox = img.getbbox()
    if not bbox:
        return False

    cropped = img.crop(bbox)

    # calculate scale to fit within target size with some margin
    margin = 4
    max_w = TARGET_SIZE[0] - 2 * margin
    max_h = TARGET_SIZE[1] - 2 * margin
    w, h = cropped.size
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    # create square canvas and paste centered
    final = Image.new("L", TARGET_SIZE, 255)
    paste_x = (TARGET_SIZE[0] - new_w) // 2
    paste_y = (TARGET_SIZE[1] - new_h) // 2
    final.paste(resized, (paste_x, paste_y))

    # binarize to clean up anti-aliasing artifacts
    threshold = 180
    final = final.point(lambda p: 0 if p < threshold else 255)

    final.save(save_path)
    return True


def generate_dataset():
    """Main function to generate the full synthetic dataset."""
    total_images = 0

    for class_name, codepoints in SYMBOL_CLASSES.items():
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        class_count = 0

        # calculate how many base renders this class will have
        num_bases = len(FONTS) * len(FONT_SIZES) * len(codepoints)
        augs_per_base = max(1, TARGET_PER_CLASS // num_bases)

        for font_name, font_path in FONTS.items():
            if not os.path.exists(font_path):
                print(f"Font not found: {font_path}, skipping")
                continue

            for font_size in FONT_SIZES:
                font = ImageFont.truetype(font_path, font_size)

                for codepoint in codepoints:
                    base_img = render_base(font, codepoint)
                    if base_img is None:
                        print(f"  Could not render {class_name} from {font_name} at size {font_size}")
                        continue

                    padded_base = add_padding(base_img)

                    for aug_i in range(augs_per_base):
                        augmented = augment(padded_base.copy())
                        filename = f"{font_name}_{font_size}_{codepoints.index(codepoint)}_{aug_i}.png"
                        save_path = os.path.join(class_dir, filename)

                        if process_and_save(augmented, save_path):
                            class_count += 1

        total_images += class_count
        print(f"{class_name}: {class_count} images")

    print(f"\nTotal: {total_images} images across {len(SYMBOL_CLASSES)} classes")
    print(f"Saved to: {OUTPUT_DIR}/")


def main():
    print("Generating synthetic music symbol dataset...")
    print(f"Target size: {TARGET_SIZE}")
    print(f"Fonts: {list(FONTS.keys())}")
    print(f"Font sizes: {FONT_SIZES}")
    print(f"Seed: {SEED}\n")
    generate_dataset()


if __name__ == "__main__":
    main()