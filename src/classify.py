# Aafi Mansuri, Terry Zhen
# Symbol Classification
# Loads trained CNN and classifies each segmented symbol

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from cnn_architecture import MusicCNN

TARGET_SIZE = (64, 64)

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Class index mappings
IDX_TO_CLASS = {
    0: "block_rest",
    1: "eighth_note",
    2: "eighth_rest",
    3: "half_note",
    4: "quarter_note",
    5: "quarter_rest",
    6: "whole_note"
}

def preprocess_crop(image):
    """
    Resize a cropped symbol image to TARGET_SIZE preserving aspect ratio,
    centered on a white canvas. Matches the training data preprocessing.
    Input: numpy array (H x W), values 0 and 255
    Output: PIL Image ready for normalization
    """
    pil_img = Image.fromarray(image).convert("L")

    # crop to non-white pixels
    bbox = pil_img.getbbox()
    if bbox:
        pil_img = pil_img.crop(bbox)

    # scale to fit within target size with margin
    margin = 4
    max_w = TARGET_SIZE[0] - 2 * margin
    max_h = TARGET_SIZE[1] - 2 * margin
    w, h = pil_img.size
    if w == 0 or h == 0:
        return Image.new("L", TARGET_SIZE, 255)

    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

    # center on white canvas
    canvas = Image.new("L", TARGET_SIZE, 255)
    paste_x = (TARGET_SIZE[0] - new_w) // 2
    paste_y = (TARGET_SIZE[1] - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    # binarize to match training data
    canvas = canvas.point(lambda p: 0 if p < 180 else 255)

    return canvas

def load_model(model_path="music_cnn.pt"):
    """Load the trained CNN from disk and set to eval mode."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def classify_symbols(symbols, model, device):
    """
    Run each symbol crop through the CNN and attach predictions and confidence.
    """
    results = []


    for sym in symbols:
        # preprocess to match training data format
        pil_img = preprocess_crop(sym["image"])

        # Apply the normalize
        tensor = normalize(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)

        label = IDX_TO_CLASS.get(pred_idx.item(), f"class_{pred_idx.item()}")
        confidence = conf.item()

        results.append({
            **sym,
            "label":      label,
            "confidence": confidence,
        })

    return results


if __name__ == "__main__":
    import sys
    from preprocess import binarize
    from staff import detect_staff_lines, group_into_staves, group_into_systems, remove_staff_lines
    from segment import find_symbols, filter_symbols

    if len(sys.argv) < 2:
        print("Usage: python classify.py <image_path>")
        sys.exit(1)

    # Run preprocess pipeline
    binary = binarize(sys.argv[1])
    staff_rows = detect_staff_lines(binary)
    staves = group_into_staves(staff_rows)
    staves = group_into_systems(staves)
    cleaned = remove_staff_lines(binary, staff_rows)

    # Retrieve candidates and symbols
    candidates = find_symbols(cleaned, staves)
    symbols = filter_symbols(candidates, staves, cleaned)

    print(f"Segmented {len(symbols)} symbols\n")

    # Load model and classify symbols
    model, device = load_model("src/music_cnn.pt")
    results = classify_symbols(symbols, model, device)

    # Print results
    for i, r in enumerate(results):
        print(
            f"sys={r['system_index']}  staff={r['staff_index']}"
            f"  x={r['x']:4.0f}  ->  {r['label']}  ({r['confidence']:.2%})"
        )
        del r['confidence']