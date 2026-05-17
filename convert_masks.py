import cv2
import numpy as np
from pathlib import Path

REMAP = {
    11:  0,   # Black Background  -> background
    12:  1,   # Abdominal Wall    -> tissue
    13:  2,   # Liver             -> organ
    21:  1,   # Gastrointestinal  -> tissue
    22:  1,   # Fat               -> tissue
    31:  3,   # Cystic Duct       -> tumor
    32:  2,   # Gallbladder       -> organ
    50:  1,   # Connective Tissue -> tissue
    255: 0,   # Unlabeled         -> background
}

src_root = Path("data/cholecseg")
img_out  = Path("data/prepared/images")
mask_out = Path("data/prepared/masks")
img_out.mkdir(parents=True, exist_ok=True)
mask_out.mkdir(parents=True, exist_ok=True)

wm_files = sorted(src_root.rglob("*_watershed_mask.png"))
print(f"Found {len(wm_files)} watershed masks")

skipped, saved = 0, 0
for wm_path in wm_files:
    stem     = wm_path.stem.replace("_watershed_mask", "")
    img_path = wm_path.parent / f"{stem}.png"
    if not img_path.exists():
        skipped += 1
        continue

    wm = cv2.imread(str(wm_path), cv2.IMREAD_GRAYSCALE)
    class_mask = np.zeros_like(wm, dtype=np.uint8)
    for src_val, dst_cls in REMAP.items():
        class_mask[wm == src_val] = dst_cls

    # Skip frames that are entirely background
    if np.unique(class_mask).tolist() == [0]:
        skipped += 1
        continue

    out_name = (
        wm_path.parent.parent.name + "_"
        + wm_path.parent.name + "_"
        + stem + ".png"
    )
    cv2.imwrite(str(img_out  / out_name), cv2.imread(str(img_path)))
    cv2.imwrite(str(mask_out / out_name), class_mask)
    saved += 1

    if saved % 100 == 0:
        print(f"  Saved {saved} pairs so far...")

print()
print(f"Done.   Saved: {saved}   Skipped: {skipped}")
print(f"Images -> {img_out}")
print(f"Masks  -> {mask_out}")

# Verify a sample
sample_masks = sorted(mask_out.glob("*.png"))[:5]
print()
print("Sample mask class distributions:")
for mp in sample_masks:
    m    = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    vals, counts = np.unique(m, return_counts=True)
    total = m.size
    dist  = {int(v): str(round(int(c) / total * 100, 1)) + "%" 
             for v, c in zip(vals, counts)}
    print(f"  {mp.name[:50]:<50}  {dist}")
