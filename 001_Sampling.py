#!/usr/bin/env python3
import os
import shutil
import random
from PIL import Image

from path_config import CLEAN_DATASET, DATA_ROOT


SRC_ROOT = DATA_ROOT / "DomainNet"
DST_ROOT = DATA_ROOT / CLEAN_DATASET
SAMPLES_PER_DOMAIN = 500
SEED = 0
OUT_SIZE = 512

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff")

# center-crop -> resize
def center_crop_resize_to_square(img: Image.Image, out_size: int) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    side = min(w, h)

    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side

    img = img.crop((left, top, right, bottom))
    img = img.resize((out_size, out_size), resample=Image.BICUBIC)
    return img

def list_images_recursive(root_dir: str):
    paths = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(r, f))
    return paths

def main():
    rng = random.Random(SEED)

    if not os.path.isdir(SRC_ROOT):
        raise FileNotFoundError(SRC_ROOT)

    if os.path.exists(DST_ROOT):
        shutil.rmtree(DST_ROOT)
    os.makedirs(DST_ROOT, exist_ok=True)

    domains = sorted(
        d for d in os.listdir(SRC_ROOT)
        if os.path.isdir(os.path.join(SRC_ROOT, d))
    )

    if not domains:
        raise ValueError(f"No domain directories found under {SRC_ROOT}")

    for domain in domains:
        domain_src = os.path.join(SRC_ROOT, domain)
        domain_dst = os.path.join(DST_ROOT, domain)
        os.makedirs(domain_dst, exist_ok=True)

        all_imgs = sorted(list_images_recursive(domain_src))

        if len(all_imgs) == 0:
            continue

        k = min(SAMPLES_PER_DOMAIN, len(all_imgs))
        sampled = rng.sample(all_imgs, k)

        for i, src_path in enumerate(sampled):
            try:
                with Image.open(src_path) as img:
                    out = center_crop_resize_to_square(img, OUT_SIZE)

                out_name = f"{domain}_{i:04d}.png"
                out_path = os.path.join(domain_dst, out_name)
                out.save(out_path, format="PNG", optimize=True)
            except Exception:
                continue

if __name__ == "__main__":
    main()
