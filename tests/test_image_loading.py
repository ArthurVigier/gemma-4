import pytest
from pathlib import Path
from PIL import Image
import os
from arc_drone.auair_eval import _load_images, _make_mosaic

def test_load_real_images_on_pod():
    # Attempt to find real images in common pod locations
    possible_paths = ["/workspace/gemma-4/images", "images"]
    base_dir = None
    for p in possible_paths:
        if Path(p).exists() and any(Path(p).glob("*.jpg")):
            base_dir = p
            break
    
    if not base_dir:
        pytest.skip("No real images found for testing. Please ensure AU-AIR is unzipped.")
        
    # Get 4 real images
    img_files = sorted(list(Path(base_dir).glob("*.jpg")))[:4]
    img_paths = [str(f.absolute()) for f in img_files]
    
    # Test loading
    loaded = _load_images(img_paths, T=4, images_path=base_dir)
    assert len(loaded) == 4
    for img in loaded:
        assert isinstance(img, Image.Image)
        # Verify it is not our fallback gray (80,80,80)
        extrema = img.getextrema() # ((minR, maxR), (minG, maxG), (minB, maxB))
        # fallback gray has min=max=80
        is_gray = all(ex[0] == 80 and ex[1] == 80 for ex in extrema)
        assert not is_gray, f"Image {img} appears to be the fallback gray!"

def test_make_mosaic():
    imgs = [Image.new("RGB", (10, 10), color=(i*10, 0, 0)) for i in range(4)]
    mosaic = _make_mosaic(imgs)
    assert mosaic.size == (20, 20)
    # Check if pixels are different (proves paste worked)
    assert mosaic.getpixel((0, 0)) != mosaic.getpixel((15, 15))

