import pytest
from pathlib import Path
from PIL import Image
import os
from arc_drone.auair_eval import _load_images, _make_mosaic

def test_load_real_images_on_pod():
    possible_paths = ["/workspace/gemma-4/images", "images"]
    base_dir = None
    for p in possible_paths:
        if Path(p).exists() and any(Path(p).glob("*.jpg")):
            base_dir = p
            break
    
    if not base_dir:
        pytest.skip("No real images found for testing.")
        
    img_files = sorted(list(Path(base_dir).glob("*.jpg")))[:4]
    img_paths = [str(f.absolute()) for f in img_files]
    
    loaded = _load_images(img_paths, T=4, images_path=base_dir)
    assert len(loaded) == 4
    for img in loaded:
        assert isinstance(img, Image.Image)
        extrema = img.getextrema()
        is_gray = all(ex[0] == 80 and ex[1] == 80 for ex in extrema)
        assert not is_gray

def test_make_mosaic():
    imgs = [Image.new("RGB", (10, 10), color=(i*10, 0, 0)) for i in range(4)]
    mosaic = _make_mosaic(imgs)
    # 2x2 of 224x224 (since _make_mosaic now forces resizing to 224)
    assert mosaic.size == (448, 448)
    assert mosaic.getpixel((0, 0)) != mosaic.getpixel((250, 250))
