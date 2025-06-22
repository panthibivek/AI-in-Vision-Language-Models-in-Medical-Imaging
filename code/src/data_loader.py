
import os
from pathlib import Path

CODEBASE_DIR = Path(__file__).resolve().parent.parent

def load_image_paths(dir : str):
    image_extensions = ['.jpg', '.jpeg', '.png']
    data_dir = CODEBASE_DIR / dir

    image_paths = [
        str(file.resolve())
        for file in data_dir.rglob('*')
        if file.suffix.lower() in image_extensions
    ]
    return image_paths

if __name__=="__main__":
    x_ray_images = load_image_paths(dir="data/chest_xrays/images/")
    print(len(x_ray_images))

    nova_brain_images = load_image_paths(dir="data/nova_brain/images/")
    print(len(nova_brain_images))
