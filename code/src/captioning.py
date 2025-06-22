
import re
import os
import sys
import cv2
import json
import time
from google import genai
from pathlib import Path
from google.genai import types
from dotenv import load_dotenv
from multiprocessing import Pool


CODEBASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_image_paths

load_dotenv() 


def caption_single_image(path_to_image : str):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    std_detection_prompt = f"""Please describe the provided brain MRI image.
    Important: Only return the description.
    """
    with open(path_to_image, 'rb') as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model=os.getenv("model"),
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            std_detection_prompt
        ]
    )
    print(f"LLM response: {response.text}")

    # because of rate limit
    time.sleep(30)
    return response.text

def get_caption_wrapper(args):
    path_to_image = args
    try:
        response = caption_single_image(path_to_image)
        return {
            "imageID" : os.path.splitext(os.path.basename(path_to_image))[0],
            "caption" : str(response)
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
        "imageID" : os.path.splitext(os.path.basename(path_to_image))[0],
        "caption" : None
        }


def get_caption_parallel(image_dir : str):
    if not Path.is_dir(CODEBASE_DIR / image_dir):
        raise ValueError("Wrong image dir. Please specify the relative path! :)")
    past_flag_ = False

    if not Path.is_file(Path(os.path.dirname(CODEBASE_DIR / image_dir)) / "captions_results.json"):
        image_paths = load_image_paths(dir=image_dir)
    else:
        with open(Path(os.path.dirname(CODEBASE_DIR / image_dir)) / "captions_results.json", 'r') as file:
            past_annotation = json.load(file)
        
        image_paths = []
        for ele in past_annotation:
            if not ele["caption"]:
                image_paths.append(CODEBASE_DIR / image_dir / f"{ele['imageID']}.png")
        
        past_flag_ = True

    tasks = [(path) for path in image_paths]

    with Pool(processes=4) as pool:
        results = pool.map(get_caption_wrapper, tasks)

    if past_flag_:
        annotation_lookup = {item["imageID"]: item["caption"] for item in results}
        for item in past_annotation:
            if item["caption"] is None:
                image_id = item["imageID"]
                item["caption"] = annotation_lookup.get(image_id)
        with open(Path(os.path.dirname(CODEBASE_DIR / image_dir)) / "captions_results.json", 'w') as file:
            json.dump(past_annotation, file)

    else:
        with open(Path(os.path.dirname(CODEBASE_DIR / image_dir)) / "captions_results.json", 'w') as file:
            json.dump(results, file)


if __name__=="__main__":
    get_caption_parallel(
        image_dir="data/nova_brain/images/"
    )

