
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


def detect_single_image(path_to_image : str):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    std_detection_prompt = f"""Please locate any abnormal areas in the MRI image, give the correct label and output the bounding boxes. Be as precise as possible. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
    Important: Only return the list of json object with keys box_2d and label nothing else.
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

def get_bbox(path_to_image: str, destination_path: str):
    image = cv2.imread(path_to_image)
    height, width = image.shape[:2]
    response_text = detect_single_image(path_to_image)

    try:
        lines = response_text.strip().split('\n')
        json_content_lines = lines[1:-1]
        clean_json_string = "\n".join(json_content_lines)
        response_json = json.loads(clean_json_string)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return {
        "imageID" : os.path.splitext(os.path.basename(path_to_image))[0],
        "annotation" : None
        }

    for item in response_json:
        y0, x0, y1, x1 = item["box_2d"]

        x_min = int(x0 / 1000 * width)
        y_min = int(y0 / 1000 * height)
        x_max = int(x1 / 1000 * width)
        y_max = int(y1 / 1000 * height)

        label = item["label"]

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0,0,255), thickness=2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imwrite(destination_path, image)

    annotation_json = {
        "imageID" : os.path.splitext(os.path.basename(path_to_image))[0],
        "annotation" : response_json
    }
    return annotation_json

def get_bbox_wrapper(args):
    path_to_image, destination_path = args
    try:
        response = get_bbox(path_to_image, destination_path)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return {
        "imageID" : os.path.splitext(os.path.basename(path_to_image))[0],
        "annotation" : None
        }


def get_bbox_parallel(image_dir : str, destination_dir : str):
    if not Path.is_dir(CODEBASE_DIR / image_dir):
        raise ValueError("Wrong image dir. Please specify the relative path! :)")
    past_flag_ = False

    if not Path.is_dir(CODEBASE_DIR / destination_dir):
        os.makedirs(CODEBASE_DIR / destination_dir, exist_ok=True)
        image_paths = load_image_paths(dir=image_dir)
    else:
        with open(Path(os.path.dirname(CODEBASE_DIR / destination_dir)) / "detection_annotation_results.json", 'r') as file:
            past_annotation = json.load(file)
        
        image_paths = []
        for ele in past_annotation:
            if not ele["annotation"]:
                image_paths.append(CODEBASE_DIR / image_dir / f"{ele['imageID']}.png")
        
        past_flag_ = True


    tasks = [
        (path, os.path.join(CODEBASE_DIR / destination_dir, os.path.basename(path)))
        for path in image_paths
    ]

    with Pool(processes=4) as pool:
        results = pool.map(get_bbox_wrapper, tasks)

    if past_flag_:
        annotation_lookup = {item["imageID"]: item["annotation"] for item in results}
        for item in past_annotation:
            if item["annotation"] is None:
                image_id = item["imageID"]
                item["annotation"] = annotation_lookup.get(image_id)
        with open(Path(os.path.dirname(CODEBASE_DIR / image_dir)) / "detection_annotation_results.json", 'w') as file:
            json.dump(past_annotation, file)

    else:
        with open(Path(os.path.dirname(CODEBASE_DIR / image_dir)) / "detection_annotation_results.json", 'w') as file:
            json.dump(results, file)


if __name__=="__main__":
    get_bbox_parallel(
        image_dir="data/nova_brain/images/",
        destination_dir="data/nova_brain/image_detection_results/"
    )

