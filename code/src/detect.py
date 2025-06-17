
import os
import cv2
import json
import pathlib
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv() 


def detect_single_image(path_to_image : str):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    std_detection_prompt = "Detect all of the items in the image. There can be multiple objects. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
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
    return response.text


def get_bbox(path_to_image: str, destination_path: str):
    image = cv2.imread(path_to_image)
    height, width = image.shape[:2]

    response = detect_single_image(path_to_image)

    lines = response.strip().split('\n')
    json_content_lines = lines[1:-1]
    clean_json_string = "\n".join(json_content_lines)

    try:
        response_json = json.loads(clean_json_string)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    for item in response_json:
        y0, x0, y1, x1 = item["box_2d"]

        x_min = int(x0 / 1000 * width)
        y_min = int(y0 / 1000 * height)
        x_max = int(x1 / 1000 * width)
        y_max = int(y1 / 1000 * height)

        label = item["label"]

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite(destination_path, image)


if __name__=="__main__":
    test_image_path = pathlib.Path("D:/TUM/AI_in_VLM/code/data/test_image.jpg")
    destination_path = pathlib.Path("D:/TUM/AI_in_VLM/code/data/test_image_output.jpg")
    get_bbox(test_image_path, destination_path)
