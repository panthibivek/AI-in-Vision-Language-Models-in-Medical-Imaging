
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

load_dotenv()


def detect_disease(case: json):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    std_detection_prompt = f"""Based on the clinical history and image findings, provide your diagnosis for the disease."
    Important: Only return the diagnosis for the disease as text.

    Case Details:
    {case} 
    """

    response = client.models.generate_content(
        model=os.getenv("model"),
        contents=[
            std_detection_prompt
        ]
    )
    print(f"LLM response: {response.text}")

    # because of rate limit
    time.sleep(10)
    return response.text

def disease_detection_classification(ground_truth, prediction):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    std_detection_prompt = f"""Based on the ground truth disease and predicted disease, classify if the diseases match. Return 1 if correct, otherwise 0"
    Important: Only return 1 or 0.

    Ground truth diagnosis: {ground_truth}
    Predicted diagnosis: {prediction}
    {case} 
    """

    response = client.models.generate_content(
        model=os.getenv("model"),
        contents=[
            std_detection_prompt
        ]
    )
    print(f"LLM response: {response.text}")

    # because of rate limit
    time.sleep(10)
    return response.text

if __name__=="__main__":
    with open(CODEBASE_DIR / "data/nova_brain/annotations.json", 'r') as file:
        annotation_json = json.load(file)
    
    disease_detection_annotations = []
    output_filename = CODEBASE_DIR / "data/nova_brain/disease_detection_annotations.json"

    # if not output_filename.is_file():
    #     with open(output_filename, 'w') as file:
    #         json.dump([], file)

    # for case in annotation_json.keys():
    #     case_input = {
    #         'clinical_history': annotation_json[case]['clinical_history'],
    #         'image_findings': annotation_json[case]['image_findings']
    #     }
    #     llm_response = detect_disease(case_input)

    #     case_json = annotation_json[case]
    #     case_json["llm_disease_detection"] = str(llm_response)
        
    #     with open(output_filename, 'r') as file:
    #         disease_detection_annotations = json.load(file)
    #     disease_detection_annotations.append(case_json)

    #     with open(output_filename, 'w') as file:
    #         json.dump(disease_detection_annotations, file)


    # detection accuracy
    with open(output_filename, 'r') as file:
        disease_detection_annotations = json.load(file)
    
    classifications = []
    for case in disease_detection_annotations:
        classification = disease_detection_classification(case["final_diagnosis"], case["llm_disease_detection"])
        if classification == "1":
            classifications.append(1)
        else:
            classifications.append(0)
    
    print(f"Accuracy = {sum(classifications)/len(classifications)}")
    