import os
import json
from tqdm import tqdm

import cv2 as cv




FOLDER = "annotated_top"


images = os.listdir(FOLDER)

with open('annotations.json', 'r') as f:
    annotations = json.load(f)

for image in tqdm(annotations['images']):
    start_idx = image['file_name'].rfind('\\')
    new_name = image['file_name'][start_idx+1:]
    new_name = new_name.replace('.png', '.jpg')
    image['file_name'] = new_name

for image in tqdm(images):
    if image.endswith(".png"):
        img = cv.imread(os.path.join(FOLDER, image))
        cv.imwrite(os.path.join(FOLDER, image[:-4] + ".jpg"), img)
        #delete the png
        os.remove(os.path.join(FOLDER, image))

annotations['categories'] = [
    {
        "id": 1,
        "name": "logo"
    },
    {
        "id": 2,
        "name": "sticker"
    }
]


for annotation in annotations['annotations']:
    if annotation['category_id'] == 1:
        annotation['category_id'] = 2
    elif annotation['category_id'] == 0:
        annotation['category_id'] = 1



for image in annotations["images"]:
    if image["file_name"].endswith(".png"):
        image["file_name"] = image["file_name"][:-4] + ".jpg"

with open('annotations.json', 'w') as f:
    json.dump(annotations, f, indent=4)
    

