import os
import json
from tqdm import tqdm

import cv2 as cv


images = os.listdir("test_data")
images = [image for image in images if image.endswith(".jpg")]
with open("annotations_test_data.json", "r") as f:
    annotations = json.load(f)

# cv.namedWindow("image", cv.WINDOW_NORMAL)

# check that the folder exists
if not os.path.exists("combined_bboxes"):
    os.mkdir("combined_bboxes")
    

for image_name in tqdm(images):

    img = cv.imread(os.path.join("test_data", image_name))
    image_id = [image['id'] for image in annotations['images'] if image['file_name'] == image_name]

    if len(image_id) > 1:
        x = 5
    image_id = image_id[0]    
    assert image_id != None, "Image id not found"

    # find the annotations for the image
    image_annotations = [annotation for annotation in annotations['annotations'] if annotation['image_id'] == image_id]

    for annotation in image_annotations:
        x, y, w, h = annotation["bbox"]
        color = (0, 255, 0) if annotation["category_id"] == 1 else (0, 0, 255)
        cv.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)



    cv.imwrite(os.path.join("combined_bboxes", image_name), img)
    # cv.imshow("image", img)
    # key = cv.waitKey(0)
    # if key == ord('q'):
    #     break
