import os
import json
import cv2 as cv

folder = "annotated_top"

images = os.listdir(folder)
with open("annotations.json", "r") as f:
    annotations = json.load(f)

for image in images:
    if image.endswith(".png"):
        img = cv.imread(os.path.join(folder, image))
        cv.imwrite(os.path.join(folder, image[:-4] + ".jpg"), img)
        #delete the png
        os.remove(os.path.join(folder, image))

for image in annotations["images"]:
    if image["file_name"].endswith(".png"):
        image["file_name"] = image["file_name"][:-4] + ".jpg"

with open("annotations.json", "w") as f:
    json.dump(annotations, f, indent=4)

