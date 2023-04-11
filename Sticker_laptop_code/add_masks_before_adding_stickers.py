import os
import json

import cv2 as cv
import numpy as np

import copy

from tqdm import tqdm


IMAGES_PATH = 'annotated_top_open'
ANNOTATIONS_PATH = 'annotations_top_open.json'

# cv.namedWindow('mask', cv.WINDOW_NORMAL)
# cv.namedWindow('img', cv.WINDOW_NORMAL)

images = os.listdir(IMAGES_PATH)
images = [image for image in images if image.endswith('.jpg')]
if not os.path.exists(IMAGES_PATH + '/masks'):
    os.mkdir(IMAGES_PATH + '/masks')

with open(ANNOTATIONS_PATH, 'r') as f:
    annotations = json.load(f)

for image in tqdm(images):
    img = cv.imread(os.path.join(IMAGES_PATH, image))
    img_id = [copy.deepcopy(annotation['id']) for annotation in annotations['images'] if annotation['file_name'] == image][0]
    img_annotations = [annotation for annotation in annotations['annotations'] if annotation['image_id'] == img_id]

    for annotation in img_annotations:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        segmentation = np.array(copy.deepcopy(annotation['segmentation'][0]))
        # convert to [[x1, y1], [x2, y2], ...] from [x1, y1, x2, y2, ...]
        segmentation = segmentation.reshape(-1, 2).astype(np.int32)
        segmentation = np.array(segmentation)
        mask = cv.fillPoly(mask, [segmentation], 255)
        test = np.unique(mask)
        mask_name = str(annotation['id']) + '.png' # must be png for lossless compression
        cv.imwrite(os.path.join(IMAGES_PATH + '/masks', mask_name), mask)
        annotation["mask"] = mask_name

with open('annotations_top_open.json', 'w') as f:
    json.dump(annotations, f, indent=4)