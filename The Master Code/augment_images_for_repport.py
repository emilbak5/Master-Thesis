import albumentations as A

import cv2


import copy
import numpy as np
import os
import json
from tqdm import tqdm
import time

IMAGES_FOLDER_PATH = 'data_stickers/train'
IMAGES_MASKS_FOLDER_PATH = 'data_stickers/train/masks'
ANNOTATIONS_PATH = 'data_stickers/train/annotations.coco.json'

DATASET_MULTIPLICATION_SIZE = 1


def convert_np_arrays_to_lists(data):
    if isinstance(data, dict):
        return {k: convert_np_arrays_to_lists(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_np_arrays_to_lists(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def find_types(data):
    encountered_types = []
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, (list, tuple)):
        items = enumerate(data)
    else:
        # check if type is a numpy type
        if type(data).__module__ == np.__name__:
            x = 5
        return [type(data)]
    
    for key, value in items:
        val_type = type(value)
        if val_type not in encountered_types:
            encountered_types.append(val_type)
        nested_types = find_types(value)
        for t in nested_types:
            if t not in encountered_types:
                encountered_types.append(t)
                
    return encountered_types

def augment_images():

    image_count = 0
    annotation_count = 0

    images = os.listdir(IMAGES_FOLDER_PATH)
    images = [image for image in images if image.endswith('.jpg')]
    with open(ANNOTATIONS_PATH, 'r') as f:
        annotations = json.load(f)

    # check if folder exists
    if not os.path.exists('data_stickers/augmented/masks'):
        os.makedirs('data_stickers/augmented/masks')

    augmentations = [
                                A.RandomSizedBBoxSafeCrop(height=2048, width=2448, erosion_rate=0.0, p=1), # erosion rate determines how much of bbox must remain
                                A.SafeRotate(p=1, limit=180, border_mode=cv2.BORDER_CONSTANT, value=0),
                                A.VerticalFlip(p=1),
                                A.HorizontalFlip(p=1),
                                # A.GaussNoise(p=1),
                                A.RandomBrightnessContrast(p=1, brightness_limit=[-0.1, 0.2]),
                                A.ColorJitter(p=1, brightness=0, contrast=0, saturation=0.4, hue=0.4),
                                A.GaussianBlur(p=1, blur_limit=17),

                                ]

    # # make a cv2 window
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('old_image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('old_masks', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('new_masks', cv2.WINDOW_NORMAL)

    for i in range(DATASET_MULTIPLICATION_SIZE):
        for image_name in tqdm(images[1512:1513]):
            number_of_augments = np.random.randint(1, 7)
            for augment in augmentations:
                transform = A.Compose([
                                    A.SomeOf([augment

                                    ], n=1),
                                    ],
                                    bbox_params=A.BboxParams(format='coco', min_visibility=0.1),
                                    # keypoint_params=A.KeypointParams(format='xy', remove_invisible=True, label_fields=['class_labels'])
                                    )

                image = cv2.imread(os.path.join(IMAGES_FOLDER_PATH, image_name))
                cv2.imwrite('augmented_images_for_report/og_image' + str(10000 + image_count) + '.jpg', image)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # cv2.imshow('image', image)
                image_id = [copy.deepcopy(image['id']) for image in annotations['images'] if image['file_name'] == image_name]
                # find the annotations for the image
                image_annotations = [copy.deepcopy(annotation) for annotation in annotations['annotations'] if annotation['image_id'] == image_id[0]]
                annotation_ids = [copy.deepcopy(annotation['id']) for annotation in image_annotations]
                # get the bounding boxes
                masks = [cv2.imread(os.path.join(IMAGES_MASKS_FOLDER_PATH, annotation['mask']), cv2.IMREAD_GRAYSCALE) for annotation in image_annotations]
                # masks = [cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) for mask in masks]

                bbox_category_ids = [copy.deepcopy(annotation['category_id']) for annotation in image_annotations]
                bboxes = [copy.deepcopy(annotation["bbox"]) for annotation in image_annotations]
                for bbox, bbox_category_id in zip(bboxes, bbox_category_ids):
                    bbox.append(str(bbox_category_id))

            
                # start time
                start = time.time()
                transformed = transform(image=image, masks=masks, bboxes=bboxes)
                end = time.time()
                # print(f"Time taken to transform image: {end - start}")
                transformed_image = transformed['image']
                transformed_bboxes_ = transformed['bboxes']
                transformed_bboxes_ = [list(bbox[:-1]) for bbox in transformed_bboxes_]
                # transformed_cla_ss_labels = transformed['class_labels']
                transformed_masks = transformed['masks']

            

                
                transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)


                cv2.imwrite('augmented_images_for_report/augmented' + str(10000 + image_count) + '.jpg', transformed_image)
                end = time.time()
                image_count += 1










                            

if __name__ == '__main__':
    augment_images()    

