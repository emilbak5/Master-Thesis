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

    

    # # make a cv2 window
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('old_image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('old_masks', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('new_masks', cv2.WINDOW_NORMAL)

    for i in range(DATASET_MULTIPLICATION_SIZE):
        for image_name in tqdm(images):
            number_of_augments = np.random.randint(1, 6)

            transform = A.Compose([
                                A.SomeOf([
                                    A.RandomSizedBBoxSafeCrop(height=2048, width=2448, erosion_rate=0.0, p=1), # erosion rate determines how much of bbox must remain
                                    # A.Rotate(p=1, limit=180, border_mode=cv2.BORDER_CONSTANT, value=0),
                                    A.VerticalFlip(p=1),
                                    A.HorizontalFlip(p=1),
                                    A.GaussNoise(p=1),
                                    A.RandomBrightnessContrast(p=1, brightness_limit=[-0.1, 0.2]),
                                    A.ColorJitter(p=1, brightness=0, contrast=0, saturation=0.4, hue=0.4),
                                ], n=number_of_augments),
                                ],
                                bbox_params=A.BboxParams(format='coco', min_visibility=0.1),
                                # keypoint_params=A.KeypointParams(format='xy', remove_invisible=True, label_fields=['class_labels'])
                                )

            image = cv2.imread(os.path.join(IMAGES_FOLDER_PATH, image_name))
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



            combined_masks_old = np.zeros_like(masks[0])
            combined_masks_new = np.zeros_like(transformed_masks[0])
            transformed_bboxes = []
            for old_mask, new_mask in zip(masks, transformed_masks):
                combined_masks_old = np.bitwise_or(combined_masks_old, old_mask)
                combined_masks_new = np.bitwise_or(combined_masks_new, new_mask)
                # find the bounding box of the new mask
                x_min, y_min, w, h = cv2.boundingRect(new_mask)
                transformed_bboxes.append([x_min, y_min, w, h])
            
            # combined_masks_new = cv2.cvtColor(combined_masks_new, cv2.COLOR_GRAY2BGR)
            # for bbox in transformed_bboxes:
                # cv2.rectangle(transformed_image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
                # cv2.rectangle(combined_masks_new, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
            # for bbox in transformed_bboxes_:
            #     cv2.rectangle(transformed_image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 0, 255), 2)




            # cv2.imshow('old_masks', combined_masks_old)
            # cv2.imshow('new_masks', combined_masks_new)
            # cv2.imshow('image', transformed_image)
            # cv2.imshow('old_image', image)
            # cv2.waitKey(0)

            
            

            # bounding_boxes_new = []
            # for transformed_segmentation_split in transformed_segmentations_splits:
            #     transformed_segmentation_split = np.array(transformed_segmentation_split).astype(np.int32)
            #     #find smallest x and y and largest x and y in transformed_segmentation_split
            #     x_min = int(np.min(transformed_segmentation_split[:, 0]))
            #     y_min = int(np.min(transformed_segmentation_split[:, 1]))
            #     x_max = int(np.max(transformed_segmentation_split[:, 0]))
            #     y_max = int(np.max(transformed_segmentation_split[:, 1]))


            #     bounding_box = [x_min, y_min, x_max - x_min, y_max - y_min]
            #     bounding_boxes_new.append(bounding_box)

            #     transformed_segmentation_split = transformed_segmentation_split.reshape(-1,1,2)
                # cv2.polylines(transformed_image, [transformed_segmentation_split], True, (0, 255, 0), 2)
                # cv2.rectangle(transformed_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            annotations['images'].append({
                "id": 10000 + image_count,
                "width": transformed_image.shape[1],
                "height": transformed_image.shape[0],
                "file_name": 'augmented_image' + str(10000 + image_count) + '.jpg',
            })
            
            assert len(transformed_bboxes) == len(annotation_ids)

            for transformed_bbox, category_id, mask in zip(transformed_bboxes, bbox_category_ids, transformed_masks):
                
                new_anno = {
                    "id": 10000 + annotation_count,
                    "image_id": 10000 + image_count,
                    "category_id": category_id,
                    "segmentation": [0, 0],
                    "bbox": transformed_bbox,
                    "mask": str(10000 + annotation_count) + '.png',
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": transformed_bbox[2] * transformed_bbox[3]
                }
                annotations['annotations'].append(new_anno)
                # start time
                start = time.time()
                cv2.imwrite('data_stickers/augmented/masks/' + str(10000 + annotation_count) + '.png', mask)
                cv2.imwrite('data_stickers/train/masks/' + str(10000 + annotation_count) + '.png', mask)
                end = time.time()
                #print(f"Time taken to save mask: {end - start}")
                annotation_count += 1



            cv2.imwrite('data_stickers/train/' + 'augmented_image' + str(10000 + image_count) + '.jpg', transformed_image)
            if not os.path.exists('data_stickers/augmented/'):
                os.makedirs('data_stickers/augmented/')
            start = time.time()
            cv2.imwrite('data_stickers/augmented/' + 'augmented_image' + str(10000 + image_count) + '.jpg', transformed_image)
            end = time.time()
            image_count += 1


    test = find_types(annotations)

    with open('data_stickers/train/annotations.coco.json', 'w') as outfile:
        json.dump(annotations, outfile, indent=4)
    with open('data_stickers/augmented/annotations.coco.json', 'w') as outfile:
        json.dump(annotations, outfile, indent=4)
            # cv2.imshow('image', image)
            # cv2.imshow('transformed_image', transformed_image)
            # # if esc is pressed, exit
            # if cv2.waitKey(0) == 27:
            #     break







                            

if __name__ == '__main__':
    augment_images()    

