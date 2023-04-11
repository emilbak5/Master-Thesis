import albumentations as A

import cv2


import copy
import numpy as np
import os
import json
from tqdm import tqdm

IMAGES_FOLDER_PATH = 'annotated_top'
IMAGES_MASKS_FOLDER_PATH = 'annotated_top/masks'
ANNOTATIONS_PATH = 'annotations_test.json'

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

    

    # make a cv2 window
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('transformed_image', cv2.WINDOW_NORMAL)

    for i in range(DATASET_MULTIPLICATION_SIZE):
        for image_name in tqdm(images):
            number_of_augments = np.random.randint(1, 5)

            transform = A.Compose([
                                A.SomeOf([
                                    A.OneOf([
                                        A.Rotate(p=1, limit=180),
                                        A.VerticalFlip(p=1),
                                        A.HorizontalFlip(p=1),
                                    ], p=1),
                                    A.GaussNoise(p=1),
                                    A.RandomBrightnessContrast(p=1, brightness_limit=[-0.1, 0.2]),
                                    A.ColorJitter(p=1, brightness=0, contrast=0),
                                ], n=number_of_augments),
                                ],
                                # bbox_params=A.BboxParams(format='coco', min_visibility=0.01),
                                keypoint_params=A.KeypointParams(format='xy', remove_invisible=True, label_fields=['class_labels'])
                                )

            image = cv2.imread(os.path.join(IMAGES_FOLDER_PATH, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imshow('image', image)
            image_id = [copy.deepcopy(image['id']) for image in annotations['images'] if image['file_name'] == image_name]
            # find the annotations for the image
            image_annotations = [copy.deepcopy(annotation) for annotation in annotations['annotations'] if annotation['image_id'] == image_id[0]]
            annotation_ids = [copy.deepcopy(annotation['id']) for annotation in image_annotations]
            # get the bounding boxes
            segmentations = [copy.deepcopy(annotation['segmentation']) for annotation in image_annotations]
            segmentations_new = []
            for segmentation in segmentations:
                # convert segmentation to this formart [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] from  [[x1, y1, x2, y2, x3, y3, x4, y4]]
                segmentation = [tuple(segmentation[0][i:i+2]) for i in range(0, len(segmentation[0]), 2)]
                segmentations_new.append(segmentation)
            
            segmentations_lengths = [len(segmentation) for segmentation in segmentations_new]
            bboxes = [copy.deepcopy(annotation["bbox"]) for annotation in image_annotations]
            class_labels = [annotation["category_id"] for annotation in image_annotations]
            class_labels_for_segment = []
            for i, label in enumerate(class_labels):
                class_label_for_segment = [label] * len(segmentations_new[i])
                class_labels_for_segment.append(class_label_for_segment)

            # flatten segmentations_new and class_labels_for_segment
            segmentations_new = [segmentation for segmentation in segmentations_new for segmentation in segmentation]
            class_labels_for_segment = [class_label for class_label in class_labels_for_segment for class_label in class_label]


            # transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            for bounding_box in bboxes:
                bounding_box.append('None')

            transformed = transform(image=image, keypoints=segmentations_new, class_labels=class_labels_for_segment)
            transformed_image = transformed['image']
            # transformed_bboxes = transformed['bboxes']
            # transformed_class_labels = transformed['class_labels']
            transformed_segmentations = transformed['keypoints']

            transformed_segmentations_splits = []
            # split transformed_segmentations into segments
            for length in segmentations_lengths:
                transformed_segmentations_splits.append(transformed_segmentations[:length])
                transformed_segmentations = transformed_segmentations[length:]



            # for segmentation, class_label in zip(transformed_segmentations, transformed_class_labels):
                # segmentation = [tuple(segmentation[i]) for i in range(len(segmentation))]
            bounding_boxes_new = []
            for transformed_segmentation_split in transformed_segmentations_splits:
                transformed_segmentation_split = np.array(transformed_segmentation_split).astype(np.int32)
                #find smallest x and y and largest x and y in transformed_segmentation_split
                x_min = int(np.min(transformed_segmentation_split[:, 0]))
                y_min = int(np.min(transformed_segmentation_split[:, 1]))
                x_max = int(np.max(transformed_segmentation_split[:, 0]))
                y_max = int(np.max(transformed_segmentation_split[:, 1]))


                bounding_box = [x_min, y_min, x_max - x_min, y_max - y_min]
                bounding_boxes_new.append(bounding_box)

                transformed_segmentation_split = transformed_segmentation_split.reshape(-1,1,2)
                # cv2.polylines(transformed_image, [transformed_segmentation_split], True, (0, 255, 0), 2)
                # cv2.rectangle(transformed_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            annotations['images'].append({
                "id": 10000 + image_count,
                "width": transformed_image.shape[1],
                "height": transformed_image.shape[0],
                "file_name": 'augmented_image' + str(10000 + image_count) + '.jpg',
            })
            transformed_segmentations_splits_original_format = []
            for transformed_segmentations_split in transformed_segmentations_splits:
                # return transformed_segmentations_split to the original format: [x1, y1, x2, y2, x3, y3, x4, y4]
                transformed_segmentations_split = [float(segmentation) for segmentation in transformed_segmentations_split for segmentation in segmentation]
                transformed_segmentations_splits_original_format.append(transformed_segmentations_split)
            
            assert len(transformed_segmentations_splits_original_format) == len(bounding_boxes_new) == len(class_labels) == len(annotation_ids)

            for transformed_segment, transformed_bbox, class_label, id in zip(transformed_segmentations_splits_original_format, bounding_boxes_new, class_labels, annotation_ids):
                
                new_anno = {
                    "id": 10000 + annotation_count,
                    "image_id": 10000 + image_count,
                    "category_id": class_label,
                    "segmentation": [transformed_segment],
                    "bbox": transformed_bbox,
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": transformed_bbox[2] * transformed_bbox[3]
                }
                annotations['annotations'].append(new_anno)
                annotation_count += 1
            # cv2.putText(transformed_image, str(class_label[0]), (int(segmentation[0][0]), int(segmentation[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # for segmentation, class_label in zip(segmentations_new, class_labels_for_segment):
                # segmentation = [tuple(segmentation[i]) for i in range(len(segmentation))]
            # cv2.polylines(image, [transformed_segmentations], True, (0, 255, 0), 2)
            # cv2.putText(image, str(class_label[0]), (int(segmentation[0][0]), int(segmentation[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # for bbox, class_label in zip(transformed_bboxes, transformed_class_labels):
            #     x, y, w, h = bbox
            #     cv2.rectangle(transformed_image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            #     cv2.putText(transformed_image, str(class_label), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # for bbox, class_label in zip(bboxes, class_labels):
            #     x, y, w, h = bbox
            #     cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            #     cv2.putText(image, str(class_label), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # cv2.imwrite('data_stickers/train/' + 'augmented_image' + str(10000 + image_count) + '.jpg', transformed_image)
            if not os.path.exists('data_stickers/augmented/'):
                os.makedirs('data_stickers/augmented/')
            cv2.imwrite('data_stickers/augmented/' + 'augmented_image' + str(10000 + image_count) + '.jpg', transformed_image)
            image_count += 1


    test = find_types(annotations)

    # with open('data_stickers/train/annotations.coco.json', 'w') as outfile:
    #     json.dump(annotations, outfile, indent=4)
    with open('data_stickers/augmented/annotations.coco.json', 'w') as outfile:
        json.dump(annotations, outfile, indent=4)
            # cv2.imshow('image', image)
            # cv2.imshow('transformed_image', transformed_image)
            # # if esc is pressed, exit
            # if cv2.waitKey(0) == 27:
            #     break







                            

if __name__ == '__main__':
    augment_images()    

