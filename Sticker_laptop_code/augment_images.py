import albumentations as A

import cv2

import os
import json

IMAGES_FOLDER_PATH = 'combined'
ANNOTATIONS_PATH = 'annotations.json'

def augment_images():

    images = os.listdir(IMAGES_FOLDER_PATH)
    with open(ANNOTATIONS_PATH, 'r') as f:
        annotations = json.load(f)

    transform = A.Compose([
                            A.RandomBrightnessContrast(p=0.1),
                            A.RandomCrop(width=450, height=450, p=0.1),
                            A.Rotate(p=0.5, limit=90),
                            A.RGBShift(p=0.1),
                            A.RandomSnow(p=0.1),
                            A.VerticalFlip(p=0.1),
                            A.HorizontalFlip(p=0.1),
                          ],
                           bbox_params=A.BboxParams(format='coco', min_visibility=0.01, label_fields=['class_labels'])
                           )

    # make a cv2 window
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('transformed_image', cv2.WINDOW_NORMAL)

    for image_name in images[20:]:
        image = cv2.imread(os.path.join(IMAGES_FOLDER_PATH, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('image', image)
        image_id = [image['id'] for image in annotations['images'] if image['file_name'] == image_name]
        # find the annotations for the image
        image_annotations = [annotation for annotation in annotations['annotations'] if annotation['image_id'] == image_id[0]]
        # get the bounding boxes
        bboxes = [annotation["bbox"] for annotation in image_annotations]
        class_labels = [annotation["category_id"] for annotation in image_annotations]

        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        for bbox, class_label in zip(transformed_bboxes, transformed_class_labels):
            x, y, w, h = bbox
            cv2.rectangle(transformed_image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            cv2.putText(transformed_image, str(class_label), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        for bbox, class_label in zip(bboxes, class_labels):
            x, y, w, h = bbox
            cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            cv2.putText(image, str(class_label), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('image', image)
        cv2.imshow('transformed_image', transformed_image)
        # if esc is pressed, exit
        if cv2.waitKey(0) == 27:
            break







                            

if __name__ == '__main__':
    augment_images()    

