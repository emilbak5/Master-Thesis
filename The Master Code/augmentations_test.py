import albumentations as A

import cv2


import copy
import numpy as np
import os
import json
from tqdm import tqdm
import itertools
import math


IMAGES_FOLDER_PATH = 'augmentation_test/train_og'
IMAGES_MASKS_FOLDER_PATH = 'augmentation_test/train_og/masks'
ANNOTATIONS_PATH = 'augmentation_test/train/annotations.coco.json'

DATASET_MULTIPLICATION_SIZE = 3
def augment_images(some_of_transform):

    image_count = 0

    images = os.listdir(IMAGES_FOLDER_PATH)
    images = [image for image in images if image.endswith('.jpg')]
    with open(ANNOTATIONS_PATH, 'r') as f:
        annotations = json.load(f)



    for i in range(DATASET_MULTIPLICATION_SIZE):
        for image_name in tqdm(images):
            number_of_augments = np.random.randint(1, len(some_of_transform) + 1)


            transform = A.Compose([
                                A.SomeOf(some_of_transform, n=number_of_augments),
                                ],
                                bbox_params=A.BboxParams(format='coco', min_visibility=0.1),
                                )

            image = cv2.imread(os.path.join(IMAGES_FOLDER_PATH, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imshow('image', image)
            image_id = [image['id'] for image in annotations['images'] if image['file_name'] == image_name]
            # find the annotations for the image
            image_annotations = [annotation for annotation in annotations['annotations'] if annotation['image_id'] == image_id[0]]
            # print(f"Time taken to find annotations: {end - start}")
            # get the bounding boxes
            masks = [cv2.imread(os.path.join(IMAGES_MASKS_FOLDER_PATH, annotation['mask']), cv2.IMREAD_GRAYSCALE) for annotation in image_annotations]
            # masks = [cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) for mask in masks]

            bbox_category_ids = [annotation['category_id'] for annotation in image_annotations]
            bboxes = [annotation["bbox"] for annotation in image_annotations]
            for bbox, bbox_category_id in zip(bboxes, bbox_category_ids):
                bbox.append(str(bbox_category_id))

        
            # start time
            transformed = transform(image=image, masks=masks, bboxes=bboxes)
            transformed_image = transformed['image']


            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite('augmentation_test/train/' + 'augmented_image' + str(10000 + image_count) + '.jpg', transformed_image)

            image_count += 1


def get_combinations(lst):
    unique_combinations = set()
    for i in range(1, len(lst) + 1):
        for combination in itertools.combinations(lst, i):
            unique_combinations.add(tuple(sorted(combination)))
        
    unique_combinations = list(unique_combinations)
        # sort the list after the length of each tuple
    unique_combinations.sort(key=lambda x: len(x))


    return unique_combinations

def count_combinations(lst):

    n = len(lst)
    count = 0
    for k in range(1, n + 1):
        count += math.comb(n, k)
    return count


def get_mean_image(images_path):

    images = os.listdir(images_path)
    images = [image for image in images if image.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    # remove all images that starts with a
    # images = [image for image in images if not image.startswith('a')]

    mean_image = np.zeros((2048, 2448, 3), dtype=np.float32)
    for image_name in tqdm(images):
        image = cv2.imread(os.path.join(images_path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean_image += image

    mean_image /= len(images)
    return mean_image


def mutual_information(image1, image2):
    # Calculate histograms of the two images
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    
    # Normalize the histograms
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Calculate joint histogram
    image1 = image1.astype(np.uint8)
    image2 = image2.astype(np.uint8)
    joint_hist = np.zeros((256, 256))
    for i in tqdm(range(image1.shape[0])):
        for j in range(image1.shape[1]):
            joint_hist[image1[i,j], image2[i,j]] += 1 
    
    # Normalize the joint histogram
    joint_hist = joint_hist / np.sum(joint_hist)
    
    # Calculate entropy of the two images and the joint histogram
    entropy1 = -np.sum(hist1 * np.log2(hist1 + (hist1 == 0)))
    entropy2 = -np.sum(hist2 * np.log2(hist2 + (hist2 == 0)))
    joint_entropy = -np.sum(joint_hist * np.log2(joint_hist + (joint_hist == 0)))
    
    # Calculate mutual information
    mutual_info = entropy1 + entropy2 - joint_entropy
    
    return mutual_info

                     

if __name__ == '__main__':
    possible_transforms = ['1', '2', '3', '4', '5', '6', '7']
    transforms_dict = {'1': A.RandomSizedBBoxSafeCrop(height=2048, width=2448, erosion_rate=0.0, p=1),
                        '2': A.VerticalFlip(p=1),
                        '3': A.HorizontalFlip(p=1),
                        '4': A.GaussNoise(p=1, var_limit=(50.0, 400.0)),
                        '5': A.RandomBrightnessContrast(p=1, brightness_limit=[-0.1, 0.2], contrast_limit=[-0.1, 0.2]),
                        '6': A.ColorJitter(p=1, brightness=0, contrast=0, saturation=0.8, hue=0.4),
                        '7': A.GaussianBlur(p=1, blur_limit=17),
                        }
    transforms_naming_dict = {'1': 'RandomSizedBBoxSafeCrop',
                        '2': 'VerticalFlip',
                        '3': 'HorizontalFlip',
                        '4': 'GaussNoise',
                        '5': 'RandomBrightnessContrast',
                        '6': 'ColorJitter',
                        '7': 'GaussianBlur',
                        }
    
    # some_of_transform = [transforms_dict[transform] for transform in ['1', '2', '3', '4', '5', '6', '7']]
    # augment_images(some_of_transform)

    combinations = get_combinations(possible_transforms)
    print(f"Number of combinations: {len(combinations)}, double checked to be {count_combinations(possible_transforms)}")

    validation_mean_image = get_mean_image('augmentation_test/valid')
    test_mean_image = get_mean_image('augmentation_test/test')
    train_mean_image = get_mean_image('augmentation_test/train')

    cv2.imwrite('augmentation_test/mean_images/' + 'train_mean_image_without_aug.jpg', train_mean_image)
    cv2.imwrite('augmentation_test/mean_images/' + 'validation_mean_image_without_aug.jpg', validation_mean_image)
    cv2.imwrite('augmentation_test/mean_images/' + 'test_mean_image_without_aug.jpg', test_mean_image)

    mutual_info_valtest = mutual_information(validation_mean_image, test_mean_image)
    print(f"Mutual information between validation and test: {mutual_info_valtest}")
    mutual_info_trainval = mutual_information(train_mean_image, validation_mean_image)
    print(f"Mutual information between train and validation: {mutual_info_trainval}")
    mutual_info_train_test = mutual_information(train_mean_image, test_mean_image)
    print(f"Mutual information between train and test: {mutual_info_train_test}")

    # save mutual info as csv
    with open('augmentation_test/mutual_info.csv', 'w') as f:
        f.write('train_val,train_test,val_test,combination\n')
        f.write(f'{mutual_info_trainval},{mutual_info_train_test},{mutual_info_valtest},"no augment"\n')

    

    for i, combination in tqdm(enumerate(combinations)):
        some_of_transform = [transforms_dict[transform] for transform in combination]
        some_of_transform_names = [transforms_naming_dict[transform] for transform in combination]
        print(f"Augmenting with: {combination}, {some_of_transform_names}")
        
        augment_images(some_of_transform)

        train_mean_image = get_mean_image('augmentation_test/train')
        cv2.imwrite('augmentation_test/mean_images/' + 'train_mean_image' + str(i) + '.jpg', train_mean_image)

        mutual_info_trainval = mutual_information(train_mean_image, validation_mean_image)
        print(f"Mutual information between train and validation: {mutual_info_trainval}")
        mutual_info_train_test = mutual_information(train_mean_image, test_mean_image)
        print(f"Mutual information between train and test: {mutual_info_train_test}")

        combination_joined = ','.join(combination)
        with open('augmentation_test/mutual_info.csv', 'a') as f:
            f.write(f'{mutual_info_trainval},{mutual_info_train_test},{mutual_info_valtest},{combination_joined}\n')





