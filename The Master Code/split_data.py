import torch
import glob
import json
import shutil
import os
from tqdm import tqdm


def split_data(data_raw_path='data_raw_stickers/', val_size=0.2, test_size=0.1, random_seed=42):

    # get a list of all the images in the dataset
    images_list = glob.glob(data_raw_path + "*.jpg")
    dataset_size = len(images_list)

    val_length = int(dataset_size * val_size)
    test_length = int(dataset_size * test_size)
    train_length = dataset_size - val_length - test_length

    assert train_length + val_length + test_length == dataset_size, "The dataset size is not equal to the sum of the splits"

    # create the splits
    train_set, val_set, test_set = torch.utils.data.random_split(images_list, [train_length, val_length, test_length], generator=torch.Generator().manual_seed(random_seed))

    # train_set = train_set.dataset
    # val_set = val_set.dataset
    # test_set = test_set.dataset
    train_set = [images_list[image] for image in train_set.indices]
    val_set = [images_list[image] for image in val_set.indices]
    test_set = [images_list[image] for image in test_set.indices]

    # remove data_raw/ from the file names
    train_set = [image[18:] for image in train_set]
    val_set = [image[18:] for image in val_set]
    test_set = [image[18:] for image in test_set]


    # read data_raw\_annotations.coco.json
    with open(data_raw_path + "result.json") as f:
        annotations = json.load(f)
    
    train_annotations = {'categories': annotations['categories'], 'images': [], 'annotations': []}
    val_annotations = {'categories': annotations['categories'], 'images': [], 'annotations': []}
    test_annotations = {'categories': annotations['categories'], 'images': [], 'annotations': []}



    # create the annotations for each split
    for image in annotations['images']:
        if image['file_name'][61:] in train_set:
            train_annotations['images'].append(image)
        elif image['file_name'][61:] in val_set:
            val_annotations['images'].append(image)
        elif image['file_name'][61:] in test_set:
            test_annotations['images'].append(image)
        else:
            print("Image not in any split")
    
    for annotation in annotations['annotations']:
        for image in train_annotations['images']:
            if annotation['image_id'] == image['id']:
                train_annotations['annotations'].append(annotation)
        for image in val_annotations['images']:
            if annotation['image_id'] == image['id']:
                val_annotations['annotations'].append(annotation)
        for image in test_annotations['images']:
            if annotation['image_id'] == image['id']:
                test_annotations['annotations'].append(annotation)

    # # for each annotation change the category_id to 1'
    # for annotation in train_annotations['annotations']:
    #     annotation['category_id'] = 1

    
    # create the folders for the splits if they don't exist
    if not os.path.exists("data_stickers/train"):
        os.makedirs("data_stickers/train")
    if not os.path.exists("data_stickers/valid"):
        os.makedirs("data_stickers/valid")
    if not os.path.exists("data_stickers/test"):
        os.makedirs("data_stickers/test")
    
    
    # write the annotations to json files
    with open("data_stickers/train/annotations.coco.json", 'w') as f:
        json.dump(train_annotations, f, indent=4)
    with open("data_stickers/valid/annotations.coco.json", 'w') as f:
        json.dump(val_annotations, f, indent=4)
    with open("data_stickers/test/annotations.coco.json", 'w') as f:
        json.dump(test_annotations, f, indent=4)

    # move the images to their respective folders
    for image in tqdm(train_set):
        shutil.copy(data_raw_path + image, "data_stickers/train/" + image)
    for image in tqdm(val_set):
        shutil.copy(data_raw_path + image, "data_stickers/valid/" + image)
    for image in tqdm(test_set):
        shutil.copy(data_raw_path + image, "data_stickers/test/" + image)




if __name__ == "__main__":
    split_data()
    

    
    


