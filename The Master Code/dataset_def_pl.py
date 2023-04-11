import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_lightning.utilities.fetching import AbstractDataFetcher


import pytorch_lightning as pl

import json
import os
from PIL import Image
import cv2 as cv





def collate_fn(batch):
    return tuple(zip(*batch))



class StickerDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, transforms=None):
        super(StickerDataset, self).__init__()

        self.transforms = transforms

        self.images = os.listdir(os.path.join(images_path))
        self.images = [images_path + '/' + image for image in self.images]
        self.images = [file for file in self.images if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

        with open(images_path + '/annotations.coco.json') as f:
            self.annotations = json.load(f)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")

        # get the image name
        image_name = self.images[idx].split('/')[-1]

        # get the image id and annotations for the image
        image_id = [self.annotations['images'][i]['id'] for i in range(len(self.annotations['images'])) if self.annotations['images'][i]['file_name'] == image_name]
        if len(image_id) != 1:
            print(image_name)
        assert len(image_id) == 1
        image_id = image_id[0]

        annotations = [annotation for annotation in self.annotations['annotations'] if annotation['image_id'] == image_id]

        
        boxes = []
        areas = []
        labels = []
        for annotation in annotations:
            x, y, w, h = annotation['bbox']
            area = annotation['area']
            label = annotation['category_id']
            areas.append(area)
            boxes.append([x, y, x+w, y+h])
            labels.append(label)
            
        

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.int64)
        image_id = torch.tensor([image_id])




        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = areas
        target['iscrowd'] = iscrowd
        target['image_name'] = image_name



        if self.transforms is not None:
            img = self.transforms(img)

        return img, target



class StickerData(pl.LightningDataModule):

    def __init__(self, train_folder, valid_folder, test_folder, batch_size=2, num_workers=2):
        super().__init__()
        
        self.train_folder = train_folder
        self.valid_folder = valid_folder
        self.test_folder = test_folder
        self.batch_size = batch_size
        self.num_workers = num_workers


        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    
    
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders

        if stage == "fit":
            self.train_dataset = StickerDataset(self.train_folder, transforms=self.transform)
            self.valid_dataset = StickerDataset(self.valid_folder, transforms=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = StickerDataset(self.test_folder, transforms=self.transform)
    
    def train_dataloader(self):
        if self.num_workers == 0:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=False)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        if self.num_workers == 0:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=False)
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        if self.num_workers == 0:
            return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=False)
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
    
