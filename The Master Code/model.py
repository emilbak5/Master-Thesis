import numpy as np
import torch
import torchvision

import os
import json
from PIL import Image
import cv2 as cv

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



class StickerDetector(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(StickerDetector, self).__init__()

        # load the pretrained model: Mask R-CNN
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # self.model = torch.hub.load('pytorch/vision:v0.6.0', 'maskrcnn_resnet50_fpn', pretrained=True)

        # get the number of input features for the classifier (is needed when changing the nr of classes)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    def forward(self, images, targets=None):
        output = self.model(images, targets)
        return output



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
        # img = cv.imread(self.images[idx])
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.open(self.images[idx]).convert("RGB")

        # get the image name
        image_name = self.images[idx].split('/')[-1]

        # get the image id and annotations for the image
        image_id = [self.annotations['images'][i]['id'] for i in range(len(self.annotations['images'])) if self.annotations['images'][i]['file_name'] == image_name]
        assert len(image_id) == 1
        image_id = image_id[0]

        annotations = [annotation for annotation in self.annotations['annotations'] if annotation['image_id'] == image_id]
        
        boxes = []
        areas = []
        for annotation in annotations:
            x, y, w, h = annotation['bbox']
            area = annotation['area']
            areas.append(area)
            boxes.append([x, y, x+w, y+h])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(annotations),), dtype=torch.int64)
        # Convert all to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(annotations),), dtype=torch.int64)

        areas = torch.as_tensor(areas, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([image_id])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = areas
        target['iscrowd'] = iscrowd



        if self.transforms is not None:
            img = self.transforms(img)

        return img, target



# dataset = StickerDataset("data/train", "data/train/train_annotations.coco.json")
# for i in range(10):
#     img, target = dataset[i]
#     # img = img.numpy()
#     # img = img.
#     # get the bounding boxes
#     boxes = target['boxes']
#     # draw the bounding boxes in the image
#     for box in boxes:
#         x1, y1, x2, y2 = box.numpy()
#         cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

#     # show img
#     cv.imshow("img", img)
#     cv.waitKey(0)

