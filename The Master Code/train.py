import numpy as np
import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
from model import StickerDetector, StickerDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def main():



    # define the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # tranforms for the dataset that includes resizing, normalization and converting to tensor
    my_transforms = transforms.Compose([
            # transforms.ConvertImageDtype(torch.float),
            # transforms.Resize(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # define the dataset
    dataset = StickerDataset(images_path='data/train', annotations_path='data/train/train_annotations.coco.json', transforms=my_transforms)

    # define the data loader
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=collate_fn)

    # define the model
    model = StickerDetector()
    model.to(device)

    # define the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # define the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # train the model
    model.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, targets in tqdm(data_loader):
            # move the images and targets to the device
            images = list(image.to(device) for image in images)
            
            # copy targets and remove all keys exept boxes and labels
            boxes_labels = []
            for target in targets:
                boxes_labels.append({'boxes': target['boxes'], 'labels': target['labels']})
            
            boxes_labels = [{k: v.to(device) for k, v in t.items()} for t in boxes_labels]


            # train the model
            loss_dict = model(images, boxes_labels)

            # get the losses
            losses = sum(loss for loss in loss_dict.values())

            # backpropagate the losses
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # update the learning rate
        lr_scheduler.step()

        # print the losses
        print(f'Epoch {epoch+1}/{num_epochs}')
        for key, value in loss_dict.items():
            print(f'{key}: {value}')

        print()






if __name__ == '__main__':
    main()










