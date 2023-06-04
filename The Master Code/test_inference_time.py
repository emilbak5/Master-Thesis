from dataset_def_pl import StickerData
from model_def_pl import StickerDetector

import torch
import torchvision
import os
from tqdm import tqdm
import time
from torchvision import transforms



NUM_WORKERS = 1
BATCH_SIZE = 6

NUM_CLASSES = 3  # logo + sticker + background

WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
LEARNING_RATE = 0.005


MODEL_NAMES = ['fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large']


CONFIG = {
    "lr": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE
    }

TRANSFORM = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def inference_time_test():

    # load the images with torch from the test folder
    images = []
    num_images = 0
    for image_name in os.listdir('data_stickers/train'):
        if num_images == 500:
            break
        if image_name.endswith('.jpg'):
            # if not image_name.startswith('augmented_'):

            image = torchvision.io.read_image('data_stickers/train/' + image_name)
            images.append(image)
            num_images += 1

    for model_name in MODEL_NAMES:
        print('Testing model: ', model_name)

        checkpoint_path = os.path.join('lightning_logs', model_name, 'version_0', 'checkpoints')
        checkpoint_path = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])
        # checkpoint_path = 'lightning_logs/fasterrcnn_resnet50_fpn/version_0/checkpoints/epoch=23-step=720.ckpt'
        # checkpoint_path = 'lightning_logs/' + model_name + '/version_0/checkpoints'
        
        

        model = StickerDetector(num_classes=NUM_CLASSES, config=CONFIG, model_name=model_name)
        model = model.load_from_checkpoint(checkpoint_path)
        model.cuda()
        model.eval()

        with torch.no_grad():
            # start timer
            start = time.time()
            for image in tqdm(images):
                
                image = TRANSFORM(image.float())
                image = [image]
                image[0] = image[0].cuda()
                image = tuple(image)

                model(image)
        
        # end timer
        time_taken = time.time() - start
        print(f'Time elapsed for {model_name}: {time_taken}')
        print(f'Time taken per image for {model_name}: {time_taken/len(images)}')



if __name__ == '__main__':
    inference_time_test()
            



