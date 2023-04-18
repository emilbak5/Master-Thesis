from dataset_def_pl import StickerData
from model_def_pl import StickerDetector
from utils import show_10_images_with_bounding_boxes


import os



NUM_IMAGES = 33

NUM_WORKERS = 0
BATCH_SIZE = 6

NUM_CLASSES = 3  # logo + sticker + background

WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

# MODEL_NAME = 'fasterrcnn_resnet50_fpn'
# LEARNING_RATE = 0.005 # used for fasterrcnn_resnet50_fpn

MODEL_NAME = 'fasterrcnn_resnet50_fpn_v2'
LEARNING_RATE = 0.005 # used for fasterrcnn_resnet50_fpn

# MODEL_NAME = 'ssd300_vgg16'
# LEARNING_RATE = 0.0005 # better for ssd300_vgg16

# MODEL_NAME = 'retinanet_resnet50_fpn'
# LEARNING_RATE = 0.005

TRAINING_VERSION = 2

CHECKPOINT_PATH = 'lightning_logs/' + MODEL_NAME + '/version_' + str(TRAINING_VERSION) + '/checkpoints'

CONFIG = {
    "lr": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY,
    }


if __name__ == '__main__':

    checkpoint_path = os.path.join(CHECKPOINT_PATH, os.listdir(CHECKPOINT_PATH)[0])

    data_module = StickerData(train_folder='data_stickers/train', valid_folder='data_stickers/valid', test_folder='data_stickers/test', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = StickerDetector(num_classes=NUM_CLASSES, config=CONFIG, batch_size=BATCH_SIZE, model_name=MODEL_NAME)

    show_10_images_with_bounding_boxes(model, data_module, checkpoint_path, num_images=NUM_IMAGES)