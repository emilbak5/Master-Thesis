import pytorch_lightning as pl
import torchmetrics

from dataset_def_pl import StickerData
from model_def_pl import StickerDetector

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from lightning.pytorch.accelerators import find_usable_cuda_devices

from utils import push_results_to_iphone
import torch
import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")



NUM_WORKERS = 2
NUM_CLASSES = 3 # logo + sticker + background
    


# MODEL_NAME = 'fasterrcnn_resnet50_fpn'
# LEARNING_RATE = 0.005 # used for fasterrcnn_resnet50_fpn

MODEL_NAME = 'fasterrcnn_resnet50_fpn_v2'
LEARNING_RATE = 0.005 # used for fasterrcnn_resnet50_fpn
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
BATCH_SIZE = 3

# MODEL_NAME = 'ssd300_vgg16'
# LEARNING_RATE = 0.0004 # better for ssd300_vgg16

# MODEL_NAME = 'ssdlite320_mobilenet_v3_large'
# LEARNING_RATE = 0.0004

# MODEL_NAME = 'retinanet_resnet50_fpn'
# LEARNING_RATE = 0.001

# MODEL_NAME = 'retinanet_resnet50_fpn_v2'
# LEARNING_RATE = 0.002




CONFIG = {
    "lr": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE
    }

if __name__ == '__main__':


    torch.set_float32_matmul_precision("medium")

    logger = TensorBoardLogger('lightning_logs', name=MODEL_NAME, default_hp_metric=True, log_graph=False)
    early_stopping_callback = EarlyStopping(monitor='Validation/mAP', min_delta=0.001, patience=4, verbose=True, mode='max', check_on_train_epoch_end=False)
    SWA_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    # torch check for cuda devices
    # print(torch.cuda.is_available())
    # print(find_usable_cuda_devices(1))
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=find_usable_cuda_devices(1), 
        max_epochs=70, 
        logger=logger, 
        limit_train_batches=0.25,
        check_val_every_n_epoch=1,
        # val_check_interval=0.1, 
        log_every_n_steps=1, 
        auto_scale_batch_size='binsearch', 
        num_sanity_val_steps=0,
        accumulate_grad_batches=4,
        
        callbacks=[early_stopping_callback, SWA_callback]
        )


    data_module = StickerData(train_folder='data_stickers/train', valid_folder='data_stickers/valid', test_folder='data_stickers/test', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = StickerDetector(num_classes=NUM_CLASSES, config=CONFIG, model_name=MODEL_NAME)
    
    # trainer.tune(model, datamodule=data_module)

    torch.cuda.empty_cache()
    data_module.setup('fit')
    trainer.validate(model, datamodule=data_module)
    trainer.fit(model, data_module)

    # push_results_to_iphone(trainer=trainer, model=model, datamodule=data_module)



