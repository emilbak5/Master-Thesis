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
    


MODEL_NAME = 'fasterrcnn_resnet50_fpn'
LEARNING_RATE = 0.008411556810718087 # from best experiment
WEIGHT_DECAY = 0.00018685388268808122
MOMENTUM = 0.9709801785257793
BATCH_SIZE = 1

# MODEL_NAME = 'fasterrcnn_resnet50_fpn_v2'
# LEARNING_RATE = 0.008661298311453684 # from best experiment
# WEIGHT_DECAY = 3.937587898215636e-05
# MOMENTUM = 0.9288785880822662
# BATCH_SIZE = 1

# MODEL_NAME = 'ssd300_vgg16'
# LEARNING_RATE = 0.0078 # better for ssd300_vgg16
# WEIGHT_DECAY = 4.554e-05
# MOMENTUM = 0.9848
# BATCH_SIZE = 2

# MODEL_NAME = 'ssdlite320_mobilenet_v3_large'
# LEARNING_RATE = 0.0078 # better for ssd300_vgg16
# WEIGHT_DECAY = 4.554e-05
# MOMENTUM = 0.9848
# BATCH_SIZE = 2

# MODEL_NAME = 'retinanet_resnet50_fpn'
# LEARNING_RATE = 0.004404380413553278 # best from experimentz
# WEIGHT_DECAY = 1.6686655768554885e-05
# MOMENTUM = 0.9449407848243583
# BATCH_SIZE = 2

# MODEL_NAME = 'retinanet_resnet50_fpn_v2'
# LEARNING_RATE = 0.003073905696281962 # best from experiment
# WEIGHT_DECAY = 0.0023199534032866923
# MOMENTUM = 0.9780596592487132
# BATCH_SIZE = 1



CONFIG = {
    "lr": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE
    }

if __name__ == '__main__':


    torch.set_float32_matmul_precision("medium")

    logger = TensorBoardLogger('lightning_logs', name=MODEL_NAME, default_hp_metric=True, log_graph=False)
    early_stopping_callback = EarlyStopping(monitor='Validation/mAP', min_delta=0.001, patience=6, verbose=True, mode='max', check_on_train_epoch_end=False)
    SWA_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    # torch check for cuda devices
    # print(torch.cuda.is_available())
    # print(find_usable_cuda_devices(1))
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=find_usable_cuda_devices(1), 
        max_epochs=70, 
        logger=logger, 
        # limit_train_batches=0.50,
        # limit_val_batches=0.1,
        check_val_every_n_epoch=1,
        # val_check_interval=0.1, 
        log_every_n_steps=1, 
        auto_scale_batch_size='binsearch', 
        num_sanity_val_steps=0,
        accumulate_grad_batches=CONFIG["batch_size"] * 2,
        
        callbacks=[early_stopping_callback, SWA_callback]
        )


    data_module = StickerData(train_folder='data_stickers/train', valid_folder='data_stickers/valid', test_folder='data_stickers/test', batch_size=2, num_workers=NUM_WORKERS)

    model = StickerDetector(num_classes=NUM_CLASSES, config=CONFIG, model_name=MODEL_NAME)
    
    # trainer.tune(model, datamodule=data_module)

    torch.cuda.empty_cache()
    data_module.setup('fit')
    trainer.validate(model, datamodule=data_module)
    trainer.fit(model, data_module)

    # push_results_to_iphone(trainer=trainer, model=model, datamodule=data_module)



