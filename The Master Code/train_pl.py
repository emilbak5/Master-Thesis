import pytorch_lightning as pl

from dataset_def_pl import StickerData
from model_def_pl import StickerDetector

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from lightning.pytorch.accelerators import find_usable_cuda_devices

from utils import push_results_to_iphone
import torch




NUM_WORKERS = 4
BATCH_SIZE = 4

NUM_CLASSES = 3 # logo + sticker + background
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

CONFIG = {
    "lr": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY,
    }

if __name__ == '__main__':

    torch.set_float32_matmul_precision("medium")

    logger = TensorBoardLogger('lightning_logs', name='sticker_detection_v2', default_hp_metric=True)
    early_stopping_callback = EarlyStopping(monitor='Validation/mAP', min_delta=0.001, patience=4, verbose=True, mode='max', check_on_train_epoch_end=False)
    # torch check for cuda devices
    # print(torch.cuda.is_available())
    # print(find_usable_cuda_devices(1))
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=find_usable_cuda_devices(1), 
        max_epochs=70, 
        logger=logger, 
        check_val_every_n_epoch=1, 
        log_every_n_steps=1, 
        auto_scale_batch_size='binsearch', 
        num_sanity_val_steps=0,
        callbacks=[early_stopping_callback]
        )


    data_module = StickerData(train_folder='data_stickers/train', valid_folder='data_stickers/valid', test_folder='data_stickers/test', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = StickerDetector(num_classes=NUM_CLASSES, config=CONFIG, batch_size=BATCH_SIZE)
    
    # trainer.tune(model, datamodule=data_module)

    torch.cuda.empty_cache()
    trainer.validate(model, datamodule=data_module)
    trainer.fit(model, data_module)

    push_results_to_iphone(trainer=trainer, model=model, datamodule=data_module)



