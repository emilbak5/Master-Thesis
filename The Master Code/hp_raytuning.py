from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from functools import partial
from ray_lightning.tune import TuneReportCallback, get_tune_resources
from ray_lightning import RayStrategy


from pytorch_lightning.callbacks import EarlyStopping
from lightning.pytorch.accelerators import find_usable_cuda_devices
import torch

import pytorch_lightning as pl

from dataset_def_pl import StickerData
from model_def_pl import StickerDetector

import os


DATA_TRAIN_PATH = 'C:/Users/emilb/OneDrive/Skrivebord/Master-Thesis/The Master Code/data_stickers/train'
DATA_VALID_PATH = 'C:/Users/emilb/OneDrive/Skrivebord/Master-Thesis/The Master Code/data_stickers/valid'
DATA_TEST_PATH = 'C:/Users/emilb/OneDrive/Skrivebord/Master-Thesis/The Master Code/data_stickers/test'

LOGGER_PATH = 'C:/Users/emilb/OneDrive/Skrivebord/Master-Thesis/The Master Code/lightning_logs'

MODEL_NAME = 'fasterrcnn_resnet50_fpn_v2'


NUM_WORKERS = 1
BATCH_SIZE = 2
NUM_EPOCHS = 50
NUM_SAMPLES = 1



def train_sticker_tune(config, num_epochs=10, num_gpus=0):


    torch.set_float32_matmul_precision("medium")

    logger = TensorBoardLogger(LOGGER_PATH, name='sticker_detection_stickers', default_hp_metric=True)
    early_stopping_callback = EarlyStopping(monitor='Validation/mAP', min_delta=0.01, patience=3, verbose=True, mode='max', check_on_train_epoch_end=False)

    tune_callback = TuneReportCallback({'map': 'Validation/mAP' }, on='validation_end')

    trainer = pl.Trainer(
        gpus=num_gpus,
        max_epochs=num_epochs, 
        logger=logger, 
        check_val_every_n_epoch=1,
        progress_bar_refresh_rate=0,
        # val_check_interval=0.1, 
        log_every_n_steps=1, 
        num_sanity_val_steps=0,
        
        callbacks=[early_stopping_callback, tune_callback]
        )

    data_module = StickerData(train_folder=DATA_TRAIN_PATH, valid_folder=DATA_VALID_PATH, test_folder=DATA_TEST_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = StickerDetector(num_classes=3, config=config, batch_size=BATCH_SIZE, model_name=MODEL_NAME)
    
    trainer.fit(model, data_module)



def tune_sticker_asha(num_samples=NUM_SAMPLES, num_epochs=NUM_EPOCHS, gpus_per_trial=1):
    
    
    config = {
        "lr": tune.loguniform(1e-3, 1e-1),
        "momentum": tune.uniform(0.1, 0.99),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        }
    

    # trainable = tune.with_parameters(
    #     train_sticker_tune,
    #     num_epochs=num_epochs,
    #     num_gpus=gpus_per_trial)
    
    reporter = CLIReporter(
        parameter_columns=["lr", "momentum", "weight_decay"],
        metric_columns=["training_iteration"]
        )

    analysis = tune.run(
                        partial(
                        train_sticker_tune,
                        num_epochs=num_epochs,
                        num_gpus=gpus_per_trial
                        ),
                        resources_per_trial={
                            "cpu": 1,
                            "gpu": gpus_per_trial
                        },
                        metric="Validation/mAP",
                        mode="max",
                        config=config,
                        num_samples=num_samples,
                        progress_reporter=reporter,
                        name="tune_sticker")
    
    print(analysis.best_config)

    # scheduler = ASHAScheduler(
    #     max_t=num_epochs,
    #     grace_period=1,
    #     reduction_factor=2)
    
    # reporter = CLIReporter(
    #     parameter_columns=["lr", "momentum", "weight_decay"],
    #     metric_columns=["loss", "val_map", "training_iteration"])
    

    # train_fn_with_parameters = tune.with_parameters(train_sticker_tune,
    #                                                 num_epochs=num_epochs,
    #                                                 num_gpus=gpus_per_trial
    #                                                 )

    # tuner = tune.Tuner(
    #     tune.with_resources(train_sticker_tune, {"cpu": 1.0, "gpu": 1.0}),  
    #     tune_config=tune.TuneConfig(
    #             metric="Validation/mAP",
    #             mode="max",
    #             num_samples=2
    #         ),
    #         param_space=config,
    #         run_config=air.RunConfig(name="tune_mnist"),
    #     )

    # results = tuner.fit()
    # print("Best hyperparameters found were: ", results.best_config)

    # # analysis = tune.run(
    # #         train_sticker_tune,
    # #         metric="Validation/mAP",
    # #         mode="max",
    # #         config=config,
    # #         num_samples=2,
    # #         resources_per_trial=get_tune_resources(num_workers=2),
    # #         name="tune_mnist")
            
    # # print("Best hyperparameters found were: ", analysis.best_config)




    # train_fn_with_parameters = tune.with_parameters(train_sticker_tune,
    #                                                 num_epochs=num_epochs,
    #                                                 num_gpus=gpus_per_trial
    #                                                 )

    # resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

    # tuner = tune.Tuner(
    #     tune.with_resources(
    #         train_fn_with_parameters,
    #         resources=resources_per_trial
    #     ),
    #     tune_config=tune.TuneConfig(
    #         metric="loss",
    #         mode="min",
    #         scheduler=scheduler,
    #         num_samples=num_samples,
    #     ),
    #     run_config=air.RunConfig(
    #         name="tune_sticker_asha",
    #         progress_reporter=reporter,
    #     ),
    #     param_space=config,
    # )


    # results = tuner.fit()

    # print("Best hyperparameters found were: ", results.get_best_result().config)

if __name__ == "__main__":
    tune_sticker_asha()