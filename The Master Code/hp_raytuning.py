from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from functools import partial
from ray_lightning import RayStrategy


from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
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


# MODEL_NAME = 'fasterrcnn_resnet50_fpn'

# MODEL_NAME = 'fasterrcnn_resnet50_fpn_v2'

MODEL_NAME = 'ssd300_vgg16'

# MODEL_NAME = 'ssdlite320_mobilenet_v3_large'

# MODEL_NAME = 'retinanet_resnet50_fpn'

# MODEL_NAME = 'retinanet_resnet50_fpn_v2'


NUM_WORKERS = 2
NUM_EPOCHS = 120
NUM_SAMPLES = 50



def train_sticker_tune(config, num_epochs=10, num_gpus=0, tensor_board_name='ray_tune/' + MODEL_NAME):


    torch.set_float32_matmul_precision("medium")

    logger = TensorBoardLogger(LOGGER_PATH, name=tensor_board_name, default_hp_metric=True)

    early_stopping_callback = EarlyStopping(monitor='Validation/mAP', min_delta=0.01, patience=3, verbose=True, mode='max', check_on_train_epoch_end=False)
    tune_callback = TuneReportCallback({'Validation/mAP': 'Validation/mAP' }, on='validation_end')
    SWA_callback = StochasticWeightAveraging(swa_lrs=1e-2)


    trainer = pl.Trainer(
        gpus=num_gpus,
        max_epochs=num_epochs, 
        logger=logger, 
        check_val_every_n_epoch=1,
        progress_bar_refresh_rate=0,
        # val_check_interval=0.1, 
        limit_train_batches=0.50,
        # limit_val_batches=0.01,
        accumulate_grad_batches=config["batch_size"] * 2, #this will give the effective batch size

        log_every_n_steps=1, 
        num_sanity_val_steps=0,
        
        callbacks=[early_stopping_callback, tune_callback, SWA_callback]  
        )

    data_module = StickerData(train_folder=DATA_TRAIN_PATH, valid_folder=DATA_VALID_PATH, test_folder=DATA_TEST_PATH, batch_size=2, num_workers=NUM_WORKERS)

    model = StickerDetector(num_classes=3, config=config, model_name=MODEL_NAME)

    torch.cuda.empty_cache()
    data_module.setup('fit')
    trainer.validate(model, datamodule=data_module)
    
    trainer.fit(model, data_module)



def tune_sticker_asha(num_samples=NUM_SAMPLES, num_epochs=NUM_EPOCHS, gpus_per_trial=1):
    
    # list all the folders in lightning_logs and find all those who starts with ray_tune
    tune_folders = [f for f in os.listdir(LOGGER_PATH) if f.startswith('ray_tune')]
    # all the folders have a number after them eg. ray_tune_1, ray_tune_2, ray_tune_3
    # we want to find the highest number and add 1 to it
    if len(tune_folders) > 0:
        tune_folder_numbers = [int(f.split('_')[-1]) for f in tune_folders]
        tune_folder_numbers.sort()
        tune_folder_number = tune_folder_numbers[-1] + 1
    else:
        tune_folder_number = 1
    
    tensor_board_name = 'ray_tune_' + str(tune_folder_number) + '/' + MODEL_NAME


    scheduler = ASHAScheduler(
                                max_t=5000,
                                grace_period=5,
                                reduction_factor=2,
                                )
    
    # scheduler = PopulationBasedTraining(
    #                                     perturbation_interval=4,
    #                                     hyperparam_mutations={
    #                                         "lr": tune.loguniform(1e-4, 1e-2),
    #                                         "momentum": tune.uniform(0.7, 0.99),
    #                                         "weight_decay": tune.loguniform(1e-5, 1e-3),
    #                             })
    config = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "momentum": tune.uniform(0.7, 1.0),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([1, 2, 4, 8, 16]), # all these will be multiplied with 2, but is necessary for the for the dataloader in SSD300
        }
    

    # trainable = tune.with_parameters(
    #     train_sticker_tune,
    #     num_epochs=num_epochs,
    #     num_gpus=gpus_per_trial)
    
    reporter = CLIReporter(
        parameter_columns=["lr", "momentum", "weight_decay", "batch_size"],
        metric_columns=['Validation/mAP', "training_iteration"],
        print_intermediate_tables = True
        )

    analysis = tune.run(
                        partial(
                        train_sticker_tune,
                        num_epochs=num_epochs,
                        num_gpus=gpus_per_trial,
                        tensor_board_name=tensor_board_name
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
                        scheduler=scheduler,
                        name=f"tune_{MODEL_NAME}_asha")
    
    print("The final hyperparameters that performed best: ", analysis.best_config)



if __name__ == "__main__":
    tune_sticker_asha()