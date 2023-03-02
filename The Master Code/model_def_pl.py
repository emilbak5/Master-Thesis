import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import Accuracy

import pytorch_lightning as pl

from utils import make_images_for_tensorboard


def collate_fn(batch):
    return tuple(zip(*batch))

class StickerDetector(pl.LightningModule):
    def __init__(self, num_classes=3, config=None, batch_size=2):
        super(StickerDetector, self).__init__()

        learning_rate = config['lr']
        momentum = config['momentum']
        weight_decay = config['weight_decay']


        # load the pretrained model: Mask R-CNN
        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        # self.model = torch.hub.load('pytorch/vision:v0.6.0', 'maskrcnn_resnet50_fpn', pretrained=True)

        # get the number of input features for the classifier (is needed when changing the nr of classes)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.map_metric = MeanAveragePrecision(box_format='xywh')


        hparams = {
            'learning_rate': learning_rate,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'batch_size': batch_size
        }
        self.hparams.update(hparams)
        self.save_hyperparameters()




    def forward(self, images, targets=None):
        output = self.model(images, targets)
        return output


    def training_step(self, train_batch, batch_idx):
        images, targets = train_batch

        boxes_labels = []
        for target in targets:
            boxes_labels.append({'boxes': target['boxes'], 'labels': target['labels']})

        loss_dict = self.model(images, boxes_labels)


        loss_dict["loss"] = sum(loss for loss in loss_dict.values()) / len(loss_dict)



        return loss_dict

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_loss_classifier = torch.stack([x['loss_classifier'] for x in outputs]).mean()
        avg_loss_box_reg = torch.stack([x['loss_box_reg'] for x in outputs]).mean()
        avg_loss_objectness = torch.stack([x['loss_objectness'] for x in outputs]).mean()
        avg_loss_rpn_box_reg = torch.stack([x['loss_rpn_box_reg'] for x in outputs]).mean()


        self.log('Train/Loss', avg_loss, sync_dist=True)
        self.log('Train/Loss_classifier', avg_loss_classifier, sync_dist=True)
        self.log('Train/Loss_box_reg', avg_loss_box_reg, sync_dist=True)
        self.log('Train/Loss_objectness', avg_loss_objectness, sync_dist=True)
        self.log('Train/Loss_rpn_box_reg', avg_loss_rpn_box_reg, sync_dist=True)



    def validation_step(self, val_batch, batch_idx):
        images, targets = val_batch

        model_preds = self.model(images)

        self.map_metric.update(model_preds, targets)

        return {'model_preds': model_preds, 'targets': targets}


    def validation_epoch_end(self, outputs):

        map_results = self.map_metric.compute()
        self.log('Validation/mAP', map_results["map"], sync_dist=True)
        self.log('Validation/mAP_50', map_results["map_50"], sync_dist=True)
        self.log('Validation/mAP_75', map_results["map_75"], sync_dist=True)
        self.map_metric.reset()

        
        image = make_images_for_tensorboard(outputs[0]['model_preds'], outputs[0]['targets'])
        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_image('Validation/Example', image, self.current_epoch)

        


    
    def configure_optimizers(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=self.hparams["learning_rate"], momentum=self.hparams["momentum"], weight_decay=self.hparams["weight_decay"])
        # use adam optimizer
        # optimizer = torch.optim.Adam(parameters, lr=self.hparams["learning_rate"], weight_decay=self.hparams["weight_decay"])
        return optimizer




    




