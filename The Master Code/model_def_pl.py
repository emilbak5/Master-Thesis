import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead, det_utils
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
# from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchmetrics

import time

import pytorch_lightning as pl

from utils import make_images_for_tensorboard


def collate_fn(batch):
    return tuple(zip(*batch))

class StickerDetector(pl.LightningModule):
    def __init__(self, num_classes=3, config=None, model_name='fasterrcnn_resnet50_fpn'):
        super(StickerDetector, self).__init__()

        self.first_batch = True

        learning_rate = config['lr']
        momentum = config['momentum']
        weight_decay = config['weight_decay']
        batch_size = config['batch_size']

        # self.example_input_array = torch.Tensor(batch_size, 3, 2048, 2448)
        
        


        # load the pretrained model: Mask R-CNN
        self.model_name = model_name

        if self.model_name == 'fasterrcnn_resnet50_fpn':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT', trainable_backbone_layers=5)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        elif self.model_name == 'fasterrcnn_resnet50_fpn_v2':    
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', trainable_backbone_layers=5)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        elif self.model_name == 'ssd300_vgg16':
            self.model = torchvision.models.detection.ssd300_vgg16(weights='DEFAULT', trainable_backbone_layers=5)
            out_channgels = det_utils.retrieve_out_channels(self.model.backbone, (300, 300))
            anchor_generator = self.model.anchor_generator
            num_anchors = anchor_generator.num_anchors_per_location()
            self.model.head.classification_head = SSDClassificationHead(out_channgels, num_anchors, num_classes)
        
        elif self.model_name == 'ssdlite320_mobilenet_v3_large':
            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights='DEFAULT', trainable_backbone_layers=6)
            out_channgels = det_utils.retrieve_out_channels(self.model.backbone, (320, 320))
            anchor_generator = self.model.anchor_generator
            num_anchors = anchor_generator.num_anchors_per_location()
            self.model.head.classification_head = SSDClassificationHead(out_channgels, num_anchors, num_classes)

        elif self.model_name == 'retinanet_resnet50_fpn':
            kwargs = {'score_thresh': 0.0005}
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(weights='DEFAULT', trainable_backbone_layers=5)
            num_anchors = self.model.head.classification_head.num_anchors
            out_channgels = self.model.backbone.out_channels
            self.model.head.classification_head = RetinaNetClassificationHead(in_channels=out_channgels, num_anchors=num_anchors, num_classes=num_classes)
            self.model.score_thresh = 0.0005

        elif self.model_name == 'retinanet_resnet50_fpn_v2':
            self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights='DEFAULT', trainable_backbone_layers=5)
            num_anchors = self.model.head.classification_head.num_anchors
            out_channgels = self.model.backbone.out_channels
            self.model.head.classification_head = RetinaNetClassificationHead(in_channels=out_channgels, num_anchors=num_anchors, num_classes=num_classes)
            self.model.score_thresh = 0.0005


            # in_features = self.model.retinanet_head.cls_score.in_features
            # self.model.retinanet_head = FastRCNNPredictor(in_features, num_classes)
        
        
        
        # self.model.eval()
        # script_model = torch.jit.script(self.model)
        # prototype_array = torch.rand(32, 3, 28, 27).cuda()

        # self.logger.experiment.add_graph(script_model, prototype_array)



        # self.map_metric = MeanAveragePrecision(box_format='xywh')
        self.map_metric = torchmetrics.MAP()
        self.accuracy_metric = torchmetrics.Accuracy(num_classes=num_classes - 1)




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
        x = 5

        return output


    def training_step(self, train_batch, batch_idx):
        

            
        images, targets = train_batch

        boxes_labels = []
        for target in targets:
            boxes_labels.append({'boxes': target['boxes'], 'labels': target['labels']})


        loss_dict = self.model(images, boxes_labels)

        # if self.model_name == 'fasterrcnn_resnet50_fpn' or self.model_name == 'fasterrcnn_resnet50_fpn_v2':
        #     loss_dict["loss"] = self.calc_weighted_average_loss_for_rcnn(loss_dict)
        # else:    
        loss_dict["loss"] = sum(loss for loss in loss_dict.values()) / len(loss_dict)

        if self.first_batch:
            # prototype_array = torch.rand(32, 3, 2048, 2448).cuda()
            # script_model = torch.jit.script(self.model, prototype_array)
            # self.logger.experiment.add_graph(script_model, prototype_array)
            # log the loss
            if self.model_name == 'fasterrcnn_resnet50_fpn' or self.model_name == 'fasterrcnn_resnet50_fpn_v2':
                self.logger.experiment.add_scalar('Train/Loss', loss_dict["loss"], 0)
                self.logger.experiment.add_scalar('Train/Loss_classifier', loss_dict["loss_classifier"], 0)
                self.logger.experiment.add_scalar('Train/Loss_box_reg', loss_dict["loss_box_reg"], 0)
                self.logger.experiment.add_scalar('Train/Loss_objectness', loss_dict["loss_objectness"], 0)
                self.logger.experiment.add_scalar('Train/Loss_rpn_box_reg', loss_dict["loss_rpn_box_reg"], 0)
            else: # same for ssd and retinanet
                self.logger.experiment.add_scalar('Train/Loss', loss_dict["loss"], 0)
                self.logger.experiment.add_scalar('Train/Loss_classifier', loss_dict["classification"], 0)
                self.logger.experiment.add_scalar('Train/Loss_box_reg', loss_dict["bbox_regression"], 0)



            self.first_batch = False

        return loss_dict

    def training_epoch_end(self, outputs):
        
        if self.model_name == 'fasterrcnn_resnet50_fpn':
            self.log_train_fasterrcnn(outputs=outputs)
        elif self.model_name == 'fasterrcnn_resnet50_fpn_v2':
            self.log_train_fasterrcnn(outputs=outputs)
        elif self.model_name == 'ssd300_vgg16':
            self.log_train_loss_SSD(outputs=outputs)
        elif self.model_name == 'ssdlite320_mobilenet_v3_large':
            self.log_train_loss_SSD(outputs=outputs)
        elif self.model_name == 'retinanet_resnet50_fpn':
            self.log_train_loss_retinanet(outputs=outputs)
        elif self.model_name == 'retinanet_resnet50_fpn_v2':
            self.log_train_loss_retinanet(outputs=outputs)



    def validation_step(self, val_batch, batch_idx):
        images, targets = val_batch

        model_preds = self.model(images)

        self.map_metric.update(model_preds, targets)


        return {'model_preds': model_preds, 'targets': targets}


    def validation_epoch_end(self, outputs):

        # start timer
        map_results = self.map_metric.compute()
        # end timer
        # print("MAP calculation time: ", end_time - start_time)

        self.log('Validation/mAP', map_results["map"], sync_dist=True)
        self.log('Validation/mAP_50', map_results["map_50"], sync_dist=True)
        self.log('Validation/mAP_75', map_results["map_75"], sync_dist=True)
        self.map_metric.reset()

        
        image = make_images_for_tensorboard(outputs[0]['model_preds'], outputs[0]['targets'])
        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_image('Validation/Example', image, self.global_step)

        


    
    def configure_optimizers(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=self.hparams["learning_rate"], momentum=self.hparams["momentum"], weight_decay=self.hparams["weight_decay"])

        # if self.model_name == "fasterrcnn_resnet50_fpn" or self.model_name == "fasterrcnn_resnet50_fpn_v2" or \
        #                                                    self.model_name == "retinanet_resnet50_fpn" or \
        #                                                    self.model_name == "retinanet_resnet50_fpn_v2":
            
        #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)
        #     return [optimizer], [lr_scheduler]
        
        # elif self.model_name == self.model_name == "ssdlite320_mobilenet_v3_large":
        #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=120)
        #     return [optimizer], [lr_scheduler]
        
        # elif self.model_name == "ssd300_vgg16":
        #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[80, 110], gamma=0.1)
        #     return [optimizer], [lr_scheduler]

        return optimizer
    

    def log_train_loss_SSD(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_loss_classifier = torch.stack([x['classification'] for x in outputs]).mean()
        avg_loss_box_reg = torch.stack([x['bbox_regression'] for x in outputs]).mean()

        self.log('Train/Loss', avg_loss, sync_dist=True)
        self.log('Train/Loss_classifier', avg_loss_classifier, sync_dist=True)
        self.log('Train/Loss_box_reg', avg_loss_box_reg, sync_dist=True)

    def log_train_fasterrcnn(self, outputs):

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
    
    def log_train_loss_retinanet(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_loss_classifier = torch.stack([x['classification'] for x in outputs]).mean()
        avg_loss_box_reg = torch.stack([x['bbox_regression'] for x in outputs]).mean()

        self.log('Train/Loss', avg_loss, sync_dist=True)
        self.log('Train/Loss_classifier', avg_loss_classifier, sync_dist=True)
        self.log('Train/Loss_box_reg', avg_loss_box_reg, sync_dist=True)


   
    def calc_weighted_average_loss_for_rcnn(self, loss_dict):

        classification_proporition = 0.4
        box_reg_proporition = 0.2
        objectness_proporition = 0.2
        rpn_box_reg_proporition = 0.2

        # loss = loss_dict["loss_classifier"] + loss_dict["loss_box_reg"] + loss_dict["loss_objectness"] + loss_dict["loss_rpn_box_reg"]
        # loss = loss / 4

        # Normalize the new losses with the proportions
        loss =  (loss_dict["loss_classifier"] * classification_proporition) + \
                (loss_dict["loss_box_reg"] * box_reg_proporition) + \
                (loss_dict["loss_objectness"] * objectness_proporition) + \
                (loss_dict["loss_rpn_box_reg"] * rpn_box_reg_proporition)



        return loss






    




