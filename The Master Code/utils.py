import torch
import pytorch_lightning as pl

from torchvision import transforms
from torchvision.utils import draw_bounding_boxes, make_grid
import torchmetrics

from dataset_def_pl import StickerData

from tqdm import tqdm
import random
import http.client, urllib

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

LABELS = {'Background': 0, 'Logo': 1, 'Sticker': 2}
# MIN_SCORE = 0.5



def push_results_to_iphone(trainer, model, datamodule):

    results = trainer.validate(model, datamodule=datamodule)

    map = results[0]['Validation/mAP']
    map_50 = results[0]['Validation/mAP_50']
    map_75 = results[0]['Validation/mAP_75']
    message = f"mAP: {map} \nmAP_50: {map_50} \nmAP_75: {map_75}"


    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
    urllib.parse.urlencode({
        "token": "anda9mq8fyqx3gueybj8hgfpw4eety",
        "user": "uixvg4qtfyh5ccyesbdegjgu1ii6ac",
        "title": "Training Finished",
        "message": message,
    }), { "Content-type": "application/x-www-form-urlencoded" })
    conn.getresponse()



def show_10_images_with_bounding_boxes(model, dataset: StickerData, check_point_path, num_images=2):
    map_metric = torchmetrics.MAP()

    save_path = check_point_path[:check_point_path.rfind('\\')]
    model = model.load_from_checkpoint(check_point_path)
    # dataset.setup(stage='test')
    dataset.setup(stage='fit')

    model.cuda()
    model.eval()

    # dataloader = dataset.test_dataloader()
    dataloader = dataset.train_dataloader()

    best_min_score = 0
    prev_map = 0

    # make for loop that goes from 0.1 to 1.0 in steps of 0.1
    for MIN_SCORE in np.arange(1.0, 0.0, -0.05):
    # for MIN_SCORE in [0.6]:
        lowest_scores = []

        # loop through the dataloader using tqdm while enumerating the index
        images_for_grid = []
        with torch.no_grad():
            for i, (image, target) in tqdm(enumerate(dataloader), total=num_images):
                torch.cuda.empty_cache()

                # move the image and target to cuda. first convert to list because the target is a list of tensors
                image = list(image)
                image[0] = image[0].cuda()
                image = tuple(image)
                # target[0]['boxes'] = target[0]['boxes'].cuda()
                # target[0]['labels'] = target[0]['labels'].cuda()
                # target[0]['image_id'] = target[0]['image_id'].cuda()
                preds = model(image)


                # only keep preds if the score is > MIN_SCORE. do it with a for loop
                for j in range(len(preds[0]['scores'])-1, -1, -1):
                    if preds[0]['scores'][j] < MIN_SCORE or preds[0]['labels'][j] != 2:
                        preds[0]['boxes'] = torch.cat((preds[0]['boxes'][:j], preds[0]['boxes'][j+1:]))
                        preds[0]['labels'] = torch.cat((preds[0]['labels'][:j], preds[0]['labels'][j+1:]))
                        preds[0]['scores'] = torch.cat((preds[0]['scores'][:j], preds[0]['scores'][j+1:]))
                
                # do the same for target
                for j in range(len(target[0]['labels'])-1, -1, -1):
                    if target[0]['labels'][j] != 2:
                        target[0]['boxes'] = torch.cat((target[0]['boxes'][:j], target[0]['boxes'][j+1:]))
                        target[0]['labels'] = torch.cat((target[0]['labels'][:j], target[0]['labels'][j+1:]))
                        target[0]['image_id'] = torch.cat((target[0]['image_id'][:j], target[0]['image_id'][j+1:]))


                # move the preds and target to cpu
                for pred in preds:
                    for key, value in pred.items():
                        pred[key] = value.cpu()



                map_metric.update(preds, target)

                # invert the normalization
                image_name =  target[0]['image_name']
                # image = cv.imread('data_stickers/test/' + image_name)

                # if image_name == 'image_open48.jpg':
                #     print('here')

                # conver bounding boxes to numpy
                boxes = preds[0]['boxes'].cpu().detach().numpy()
                labels = preds[0]['labels'].cpu().detach().numpy()
                scores = preds[0]['scores'].cpu().detach().numpy()
                # convert target boxes to numpy
                target_boxes = target[0]['boxes'].cpu().detach().numpy()
                target_labels = target[0]['labels'].cpu().detach().numpy()


                # make two new lists for target boxes and scores that only keeps the items if the label is a sticker
                boxes_temp = [boxes[i] for i in range(len(boxes)) if labels[i] == 2]
                scores_temp = [scores[i] for i in range(len(scores)) if labels[i] == 2]

                target_boxes_temp = [target_boxes[i] for i in range(len(target_boxes)) if target_labels[i] == 2]



                # if len(scores_temp) > len(target_boxes_temp):
                #     highest_scores = scores_temp[:len(target_boxes_temp)]
                #     min_score = np.min(highest_scores)
                #     lowest_scores.append(min_score)
                # else:
                #     lowest_scores.append(np.min(scores_temp))

                # loop BACKWARDS through the scores. if a score is < MIN_SCORE then remove the bounding box and label
                # for j in range(len(scores)-1, -1, -1):
                #     if scores[j] < MIN_SCORE:
                #         boxes = np.delete(boxes, j, 0)
                #         labels = np.delete(labels, j, 0)
                #         scores = np.delete(scores, j, 0)



                labels = [key for value in labels for key, val in LABELS.items() if val == value]
                target_labels = [key for value in target_labels for key, val in LABELS.items() if val == value]

                # draw bounding boxes on image. target should be green and prediction should be red
                # for box, label in zip(target_boxes, target_labels):
                #     if label == 'Sticker' or label == 'Logo':
                #         image = cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                #         image = cv.putText(image, str(label), (int(box[0]), int(box[1])), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv.LINE_AA)
                # for box, label, score in zip(boxes, labels, scores):
                #     if label == 'Sticker' or label == 'Logo':
                #         image = cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                #         image = cv.putText(image, str(label + ' ' + str(score)[:5]), (int(box[0]), int(box[1])), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv.LINE_AA) # (255,0,0) is blue


                # images_for_grid.append(image)

                # cv.imwrite(save_path + '/' + image_name, image)
        

                if i == num_images - 1:
                    break
        

        # print('lowest scores: ', lowest_scores)
        map_results = map_metric.compute()
        # print(f'mAP: {map_results["map"]}, mAP_50: {map_results["map_50"]}, mAP_75: {map_results["map_75"]}')
        if map_results["map"] > prev_map:
            prev_map = map_results["map"]
            best_min_score = MIN_SCORE
            print(f"new best mAP: {map_results['map']}, mAP_50: {map_results['map_50']}, mAP_75: {map_results['map_75']} with min_score: {best_min_score}")
        map_metric.reset()
    # row1 = [images_for_grid[i] for i in range(0, len(images_for_grid), 2)]
    # row2 = [images_for_grid[i] for i in range(1, len(images_for_grid), 2)]
    
    # # create a grid of images
    # row1 = np.hstack(row1)
    # row2 = np.hstack(row2)
    # images = np.vstack((row1, row2))


    # # show the grid of images rezing the window to fit the image
    # save_path = check_point_path[:check_point_path.rfind('\\')]
    # cv.imwrite(save_path + '/test_images.jpg', images)

    # plt_image = cv.cvtColor(images, cv.COLOR_BGR2RGB)
    # imgplot = plt.imshow(plt_image)
    # plt.show()
    # cv.namedWindow('image', cv.WINDOW_NORMAL)
    # cv.imshow('image', images)

    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    return None


def make_images_for_tensorboard(pred, target):
    MIN_SCORE = 0.1
    # get a random number between 0 and the length of the target
    random_image_nr = random.randint(0, len(target) - 1)

    image = cv.imread('C:/Users/emilb/OneDrive/Skrivebord/Master-Thesis/The Master Code/data_stickers/valid/' + target[random_image_nr]['image_name'])
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    labels_target = target[random_image_nr]['labels'].tolist()
    labels_pred = pred[random_image_nr]['labels'].tolist()

    boxes_target = target[random_image_nr]['boxes']
    boxes_pred = pred[random_image_nr]['boxes'].tolist()

    scores = pred[random_image_nr]['scores'].tolist()


    # loop BACKWARDS through the scores. if a score is < 0.5 then remove the bounding box and label
    for j in range(len(scores)-1, -1, -1):
        if scores[j] < MIN_SCORE:
            boxes_pred = np.delete(boxes_pred, j, 0)
            labels_pred = np.delete(labels_pred, j, 0)

    # Convert labels to string
    labels_target = [key for value in labels_target for key, val in LABELS.items() if val == value]
    labels_pred = [key for value in labels_pred for key, val in LABELS.items() if val == value]

    if type(boxes_pred) == list:
        boxes_pred = np.array(boxes_pred)

    # convert to tensor from numpy

    boxes_pred = torch.from_numpy(boxes_pred)
    

    bb_image = draw_bounding_boxes(image=image, boxes=boxes_target, labels=labels_target, colors=(0, 255, 0), width=3)

    # make sure the tensor boxes_pred is not empty
    if boxes_pred.shape[0] != 0:
        bb_image = draw_bounding_boxes(image=bb_image, boxes=boxes_pred, labels=labels_pred, colors=(0, 0, 255), width=3)
    
    
    

    #resize image to 80 % of original size
    bb_image = transforms.Resize(size=(int(bb_image.shape[1] * 0.8), int(bb_image.shape[2] * 0.8)))(bb_image)
    # # get two random images from image in a list
    # random_ints = random.sample(range(0, len(target)), 4)
    # bbox_images = []
    # for random_int in random_ints:

    #     image = cv.imread('data_stickers/valid/' + target[random_int]['image_name'])
    #     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    #     image = torch.from_numpy(image)
    #     image = image.permute(2, 0, 1)
        
    #     bb_image = draw_bounding_boxes(image=image, boxes=target[random_int]['boxes'], colors=(0, 255, 0))
    #     bb_image = draw_bounding_boxes(image=bb_image, boxes=pred[random_int]['boxes'], colors=(0, 0, 255))

    #     bbox_images.append(bb_image)



    # # convert bbox_images to tensor (b, c, h, w) 
    # bbox_images = torch.stack(bbox_images)
    # # make a grid of images
    # grid = make_grid(bbox_images, nrow=2)
    # # shrink grid to 1/2 size
    # # grid = transforms.Resize(size=(grid.shape[1]//2, grid.shape[2]//2))(grid)

    return bb_image




