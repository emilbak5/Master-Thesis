import cv2 as cv
import numpy as np
import random
import json
from scipy import ndimage
from shapely.geometry import Polygon, MultiPolygon
import os
import re
import copy


LAPTOP_PATH_TOP = 'annotated_top_open'
COMBINED_PATH = 'combined'
COMBINED_MASKS_PATH = 'combined/masks'
STICKER_PATH_WHITE = 'stickers_cut_white'
STICKER_PATH_BLACK = 'stickers_cut_black'

WHITE = 0
BLACK = 1





def trackbar_callback(val):
    pass

def get_random_sticker(stickers_white, stickers_black):
    color = random.randint(WHITE, BLACK)
    color = BLACK
    sticker = None
    while True:
        if color == WHITE:
            sticker = random.choice(stickers_white)
            sticker = cv.imread(os.path.join(STICKER_PATH_WHITE, sticker))
            cv.setTrackbarPos("sticker_trackbar", "Sticker", 252)

        else:
            # sticker = random.choice(stickers_black)
            # sticker = cv.imread(os.path.join(STICKER_PATH_BLACK, sticker))
            # cv.setTrackbarPos("sticker_trackbar", "Sticker", 20)'
            sticker = random.choice(stickers_black)
            sticker = cv.imread(os.path.join(STICKER_PATH_BLACK, sticker))
            sticker[np.all(sticker == (76,112,71), axis=-1)] = (0,0,0)
            cv.setTrackbarPos("sticker_trackbar", "Sticker", 1)
        
        if sticker.shape[0] < 1000 and sticker.shape[1] < 1000:
            break
    
    return sticker, color
    
def get_mask(sticker, color):

    thresh = cv.getTrackbarPos("sticker_trackbar", "Sticker")
    # convert image to binary
    gray = cv.cvtColor(sticker, cv.COLOR_BGR2GRAY)
    if color == WHITE:
        ret, thresh = cv.threshold(gray, thresh=thresh, maxval=255, type=cv.THRESH_BINARY_INV)
    else:
        thresh = cv.inRange(gray, thresh, 255)
        # thresh = cv.bitwise_not(thresh)

            # check that no white pixel is without white neighbors
    kernel = np.ones((3,3),np.uint8)
    thresh = cv.erode(thresh,kernel,iterations = 2)
    thresh = cv.dilate(thresh,kernel,iterations = 2)

    return thresh

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        img, mask, sticker, annotations, annotation_nr, i = param


        # gray = cv.cvtColor(sticker, cv.COLOR_BGR2GRAY)
        # find all pixels that are not black
        # mask = cv.inRange(gray, 0, 230)
        # mask = cv.bitwise_not(mask)
        
        # show ret
        # cv.imshow("test", gray)
        # cv.waitKey(0)
        # cv.imshow("ret", mask)
        # cv.waitKey(0)

        # get bounding box of mask
        assert mask.shape == sticker.shape[:2]

        x_bb, y_bb, w, h = cv.boundingRect(mask)
        x1 = x
        y1 = y
        x2, y2 = x + w, y + h
        mask = mask[y_bb:y_bb + h, x_bb:x_bb + w]
        # resize sticker the same way as mask. remember there is a 3rd dimension
        sticker = sticker[y_bb:y_bb + h, x_bb:x_bb + w]

        cv.imshow("Sticker", sticker)
        # cv.imshow("mask", mask)

        polygons = []
        contour, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour = sorted(contour, key=cv.contourArea, reverse=True)

        # mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        # cv.drawContours(mask, contour, -1, (0, 0, 255), 3)

        # offset the contours to the correct position
        for point in contour[0]:
            point[0][0] = point[0][0] + x1
            point[0][1] = point[0][1] + y1
        

        
        # cv.drawContours(img, contour[0], -1, (0, 0, 255), 3)

        # convert contour to a flattened list of points in this format [x1, y1, x2, y2, ...]
        contour = contour[0].flatten().tolist()


        

        # # get the width and height of the sticker
        # h, w, _ = sticker.shape
        # # get the x and y of the top left corner of the sticker
        # x1, y1 = x, y
        # # get the x and y of the bottom right corner of the sticker
        # x2, y2 = x + w, y + h


        # at all the pixels in ret that are black (0) set the pixels in the laptop to the pixel in the sticker
        annotation_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for j in range(mask.shape[0]):
            for k in range(mask.shape[1]):
                if mask[j, k] == 255:
                    img[y1 + j, x1 + k] = sticker[j, k]
                    annotation_mask[y1 + j, x1 + k] = 255
        # draw a rectangle around the sticker
        # cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        


        # add to dict. must have id, image_id, category_id, bbox, area, segmentation, iscrowd, category_id is 2 for stickers       
        annotations['annotations'].append({'id': annotation_nr[0], 'image_id': i, 'category_id': 2, 'bbox': [x1, y1, w, h], 'mask': f"{annotation_nr[0]}.png", 'area': w * h, 'segmentation': [contour], 'iscrowd': 0}) # contour must be added
        cv.imwrite(os.path.join(COMBINED_MASKS_PATH, f"{annotation_nr[0]}.png"), annotation_mask)
        annotation_nr[0] += 1
        cv.imshow("laptop", img)




# make 3 windows
cv.namedWindow("laptop", cv.WINDOW_NORMAL)
cv.namedWindow("Sticker", cv.WINDOW_NORMAL)
cv.resizeWindow("Sticker", 600, 300)
cv.namedWindow("Extracted Sticker", cv.WINDOW_NORMAL)

cv.createTrackbar("sticker_trackbar", "Sticker", 0, 254, trackbar_callback)
cv.setTrackbarMin('sticker_trackbar', "Sticker", 1)



def add_to_pc():
    # check if folder exists
    if not os.path.exists(COMBINED_PATH):
        os.makedirs(COMBINED_PATH)
    # check if folder exists
    if not os.path.exists(COMBINED_MASKS_PATH):
        os.makedirs(COMBINED_MASKS_PATH)

    # list the laptops
    laptops = os.listdir(LAPTOP_PATH_TOP)
    stickers_white = os.listdir(STICKER_PATH_WHITE)
    stickers_black = os.listdir(STICKER_PATH_BLACK)
    previously_annotated = os.listdir(COMBINED_PATH)
    previously_annotated = [string for string in previously_annotated if string.endswith('.jpg')]
    laptops = [string for string in laptops if string.endswith('.jpg')]


    


    annotations = {'categories': [{"id": 1, 'name': "Sticker", 'supercategory': 'none'}, {"id": 2, 'name': 'logo', 'supercategory': 'none'}],
        'images': [],
        'annotations': []}


    annotation_nr = [0]

    if os.path.exists('annotations.json'):
        with open('annotations.json', 'r') as f:
            annotations = json.load(f)
    
        #   find highest annotation number in json file
        annotation_ids = [annotation['id'] for annotation in annotations['annotations']]
        annotation_nr[0] = max(annotation_ids) + 1
        # for annotation in annotations['annotations']:
        #     if annotation['id'] > annotation_nr[0]:
        #         annotation_nr[0] = annotation['id']
        # annotation_nr[0] += 1

    if not previously_annotated:
        i = 0
    else:
        # image_numbers = [int(re.search(r'\d+', string).group()) for string in previously_annotated]
        image_ids = [image["id"] for image in annotations["images"]]
        i = max(image_ids) + 1

    while(True):
        
        # randomize the laptops list
        random.shuffle(laptops)
        
        for laptop in laptops:
            image_name = 'image_top' + str(i) + '.jpg'
            img = cv.imread(LAPTOP_PATH_TOP + '/' + laptop)
            coppied_annotations = []
            img_id = None
            for image_anno in annotations['images']:
                # find the image id of the laptop
                if image_anno['file_name'] == laptop: 
                    img_id = image_anno['id']
            
            for annotation in annotations['annotations']:
                if annotation['image_id'] == img_id:
                    coppied_annotations.append(copy.deepcopy(annotation))
            

            assert coppied_annotations != None, "No annotations found for image " + laptop

            cv.imshow("laptop", img)

            sticker, color = get_random_sticker(stickers_white, stickers_black)
            # get a random number for rotation between 0 and 360
            rotation = random.randint(0, 360)
            # rotate the sticker
            if color == WHITE:
                sticker = ndimage.rotate(sticker, rotation, cval=255)
            else:
                # sticker = ndimage.rotate(sticker, rotation, cval=0)
                sticker = ndimage.rotate(sticker, rotation, cval=0)

            changed = True
            prev_track_val = cv.getTrackbarPos("sticker_trackbar", "Sticker")
            while(True):
                # make a brown background
                


                # cv.resizeWindow("Sticker", sticker.shape[1], sticker.shape[0])
                cv.resizeWindow("Extracted Sticker", sticker.shape[1], sticker.shape[0])
                if changed or prev_track_val != cv.getTrackbarPos("sticker_trackbar", "Sticker"):
                    brown_background = np.zeros((sticker.shape[0], sticker.shape[1] , 3), np.uint8)
                    brown_background[:] = (63, 0, 127) # brown
                    mask = get_mask(sticker, color)
                    for j in range(mask.shape[0]):
                        for k in range(mask.shape[1]):
                            if mask[j, k] == 255:
                                brown_background[j, k] = sticker[j, k]
                    changed = False
                    prev_track_val = cv.getTrackbarPos("sticker_trackbar", "Sticker")



                cv.imshow("Extracted Sticker", brown_background)
                cv.imshow("Sticker", sticker)


                key = cv.waitKeyEx(10)
                if key == 2490368: # up arrow
                    sticker = cv.resize(sticker, (int(sticker.shape[1] * 1.1), int(sticker.shape[0] * 1.1)))
                    changed = True
                elif key == 2621440: # down arrow
                    sticker = cv.resize(sticker, (int(sticker.shape[1] * 0.9), int(sticker.shape[0] * 0.9)))
                    changed = True
                ## detect where the mouse is and place the sticker there
                elif key == 2555904: # right arrow
                    contour, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    # find the largest contour
                    contour = sorted(contour, key=cv.contourArea, reverse=True)

                    x_bb, y_bb, w, h = cv.boundingRect(mask)

                    mask_temp = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
                    cv.rectangle(mask_temp, (x_bb, y_bb), (x_bb + w, y_bb + h), (0, 255, 0), 2)
                    cv.drawContours(mask_temp, contour[0], -1, (0, 0, 255), 4)
                    cv.imshow("Extracted Sticker", mask_temp)

                    while True:
                        cv.waitKey(10)
                        cv.setMouseCallback("laptop", mouse_callback, param=(img, mask, sticker, annotations, annotation_nr, i))
                        if cv.waitKeyEx(0) == 2555904:

                            sticker, color = get_random_sticker(stickers_white, stickers_black)
                            # get a random number for rotation between 0 and 360
                            rotation = random.randint(0, 360)
                            # rotate the sticker
                            if color == WHITE:
                                sticker = ndimage.rotate(sticker, rotation, cval=255)
                            else:
                                # sticker = ndimage.rotate(sticker, rotation, cval=0)
                                sticker = ndimage.rotate(sticker, rotation, cval=0)
                            changed = True
                            break
                    # sticker = random.choice(stickers)
                    # sticker = cv.imread(STICKER_PATH + '/' + sticker)
                    # annotation_nr += 1

                # if space bar is pressed, dont save the image and break 
                elif key == 32:
                    break
                


                # if enter is press break
                elif key == 13:
                    # save the annotated image
                    cv.imwrite('combined/' + image_name, img)
                    cv.imwrite('C:/Users/emilb/OneDrive/Skrivebord/label-studio-data/images/image' + str(i) + '.jpg', img)
                    annotations['images'].append({'id': i, 'license': 0, 'file_name': image_name, 'height': img.shape[0], 'width': img.shape[1], 'date_captured': '2021-01-01 00:00:00'})
                    for annotation in coppied_annotations:
                        annotation['id'] = annotation_nr[0]
                        annotation['image_id'] = i

                        annotations['annotations'].append(annotation)
                        annotation_nr[0] += 1
                    # save the annotations
                    i += 1
                    with open('annotations.json', 'w') as f:
                        json.dump(annotations, f, indent=4)


                    break
            
        







if __name__ == "__main__":
    add_to_pc()