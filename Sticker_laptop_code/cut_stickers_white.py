import os
import json

from tqdm import tqdm
import cv2  as cv
import numpy as np


SHOW_IMAGES = False

# make window that is 400x400
cv.namedWindow("ret", cv.WINDOW_NORMAL)
cv.resizeWindow("ret", 400, 400)

def cut_stickers():
    # get a list of the images in the folder "stickers_og"
    images = os.listdir("sticker_white_original")
    # make a dict to store the name of the images along with their with and height
    stickers = {}
    # loop through the images
    for image in tqdm(images):
        # read the image
        img = cv.imread("sticker_white_original/" + image)
        #convert image to binary
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, thresh=245, maxval=255, type=cv.THRESH_BINARY_INV)
        # thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 2)
        # blur = cv.GaussianBlur(gray,(5,5),0)
        # ret3,thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        # check that no white pixel is without white neighbors
        kernel = np.ones((3,3),np.uint8)
        thresh = cv.erode(thresh,kernel,iterations = 2)
        thresh = cv.dilate(thresh,kernel,iterations = 2)

        # show ret image
        if SHOW_IMAGES:
            cv.imshow("ret", thresh)
            cv.waitKey(0)   
        # make a boundingbox the colored pixels
        x, y, w, h = cv.boundingRect(thresh)
        # get the contours
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        # draw the boundingbox
        # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # crop the image
        crop = img[y:y + h, x:x + w]
        # show the image
        # if SHOW_IMAGES:
        #     cv.imshow("image", crop)
        #     cv.waitKey(0)
        # save the image
        if crop.size == 0:
            print("image number: " + image + " is empty")
            continue

        cv.imwrite("stickers_cut_white/" + image, crop)
        # image name and size to dict
        stickers[image] = {"width": w, "height": h}
        # save the dict to a json file
        with open("stickers.json", "w") as f:
            json.dump(stickers, f, indent=4)
        



if __name__ == "__main__":
    cut_stickers()
