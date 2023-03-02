import os
import json

from tqdm import tqdm
import cv2  as cv
import numpy as np


SHOW_IMAGES = False

# define window that is resizable
cv.namedWindow("image", cv.WINDOW_NORMAL)

def cut_stickers():
    # get a list of the images in the folder "stickers_og"
    images = os.listdir("sticker_black_original")
    # make a dict to store the name of the images along with their with and height
    stickers = {}
    # loop through the images
    for image in tqdm(images):
        # read the image
        img = cv.imread("sticker_black_original/" + image)
        #convert image to binary
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thresh = cv.inRange(gray, 20, 255)

        # check that no white pixel is without white neighbors
        kernel = np.ones((3,3),np.uint8)
        thresh = cv.erode(thresh,kernel,iterations = 2)
        thresh = cv.dilate(thresh,kernel,iterations = 2)
        # thresh = cv.bitwise_not(mask)        # show ret image
        if SHOW_IMAGES:
            cv.imshow("image", thresh)
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
        if SHOW_IMAGES:
            cv.imshow("image", crop)
            cv.waitKey(0)
        # save the image
        cv.imwrite("stickers_cut_black/" + image, crop)
        # image name and size to dict
        stickers[image] = {"width": w, "height": h}
        # save the dict to a json file
        with open("stickers.json", "w") as f:
            json.dump(stickers, f, indent=4)



if __name__ == "__main__":
    cut_stickers()
