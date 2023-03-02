from pypylon import pylon
import cv2 as cv

import numpy as np
import os


SAVE_PATH_TOP = "grapped_computer_top"
SAVE_PATH_OPEN = "grapped_computer_open"
SAVE_PATH_BOTTOM = "grapped_computer_bottom"

SAVE_PATH_STICKER_WHITE = "sticker_white_original"
SAVE_PATH_STICKER_BLACK = "sticker_black_original"
# SAVE_PATH = "sticker_og"

def grab_images():

    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    for device in devices:
        print(device.GetFriendlyName())
    # Create an instant camera object with the camera device found first.
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.PixelFormat.SetValue("BGR8")


    print("Using device ", camera.GetDeviceInfo().GetModelName())
    # print camera resolution
    print("Camera resolution: ", camera.Width.GetValue(), "x", camera.Height.GetValue())
    cv.namedWindow("Grabbed", cv.WINDOW_NORMAL)
    cv.namedWindow("Camera", cv.WINDOW_NORMAL)
    grabbed = None

    # list all images in SAVE_PATH
    images = os.listdir(SAVE_PATH_TOP)
    # get the last image number
    if len(images) > 0:
        i_top = len(images) + 1
    else:
        i_top = 0

    images = os.listdir(SAVE_PATH_OPEN)
    # get the last image number
    if len(images) > 0:
        i_open = len(images) + 1
    else:
        i_open = 0
    
    images = os.listdir(SAVE_PATH_BOTTOM)
    # get the last image number
    if len(images) > 0:
        i_bottom = len(images) + 1
    else:
        i_bottom = 0

    images = os.listdir(SAVE_PATH_STICKER_WHITE)
    # get the last image number
    if len(images) > 0:
        i_white = len(images)
    else:
        i_white = 0
    
    images = os.listdir(SAVE_PATH_STICKER_BLACK)
    # get the last image number
    if len(images) > 0:
        i_black = len(images)
    else:
        i_black = 0


    
    while True:
        # display what the camera sees
        display = camera.GrabOne(1000)
        display = display.GetArray()
        cv.imshow("Camera", display)


        # wait for key press
        key = cv.waitKey(1)
        # if keypressed space
        if key == 32: # space
            # show grabbed image in Grabbed window
            grabbed = camera.GrabOne(1000)
            grabbed = grabbed.GetArray()
            print(grabbed.shape)
            cv.imshow("Grabbed", grabbed)
        # if keypressed 1
        elif key == 49: # 1
            # save grabbed image
            if grabbed is not None:
                cv.imwrite(SAVE_PATH_TOP + "/image_top" + str(i_top) + ".png", grabbed)
                print(f"Grabbed image number {i_top} saved")
                i_top += 1
                grabbed = None
        # if keypressed 2
        elif key == 50: # 2
            # save grabbed image
            if grabbed is not None:
                cv.imwrite(SAVE_PATH_OPEN + "/image_open" + str(i_open) + ".png", grabbed)
                print(f"Grabbed image number {i_open} saved")
                i_open += 1
                grabbed = None
        elif key == 51:
            if grabbed is not None:
                cv.imwrite(SAVE_PATH_BOTTOM + "/image_bottom" + str(i_bottom) + ".png", grabbed)
                print(f"Grabbed image number {i_bottom} saved")
                i_bottom += 1
                grabbed = None
        # if keypressed 5
        elif key == 53: # 5
            # save grabbed image
            if grabbed is not None:
                cv.imwrite(SAVE_PATH_STICKER_WHITE + "/sticker_white" + str(i_white) + ".png", grabbed)
                print(f"Grabbed image number {i_white} saved")
                i_white += 1
                grabbed = None
        # if keypressed 6
        elif key == 54: # 6
            # save grabbed image
            if grabbed is not None:
                cv.imwrite(SAVE_PATH_STICKER_BLACK + "/sticker_black" + str(i_black) + ".png", grabbed)
                print(f"Grabbed image number {i_black} saved")
                i_black += 1
                grabbed = None





if __name__ == "__main__":
    grab_images()