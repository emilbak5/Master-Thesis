import numpy as np
import os
import cv2

from tqdm import tqdm


cv2.namedWindow('mean_image_valid', cv2.WINDOW_NORMAL)
cv2.namedWindow('mean_image_train', cv2.WINDOW_NORMAL)
cv2.namedWindow('mean_image_test', cv2.WINDOW_NORMAL)

def get_mean_image(images_path):

    images = os.listdir(images_path)
    images = [image for image in images if image.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    # remove all images that starts with a
    # images = [image for image in images if not image.startswith('a')]

    mean_image = np.zeros((2048, 2448, 3), dtype=np.float32)
    for image_name in tqdm(images):
        image = cv2.imread(os.path.join(images_path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean_image += image

    mean_image /= len(images)
    return mean_image

def mutual_information(image1, image2):
    # Calculate histograms of the two images
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    
    # Normalize the histograms
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Calculate joint histogram
    image1 = image1.astype(np.uint8)
    image2 = image2.astype(np.uint8)
    joint_hist = np.zeros((256, 256))
    for i in tqdm(range(image1.shape[0])):
        for j in range(image1.shape[1]):
            joint_hist[image1[i,j], image2[i,j]] += 1 
    
    # Normalize the joint histogram
    joint_hist = joint_hist / np.sum(joint_hist)
    
    # Calculate entropy of the two images and the joint histogram
    entropy1 = -np.sum(hist1 * np.log2(hist1 + (hist1 == 0)))
    entropy2 = -np.sum(hist2 * np.log2(hist2 + (hist2 == 0)))
    joint_entropy = -np.sum(joint_hist * np.log2(joint_hist + (joint_hist == 0)))
    
    # Calculate mutual information
    mutual_info = entropy1 + entropy2 - joint_entropy
    
    return mutual_info



if __name__ == '__main__':
    IMAGES_FOLDER_VALID = "data_stickers/valid"
    IMAGES_FOLDER_TRAIN = "data_stickers/train"
    IMAGES_FOLDER_TEST = "data_stickers/test"
    mean_image_valid = get_mean_image(IMAGES_FOLDER_VALID)
    mean_image_train = get_mean_image(IMAGES_FOLDER_TRAIN)
    mean_image_test = get_mean_image(IMAGES_FOLDER_TEST)

    cv2.imshow("mean_image_valid", mean_image_valid.astype(np.uint8))
    cv2.imshow("mean_image_train", mean_image_train.astype(np.uint8))
    cv2.imshow("mean_image_test", mean_image_test.astype(np.uint8))

    cv2.waitKey(0)


    print(f"MI valid - train: {mutual_information(mean_image_valid, mean_image_train)}")
    print(f"MI valid - test: {mutual_information(mean_image_valid, mean_image_test)}")
    print(f"MI train - test: {mutual_information(mean_image_train, mean_image_test)}")

    cv2.imwrite("mean_images/mean_image_valid.png", mean_image_valid.astype(np.uint8))
    cv2.imwrite("mean_images/mean_image_train.png", mean_image_train.astype(np.uint8))
    cv2.imwrite("mean_images/mean_image_test.png", mean_image_test.astype(np.uint8))