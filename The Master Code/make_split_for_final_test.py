import os
import numpy as np
import shutil
from tqdm import tqdm




IMAGES_FOLDER_PATH = 'data_stickers/train'


# create new folder called final_test
if not os.path.exists('final_test'):
    os.makedirs('final_test')

#within this folder, create a 5 folders called images100, images200, images300, images400, images500

for i in range(100, 600, 100):
    if not os.path.exists('final_test/images' + str(i)):
        os.makedirs('final_test/images' + str(i))


for i in tqdm(range(100, 600, 100)):
    # copy i images from the train folder to its corresponding folder in final_test
    images = os.listdir(IMAGES_FOLDER_PATH)
    images = [image for image in images if image.endswith('.jpg')]
    # shuffle the images
    np.random.shuffle(images)
    # copy the first i images to the final_test folder
    for image in images[:i]:
        shutil.copy(IMAGES_FOLDER_PATH + '/' + image, 'final_test/images' + str(i) + '/' + image)
    


