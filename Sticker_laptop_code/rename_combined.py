import os


images = os.listdir("combined")

for i, image in enumerate(images):
    os.rename("combined/" + image, "combined/image" + str(i) + ".jpg")