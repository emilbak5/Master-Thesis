import os


FOLDERS = [
    'grapped_computer_top',
    'grapped_computer_open',
    'grapped_computer_bottom',
    'sticker_white_original',
    'sticker_black_original'
]

def clear_folders():
    for folder in FOLDERS:
        images = os.listdir(folder)
        for image in images:
            os.remove(folder + '/' + image)


if __name__ == "__main__":
    clear_folders()
