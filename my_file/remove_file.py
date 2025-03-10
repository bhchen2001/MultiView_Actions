# remove the file under /home/bhchen/action_recognition/dataset/nturgb+d_rgb with 'S006' 'S007' 'S008' 'S009' 'S010' and end with '.avi'
# filename format: S006C001P001R001A001_rgb.avi

import os
import shutil

def filter_small_dataset():
    path = '/home/bhchen/action_recognition/dataset/nturgb+d_rgb'
    filter_list = ['S017']
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.avi') and any(filter in file for filter in filter_list):
                print('Remove file:', os.path.join(root, file))
                os.remove(os.path.join(root, file))

if __name__ == '__main__':
    filter_small_dataset()