import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config
from utils.load_utils import load_image

box_width = 600
box_height = 600

DATAPATH = config.SCHULTHESS_DATAPATH
test_datapath = os.path.join(DATAPATH, 'test')
train_datapath = os.path.join(DATAPATH, 'train')

datapaths = [test_datapath, train_datapath]

if __name__ == '__main__':
    for datapath in datapaths:
        SAVE_PATH = os.path.join(DATAPATH, f'{box_width}x{box_height}_imgs', os.path.basename(datapath))
        os.makedirs(SAVE_PATH, exist_ok=True)

        filenames = []
        for root, dirs, files in os.walk(datapath):
            for file in files:
                if file.endswith('.png'):
                    filenames.append(os.path.join(root, file))

        subdirs = os.listdir(datapath)
        for subdir in subdirs:
            os.makedirs(os.path.join(SAVE_PATH, subdir), exist_ok=True)

        for file in filenames:
            img = load_image(file)
            img = np.array(img)

            _, thres = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

            thres = cv2.bitwise_not(thres)

            row_sums = np.sum(thres, axis=1)
            joint_row = np.argmax(row_sums)

            center_col = img.shape[1] // 2

            joint_center_x = center_col
            joint_center_y = joint_row

            top = max(joint_center_y - box_height // 2, 0)
            bottom = min(joint_center_y + box_height // 2, img.shape[0])
            left = max(joint_center_x - box_width // 2, 0)
            right = min(joint_center_x + box_width // 2, img.shape[1])

            roi = img[top:bottom, left:right]
            roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)

            img_savepath = os.path.join(SAVE_PATH, os.path.basename(os.path.dirname(file)), os.path.basename(file))

            cv2.imwrite(img_savepath, roi)
            print(f'Saved: {os.path.basename(file)}')

        