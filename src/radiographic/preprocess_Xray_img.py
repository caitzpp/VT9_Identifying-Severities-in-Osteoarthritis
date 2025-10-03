import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import config
from utils.load_utils import load_image

box_width = 800
box_height = 800

t=100

DATAPATH = config.SCHULTHESS_DATAPATH
test_datapath = os.path.join(DATAPATH, 'test')
train_datapath = os.path.join(DATAPATH, 'train')

datapaths = [test_datapath, train_datapath]
# datapaths = [test_datapath]

if __name__ == '__main__':
    df = pd.DataFrame(columns=['file', 'top', 'bottom', 'left', 'right', 'joint_center_x', 'joint_center_y'])
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

            _, thres = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)

            thres = cv2.bitwise_not(thres)

            row_sums = np.sum(thres, axis=1)
            joint_row = np.argmax(row_sums)
            while joint_row > 650 or joint_row<300 :
                row_sums[joint_row] = 0
                joint_row = np.argmax(row_sums)

            center_col = img.shape[1] // 2
            # col_sums = np.sum(thres, axis=0)
            # # print(col_sums[center_col])
            # center_col = np.argmin(col_sums[512-100:512+100]) + 512 - 100
            # while center_col > 550 or center_col<450 :
            #     col_sums[center_col] = 0
            #     center_col = np.argmin(col_sums)

            joint_center_x = center_col
            joint_center_y = joint_row

            top = max(joint_center_y - box_height // 2, 0)
            bottom = min(joint_center_y + box_height // 2, img.shape[0])
            left = max(joint_center_x - box_width // 2, 0)
            right = min(joint_center_x + box_width // 2, img.shape[1])

            new_row = pd.DataFrame({
                'file': [file],
                'top': [top],
                'bottom': [bottom],
                'left': [left],
                'right': [right],
                'joint_center_x': [joint_center_x],
                'joint_center_y': [joint_center_y]
            })

            df = pd.concat([df, new_row], ignore_index=True)

            roi = img[top:bottom, left:right]
            roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)

            img_savepath = os.path.join(SAVE_PATH, os.path.basename(os.path.dirname(file)), os.path.basename(file))

            cv2.imwrite(img_savepath, roi)
            print(f'Saved: {os.path.basename(file)}')
    df.to_csv(os.path.join(DATAPATH, f'{box_width}x{box_height}_imgs', f'crop_coordinates_t{t}.csv'), index=False)

        