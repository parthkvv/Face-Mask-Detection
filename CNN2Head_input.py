import numpy as np
import os
import cv2
import scipy
import random

DATASET_FOLDER = './content/gdrive/MyDrive/'
NUM_SMILE_IMAGE = 4000
SMILE_SIZE = 48
EMOTION_SIZE = 48


def getFacemaskImage():
    print('Load smile image...................')
    X1 = np.load(DATASET_FOLDER + 'train_part0.npy')
    X2 = np.load(DATASET_FOLDER + 'test_part0.npy')

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done !')
    print('Number of facemask train data: ',str(len(train_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data
