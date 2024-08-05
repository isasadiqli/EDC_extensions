import os
import random
from shutil import copyfile
import pandas as  pd
import numpy as np
import cv2
import argparse

from extentions.image_transformations import do_transformation

random.seed(1)

parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data-folder', default='/data/disk2T1/guoj/ISIC2018/original', type=str)
parser.add_argument('--save-folder', default='/data/disk2T1/guoj/ISIC2018', type=str)
config = parser.parse_args()

source_dir = config.data_folder
target_dir = config.save_folder

train_csv = np.array(pd.read_csv(
    os.path.join(source_dir, 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')))
valid_csv = np.array(pd.read_csv(
    os.path.join(source_dir, 'ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv')))

train_normal_path = []
valid_normal_path = []
valid_abnormal_path = []


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

printProgressBar(0, len(train_csv), prefix = 'Progress:', suffix = 'Complete', length = 50)
for i, line in enumerate(train_csv):
    if line[2] == 1:
        train_normal_path.append(os.path.join(source_dir, 'ISIC2018_Task3_Training_Input', line[0] + '.jpg'))
    printProgressBar(i + 1, len(train_csv), prefix='Progress:', suffix='Complete', length=50)

printProgressBar(0, len(valid_csv), prefix = 'Progress:', suffix = 'Complete', length = 50)
for i, line in enumerate(valid_csv):
    if line[2] == 1:
        valid_normal_path.append(os.path.join(source_dir, 'ISIC2018_Task3_Validation_Input', line[0] + '.jpg'))
    else:
        valid_abnormal_path.append(os.path.join(source_dir, 'ISIC2018_Task3_Validation_Input', line[0] + '.jpg'))
    printProgressBar(i + 1, len(valid_csv), prefix='Progress:', suffix='Complete', length=50)

target_train_normal_dir = os.path.join(target_dir, 'train', 'NORMAL')
if not os.path.exists(target_train_normal_dir):
    os.makedirs(target_train_normal_dir)

target_test_normal_dir = os.path.join(target_dir, 'test', 'NORMAL')
if not os.path.exists(target_test_normal_dir):
    os.makedirs(target_test_normal_dir)

target_test_abnormal_dir = os.path.join(target_dir, 'test', 'ABNORMAL')
if not os.path.exists(target_test_abnormal_dir):
    os.makedirs(target_test_abnormal_dir)



transformation_type = 'rotate'
printProgressBar(0, len(train_normal_path), prefix = 'Progress:', suffix = 'Complete', length = 50)
for i, f in enumerate(train_normal_path):

    f = do_transformation(f, transformation_type=transformation_type)  # I can choose the transformation type from here

    copyfile(f, os.path.join(target_train_normal_dir, os.path.basename(f)))
    printProgressBar(i + 1, len(train_normal_path), prefix='Progress:', suffix='Complete', length=50)


printProgressBar(0, len(valid_normal_path), prefix = 'Progress:', suffix = 'Complete', length = 50)
for i, f in enumerate(valid_normal_path):

    f = do_transformation(f, transformation_type=transformation_type)  # I can choose the transformation type from here

    copyfile(f, os.path.join(target_test_normal_dir, os.path.basename(f)))
    printProgressBar(i + 1, len(valid_normal_path), prefix='Progress:', suffix='Complete', length=50)


printProgressBar(0, len(valid_abnormal_path), prefix = 'Progress:', suffix = 'Complete', length = 50)
for i, f in enumerate(valid_abnormal_path):

    f = do_transformation(f, transformation_type=transformation_type)  # I can choose the transformation type from here

    copyfile(f, os.path.join(target_test_abnormal_dir, os.path.basename(f)))
    printProgressBar(i + 1, len(valid_abnormal_path), prefix='Progress:', suffix='Complete', length=50)
