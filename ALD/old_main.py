from __future__ import print_function, division

import csv
import os

import cv2
import torch.optim as optim
import torch
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from ALD.dataLoader import Rescale, RandomCrop, ToTensor, LandmarksDataset
import ALD.models
import ALD.train
import ALD.lossFunction
import argparse
from PIL import Image

plt.ion()  # interactive mode

convert_tensor = transforms.ToTensor()

def get_original_name(_path):
    return os.listdir(_path)[0]


parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--landmarkNum", type=int, default=83)
parser.add_argument("--image_scale", default=(800, 640), type=tuple)
parser.add_argument("--use_gpu", type=int, default=0)
parser.add_argument("--spacing", type=float, default=0.1)
parser.add_argument("--R1", type=int, default=41)
parser.add_argument("--R2", type=int, default=41)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--data_enhanceNum", type=int, default=1)
parser.add_argument("--stage", type=str, default="train")
parser.add_argument("--saveName", type=str, default="test1")
parser.add_argument("--testName", type=str, default="30cepha100_fusion_unsuper.pkl")
parser.add_argument("--dataRoot", type=str, default="../image_to_test/A/")
parser.add_argument("--supervised_dataset_train", type=str, default="test/")
parser.add_argument("--supervised_dataset_test", type=str, default="test/")
parser.add_argument("--unsupervised_dataset", type=str, default="test/")
parser.add_argument("--trainingSetCsv", type=str, default="image_name.csv")
parser.add_argument("--testSetCsv", type=str, default="image_name.csv")
parser.add_argument("--unsupervisedCsv", type=str, default="image_name.csv")
parser.add_argument("--image_name", type=str, default="image.jpg")


def create_csv(_name):
    _f = open("../image_to_test/A/image_name.csv", "w", newline="\n")
    writer = csv.writer(_f)
    writer.writerow(["Name"] + ["col"] * 166)
    writer.writerow([_name] + ["0.0"] * 166)
    _f.close()


def ald(_img):
    config = parser.parse_args()

    model_ft = torch.load('../model_700_83_points_positive.pt', map_location=torch.device("cpu"))

    _img = _img.resize(config.image_scale)
    _image = convert_tensor(_img)
    _image = _image.unsqueeze(0)

    return ALD.train.predict(model_ft, _image, config)


if __name__ == "__main__":
    img = Image.open('../image_to_test/A/test/{}'.format(get_original_name("../image_to_test/A/test")))
    points = ald(img)
    print(points[0])
