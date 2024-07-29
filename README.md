# Automated Cephalometric Points Marking System

## Introduction
This repository contains the source code used in the research described in the article: “Automated Cephalometric Points Marking System”

## Requeirements
Python 3.10

All required libraries can be found in the requirements.txt file

## Dataset
The data/images_to_process folder contains a collection of images derived from the public Image_Annotation_EvaluationCode collection. In addition, there are 4 “400_senior_*.csv” files in the data folder, which contain the prepared information needed to train the model. Each file contains information about the location of a landmarks. In addition, in the “config.py” file it is possible to change selected parameters.

## Image Augmentation 
In order to create the image collections described in the mentioned article, use the “image_augumentation.py” script, in which we indicate the locations of the folder containing the images to be augmented next.

## Training
To perform the process of training the model on the proposed dataset, run the “train_model.py” script. Information about the data for the training process itself can be found in the “config.py” file. 

## Prediction
In order to detect the location of landmarks for a given set of images, you need to run the “predict.py” script. In this script you need to specify the path to the model, by default it is set to the base model discussed in the article. Information about the data for the prediction process can be found in the “config.py” file. 