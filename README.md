# Supplemental Code for Automated Cephalometric Points Marking System

[Paper](https://doi.org/10.3390/diagnostics16111638)

[Kaja Szwarczyńska](https://orcid.org/0000-0003-0645-0064)<sup>1</sup>, [Eryk Kosmala](https://orcid.org/0009-0008-3953-1622)<sup>2</sup>, [Maciej Antczak](https://orcid.org/0000-0002-5320-2023)<sup>2,3</sup>, [Ivo Domagała](https://orcid.org/0000-0002-7976-0605)<sup>1</sup>, Barbara Biedziak<sup>1</sup> and [Jędrzej Musiał](https://orcid.org/0000-0003-3018-5010)<sup>2</sup>

<sup>1</sup>Department of Orthodontics and Craniofacial Anomalies, Poznan University of Medical Sciences, Fredry 10, 60-812 Poznan, Poland

<sup>2</sup>Institute of Computing Science; European Centre for Bioinformatics and Genomics, Poznan University of Technology, 60-965 Poznan, Poland

<sup>3</sup>Institute of Bioorganic Chemistry, Polish Academy of Sciences, 61-704 Poznan, Poland

## Introduction
This repository contains the source code used in the research described in the article: “Automated Cephalometric Points Marking System”. The code uses the tool discussed in the article: Anatomic-Landmark-Detection [1].

## Requeirements
- Python 3.10
- All required libraries can be found in the `requirements.txt` file

## Dataset
The `data/images_to_process` folder contains a collection of images derived from the public [Image_Annotation_EvaluationCode](https://figshare.com/s/37ec464af8e81ae6ebbf) collection. In addition, there are 4 `400_senior_*.csv` files in the data folder, which contain the prepared information needed to train the model. Each file contains information about the location of a landmarks. In addition, in the `config.py` file it is possible to change selected parameters.

## Image Augmentation 
In order to create the image collections described in the mentioned article, use the `image_augumentation.py` script, in which we indicate the locations of the folder containing the images to be augmented next.

## Training
To perform the process of training the model on the proposed dataset, run the `train_model.py` script. Information about the data for the training process itself can be found in the `config.py` file. 

## Prediction
In order to detect the location of landmarks for a given set of images, you need to run the `predict.py` script. In this script you need to specify the path to the model, by default it is set to the base model discussed in the article. Information about the data for the prediction process can be found in the `config.py` file. 

[1] R. Chen, Y. Ma, N. Chen, D. Lee, and W. Wang, “Cephalometric landmark detection by attentive feature pyramid fusion and regression-voting,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part III. Berlin, Heidelberg: Springer-Verlag, 2019, p. 873–881. [Online]. Available: https://doi.org/10.1007/978-3-030-32248-9_97
