# Pneumonia Detection from Chest X-Ray Images

This project's goal is to detect pneumonia from chest X-ray images. It uses deep learning models and the pytorch framework to classify images as either 'NORMAL' or 'PNEUMONIA'. The project has two models defined and available for training and evaluation and prediction. One is a basic model, made up of 5 layers that is trained from scratch. The other is the well-known ResNet50 model, applied with transfer learning. Due to our limited training dataset, around 5k images, you can see that the basic model performs reasonably well for this task.

The dataset used in this project is  the 'Chest X-Ray Images (Pneumonia)' from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). This dataset is made up of test/train/val directories that have the following image counts:
* test: 624
* train: 5216
* val: 16

Because the original val set was unusable (only 8 samples per class), I manually copy-pasted it into the train directory and performed an 80/20 stratified split, ending up with ~1k validation images.

### Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Project Structure](#project-structure)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Configuration](#configuration)
* [Model Performance](#model-performance)
* [Grad-Cam](#grad-cam)

### Introduction

Pneumonia is an infection that inflames the air sacs in one or both lungs. This project leverages computer vision and deep
learning to build a model that can assist in this diagnostic process, by analyzing chest X-ray images.

### Features

#### Exploratory Data Analysis

The EDA, found in the `notebooks` directory revealed several key insights:
* Class imbalance: The dataset is imbalanced, with significant more "PNEUMONIA" than "NORMAL" images
* Image size variability: the original images come in various shapes and sizes, necessitating a resizing preprocess step
* Pixel intensity distribution: "PNEUMONIA" images tend to have a higher concentration of pixels in the brighter range, 
which aligns with the radiological finding of white spots (infiltrates) in the lungs of pneumonia patients

The `data_processing.py` script handles the following preprocessing steps:
* Image resizing: all images are resized to a uniform dimension of 256 by 256 pixels
* Data augmentation: to address the class imbalance, the training data is augmented with random rotations, shifts, shears and zooms
* Normalization: Pixel values are scaled to a range between 0 and 1

### Project Structure

The project is organized as follows:
```
.
├── CLAUDE.md
├── README.md
├── config
│   ├── base_cnn.yaml
│   ├── base_config.yaml
│   └── resnet50.yaml
├── data
│   ├── processed
│   └── raw
├── models
├── notebooks
│   └── 01_Exploratory_Data_Analysis.ipynb
├── outputs
│   ├── grad_cam
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── data_processing.py
│   ├── evaluate.py
│   ├── grad_cam.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── util.py
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── test_data_processing.py
    ├── test_model.py
    └── test_utils.py

```

* `config` contains YAML files for configuring the models and data paths
* `data` where the raw and processed data is stored
* `models` where the trained models are stored
* `notebooks` contains Jupyter Notebook for EDA
* `src` python source code for data processing, training and evaluation
* `tests` where the unit tests are defined
* `outputs` convenient directory to collect different outputs (learning curves, confusion matrix, etc.)
* `requirements.txt` required packages for this project

## Getting Started

In order to get this up and running please do the following:

### Pre-requisites:

* Python 3.x
* pip
* Download the dataset 'Chest X-Ray Images (Pneumonia)' from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

1. Clone the repo
    `git clone https://github.com/RaresPopa22/ChestXRayImages`
2. Install Python packages
    `pip install -r requirements.txt`


## Usage

Because of their large size, the pre-trained models were not pushed upstream. To train and evaluate their performance, please follow these steps:

### Training the Models

In order to train a model, run the `train.py` script with the desired configuration:

* Train base CNN model: 
    `python -m src.train --config config/base_cnn.yaml`
* Train ResNet50 model:
    `python -m src.train --config config/resnet50.yaml`


### Evaluating the models

After training, you can evaluate and compare the models using the evaluate.py script:
    `python -m src.evaluate --configs config/base_cnn.yaml config/resnet50.yaml`

This will print a classification report for each model, a comparison summary, and will plot a Precision-Recall curve plot.

### Making individual predictions

Using one of the models defined in this project, you can make a prediction for a single image by running the following command:
    `python -m src.predict --config config/base_cnn.yaml --input data/raw/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg --output predict.json`

Tweak accordingly the input image path.

### Running the diagnostic GradCam

If you want to run the grad_cam script yourself, here's an example for base_cnn:
    `python -m src.grad_cam --configs config/base_cnn.yaml`

### Running the unit tests suite

There are 19 unit tests placed in different files, all under tests directory. They can be easily run locally, for all suite:
    `pytest tests`
or if one specific file is targeted:
    `pytest tests/test_model.py`

### CI/CD

I previously mentioned the unit tests, so this section comes in extension of that. The unit tests are run as part of the ci pipeline to ensure that no breaking changes are introduced. It install the required dependencies defined in requirements.txt and next it runs the whole suite of unit tests. They are being run on pull requests or pushes on that PR opened.

## Configuration

This project uses YAML files for configuration, making it easy to manage model parameters and data paths.

* `base_config.yaml` contains the base configuration, including data paths
* `resnet50.yaml` and `base_cnn.yaml` contains model-specific parameters

## Model Performance

The models were evaluated on a held-out test set. The main metrics used for comparison are the Area Under the Precision-Recall
Curve (AUPRC), ROC AUC and F1-Score

The table below summarizes the performance of the trained models:


| Model    | AUPRC  | ROC AUC | Recall (Pneumonia) | Precision (Pneumonia) | F1-Score (Pneumonia) |
|----------|--------|---------|--------------------|-----------------------|----------------------|
| resnet50 | 0.968 | 0.958  | 0.98               | 0.91                  | 0.92                 |
| base_cnn | 0.966   | 0.953  | 0.98               | 0.86                  | 0.92                 |

Here are the detailed classification reports:

### BASE CNN
```
INFO:__main__:Using optimal threshold=0.37587299942970276
INFO:__main__:Classification report for base_cnn
INFO:__main__:
              precision    recall  f1-score   support

      NORMAL       0.96      0.75      0.84       234
   PNEUMONIA       0.87      0.98      0.92       390

    accuracy                           0.89       624
   macro avg       0.91      0.87      0.88       624
weighted avg       0.90      0.89      0.89       624
```

### ResNet50

```
INFO:__main__:Using optimal threshold=0.23028059303760529
INFO:__main__:Classification report for resnet50
INFO:__main__:
              precision    recall  f1-score   support

      NORMAL       0.96      0.74      0.84       234
   PNEUMONIA       0.86      0.98      0.92       390

    accuracy                           0.89       624
   macro avg       0.91      0.86      0.88       624
weighted avg       0.90      0.89      0.89       624
```

And the final report:
```
             AUPRC   ROC AUC  F1-Score (PNEUMONIA)
Model                                             
base_cnn  0.966043  0.952783              0.920482
resnet50  0.968381  0.958306              0.919568
```

Both models perform well, with the resnet50 model having a slight edge in the AUPRC, but base_cnn having a slight better precision. The high precision and recall scores indicate that the models are effective at distinguishing between normal and pneumonia X-rays. This also points to
the fact that we don't need a deep neural network here, as the real constraint is the dataset, for training we have around 5k images. Training
a simple model from scratch proved to be very close to a top notch fined tuned model like resnet50.

![outputs/PRAUC.png](outputs/PRAUC.png)

Having in mind that we are on the eval subject, here are the confusion matrices for both models. Based on this I would say BaseCNN wins by a
whisker, recalling one extra Pneumonia case, which real world, it means more than just a number. Also BaseCNN has a better precision for the
positive case.

### Base_CNN
![outputs/base_cnn_confusion_matrix.png](outputs/base_cnn_confusion_matrix.png)

### Resnet50
![outputs/resnet50_confusion_matrix.png](outputs/resnet50_confusion_matrix.png)


In the end of this chapter, I will like to present here the learning curves for both models. You can see that resnet50 had already very good initial weights, as its
learning curve is smoother for both training and eval. On the other hand for the base_cnn we have a jumpy learning curve for eval, and this
is because of the very small eval set that we used here, around 1k images.

### Base_CNN
![outputs/learning_curve_base_cnn.png](outputs/learning_curve_base_cnn.png)

### Resnet50
![outputs/learning_curve_resnet50.png](outputs/learning_curve_resnet50.png)

## Grad-Cam

Let's take a look at why our two models made a particular decision, or at least try to guess. We'll do that using the heatmap
that highlights the regions that were the most important in the prediction.

We ran 10 examples, 5 positive, patient had pneumonia, and 5 negative, normal lungs. Anyone is invited to draw their own conclusions but here is what I made of this little experiment:

#### BASE CNN
- on pneumonia cases, it focuses on the left side of the patient, in the lower left lung area
- on 1.png it highlights the shoulder area, which is not clinically meaningful at all, the model
may pick up on non important medical stuff
- on normal cases, it highlights the abdomen area, it might have learned if a bright central structure exists, then the patient has clear lungs
- it shows how fragile the model heuristics are

#### RESNET50
- on normal cases it shows almost no activation, which is a very good principle
- on pneumonia cases, it has an erratic hotspot pattern, it even focuses on the armpit areas in some examples
- it also misclassified 0.png, in there we can see it looks at the edges of the image

I think BaseCNN once again shows a better predictive behavior overall, being more consistently confident. But at the end of the day neither model is reliably looking at the right thing every time.

