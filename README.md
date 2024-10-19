# ViT-CAMNet-Model-

# Diabetic Retinopathy Detection and Classification

## About

Diabetic Retinopathy (DR) is a common complication of diabetes mellitus, which causes lesions on the retina that effect vision. If it is not detected early, it can lead to blindness. In this project, I use tensorflow to develop neural networks which can detect diabetic retinopathy from retinal images. 

## Data

The dataset is obtained from a ([Diabetic Retinopathy Dataset)](https://www.kaggle.com/datasets/sachinkumar413/diabetic-retinopathy-dataset). This dataset consists of images categorized into five classes, labeled 0 to 4, with each label representing a progressive increase in disease severity: 0 Healthy, 1- Mild DR, 2- Moderate DR, 3- Proliferative DR, and 4- Severe DR. Each image has dimensions of 256×256 pixels, and the dataset contains a total of 2,750 images. However, the dataset is highly imbalanced, with the class distribution as follows: 1,000 Healthy, 370 Mild, 900 Moderate, 290 Proliferative, and 190 Severe DR images.

<p align = "center">
<img align="center" src="images/original_dataset.png" alt="Original Dataset"/>
</p>

## Data Preprocessing & Augmentation*

The preprocessing pipeline consists of the following:
1. Random oversampling
2. Resizing
3. Normalization
4.  ZCA (Zero Component Analysis) Whitening
