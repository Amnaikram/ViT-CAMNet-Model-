# ViT-CAMNet-Model-

# Diabetic Retinopathy Detection and Classification

## About

Diabetic Retinopathy (DR) is a common complication of diabetes mellitus, which causes lesions on the retina that effect vision. If it is not detected early, it can lead to blindness. In this project, I use tensorflow to develop neural networks which can detect diabetic retinopathy from retinal images. 

## Data 

The dataset is obtained from a ([Diabetic Retinopathy Dataset)](https://www.kaggle.com/datasets/sachinkumar413/diabetic-retinopathy-dataset). This dataset consists of images categorized into five classes, labeled 0 to 4, with each label representing a progressive increase in disease severity: 0 Healthy, 1- Mild DR, 2- Moderate DR, 3- Proliferative DR, and 4- Severe DR. Each image has dimensions of 256×256 pixels, and the dataset contains a total of 2,750 images. However, the dataset is highly imbalanced, with the class distribution as follows: 1,000 Healthy, 370 Mild, 900 Moderate, 290 Proliferative, and 190 Severe DR images.

<p align = "center">
<img align="center" src="dataset.png" alt="Original Dataset"/>
</p>

## Data Preprocessing & Augmentation (Refer 

The preprocessing pipeline consists of the following:
1. Random oversampling
2. Resizing
3. Normalization
4.  ZCA (Zero Component Analysis) Whitening

<p align = "center">
<img align="center" src="Preprocessed ViT.png" alt="Original Dataset"/>
</p>

## Vision Transformer

The Vision Transformer (ViT) model adapts the transformer architecture from the field of natural language processing (NLP) to image recognition tasks. Transformers initially transformed natural language processing (NLP) by allowing models to capture long-range dependencies and context within text. Drawing inspiration from this success, researchers have begun applying transformers to computer vision, aiming
 to potentially supplant convolutional neural networks (CNNs). ViT has three variants,  namely Base, Large and Huge. We applied fine-tuned vit base patch32 224 to perform multi-classification of DR. The specific variant vit base patch32 224 is used for multi level classification of DR.
 
## Training and Evaluation


The program directly accesses images from the directory to train the model without converting the entire dataset to numpy format (which consumes a lot of space ~20GB).

## Results

The current models return the following scores for binary classification (DR vs No DR) on the dataset.
| Model | Accuracy |
| :-----: | :-----: |
| Standard CNN (Training) | 82.2% |
| Standard CNN (Validation) | 82.2% |
| InceptionV3 (Training) | 86.0% |
| InceptionV3 (Validation) | 85.8% |
