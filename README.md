# ViT-CAMNet-Model-

# Diabetic Retinopathy Detection

## About

Diabetic Retinopathy (DR) is a common complication of diabetes mellitus, which causes lesions on the retina that effect vision. If it is not detected early, it can lead to blindness. In this project, I use tensorflow to develop neural networks which can detect diabetic retinopathy from retinal images. 

## Data

The dataset is obtained from a ([Diabetic Retinopathy Dataset)](https://www.kaggle.com/datasets/sachinkumar413/diabetic-retinopathy-dataset). The original training data consists of about 35000 images taken of different people, using different cameras, and of different sizes. The dataset is heavily biased as most of the images are classified as not having DR. Pertaining to the preprocessing section, this data is also very noisy, and requires multiple preprocessing steps to get all images to a useable format for training a model.

<p align = "center">
<img align="center" src="images/original_dataset.png" alt="Original Dataset"/>
</p>
