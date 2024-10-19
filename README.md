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

   ## Neural Network Architecture

I train two kinds of neural networks in this project. The first is a standard Convolution Neural Network (CNN) consisting of basic convolution and pooling layers. The second is the InceptionV3 model developed by Google. The architectures for both the models are included in the [neural_nets](https://github.com/ramanakshay/Diabetic-Retinopathy-Detection/tree/main/neural_nets) folder.

## Training

The data is split into two parts. 80% of the data is used for training and the remaining 20% is used for validation. The training data is the data used to train the model and the validation data is used to tune the model’s hyper parameters (optimizer, learning rate, batch size, epoch size...). We perform transfer learning on the InceptionV3 model by using a pre-trained network trained on the ImageNet dataset. 

## Using Google Colab

<p align = "center">
<img align="center" src="images/colab_upload.png" alt="Upload"/>
</p>

To train the model using Google Colab, first import the neural network architecture (.pynb file) from the neural_nets folder. Then, import the .csv file containing the labels for the augmented dataset. Import the dataset by unzipping the image folder from Google Drive using the following command - 

`!unzip '/content/drive/MyDrive/filepath/filename.zip'`

The program directly accesses images from the directory to train the model without converting the entire dataset to numpy format (which consumes a lot of space ~20GB).

## Results

The current models return the following scores for binary classification (DR vs No DR) on the dataset.
| Model | Accuracy |
| :-----: | :-----: |
| Standard CNN (Training) | 82.2% |
| Standard CNN (Validation) | 82.2% |
| InceptionV3 (Training) | 86.0% |
| InceptionV3 (Validation) | 85.8% |
