# Vegetables-Classifier using Convolutional Neural Network
This repository contains code for a machine learning project that uses a Convolutional Neural Network (CNN) to classify vegetable images. The goal of this project is to train a model that can accurately identify different types of vegetables based on input images.


### 1. Project Overview
The project involves training a CNN using TensorFlow and Keras libraries to classify vegetable images into 15 different classes. The dataset used for training, validation, and testing consists of images of various vegetables.


### 2. Getting Started
To run this project, follow these steps:

Clone the repository to your local machine.
Install the required libraries, including TensorFlow, Keras, Matplotlib, and NumPy.
Set up the dataset directory structure according to the instructions provided in the code.
Run the code to train the classifier.

### 3. Dataset
The dataset used in this project consists of 15 classes of vegetable images. The images are divided into three sets: training, validation, and testing. The training set contains 15,000 images, the validation set contains 3,000 images, and the testing set contains 3,000 images.

### 4. Preprocessing
The input images are preprocessed using the ImageDataGenerator class from Keras. The images are rescaled and augmented by applying horizontal flips to increase the diversity of the training data.

### 5. Model Architecture
The model is a sequential CNN with the following layers:

Convolutional layers with ReLU activation and max pooling.
Flatten layer to convert the 2D feature maps into 1D.
Dense layers with ReLU activation and batch normalization.
Output layer with softmax activation for multi-class classification.

### 6. Model Training
The model is compiled with the RMSprop optimizer and categorical cross-entropy loss function. During training, an early stopping callback is used to monitor the accuracy and stop training if there is no improvement for a specified number of epochs.

### 7. Training Results
The model is trained for 50 epochs with a batch size of 10. The training and validation accuracies and losses are recorded for each epoch.

### 8. Evaluation
The model's performance is evaluated using the testing set. The final accuracy and loss metrics are calculated on this independent set of images to assess the model's generalization ability.

### 9. Conclusion
This project demonstrates the application of a CNN for vegetable image classification. By training the model on a large dataset and fine-tuning the architecture, we achieved impressive accuracy on both the training and testing sets.

Feel free to explore the code, dataset, and experiment with different hyperparameters to further improve the model's performance.
