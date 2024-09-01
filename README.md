# Crop-Disease-Prediction
This project focuses on identifying plant diseases using a deep learning model built on the MobileNet architecture. The model is designed to take an image of a plant leaf as input and predict the disease affecting the plant, if any.

## Overview
LeafDisease_MobileNet is a TensorFlow-based deep learning model designed to classify plant leaf images into various categories of diseases or healthy states. The model leverages the power of MobileNet, a lightweight deep learning architecture, to perform efficient and accurate image classification tasks, making it suitable for deployment on devices with limited computational resources.

## Model Architecture
The model is composed of the following key components:

### Input Layer:

The model starts with a Keras Input layer that accepts images of a specified shape. This layer forms the basis of the image data that will be passed through the network for feature extraction and classification.
python
Copy code
inputs = keras.Input(shape=input_shape)

### Base MobileNet Model:

A pre-trained MobileNet model is used as the backbone for feature extraction. The model is set to not train further (training=False), which allows the use of pre-learned features from the large-scale ImageNet dataset.
python
Copy code
x = base_model(inputs, training=False)


### Global Average Pooling:

This layer performs downsampling by computing the average of each feature map, significantly reducing the spatial dimensions and making the model more robust against variations in the input image.
python
Copy code
x = tf.keras.layers.GlobalAveragePooling2D()(x)

### Dropout Layer:

A dropout layer with a rate of 20% is applied to prevent overfitting by randomly setting a fraction of input units to zero during training.
python
Copy code
x = tf.keras.layers.Dropout(0.2)(x)

### Dense Output Layer:

The final layer is a dense (fully connected) layer with a softmax activation function. The number of units in this layer corresponds to the number of disease categories (including healthy leaves), and the softmax function outputs a probability distribution across these categories.
python
Copy code
x = tf.keras.layers.Dense(len(categories), activation="softmax")(x)

### Model Compilation:

The model is compiled into a final form that can be trained and evaluated. The complete model takes the input from the first layer and produces an output in the form of disease classification.
python
Copy code
model = keras.Model(inputs=inputs, outputs=x, name="LeafDisease_MobileNet")

## Dataset
The model was trained and tested on a comprehensive dataset that includes images of plant leaves affected by various diseases as well as healthy leaves. The dataset is structured into categories, each representing a different type of plant disease or a healthy state.

Source: The dataset used for training and testing is available on Kaggle: Plant Disease Dataset
Categories: The dataset contains images of leaves categorized into multiple classes, including but not limited to diseases like Apple Scab, Apple Black Rot, and healthy leaves for various plants.

## How It Works
Input: The model takes a preprocessed image of a plant leaf as input.
Prediction: The image is passed through the model layers, where the MobileNet base extracts features, followed by classification layers that predict the likelihood of various diseases.
Output: The model outputs the predicted class (disease or healthy state) along with a probability score.

## Usage
To use the model in your project:

Load the trained model and preprocess the input image to match the input shape expected by the model.
Use the model to predict the class of the input image.
Display or utilize the prediction as needed in your application.
