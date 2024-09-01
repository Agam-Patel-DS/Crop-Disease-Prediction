# -*- coding: utf-8 -*-
"""Crop Disease Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-NPfEHP3IfxNUP7P1VJhGjpFuuxMGKIV
"""

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np

import os

!kaggle datasets download -d vipoooool/new-plant-diseases-dataset

image_size = 224
target_size = (image_size, image_size)
input_shape = (image_size, image_size, 3)

batch_size = 32
epochs = 25

base_dir = "/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = os.path.join(base_dir,"train")
test_dir = os.path.join(base_dir,"valid")

!unzip '/content/new-plant-diseases-dataset.zip'

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0,
                                                             shear_range = 0.2,
                                                             zoom_range = 0.2,
                                                             width_shift_range = 0.2,
                                                             height_shift_range = 0.2,
                                                             fill_mode="nearest")

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0)

train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size = (image_size, image_size),
                                               batch_size = batch_size,
                                               class_mode = "categorical")

test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size = (image_size, image_size),
                                             batch_size = batch_size,
                                             class_mode = "categorical")

categories = list(train_data.class_indices.keys())
print(train_data.class_indices)

import json
with open('class_indices.json','w') as f:
  json.dump(train_data.class_indices, f)

from IPython.display import FileLink
FileLink(r'class_indices.json')

base_model = tf.keras.applications.MobileNet(weights = "imagenet",
                                             include_top = False,
                                             input_shape = input_shape)

base_model.trainable = False

inputs = keras.Input(shape = input_shape)

x = base_model(inputs, training = False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(len(categories),
                          activation="softmax")(x)

model = keras.Model(inputs = inputs,
                    outputs = x,
                    name="LeafDisease_MobileNet")

optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer = optimizer,
              loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
              metrics=[keras.metrics.CategoricalAccuracy(),
                       'accuracy'])

history = model.fit(train_data,
                    validation_data=test_data,
                    epochs=epochs,
                    steps_per_epoch=150,
                    validation_steps=100)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,loss,c="red",label="Training")
plt.plot(epochs,val_loss,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

epochs = range(len(acc))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,acc,c="red",label="Training")
plt.plot(epochs,val_acc,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# prompt: save the model as h5 extension

model.save('plant_disease.h5')

import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# prompt: use the same function as below but add a section which gives suggestions depending on the diseases also

from google.colab import files
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from IPython.display import FileLink
from PIL import Image
import io

def predict_image(image_path):
  """
  Predicts the disease of a plant image.

  Args:
    image_path: Path to the image file.

  Returns:
    A tuple containing the predicted disease and a list of suggestions.
  """
  # Load the image
  image = Image.open(image_path)
  image = image.resize((224, 224))
  image = np.array(image)
  image = image / 255.0
  image = np.expand_dims(image, axis=0)

  # Load the model
  model = tf.keras.models.load_model('plant_disease.h5')

  # Make the prediction
  prediction = model.predict(image)
  predicted_class = np.argmax(prediction)

  # Load the class indices
  with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

  # Get the predicted disease
  for key, value in class_indices.items():
    if value == predicted_class:
      predicted_disease = key
      break

  # Display the image
  plt.imshow(image[0])
  plt.title(f"Predicted Disease: {predicted_disease}")
  plt.axis('off')
  plt.show()
  # Provide suggestions based on the disease
  suggestions = []
  if predicted_disease == "Tomato___Bacterial_spot":
    suggestions = [
        "Use copper-based fungicides.",
        "Remove infected leaves and fruits.",
        "Maintain good sanitation.",
    ]
  elif predicted_disease == "Tomato___Early_blight":
    suggestions = [
        "Use fungicides with active ingredients like chlorothalonil or mancozeb.",
        "Remove infected leaves.",
        "Provide good air circulation."
    ]
  elif predicted_disease == "Tomato___Late_blight":
    suggestions = [
        "Use fungicides with active ingredients like chlorothalonil or mancozeb.",
        "Remove infected leaves and fruits.",
        "Maintain good sanitation."
    ]
  elif predicted_disease == "Tomato___Leaf_Mold":
    suggestions = [
        "Use fungicides with active ingredients like myclobutanil or azoxystrobin.",
        "Remove infected leaves.",
        "Provide good air circulation."
    ]
  elif predicted_disease == "Tomato___Septoria_leaf_spot":
    suggestions = [
        "Use fungicides with active ingredients like chlorothalonil or mancozeb.",
        "Remove infected leaves.",
        "Provide good air circulation."
    ]
  elif predicted_disease == "Tomato___Spider_mites Two-spotted_spider_mite":
    suggestions = [
        "Use miticides like pyridaben or acetamiprid.",
        "Remove infected leaves.",
        "Maintain good sanitation."
    ]
  elif predicted_disease == "Tomato___Target_Spot":
    suggestions = [
        "Use fungicides with active ingredients like chlorothalonil or mancozeb.",
        "Remove infected leaves.",
        "Provide good air circulation."
    ]
  elif predicted_disease == "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
    suggestions = [
        "Remove infected plants.",
        "Use insecticides to control whiteflies.",
        "Use resistant varieties."
    ]
  elif predicted_disease == "Tomato___Tomato_mosaic_virus":
    suggestions = [
        "Remove infected plants.",
        "Use insecticides to control aphids.",
        "Use resistant varieties."
    ]
  elif predicted_disease == "Potato___Early_blight":
    suggestions = [
        "Use fungicides with active ingredients like chlorothalonil or mancozeb.",
        "Remove infected leaves.",
        "Provide good air circulation."
    ]
  elif predicted_disease == "Potato___Late_blight":
    suggestions = [
        "Use fungicides with active ingredients like chlorothalonil or mancozeb.",
        "Remove infected leaves and tubers.",
        "Maintain good sanitation."
    ]
  elif predicted_disease == "Potato___healthy":
    suggestions = [
        "Continue to monitor your plants for any signs of disease."
    ]
  elif predicted_disease == "Pepper___Bacterial_spot":
    suggestions = [
        "Use copper-based fungicides.",
        "Remove infected leaves and fruits.",
        "Maintain good sanitation."
    ]
  elif predicted_disease == "Pepper___healthy":
    suggestions = [
        "Continue to monitor your plants for any signs of disease."
    ]

  return predicted_disease, suggestions

path = input("Enter the path of the image: ")
predict_image(path)

