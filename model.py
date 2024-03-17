import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Define the dataset directory and subdirectories
dataset_dir = 'D:\MY PROJECT'
normal_dir = os.path.join(dataset_dir, 'D:\MY PROJECT\\normal')
abnormal_dir = os.path.join(dataset_dir, 'D:\MY PROJECT\\abnormal')
infarction_dir = os.path.join(dataset_dir, 'D:\MY PROJECT\\myocardiac 240 X 12 = 2880')

# Initialize empty lists for image paths and labels
image_paths = []
labels = []

# Loading normal ECG images
for image_filename in os.listdir(normal_dir):
    image_path = os.path.join(normal_dir, image_filename)
    image_paths.append(image_path)
    labels.append('normal')  # 'normal' represents normal

# Loading abnormal ECG images
for image_filename in os.listdir(abnormal_dir):
    image_path = os.path.join(abnormal_dir, image_filename)
    image_paths.append(image_path)
    labels.append('abnormal')  # 'abnormal' represents abnormal

# Loading myocardial infarction ECG images
for image_filename in os.listdir(infarction_dir):
    image_path = os.path.join(infarction_dir, image_filename)
    image_paths.append(image_path)
    labels.append('myocardial_infarction')

# Split the data into training, validation, and test sets
test_size = 0.1
validation_size = 0.2
X_train, X_temp, y_train, y_temp = train_test_split(image_paths, labels, test_size=test_size, random_state=30)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size / (1 - test_size), random_state=30)

target_size = (224, 224)

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image

X_train_preprocessed = [preprocess_image(image_path, target_size) for image_path in X_train]
X_val_preprocessed = [preprocess_image(image_path, target_size) for image_path in X_val]
X_test_preprocessed = [preprocess_image(image_path, target_size) for image_path in X_test]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

X_train_array = np.array(X_train_preprocessed)
X_val_array = np.array(X_val_preprocessed)
X_test_array = np.array(X_test_preprocessed)

y_train_array = np.array(y_train_encoded)
y_val_array = np.array(y_val_encoded)
y_test_array = np.array(y_test_encoded)

# Define the CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3 classes: normal, abnormal, myocardial infarction
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 10
batch_size = 32

history = model.fit(X_train_array, y_train_array,
                    validation_data=(X_val_array, y_val_array),
                    epochs=epochs, batch_size=batch_size)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_array, y_test_array)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy (in %): {test_accuracy * 100:.2f}%")
