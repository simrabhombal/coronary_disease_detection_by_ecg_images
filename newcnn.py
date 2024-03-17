import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam 

np.random.seed(42)

# dataset directory and subdirectories
dataset_dir = 'D:\\MY PROJECT'
normal_dir = os.path.join(dataset_dir, 'normal')
abnormal_dir = os.path.join(dataset_dir, 'abnormal')
infarction_dir = os.path.join(dataset_dir, 'myocardiac 240 X 12 = 2880')


# Initialize empty lists for image paths and labels
image_paths = []
labels = []

# Load normal ECG images
for image_filename in os.listdir(normal_dir):
    image_path = os.path.join(normal_dir, image_filename)
    image_paths.append(image_path)
    labels.append('normal')  # normal

# Load abnormal ECG images
for image_filename in os.listdir(abnormal_dir):
    image_path = os.path.join(abnormal_dir, image_filename)
    image_paths.append(image_path)
    labels.append('abnormal')  # abnormal

# Load myocardial infarction ECG images
for image_filename in os.listdir(infarction_dir):
    image_path = os.path.join(infarction_dir, image_filename)
    image_paths.append(image_path)
    labels.append('myocardial_infarction')  # myocardial infarction

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(image_paths, labels, test_size=0.15, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Define the target size for image resizing
target_size = (224, 224)

# Preprocess and normalize images
def preprocess_image(image_path, target_size):
    # Read the image
    image = cv2.imread(image_path)
    
    # Resize the entire image to the target size
    image = cv2.resize(image, target_size)
    
    image = image / 255.0 # Normalize pixel values to the range [0, 1]

    return image

# Modify the preprocessing for training, validation, and test sets
X_train_preprocessed = [preprocess_image(image_path, target_size) for image_path in X_train]
X_val_preprocessed = [preprocess_image(image_path, target_size) for image_path in X_val]
X_test_preprocessed = [preprocess_image(image_path, target_size) for image_path in X_test]

# Convert labels to numeric values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# Convert preprocessed images and labels to numpy arrays
X_train_array = np.array(X_train_preprocessed)
X_val_array = np.array(X_val_preprocessed)
X_test_array = np.array(X_test_preprocessed)

y_train_array = np.array(y_train_encoded)
y_val_array = np.array(y_val_encoded)
y_test_array = np.array(y_test_encoded)

# CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')
])

# Compile CNN model
cnn_model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Training CNN model
cnn_model.fit(X_train_array, y_train_array,
              validation_data=(X_val_array, y_val_array),
              epochs=16, batch_size=32)

# Predictions on the validation set using CNN model
y_val_pred_cnn = np.argmax(cnn_model.predict(X_val_array), axis=1)

# Combine training data and labels for the Balanced Random Forest Classifier
X_train_combined = np.array(X_train_preprocessed)
y_train_combined = np.array(y_train_encoded)

# Flatten each image in X_train_combined
X_train_combined_flattened = X_train_combined.reshape(X_train_combined.shape[0], -1)

# Shuffle the data to ensure randomness
X_train_combined_flattened, y_train_combined = shuffle(X_train_combined_flattened, y_train_combined, random_state=42)

# Define and train the Balanced Random Forest Classifier
brf_classifier = RandomForestClassifier(class_weight='balanced', random_state=42)
brf_classifier.fit(X_train_combined_flattened, y_train_combined)

# Predictions on the validation set using Balanced Random Forest Classifier
y_val_pred_brf = brf_classifier.predict(X_val_array.reshape(X_val_array.shape[0], -1))

# Evaluate the CNN model
print("CNN Model Metrics:")
print("Validation Accuracy:", accuracy_score(y_val_array, y_val_pred_cnn))
print("CNN Model Classification Report:\n", classification_report(y_val_array, y_val_pred_cnn))

# Evaluate the Balanced Random Forest Classifier
print("\nBalanced Random Forest Classifier Metrics:")
print("Validation Accuracy:", accuracy_score(y_val_array, y_val_pred_brf))
print("Balanced Random Forest Classifier Classification Report:\n", classification_report(y_val_array, y_val_pred_brf))


new_image_path = r"D:\demo\test\normal\Normal(284).jpg"
new_image = preprocess_image(new_image_path, target_size)
new_image = np.reshape(new_image, (1, 224, 224, 3))
predictions = cnn_model.predict(new_image)
predicted_class_index = np.argmax(predictions)
predicted_class_label = label_encoder.classes_[predicted_class_index]
print("Predicted Class Label using CNN model:", predicted_class_label)
