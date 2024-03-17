import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

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

# Set up early stopping using the warm_start parameter
early_stopping_rounds = 10  # Adjust the number of rounds as needed
best_validation_score = float('-inf')  # Initialize to negative infinity

# Create an instance of the Random Forest classifier with early stopping
random_forest_classifier = RandomForestClassifier(n_estimators=1, warm_start=True)  # Start with 1 estimator

for _ in range(100):  # You can adjust the maximum number of iterations
    # Train the Random Forest classifier with one more estimator
    random_forest_classifier.n_estimators += 1
    random_forest_classifier.fit(X_train_array.reshape(X_train_array.shape[0], -1), y_train_array)
    
    # Evaluate the model on the validation set
    validation_score = random_forest_classifier.score(X_val_array.reshape(X_val_array.shape[0], -1), y_val_array)
    
    # Check if validation performance is improving
    if validation_score > best_validation_score:
        best_validation_score = validation_score
        best_estimator_count = random_forest_classifier.n_estimators
    else:
        # Validation performance starts to degrade; stop training
        print(f"Validation performance degrading. Best estimator count: {best_estimator_count}")
        break

# Evaluate the Random Forest classifier on the test set
test_accuracy = random_forest_classifier.score(X_test_array.reshape(X_test_array.shape[0], -1), y_test_array)
print(f"Test Accuracy (in %): {test_accuracy * 100:.2f}%")
