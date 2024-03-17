import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Define the dataset directory and subdirectories
dataset_dir = 'D:\\MY PROJECT'
normal_dir = os.path.join(dataset_dir, 'D:\\MY PROJECT\\normal')
abnormal_dir = os.path.join(dataset_dir, 'D:\\MY PROJECT\\abnormal')
infarction_dir = os.path.join(dataset_dir, 'D:\\MY PROJECT\\myocardiac 240 X 12 = 2880')

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
test_size = 0.2
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

# Create your Multinomial Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()

# Define the number of folds (k) and create a KFold object
k = 5  # You can adjust the number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Perform cross-validation and obtain accuracy scores
accuracy_scores = cross_val_score(naive_bayes_classifier, X_train_array.reshape(X_train_array.shape[0], -1), y_train_array, cv=kf, scoring='accuracy')

# Print the accuracy scores for each fold
for i, accuracy in enumerate(accuracy_scores):
    print(f'Fold {i + 1} Accuracy: {accuracy * 100:.2f}%')

# Calculate the mean and standard deviation of accuracy scores
mean_accuracy = accuracy_scores.mean()
std_accuracy = accuracy_scores.std()
print(f'Mean Accuracy: {mean_accuracy * 100:.2f}%')
print(f'Standard Deviation: {std_accuracy * 100:.2f}%')

# Perform cross-validation and obtain predictions
predictions = []
for train_index, test_index in kf.split(X_train_array):
    X_train_fold, X_val_fold = X_train_array[train_index], X_train_array[test_index]
    y_train_fold, y_val_fold = y_train_array[train_index], y_train_array[test_index]

    naive_bayes_classifier.fit(X_train_fold.reshape(X_train_fold.shape[0], -1), y_train_fold)
    fold_predictions = naive_bayes_classifier.predict(X_val_fold.reshape(X_val_fold.shape[0], -1))
    predictions.extend(fold_predictions)

# Print the classification report on the validation set
classification_rep = classification_report(y_train_array, predictions, target_names=label_encoder.classes_)
print("Multinomial Naive Bayes Classification Report on Validation Set:\n", classification_rep)

# Train the final classifier on the entire training set
naive_bayes_classifier.fit(X_train_array.reshape(X_train_array.shape[0], -1), y_train_array)

# Evaluate the classifier on the test set
test_predictions = naive_bayes_classifier.predict(X_test_array.reshape(X_test_array.shape[0], -1))

# Print the test accuracy
test_accuracy = naive_bayes_classifier.score(X_test_array.reshape(X_test_array.shape[0], -1), y_test_array)
print(f"Test Accuracy  of Naive bayes (in %): {test_accuracy * 100:.2f}%")

# Generate and print the classification report on the test set
classification_rep_test = classification_report(y_test_array, test_predictions, target_names=label_encoder.classes_)
print("Multinomial Naive Bayes Classification Report on Test Set:\n", classification_rep_test)
