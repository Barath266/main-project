from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import time
import random
from flask import Flask, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.callbacks import CSVLogger
from keras.utils import to_categorical

app = Flask(__name__)

def load_and_preprocess_dataset(data_dir, image_size, test_size, random_state=42):
    files = os.listdir(os.path.join(data_dir, 'full-fundus'))

    categories = ['REFUGE1-train', 'EyePACS-Glaucoma', 'OIA-ODIR-TRAIN', 'G1020']
    file_paths = {category: [] for category in categories}

    for category in categories:
        file_paths[category] = [os.path.join(data_dir, 'full-fundus', i) for i in files if category in i]

    if all(len(paths) == 0 for paths in file_paths.values()):
        raise ValueError("Dataset is empty. Adjust your data directory or test_size parameter.")

    img = []
    idx = []

    for category, paths in file_paths.items():
        for path in paths:
            r = cv2.imread(path)
            r = cv2.resize(r, (image_size, image_size))
            img.append(r)
            idx.append(categories.index(category))

    img_array = np.array(img)
    idx_array = np.array(idx)

    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=-1) if img_array.shape[-1] == 3 else img_array

    img_array = img_array / 255.0

    x_train, x_test, y_train, y_test = train_test_split(
        img_array, idx_array, test_size=test_size, random_state=random_state, stratify=idx_array
    )

    # Flatten the training set
    x_train_flat = x_train.reshape(x_train.shape[0], -1)

    return x_train_flat

# Load the PCA model
pca = PCA(n_components=100)  # Assuming you used 100 components during training

# Define x_train_flat globally
x_train_flat = load_and_preprocess_dataset(r"D:\final project\data set\full-fundus", image_size=128, test_size=0.2)

pca.fit(x_train_flat)

# Load the VIT model
vit_model = tf.keras.models.load_model(r"C:\Users\barat\OneDrive\Desktop\vit_model.h5") 
pca_model= tf.keras.models.load_model(r"C:\Users\barat\OneDrive\Desktop\old saved model\pca_with_cnn.h5")
hybrid_model=tf.keras.models.load_model(r"C:\Users\barat\OneDrive\Desktop\old saved model\hybridmodel.h5")

categories = ['REFUGE1-train', 'EyePACS-Glaucoma', 'OIA-ODIR-TRAIN', 'G1020']

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    model_type = request.form['model']  # Get the model type from the request

    # Save the received image to the desktop with a unique filename
    unique_filename = save_image(file)

    # Preprocess the input image
    image = cv2.imread(unique_filename)
    resized_image = cv2.resize(image, (128, 128))
    normalized_image = resized_image / 255.0

    if model_type == 'pcawithcnn':
        image_flat = normalized_image.reshape(1, -1)
        image_pca = pca.transform(image_flat)
        image_reshaped = image_pca.reshape(1, 100, 1)
    else:
        image_reshaped = np.expand_dims(normalized_image, axis=0)

    # Make predictions based on the selected model type
    if model_type == 'vit':
        print("Im inside vit block")
        model = vit_model
    elif model_type == 'pcawithcnn':
        print("Im inside pca with cnn block")
        model = pca_model
    elif model_type == 'hybrid':
        print("Im inside hybrid model block")
        model = hybrid_model
    else:
        return jsonify({'error': 'Invalid model type'})

    # Make predictions
    try:
        predictions = model.predict(image_reshaped)
        predicted_class = np.argmax(predictions)
        predicted_category = categories[predicted_class]
    except Exception as e:
        return jsonify({'error': str(e)})

    # Delete the temporary image file
    os.remove(unique_filename)

    # Return the predicted category in JSON format
    response = {'predicted_category': predicted_category}
    return jsonify(response)


def save_image(file):
    # Generate a unique filename
    timestamp = int(time.time())
    random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', k=8))
    unique_filename = f"temp_image_{timestamp}_{random_string}.png"

    # Specify the desktop path
    desktop_path = r"C:\Users\barat\OneDrive\Desktop"

    # Save the received image to the desktop with the unique filename
    file.save(os.path.join(desktop_path, unique_filename))

    return os.path.join(desktop_path, unique_filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
