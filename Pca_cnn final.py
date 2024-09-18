#!/usr/bin/env python
# coding: utf-8

# In[41]:


#pca
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Function to load and preprocess dataset
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

    return x_train, x_test, y_train, y_test

# Load and preprocess the dataset
data_dir =  "D:\\final project\\data set\\full-fundus"
image_size = 128
(x_train, x_test, y_train, y_test) = load_and_preprocess_dataset(data_dir, image_size, test_size=0.2)

# Reshape the data if necessary
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Perform PCA
pca = PCA(n_components=100)  # Choose the number of components you desire
pca.fit(x_train_flat)

# Transform both training and testing data
x_train_pca = pca.transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)

# Now x_train_pca and x_test_pca contain the PCA-transformed data


# In[42]:


y_train


# In[43]:


import tensorflow as tf

ytrain=tf.keras.utils.to_categorical(y_train, num_classes=4)

ytrain


# In[44]:


ytest=tf.keras.utils.to_categorical(y_test,num_classes=4)


# In[45]:


from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Define the model
model = Sequential()
model.add(Conv1D(128, kernel_size=3, activation='relu', input_shape=(100, 1)))  # Adjust input shape according to PCA components
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))  # Binary classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Fit the model
history = model.fit(x_train_pca[..., np.newaxis], ytrain, epochs=100, batch_size=32, validation_data=(x_test_pca[..., np.newaxis], ytest[..., np.newaxis]))


# In[46]:


loss, accuracy = model.evaluate(x_test_pca[..., np.newaxis], ytest[..., np.newaxis])
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')


# In[48]:


import numpy as np

from sklearn.metrics import f1_score,accuracy_score,classification_report
# Make predictions
y_pred = model.predict(x_test_pca[..., np.newaxis])

ypred = np.argmax(y_pred,axis=1)


# Calculate F1 score
f1 = f1_score(y_test[..., np.newaxis], ypred, average='weighted')
print("F1 score:", f1)

# Classification report
print(classification_report(y_test[..., np.newaxis], ypred))
print("Accuracy: ",accuracy_score(y_test[..., np.newaxis],ypred))


# In[49]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

