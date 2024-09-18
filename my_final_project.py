#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models



import os
import re

l = []

# List all files in the specified directory
directory ="D:\\final project\\data set\\full-fundus\\full-fundus"
files = os.listdir(directory)

for filename in files:
    # Use regular expression to find uppercase letters in the filenames
    uppercase_letters = re.findall('[A-Z]+', filename)

    if uppercase_letters:
        # Split the found uppercase letters into separate strings
        uppercase_letters = uppercase_letters[0].split()
        l.append(uppercase_letters)

# Display the results
print(l)


# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'l' is a list of values you want to visualize
# For example, let's say 'l' is a list of categories or labels

# Example:

# Create a DataFrame from the list
k = pd.DataFrame(l, columns=['Category'])

# Create a count plot using Seaborn
sns.countplot(x='Category', data=k)

# Show the plot
plt.show()


# In[19]:


data_dir = "D:\\final project\\data set\\full-fundus"
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

data_dir = "D:\\final project\\data set\\full-fundus"
image_size =128
(x_train, x_test, y_train, y_test) = load_and_preprocess_dataset(data_dir, image_size, test_size=0.2)


# In[20]:


original_size = x_train.size  # Total number of elements in the original array
print("Original size:", original_size)


# In[21]:


print(x_train.size)
print(x_train.shape)


# In[22]:


print(x_test.size)
print(x_test.shape)


# In[23]:


x_train=x_train.reshape(6268, 128, 128, 3)
x_test=x_test.reshape(1568, 128, 128, 3)


# In[24]:


ytrain=y_train
ytest=y_test


# In[25]:


def vision_transformer(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
    input_layer = layers.Input(shape=(image_size, image_size, 3))

    # Patching the input image
    x = layers.Conv2D(dim, patch_size, strides=patch_size, padding="valid", activation="relu")(input_layer)

    # Transformer Encoder
    for _ in range(depth):
        # Multi-Head Self-Attention
        attention_output = layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads)(x, x)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)

        # MLP Block
        mlp_output = layers.Conv1D(filters=mlp_dim, kernel_size=1, activation="relu")(attention_output)
        mlp_output = layers.Conv1D(filters=dim, kernel_size=1)(mlp_output)
        x = layers.LayerNormalization(epsilon=1e-6)(mlp_output + attention_output)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully-connected layer with the correct number of units for CIFAR-10 (10 classes)
    output_layer = layers.Dense(4, activation="softmax")(x)

    model = models.Model(inputs=input_layer, outputs=output_layer, name="vision_transformer")
    return model


# In[26]:


num_classes = len(np.unique(y_train))  # Automatically determine the number of classes
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Parameters for Vision Transformer
image_size = 128  # You may need to adjust this based on the actual size of your images
patch_size = 8
dim = 64
depth = 3
heads = 4
mlp_dim = 128

# Create the Vision Transformer model
vit_model = vision_transformer(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim)

# Compile the model
vit_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train the model with EarlyStopping callback
hs = vit_model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))


# In[28]:


vit_model.evaluate(x_test,y_test)


# In[29]:


import numpy as np
# Make predictions
y_pred = vit_model.predict(x_test)

# Define a threshold
threshold = 0.48

# Apply threshold to convert probabilities to binary predictions
y_pred_binary = np.where(y_pred < threshold, 0,1 )


# Calculate F1 score
f1 = f1_score(y_test, y_pred_binary, average='weighted')
print("F1 score:", f1)

# Classification report
print(classification_report(y_test, y_pred_binary))
print("Accuracy:",accuracy_score(y_test,y_pred_binary))


# In[34]:


ypred=vit_model.predict(x_test)


# In[35]:


from sklearn.metrics import confusion_matrix
newpred=np.argmax(ypred,axis=1)


# In[30]:


# Plotting training and validation accuracy
plt.plot(hs.history['accuracy'], label='Training Accuracy')
plt.plot(hs.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




