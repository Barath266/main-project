#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import tensorflow as tf
l=[]
# List all files in the specified directory
files = os.listdir(r"C:\softwares\data set\full-fundus\full-fundus")
for i in files:
    uppercase_letters = re.findall('[A-Z]+',i)
    uppercase_letters=uppercase_letters[0].split()
    
    l.append(uppercase_letters)
# Use regular expression to find uppercase letters in the filenames


# Display the result


# In[2]:


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


# In[3]:


image_size=256
import cv2
import  numpy as np
import os
from sklearn.model_selection import train_test_split
data_dir =  "D:\\final project\\data set\\full-fundus"
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
(x_train, x_test, y_train, y_test) = load_and_preprocess_dataset(data_dir, image_size, test_size=0.2)


# In[4]:


x_train=x_train.reshape(6268, 256,256, 3)
x_test=x_test.reshape(1568,256,256, 3)


# In[5]:


import numpy as np
from tensorflow.keras.utils import to_categorical

set(y_test)


# In[6]:


y_test


# In[7]:


ytrain=to_categorical(y_train, num_classes=4)
ytest=to_categorical(y_test, num_classes=4)


# In[8]:


ytest


# In[9]:


ytrain.shape


# In[10]:


from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
from keras.callbacks import CSVLogger

# Load the InceptionV3 model with pre-trained weights
ins = InceptionV3(weights=None, include_top=False, input_shape=(256, 256, 3))

# Access the desired layer by name
desired_layer = ins.layers[13]
input_layer = ins.input



# First CNN branch
branch1 = Conv2D(64, (3, 3), activation='relu')(input_layer)
branch1 = MaxPooling2D((2, 2))(branch1)
branch1 = Conv2D(128, (3, 3), activation='relu')(branch1)
branch1 = MaxPooling2D((2, 2))(branch1)

# Second CNN branch
branch2 = Conv2D(64, (3, 3), activation='relu')(input_layer)
branch2 = MaxPooling2D((2, 2))(branch2)
branch2 = Conv2D(128, (3, 3), activation='relu')(branch2)
branch2 = MaxPooling2D((2, 2))(branch2)

# Concatenate the outputs of both branches
merged1 = Concatenate()([branch1, branch2])

merged = Concatenate()([desired_layer.output, merged1])
flattened_output = Flatten()(merged)
dense_layer1 = Dense(128, activation='relu')(flattened_output)
dense_layer2 = Dense(64, activation='relu')(dense_layer1)
final_output = Dense(4, activation='softmax')(dense_layer2)

# Create a new model to extract features from the desired layer
feature_model = Model(inputs=ins.input, outputs=final_output)
feature_model.summary()


# In[11]:


feature_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[12]:


Shistory=feature_model.fit(x_train,ytrain,validation_split=0.2,epochs=100)


# In[20]:


feature_model.save('hybrid_model.keras')


# In[14]:


feature_model.evaluate(x_test,ytest)


# In[15]:


ypred=feature_model.predict(x_test)


# In[16]:


from sklearn.metrics import confusion_matrix
newpred=np.argmax(ypred,axis=1)


# In[17]:


con_mat=confusion_matrix(y_test,newpred)


# In[18]:


con_mat


# In[19]:


import seaborn as sns

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(con_mat, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()

