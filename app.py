import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st 

st.header('ALPHA PRODUCT FINDER')

# Load the pre-computed features and filenames
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# Function to extract item names from filenames
def extract_item_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]  # Assuming filename without extension is the item name

def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result/norm(result)
    return norm_result

# Initialize the pre-trained model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Fit the Nearest Neighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# File upload
upload_file = st.file_uploader("Upload Image")

if upload_file is not None:
    # Ensure the 'upload' directory exists
    upload_dir = 'upload'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    # Save the uploaded file to the 'upload' directory
    file_path = os.path.join(upload_dir, upload_file.name)
    with open(file_path, 'wb') as f:
        f.write(upload_file.getbuffer())
    
    # Display the uploaded image
    st.subheader('Uploaded Image')
    st.image(upload_file)

    # Extract features and get recommendations
    input_img_features = extract_features_from_images(file_path, model)
    distance, indices = neighbors.kneighbors([input_img_features])

    # Display recommended images with item names
    st.subheader('Recommended Items')
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.image(filenames[indices[0][1]])
        st.caption(extract_item_name(filenames[indices[0][1]]))
    with col2:
        st.image(filenames[indices[0][2]])
        st.caption(extract_item_name(filenames[indices[0][2]]))
    with col3:
        st.image(filenames[indices[0][3]])
        st.caption(extract_item_name(filenames[indices[0][3]]))
    with col4:
        st.image(filenames[indices[0][4]])
        st.caption(extract_item_name(filenames[indices[0][4]]))
    with col5:
        st.image(filenames[indices[0][5]])
        st.caption(extract_item_name(filenames[indices[0][5]]))
