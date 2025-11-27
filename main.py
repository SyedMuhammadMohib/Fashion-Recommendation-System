import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
        .block-container::before, .block-container::after {
            display: none !important;
        }
        div[data-testid="column"] {
            background: transparent !important;
            border: none !important;
        }
        .st-emotion-cache-16idsys, .st-emotion-cache-1wivap2 {
            background: transparent !important;
            border: none !important;
        }
        hr { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# ---------------------- MODEL & DATA LOADING ----------------------
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# ---------------------- FUNCTIONS ----------------------
def save_uploaded_image(uploaded_img):
    img_path = os.path.join("uploads", uploaded_img.name)

    with open(img_path, "wb") as f:
        f.write(uploaded_img.getbuffer())

    return img_path


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    expanded_img_arr = np.expand_dims(img_arr, axis=0)
    preprocessed_img = preprocess_input(expanded_img_arr)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)


def recommend(features, feature_list, n_neighbors=5):
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# ---------------------- UI ----------------------
st.title("👗 Fashion Recommender System")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_path = save_uploaded_image(uploaded_file)

    if image_path:
        st.subheader("🔺 Your Uploaded Image")
        st.image(Image.open(image_path), width=250)

        features = extract_features(image_path, model)
        indices = recommend(features, feature_list)

        st.subheader("🛍️ Recommended Fashion Items")

        # --------- FIXED: REPLACED use_column_width WITH use_container_width ----------
        cols = st.columns(5)
        for idx, col in zip(indices[0], cols):
            col.image(Image.open(filenames[idx]), use_container_width=True)
