import os
import argparse
from PIL import Image
from pickle import load
import numpy as np

parser = argparse.ArgumentParser(description='Process an image.')
parser.add_argument('image_path', type=str, help='Path to the image file.')

args = parser.parse_args()

image_path = args.image_path

image_name = os.path.splitext(os.path.basename(image_path))[0]

image = Image.open(image_path).convert('L')

(width, height) = image.size
# Calculate the minimum dimension to ensure a 1:1 aspect ratio
min_dim = min(width, height)
# Calculate the left and top coordinates to center the crop
left = (width - min_dim) // 2
top = (height - min_dim) // 2
# Crop the image to a 1:1 aspect ratio
image = image.crop((left, top, left + min_dim, top + min_dim))
image = image.resize((48, 48))

image.save(f'resized_{image_name}.png')

scaler_pkl = os.path("/dump/standardscaler_pca_normalizers_dump.pkl")

# scale image with standard scaler
label_encoder, pca, scaler = load(open("/dump/labelencoder_standardscaler_pca_normalizers_dump.pkl", "rb"))
image = pca.transform(image)
image = scaler.transform(image)

# machine learning
svc_model = load(open("/final/knn_model_standardscaler_grisearch_pca_dump.pkl", "rb"))
y = svc_model.predict(image)

label = label_encoder.inverse_transform(y)
# label = label_encoder.inverse_transform([y])

print(label)