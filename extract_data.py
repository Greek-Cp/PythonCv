import cv2
import numpy as np
import pandas as pd

def extract_features(image_path):
    # Load the target image
    image = cv2.imread(image_path)

    # Calculate the average value of each channel (R, G, B)
    r, g, b = cv2.split(image)
    avg_r = np.mean(r)
    avg_g = np.mean(g)
    avg_b = np.mean(b)

    return [avg_r, avg_g, avg_b]


image_path = "img/cabe.png"
features = extract_features(image_path)

# Reshape features into a matrix
feature_matrix = np.reshape(features, (1, len(features)))

# Create a DataFrame for the feature matrix
data = pd.DataFrame(feature_matrix, columns=["Average R", "Average G", "Average B"])

# Export data to Excel
data.to_excel("res.xlsx", index=False)
