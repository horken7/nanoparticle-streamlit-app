from collections import Counter
import numpy as np
import cv2
import matplotlib.pyplot as plt

def normalize_boxes(boxes, width, height):
    return [[x1 / width, y1 / height, x2 / width, y2 / height] for x1, y1, x2, y2 in boxes]

def denormalize_box(box, width, height):
    x1, y1, x2, y2 = box
    return [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]

def plot_class_distribution(labels_fused, class_labels):
    class_counts = Counter(labels_fused)
    
    plt.figure(figsize=(8, 5))
    plt.bar([class_labels.get(cls, f"Class {cls}") for cls in class_counts.keys()], 
            class_counts.values(), color='blue', width=0.2)
    plt.xlabel("Detected Classes")
    plt.ylabel("Number of Detections")
    plt.title("Number of Detections per Class")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()  # Display the plot directly in the Streamlit app

def read_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
