from ultralytics import YOLO
import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from detection import detect_nanoparticles

# Dictionary of class names
class_labels = {
    0: "LCN",        # Class 0
    1: "MCV",        # Class 1
    2: "TMCV",       # Class 2
    3: "V",          # Class 3
    4: "scal_bar"    # Class 4
}

# Color dictionary by class
class_colors = {
    0: (0, 0, 255),    # Class 0
    1: (0, 255, 0),    # Class 1
    2: (243, 243, 0),  # Class 2
    3: (0, 223, 183),  # Class 3
    4: (255, 0, 0)     # Class 4
}

def draw_detection_boxes(image, results):
    """Draw bounding boxes with labels on the image"""
    img_height, img_width = image.shape[:2]
    f = img_height / 1024
    thickness = max(1, int(f * 2))
    font_scale = max(0.5, f * 1)
    
    # Create a copy of the image for drawing
    annotated_img = image.copy()
    
    for result in results:
        label_id, confidence, box = result
        x1, y1, x2, y2 = box
        
        # Get color and class name
        color = class_colors.get(label_id, (255, 255, 255))
        label_name = class_labels.get(label_id, f"Class {label_id}")
        
        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        text = f"{label_name} {confidence:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Draw text background
        cv2.rectangle(annotated_img, 
                     (x1, y1 - text_height - baseline), 
                     (x1 + text_width, y1), 
                     color, -1)
        
        # Add text
        text_color = (255, 255, 255)
        if color == (243, 243, 0):  # Yellow needs dark text
            text_color = (0, 0, 0)
            
        cv2.putText(annotated_img, text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                   text_color, thickness)
    
    return annotated_img

def plot_class_distribution(results):
    """Create histogram of detected classes"""
    if not results:
        return None
    
    # Extract all class labels
    all_labels = [result[0] for result in results]
    class_counts = Counter(all_labels)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [class_labels.get(cls, f"Class {cls}") for cls in class_counts.keys()],
        class_counts.values(), 
        color='blue', 
        width=0.4
    )
    
    ax.set_xlabel("Detected Classes")
    ax.set_ylabel("Number of Detections")
    ax.set_title("Number of Detections per Class")
    plt.xticks(rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig

def calculate_size_distribution(results, image):
    """Calculate size distribution based on scale bar"""
    if not results:
        return None, None
    
    # Find the scale bar (class 4)
    scale_bar = None
    for result in results:
        label, _, box = result
        if label == 4:  # Scale bar class
            scale_bar = box
            break
            
    if not scale_bar:
        return None, None
        
    # Calculate pixels to nm conversion
    scale_bar_length_nm = 200  # Standard 200nm scale bar
    scale_bar_pixels = scale_bar[2] - scale_bar[0]
    pixel_to_nm = scale_bar_length_nm / scale_bar_pixels
    
    # Calculate object sizes by class
    object_sizes_nm = {cls: [] for cls in set(r[0] for r in results) if cls != 4}
    
    for result in results:
        label, _, box = result
        if label != 4:  # Ignore scale bar
            x1, y1, x2, y2 = box
            length_nm = (x2 - x1) * pixel_to_nm
            width_nm = (y2 - y1) * pixel_to_nm
            max_dimension_nm = max(length_nm, width_nm)
            if label in object_sizes_nm:
                object_sizes_nm[label].append(max_dimension_nm)
                
    # Create size distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for cls, values in object_sizes_nm.items():
        if values:
            color = np.array(class_colors.get(cls, (255, 255, 255))) / 255.0
            ax.hist(values, bins=10, alpha=0.6, width=4, 
                   color=color, 
                   label=class_labels.get(cls, f"Class {cls}"))
    
    ax.set_xlabel("Diameter (nm)")
    ax.set_ylabel("Number of Detections")
    ax.set_title("Distribution of Object Sizes per Class (in nm)")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create summary dataframe
    summary_data = []
    for cls, values in object_sizes_nm.items():
        if values:
            mean_size_nm = np.mean(values)
            std_dev_size_nm = np.std(values)
            summary_data.append([
                class_labels.get(cls, f"Class {cls}"), 
                mean_size_nm, 
                std_dev_size_nm
            ])
                
    if summary_data:
        df_summary = pd.DataFrame(
            summary_data, 
            columns=["Class", "Mean Diameter (nm)", "Std Dev (nm)"]
        )
        return fig, df_summary
    
    return fig, None

def main():
    st.title("Nanoparticle Detection App")
    st.write("Upload an image to detect nanoparticles.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Display original image
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

        # Perform detection
        with st.spinner("Detecting nanoparticles..."):
            results = detect_nanoparticles(image)

        # Display results
        if results:
            st.write(f"**Detected {len(results)} objects**")
            
            # Convert results to DataFrame for table display
            results_df = pd.DataFrame(
                [[class_labels.get(r[0], f"Class {r[0]}"), r[1], r[2]] for r in results],
                columns=["Class", "Confidence", "Bounding Box"]
            )
            st.dataframe(results_df)
            
            # Display annotated image
            annotated_img = draw_detection_boxes(image, results)
            st.image(annotated_img, channels="BGR", caption="Detection Results", use_column_width=True)
            
            # Class distribution chart
            st.subheader("Class Distribution")
            class_fig = plot_class_distribution(results)
            if class_fig:
                st.pyplot(class_fig)
            
            # Size distribution analysis
            st.subheader("Size Distribution Analysis")
            size_fig, size_df = calculate_size_distribution(results, image)
            
            if size_fig and size_df is not None:
                st.pyplot(size_fig)
                st.subheader("Size Statistics")
                st.dataframe(size_df)
            elif size_fig:
                st.pyplot(size_fig)
            else:
                st.write("No scale bar detected. Size analysis is not available.")
            
        else:
            st.write("No nanoparticles detected.")

if __name__ == "__main__":
    main()