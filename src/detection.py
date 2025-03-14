from ultralytics import YOLO
import cv2
import os
import numpy as np
from ensemble_boxes import soft_nms

# Load your models (ensure the model files exist in the models folder)
model_N = YOLO('models/YOLOv8n.pt')
model_S = YOLO('models/YOLOv8s.pt')
model_M = YOLO('models/YOLOv8m.pt')

def extract_boxes_confidences_labels(results):
    boxes = []
    scores = []
    labels = []
    # Assuming results[0].boxes.data contains detections for each model.
    for r in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = r.tolist()
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        labels.append(int(cls))
    return boxes, scores, labels

def detect_nanoparticles(image):
    # Save the image temporarily
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, image)

    # Run inference with the three models
    results_N = model_N(temp_path, verbose=False)
    results_S = model_S(temp_path, verbose=False)
    results_M = model_M(temp_path, verbose=False)

    boxes_N, scores_N, labels_N = extract_boxes_confidences_labels(results_N)
    boxes_S, scores_S, labels_S = extract_boxes_confidences_labels(results_S)
    boxes_M, scores_M, labels_M = extract_boxes_confidences_labels(results_M)

    # Remove the temporary image file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Get image dimensions
    height, width = image.shape[:2]

    # Normalize boxes to values between 0 and 1
    def normalize_boxes(boxes, width, height):
        normalized = []
        for box in boxes:
            x1, y1, x2, y2 = box
            normalized.append([x1/width, y1/height, x2/width, y2/height])
        return normalized

    boxes_N_norm = normalize_boxes(boxes_N, width, height)
    boxes_S_norm = normalize_boxes(boxes_S, width, height)
    boxes_M_norm = normalize_boxes(boxes_M, width, height)

    # Convert to numpy arrays; if a model returned no detections, use an empty array.
    boxes_list = [
        np.array(boxes_N_norm) if boxes_N_norm else np.empty((0, 4)),
        np.array(boxes_S_norm) if boxes_S_norm else np.empty((0, 4)),
        np.array(boxes_M_norm) if boxes_M_norm else np.empty((0, 4))
    ]
    scores_list = [
        np.array(scores_N) if scores_N else np.empty((0,)),
        np.array(scores_S) if scores_S else np.empty((0,)),
        np.array(scores_M) if scores_M else np.empty((0,))
    ]
    labels_list = [
        np.array(labels_N) if labels_N else np.empty((0,)),
        np.array(labels_S) if labels_S else np.empty((0,)),
        np.array(labels_M) if labels_M else np.empty((0,))
    ]

    # If no boxes detected across models, return an empty list
    if not any(b.size > 0 for b in boxes_list):
        return []

    # Fuse detections using soft NMS (adjust thresholds if necessary)
    boxes_fused, scores_fused, labels_fused = soft_nms(
        boxes_list, scores_list, labels_list, iou_thr=0.3, sigma=0.5, thresh=0.5
    )

    # If soft_nms returns empty lists, return empty detections
    if len(boxes_fused) == 0:
        return []

    # Convert the normalized fused boxes back to pixel coordinates for display
    detections = []
    for i in range(len(boxes_fused)):
        nb = boxes_fused[i]
        x1 = nb[0] * width
        y1 = nb[1] * height
        x2 = nb[2] * width
        y2 = nb[3] * height
        box_pixels = [int(x1), int(y1), int(x2), int(y2)]
        detections.append([labels_fused[i], scores_fused[i], box_pixels])
    return detections