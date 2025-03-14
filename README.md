# Nanoparticle Detection Streamlit Application

This project is a Streamlit application designed for detecting nanoparticles in images using YOLO (You Only Look Once) models. The application allows users to upload images and receive real-time detection results, including bounding boxes around detected nanoparticles.

## Project Structure

```
nanoparticle-streamlit-app
├── src
│   ├── app.py          # Main entry point of the Streamlit application
│   ├── detection.py    # Logic for loading models and performing detections
│   └── utils.py        # Utility functions for image processing and plotting
├── models
│   ├── YOLOv8n.pt      # Trained weights for the YOLOv8n model
│   ├── YOLOv8s.pt      # Trained weights for the YOLOv8s model
│   └── YOLOv8m.pt      # Trained weights for the YOLOv8m model
├── requirements.txt     # List of dependencies for the project
├── .gitignore           # Files and directories to ignore by Git
└── README.md            # Documentation for the project
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd nanoparticle-streamlit-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501` to access the application.

3. Upload an image containing nanoparticles. The application will process the image and display the detection results, including bounding boxes around detected nanoparticles.

## Functionality

- **Image Upload**: Users can upload images for analysis.
- **Detection**: The application uses YOLO models to detect nanoparticles in the uploaded images.
- **Results Display**: Detected nanoparticles are highlighted with bounding boxes, and additional statistics are provided.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.