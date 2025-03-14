# Nanoparticle Detection Streamlit Application

This project is a Streamlit application designed for detecting and analyzing nanoparticles in TEM (Transmission Electron Microscopy) images using YOLO (You Only Look Once) models. Based on research published at [Zenodo (Record 14995364)](https://zenodo.org/records/14995364), this application identifies different nanostructures and provides morphological analysis with accurate size measurements.

## Research Background

This project applies YOLOv8 for the detection and morphological analysis of polymer nanoparticles observed via Transmission Electron Microscopy (TEM). The goal is to classify different nanostructures and measure their dimensions based on a scale bar for accurate size conversion.

### Detected Nanoparticle Types

Based on the study "Multicompartment Vesicles: A Key Intermediate Structure in Polymerization-Induced Self-Assembly of Graft Copolymers," we classify and detect the following morphologies:

- **LCN** (Large Compound Nano-object): Formed by fusion of Multicompartment Vesicles (MCV), appearing as spherical or cylindrical objects.
- **MCV** (Multicompartment Vesicle): Intermediate structures containing multiple hydrophilic cores within a membrane, attributed to phase separation of dextran and residual monomer.
- **TMCV** (Thick Membrane Multicompartment Vesicle): A transitional form of MCVs with a thicker membrane before merging into LCNs.
- **V** (Unilamellar Vesicle, ULV): Formed at a lower polymerization degree (X = 100), these single-layer vesicles are thinner but grow progressively.
- **Scale Bar**: A detected reference scale to convert pixel measurements into nanometers, set at 200 nm.

## Technical Approach

The application utilizes three different YOLOv8 models (YOLOv8n, YOLOv8s, and YOLOv8m) to perform multi-scale detection of nanoparticles. The models process the input TEM images separately, and their predictions are merged using a weighted box fusion (WBF) technique. This improves detection accuracy and reduces false positives.

After detection, the bounding boxes are converted from pixels to nanometers based on the scale bar, allowing precise size measurements and statistical analysis of nanoparticle distribution.

### Training Configuration

The models were trained using the following hardware and software setup:
- CPU: Intel Core(TM) i7-1068NG7 2.30GHz
- OS: Ubuntu 20.04
- Python Version: 3.12.9
- Torch Version: 2.2.2 (CPU)
- Framework: PyTorch 2.0 + Ultralytics YOLOv8

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