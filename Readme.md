# Brain Tumor Segmentation using YOLOv11 and SimpleITK

A comprehensive solution for automated brain tumor segmentation in MRI images using YOLOv11 for initial detection and SimpleITK for 3D post-processing refinement.

## Features

- YOLOv11-based instance segmentation for tumor detection
- Advanced 3D post-processing pipeline using SimpleITK
- Level set-based contour optimization
- Integration with 3D Slicer for visualization
- Support for NRRD format medical images

## Requirements

```
roboflow
ultralytics
numpy
opencv-python
torch
SimpleITK
pynrrd
tqdm
matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model using `brain_tumor_pj.ipynb`

1. Prepare your dataset using Roboflow:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
version = project.version(1)
dataset = version.download("yolov11")
```

2. Train the YOLOv11 model:
```python
from ultralytics import YOLO

# Load the base model
model = YOLO("yolo11n-seg.pt")

# Train the model
results = model.train(
    data="/path/to/data.yaml",
    epochs=100,
    imgsz=640
)
```

### Inference and Post-processing using `inference.py`

1. For basic inference on a single image:
```python
model = YOLO('path/to/best.pt')
results = model("path/to/image.jpg", save=True)
```

2. For full 3D volume processing with post-processing:
```python
from process_3d import process_3d_with_multiple_outputs

# Process a 3D volume and get three different outputs
yolo_output, morph_output, refined_output = process_3d_with_multiple_outputs(
    "input.nrrd",
    "output_yolo.nrrd",
    "output_morph.nrrd",
    "output_refined.nrrd"
)
```

## Pipeline Overview

The segmentation pipeline consists of three main stages:

1. **Initial Segmentation (YOLOv11)**
   - Processes 2D slices from the 3D volume
   - Provides initial tumor detection and segmentation

2. **Morphological Refinement**
   - Applies 3D morphological operations
   - Smooths boundaries and removes artifacts
   - Implemented using SimpleITK

3. **Level Set Refinement**
   - Uses geodesic active contours
   - Optimizes tumor boundaries
   - Ensures spatial consistency

## Visualization

Results can be visualized using 3D Slicer:

1. Open 3D Slicer
2. Load the original NRRD file
3. Load the segmentation output file
4. Use the segmentation overlay tools for visualization


## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv11 implementation
- [SimpleITK](https://simpleitk.org/) for medical image processing tools
- [3D Slicer](https://www.slicer.org/) for visualization support

## Contact

- Yushan Xie - yushan.xie@vanderbilt.edu
- Yiming Pan - yiming.pan@vanderbilt.edu
- Yifei Wu - yifei.wu@vanderbilt.edu