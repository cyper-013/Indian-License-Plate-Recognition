# Indian License Plate Recognition (LPR)

A real-time license plate recognition system for Indian vehicles. It detects and tracks vehicles, locates their license plates, reads the text with OCR, validates it against the Indian plate format, and logs each result to a CSV file.

## Features

- Vehicle detection (car, motorcycle, bus, truck) with YOLOv8
- Multi-object tracking with SORT, so each plate is tied to a specific vehicle
- License plate detection with a dedicated YOLO model
- Text recognition with EasyOCR, after grayscale, Gaussian blur and Otsu thresholding
- Indian plate validation and formatting (`SS DD SS NNNN`, e.g. `MH 12 AB 1234`)
- Best reading kept per tracked vehicle
- Live display with vehicle box, plate box, plate text and FPS counter
- Incremental CSV logging with timestamp, plate text, confidence, bounding boxes and frame number

## How it works

1. **Detect vehicles** in each frame with YOLOv8n (COCO classes 2, 3, 5, 7).
2. **Track vehicles** across frames with SORT to get stable IDs.
3. **Detect plates** with a license plate detection model.
4. **Match each plate to a vehicle** by checking which tracked vehicle box contains it.
5. **Preprocess and read** the plate crop with EasyOCR.
6. **Format and validate** the text as an Indian plate, then store and log the best result per vehicle.

## Requirements

- Python 3.10 recommended
- A webcam (the default input is camera index 0)
- Optional: an NVIDIA GPU with a CUDA build of PyTorch for faster processing

Python packages:

```
torch
torchvision
ultralytics
easyocr
opencv-python
numpy
filterpy
scikit-image
lap
```

## Installation

```bash
# 1. Create and activate an environment
conda create -n lpr python=3.10
conda activate lpr

# 2. Install PyTorch (see https://pytorch.org for the command matching your CUDA version)
pip install torch torchvision

# 3. Install the other dependencies
pip install ultralytics easyocr opencv-python numpy filterpy scikit-image lap

# 4. Clone SORT next to the script
git clone https://github.com/abewley/sort.git
```

### Model files

Place these in the project folder:

- `yolov8n.pt`: the official Ultralytics YOLOv8n COCO model (AGPL-3.0). Ultralytics downloads it automatically on first run.
- `license_plate_detector.pt`: a single-class (`license_plate`) YOLOv8n plate detector, obtained from
  [Automatic-License-Plate-Recognition-using-YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8)
  by Muhammad Zeerak Khan. Please download it from there and see that repository for its terms.

## Usage

```bash
python main.py
```

Press `q` to quit. Detected plates are printed to the console and appended to `indian_plates_log.csv`.

### Output format

| Column | Description |
|---|---|
| Timestamp | Time the plate was logged |
| License Plate | Formatted plate text |
| Confidence | Plate detector confidence score |
| Vehicle BBox | `[x1, y1, x2, y2]` of the vehicle |
| Plate BBox | `[x1, y1, x2, y2]` of the plate |
| Frame Number | Frame index at the time of logging |

## Troubleshooting

- **`No module named 'pkg_resources'`**: run `pip install "setuptools<81"`.
- **`Weights only load failed` (PyTorch 2.6+)**: the plate detector was saved with an older Ultralytics release. Upgrade Ultralytics, or, for files you trust, load with `weights_only=False`.
- **`module 'PIL.Image' has no attribute 'ANTIALIAS'`**: run `pip install "pillow<10"` or upgrade EasyOCR.
- **`Neither CUDA nor MPS are available`**: you have a CPU-only PyTorch build. Install the CUDA build from [pytorch.org](https://pytorch.org).

## Known limitations

- Validation accepts only 10-character plates in the `SS DD SS NNNN` pattern, so 9-character, BH-series and older formats are rejected.
- The plate detector was not trained or evaluated by this project, so accuracy on Indian plates is not guaranteed.
- Recognition accuracy depends on lighting, camera angle, plate condition and resolution.
- Input is currently a live webcam feed.

## Credits

This project is built on top of
[automatic-number-plate-recognition-python-yolov8](https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8)
by Computer Vision Engineer, which provided the base pipeline (vehicle detection, SORT tracking, plate detection and OCR). Many thanks for the starting point.

This version extends it with a class-based structure, real-time webcam processing, Indian plate validation and formatting, per-vehicle best-result selection and incremental CSV logging.

The license plate detection model comes from
[Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8).

## Dependencies and acknowledgements

- [SORT](https://github.com/abewley/sort) by Alex Bewley, for multi-object tracking
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)
