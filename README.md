# Automated License Plate Recognition (ALPR)

This project implements a classical image processing pipeline combined with Optical Character Recognition (OCR) to automatically identify and extract license plate characters from an image of a car.

Developed as part of the **i2i Academy Applied Image Processing** curriculum.

## Pipeline Overview

The ALPR module processes images in a step-by-step workflow:
1. **Grayscale Conversion & Smoothing:** Reduces noise and prepares the image using a Bilateral Filter.
2. **Edge Detection:** Extracts shape outlines using Canny Edge Detection.
3. **Contour Analysis & Segmentation:** Finds shapes, isolates the rectangular plate region using contour vertices, and crops it.
4. **OCR Character Recognition:** Extracts the text using `easyocr`.

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install opencv-python numpy easyocr torch torchvision
```

## How to Run

1. Place your car image in the project root and name it `car.png`.
2. Run the plate recognition script:

```bash
python plate_recognition.py
```

The script will guide you step-by-step, saving intermediate images (`step1_gray_blur.png`, `step2_edges.png`, `step3_contours.png`, and `step4_cropped_plate.png`) in the root directory.
