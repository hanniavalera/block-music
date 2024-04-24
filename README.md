# Real-Time Text Recognition from Webcam Feed

This Python script captures video from a webcam and uses Tesseract OCR to perform real-time text recognition on the video feed. It's ideal for educational purposes, interactive installations, or any application that requires instant text detection from a live camera feed.

## Prerequisites

- Python 3.x
- OpenCV (`cv2` library)
- Pytesseract
- Tesseract OCR

## Installation

1. **Install Python 3.x**

2. **Install the required Python libraries**

3. **Install Tesseract OCR**


## How It Works
1. The script initializes the webcam and captures frames continuously.
2. Each frame is converted to grayscale to simplify the image processing.
3. Tesseract OCR processes the grayscale images to detect and decode text.
4. The video frame along with any detected text is displayed in real-time.
5. The script runs in a loop until the 'q' key is pressed.

