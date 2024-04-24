import cv2
import pytesseract
import os

# Check if a custom path to Tesseract is provided via an environment variable
tesseract_path = os.getenv('TESSERACT_PATH')
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 is typically the built-in webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform OCR using Tesseract on the grayscale image
    text = pytesseract.image_to_string(gray, lang='eng')
    print("Detected text:", text)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()