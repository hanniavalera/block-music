import cv2
import pytesseract
import os
from pydub import AudioSegment
from pydub.generators import Sine
import simpleaudio as sa

# Define frequency mappings for pitches
pitch_mappings = {
    1: 440,  # A4
    2: 494,  # B4
    3: 523  # C5
}

# Default values
default_pitch = 1
default_repeat = 2

# Function to play a single note with a specific pitch
def play_note(pitch, duration=500):  # Default duration 500ms
    frequency = pitch_mappings.get(pitch, pitch_mappings[default_pitch])
    tone = Sine(frequency)
    sound = tone.to_audio_segment(duration=duration)
    play_obj = sa.play_buffer(sound.raw_data, 1, 2, 44100)
    play_obj.wait_done()

# Initialize the camera
cap = cv2.VideoCapture(0)
# set to a reduced frame rate as nothing will essentially be moving in the frame
cap.set(cv2.CAP_PROP_FPS, 10)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform OCR using Tesseract on the grayscale image
    config = '--psm 6'  # Assume a single uniform block of text.
    text = pytesseract.image_to_string(gray, config=config)
    print("Recognized text:", text)

    # Normalize and split the text
    commands = text.upper().split()
    i = 0
    while i < len(commands):
        cmd = commands[i]
        if "NOTE" in cmd:
            # Check for subsequent pitch specification
            pitch = default_pitch
            if i + 1 < len(commands) and "PITCH" in commands[i+1]:
                pitch = int(commands[i+1][-1])
                i += 1
            play_note(pitch)
        elif "BEGIN" in cmd and "LOOP" in cmd:
            loop_start = i
        elif "END" in cmd and "LOOP" in cmd:
            loop_end = i
            repeat_count = default_repeat
            # Look for repeat specification
            if i + 1 < len(commands) and "REPEAT:" in commands[i+1]:
                repeat_count = int(commands[i+1].split(":")[1])
                i += 1
            # Execute the loop
            for _ in range(repeat_count):
                j = loop_start + 1
                while j < loop_end:
                    if "NOTE" in commands[j]:
                        pitch = default_pitch
                        if j + 1 < len(commands) and "PITCH" in commands[j+1]:
                            pitch = int(commands[j+1][-1])
                            j += 1
                        play_note(pitch)
                    j += 1
            i = loop_end
        i += 1

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()