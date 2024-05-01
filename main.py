import cv2
import pytesseract
import os
from pydub import AudioSegment
from pydub.generators import Sine
import simpleaudio as sa
import numpy as np

# Define frequency mappings for pitches
pitch_mappings = {
    1: 440,  # A4
    2: 494,  # B4
    3: 523  # C5
}

# Default values
default_pitch = 2
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
cap.set(cv2.CAP_PROP_FPS, 1)
# use the full resolution of the camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

text_lengths = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    config = '--psm 6'  # Assume a single uniform block of text.
    text = pytesseract.image_to_string(gray, config=config)
    print("Recognized text:", text)

    current_length = len(text.replace(" ", ""))  # Consider non-space characters only
    text_lengths.append(current_length)
    commands = []
    if len(text_lengths) > 1 and current_length > np.mean(text_lengths[:-1]):
        commands = text.upper().split()
        i = 0
        while i < len(commands):
            cmd = commands[i]
            if "NOTE" in cmd:
                pitch = default_pitch
                if i + 1 < len(commands) and "PITCH" in commands[i+1]:
                    try:
                        pitch = int(commands[i+1].split(' ')[-1])
                    except ValueError:
                        pitch = default_pitch
                    i += 1
                play_note(pitch)
            elif "BEGIN" in cmd and "LOOP" in cmd:
                loop_start = i
                repeat_count = default_repeat  # Set default repeat count
                while i < len(commands) and "END" not in commands[i]:
                    i += 1
                    if "REPEAT:" in commands[i] and commands[i].split(":")[1].isnumeric():
                        try:
                            repeat_count = int(commands[i].split(":")[1])
                        except ValueError:
                            repeat_count = default_repeat
                for _ in range(repeat_count):
                    j = loop_start + 1
                    while j < i:
                        if "NOTE" in commands[j]:
                            pitch = default_pitch
                            if j + 1 < len(commands) and "PITCH" in commands[j+1]:
                                try:
                                    pitch = int(commands[j+1].split(' ')[-1])
                                except ValueError:
                                    pitch = default_pitch
                                j += 1
                            play_note(pitch)
                        j += 1
                # omit the begin, end and repeat commands from the list, so we can end the loop
                commands = commands[:loop_start] + commands[i+1:]
                i = loop_start - 1
            i += 1

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended.")