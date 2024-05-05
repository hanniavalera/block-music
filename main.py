import cv2
import pytesseract
import os
from pydub import AudioSegment
from pydub.generators import Sine
import simpleaudio as sa
import numpy as np
from collections import deque

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

# Edit distance implementation for command recognition
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        memo = {}
        
        def dfs(i, j):
            if i == 0 or j == 0: return j or i
                        
            if (i, j) in memo:
                return memo[(i, j)]
            
            if word1[i-1] == word2[j-1]:
                ans = dfs(i-1, j-1)
            else: 
                ans = 1 + min(dfs(i, j-1), dfs(i-1, j), dfs(i-1, j-1))
                
            memo[(i, j)] = ans
            return memo[(i, j)]
        
        return dfs(len(word1), len(word2))

def parse_command_argument(command):
    parts = command.split()
    for part in parts:
        if part.isdigit():  # Check if part is a digit
            return int(part)
    # Fall back to default if no digit is found
    return None

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

text_lengths = []
edit_distance_solver = Solution()
command_keywords = ["START", "NOTE", "BEGIN LOOP", "END LOOP", "END", "PITCH", "REPEAT"]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    config = '--psm 6'
    text = pytesseract.image_to_string(gray, config=config)
    print("Recognized text:", text)

    current_length = len(text.replace(" ", ""))
    text_lengths.append(current_length)
    if len(text_lengths) > 1 and current_length > np.mean(text_lengths[:-1]):
        words = text.upper().split()
        for word in words:
            for keyword in command_keywords:
                if edit_distance_solver.minDistance(word, keyword) <= 3:  # Increased threshold
                    command_argument = parse_command_argument(text)
                    if "PITCH" in keyword and command_argument:
                        play_note(command_argument)
                    elif "REPEAT" in keyword and command_argument:
                        # Implement repeat logic
                        pass

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended.")