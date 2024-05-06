import cv2
import pytesseract
from pydub import AudioSegment
from pydub.playback import play
from pydub.generators import Sine
import numpy as np
import re

pitch_mappings = {
    'ONE': 440,  # A4
    'TWO': 494,  # B4
    'THREE': 523  # C5
}

default_pitch = 2
default_repeat = 2

# Convert spelled-out numbers to integer values
number_map = {'ONE': 1, 'TWO': 2, 'THREE': 3}

def play_note(pitch, duration=500):  # Default duration 500ms
    frequency = pitch_mappings.get(pitch, pitch_mappings['TWO'])
    sound = Sine(frequency).to_audio_segment(duration=duration)
    play(sound)

def minDistance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

command_keywords = ["START", "NOTE", "BEGIN LOOP", "STOP LOOP", "END", "PITCH", "REPEAT"]
threshold = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')
    print("Recognized text:", text)

    words = text.upper().split()
    for i, word in enumerate(words):
        for keyword in command_keywords:
            if minDistance(word, keyword) <= threshold:
                if "PITCH" in keyword:
                    # Look for the next word that is a key in number_map
                    pitch = next((w for w in words[i+1:i+3] if w in pitch_mappings), 'TWO')
                    play_note(pitch)
                elif "REPEAT" in keyword:
                    # Similar logic for repeat, assuming it needs to trigger something repeatedly
                    repeat = next((number_map[w] for w in words[i+1:i+3] if w in number_map), default_repeat)
                    # Implement logic using 'repeat' value as needed

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended.")