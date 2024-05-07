import cv2
import pytesseract
from pydub import AudioSegment
from pydub.playback import play
from pydub.generators import Sine
import numpy as np
import time

# Setup initial configurations - make it like do, re, mi, fa, so, la, ti
pitch_mappings = {'ONE': 261.63, 'TWO': 293.66, 'THREE': 329.63, 'FOUR': 349.23, 'FIVE': 392.00, 'SIX': 440.00, 'SEVEN': 493.88}
number_map = {'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5, 'SIX': 6, 'SEVEN': 7}
default_repeat = 2
# randomize pitch for default from one to three
default_pitch = np.random.choice(['ONE', 'TWO', 'THREE'])
command_keywords = ["START", "NOTE", "BEGIN LOOP", "STOP LOOP", "END", "PITCH", "REPEAT"]
threshold = 3

# add a vague tesseract path to the pytesseract
pytesseract.pytesseract.tesseract_cmd = '/Users/sojioduneye/miniconda3/envs/block-music/bin/tesseract'
def play_note(pitch, duration=1000):
    frequency = pitch_mappings.get(pitch, pitch_mappings[default_pitch])
    sound = Sine(frequency).to_audio_segment(duration=duration)
    play(sound)

def minDistance(word1, word2):
    m = len(word1)
    n = len(word2)
    # dp[i][j] := min # Of operations to convert word1[0..i) to word2[0..j)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
      dp[i][0] = i

    for j in range(1, n + 1):
      dp[0][j] = j

    for i in range(1, m + 1):
      for j in range(1, n + 1):
        if word1[i - 1] == word2[j - 1]:
          dp[i][j] = dp[i - 1][j - 1]
        else:
          dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

    return dp[m][n]

def process_commands(commands, words, index):
    while index < len(words):
        word = words[index]
        if 'END' in word and index >= len(words) * 4 / 5 or word not in command_keywords and index >= len(words) * 4 / 5:
            return index, True  # Signal to end the program
        for keyword in command_keywords:
            if minDistance(word, keyword) <= threshold:
                if "BEGIN LOOP" in keyword:
                    # you will look for the repeat if any and if not default to 2
                    repeat = next((w for w in words[index+1:index+3] if w in number_map), default_repeat)
                    for _ in range(number_map.get(repeat, default_repeat)):
                        process_commands(commands, words, index+1)
                elif "STOP LOOP" in keyword:
                    return index, False
                elif "NOTE" in keyword:
                    pitch = next((w for w in words[index+1:index+3] if w in pitch_mappings), 'TWO')
                    commands.append((play_note, pitch))
        index += 1
    return index, False

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 10)  # Limiting to 10 frames per second
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

start_time = time.time()
text_lengths = []

while time.time() - start_time < 120:  # Run for a minute
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')
    words = text.upper().split()

    current_length = len(words)

    if len(text_lengths) > 10:  # Use last 10 frames to get a stable average
        avg_length = sum(text_lengths[-10:]) / 10
    else:
        avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0

    if current_length > avg_length:
        commands = []
        index, should_end = process_commands(commands, words, 0)
        for command in commands:  # Execute commands
            command[0](command[1])
        if should_end:
            break

    text_lengths.append(current_length)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended.")