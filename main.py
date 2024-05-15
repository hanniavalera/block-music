import cv2
import pytesseract
import simpleaudio as sa
import numpy as np
import time
import os
from collections import deque

# Setup initial configurations - make it like do, re, mi, fa, so, la, ti
pitch_mappings = {'ONE': 261.63, 'TWO': 293.66, 'THREE': 329.63, 'FOUR': 349.23, 'FIVE': 392.00, 'SIX': 440.00, 'SEVEN': 493.88}
number_map = {'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5, 'SIX': 6, 'SEVEN': 7}
default_repeat = 2
default_pitch = np.random.choice(['ONE', 'TWO', 'THREE'])
command_keywords = ["START", "NOTE", "BEGIN LOOP", "STOP LOOP", "END", "PITCH", "REPEAT"]
threshold = 3

def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = amplitude * np.sin(frequency * t * 2 * np.pi)
    audio = wave * (2**15 - 1) / np.max(np.abs(wave))
    audio = audio.astype(np.int16)
    return audio

def play_note_simpleaudio(pitch, duration=1):
    frequency = pitch_mappings.get(pitch, pitch_mappings[default_pitch])
    audio = generate_sine_wave(frequency, duration)
    play_obj = sa.play_buffer(audio, 1, 2, 44100)
    play_obj.wait_done()

def minDistance(word1, word2):
    m = len(word1)
    n = len(word2)
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
        print(f"Processing word: {word}")  # Debugging statement
        if ('END' in word or word not in command_keywords) and index >= len(words) * 4 / 5:
            print("End command found or invalid command")  # Debugging statement
            return index, True  # Signal to end the program
        for keyword in command_keywords:
            if minDistance(word, keyword) <= threshold:
                print(f"Keyword matched: {keyword}")  # Debugging statement
                if "BEGIN LOOP" in keyword:
                    repeat = next((w for w in words[index+1:index+3] if w in number_map), default_repeat)
                    repeat_count = number_map.get(repeat, default_repeat)
                    for _ in range(repeat_count):
                        new_index, should_end = process_commands(commands, words, index+1)
                        if should_end:
                            return new_index, should_end
                elif "STOP LOOP" in keyword:
                    return index, False
                elif "NOTE" in keyword:
                    pitch = next((w for w in words[index+1:index+3] if w in pitch_mappings), 'TWO')
                    commands.append((play_note_simpleaudio, pitch))
        index += 1
    return index, False

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

start_time = time.time()
text_frames = []
command_queue = deque()  # Queue to process commands

while time.time() - start_time < 120:  # Run for two minutes
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')
    words = text.upper().split()

    text_frames.append(words)
    print(f"Captured text: {words}")  # Debugging statement

    if len(text_frames) > 10:  # Use last 10 frames to get a stable average
        avg_length = sum(len(f) for f in text_frames[-10:]) / 10
    else:
        avg_length = sum(len(f) for f in text_frames) / len(text_frames) if text_frames else 0

    # Find the frame with length closest to the average
    if len(text_frames) >= 10:
        closest_frame = min(text_frames, key=lambda x: abs(len(x) - avg_length))
        text_frames = text_frames[-10:]  # Keep only the last 10 frames for averaging

        commands = []
        index, should_end = process_commands(commands, closest_frame, 0)
        command_queue.extend(commands)  # Add commands to the queue
        if should_end:
            break

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended.")

# Process commands from the queue
while command_queue:
    command = command_queue.popleft()
    command[0](command[1])

# Add a sleep to pause execution after ending
if should_end:
    time.sleep(10)