import pytesseract
import simpleaudio as sa
import numpy as np
import time
import os
from collections import deque
import openai

# Setup initial configurations - make it like do, re, mi, fa, so, la, ti
pitch_mappings = {'ONE': 261.63, 'TWO': 293.66, 'THREE': 329.63, 'FOUR': 349.23, 'FIVE': 392.00, 'SIX': 440.00, 'SEVEN': 493.88}
number_map = {'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5, 'SIX': 6, 'SEVEN': 7}
default_repeat = 2
default_pitch = np.random.choice(['ONE', 'TWO', 'THREE'])
command_keywords = ["START", "NOTE", "BEGIN LOOP", "STOP LOOP", "END", "PITCH", "REPEAT"]
threshold = 3

def recognize_text_from_image(file_path):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    with open(file_path, 'rb') as image_file:
        image_data = image_file.read()
    response = openai.Image.create(file={"name": "tmp.img", "content": image_data}, purpose="feature-extraction")
    return response['choices'][0]['text']

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
        if ('END' in word or word not in command_keywords) and index >= len(words) * 4 / 5:
            return index, True  # Signal to end the program
        for keyword in command_keywords:
            if minDistance(word, keyword) <= threshold:
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

file_path = 'tmp.img'
text = recognize_text_from_image(file_path)
words = text.upper().split()

commands = []
index, should_end = process_commands(commands, words, 0)
command_queue = deque(commands)  # Add commands to the queue

# Process commands from the queue
while command_queue:
    command = command_queue.popleft()
    command[0](command[1])

# Add a sleep to pause execution after ending
if should_end:
    time.sleep(10)