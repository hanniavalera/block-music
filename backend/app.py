# backend/app.py
import base64
import numpy as np
import time
import os
import wave
from collections import deque
import openai
from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
import uuid
import requests

# Setup initial configurations - make it like do, re, mi, fa, so, la, ti
pitch_mappings = {'ONE': 261.63, 'TWO': 293.66, 'THREE': 329.63, 'FOUR': 349.23, 'FIVE': 392.00, 'SIX': 440.00, 'SEVEN': 493.88}
number_map = {'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5, 'SIX': 6, 'SEVEN': 7}
default_repeat = 2
default_pitch = np.random.choice(['ONE', 'TWO', 'THREE'])
command_keywords = ["START", "NOTE", "BEGIN LOOP", "STOP LOOP", "END", "PITCH", "REPEAT"]
threshold = 3

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
MUSIC_FOLDER = 'music'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MUSIC_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            # Process the image to generate music
            music_filename = process_image_to_music(file_path)
            return jsonify({'musicUrl': music_filename}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

def recognize_text_from_image(file_path):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # Encode the image to base64
    with open(file_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Create the payload for OpenAI Vision API
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "I have an image of a music program written in a custom programming language for kids. "
                            "The language has the following elements: START, END, BEGIN LOOP, STOP LOOP, NOTE, PITCH, and REPEAT. "
                            "Here are the rules:\n"
                            "- The program starts with START and ends with END.\n"
                            "- BEGIN LOOP specifies the start of a loop and can have an attached REPEAT specifying the number of iterations (1 to 7).\n"
                            "- STOP LOOP marks the end of a loop.\n"
                            "- NOTE specifies a musical note and can have an attached PITCH (1 to 7).\n"
                            "- REPEAT specifies how many times a NOTE should be played (1 to 7).\n"
                            "Please interpret the image, extract the commands according to these rules, and list them in order."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()

    # Log the response for debugging
    print("OpenAI API response:", response_data)

    if 'choices' in response_data and response_data['choices']:
        return response_data['choices'][0]['message']['content']
    else:
        raise ValueError("Unexpected response from OpenAI API")

def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = amplitude * np.sin(frequency * t * 2 * np.pi)
    audio = wave * (2**15 - 1) / np.max(np.abs(wave))
    audio = audio.astype(np.int16)
    return audio

def save_note_to_file(pitch, duration, file_path):
    frequency = pitch_mappings.get(pitch, pitch_mappings[default_pitch])
    audio = generate_sine_wave(frequency, duration)

    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16 bits per sample
        wav_file.setframerate(44100)
        wav_file.writeframes(audio.tobytes())

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
                    commands.append((save_note_to_file, pitch))
        index += 1
    return index, False

def process_image_to_music(file_path):
    text = recognize_text_from_image(file_path)
    words = text.upper().split()

    commands = []
    index, should_end = process_commands(commands, words, 0)
    command_queue = deque(commands)  # Add commands to the queue

    # Generate a unique filename for the output music file
    music_filename = str(uuid.uuid4()) + '.wav'
    music_path = os.path.join(MUSIC_FOLDER, music_filename)

    # Process commands from the queue and save the notes to the file
    for i, command in enumerate(command_queue):
        note_file_path = os.path.join(MUSIC_FOLDER, f"note_{i}.wav")
        command[0](command[1], 1, note_file_path)  # Save each note to a separate file

    # Combine all note files into one music file
    with wave.open(music_path, 'wb') as output_file:
        output_file.setnchannels(1)  # Mono
        output_file.setsampwidth(2)  # 16 bits per sample
        output_file.setframerate(44100)
        
        for i in range(len(command_queue)):
            note_file_path = os.path.join(MUSIC_FOLDER, f"note_{i}.wav")
            with wave.open(note_file_path, 'rb') as note_file:
                output_file.writeframes(note_file.readframes(note_file.getnframes()))

    return music_filename

@app.route('/<path:filename>', methods=['GET'])
def serve_file(filename):
    return send_from_directory(MUSIC_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)