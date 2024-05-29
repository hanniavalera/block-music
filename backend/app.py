# backend/app.py
import base64
import numpy as np
import os
import wave
import openai
from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
import uuid
import requests

# Setup initial configurations - make it like do, re, mi, fa, so, la, ti
pitch_mappings = {'ONE': 261.63, 'TWO': 293.66, 'THREE': 329.63, 'FOUR': 349.23, 'FIVE': 392.00, 'SIX': 440.00, 'SEVEN': 493.88}
default_pitch = 'ONE'

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
MUSIC_FOLDER = 'music'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MUSIC_FOLDER, exist_ok=True)

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
            return jsonify({'musicUrl': f'/music/{music_filename}'}), 200
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
                            "- NOTE specifies a musical note and can have an attached PITCH (ONE TO SEVEN).\n"
                            "- REPEAT specifies how many times a NOTE should be played (ONE TO SEVEN).\n"
                            "Please interpret the image, extract the notes and pitches that would result from running the program.\n"
                            "Provice a sequence of pairs in the format NOTE PITCH, in which NOTE stays the same and PITCH is a number from ONE to SEVEN.\n"
                            "Output the sequence solely as a comma-separated list with NOTE PITCH with no additional explanation.\n"
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

def find_pitch_word(string_list, pitch_mappings):
    for string in string_list:
        words = string.split()
        for word in words:
            if word in pitch_mappings:
                return word
    return None

def process_image_to_music(file_path):
    text = recognize_text_from_image(file_path)
    # Split the recognized text into a list of commands
    commands = text.split(",")
    print("Commands:", commands)

    # Generate a unique filename for the output music file
    music_filename = str(uuid.uuid4()) + '.wav'
    music_path = os.path.join(MUSIC_FOLDER, music_filename)
    audio_frames = []

    # Process the commands and generate the notes
    for command in commands:
        command = command.strip()
        if command.startswith("NOTE"):
            # find the number within the command string, if not do default pitch
            pitch = find_pitch_word(command.split(), pitch_mappings) or default_pitch
            note_file_path = os.path.join(MUSIC_FOLDER, f"note_{uuid.uuid4()}.wav")
            save_note_to_file(pitch, 1, note_file_path)
            with wave.open(note_file_path, 'rb') as note_file:
                audio_frames.append(note_file.readframes(note_file.getnframes()))

    # Combine all note files into one music file
    with wave.open(music_path, 'wb') as output_file:
        output_file.setnchannels(1)  # Mono
        output_file.setsampwidth(2)  # 16 bits per sample
        output_file.setframerate(44100)
        for frames in audio_frames:
            output_file.writeframes(frames)

    return music_filename

@app.route('/music/<filename>')
def get_music_file(filename):
    return send_from_directory(MUSIC_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)