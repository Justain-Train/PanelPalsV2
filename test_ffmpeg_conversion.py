#!/usr/bin/env python3
"""
Quick test to check if ffmpeg conversion works with ElevenLabs audio
"""
import subprocess
import tempfile
import os
from elevenlabs import generate, Voice, VoiceSettings

# Get API key from .env
from dotenv import load_dotenv
load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

print("🎤 Generating test audio from ElevenLabs...")
audio_bytes = generate(
    text="Hello, this is a test.",
    voice=Voice(
        voice_id=ELEVENLABS_VOICE_ID,
        settings=VoiceSettings(stability=0.5, similarity_boost=0.75)
    ),
    model="eleven_turbo_v2",
    api_key=ELEVENLABS_API_KEY
)

# Convert generator to bytes
if not isinstance(audio_bytes, bytes):
    audio_bytes = b''.join(audio_bytes)

print(f"✅ Generated {len(audio_bytes)} bytes of audio")

# Save to temp MP3 file
with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
    mp3_file.write(audio_bytes)
    mp3_path = mp3_file.name

print(f"💾 Saved to: {mp3_path}")

# Try to convert with ffmpeg
wav_path = mp3_path.replace('.mp3', '.wav')

print("🔄 Converting with ffmpeg...")
try:
    result = subprocess.run([
        'ffmpeg', '-i', mp3_path,
        '-ar', '44100',
        '-ac', '1',
        '-y',
        '-loglevel', 'error',
        wav_path
    ], check=True, capture_output=True, timeout=30)
    
    print(f"✅ Conversion successful!")
    print(f"📁 WAV file: {wav_path}")
    print(f"📊 WAV size: {os.path.getsize(wav_path)} bytes")
    
    # Play it
    print("🔊 Playing audio...")
    subprocess.run(['afplay', wav_path])
    
except subprocess.TimeoutExpired:
    print("❌ FFmpeg timed out!")
except subprocess.CalledProcessError as e:
    print(f"❌ FFmpeg failed: {e}")
    print(f"stderr: {e.stderr.decode()}")
finally:
    # Cleanup
    if os.path.exists(mp3_path):
        os.unlink(mp3_path)
    if os.path.exists(wav_path):
        os.unlink(wav_path)
