# PortAudio Installation Guide

## The Problem
The error "PortAudio library not found" occurs when trying to install `sounddevice`, which is required for audio recording features in the Legal Advisor app.

## Solutions by Operating System

### 🪟 **Windows**

#### Method 1: Using Conda (Recommended)
```bash
# Install conda if you don't have it: https://anaconda.org/
conda install portaudio
pip install sounddevice
```

#### Method 2: Pre-compiled wheels
```bash
# Use pre-compiled wheel that includes PortAudio
pip install sounddevice --only-binary=all
```

#### Method 3: Manual PortAudio Installation
1. Download PortAudio from: http://www.portaudio.com/download.html
2. Extract and follow Windows build instructions
3. Add to system PATH
4. Then install sounddevice: `pip install sounddevice`

### 🐧 **Linux (Ubuntu/Debian)**

```bash
# Install PortAudio development libraries
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev

# Install Python audio packages
pip install sounddevice SpeechRecognition pyttsx3
```

### 🍎 **macOS**

```bash
# Using Homebrew
brew install portaudio

# Install Python audio packages
pip install sounddevice SpeechRecognition pyttsx3
```

## Alternative: Voice Features Without PortAudio

If you can't install PortAudio, you can create a requirements file without voice features:

### Create requirements_no_voice.txt:
```
streamlit>=1.28.0
PyMuPDF>=1.23.0
python-docx>=0.8.11
numpy>=1.24.0
scikit-learn>=1.0.0
scipy>=1.10.0
langchain>=0.1.0
langchain-community>=0.0.10
spacy>=3.7.0
langdetect>=1.0.9
googletrans>=4.0.0
google-generativeai>=0.3.0
python-dotenv>=1.0.0
```

Then install with:
```bash
pip install -r requirements_no_voice.txt
```

## Docker Solution

If you're using Docker, add this to your Dockerfile:

```dockerfile
# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
```

## Cloud Deployment Notes

### Streamlit Cloud
- Voice features may not work in web-based environments
- Use `requirements_no_voice.txt` for cloud deployment

### Heroku
Add to your `Aptfile`:
```
portaudio19-dev
```

## Testing Audio Installation

After installation, test with:

```python
import sounddevice as sd
print("Available audio devices:")
print(sd.query_devices())
```

## Troubleshooting

### Error: "Microsoft Visual C++ 14.0 is required" (Windows)
- Install Microsoft Visual C++ Build Tools
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Error: "Failed building wheel for sounddevice"
- Try installing with: `pip install sounddevice --no-cache-dir`
- Or use conda: `conda install sounddevice`

### Error: "No module named '_portaudio'"
- Reinstall sounddevice: `pip uninstall sounddevice && pip install sounddevice`

## Quick Test Commands

```bash
# Test if PortAudio is available
python -c "import sounddevice; print('PortAudio available!')"

# Test speech recognition
python -c "import speech_recognition; print('SpeechRecognition available!')"

# Test text-to-speech
python -c "import pyttsx3; print('Text-to-speech available!')"
```
