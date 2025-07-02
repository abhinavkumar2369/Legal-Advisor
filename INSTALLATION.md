# Legal Advisor - Installation Guide

## Installation Options

### Option 1: Core Features Only (Recommended for initial testing)
```bash
pip install -r requirements_core.txt
```

### Option 2: Full Features (May require additional system dependencies)
```bash
pip install -r requirements.txt
```

### Option 3: Step-by-step Installation (If you encounter issues)

1. **Install core dependencies first:**
   ```bash
   pip install streamlit PyMuPDF python-docx numpy scikit-learn
   ```

2. **Install AI/NLP dependencies:**
   ```bash
   pip install langchain langchain-community google-generativeai
   ```

3. **Install optional features (if needed):**
   ```bash
   # For language detection
   pip install langdetect

   # For audio features (may require system dependencies)
   pip install sounddevice scipy SpeechRecognition pyttsx3

   # For better embeddings (may require compilation)
   pip install sentence-transformers faiss-cpu
   ```

## Troubleshooting Common Issues

### 1. sentencepiece compilation error
**Problem:** Building wheel for sentencepiece failed
**Solution:** Use the fallback TF-IDF embedder (already implemented in the code)

### 2. sounddevice/audio issues
**Problem:** Audio recording not working
**Solutions:**
- On Windows: Install Microsoft Visual C++ Build Tools
- On Linux: `sudo apt-get install portaudio19-dev python3-dev`
- On macOS: `brew install portaudio`

### 3. spacy model missing
**Problem:** Spacy model not found
**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### 4. cmake/pkg-config missing (Linux)
**Problem:** cmake not found during compilation
**Solution:**
```bash
sudo apt-get update
sudo apt-get install cmake pkg-config
```

## Feature Status Based on Dependencies

| Feature | Required Package | Status |
|---------|-----------------|--------|
| Core Document Processing | PyMuPDF, python-docx | ✅ Always Available |
| Basic Text Analysis | scikit-learn | ✅ Always Available |
| AI Q&A | google-generativeai | ✅ Always Available |
| Voice Recognition | sounddevice, SpeechRecognition | ⚠️ Optional |
| Text-to-Speech | pyttsx3 | ⚠️ Optional |
| Advanced Embeddings | sentence-transformers | ⚠️ Optional (has fallback) |
| Language Detection | langdetect | ⚠️ Optional |

## Running the Application

1. **With core features only:**
   ```bash
   streamlit run app.py
   ```

2. **Check which features are available:**
   The app will show warnings for disabled features in the UI.

## Environment-Specific Notes

### Cloud Platforms (Streamlit Cloud, Heroku, etc.)
- Use `requirements_core.txt` for better compatibility
- Some audio features may not work in web environments

### Local Development
- Use `requirements.txt` for full features
- Install system dependencies as needed

### Docker
- Add system dependencies to your Dockerfile:
  ```dockerfile
  RUN apt-get update && apt-get install -y \
      cmake \
      pkg-config \
      portaudio19-dev \
      python3-dev
  ```
