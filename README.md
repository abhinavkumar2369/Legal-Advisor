# Legal Ease AI - Installation Guide

## Quick Setup (Windows)

### Option 1: Automatic Installation
Run the provided batch file:
```bash
install.bat
```

### Option 2: Manual Installation

1. **Install basic dependencies first:**
```bash
pip install streamlit PyMuPDF python-docx numpy sentence-transformers faiss-cpu spacy argostranslate
```

2. **Install llama-cpp-python (if you want local LLM support):**

   **For CPU-only:**
   ```bash
   pip install llama-cpp-python --force-reinstall --no-cache-dir
   ```

   **For GPU support (NVIDIA):**
   ```bash
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
   ```

   **If llama-cpp-python fails to install:**
   - The app will still work without it (basic features only)
   - You can download pre-compiled wheels from: https://github.com/abetlen/llama-cpp-python/releases

3. **Install LangChain components:**
```bash
pip install langchain langchain-community
```

## Running the Application

```bash
streamlit run app.py
```

## Features Available Without LLM

Even without a local LLM model, you can still use:
- ✅ Document parsing (PDF/DOCX)
- ✅ Document statistics
- ✅ Legal term glossary
- ✅ Basic document preview
- ✅ Export functionality

## Adding LLM Support

1. Download a GGUF model file (e.g., from HuggingFace)
2. Place it in a `models/` directory
3. Update the model path in the sidebar
4. Click "Load Model"

## Troubleshooting

### llama-cpp-python Installation Issues:
- Make sure you have Visual Studio Build Tools installed
- Try installing with `--no-cache-dir` flag
- Consider using pre-compiled wheels
- The app works without it!

### Memory Issues:
- Use smaller model files (Q4_K_M quantization)
- Reduce chunk size in document processing
- Close other applications

### File Upload Issues:
- Ensure PDF/DOCX files are not corrupted
- Try smaller files first
- Check file permissions

## Model Recommendations

For local LLM models, try these GGUF files:
- **Small/Fast:** mistral-7b-instruct-v0.1.Q4_K_M.gguf (~4GB)
- **Balanced:** llama-2-7b-chat.Q5_K_M.gguf (~5GB)
- **Large/Accurate:** mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf (~26GB)

Download from: https://huggingface.co/models?search=gguf
