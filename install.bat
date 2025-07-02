@echo off
echo Installing Legal Ease AI...
echo.

REM Create models directory
if not exist "models" mkdir models

echo Step 1: Installing basic dependencies...
pip install streamlit PyMuPDF python-docx numpy sentence-transformers faiss-cpu spacy argostranslate

echo.
echo Step 2: Installing LangChain (optional for LLM support)...
pip install langchain langchain-community

echo.
echo Step 3: Attempting to install llama-cpp-python...
echo (This may take a while and might fail - that's okay!)

REM Try to install llama-cpp-python
pip install llama-cpp-python --force-reinstall --no-cache-dir

echo.
echo Installation complete!
echo.
echo To run the application:
echo   streamlit run app.py
echo.
echo Note: If llama-cpp-python failed to install, the app will still work
echo with basic features. You can add LLM support later.
echo.
pause
