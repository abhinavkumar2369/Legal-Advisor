import streamlit as st
import os
import time
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.info("python-dotenv not installed. Using system environment variables only.")

from utils.parser import parse_document
try:
    from utils.embedder import DocumentEmbedder
except ImportError:
    from utils.embedder_fallback import DocumentEmbedder
    st.warning("Using fallback TF-IDF embedder due to dependency issues")
from utils.glossary import LegalGlossary
from agents.legal_agents import LegalAgent
import json

# Optional imports for voice features
VOICE_AVAILABLE = False
try:
    import sounddevice as sd
    import numpy as np
    import scipy.io.wavfile as wav
    import speech_recognition as sr
    import tempfile
    from utils.voice_assistant import record_audio, save_audio_to_wav, transcribe_audio, speak_text
    VOICE_AVAILABLE = True
except ImportError:
    st.info("Voice features disabled due to missing dependencies. Install sounddevice, scipy, and SpeechRecognition to enable voice features.")

# Optional language detection
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    st.info("Language detection disabled. Install langdetect to enable this feature.")





st.set_page_config(page_title="Legal Ease AI", page_icon="⚖️", layout="wide")

# Initialize session state
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'glossary' not in st.session_state:
    st.session_state.glossary = LegalGlossary()
if 'document_info' not in st.session_state:
    st.session_state.document_info = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

def get_file_info(uploaded_file):
    """Extract basic information about the uploaded file"""
    file_info = {
        'name': uploaded_file.name,
        'size': uploaded_file.size,
        'type': uploaded_file.type,
        'size_mb': round(uploaded_file.size / (1024 * 1024), 2),
        'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return file_info

def process_and_analyze_document(uploaded_file):
    """Process document and automatically generate analysis"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Parse document
        status_text.text("📄 Parsing document...")
        progress_bar.progress(20)
        chunks = parse_document(uploaded_file)
        
        if not chunks:
            st.error("Failed to parse document. Please check the file format.")
            return False
        
        # Step 2: Create embeddings
        status_text.text("🔍 Creating embeddings...")
        progress_bar.progress(40)
        embedder = DocumentEmbedder()
        embedder.embed_documents(chunks)
        st.session_state.embedder = embedder
        
        # Step 3: Initialize agent
        status_text.text("🤖 Initializing AI agent...")
        progress_bar.progress(60)
        agent = LegalAgent()
        agent.set_retriever(embedder)
        st.session_state.agent = agent
        
        # Step 4: Generate analysis
        status_text.text("📊 Generating analysis...")
        progress_bar.progress(80)
        
        analysis_results = {}
        
        # Generate multiple types of summaries
        if st.session_state.agent and st.session_state.agent.model:
            # AI-powered analysis
            analysis_results['quick_summary'] = agent.summarize_document(max_length=200)
            analysis_results['detailed_summary'] = agent.summarize_document(max_length=800)
            analysis_results['key_points'] = agent.extract_key_points()
        else:
            # Fallback analysis without AI model
            analysis_results['quick_summary'] = agent.get_basic_summary() if agent else "No summary available"
            analysis_results['detailed_summary'] = st.session_state.embedder.get_document_summary(max_chunks=5)
            analysis_results['key_points'] = "AI model not loaded. Key points extraction requires a local LLM model."
        
        # Legal terms analysis (works without LLM)
        doc_text = st.session_state.embedder.get_document_summary(max_chunks=10)
        analysis_results['legal_terms'] = st.session_state.glossary.find_legal_terms(doc_text)
        
        # Document statistics
        total_words = sum(chunk.get('word_count', 0) for chunk in chunks)
        analysis_results['stats'] = {
            'total_chunks': len(chunks),
            'total_words': total_words,
            'estimated_reading_time': f"{total_words // 200} minutes",
            'legal_terms_found': len(analysis_results.get('legal_terms', []))
        }
        
        st.session_state.analysis_results = analysis_results
        st.session_state.document_processed = True
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return False

def display_document_preview(file_info):
    """Display document preview information"""
    st.subheader("📄 Document Preview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("File Name", file_info['name'])
    with col2:
        st.metric("File Size", f"{file_info['size_mb']} MB")
    with col3:
        st.metric("File Type", file_info['type'].split('/')[-1].upper())
    with col4:
        st.metric("Upload Time", file_info['upload_time'])

def display_analysis_results():
    """Display comprehensive analysis results"""
    if not st.session_state.analysis_results:
        return
    
    results = st.session_state.analysis_results
    
    # Statistics Overview
    st.subheader("📊 Document Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    stats = results.get('stats', {})
    with col1:
        st.metric("Total Chunks", stats.get('total_chunks', 0))
    with col2:
        st.metric("Total Words", stats.get('total_words', 0))
    with col3:
        st.metric("Reading Time", stats.get('estimated_reading_time', 'N/A'))
    with col4:
        st.metric("Legal Terms", stats.get('legal_terms_found', 0))
    
    # Summary Tabs
    st.subheader("📋 Document Analysis")
    summary_tabs = st.tabs(["Quick Summary", "Detailed Summary", "Key Points", "Legal Terms"])
    
    with summary_tabs[0]:
        st.markdown("### 🔍 Quick Overview")
        if 'quick_summary' in results and results['quick_summary']:
            st.write(results['quick_summary'])
        else:
            st.info("💡 **Quick summary requires an AI model.** Load a local LLM model in the sidebar for AI-powered analysis.")
            # Show basic document info as fallback
            stats = results.get('stats', {})
            st.write(f"**Document Preview:** This document contains {stats.get('total_words', 0):,} words across {stats.get('total_chunks', 0)} sections with an estimated reading time of {stats.get('estimated_reading_time', 'N/A')}.")
    
    with summary_tabs[1]:
        st.markdown("### 📝 Detailed Analysis")
        if 'detailed_summary' in results and results['detailed_summary']:
            st.write(results['detailed_summary'])
        else:
            st.info("💡 **Detailed analysis requires an AI model.** Load a local LLM model for comprehensive document analysis.")
            # Show document preview as fallback
            if st.session_state.embedder:
                preview = st.session_state.embedder.get_document_summary(max_chunks=3)
                st.markdown("**Document Preview (First 3 sections):**")
                st.text_area("", preview, height=200, disabled=True)
    
    with summary_tabs[2]:
        st.markdown("### 🎯 Key Points")
        if 'key_points' in results and results['key_points'] and not results['key_points'].startswith("AI model not loaded"):
            st.write(results['key_points'])
        else:
            st.info("💡 **Key points extraction requires an AI model.** Load a local LLM model to automatically extract important clauses and terms.")
            # Manual key points suggestion
            st.markdown("**💭 Manual Review Suggestions:**")
            st.markdown("""
            When reviewing this document, pay attention to:
            - Payment terms and amounts
            - Termination conditions
            - Obligations of each party
            - Deadlines and time limits
            - Penalties or consequences
            - Confidentiality requirements
            """)
            
    with summary_tabs[3]:
        st.markdown("### ⚖️ Legal Terms Found")
        legal_terms = results.get('legal_terms', [])
        if legal_terms:
            for term in legal_terms:
                with st.expander(f"📖 {term['term'].title()}"):
                    st.write(term['definition'])
        else:
            st.info("No legal terms found or glossary not available.")

def export_analysis():
    """Export analysis results"""
    if not st.session_state.analysis_results:
        st.warning("No analysis results to export.")
        return
    
    st.subheader("📤 Export Analysis")
    
    export_format = st.selectbox("Choose export format:", ["JSON", "Text Summary", "Markdown"])
    
    if st.button("Generate Export", type="primary"):
        results = st.session_state.analysis_results
        file_info = st.session_state.document_info
        
        if export_format == "JSON":
            export_data = {
                'document_info': file_info,
                'analysis_results': results,
                'export_timestamp': datetime.now().isoformat()
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"analysis_{file_info['name'].split('.')[0]}.json",
                mime="application/json"
            )
        
        elif export_format == "Text Summary":
            text_export = f"""
Legal Document Analysis Report
==============================

Document: {file_info['name']}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

QUICK SUMMARY:
{results.get('quick_summary', 'Not available')}

DETAILED ANALYSIS:
{results.get('detailed_summary', 'Not available')}

KEY POINTS:
{results.get('key_points', 'Not available')}

STATISTICS:
- Total Words: {results.get('stats', {}).get('total_words', 'N/A')}
- Reading Time: {results.get('stats', {}).get('estimated_reading_time', 'N/A')}
- Legal Terms Found: {results.get('stats', {}).get('legal_terms_found', 'N/A')}
"""
            st.download_button(
                label="Download Text Report",
                data=text_export,
                file_name=f"analysis_{file_info['name'].split('.')[0]}.txt",
                mime="text/plain"
            )
def record_audio(duration=5, fs=44100):
    if not VOICE_AVAILABLE:
        st.error("Voice features not available")
        return None, None
    st.info("🎤 Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio, fs

def save_wav(audio, fs):
    if not VOICE_AVAILABLE:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(tmp.name, fs, audio)
    return tmp.name

def transcribe_audio(file_path):
    if not VOICE_AVAILABLE:
        return "Voice features not available"
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "❌ Could not understand audio."
    except sr.RequestError:
        return "❌ Could not request results."

from langdetect import detect  # <- Add at the top

def voice_control_ui():
    st.header("🎤 Voice Assistant")

    if not VOICE_AVAILABLE:
        st.warning("🔇 Voice features are disabled due to missing dependencies.")
        st.info("To enable voice features, install: sounddevice, scipy, SpeechRecognition, pyttsx3")
        return

    if 'voice_enabled' not in st.session_state:
        st.session_state.voice_enabled = False

    enabled = st.checkbox("Enable Voice Assistant", value=st.session_state.voice_enabled)
    st.session_state.voice_enabled = enabled

    if enabled:
        st.markdown("Use your voice to ask document-related questions.")

        if st.button("🎙️ Start Voice Command"):
            try:
                # Step 1: Record
                with st.spinner("🎧 Listening... Please speak clearly."):
                    audio, fs = record_audio()
                    if audio is None:
                        st.error("Failed to record audio.")
                        return

                # Step 2: Save to WAV
                wav_path = save_audio_to_wav(audio, fs)
                if not wav_path:
                    st.error("Failed to save audio.")
                    return

                # Step 3: Transcribe
                with st.spinner("🧠 Transcribing..."):
                    command = transcribe_audio(wav_path)

                # Display transcription
                st.success("📝 You said:")
                st.write(f"**{command}**")

                # Step 3.5: Detect Language (if available)
                language = 'en'  # default
                if LANGDETECT_AVAILABLE:
                    try:
                        language = detect(command)
                    except:
                        language = 'en'

                # Step 4: Ask LLM
                if st.session_state.agent and st.session_state.agent.model:
                    with st.spinner("🤖 Thinking..."):
                        answer = st.session_state.agent.ask_question(command)
                        st.markdown("**🤖 Answer:**")
                        st.write(answer)

                        # Speak the answer
                        if VOICE_AVAILABLE:
                            try:
                                speak_text(answer)
                            except:
                                st.info("Text-to-speech not available")
                else:
                    st.warning("🤖 Please load a model to use voice Q&A.")
            except Exception as e:
                st.error(f"❌ Voice interaction failed: {str(e)}")
    else:
        st.info("🔇 Voice assistant is disabled. Enable the checkbox to use it.")





def main():
    st.title("⚖️ Legal Ease AI")
    st.markdown("**AI-Powered Legal Document Analysis & Summary Generator**")

    # Sidebar for settings and model configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key Configuration
        st.subheader("🔑 API Configuration")
        
        # Check if API key is available from environment
        env_api_key = os.getenv('GEMINI_API_KEY')
        if env_api_key:
            st.success("✅ API key loaded from environment")
            api_key_input = env_api_key
        else:
            st.warning("⚠️ No API key found in environment")
            api_key_input = st.text_input(
                "Gemini API Key",
                type="password",
                help="Enter your Google Gemini API key. Get it from Google AI Studio."
            )
        
        # Initialize or update agent with API key
        if api_key_input:
            if 'agent' not in st.session_state or st.session_state.agent is None:
                agent = LegalAgent(gemini_api_key=api_key_input)
                if st.session_state.embedder:
                    agent.set_retriever(st.session_state.embedder)
                st.session_state.agent = agent
        
        st.divider()

        # Model path input (keeping for backward compatibility)
        st.subheader("🤖 Local Model (Optional)")
        model_path = st.text_input(
            "Local Model Path", 
            value="",
            help="Path to your local LLM model file (optional - Gemini AI is the default)"
        )

        if st.button("Load Local Model") and model_path:
            st.info("Local model loading not implemented in this version. Using Gemini AI.")

        st.divider()

        # Status indicators
        st.subheader("📊 Status")
        if st.session_state.agent and st.session_state.agent.model:
            st.success("🤖 Gemini AI: Connected")
        else:
            st.warning("🤖 Gemini AI: Not connected")
            if not os.getenv('GEMINI_API_KEY') and not api_key_input:
                st.info("💡 Add your API key above to enable AI features")

        if st.session_state.document_processed:
            st.success("📄 Document: Processed")
        else:
            st.info("📄 Document: Not uploaded")

        # Voice Assistant Control (in sidebar for toggle)
        voice_control_ui()

    # Main content area
    if not st.session_state.document_processed:
        st.header("📁 Upload & Analyze Document")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose a legal document to analyze",
                type=['pdf', 'docx'],
                help="Supported formats: PDF, DOCX"
            )

            if uploaded_file:
                file_info = get_file_info(uploaded_file)
                st.session_state.document_info = file_info
                display_document_preview(file_info)

        with col2:
            if uploaded_file:
                st.markdown("### 🚀 Ready to analyze!")
                if st.button("Analyze Document", type="primary", use_container_width=True):
                    success = process_and_analyze_document(uploaded_file)
                    if success:
                        st.success("✅ Document analyzed successfully!")
                        st.rerun()

        if not uploaded_file:
            st.markdown("""
                ### 🔍 How to use Legal Ease AI:
                1. **Upload** a legal document (PDF or DOCX)
                2. **Preview** document information
                3. **Analyze** with our AI engine
                4. **Review** summaries and key insights
                5. **Export** results for your records

                #### 📋 Features:
                - ✅ Quick Summary
                - ✅ Detailed Analysis
                - ✅ Key Points
                - ✅ Legal Glossary
                - ✅ Q&A System
                - ✅ Export Options
                - 🎤 Voice Assistant
            """)

    else:
        st.header("📊 Analysis Results")
        display_analysis_results()

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Analyze New Document", type="secondary"):
                st.session_state.document_processed = False
                st.session_state.embedder = None
                st.session_state.analysis_results = {}
                st.session_state.document_info = {}
                st.rerun()

        with col2:
            if st.button("📤 Export Analysis", type="secondary"):
                export_analysis()

        with col3:
            if st.button("❓ Ask Questions", type="secondary"):
                st.session_state.show_qa = True

        if st.session_state.get('show_qa', False):
            st.divider()
            st.subheader("❓ Ask Questions About Your Document")

            question = st.text_input(
                "What would you like to know?",
                placeholder="e.g., What are the termination conditions?"
            )

            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("Get Answer", type="primary") and question:
                    if st.session_state.agent and st.session_state.agent.model:
                        with st.spinner("Finding answer..."):
                            answer = st.session_state.agent.ask_question(question)
                            st.write("**Answer:**")
                            st.write(answer)
                    else:
                        st.warning("Please load a model first to use the Q&A feature.")

            with col2:
                if st.button("Close Q&A"):
                    st.session_state.show_qa = False
                    st.rerun()

            st.markdown("**Sample Questions:**")
            sample_questions = [
                "What are the main obligations of each party?",
                "What are the termination conditions?",
                "Are there any penalties or fees mentioned?",
                "What is the duration of this agreement?",
                "What are the payment terms?"
            ]

            for i, q in enumerate(sample_questions):
                if st.button(f"💡 {q}", key=f"sample_q_{i}"):
                    if st.session_state.agent and st.session_state.agent.model:
                        with st.spinner("Finding answer..."):
                            answer = st.session_state.agent.ask_question(q)
                            st.write(f"**Question:** {q}")
                            st.write(f"**Answer:** {answer}")
                    else:
                        st.warning("Please load a model first.")



if __name__ == "__main__":
    main()
