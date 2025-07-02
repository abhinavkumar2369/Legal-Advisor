# requirements.txt


# glossary.json
{
  "indemnity": "A contractual obligation of one party to compensate the loss incurred by the other party.",
  "non-compete": "A clause preventing you from working for competitors after leaving the job.",
  "arbitration": "A method of dispute resolution outside of court where a neutral third party makes a binding decision.",
  "liquidated damages": "A predetermined amount of money that must be paid if a contract is breached.",
  "force majeure": "Unforeseeable circumstances that prevent a party from fulfilling a contract.",
  "breach of contract": "Failure to perform any duty or obligation specified in a contract.",
  "termination clause": "A provision that specifies the conditions under which a contract can be ended.",
  "confidentiality": "An obligation to keep certain information secret and not disclose it to others.",
  "liability": "Legal responsibility for one's acts or omissions.",
  "warranty": "A guarantee or promise that certain facts or conditions are true or will happen."
}

# utils/parser.py
import fitz
from docx import Document
import re
from typing import List, Dict
import streamlit as st
import os

class DocumentParser:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def parse_pdf(self, file_path: str) -> List[Dict]:
        try:
            doc = fitz.open(file_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text() + "\n\n"
            doc.close()
            return self._create_chunks(self._clean_text(text))
        except Exception as e:
            st.error(f"Error parsing PDF: {str(e)}")
            return []
    
    def parse_docx(self, file_path: str) -> List[Dict]:
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return self._create_chunks(self._clean_text(text))
        except Exception as e:
            st.error(f"Error parsing DOCX: {str(e)}")
            return []
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\%\$\&\@]', '', text).strip()
    
    def _create_chunks(self, text: str) -> List[Dict]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append({
                'text': ' '.join(chunk_words),
                'chunk_id': len(chunks),
                'word_count': len(chunk_words)
            })
        return chunks

def parse_document(uploaded_file) -> List[Dict]:
    parser = DocumentParser()
    with open(f"temp_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if uploaded_file.name.endswith('.pdf'):
        chunks = parser.parse_pdf(f"temp_{uploaded_file.name}")
    elif uploaded_file.name.endswith('.docx'):
        chunks = parser.parse_docx(f"temp_{uploaded_file.name}")
    else:
        st.error("Unsupported file format")
        return []
    
    os.remove(f"temp_{uploaded_file.name}")
    return chunks

# utils/embedder.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict
import streamlit as st

class DocumentEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        self.index = None
        self.chunks = []
        self.vectorstore_path = "vectorstore"
        os.makedirs(self.vectorstore_path, exist_ok=True)
    
    def embed_documents(self, chunks: List[Dict]) -> None:
        try:
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.model.encode(texts)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings.astype(np.float32))
            self.chunks = chunks
            self._save_vectorstore()
            st.success(f"Successfully embedded {len(chunks)} document chunks")
        except Exception as e:
            st.error(f"Error embedding documents: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        try:
            if self.index is None:
                self._load_vectorstore()
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(query_embedding.astype(np.float32), k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    chunk['similarity_score'] = float(distances[0][i])
                    results.append(chunk)
            return results
        except Exception as e:
            st.error(f"Error during similarity search: {str(e)}")
            return []
    
    def _save_vectorstore(self) -> None:
        try:
            faiss.write_index(self.index, os.path.join(self.vectorstore_path, "index.faiss"))
            with open(os.path.join(self.vectorstore_path, "chunks.pkl"), "wb") as f:
                pickle.dump(self.chunks, f)
        except Exception as e:
            st.error(f"Error saving vectorstore: {str(e)}")
    
    def _load_vectorstore(self) -> bool:
        try:
            index_path = os.path.join(self.vectorstore_path, "index.faiss")
            chunks_path = os.path.join(self.vectorstore_path, "chunks.pkl")
            if os.path.exists(index_path) and os.path.exists(chunks_path):
                self.index = faiss.read_index(index_path)
                with open(chunks_path, "rb") as f:
                    self.chunks = pickle.load(f)
                return True
            return False
        except Exception as e:
            st.error(f"Error loading vectorstore: {str(e)}")
            return False
    
    def get_document_summary(self, max_chunks: int = 3) -> str:
        if not self.chunks:
            return "No document loaded"
        summary_chunks = self.chunks[:max_chunks]
        summary_text = "\n\n".join([chunk['text'] for chunk in summary_chunks])
        return summary_text[:2000] + "..." if len(summary_text) > 2000 else summary_text

# utils/glossary.py
import json
import re
from typing import Dict, List, Optional
import streamlit as st

class LegalGlossary:
    def __init__(self, glossary_path: str = "glossary.json"):
        self.glossary_path = glossary_path
        self.glossary = self._load_glossary()
    
    def _load_glossary(self) -> Dict:
        try:
            with open(self.glossary_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            st.warning(f"Glossary file {self.glossary_path} not found.")
            return {}
        except json.JSONDecodeError:
            st.error(f"Error reading glossary file {self.glossary_path}")
            return {}
    
    def explain_term(self, term: str) -> Optional[str]:
        term_lower = term.lower()
        if term_lower in self.glossary:
            return self.glossary[term_lower]
        for key, value in self.glossary.items():
            if term_lower in key or key in term_lower:
                return f"Related term '{key}': {value}"
        return None
    
    def find_legal_terms(self, text: str) -> List[Dict]:
        found_terms = []
        text_lower = text.lower()
        for term, definition in self.glossary.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                found_terms.append({
                    'term': term,
                    'definition': definition,
                    'start': match.start(),
                    'end': match.end()
                })
        return found_terms
    
    def get_all_terms(self) -> Dict:
        return self.glossary

# agents/legal_agent.py
from langchain.llms import LlamaCpp
from typing import List, Dict, Optional
import streamlit as st
import os

class LegalAgent:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        self.llm = None
        self.retriever = None
        self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        try:
            if os.path.exists(self.model_path):
                self.llm = LlamaCpp(
                    model_path=self.model_path,
                    temperature=0.1,
                    max_tokens=512,
                    n_ctx=2048,
                    verbose=False
                )
                st.success("Local LLM loaded successfully")
            else:
                st.error(f"Model file not found at {self.model_path}")
        except Exception as e:
            st.error(f"Error loading LLM: {str(e)}")
    
    def set_retriever(self, embedder) -> None:
        self.retriever = embedder
    
    def ask_question(self, question: str) -> str:
        if not self.llm:
            return "LLM not available. Please check model configuration."
        if not self.retriever:
            return "No document loaded. Please upload a document first."
        
        try:
            relevant_chunks = self.retriever.similarity_search(question, k=3)
            if not relevant_chunks:
                return "No relevant information found in the document."
            
            context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
            prompt = f"""Based on the following legal document content, answer the question clearly and concisely.

Context:
{context}

Question: {question}

Answer:"""
            
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    def summarize_document(self, max_length: int = 500) -> str:
        if not self.llm or not self.retriever:
            return "LLM or document not available."
        
        try:
            doc_text = self.retriever.get_document_summary(max_chunks=5)
            prompt = f"""Summarize the following legal document in plain English. Focus on:
1. Document type and parties involved
2. Key obligations and rights
3. Important terms and conditions

Document content:
{doc_text}

Summary:"""
            
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def simplify_text(self, text: str) -> str:
        if not self.llm:
            return "LLM not available."
        
        try:
            prompt = f"""Rewrite the following legal text in simple, plain English:

Legal text:
{text}

Plain English version:"""
            
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            return f"Error simplifying text: {str(e)}"

# app.py
import streamlit as st
import os
from utils.parser import parse_document
from utils.embedder import DocumentEmbedder
from utils.glossary import LegalGlossary
from agents.legal_agent import LegalAgent

st.set_page_config(page_title="Legal Ease AI", page_icon="⚖️", layout="wide")

if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'glossary' not in st.session_state:
    st.session_state.glossary = LegalGlossary()

def process_document(uploaded_file):
    with st.spinner("Processing document..."):
        try:
            chunks = parse_document(uploaded_file)
            if chunks:
                embedder = DocumentEmbedder()
                embedder.embed_documents(chunks)
                st.session_state.embedder = embedder
                st.session_state.document_processed = True
                st.success(f"Document processed! Found {len(chunks)} chunks.")
                st.rerun()
            else:
                st.error("Failed to process document.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

def load_model(model_path):
    with st.spinner("Loading model..."):
        try:
            agent = LegalAgent(model_path)
            if st.session_state.embedder:
                agent.set_retriever(st.session_state.embedder)
            st.session_state.agent = agent
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

def main():
    st.title("⚖️ Legal Ease AI")
    st.markdown("Free AI-Powered Legal Document Assistant")
    
    with st.sidebar:
        st.header("📁 Upload")
        uploaded_file = st.file_uploader("Choose document", type=['pdf', 'docx'])
        if uploaded_file and st.button("Process Document", type="primary"):
            process_document(uploaded_file)
        
        st.header("🔧 Settings")
        model_path = st.text_input("Model Path", value="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
        if st.button("Load Model"):
            load_model(model_path)
    
    if st.session_state.document_processed:
        tabs = st.tabs(["📊 Summary", "❓ Q&A", "📚 Glossary", "🌐 Translation"])
        
        with tabs[0]:
            st.header("📊 Document Summary")
            if st.button("Generate Summary", type="primary"):
                if st.session_state.agent:
                    with st.spinner("Generating summary..."):
                        summary = st.session_state.agent.summarize_document()
                        st.write(summary)
                else:
                    st.warning("Please load model first.")
            
            if st.session_state.embedder:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chunks", len(st.session_state.embedder.chunks))
                with col2:
                    total_words = sum(chunk.get('word_count', 0) for chunk in st.session_state.embedder.chunks)
                    st.metric("Words", total_words)
                with col3:
                    st.metric("Read Time", f"{total_words // 200} min")
        
        with tabs[1]:
            st.header("❓ Ask Questions")
            question = st.text_input("Question:", placeholder="What are the termination conditions?")
            if st.button("Ask", type="primary") and question:
                if st.session_state.agent:
                    with st.spinner("Finding answer..."):
                        answer = st.session_state.agent.ask_question(question)
                        st.write(answer)
                else:
                    st.warning("Please load model first.")
            
            sample_questions = [
                "What are the main obligations?",
                "What are termination conditions?",
                "Are there penalties?",
                "What is the payment schedule?"
            ]
            for i, q in enumerate(sample_questions):
                if st.button(q, key=f"q_{i}"):
                    if st.session_state.agent:
                        answer = st.session_state.agent.ask_question(q)
                        st.write(answer)
        
        with tabs[2]:
            st.header("📚 Legal Glossary")
            search_term = st.text_input("Search term:", placeholder="indemnity")
            if search_term:
                explanation = st.session_state.glossary.explain_term(search_term)
                if explanation:
                    st.success(f"**{search_term}**: {explanation}")
                else:
                    st.warning(f"Term '{search_term}' not found.")
            
            all_terms = st.session_state.glossary.get_all_terms()
            for term, definition in all_terms.items():
                with st.expander(f"📖 {term.title()}"):
                    st.write(definition)
        
        with tabs[3]:
            st.header("🌐 Translation")
            text_to_translate = st.text_area("Text to translate:")
            target_language = st.selectbox("Translate to:", ["Hindi", "Spanish", "French", "German"])
            
            if st.button("Translate", type="primary") and text_to_translate:
                try:
                    import argostranslate.translate as translate
                    lang_codes = {"Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de"}
                    target_code = lang_codes.get(target_language, "hi")
                    with st.spinner(f"Translating to {target_language}..."):
                        translated = translate.translate(text_to_translate, "en", target_code)
                        st.write(f"**{target_language}:** {translated}")
                except ImportError:
                    st.error("Translation library not installed. Run: pip install argostranslate")
                except Exception as e:
                    st.error(f"Translation error: {str(e)}")
    else:
        st.markdown("""
        ## Welcome to Legal Ease AI! 🚀
        
        **Features:**
        - 📄 Document Analysis (PDF/DOCX)
        - 🔍 Plain English Summaries
        - ❓ Q&A System
        - 📚 Legal Glossary
        - 🌐 Translation Support
        
        **Get Started:**
        1. Upload a legal document
        2. Download a GGUF model from HuggingFace
        3. Set model path in settings
        4. Start analyzing!
        """)

if __name__ == "__main__":
    main()