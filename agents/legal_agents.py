# agents/legal_agents.py
import google.generativeai as genai
from typing import List, Dict, Optional
import streamlit as st
import os

class LegalAgent:
    def __init__(self, gemini_api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        # Get API key from environment variable or parameter
        self.api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = model_name
        self.model = None
        self.retriever = None
        self._initialize_gemini()
    
    def _initialize_gemini(self) -> None:
        """Initialize Gemini AI model"""
        try:
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                st.success("✅ Gemini AI connected successfully!")
            else:
                st.warning("⚠️ No Gemini API key provided. Please add your API key in the sidebar.")
        except Exception as e:
            st.error(f"❌ Error initializing Gemini: {str(e)}")
    
    def set_retriever(self, embedder) -> None:
        """Set the document retriever/embedder"""
        self.retriever = embedder
    
    def ask_question(self, question: str) -> str:
        """Answer questions about the document using Gemini AI"""
        if not self.model:
            return "❌ Gemini AI not available. Please check your API key."
        if not self.retriever:
            return "❌ No document loaded. Please upload a document first."
        
        try:
            # Get relevant document chunks
            relevant_chunks = self.retriever.similarity_search(question, k=3)
            if not relevant_chunks:
                return "❌ No relevant information found in the document."
            
            # Create context from relevant chunks
            context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
            
            # Create prompt for Gemini
            prompt = f"""You are a legal assistant helping users understand legal documents. Based on the following document content, answer the question clearly and concisely in plain English.

**Document Context:**
{context}

**Question:** {question}

**Instructions:**
- Answer in simple, clear language
- Focus on the specific question asked
- If the answer isn't in the provided context, say so
- Highlight important legal implications
- Keep the response concise but informative

**Answer:**"""
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"❌ Error processing question: {str(e)}"
    
    def summarize_document(self, max_length: int = 500) -> str:
        """Generate document summary using Gemini AI"""
        if not self.model:
            return "❌ Gemini AI not available. Please check your API key."
        if not self.retriever:
            return "❌ No document loaded."
        
        try:
            # Get document content
            doc_text = self.retriever.get_document_summary(max_chunks=5)
            
            # Create summary prompt
            if max_length <= 300:
                summary_type = "brief"
                instructions = "Provide a concise summary in 2-3 sentences."
            else:
                summary_type = "detailed"
                instructions = "Provide a comprehensive summary covering all key points."
            
            prompt = f"""You are a legal expert. Analyze the following legal document and provide a {summary_type} summary in plain English.

**Document Content:**
{doc_text}

**Instructions:**
- {instructions}
- Focus on identifying:
  - The type of document
  - The parties involved
  - The main obligations or responsibilities of each party
  - Any important terms, deadlines, or conditions
- Avoid legal jargon; use clear, simple language that non-lawyers can easily understand."""

            response = self.model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            return f"❌ Error generating summary: {str(e)}"

    def extract_key_points(self, max_chunks: int = 5) -> str:
        """Extract key points from the legal document using Gemini AI"""
        if not self.model:
            return "❌ Gemini AI not available. Please check your API key."
        if not self.retriever:
            return "❌ No document loaded."

        try:
            # Get top relevant chunks
            doc_text = self.retriever.get_document_summary(max_chunks=max_chunks)

            prompt = f"""You are a legal assistant. Extract the most important key points from the following legal document in plain English.

**Document Content:**
{doc_text}

**Instructions:**
- List 5 to 10 key points as bullet points
- Focus on obligations, deadlines, penalties, payment terms, and any legal responsibilities
- Use simple, clear language for non-lawyers
- Avoid repeating the full legal clauses; summarize them

**Key Points:**"""

            response = self.model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            return f"❌ Error extracting key points: {str(e)}"
