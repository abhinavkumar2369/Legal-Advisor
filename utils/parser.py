import fitz  # PyMuPDF
from docx import Document
import re
import os
import streamlit as st
from typing import List, Dict

class DocumentParser:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def parse_pdf(self, file_path: str) -> List[Dict]:
        """Parse PDF file and return text chunks"""
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
        """Parse DOCX file and return text chunks"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return self._create_chunks(self._clean_text(text))
        except Exception as e:
            st.error(f"Error parsing DOCX: {str(e)}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\%\$\&\@]', '', text)
        return text.strip()
    
    def _create_chunks(self, text: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'word_count': len(chunk_words),
                'start_index': i,
                'end_index': min(i + self.chunk_size, len(words))
            })
        
        return chunks

def parse_document(uploaded_file) -> List[Dict]:
    """Main function to parse uploaded document"""
    parser = DocumentParser()
    
    # Save uploaded file temporarily
    temp_file_path = f"temp_{uploaded_file.name}"
    
    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Parse based on file extension
        if uploaded_file.name.lower().endswith('.pdf'):
            chunks = parser.parse_pdf(temp_file_path)
        elif uploaded_file.name.lower().endswith('.docx'):
            chunks = parser.parse_docx(temp_file_path)
        else:
            st.error("Unsupported file format. Please upload PDF or DOCX files.")
            return []
        
        return chunks
    
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return []
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass  # Ignore cleanup errors
