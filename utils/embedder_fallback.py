from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os
import streamlit as st
from typing import List, Dict

class DocumentEmbedder:
    def __init__(self, model_name: str = "tfidf"):
        """Initialize the document embedder with TF-IDF as fallback"""
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.embeddings = None
            self.chunks = []
            self.vectorstore_path = "vectorstore"
            os.makedirs(self.vectorstore_path, exist_ok=True)
            self.model = "tfidf"  # For compatibility
            st.info("Using TF-IDF embeddings (fallback mode)")
        except Exception as e:
            st.error(f"Error initializing embedder: {str(e)}")
            self.vectorizer = None
    
    def embed_documents(self, chunks: List[Dict]) -> None:
        """Create embeddings for document chunks using TF-IDF"""
        if not self.vectorizer:
            st.error("Embedder not available")
            return
            
        try:
            if not chunks:
                st.warning("No chunks to embed")
                return
                
            texts = [chunk['text'] for chunk in chunks]
            st.info(f"Creating TF-IDF embeddings for {len(texts)} chunks...")
            
            # Create embeddings using TF-IDF
            self.embeddings = self.vectorizer.fit_transform(texts)
            
            # Store chunks
            self.chunks = chunks
            
            # Save to disk
            self._save_vectorstore()
            
            st.success(f"Successfully embedded {len(chunks)} document chunks using TF-IDF")
            
        except Exception as e:
            st.error(f"Error embedding documents: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar document chunks based on query"""
        try:
            if self.embeddings is None:
                self._load_vectorstore()
            
            if self.embeddings is None or not self.vectorizer:
                return []
            
            # Create query embedding
            query_embedding = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
            
            # Get top k results
            top_indices = similarities.argsort()[-k:][::-1]
            
            results = []
            for i, idx in enumerate(top_indices):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    chunk['similarity_score'] = float(similarities[idx])
                    chunk['rank'] = i + 1
                    results.append(chunk)
            
            return results
            
        except Exception as e:
            st.error(f"Error during similarity search: {str(e)}")
            return []
    
    def _save_vectorstore(self) -> None:
        """Save vectorizer and chunks to disk"""
        try:
            if self.embeddings is not None:
                with open(os.path.join(self.vectorstore_path, "embeddings.pkl"), "wb") as f:
                    pickle.dump(self.embeddings, f)
                
                with open(os.path.join(self.vectorstore_path, "vectorizer.pkl"), "wb") as f:
                    pickle.dump(self.vectorizer, f)
            
            with open(os.path.join(self.vectorstore_path, "chunks.pkl"), "wb") as f:
                pickle.dump(self.chunks, f)
                
        except Exception as e:
            st.error(f"Error saving vectorstore: {str(e)}")
    
    def _load_vectorstore(self) -> bool:
        """Load vectorizer and chunks from disk"""
        try:
            embeddings_path = os.path.join(self.vectorstore_path, "embeddings.pkl")
            vectorizer_path = os.path.join(self.vectorstore_path, "vectorizer.pkl")
            chunks_path = os.path.join(self.vectorstore_path, "chunks.pkl")
            
            if os.path.exists(embeddings_path) and os.path.exists(chunks_path):
                with open(embeddings_path, "rb") as f:
                    self.embeddings = pickle.load(f)
                
                with open(vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                
                with open(chunks_path, "rb") as f:
                    self.chunks = pickle.load(f)
                
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Error loading vectorstore: {str(e)}")
            return False
    
    def get_document_summary(self, max_chunks: int = 3) -> str:
        """Get a summary of the document using first few chunks"""
        if not self.chunks:
            return "No document loaded"
        
        summary_chunks = self.chunks[:max_chunks]
        summary_text = "\n\n".join([chunk['text'] for chunk in summary_chunks])
        
        # Truncate if too long
        if len(summary_text) > 2000:
            return summary_text[:2000] + "..."
        
        return summary_text
    
    def get_stats(self) -> Dict:
        """Get statistics about the embedded documents"""
        if not self.chunks:
            return {}
        
        total_words = sum(chunk.get('word_count', 0) for chunk in self.chunks)
        
        return {
            'total_chunks': len(self.chunks),
            'total_words': total_words,
            'average_chunk_size': total_words // len(self.chunks) if self.chunks else 0,
            'has_index': self.embeddings is not None
        }
