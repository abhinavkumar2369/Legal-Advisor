from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import streamlit as st
from typing import List, Dict

class DocumentEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the document embedder with a sentence transformer model"""
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = 384  # Dimension for all-MiniLM-L6-v2
            self.index = None
            self.chunks = []
            self.vectorstore_path = "vectorstore"
            os.makedirs(self.vectorstore_path, exist_ok=True)
        except Exception as e:
            st.error(f"Error initializing embedder: {str(e)}")
            self.model = None
    
    def embed_documents(self, chunks: List[Dict]) -> None:
        """Create embeddings for document chunks and build FAISS index"""
        if not self.model:
            st.error("Embedder model not available")
            return
            
        try:
            if not chunks:
                st.warning("No chunks to embed")
                return
                
            texts = [chunk['text'] for chunk in chunks]
            st.info(f"Creating embeddings for {len(texts)} chunks...")
            
            # Create embeddings
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings.astype(np.float32))
            
            # Store chunks
            self.chunks = chunks
            
            # Save to disk
            self._save_vectorstore()
            
            st.success(f"Successfully embedded {len(chunks)} document chunks")
            
        except Exception as e:
            st.error(f"Error embedding documents: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar document chunks based on query"""
        try:
            if self.index is None:
                self._load_vectorstore()
            
            if self.index is None or not self.model:
                return []
            
            # Create query embedding
            query_embedding = self.model.encode([query])
            
            # Search for similar chunks
            distances, indices = self.index.search(query_embedding.astype(np.float32), k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    chunk['similarity_score'] = float(distances[0][i])
                    chunk['rank'] = i + 1
                    results.append(chunk)
            
            return results
            
        except Exception as e:
            st.error(f"Error during similarity search: {str(e)}")
            return []
    
    def _save_vectorstore(self) -> None:
        """Save FAISS index and chunks to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, os.path.join(self.vectorstore_path, "index.faiss"))
            
            with open(os.path.join(self.vectorstore_path, "chunks.pkl"), "wb") as f:
                pickle.dump(self.chunks, f)
                
        except Exception as e:
            st.error(f"Error saving vectorstore: {str(e)}")
    
    def _load_vectorstore(self) -> bool:
        """Load FAISS index and chunks from disk"""
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
            'has_index': self.index is not None
        }
