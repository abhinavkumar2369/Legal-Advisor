import json
import re
import os
import streamlit as st
from typing import Dict, List, Optional

class LegalGlossary:
    def __init__(self, glossary_path: str = "glossary.json"):
        """Initialize legal glossary with terms and definitions"""
        self.glossary_path = glossary_path
        self.glossary = self._load_glossary()
    
    def _load_glossary(self) -> Dict:
        """Load glossary from JSON file or create default"""
        try:
            if os.path.exists(self.glossary_path):
                with open(self.glossary_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create default glossary
                default_glossary = self._get_default_glossary()
                self._save_glossary(default_glossary)
                return default_glossary
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            st.warning(f"Error loading glossary, using default: {str(e)}")
            return self._get_default_glossary()
    
    def _get_default_glossary(self) -> Dict:
        """Return default legal terms glossary"""
        return {
            "indemnity": "A contractual obligation of one party to compensate the loss incurred by the other party.",
            "non-compete": "A clause preventing you from working for competitors after leaving the job.",
            "arbitration": "A method of dispute resolution outside of court where a neutral third party makes a binding decision.",
            "liquidated damages": "A predetermined amount of money that must be paid if a contract is breached.",
            "force majeure": "Unforeseeable circumstances that prevent a party from fulfilling a contract.",
            "breach of contract": "Failure to perform any duty or obligation specified in a contract.",
            "termination clause": "A provision that specifies the conditions under which a contract can be ended.",
            "confidentiality": "An obligation to keep certain information secret and not disclose it to others.",
            "liability": "Legal responsibility for one's acts or omissions.",
            "warranty": "A guarantee or promise that certain facts or conditions are true or will happen.",
            "consideration": "Something of value exchanged between parties to make a contract legally binding.",
            "due diligence": "The investigation or exercise of care that a reasonable business or person is expected to take before entering into an agreement or contract.",
            "escrow": "A financial arrangement where a third party holds and regulates payment of funds required for two parties involved in a given transaction.",
            "lien": "A right to keep possession of property belonging to another person until a debt owed by that person is discharged.",
            "power of attorney": "A legal document giving one person the power to act for another person in specified or all legal or financial matters.",
            "statute of limitations": "A law that sets the maximum time after an event within which legal proceedings may be initiated.",
            "tort": "A wrongful act or an infringement of a right (other than under contract) leading to civil legal liability.",
            "jurisdiction": "The official power to make legal decisions and judgments.",
            "precedent": "A legal decision that serves as an authoritative rule or pattern in future similar or analogous cases.",
            "injunction": "A judicial order that restrains a person from beginning or continuing an action."
        }
    
    def _save_glossary(self, glossary: Dict) -> None:
        """Save glossary to JSON file"""
        try:
            with open(self.glossary_path, 'w', encoding='utf-8') as f:
                json.dump(glossary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving glossary: {str(e)}")
    
    def explain_term(self, term: str) -> Optional[str]:
        """Get explanation for a specific legal term"""
        if not term:
            return None
            
        term_lower = term.lower().strip()
        
        # Direct match
        if term_lower in self.glossary:
            return self.glossary[term_lower]
        
        # Partial match
        for key, value in self.glossary.items():
            if term_lower in key or key in term_lower:
                return f"Related term '{key}': {value}"
        
        return None
    
    def find_legal_terms(self, text: str) -> List[Dict]:
        """Find legal terms in the given text"""
        if not text:
            return []
            
        found_terms = []
        text_lower = text.lower()
        
        for term, definition in self.glossary.items():
            # Create regex pattern for whole word matching
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            
            if matches:
                found_terms.append({
                    'term': term,
                    'definition': definition,
                    'count': len(matches),
                    'positions': [match.span() for match in matches]
                })
        
        # Sort by frequency
        found_terms.sort(key=lambda x: x['count'], reverse=True)
        return found_terms
    
    def get_all_terms(self) -> Dict:
        """Get all terms in the glossary"""
        return self.glossary.copy()
    
    def add_term(self, term: str, definition: str) -> bool:
        """Add a new term to the glossary"""
        if not term or not definition:
            return False
            
        try:
            self.glossary[term.lower().strip()] = definition.strip()
            self._save_glossary(self.glossary)
            return True
        except Exception as e:
            st.error(f"Error adding term: {str(e)}")
            return False
    
    def search_terms(self, query: str) -> List[Dict]:
        """Search for terms matching the query"""
        if not query:
            return []
            
        query_lower = query.lower().strip()
        results = []
        
        for term, definition in self.glossary.items():
            # Check if query matches term or definition
            if (query_lower in term or 
                query_lower in definition.lower() or 
                term in query_lower):
                
                results.append({
                    'term': term,
                    'definition': definition,
                    'relevance_score': self._calculate_relevance(query_lower, term, definition)
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results
    
    def _calculate_relevance(self, query: str, term: str, definition: str) -> float:
        """Calculate relevance score for search results"""
        score = 0.0
        
        # Exact term match gets highest score
        if query == term:
            score += 100
        elif query in term:
            score += 50
        elif term in query:
            score += 30
        
        # Definition match gets lower score
        if query in definition.lower():
            score += 10
        
        return score
