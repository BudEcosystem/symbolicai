"""
FAISS-enhanced phrase matcher for intelligent multi-word phrase matching.
Uses semantic similarity to find phrase boundaries and match multi-word phrases.
"""

import re
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass
from collections import defaultdict

from .faiss_manager import FAISSManager
from .model_manager import Model2VecManager
from .multi_level_cache import get_global_cache


@dataclass
class PhraseCandidate:
    """Represents a potential phrase match"""
    text: str
    start_pos: int
    end_pos: int
    word_count: int
    embedding: Optional[np.ndarray] = None
    similarity_score: float = 0.0
    context_score: float = 0.0
    combined_score: float = 0.0


class FAISSPhraseIndex:
    """Manages FAISS indices for phrase embeddings"""
    
    def __init__(self, model_manager: Model2VecManager, dimension: int = 256):
        self.model_manager = model_manager
        self.dimension = dimension
        self.faiss_manager = FAISSManager()
        
        # Separate indices for different phrase lengths
        self.phrase_indices = {}  # length -> faiss index
        self.phrase_data = {}     # length -> list of (text, metadata)
        self.cache = get_global_cache()
        
    def add_phrase_prototypes(self, prototypes: List[str], metadata: Optional[List[Dict]] = None):
        """Add phrase prototypes to appropriate indices based on word count"""
        phrases_by_length = defaultdict(list)
        metadata_by_length = defaultdict(list)
        
        for i, phrase in enumerate(prototypes):
            word_count = len(phrase.split())
            phrases_by_length[word_count].append(phrase)
            if metadata:
                metadata_by_length[word_count].append(metadata[i])
            else:
                metadata_by_length[word_count].append({})
        
        # Process each phrase length group
        for length, phrases in phrases_by_length.items():
            if length not in self.phrase_indices:
                # Create new index for this phrase length
                self.phrase_indices[length] = self.faiss_manager.create_auto_index(
                    self.dimension, len(phrases)
                )
                self.phrase_data[length] = []
            
            # Get embeddings
            embeddings = self.model_manager.embed_sync(phrases)
            
            # Add to FAISS index
            self.faiss_manager.add_embeddings(
                self.phrase_indices[length],
                embeddings
            )
            
            # Store phrase data
            self.phrase_data[length].extend(list(zip(phrases, metadata_by_length[length])))
    
    def search_similar_phrases(
        self,
        query: str,
        k: int = 5,
        length_range: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[str, float, Dict]]:
        """Search for similar phrases across indices"""
        query_embedding = self.model_manager.embed_sync([query])[0]
        query_length = len(query.split())
        
        results = []
        
        # Determine which indices to search
        if length_range:
            min_len, max_len = length_range
            lengths_to_search = [l for l in self.phrase_indices.keys() 
                               if min_len <= l <= max_len]
        else:
            # Search phrases with similar lengths (±2 words)
            lengths_to_search = [l for l in self.phrase_indices.keys() 
                               if abs(l - query_length) <= 2]
        
        # Search each relevant index
        for length in lengths_to_search:
            if length not in self.phrase_indices:
                continue
                
            index = self.phrase_indices[length]
            data = self.phrase_data[length]
            
            # Search in FAISS
            distances, indices = self.faiss_manager.search(
                index,
                query_embedding.reshape(1, -1),
                k=min(k, len(data)),
                embeddings=None
            )
            
            # Collect results
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(data):
                    phrase, metadata = data[idx]
                    results.append((phrase, float(dist), metadata))
        
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


class FAISSPhraseMatcher:
    """
    Enhanced phrase matcher using FAISS for intelligent multi-word matching.
    Handles:
    - Variable-length phrase matching
    - Semantic similarity for phrase boundaries
    - Context-aware phrase extraction
    - Efficient similarity search for large phrase vocabularies
    """
    
    def __init__(
        self,
        model_manager: Model2VecManager,
        max_phrase_length: int = 10,
        similarity_threshold: float = 0.3,
        use_context_scoring: bool = True,
        phrase_delimiters: Optional[List[str]] = None
    ):
        """
        Initialize FAISS-enhanced phrase matcher.
        
        Args:
            model_manager: Model manager for embeddings
            max_phrase_length: Maximum words in a phrase
            similarity_threshold: Minimum similarity for matches
            use_context_scoring: Whether to use context scoring
            phrase_delimiters: Custom phrase boundary delimiters
        """
        self.model_manager = model_manager
        self.max_phrase_length = max_phrase_length
        self.similarity_threshold = similarity_threshold
        self.use_context_scoring = use_context_scoring
        self.phrase_delimiters = phrase_delimiters or [
            '.', '!', '?', ',', ';', ':', '(', ')', '[', ']', '{', '}',
            '"', "'", '-', '—', '/', '\\', '|'
        ]
        
        # Initialize FAISS components
        self.phrase_index = FAISSPhraseIndex(model_manager)
        self.known_phrases = set()
        self.cache = get_global_cache()
        
        # Compile delimiter pattern
        self.delimiter_pattern = re.compile(
            '|'.join(re.escape(d) for d in self.phrase_delimiters)
        )
    
    def add_known_phrases(self, phrases: List[str], categories: Optional[List[str]] = None):
        """Add known phrases to the FAISS index"""
        metadata = []
        for i, phrase in enumerate(phrases):
            self.known_phrases.add(phrase.lower())
            meta = {'original': phrase}
            if categories and i < len(categories):
                meta['category'] = categories[i]
            metadata.append(meta)
        
        self.phrase_index.add_phrase_prototypes(phrases, metadata)
    
    def extract_phrase_candidates(
        self,
        text: str,
        start_pos: int,
        min_length: int = 1,
        max_length: Optional[int] = None
    ) -> List[PhraseCandidate]:
        """Extract all possible phrase candidates from a position"""
        max_length = max_length or self.max_phrase_length
        candidates = []
        
        # Find the next delimiter or end of text
        remaining_text = text[start_pos:]
        delimiter_match = self.delimiter_pattern.search(remaining_text)
        
        if delimiter_match:
            boundary_pos = start_pos + delimiter_match.start()
        else:
            boundary_pos = len(text)
        
        # Extract words up to boundary
        words_text = text[start_pos:boundary_pos].strip()
        words = words_text.split()
        
        if not words:
            return candidates
        
        # Generate all possible phrases of different lengths
        for length in range(min_length, min(len(words) + 1, max_length + 1)):
            for i in range(len(words) - length + 1):
                phrase_words = words[i:i + length]
                phrase_text = ' '.join(phrase_words)
                
                # Calculate actual positions
                if i == 0:
                    phrase_start = start_pos
                else:
                    # Find the start position of the i-th word
                    prefix = ' '.join(words[:i])
                    phrase_start = start_pos + len(prefix) + 1
                
                phrase_end = phrase_start + len(phrase_text)
                
                candidate = PhraseCandidate(
                    text=phrase_text,
                    start_pos=phrase_start,
                    end_pos=phrase_end,
                    word_count=length
                )
                candidates.append(candidate)
        
        return candidates
    
    def score_phrase_candidate(
        self,
        candidate: PhraseCandidate,
        context_before: str = "",
        context_after: str = "",
        target_type: Optional[str] = None
    ) -> PhraseCandidate:
        """Score a phrase candidate using semantic similarity and context"""
        # Get embedding
        if candidate.embedding is None:
            candidate.embedding = self.model_manager.embed_sync([candidate.text])[0]
        
        # Search for similar known phrases
        similar_phrases = self.phrase_index.search_similar_phrases(
            candidate.text,
            k=3,
            length_range=(candidate.word_count - 1, candidate.word_count + 1)
        )
        
        if similar_phrases:
            # Use best similarity score
            candidate.similarity_score = similar_phrases[0][1]
        else:
            candidate.similarity_score = 0.0
        
        # Context scoring
        if self.use_context_scoring and (context_before or context_after):
            context_score = self._compute_context_score(
                candidate.text,
                context_before,
                context_after,
                target_type
            )
            candidate.context_score = context_score
        else:
            candidate.context_score = 0.0
        
        # Combined score (weighted average)
        similarity_weight = 0.7
        context_weight = 0.3 if self.use_context_scoring else 0.0
        
        candidate.combined_score = (
            similarity_weight * candidate.similarity_score +
            context_weight * candidate.context_score
        )
        
        return candidate
    
    def _compute_context_score(
        self,
        phrase: str,
        context_before: str,
        context_after: str,
        target_type: Optional[str] = None
    ) -> float:
        """Compute context-based score for phrase candidate"""
        scores = []
        
        # Check if phrase appears in known phrases
        if phrase.lower() in self.known_phrases:
            scores.append(1.0)
        
        # Semantic coherence with context
        if context_before:
            # Check semantic similarity between phrase and preceding context
            context_embedding = self.model_manager.embed_sync([context_before])[0]
            phrase_embedding = self.model_manager.embed_sync([phrase])[0]
            coherence = self.model_manager.cosine_similarity(context_embedding, phrase_embedding)
            scores.append(float(coherence))
        
        # Type matching if specified
        if target_type:
            type_embedding = self.model_manager.embed_sync([target_type])[0]
            phrase_embedding = self.model_manager.embed_sync([phrase])[0]
            type_similarity = self.model_manager.cosine_similarity(type_embedding, phrase_embedding)
            scores.append(float(type_similarity))
        
        # Average all scores
        return np.mean(scores) if scores else 0.0
    
    def find_best_phrase_match(
        self,
        text: str,
        start_pos: int,
        parameter_name: Optional[str] = None,
        min_similarity: Optional[float] = None
    ) -> Optional[PhraseCandidate]:
        """Find the best phrase match starting from a position"""
        min_similarity = min_similarity or self.similarity_threshold
        
        # Extract context
        context_before = text[max(0, start_pos - 50):start_pos].strip()
        context_after = text[start_pos:min(len(text), start_pos + 50)].strip()
        
        # Get all candidates
        candidates = self.extract_phrase_candidates(text, start_pos)
        
        if not candidates:
            return None
        
        # Score all candidates
        scored_candidates = []
        for candidate in candidates:
            scored = self.score_phrase_candidate(
                candidate,
                context_before,
                context_after,
                parameter_name
            )
            
            # Only keep candidates above threshold
            if scored.combined_score >= min_similarity:
                scored_candidates.append(scored)
        
        if not scored_candidates:
            return None
        
        # Sort by combined score and prefer longer phrases for ties
        scored_candidates.sort(
            key=lambda c: (c.combined_score, c.word_count),
            reverse=True
        )
        
        return scored_candidates[0]
    
    def match_phrase_in_text(
        self,
        text: str,
        pattern_pos: int,
        parameter_name: Optional[str] = None
    ) -> Optional[Tuple[str, int, int]]:
        """
        Match a phrase in text at the given position.
        
        Returns:
            Tuple of (matched_phrase, start_pos, end_pos) or None
        """
        best_match = self.find_best_phrase_match(
            text,
            pattern_pos,
            parameter_name
        )
        
        if best_match:
            return (best_match.text, best_match.start_pos, best_match.end_pos)
        
        return None
    
    def extract_all_phrases(
        self,
        text: str,
        min_confidence: float = 0.5
    ) -> List[PhraseCandidate]:
        """Extract all high-confidence phrases from text"""
        phrases = []
        words = text.split()
        
        i = 0
        while i < len(words):
            # Find position of current word in text
            word_start = text.find(words[i], sum(len(w) + 1 for w in words[:i]))
            
            if word_start >= 0:
                best_match = self.find_best_phrase_match(
                    text,
                    word_start,
                    min_similarity=min_confidence
                )
                
                if best_match:
                    phrases.append(best_match)
                    # Skip words that were part of the matched phrase
                    i += best_match.word_count
                    continue
            
            i += 1
        
        return phrases
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get matcher statistics"""
        stats = {
            'num_known_phrases': len(self.known_phrases),
            'max_phrase_length': self.max_phrase_length,
            'similarity_threshold': self.similarity_threshold,
            'use_context_scoring': self.use_context_scoring,
            'phrase_indices': {}
        }
        
        for length, index in self.phrase_index.phrase_indices.items():
            stats['phrase_indices'][f'{length}_words'] = {
                'num_phrases': len(self.phrase_index.phrase_data[length]),
                'index_type': self.phrase_index.faiss_manager.get_index_type(index)
            }
        
        return stats