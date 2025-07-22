"""
Chunking strategies for context-aware matching with Naive Bayes combination.
"""

import re
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from .unified_registry import UnifiedParameterTypeRegistry
from .context_aware_expression import ContextAwareExpression


class TextChunk:
    """Represents a chunk of text with metadata"""
    
    def __init__(self, text: str, start_pos: int, end_pos: int, chunk_type: str = 'content'):
        self.text = text
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.chunk_type = chunk_type
        self.similarity_score = 0.0
        self.contains_pattern = False
    
    def __repr__(self):
        return f"TextChunk(text='{self.text[:30]}...', type={self.chunk_type}, similarity={self.similarity_score:.3f})"


class ChunkingProcessor:
    """Processes text into chunks using different strategies"""
    
    def __init__(self, registry: Optional[UnifiedParameterTypeRegistry] = None):
        self.registry = registry or UnifiedParameterTypeRegistry()
    
    def create_chunks(
        self,
        text: str,
        strategy: str,
        window_size: Union[int, str] = 'auto',
        overlap_ratio: float = 0.3
    ) -> List[TextChunk]:
        """Create chunks using the specified strategy"""
        
        if strategy == 'single':
            return [TextChunk(text, 0, len(text), 'full')]
        elif strategy == 'sliding':
            return self._sliding_window_chunks(text, window_size, overlap_ratio)
        elif strategy == 'sentences':
            return self._sentence_chunks(text)
        elif strategy == 'overlapping':
            return self._overlapping_chunks(text, window_size, overlap_ratio)
        elif strategy == 'semantic_boundary':
            return self._semantic_boundary_chunks(text, window_size)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _sliding_window_chunks(
        self,
        text: str,
        window_size: Union[int, str],
        overlap_ratio: float = 0.3
    ) -> List[TextChunk]:
        """Create sliding window chunks with overlap"""
        
        # Determine window size
        if isinstance(window_size, str):
            if window_size == 'auto':
                window_size = min(100, len(text.split()) // 3)  # 1/3 of text, max 100 words
            else:
                window_size = 50  # Default
        
        words = text.split()
        if len(words) <= window_size:
            return [TextChunk(text, 0, len(text), 'sliding')]
        
        chunks = []
        step_size = max(1, int(window_size * (1 - overlap_ratio)))
        
        for start in range(0, len(words), step_size):
            end = min(start + window_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character positions
            char_start = text.find(chunk_words[0]) if chunk_words else 0
            char_end = char_start + len(chunk_text)
            
            chunks.append(TextChunk(chunk_text, char_start, char_end, 'sliding'))
            
            if end >= len(words):
                break
        
        return chunks
    
    def _sentence_chunks(self, text: str) -> List[TextChunk]:
        """Create chunks based on sentence boundaries"""
        # Split by common sentence delimiters
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [TextChunk(text, 0, len(text), 'sentence')]
        
        chunks = []
        current_pos = 0
        
        for sentence in sentences:
            start_pos = text.find(sentence, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(sentence)
            chunks.append(TextChunk(sentence, start_pos, end_pos, 'sentence'))
            current_pos = end_pos
        
        return chunks
    
    def _overlapping_chunks(
        self,
        text: str,
        window_size: Union[int, str],
        overlap_ratio: float = 0.5
    ) -> List[TextChunk]:
        """Create overlapping chunks with higher overlap for better coverage"""
        
        if isinstance(window_size, str):
            window_size = min(75, len(text.split()) // 4)
        
        words = text.split()
        if len(words) <= window_size:
            return [TextChunk(text, 0, len(text), 'overlapping')]
        
        chunks = []
        step_size = max(1, int(window_size * (1 - overlap_ratio)))
        
        # Create overlapping windows
        for start in range(0, len(words), step_size):
            end = min(start + window_size, len(words))
            
            # Ensure we don't miss the end of the text
            if start > 0 and end == len(words):
                start = max(0, len(words) - window_size)
            
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            char_start = 0
            char_end = len(chunk_text)
            
            chunks.append(TextChunk(chunk_text, char_start, char_end, 'overlapping'))
            
            if end >= len(words):
                break
        
        return chunks
    
    def _semantic_boundary_chunks(
        self,
        text: str,
        window_size: Union[int, str]
    ) -> List[TextChunk]:
        """Create chunks based on semantic boundaries (paragraphs, topic shifts)"""
        
        # Split by paragraph boundaries first
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if len(paragraphs) <= 1:
            # No clear paragraph structure, fall back to sentence chunking
            return self._sentence_chunks(text)
        
        chunks = []
        current_pos = 0
        
        for paragraph in paragraphs:
            start_pos = text.find(paragraph, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(paragraph)
            
            # If paragraph is too long, sub-chunk it
            if isinstance(window_size, int) and len(paragraph.split()) > window_size:
                sub_chunks = self._sliding_window_chunks(paragraph, window_size)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_type = 'semantic_boundary'
                    chunks.append(sub_chunk)
            else:
                chunks.append(TextChunk(paragraph, start_pos, end_pos, 'semantic_boundary'))
            
            current_pos = end_pos
        
        return chunks


class NaiveBayesChunkCombiner:
    """Combines chunk similarities using Naive Bayes approach"""
    
    @staticmethod
    def combine_similarities(
        chunk_similarities: List[float],
        combination_method: str = 'geometric_mean',
        weights: Optional[List[float]] = None
    ) -> float:
        """
        Combine multiple chunk similarities using different methods.
        
        Args:
            chunk_similarities: Similarity scores from different chunks
            combination_method: Method to combine scores
            weights: Optional weights for chunks (must sum to 1)
            
        Returns:
            Combined similarity score
        """
        
        if not chunk_similarities:
            return 0.0
        
        # Filter out zero similarities for geometric operations
        non_zero_sims = [max(s, 0.001) for s in chunk_similarities]
        
        if combination_method == 'geometric_mean':
            return np.power(np.prod(non_zero_sims), 1.0 / len(non_zero_sims))
        
        elif combination_method == 'arithmetic_mean':
            if weights:
                return np.average(chunk_similarities, weights=weights)
            return np.mean(chunk_similarities)
        
        elif combination_method == 'weighted_geometric':
            if not weights:
                weights = [1.0 / len(chunk_similarities)] * len(chunk_similarities)
            
            weighted_product = np.prod([s ** w for s, w in zip(non_zero_sims, weights)])
            return weighted_product
        
        elif combination_method == 'max':
            return np.max(chunk_similarities)
        
        elif combination_method == 'harmonic_mean':
            return len(chunk_similarities) / np.sum([1.0 / s for s in non_zero_sims])
        
        elif combination_method == 'naive_bayes':
            # P(match|all_chunks) ∝ ∏ P(match|chunk_i)
            # Using log probabilities to avoid underflow
            log_probs = [np.log(max(s, 0.001)) for s in chunk_similarities]
            combined_log_prob = np.sum(log_probs) / len(log_probs)
            return min(np.exp(combined_log_prob), 1.0)
        
        else:
            # Default to arithmetic mean
            return np.mean(chunk_similarities)
    
    @staticmethod
    def calculate_chunk_weights(
        chunks: List[TextChunk],
        pattern_matches: List[bool],
        text_lengths: List[int]
    ) -> List[float]:
        """
        Calculate weights for chunks based on various factors.
        
        Args:
            chunks: List of text chunks
            pattern_matches: Whether each chunk contains the pattern
            text_lengths: Length of each chunk in words
            
        Returns:
            Normalized weights for chunks
        """
        
        weights = []
        
        for i, chunk in enumerate(chunks):
            weight = 1.0
            
            # Weight by text length (longer chunks get more weight)
            length_weight = np.log(max(text_lengths[i], 1)) / np.log(max(max(text_lengths), 1))
            weight *= (0.5 + 0.5 * length_weight)
            
            # Weight by pattern presence (chunks with patterns get more weight)
            if pattern_matches[i]:
                weight *= 2.0
            
            # Weight by chunk position (middle chunks often more important)
            position_weight = 1.0 - abs(i - len(chunks) / 2) / (len(chunks) / 2) if len(chunks) > 1 else 1.0
            weight *= (0.7 + 0.3 * position_weight)
            
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        return weights


class ChunkedContextMatcher:
    """Context matcher using chunking strategies with Naive Bayes combination"""
    
    def __init__(
        self,
        expression: str,
        expected_context: str,
        registry: Optional[UnifiedParameterTypeRegistry] = None,
        chunking_strategy: str = 'sliding',
        window_size: Union[int, str] = 'auto',
        combination_method: str = 'naive_bayes',
        chunk_threshold: float = 0.3
    ):
        """
        Initialize chunked context matcher.
        
        Args:
            expression: Pattern expression
            expected_context: Expected semantic context
            registry: Parameter registry
            chunking_strategy: How to split text into chunks
            window_size: Size of chunks
            combination_method: How to combine chunk scores
            chunk_threshold: Minimum threshold for individual chunks
        """
        
        self.expression = expression
        self.expected_context = expected_context
        self.registry = registry or UnifiedParameterTypeRegistry()
        self.chunking_strategy = chunking_strategy
        self.window_size = window_size
        self.combination_method = combination_method
        self.chunk_threshold = chunk_threshold
        
        self.processor = ChunkingProcessor(self.registry)
        self.combiner = NaiveBayesChunkCombiner()
        
        if not self.registry.model_manager:
            self.registry.initialize_model()
    
    def match_with_chunks(
        self,
        text: str,
        final_threshold: float = 0.5
    ) -> Tuple[Optional[Dict], List[TextChunk], float]:
        """
        Match text using chunking strategy.
        
        Returns:
            Tuple of (match_result, chunks_with_scores, combined_similarity)
        """
        
        # Create chunks
        chunks = self.processor.create_chunks(
            text, self.chunking_strategy, self.window_size
        )
        
        if not chunks:
            return None, [], 0.0
        
        # Evaluate each chunk
        chunk_similarities = []
        pattern_matches = []
        text_lengths = []
        
        for chunk in chunks:
            expr = ContextAwareExpression(
                expression=self.expression,
                expected_context=self.expected_context,
                context_threshold=0.0,  # Get raw similarities
                context_window='auto',
                registry=self.registry
            )
            
            match = expr.match_with_context(chunk.text)
            similarity = match.context_similarity if match else 0.0
            
            chunk.similarity_score = similarity
            chunk.contains_pattern = match is not None
            
            chunk_similarities.append(similarity)
            pattern_matches.append(match is not None)
            text_lengths.append(len(chunk.text.split()))
        
        # Calculate chunk weights
        weights = self.combiner.calculate_chunk_weights(
            chunks, pattern_matches, text_lengths
        )
        
        # Combine similarities
        combined_similarity = self.combiner.combine_similarities(
            chunk_similarities, 
            self.combination_method,
            weights if self.combination_method in ['arithmetic_mean', 'weighted_geometric'] else None
        )
        
        # Determine if it passes final threshold
        if combined_similarity >= final_threshold:
            # Find the best matching chunk for parameter extraction
            best_chunk_idx = np.argmax(chunk_similarities)
            best_chunk = chunks[best_chunk_idx]
            
            expr = ContextAwareExpression(
                expression=self.expression,
                expected_context=self.expected_context,
                context_threshold=0.0,
                registry=self.registry
            )
            
            best_match = expr.match_with_context(best_chunk.text)
            
            if best_match:
                # Enhance match with combined similarity
                best_match.context_similarity = combined_similarity
                return {
                    'match': best_match,
                    'combined_similarity': combined_similarity,
                    'chunk_count': len(chunks),
                    'best_chunk_similarity': chunk_similarities[best_chunk_idx]
                }, chunks, combined_similarity
        
        return None, chunks, combined_similarity
    
    def explain_chunks(
        self,
        text: str,
        final_threshold: float = 0.5
    ) -> Dict:
        """
        Provide detailed explanation of chunking and matching process.
        """
        
        result, chunks, combined_sim = self.match_with_chunks(text, final_threshold)
        
        explanation = {
            'text_length': len(text),
            'chunk_count': len(chunks),
            'chunking_strategy': self.chunking_strategy,
            'combination_method': self.combination_method,
            'combined_similarity': combined_sim,
            'passes_threshold': combined_sim >= final_threshold,
            'threshold': final_threshold,
            'chunk_details': []
        }
        
        for i, chunk in enumerate(chunks):
            explanation['chunk_details'].append({
                'chunk_index': i,
                'text_preview': chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text,
                'similarity': chunk.similarity_score,
                'contains_pattern': chunk.contains_pattern,
                'chunk_type': chunk.chunk_type,
                'length_words': len(chunk.text.split())
            })
        
        return explanation