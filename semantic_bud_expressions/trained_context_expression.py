"""
Trained context-aware expression that uses learned parameters for optimal matching.
"""

import json
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from .context_aware_expression import ContextAwareExpression, ContextMatch
from .context_aware_trainer import TrainedParameters
from .chunking_strategies import ChunkedContextMatcher
from .unified_registry import UnifiedParameterTypeRegistry


class TrainedContextMatch(ContextMatch):
    """Enhanced context match with confidence scores and training metadata"""
    
    def __init__(
        self,
        matched_text: str,
        parameters: Dict[str, Any],
        start_pos: int,
        end_pos: int,
        context_text: str,
        context_similarity: float,
        full_text: str,
        confidence: float = 1.0,
        training_metadata: Optional[Dict] = None
    ):
        super().__init__(
            matched_text=matched_text,
            parameters=parameters,
            start_pos=start_pos,
            end_pos=end_pos,
            context_text=context_text,
            context_similarity=context_similarity,
            full_text=full_text
        )
        self.confidence = confidence
        self.training_metadata = training_metadata or {}
    
    def __repr__(self):
        return (
            f"TrainedContextMatch(text='{self.matched_text}', "
            f"similarity={self.context_similarity:.2f}, "
            f"confidence={self.confidence:.2f})"
        )


class TrainedContextAwareExpression:
    """
    Context-aware expression using trained parameters for optimal performance.
    
    This class uses pre-trained parameters (threshold, window size, chunking strategy)
    to provide optimal context-aware matching for specific domains.
    """
    
    def __init__(
        self,
        trained_params: TrainedParameters,
        registry: Optional[UnifiedParameterTypeRegistry] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize with trained parameters.
        
        Args:
            trained_params: Parameters learned from training
            registry: Parameter type registry
            confidence_threshold: Minimum confidence for high-confidence matches
        """
        self.trained_params = trained_params
        self.registry = registry or UnifiedParameterTypeRegistry()
        self.confidence_threshold = confidence_threshold
        
        if not self.registry.model_manager:
            self.registry.initialize_model()
        
        # Initialize the appropriate matcher based on strategy
        if trained_params.optimal_strategy == 'single':
            self.matcher = ContextAwareExpression(
                expression=trained_params.expression,
                expected_context=trained_params.expected_context,
                context_threshold=trained_params.optimal_threshold,
                context_window=trained_params.optimal_window,
                registry=self.registry
            )
            self.use_chunking = False
        else:
            self.matcher = ChunkedContextMatcher(
                expression=trained_params.expression,
                expected_context=trained_params.expected_context,
                registry=self.registry,
                chunking_strategy=trained_params.optimal_strategy,
                window_size=trained_params.optimal_window
            )
            self.use_chunking = True
    
    @classmethod
    def from_file(
        cls,
        filepath: str,
        registry: Optional[UnifiedParameterTypeRegistry] = None
    ):
        """Load trained expression from file"""
        trained_params = TrainedParameters.load(filepath)
        return cls(trained_params, registry)
    
    def match_with_context(
        self,
        text: str,
        return_confidence: bool = True
    ) -> Optional[TrainedContextMatch]:
        """
        Match text using trained parameters.
        
        Args:
            text: Text to match
            return_confidence: Whether to calculate confidence scores
            
        Returns:
            TrainedContextMatch if successful, None otherwise
        """
        
        if self.use_chunking:
            result, chunks, combined_similarity = self.matcher.match_with_chunks(
                text, self.trained_params.optimal_threshold
            )
            
            if result and result['match']:
                base_match = result['match']
                
                # Calculate confidence based on multiple factors
                confidence = self._calculate_confidence(
                    combined_similarity,
                    result.get('chunk_count', 1),
                    result.get('best_chunk_similarity', combined_similarity)
                )
                
                return TrainedContextMatch(
                    matched_text=base_match.matched_text,
                    parameters=base_match.parameters,
                    start_pos=base_match.start_pos,
                    end_pos=base_match.end_pos,
                    context_text=base_match.context_text,
                    context_similarity=combined_similarity,
                    full_text=base_match.full_text,
                    confidence=confidence,
                    training_metadata={
                        'trained_threshold': self.trained_params.optimal_threshold,
                        'trained_strategy': self.trained_params.optimal_strategy,
                        'chunk_count': result.get('chunk_count', 1),
                        'training_metrics': self.trained_params.performance_metrics
                    }
                )
        else:
            base_match = self.matcher.match_with_context(text)
            
            if base_match:
                confidence = self._calculate_confidence(
                    base_match.context_similarity,
                    chunk_count=1,
                    best_chunk_similarity=base_match.context_similarity
                )
                
                return TrainedContextMatch(
                    matched_text=base_match.matched_text,
                    parameters=base_match.parameters,
                    start_pos=base_match.start_pos,
                    end_pos=base_match.end_pos,
                    context_text=base_match.context_text,
                    context_similarity=base_match.context_similarity,
                    full_text=base_match.full_text,
                    confidence=confidence,
                    training_metadata={
                        'trained_threshold': self.trained_params.optimal_threshold,
                        'trained_strategy': self.trained_params.optimal_strategy,
                        'training_metrics': self.trained_params.performance_metrics
                    }
                )
        
        return None
    
    def find_all_with_context(
        self,
        text: str,
        max_matches: int = 10
    ) -> List[TrainedContextMatch]:
        """Find all matches in text using trained parameters"""
        
        matches = []
        remaining_text = text
        offset = 0
        
        while remaining_text and len(matches) < max_matches:
            match = self.match_with_context(remaining_text)
            
            if match:
                # Adjust positions for original text
                match.start_pos += offset
                match.end_pos += offset
                matches.append(match)
                
                # Move past this match
                next_start = match.end_pos - offset
                remaining_text = remaining_text[next_start:]
                offset += next_start
            else:
                break
        
        return matches
    
    def _calculate_confidence(
        self,
        similarity: float,
        chunk_count: int = 1,
        best_chunk_similarity: float = 0.0
    ) -> float:
        """
        Calculate confidence score based on multiple factors.
        
        Args:
            similarity: Combined similarity score
            chunk_count: Number of chunks used
            best_chunk_similarity: Highest individual chunk similarity
            
        Returns:
            Confidence score between 0 and 1
        """
        
        # Base confidence from similarity relative to threshold
        threshold_margin = (similarity - self.trained_params.optimal_threshold) / (1.0 - self.trained_params.optimal_threshold)
        threshold_margin = max(0, min(1, threshold_margin))  # Clamp to [0, 1]
        
        # Training performance confidence
        training_f1 = self.trained_params.performance_metrics.get('f1', 0.5)
        training_confidence = training_f1
        
        # Chunk consistency confidence (only for chunked strategies)
        chunk_confidence = 1.0
        if chunk_count > 1:
            # Higher confidence if best chunk is similar to combined score
            chunk_consistency = 1.0 - abs(similarity - best_chunk_similarity)
            chunk_confidence = 0.7 + 0.3 * chunk_consistency
        
        # Combine confidence factors
        confidence = (
            0.4 * threshold_margin +
            0.3 * training_confidence +
            0.3 * chunk_confidence
        )
        
        return min(1.0, max(0.0, confidence))
    
    def get_prediction_explanation(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Get detailed explanation of why the model made its prediction.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with detailed explanation
        """
        
        explanation = {
            'input_text': text[:200] + '...' if len(text) > 200 else text,
            'trained_parameters': {
                'expression': self.trained_params.expression,
                'expected_context': self.trained_params.expected_context,
                'optimal_threshold': self.trained_params.optimal_threshold,
                'optimal_window': self.trained_params.optimal_window,
                'optimal_strategy': self.trained_params.optimal_strategy
            },
            'training_performance': self.trained_params.performance_metrics,
            'prediction': None,
            'reasoning': []
        }
        
        # Get match and detailed analysis
        if self.use_chunking:
            result, chunks, combined_similarity = self.matcher.match_with_chunks(
                text, self.trained_params.optimal_threshold
            )
            
            explanation['chunking_analysis'] = self.matcher.explain_chunks(
                text, self.trained_params.optimal_threshold
            )
            
            if result:
                match = self.match_with_context(text)
                explanation['prediction'] = {
                    'matched': True,
                    'similarity': combined_similarity,
                    'confidence': match.confidence if match else 0.0,
                    'parameters': match.parameters if match else {}
                }
                
                explanation['reasoning'].append(
                    f"Combined similarity {combined_similarity:.3f} exceeds trained threshold {self.trained_params.optimal_threshold:.3f}"
                )
                explanation['reasoning'].append(
                    f"Used chunking strategy '{self.trained_params.optimal_strategy}' with {len(chunks)} chunks"
                )
            else:
                explanation['prediction'] = {
                    'matched': False,
                    'similarity': combined_similarity,
                    'confidence': 0.0
                }
                
                explanation['reasoning'].append(
                    f"Combined similarity {combined_similarity:.3f} below trained threshold {self.trained_params.optimal_threshold:.3f}"
                )
        
        else:
            match = self.matcher.match_with_context(text)
            
            if match:
                trained_match = self.match_with_context(text)
                explanation['prediction'] = {
                    'matched': True,
                    'similarity': match.context_similarity,
                    'confidence': trained_match.confidence if trained_match else 0.0,
                    'parameters': match.parameters
                }
                
                explanation['reasoning'].append(
                    f"Context similarity {match.context_similarity:.3f} exceeds trained threshold {self.trained_params.optimal_threshold:.3f}"
                )
                explanation['reasoning'].append(
                    f"Context extracted: '{match.context_text[:100]}...'"
                )
            else:
                # Try to get similarity even if below threshold
                temp_expr = ContextAwareExpression(
                    expression=self.trained_params.expression,
                    expected_context=self.trained_params.expected_context,
                    context_threshold=0.0,
                    context_window=self.trained_params.optimal_window,
                    registry=self.registry
                )
                temp_match = temp_expr.match_with_context(text)
                similarity = temp_match.context_similarity if temp_match else 0.0
                
                explanation['prediction'] = {
                    'matched': False,
                    'similarity': similarity,
                    'confidence': 0.0
                }
                
                explanation['reasoning'].append(
                    f"Context similarity {similarity:.3f} below trained threshold {self.trained_params.optimal_threshold:.3f}"
                )
        
        return explanation
    
    def is_high_confidence(self, match: Optional[TrainedContextMatch]) -> bool:
        """Check if a match has high confidence"""
        return match is not None and match.confidence >= self.confidence_threshold
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process and results"""
        return {
            'expression': self.trained_params.expression,
            'expected_context': self.trained_params.expected_context,
            'optimal_parameters': {
                'threshold': self.trained_params.optimal_threshold,
                'window': self.trained_params.optimal_window,
                'strategy': self.trained_params.optimal_strategy
            },
            'performance_metrics': self.trained_params.performance_metrics,
            'training_timestamp': self.trained_params.training_timestamp,
            'confidence_threshold': self.confidence_threshold
        }


# Convenience functions for common use cases
def load_trained_expression(
    filepath: str,
    registry: Optional[UnifiedParameterTypeRegistry] = None
) -> TrainedContextAwareExpression:
    """Load a trained expression from file"""
    return TrainedContextAwareExpression.from_file(filepath, registry)


def create_trained_expression(
    expression: str,
    expected_context: str,
    positive_examples: List[str],
    negative_examples: List[str],
    registry: Optional[UnifiedParameterTypeRegistry] = None
) -> TrainedContextAwareExpression:
    """Train and create a new expression in one step"""
    from .context_aware_trainer import ContextAwareTrainer
    
    trainer = ContextAwareTrainer(registry)
    trained_params = trainer.train(
        expression=expression,
        expected_context=expected_context,
        positive_examples=positive_examples,
        negative_examples=negative_examples
    )
    
    return TrainedContextAwareExpression(trained_params, registry)