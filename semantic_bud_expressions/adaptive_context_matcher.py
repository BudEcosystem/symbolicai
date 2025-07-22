"""
Adaptive context matcher that works well without training but excels with training data.
Handles context length mismatches, perspective differences, and automatic parameter tuning.
"""

import re
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from .context_aware_expression import ContextAwareExpression, ContextMatch
from .unified_registry import UnifiedParameterTypeRegistry
from .chunking_strategies import ChunkingProcessor, NaiveBayesChunkCombiner
from .context_aware_trainer import ContextAwareTrainer, TrainingConfig


@dataclass
class ContextNormalizationConfig:
    """Configuration for context normalization"""
    perspective_normalization: bool = True
    length_normalization: bool = True
    semantic_expansion: bool = True
    fallback_strategies: List[str] = None
    
    def __post_init__(self):
        if self.fallback_strategies is None:
            self.fallback_strategies = ['keyword_matching', 'partial_similarity', 'fuzzy_matching']


class PerspectiveNormalizer:
    """Handles perspective normalization between contexts"""
    
    PERSPECTIVE_MAPPINGS = {
        # Second person to third person
        r'\byou\b': 'user',
        r'\byour\b': 'user',
        r'\byours\b': 'user',
        r'\byourself\b': 'user',
        
        # First person to third person
        r'\bi\b': 'user',
        r'\bme\b': 'user', 
        r'\bmy\b': 'user',
        r'\bmine\b': 'user',
        r'\bmyself\b': 'user',
        
        # Action perspective normalization
        r'\bdo\b': 'perform',
        r'\bmake\b': 'create',
        r'\bget\b': 'obtain',
        r'\bhave\b': 'possess',
        r'\bwant\b': 'need',
        r'\blike\b': 'prefer'
    }
    
    @staticmethod
    def normalize_perspective(text: str) -> str:
        """Normalize perspective in text"""
        normalized = text.lower()
        
        for pattern, replacement in PerspectiveNormalizer.PERSPECTIVE_MAPPINGS.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    @staticmethod
    def generate_perspective_variants(text: str) -> List[str]:
        """Generate different perspective variants of the text"""
        variants = [text]
        
        # Original normalized
        normalized = PerspectiveNormalizer.normalize_perspective(text)
        if normalized != text.lower():
            variants.append(normalized)
        
        # First person variant
        first_person = text.lower()
        first_person = re.sub(r'\buser\b', 'I', first_person)
        first_person = re.sub(r'\bthe user\b', 'I', first_person)
        variants.append(first_person)
        
        # Second person variant  
        second_person = text.lower()
        second_person = re.sub(r'\buser\b', 'you', second_person)
        second_person = re.sub(r'\bthe user\b', 'you', second_person)
        variants.append(second_person)
        
        return list(set(variants))  # Remove duplicates


class SemanticExpander:
    """Expands short contexts with semantically related terms"""
    
    DOMAIN_EXPANSIONS = {
        'medical': ['health', 'healthcare', 'clinical', 'treatment', 'diagnosis', 'patient', 'doctor', 'hospital'],
        'financial': ['banking', 'money', 'payment', 'transaction', 'account', 'funds', 'finance', 'bank'],
        'shopping': ['purchase', 'buy', 'cart', 'store', 'retail', 'checkout', 'order', 'ecommerce'],
        'legal': ['contract', 'agreement', 'law', 'court', 'attorney', 'jurisdiction', 'legal', 'compliance'],
        'tech': ['software', 'system', 'application', 'computer', 'technology', 'digital', 'technical'],
        'education': ['school', 'learning', 'student', 'teacher', 'academic', 'education', 'classroom']
    }
    
    COMMON_EXPANSIONS = {
        'help': ['assist', 'support', 'aid', 'guidance'],
        'problem': ['issue', 'error', 'difficulty', 'trouble'],
        'need': ['require', 'want', 'necessity'],
        'quick': ['fast', 'rapid', 'immediate', 'urgent'],
        'good': ['excellent', 'great', 'positive', 'beneficial'],
        'bad': ['poor', 'negative', 'problematic', 'concerning']
    }
    
    @classmethod
    def expand_context(cls, context: str, max_additions: int = 5) -> str:
        """Expand short context with related terms"""
        words = context.lower().split()
        
        if len(words) >= 8:  # Already long enough
            return context
        
        additions = []
        
        # Check for domain-specific expansions
        for domain, expansions in cls.DOMAIN_EXPANSIONS.items():
            if any(word in words for word in expansions[:3]):  # If matches domain
                additions.extend(expansions[:max_additions])
                break
        
        # Add common expansions
        for word in words:
            if word in cls.COMMON_EXPANSIONS:
                additions.extend(cls.COMMON_EXPANSIONS[word][:2])
        
        # Limit additions
        additions = list(set(additions))[:max_additions]
        
        if additions:
            expanded = context + " " + " ".join(additions)
            return expanded
        
        return context


class FallbackMatcher:
    """Provides fallback matching strategies when semantic matching fails"""
    
    @staticmethod
    def keyword_overlap_score(context: str, target: str) -> float:
        """Calculate keyword overlap score"""
        context_words = set(context.lower().split())
        target_words = set(target.lower().split())
        
        if not target_words:
            return 0.0
        
        overlap = context_words & target_words
        return len(overlap) / len(target_words)
    
    @staticmethod
    def fuzzy_similarity(context: str, target: str) -> float:
        """Calculate fuzzy string similarity"""
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, context.lower(), target.lower()).ratio()
        except:
            # Simple fallback
            return FallbackMatcher.keyword_overlap_score(context, target)
    
    @staticmethod
    def partial_matching_score(context: str, target: str) -> float:
        """Score based on partial phrase matching"""
        target_phrases = target.lower().split()
        context_lower = context.lower()
        
        matches = sum(1 for phrase in target_phrases if phrase in context_lower)
        return matches / len(target_phrases) if target_phrases else 0.0


class AdaptiveContextMatcher:
    """
    Adaptive context matcher that works without training but improves significantly with it.
    
    Key features:
    - Automatic parameter selection for untrained use
    - Perspective normalization (you/your → user context)
    - Context length handling (long input vs short target)
    - Fallback strategies for robustness
    - Training enhancement for optimal performance
    """
    
    def __init__(
        self,
        expression: str,
        target_context: str,
        registry: Optional[UnifiedParameterTypeRegistry] = None,
        config: Optional[ContextNormalizationConfig] = None
    ):
        """
        Initialize adaptive matcher.
        
        Args:
            expression: Pattern to match (e.g., "I need help with {issue}")
            target_context: Target context (e.g., "technical support help")
            registry: Parameter registry
            config: Normalization configuration
        """
        self.expression = expression
        self.original_target_context = target_context
        self.registry = registry or UnifiedParameterTypeRegistry()
        self.config = config or ContextNormalizationConfig()
        
        if not self.registry.model_manager:
            self.registry.initialize_model()
        
        # Initialize components
        self.perspective_normalizer = PerspectiveNormalizer()
        self.semantic_expander = SemanticExpander()
        self.fallback_matcher = FallbackMatcher()
        self.chunking_processor = ChunkingProcessor(self.registry)
        
        # Process target context
        self.processed_target_contexts = self._process_target_context(target_context)
        
        # Auto-configure if no training provided
        self.is_trained = False
        self.trained_params = None
        self.base_matchers = self._create_base_matchers()
    
    def _process_target_context(self, target_context: str) -> Dict[str, str]:
        """Process target context with various normalization strategies"""
        processed = {
            'original': target_context,
            'normalized': target_context,
            'expanded': target_context,
            'perspective_variants': []
        }
        
        if self.config.perspective_normalization:
            processed['normalized'] = self.perspective_normalizer.normalize_perspective(target_context)
            processed['perspective_variants'] = self.perspective_normalizer.generate_perspective_variants(target_context)
        
        if self.config.semantic_expansion:
            processed['expanded'] = self.semantic_expander.expand_context(processed['normalized'])
        
        return processed
    
    def _create_base_matchers(self) -> Dict[str, ContextAwareExpression]:
        """Create base matchers with different configurations for untrained use"""
        matchers = {}
        
        # Conservative matcher (high precision)
        matchers['conservative'] = ContextAwareExpression(
            expression=self.expression,
            expected_context=self.processed_target_contexts['expanded'],
            context_threshold=0.6,  # Higher threshold
            context_window='sentence',
            registry=self.registry
        )
        
        # Liberal matcher (high recall)  
        matchers['liberal'] = ContextAwareExpression(
            expression=self.expression,
            expected_context=self.processed_target_contexts['expanded'],
            context_threshold=0.35,  # Lower threshold
            context_window='auto',
            registry=self.registry
        )
        
        # Balanced matcher
        matchers['balanced'] = ContextAwareExpression(
            expression=self.expression,
            expected_context=self.processed_target_contexts['expanded'],
            context_threshold=0.45,  # Medium threshold
            context_window='auto',
            registry=self.registry
        )
        
        return matchers
    
    def train(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        optimize_for: str = 'balanced'  # 'precision', 'recall', 'balanced'
    ) -> Dict[str, Any]:
        """
        Train the matcher on provided examples.
        
        Args:
            positive_examples: Examples that should match
            negative_examples: Examples that should not match
            optimize_for: Optimization objective
            
        Returns:
            Training results and performance metrics
        """
        print(f"Training adaptive matcher for: {self.expression}")
        print(f"Target context: {self.original_target_context}")
        print(f"Optimization: {optimize_for}")
        
        # Create trainer
        trainer = ContextAwareTrainer(self.registry)
        
        # Configure training based on optimization goal
        if optimize_for == 'precision':
            config = TrainingConfig(
                optimization_metric='precision',
                threshold_range=(0.4, 0.8),
                chunking_strategies=['single', 'sentences']
            )
        elif optimize_for == 'recall':
            config = TrainingConfig(
                optimization_metric='recall', 
                threshold_range=(0.25, 0.6),
                chunking_strategies=['single', 'sliding', 'overlapping']
            )
        else:  # balanced
            config = TrainingConfig(
                optimization_metric='f1',
                threshold_range=(0.3, 0.7),
                chunking_strategies=['single', 'sliding', 'sentences']
            )
        
        # Train on expanded context for better generalization
        self.trained_params = trainer.train(
            expression=self.expression,
            expected_context=self.processed_target_contexts['expanded'],
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            config=config
        )
        
        self.is_trained = True
        
        # Create trained matcher
        from .trained_context_expression import TrainedContextAwareExpression
        self.trained_matcher = TrainedContextAwareExpression(
            self.trained_params, self.registry
        )
        
        return {
            'trained_params': self.trained_params,
            'performance': self.trained_params.performance_metrics,
            'optimal_threshold': self.trained_params.optimal_threshold,
            'optimal_strategy': self.trained_params.optimal_strategy
        }
    
    def match(
        self,
        text: str,
        confidence_threshold: float = 0.7
    ) -> Optional[Dict[str, Any]]:
        """
        Match text using trained parameters if available, otherwise use adaptive strategies.
        
        Args:
            text: Text to match against
            confidence_threshold: Minimum confidence for high-confidence results
            
        Returns:
            Match result with confidence and explanation
        """
        
        if self.is_trained:
            return self._trained_match(text, confidence_threshold)
        else:
            return self._adaptive_match(text, confidence_threshold)
    
    def _trained_match(
        self,
        text: str, 
        confidence_threshold: float
    ) -> Optional[Dict[str, Any]]:
        """Match using trained parameters"""
        
        match = self.trained_matcher.match_with_context(text)
        
        if match:
            return {
                'matched': True,
                'parameters': match.parameters,
                'similarity': match.context_similarity,
                'confidence': match.confidence,
                'high_confidence': match.confidence >= confidence_threshold,
                'method': 'trained',
                'training_metrics': self.trained_params.performance_metrics,
                'context_used': match.context_text,
                'explanation': f"Trained model (threshold: {self.trained_params.optimal_threshold:.2f})"
            }
        
        return None
    
    def _adaptive_match(
        self,
        text: str,
        confidence_threshold: float
    ) -> Optional[Dict[str, Any]]:
        """Match using adaptive strategies without training"""
        
        # Try multiple strategies and contexts
        best_match = None
        best_score = 0.0
        best_method = 'none'
        
        # Strategy 1: Try different target context variants
        for variant_name, context_variant in self.processed_target_contexts.items():
            if variant_name == 'perspective_variants':
                continue
                
            for matcher_name, matcher in self.base_matchers.items():
                # Update matcher's expected context
                matcher.expected_context = context_variant
                matcher._expected_context_embedding = None  # Reset cached embedding
                
                match = matcher.match_with_context(text)
                
                if match:
                    score = match.context_similarity
                    
                    if score > best_score:
                        best_score = score
                        best_match = match
                        best_method = f"{matcher_name}_{variant_name}"
        
        # Strategy 2: Try perspective variants
        for perspective_variant in self.processed_target_contexts['perspective_variants']:
            for matcher_name, matcher in self.base_matchers.items():
                matcher.expected_context = perspective_variant
                matcher._expected_context_embedding = None
                
                match = matcher.match_with_context(text)
                
                if match and match.context_similarity > best_score:
                    best_score = match.context_similarity
                    best_match = match
                    best_method = f"{matcher_name}_perspective"
        
        # Strategy 3: Fallback strategies if no good semantic match
        if not best_match or best_score < 0.3:
            fallback_result = self._try_fallback_strategies(text)
            
            if fallback_result and fallback_result['score'] > best_score:
                return {
                    'matched': True,
                    'parameters': fallback_result['parameters'],
                    'similarity': fallback_result['score'],
                    'confidence': fallback_result['score'] * 0.8,  # Lower confidence for fallback
                    'high_confidence': False,
                    'method': fallback_result['method'],
                    'context_used': fallback_result['context'],
                    'explanation': f"Fallback strategy: {fallback_result['method']}"
                }
        
        if best_match:
            # Estimate confidence for untrained match
            confidence = self._estimate_confidence(best_score, best_method, text)
            
            return {
                'matched': True,
                'parameters': best_match.parameters,
                'similarity': best_score,
                'confidence': confidence,
                'high_confidence': confidence >= confidence_threshold,
                'method': best_method,
                'context_used': best_match.context_text,
                'explanation': f"Adaptive strategy: {best_method} (similarity: {best_score:.2f})"
            }
        
        return None
    
    def _try_fallback_strategies(self, text: str) -> Optional[Dict[str, Any]]:
        """Try fallback strategies when semantic matching fails"""
        
        # Extract context manually
        context_extractor = self.base_matchers['balanced']
        potential_match = context_extractor.expression.match(text)
        
        if not potential_match:
            return None
        
        # Get surrounding context
        context = context_extractor.extract_context(text, position='before')
        
        if not context:
            return None
        
        best_score = 0.0
        best_method = 'none'
        
        target = self.processed_target_contexts['expanded']
        
        # Try different fallback methods
        for strategy in self.config.fallback_strategies:
            if strategy == 'keyword_matching':
                score = self.fallback_matcher.keyword_overlap_score(context, target)
            elif strategy == 'partial_similarity':
                score = self.fallback_matcher.partial_matching_score(context, target)
            elif strategy == 'fuzzy_matching':
                score = self.fallback_matcher.fuzzy_similarity(context, target)
            else:
                continue
            
            if score > best_score:
                best_score = score
                best_method = strategy
        
        if best_score >= 0.4:  # Minimum threshold for fallback
            return {
                'parameters': {param.name: param.value for param in potential_match},
                'score': best_score,
                'method': best_method,
                'context': context
            }
        
        return None
    
    def _estimate_confidence(
        self,
        similarity_score: float,
        method: str,
        text: str
    ) -> float:
        """Estimate confidence for untrained matches"""
        
        base_confidence = similarity_score
        
        # Adjust based on method reliability
        method_multipliers = {
            'conservative': 0.9,  # High precision method
            'balanced': 0.8,     # Balanced method
            'liberal': 0.7       # High recall method
        }
        
        method_type = method.split('_')[0]
        multiplier = method_multipliers.get(method_type, 0.8)
        
        # Adjust based on context length (longer contexts more reliable)
        context_length_bonus = min(0.1, len(text.split()) / 200)
        
        # Adjust based on target context processing
        if 'expanded' in method:
            expansion_bonus = 0.05
        else:
            expansion_bonus = 0
        
        confidence = base_confidence * multiplier + context_length_bonus + expansion_bonus
        
        return min(1.0, max(0.0, confidence))
    
    def get_performance_analysis(
        self,
        test_cases: List[Tuple[str, bool]]  # (text, should_match)
    ) -> Dict[str, Any]:
        """Analyze performance on test cases"""
        
        results = {
            'total_cases': len(test_cases),
            'correct_predictions': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'predictions': []
        }
        
        for text, should_match in test_cases:
            match_result = self.match(text)
            predicted_match = match_result is not None
            
            is_correct = predicted_match == should_match
            
            if is_correct:
                results['correct_predictions'] += 1
            elif predicted_match and not should_match:
                results['false_positives'] += 1
            elif not predicted_match and should_match:
                results['false_negatives'] += 1
            
            results['predictions'].append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'expected': should_match,
                'predicted': predicted_match,
                'correct': is_correct,
                'confidence': match_result.get('confidence', 0.0) if match_result else 0.0,
                'method': match_result.get('method', 'none') if match_result else 'none'
            })
        
        # Calculate metrics
        total = results['total_cases']
        tp = results['correct_predictions'] - results['false_positives']
        fp = results['false_positives']
        fn = results['false_negatives']
        tn = total - tp - fp - fn
        
        results['accuracy'] = results['correct_predictions'] / total if total > 0 else 0
        results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall']) if (results['precision'] + results['recall']) > 0 else 0
        
        return results
    
    def explain_matching_process(self, text: str) -> Dict[str, Any]:
        """Provide detailed explanation of the matching process"""
        
        explanation = {
            'input_text': text[:200] + '...' if len(text) > 200 else text,
            'target_context': self.original_target_context,
            'processed_contexts': self.processed_target_contexts,
            'is_trained': self.is_trained,
            'matching_steps': []
        }
        
        if self.is_trained:
            explanation['trained_params'] = {
                'threshold': self.trained_params.optimal_threshold,
                'window': self.trained_params.optimal_window,
                'strategy': self.trained_params.optimal_strategy,
                'performance': self.trained_params.performance_metrics
            }
        
        # Step-by-step matching process
        match_result = self.match(text)
        
        if match_result:
            explanation['result'] = match_result
            explanation['matching_steps'] = [
                f"1. Input processed with {match_result['method']} method",
                f"2. Context similarity: {match_result['similarity']:.3f}",
                f"3. Confidence calculated: {match_result['confidence']:.3f}",
                f"4. Final decision: MATCH"
            ]
        else:
            explanation['result'] = {'matched': False}
            explanation['matching_steps'] = [
                "1. All matching strategies attempted",
                "2. No strategy exceeded threshold",
                "3. Fallback strategies tried",
                "4. Final decision: NO MATCH"
            ]
        
        return explanation