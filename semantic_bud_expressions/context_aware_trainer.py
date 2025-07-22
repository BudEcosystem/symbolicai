"""
Context-aware expression training system that automatically determines optimal parameters.
"""

import json
import pickle
import time
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from dataclasses import dataclass, asdict
import warnings
from .context_aware_expression import ContextAwareExpression
from .unified_registry import UnifiedParameterTypeRegistry

# Optional sklearn imports
try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. Training functionality will be limited.")
    
    # Mock sklearn functions for basic functionality
    def roc_curve(y_true, y_score):
        return [0, 1], [0, 1], [0.5]
    
    def auc(x, y):
        return 0.5
    
    def precision_recall_curve(y_true, y_score):
        return [0, 1], [0, 1], [0.5]
    
    class StratifiedKFold:
        def __init__(self, n_splits=5, shuffle=True, random_state=None):
            self.n_splits = n_splits
            
        def split(self, X, y):
            # Simple split without stratification
            n = len(X)
            indices = list(range(n))
            fold_size = n // self.n_splits
            
            for i in range(self.n_splits):
                start = i * fold_size
                end = start + fold_size if i < self.n_splits - 1 else n
                test_indices = indices[start:end]
                train_indices = indices[:start] + indices[end:]
                yield train_indices, test_indices


@dataclass
class TrainingConfig:
    """Configuration for context-aware training"""
    threshold_range: Tuple[float, float] = (0.1, 0.9)
    threshold_step: float = 0.05
    window_sizes: List[Union[int, str]] = None
    chunking_strategies: List[str] = None
    optimization_metric: str = 'f1'  # 'f1', 'precision', 'recall', 'accuracy', 'auc'
    cross_validation_folds: int = 5
    min_examples_per_fold: int = 3
    
    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [25, 50, 100, 200, 'sentence', 'auto']
        if self.chunking_strategies is None:
            self.chunking_strategies = ['single', 'sliding', 'sentences', 'overlapping']


@dataclass
class TrainedParameters:
    """Trained parameters for context-aware matching"""
    expression: str
    expected_context: str
    optimal_threshold: float
    optimal_window: Union[int, str]
    optimal_strategy: str
    performance_metrics: Dict[str, float]
    chunk_weights: Optional[List[float]] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    training_timestamp: Optional[float] = None
    
    def save(self, filepath: str):
        """Save trained parameters to file"""
        data = asdict(self)
        data['training_timestamp'] = time.time()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load trained parameters from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ContextAwareTrainer:
    """
    Automatic trainer for context-aware expressions.
    
    Determines optimal thresholds, window sizes, and chunking strategies
    based on labeled training data.
    """
    
    def __init__(self, registry: Optional[UnifiedParameterTypeRegistry] = None):
        self.registry = registry or UnifiedParameterTypeRegistry()
        if not self.registry.model_manager:
            self.registry.initialize_model()
        self.training_history = []
    
    def prepare_training_data(
        self,
        positive_examples: List[str],
        negative_examples: List[str]
    ) -> Tuple[List[str], List[int]]:
        """Prepare training data for optimization"""
        texts = positive_examples + negative_examples
        labels = [1] * len(positive_examples) + [0] * len(negative_examples)
        
        # Shuffle while maintaining alignment
        import random
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        
        return list(texts), list(labels)
    
    def evaluate_threshold(
        self,
        expression: str,
        expected_context: str,
        texts: List[str],
        labels: List[int],
        threshold: float,
        window_size: Union[int, str] = 'auto',
        strategy: str = 'single'
    ) -> Dict[str, float]:
        """Evaluate performance at a specific threshold"""
        
        if strategy == 'single':
            predictions = self._evaluate_single_context(
                expression, expected_context, texts, threshold, window_size
            )
        else:
            predictions = self._evaluate_chunked_context(
                expression, expected_context, texts, threshold, window_size, strategy
            )
        
        return self._calculate_metrics(labels, predictions)
    
    def _evaluate_single_context(
        self,
        expression: str,
        expected_context: str,
        texts: List[str],
        threshold: float,
        window_size: Union[int, str]
    ) -> List[float]:
        """Evaluate using single context extraction"""
        expr = ContextAwareExpression(
            expression=expression,
            expected_context=expected_context,
            context_threshold=0.0,  # Get all similarity scores
            context_window=window_size,
            registry=self.registry
        )
        
        similarities = []
        for text in texts:
            match = expr.match_with_context(text)
            similarity = match.context_similarity if match else 0.0
            similarities.append(similarity)
        
        return similarities
    
    def _evaluate_chunked_context(
        self,
        expression: str,
        expected_context: str,
        texts: List[str],
        threshold: float,
        window_size: Union[int, str],
        strategy: str
    ) -> List[float]:
        """Evaluate using chunked context strategies"""
        from .chunking_strategies import ChunkingProcessor
        
        processor = ChunkingProcessor(self.registry)
        similarities = []
        
        for text in texts:
            chunks = processor.create_chunks(text, strategy, window_size)
            chunk_similarities = []
            
            for chunk in chunks:
                expr = ContextAwareExpression(
                    expression=expression,
                    expected_context=expected_context,
                    context_threshold=0.0,
                    context_window=window_size,
                    registry=self.registry
                )
                match = expr.match_with_context(chunk)
                similarity = match.context_similarity if match else 0.0
                chunk_similarities.append(similarity)
            
            if chunk_similarities:
                # Combine using Naive Bayes approach (geometric mean)
                combined_similarity = np.power(
                    np.prod([max(s, 0.001) for s in chunk_similarities]),
                    1.0 / len(chunk_similarities)
                )
                similarities.append(combined_similarity)
            else:
                similarities.append(0.0)
        
        return similarities
    
    def _calculate_metrics(self, y_true: List[int], similarities: List[float]) -> Dict[str, float]:
        """Calculate performance metrics from similarities"""
        y_true = np.array(y_true)
        similarities = np.array(similarities)
        
        # Calculate ROC AUC
        try:
            fpr, tpr, _ = roc_curve(y_true, similarities)
            roc_auc = auc(fpr, tpr)
        except:
            roc_auc = 0.5
        
        # Calculate PR AUC
        try:
            precision, recall, _ = precision_recall_curve(y_true, similarities)
            pr_auc = auc(recall, precision)
        except:
            pr_auc = 0.5
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'similarities': similarities.tolist()
        }
    
    def find_optimal_threshold(
        self,
        similarities: List[float],
        labels: List[int],
        metric: str = 'f1'
    ) -> Tuple[float, Dict[str, float]]:
        """Find optimal threshold based on the specified metric"""
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_score = 0.0
        threshold_metrics = {}
        
        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)
            
            # Calculate metrics
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            tn = np.sum((predictions == 0) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
            
            threshold_metrics[threshold] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }
            
            # Select best based on metric
            if metric == 'f1':
                score = f1
            elif metric == 'precision':
                score = precision
            elif metric == 'recall':
                score = recall
            elif metric == 'accuracy':
                score = accuracy
            else:
                score = f1
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        best_metrics = threshold_metrics[best_threshold]
        best_metrics['threshold'] = best_threshold
        
        return best_threshold, best_metrics
    
    def train(
        self,
        expression: str,
        expected_context: str,
        positive_examples: List[str],
        negative_examples: List[str],
        config: Optional[TrainingConfig] = None
    ) -> TrainedParameters:
        """
        Train context-aware expression parameters automatically.
        
        Args:
            expression: The pattern expression to train
            expected_context: Expected semantic context
            positive_examples: Examples that should match
            negative_examples: Examples that should not match
            config: Training configuration
            
        Returns:
            TrainedParameters with optimal settings
        """
        if config is None:
            config = TrainingConfig()
        
        print(f"Training context-aware expression: {expression}")
        print(f"Expected context: {expected_context}")
        print(f"Training data: {len(positive_examples)} positive, {len(negative_examples)} negative")
        
        texts, labels = self.prepare_training_data(positive_examples, negative_examples)
        
        best_params = None
        best_score = 0.0
        results = []
        
        # Test different configurations
        total_configs = len(config.window_sizes) * len(config.chunking_strategies)
        current_config = 0
        
        for window_size in config.window_sizes:
            for strategy in config.chunking_strategies:
                current_config += 1
                print(f"Testing configuration {current_config}/{total_configs}: window={window_size}, strategy={strategy}")
                
                try:
                    # Get similarities for all examples
                    similarities = self._evaluate_single_context(
                        expression, expected_context, texts, 0.0, window_size
                    ) if strategy == 'single' else self._evaluate_chunked_context(
                        expression, expected_context, texts, 0.0, window_size, strategy
                    )
                    
                    # Find optimal threshold
                    optimal_threshold, metrics = self.find_optimal_threshold(
                        similarities, labels, config.optimization_metric
                    )
                    
                    score = metrics[config.optimization_metric]
                    
                    result = {
                        'window_size': window_size,
                        'strategy': strategy,
                        'threshold': optimal_threshold,
                        'score': score,
                        'metrics': metrics,
                        'similarities': similarities
                    }
                    results.append(result)
                    
                    print(f"  Threshold: {optimal_threshold:.3f}, {config.optimization_metric}: {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = result
                
                except Exception as e:
                    print(f"  Failed: {str(e)}")
                    continue
        
        if not best_params:
            raise ValueError("No valid configuration found. Check your training data.")
        
        # Create trained parameters object
        trained_params = TrainedParameters(
            expression=expression,
            expected_context=expected_context,
            optimal_threshold=best_params['threshold'],
            optimal_window=best_params['window_size'],
            optimal_strategy=best_params['strategy'],
            performance_metrics=best_params['metrics'],
            training_timestamp=time.time()
        )
        
        self.training_history.append(trained_params)
        
        print(f"\nBest configuration:")
        print(f"  Window: {trained_params.optimal_window}")
        print(f"  Strategy: {trained_params.optimal_strategy}")
        print(f"  Threshold: {trained_params.optimal_threshold:.3f}")
        print(f"  {config.optimization_metric}: {best_score:.3f}")
        print(f"  Precision: {trained_params.performance_metrics['precision']:.3f}")
        print(f"  Recall: {trained_params.performance_metrics['recall']:.3f}")
        print(f"  F1: {trained_params.performance_metrics['f1']:.3f}")
        
        return trained_params
    
    def cross_validate(
        self,
        expression: str,
        expected_context: str,
        positive_examples: List[str],
        negative_examples: List[str],
        config: Optional[TrainingConfig] = None,
        folds: int = 5
    ) -> Dict[str, Any]:
        """Perform cross-validation to assess model stability"""
        if config is None:
            config = TrainingConfig()
        
        texts, labels = self.prepare_training_data(positive_examples, negative_examples)
        
        if len(texts) < folds * 2:
            raise ValueError(f"Need at least {folds * 2} examples for {folds}-fold cross-validation")
        
        kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(texts, labels)):
            print(f"Cross-validation fold {fold + 1}/{folds}")
            
            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            test_texts = [texts[i] for i in test_idx]
            test_labels = [labels[i] for i in test_idx]
            
            # Train on fold
            train_pos = [t for t, l in zip(train_texts, train_labels) if l == 1]
            train_neg = [t for t, l in zip(train_texts, train_labels) if l == 0]
            
            fold_params = self.train(expression, expected_context, train_pos, train_neg, config)
            
            # Test on holdout
            test_similarities = self._evaluate_single_context(
                expression, expected_context, test_texts, 
                fold_params.optimal_threshold, fold_params.optimal_window
            )
            
            test_predictions = [1 if s >= fold_params.optimal_threshold else 0 for s in test_similarities]
            test_metrics = self._calculate_metrics_from_predictions(test_labels, test_predictions)
            
            fold_results.append({
                'params': fold_params,
                'test_metrics': test_metrics
            })
        
        # Aggregate results
        cv_results = self._aggregate_cv_results(fold_results, config.optimization_metric)
        
        return cv_results
    
    def _calculate_metrics_from_predictions(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Calculate metrics from binary predictions"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
    
    def _aggregate_cv_results(self, fold_results: List[Dict], metric: str) -> Dict[str, Any]:
        """Aggregate cross-validation results"""
        scores = [result['test_metrics'][metric] for result in fold_results]
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores,
            'confidence_interval': (
                np.mean(scores) - 1.96 * np.std(scores),
                np.mean(scores) + 1.96 * np.std(scores)
            ),
            'fold_results': fold_results
        }