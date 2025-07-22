#!/usr/bin/env python3
"""
Evaluation framework for trained context-aware models.
Tests models on holdout data and generates performance reports.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import (
    ContextAwareTrainer,
    TrainedContextAwareExpression,
    TrainedParameters,
    ContextAwareExpression,
    EnhancedUnifiedParameterTypeRegistry
)
import json
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Tuple, Dict, Any


class ModelEvaluator:
    """Comprehensive evaluation framework for context-aware models"""
    
    def __init__(self, registry=None):
        self.registry = registry or EnhancedUnifiedParameterTypeRegistry()
        if not self.registry.model_manager:
            self.registry.initialize_model()
    
    def evaluate_model(
        self,
        model: TrainedContextAwareExpression,
        test_texts: List[str],
        test_labels: List[int],
        model_name: str = "Trained Model"
    ) -> Dict[str, Any]:
        """Evaluate a trained model on test data"""
        
        print(f"\nEvaluating {model_name}...")
        print("-" * 50)
        
        predictions = []
        confidences = []
        processing_times = []
        
        # Make predictions
        for text in test_texts:
            start_time = time.time()
            match = model.match_with_context(text)
            processing_time = (time.time() - start_time) * 1000  # ms
            
            prediction = 1 if match else 0
            confidence = match.confidence if match else 0.0
            
            predictions.append(prediction)
            confidences.append(confidence)
            processing_times.append(processing_time)
        
        # Calculate metrics
        predictions = np.array(predictions)
        test_labels = np.array(test_labels)
        
        # Basic metrics
        accuracy = np.mean(predictions == test_labels)
        
        tp = np.sum((predictions == 1) & (test_labels == 1))
        fp = np.sum((predictions == 1) & (test_labels == 0))
        tn = np.sum((predictions == 0) & (test_labels == 0))
        fn = np.sum((predictions == 0) & (test_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Performance metrics
        avg_processing_time = np.mean(processing_times)
        avg_confidence = np.mean(confidences)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'confusion_matrix': cm.tolist(),
            'avg_processing_time_ms': avg_processing_time,
            'avg_confidence': avg_confidence,
            'predictions': predictions.tolist(),
            'confidences': confidences
        }
        
        # Print summary
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")
        print(f"Avg Time:  {avg_processing_time:.2f}ms")
        print(f"Avg Conf:  {avg_confidence:.3f}")
        
        return results
    
    def compare_models(
        self,
        models: List[Tuple[Any, str]],  # (model, name) pairs
        test_texts: List[str],
        test_labels: List[int]
    ) -> Dict[str, Any]:
        """Compare multiple models on the same test data"""
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        results = []
        
        for model, name in models:
            result = self.evaluate_model(model, test_texts, test_labels, name)
            results.append(result)
        
        # Create comparison table
        print(f"\n{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time(ms)':<10}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['model_name']:<20} "
                  f"{result['accuracy']:<10.3f} "
                  f"{result['precision']:<10.3f} "
                  f"{result['recall']:<10.3f} "
                  f"{result['f1_score']:<10.3f} "
                  f"{result['avg_processing_time_ms']:<10.2f}")
        
        # Find best model
        best_f1 = max(results, key=lambda x: x['f1_score'])
        best_accuracy = max(results, key=lambda x: x['accuracy'])
        fastest = min(results, key=lambda x: x['avg_processing_time_ms'])
        
        print(f"\nBest F1 Score: {best_f1['model_name']} ({best_f1['f1_score']:.3f})")
        print(f"Best Accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.3f})")
        print(f"Fastest: {fastest['model_name']} ({fastest['avg_processing_time_ms']:.2f}ms)")
        
        return {
            'results': results,
            'best_f1': best_f1,
            'best_accuracy': best_accuracy,
            'fastest': fastest
        }
    
    def cross_domain_evaluation(
        self,
        models: Dict[str, TrainedContextAwareExpression],
        test_datasets: Dict[str, Tuple[List[str], List[int]]]
    ):
        """Evaluate how well models perform across different domains"""
        
        print("\n" + "="*60)
        print("CROSS-DOMAIN EVALUATION")
        print("="*60)
        
        results = {}
        
        for model_domain, model in models.items():
            results[model_domain] = {}
            
            for test_domain, (test_texts, test_labels) in test_datasets.items():
                print(f"\nTesting {model_domain} model on {test_domain} data...")
                
                result = self.evaluate_model(
                    model, test_texts, test_labels, 
                    f"{model_domain} on {test_domain}"
                )
                
                results[model_domain][test_domain] = result
        
        # Create cross-domain performance matrix
        print(f"\n{'Model \\ Test Data':<15}", end="")
        for test_domain in test_datasets.keys():
            print(f"{test_domain:<12}", end="")
        print()
        print("-" * (15 + 12 * len(test_datasets)))
        
        for model_domain in models.keys():
            print(f"{model_domain:<15}", end="")
            for test_domain in test_datasets.keys():
                f1_score = results[model_domain][test_domain]['f1_score']
                print(f"{f1_score:<12.3f}", end="")
            print()
        
        return results


def create_holdout_test_data():
    """Create holdout test data for different domains"""
    
    test_datasets = {
        'healthcare': {
            'positive': [
                "The emergency physician noted the patient has cardiac arrest symptoms",
                "Radiology report indicates the patient has bone fracture in left arm",
                "The pediatrician confirmed the patient has viral infection requiring rest",
                "Surgical consultation reveals the patient has gallbladder stones needing removal",
                "The oncologist determined the patient has tumor requiring chemotherapy treatment"
            ],
            'negative': [
                "The smartphone has battery drain issues affecting performance daily",
                "The vehicle has transmission problems requiring immediate professional attention",
                "The website has loading speed issues frustrating users significantly",
                "The application has memory leak problems causing system crashes",
                "The printer has paper jam issues disrupting office workflow"
            ]
        },
        'financial': {
            'positive': [
                "Online banking platform: transfer $2500 to investment portfolio account",
                "Credit union services: transfer funds to mortgage payment account",
                "International wire: transfer euros to foreign exchange account",
                "Payroll processing: transfer salaries to employee checking accounts",
                "Investment broker: transfer dividends to client savings accounts"
            ],
            'negative': [
                "Office relocation: transfer equipment to new building location",
                "Academic program: transfer credits to partner university system",
                "Transport logistics: transfer cargo to destination warehouse facility",
                "Human resources: transfer employee to different department division",
                "Data migration: transfer files to cloud storage platform"
            ]
        },
        'ecommerce': {
            'positive': [
                "Online marketplace: add luxury watch to cart before checkout",
                "Digital storefront: add premium software to shopping basket",
                "Retail website: add designer clothing to wishlist for later",
                "E-commerce platform: add electronics bundle to cart immediately",
                "Online store: add books and accessories to order"
            ],
            'negative': [
                "Laboratory procedure: add chemical reagent to test solution mixture",
                "Construction project: add insulation to building wall structure",
                "Meeting preparation: add agenda items to discussion list",
                "Database administration: add new records to customer table",
                "Garden maintenance: add organic fertilizer to soil preparation"
            ]
        }
    }
    
    # Convert to required format
    formatted_datasets = {}
    for domain, data in test_datasets.items():
        texts = data['positive'] + data['negative']
        labels = [1] * len(data['positive']) + [0] * len(data['negative'])
        formatted_datasets[domain] = (texts, labels)
    
    return formatted_datasets


def evaluate_chunking_strategies():
    """Evaluate different chunking strategies"""
    
    print("\n" + "="*60)
    print("CHUNKING STRATEGY EVALUATION")
    print("="*60)
    
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Create long text samples for chunking
    long_texts = [
        """
        The patient was admitted to the emergency department yesterday evening with acute symptoms.
        After thorough examination by our medical team and running comprehensive diagnostic tests,
        the attending physician determined that the patient has severe pneumonia requiring immediate
        hospitalization and intensive antibiotic treatment. The medical staff is monitoring vital signs.
        """,
        """
        Please log into your online banking account to complete the financial transaction.
        Our secure banking platform allows you to easily transfer funds between accounts.
        For this transaction, you need to transfer $5000 to your investment account
        to complete your portfolio diversification strategy as recommended by your financial advisor.
        """,
        """
        Welcome to our premium online shopping experience on our e-commerce platform.
        Browse through thousands of products in our digital marketplace catalog.
        When you're ready to make a purchase, simply add items to cart and proceed
        to our secure checkout process. Don't forget to add warranty protection to your order.
        """
    ]
    
    labels = [1, 1, 1]  # All should match their respective domains
    
    # Test different strategies
    strategies = ['single', 'sliding', 'sentences', 'overlapping']
    
    for strategy in strategies:
        print(f"\nTesting chunking strategy: {strategy}")
        print("-" * 40)
        
        # Create models with different strategies
        from semantic_bud_expressions.chunking_strategies import ChunkedContextMatcher
        
        healthcare_matcher = ChunkedContextMatcher(
            expression="patient has {condition}",
            expected_context="medical healthcare hospital doctor",
            registry=registry,
            chunking_strategy=strategy,
            window_size=100
        )
        
        matches = 0
        total_time = 0
        
        for text in long_texts[:1]:  # Test on healthcare text
            start_time = time.time()
            result, chunks, similarity = healthcare_matcher.match_with_chunks(text, 0.4)
            elapsed = time.time() - start_time
            
            if result:
                matches += 1
            total_time += elapsed
            
            print(f"  Chunks: {len(chunks)}, Similarity: {similarity:.3f}, "
                  f"Match: {'✓' if result else '✗'}, Time: {elapsed*1000:.1f}ms")
        
        print(f"  Strategy performance: {matches}/1 matches, avg time: {total_time*1000:.1f}ms")


def main():
    """Run comprehensive evaluation"""
    
    print("Context-Aware Model Evaluation Framework")
    print("=" * 60)
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Create test data
    test_datasets = create_holdout_test_data()
    
    # Try to load trained models
    try:
        models = {}
        model_files = [
            ('healthcare', 'trained_healthcare_model.json'),
            ('financial', 'trained_financial_model.json'),
            ('ecommerce', 'trained_ecommerce_model.json')
        ]
        
        for domain, filename in model_files:
            try:
                models[domain] = TrainedContextAwareExpression.from_file(filename, evaluator.registry)
                print(f"✓ Loaded {domain} model")
            except FileNotFoundError:
                print(f"✗ Could not load {domain} model from {filename}")
        
        if not models:
            print("\nNo trained models found. Run train_context_aware.py first.")
            return
        
        # 1. Individual model evaluation
        print("\n" + "="*60)
        print("INDIVIDUAL MODEL EVALUATION")
        print("="*60)
        
        individual_results = {}
        for domain, model in models.items():
            if domain in test_datasets:
                test_texts, test_labels = test_datasets[domain]
                result = evaluator.evaluate_model(model, test_texts, test_labels, f"{domain.title()} Model")
                individual_results[domain] = result
        
        # 2. Compare trained vs baseline models
        if 'healthcare' in models and 'healthcare' in test_datasets:
            print("\n" + "="*60)
            print("TRAINED VS BASELINE COMPARISON")
            print("="*60)
            
            # Create baseline model
            baseline = ContextAwareExpression(
                expression="patient has {condition}",
                expected_context="medical healthcare hospital doctor clinical treatment",
                context_threshold=0.5,
                registry=evaluator.registry
            )
            
            test_texts, test_labels = test_datasets['healthcare']
            
            comparison_models = [
                (models['healthcare'], "Trained Healthcare"),
                (baseline, "Baseline Healthcare")
            ]
            
            evaluator.compare_models(comparison_models, test_texts, test_labels)
        
        # 3. Cross-domain evaluation
        if len(models) > 1 and len(test_datasets) > 1:
            print("\n" + "="*60)
            print("CROSS-DOMAIN ROBUSTNESS")
            print("="*60)
            
            cross_results = evaluator.cross_domain_evaluation(models, test_datasets)
        
        # 4. Chunking strategy evaluation
        evaluate_chunking_strategies()
        
        # 5. Generate summary report
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print("\nKey Findings:")
        
        if individual_results:
            best_model = max(individual_results.values(), key=lambda x: x['f1_score'])
            print(f"- Best performing model: {best_model['model_name']} (F1: {best_model['f1_score']:.3f})")
            
            avg_accuracy = np.mean([r['accuracy'] for r in individual_results.values()])
            print(f"- Average accuracy across domains: {avg_accuracy:.3f}")
            
            avg_time = np.mean([r['avg_processing_time_ms'] for r in individual_results.values()])
            print(f"- Average processing time: {avg_time:.2f}ms")
        
        print("\nRecommendations:")
        print("- Use trained models for domain-specific applications")
        print("- Monitor confidence scores for production deployment")
        print("- Consider ensemble methods for cross-domain robustness")
        print("- Retrain models periodically with new data")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()