#!/usr/bin/env python3
"""
Context threshold analyzer - helps find optimal thresholds for different use cases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import (
    ContextAwareExpression,
    EnhancedUnifiedParameterTypeRegistry
)
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class ThresholdAnalyzer:
    """Analyze optimal thresholds for context matching"""
    
    def __init__(self):
        self.registry = EnhancedUnifiedParameterTypeRegistry()
        self.registry.initialize_model()
    
    def analyze_threshold_sensitivity(
        self,
        expression: str,
        expected_context: str,
        positive_examples: List[str],
        negative_examples: List[str],
        context_window: str = 'auto'
    ) -> Dict:
        """Analyze how different thresholds affect matching accuracy"""
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        results = {
            'thresholds': [],
            'true_positive_rates': [],
            'false_positive_rates': [],
            'accuracies': [],
            'f1_scores': [],
            'similarities_positive': [],
            'similarities_negative': []
        }
        
        # First, collect all similarity scores
        expr = ContextAwareExpression(
            expression=expression,
            expected_context=expected_context,
            context_threshold=0.0,  # Accept all to get scores
            context_window=context_window,
            registry=self.registry
        )
        
        pos_similarities = []
        for text in positive_examples:
            match = expr.match_with_context(text)
            if match:
                pos_similarities.append(match.context_similarity)
            else:
                pos_similarities.append(0.0)
        
        neg_similarities = []
        for text in negative_examples:
            match = expr.match_with_context(text)
            if match:
                neg_similarities.append(match.context_similarity)
            else:
                neg_similarities.append(0.0)
        
        results['similarities_positive'] = pos_similarities
        results['similarities_negative'] = neg_similarities
        
        # Test each threshold
        for threshold in thresholds:
            tp = sum(1 for sim in pos_similarities if sim >= threshold)
            fn = len(pos_similarities) - tp
            fp = sum(1 for sim in neg_similarities if sim >= threshold)
            tn = len(neg_similarities) - fp
            
            tpr = tp / len(pos_similarities) if pos_similarities else 0
            fpr = fp / len(neg_similarities) if neg_similarities else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results['thresholds'].append(threshold)
            results['true_positive_rates'].append(tpr)
            results['false_positive_rates'].append(fpr)
            results['accuracies'].append(accuracy)
            results['f1_scores'].append(f1)
        
        # Find optimal thresholds
        results['optimal_accuracy_threshold'] = thresholds[np.argmax(results['accuracies'])]
        results['optimal_f1_threshold'] = thresholds[np.argmax(results['f1_scores'])]
        
        return results
    
    def plot_threshold_analysis(self, results: Dict, title: str, save_path: str = None):
        """Plot threshold analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Threshold Analysis: {title}', fontsize=16)
        
        # ROC-like curve
        ax1 = axes[0, 0]
        ax1.plot(results['false_positive_rates'], results['true_positive_rates'], 'b-', linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC-like Curve')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy vs Threshold
        ax2 = axes[0, 1]
        ax2.plot(results['thresholds'], results['accuracies'], 'g-', linewidth=2)
        ax2.axvline(results['optimal_accuracy_threshold'], color='r', linestyle='--', 
                   label=f'Optimal: {results["optimal_accuracy_threshold"]:.2f}')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 Score vs Threshold
        ax3 = axes[1, 0]
        ax3.plot(results['thresholds'], results['f1_scores'], 'm-', linewidth=2)
        ax3.axvline(results['optimal_f1_threshold'], color='r', linestyle='--',
                   label=f'Optimal: {results["optimal_f1_threshold"]:.2f}')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Score vs Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Similarity Distribution
        ax4 = axes[1, 1]
        ax4.hist(results['similarities_positive'], bins=20, alpha=0.5, label='Positive', color='green')
        ax4.hist(results['similarities_negative'], bins=20, alpha=0.5, label='Negative', color='red')
        ax4.set_xlabel('Similarity Score')
        ax4.set_ylabel('Count')
        ax4.set_title('Similarity Score Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        return fig


def main():
    analyzer = ThresholdAnalyzer()
    
    # 1. Healthcare Context Analysis
    print("Analyzing Healthcare Context Thresholds...")
    healthcare_results = analyzer.analyze_threshold_sensitivity(
        expression="patient has {symptom}",
        expected_context="medical healthcare hospital doctor",
        positive_examples=[
            "The doctor examined the patient who has severe headache",
            "Medical records indicate the patient has diabetes",
            "In the emergency room, the patient has chest pain",
            "The nurse noted that the patient has fever",
            "During the consultation, the patient has anxiety symptoms"
        ],
        negative_examples=[
            "The computer has virus issues",
            "My car has engine problems",
            "The building has structural damage",
            "The plant has yellow leaves",
            "The software has bugs"
        ]
    )
    
    print(f"Healthcare Optimal Thresholds:")
    print(f"  For best accuracy: {healthcare_results['optimal_accuracy_threshold']:.2f}")
    print(f"  For best F1 score: {healthcare_results['optimal_f1_threshold']:.2f}")
    print(f"  Positive similarities: {[f'{s:.2f}' for s in healthcare_results['similarities_positive']]}")
    print(f"  Negative similarities: {[f'{s:.2f}' for s in healthcare_results['similarities_negative']]}")
    print()
    
    # 2. Financial Context Analysis
    print("Analyzing Financial Context Thresholds...")
    financial_results = analyzer.analyze_threshold_sensitivity(
        expression="transfer {amount} to {account}",
        expected_context="banking finance payment transaction money",
        positive_examples=[
            "Login to your bank account to transfer funds to savings",
            "Complete the wire transfer of $1000 to checking account",
            "Use online banking to transfer money to investment account",
            "PayPal allows you to transfer payments to merchant account",
            "The financial advisor recommends transfer to IRA account"
        ],
        negative_examples=[
            "Transfer the files to backup folder",
            "Transfer patient to ICU immediately",
            "Transfer data to external drive",
            "Transfer ownership to new buyer",
            "Transfer student to different class"
        ]
    )
    
    print(f"Financial Optimal Thresholds:")
    print(f"  For best accuracy: {financial_results['optimal_accuracy_threshold']:.2f}")
    print(f"  For best F1 score: {financial_results['optimal_f1_threshold']:.2f}")
    print()
    
    # 3. E-commerce Context Analysis
    print("Analyzing E-commerce Context Thresholds...")
    ecommerce_results = analyzer.analyze_threshold_sensitivity(
        expression="add {item} to cart",
        expected_context="shopping online store purchase checkout",
        positive_examples=[
            "Browse our online store and add items to cart",
            "Ready to checkout? Add the product to cart",
            "Shopping for electronics, add laptop to cart",
            "During the sale, add multiple items to cart",
            "E-commerce site lets you add products to cart"
        ],
        negative_examples=[
            "Add sugar to the recipe",
            "Add user to database",
            "Add numbers to calculate total",
            "Add employee to team",
            "Add comment to document"
        ]
    )
    
    print(f"E-commerce Optimal Thresholds:")
    print(f"  For best accuracy: {ecommerce_results['optimal_accuracy_threshold']:.2f}")
    print(f"  For best F1 score: {ecommerce_results['optimal_f1_threshold']:.2f}")
    print()
    
    # 4. Test context window strategies
    print("\nTesting Different Context Windows...")
    
    test_text = """
    The patient was brought to the emergency department with severe symptoms.
    After initial assessment, the patient has difficulty breathing and chest pain.
    The medical team is preparing for immediate intervention.
    """
    
    windows = ['auto', 'sentence', 50, 100]
    for window in windows:
        expr = ContextAwareExpression(
            expression="patient has {symptoms}",
            expected_context="medical emergency hospital",
            context_threshold=0.0,
            context_window=window,
            registry=analyzer.registry
        )
        
        match = expr.match_with_context(test_text)
        if match:
            print(f"\nWindow '{window}':")
            print(f"  Context: '{match.context_text[:60]}...'")
            print(f"  Similarity: {match.context_similarity:.3f}")
    
    # 5. Recommendations
    print("\n" + "="*60)
    print("THRESHOLD RECOMMENDATIONS:")
    print("="*60)
    print()
    print("1. General Purpose (balanced): 0.45 - 0.50")
    print("   - Good balance between precision and recall")
    print("   - Works well across different industries")
    print()
    print("2. High Precision (few false positives): 0.55 - 0.65")
    print("   - Use for legal, medical, financial contexts")
    print("   - When false positives are costly")
    print()
    print("3. High Recall (catch most matches): 0.35 - 0.45")
    print("   - Use for e-commerce, search, discovery")
    print("   - When missing matches is costly")
    print()
    print("4. Domain-Specific Tuning:")
    print("   - Always test with your specific data")
    print("   - Consider the cost of false positives vs false negatives")
    print("   - Use similarity score distributions to guide selection")
    print()
    print("5. Context Window Selection:")
    print("   - 'auto': Best for most cases")
    print("   - 'sentence': Good for structured text")
    print("   - 50-100 words: Good for short contexts")
    print("   - Larger windows: Better for long documents")


if __name__ == "__main__":
    main()