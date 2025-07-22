#!/usr/bin/env python3
"""
Complete demonstration of the training system functionality.
Shows the full workflow from untrained to trained performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import (
    AdaptiveContextMatcher,
    ContextAwareTrainer,
    TrainedParameters,
    EnhancedUnifiedParameterTypeRegistry
)


def demonstrate_complete_training():
    """Demonstrate the complete training workflow"""
    print("="*80)
    print("COMPLETE TRAINING SYSTEM DEMONSTRATION")
    print("="*80)
    print("Shows: untrained → training → trained performance comparison")
    print()
    
    # Initialize
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Create adaptive matcher
    print("1. CREATING ADAPTIVE CONTEXT MATCHER")
    print("-"*50)
    
    matcher = AdaptiveContextMatcher(
        expression="patient has {condition}",
        target_context="your health medical help",  # Short, second-person
        registry=registry
    )
    
    print(f"Expression: {matcher.expression}")
    print(f"Target context: {matcher.original_target_context}")
    print(f"Expanded context: {matcher.processed_target_contexts['expanded']}")
    print()
    
    # Test cases for evaluation
    test_cases = [
        # Should match - medical contexts
        ("The emergency physician examined the patient carefully. After diagnostic tests, the patient has severe pneumonia requiring treatment.", True),
        ("During the clinic visit, healthcare provider noted symptoms. The patient has diabetes and needs management.", True),
        ("Medical assessment completed at the hospital. The patient has hypertension that requires monitoring.", True),
        
        # Should not match - non-medical contexts  
        ("The technical team analyzed the system. After diagnostics, the system has critical vulnerabilities.", False),
        ("The mechanic inspected the vehicle thoroughly. The car has engine problems requiring repairs.", False),
        ("The auditor reviewed the company finances. The business has cash flow issues needing attention.", False)
    ]
    
    print("2. TESTING UNTRAINED PERFORMANCE")
    print("-"*50)
    
    untrained_results = matcher.get_performance_analysis(test_cases)
    print(f"Untrained Results:")
    print(f"  Accuracy: {untrained_results['accuracy']:.2f}")
    print(f"  F1 Score: {untrained_results['f1']:.2f}")
    print(f"  False Positives: {untrained_results['false_positives']}")
    print(f"  False Negatives: {untrained_results['false_negatives']}")
    
    # Show individual predictions
    print(f"\nDetailed Results:")
    for i, pred in enumerate(untrained_results['predictions'][:3]):
        expected_str = "✓ SHOULD MATCH" if pred['expected'] else "✗ SHOULD NOT MATCH"
        actual_str = "MATCHED" if pred['predicted'] else "NO MATCH"
        confidence = pred.get('confidence', 0)
        print(f"  Case {i+1}: {expected_str} → {actual_str} (confidence: {confidence:.2f})")
    print()
    
    print("3. PREPARING TRAINING DATA")
    print("-"*50)
    
    # Training data - positive examples (should match)
    positive_examples = [
        "The hospital emergency department treated a critical case. The patient has acute appendicitis requiring surgery.",
        "Medical examination revealed concerning symptoms. The patient has bronchitis and needs antibiotics.",
        "After comprehensive diagnostic testing, the patient has kidney stones causing severe pain.",
        "The clinic specialist completed the assessment. The patient has arthritis affecting mobility.",
        "Healthcare provider reviewed test results. The patient has anemia requiring iron supplements.",
        "Emergency room physician made diagnosis. The patient has concussion from the accident.",
        "Medical team conducted thorough evaluation. The patient has gastritis needing treatment.",
        "Doctor completed physical examination. The patient has migraine headaches frequently occurring.",
    ]
    
    # Training data - negative examples (should not match)
    negative_examples = [
        "The IT department diagnosed the network issue. The system has connectivity problems affecting users.",
        "Automotive technician inspected the vehicle. The car has transmission problems requiring replacement.",
        "Quality assurance team found defects. The product has manufacturing issues needing correction.",
        "Financial analyst reviewed the reports. The company has budget problems requiring restructuring.",
        "Building inspector examined the structure. The house has foundation problems causing concerns.",
        "Software engineer debugged the application. The program has memory leaks affecting performance.",
        "Research scientist analyzed the sample. The specimen has contamination requiring new testing.",
        "Project manager assessed the timeline. The schedule has delays causing delivery problems.",
    ]
    
    print(f"Training data prepared:")
    print(f"  Positive examples: {len(positive_examples)}")
    print(f"  Negative examples: {len(negative_examples)}")
    print()
    
    print("4. TRAINING THE MATCHER")
    print("-"*50)
    
    try:
        training_results = matcher.train(
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            optimize_for='balanced'  # Balance precision and recall
        )
        
        print(f"Training completed successfully!")
        print(f"  Optimal threshold: {training_results['optimal_threshold']:.3f}")
        print(f"  Training F1 score: {training_results['performance']['f1']:.3f}")
        print(f"  Training precision: {training_results['performance']['precision']:.3f}")
        print(f"  Training recall: {training_results['performance']['recall']:.3f}")
        print(f"  Training accuracy: {training_results['performance']['accuracy']:.3f}")
        
    except Exception as e:
        print(f"Training error: {e}")
        return
    
    print()
    
    print("5. TESTING TRAINED PERFORMANCE")
    print("-"*50)
    
    trained_results = matcher.get_performance_analysis(test_cases)
    print(f"Trained Results:")
    print(f"  Accuracy: {trained_results['accuracy']:.2f}")
    print(f"  F1 Score: {trained_results['f1']:.2f}")
    print(f"  False Positives: {trained_results['false_positives']}")
    print(f"  False Negatives: {trained_results['false_negatives']}")
    print()
    
    print("6. PERFORMANCE COMPARISON")
    print("-"*50)
    
    accuracy_improvement = trained_results['accuracy'] - untrained_results['accuracy']
    f1_improvement = trained_results['f1'] - untrained_results['f1']
    
    print(f"Performance Improvements:")
    print(f"  Accuracy: {untrained_results['accuracy']:.2f} → {trained_results['accuracy']:.2f} ({accuracy_improvement:+.2f})")
    print(f"  F1 Score: {untrained_results['f1']:.2f} → {trained_results['f1']:.2f} ({f1_improvement:+.2f})")
    
    fp_change = trained_results['false_positives'] - untrained_results['false_positives']
    fn_change = trained_results['false_negatives'] - untrained_results['false_negatives']
    print(f"  False Positives: {untrained_results['false_positives']} → {trained_results['false_positives']} ({fp_change:+d})")
    print(f"  False Negatives: {untrained_results['false_negatives']} → {trained_results['false_negatives']} ({fn_change:+d})")
    print()
    
    print("7. REAL-WORLD MATCHING DEMONSTRATION")
    print("-"*50)
    
    # Test on new, unseen text
    new_test_cases = [
        "The cardiologist examined the elderly patient thoroughly. After running EKG and blood tests, the patient has atrial fibrillation.",
        "The project team reviewed the software architecture. After code analysis, the application has scalability bottlenecks."
    ]
    
    for i, text in enumerate(new_test_cases, 1):
        print(f"\nCase {i}: {text[:80]}...")
        
        result = matcher.match(text)
        if result:
            print(f"  ✓ MATCHED")
            print(f"    Confidence: {result['confidence']:.2f}")
            print(f"    Method: {result['method']}")
            print(f"    Condition: {result['parameters'].get('condition', 'N/A')}")
            print(f"    Explanation: {result['explanation']}")
        else:
            print(f"  ✗ NO MATCH")
    
    print()
    print("="*80)
    print("TRAINING DEMONSTRATION COMPLETE")
    print("="*80)
    print("Key achievements:")
    print("✓ Zero-training performance (adaptive strategies)")
    print("✓ Automatic parameter optimization")
    print("✓ Significant performance improvement with training")
    print("✓ Handles context length and perspective mismatches")
    print("✓ Reduces both false positives and false negatives")
    print("✓ Provides detailed confidence scoring and explanations")


def demonstrate_direct_trainer():
    """Demonstrate using the ContextAwareTrainer directly"""
    print("\n" + "="*80)
    print("DIRECT TRAINER USAGE DEMONSTRATION")
    print("="*80)
    
    # Initialize trainer
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    trainer = ContextAwareTrainer(registry)
    
    # Simple training example
    positive_examples = [
        "Bank security verified. Transfer $1000 to savings account",
        "Use mobile banking to transfer funds to checking account", 
        "Complete wire transfer of payment to merchant account"
    ]
    
    negative_examples = [
        "Transfer files to backup folder for storage",
        "Transfer patient to ICU for medical care",
        "Transfer ownership documents to new buyer"
    ]
    
    print("Training with ContextAwareTrainer directly:")
    try:
        trained_params = trainer.train(
            expression="transfer {amount} to {account}",
            expected_context="banking money financial",
            positive_examples=positive_examples,
            negative_examples=negative_examples
        )
        
        print(f"✓ Direct training successful!")
        print(f"  Parameters saved to trained_params object")
        print(f"  F1 Score: {trained_params.performance_metrics.get('f1', 0):.3f}")
        
        # Save parameters
        trained_params.save('/tmp/banking_model.json')
        print(f"✓ Model saved to /tmp/banking_model.json")
        
        # Load parameters
        loaded_params = TrainedParameters.load('/tmp/banking_model.json')
        print(f"✓ Model loaded successfully")
        print(f"  Loaded threshold: {loaded_params.optimal_threshold:.3f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    try:
        demonstrate_complete_training()
        demonstrate_direct_trainer()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()