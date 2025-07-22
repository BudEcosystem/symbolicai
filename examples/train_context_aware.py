#!/usr/bin/env python3
"""
Training examples for context-aware expressions across different domains.
Shows how to train models for healthcare, finance, e-commerce, etc.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import (
    ContextAwareTrainer,
    TrainedContextAwareExpression,
    TrainingConfig,
    EnhancedUnifiedParameterTypeRegistry
)
import json


def train_healthcare_model():
    """Train a model for healthcare context matching"""
    print("=" * 60)
    print("Training Healthcare Context Model")
    print("=" * 60)
    
    # Create registry
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Training data
    positive_examples = [
        "The doctor examined the patient who has severe headache and prescribed medication",
        "Medical records indicate the patient has diabetes and requires insulin therapy",
        "In the emergency room, the patient has chest pain and needs immediate attention",
        "The nurse noted that the patient has fever and administered antibiotics",
        "During the consultation, the patient has anxiety symptoms and was referred to therapy",
        "Clinical assessment shows the patient has hypertension and needs monitoring",
        "The specialist confirmed the patient has arthritis and recommended treatment",
        "Hospital admission notes: patient has pneumonia and requires oxygen support",
        "The surgeon explained that the patient has appendicitis and needs surgery",
        "After examination, the patient has migraine and was given pain relief"
    ]
    
    negative_examples = [
        "The computer has virus issues and needs antivirus software",
        "My car has engine problems and requires immediate repair",
        "The building has structural damage and needs renovation work",
        "The plant has yellow leaves and requires more water",
        "The software has bugs and needs debugging by developers",
        "The machine has mechanical failure and requires maintenance",
        "The project has budget issues and needs additional funding",
        "The website has performance problems and requires optimization",
        "The device has battery problems and needs replacement",
        "The system has security vulnerabilities and requires patches"
    ]
    
    # Create trainer
    trainer = ContextAwareTrainer(registry)
    
    # Train model
    trained_params = trainer.train(
        expression="patient has {condition}",
        expected_context="medical healthcare hospital doctor clinical treatment",
        positive_examples=positive_examples,
        negative_examples=negative_examples
    )
    
    # Save trained model
    trained_params.save('trained_healthcare_model.json')
    print(f"\nModel saved to: trained_healthcare_model.json")
    
    # Test the trained model
    print("\nTesting trained model:")
    trained_expr = TrainedContextAwareExpression(trained_params, registry)
    
    test_cases = [
        "The physician determined that the patient has bronchitis",  # Should match
        "The application has connectivity issues",  # Should not match
        "Medical examination revealed the patient has cardiac arrhythmia"  # Should match
    ]
    
    for test_text in test_cases:
        match = trained_expr.match_with_context(test_text)
        if match:
            print(f"✓ MATCH: '{test_text[:50]}...' -> {match.parameters} (confidence: {match.confidence:.2f})")
        else:
            print(f"✗ NO MATCH: '{test_text[:50]}...'")
    
    return trained_params


def train_financial_model():
    """Train a model for financial context matching"""
    print("\n" + "=" * 60)
    print("Training Financial Context Model")
    print("=" * 60)
    
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    positive_examples = [
        "Login to your bank account to transfer $1000 to savings account",
        "Complete the wire transfer of $5000 to checking account immediately",
        "Use online banking to transfer funds to investment account",
        "PayPal allows you to transfer payments to merchant account",
        "The financial advisor recommends to transfer money to retirement account",
        "ATM transaction: transfer $500 to credit account for payment",
        "Mobile banking app lets you transfer amounts to joint account",
        "Electronic transfer of $2000 to business account was completed",
        "Currency exchange: transfer euros to dollar account",
        "Automated transfer from payroll to employee accounts"
    ]
    
    negative_examples = [
        "Transfer the files to backup folder on server",
        "Transfer patient to ICU immediately for surgery",
        "Transfer data to external drive for storage",
        "Transfer ownership to new buyer after sale",
        "Transfer student to different class next semester",
        "Transfer call to technical support department",
        "Transfer luggage to connecting flight gate",
        "Transfer employee to different department",
        "Transfer blame to someone else unfairly",
        "Transfer skills learned to new project"
    ]
    
    trainer = ContextAwareTrainer(registry)
    
    # Use custom training config for financial domain
    config = TrainingConfig(
        optimization_metric='precision',  # Financial needs high precision
        threshold_range=(0.4, 0.8),
        window_sizes=[50, 100, 'sentence', 'auto']
    )
    
    trained_params = trainer.train(
        expression="transfer {amount} to {account}",
        expected_context="banking financial payment transaction money account funds",
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        config=config
    )
    
    trained_params.save('trained_financial_model.json')
    print(f"\nModel saved to: trained_financial_model.json")
    
    # Test the model
    print("\nTesting trained model:")
    trained_expr = TrainedContextAwareExpression(trained_params, registry)
    
    test_cases = [
        "Banking app: transfer $300 to checking account",
        "Transfer the document to legal team",
        "Cryptocurrency wallet: transfer 0.5 BTC to cold storage"
    ]
    
    for test_text in test_cases:
        match = trained_expr.match_with_context(test_text)
        if match:
            print(f"✓ MATCH: '{test_text[:50]}...' -> {match.parameters} (confidence: {match.confidence:.2f})")
        else:
            print(f"✗ NO MATCH: '{test_text[:50]}...'")
    
    return trained_params


def train_ecommerce_model():
    """Train a model for e-commerce context matching"""
    print("\n" + "=" * 60)
    print("Training E-commerce Context Model")
    print("=" * 60)
    
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    positive_examples = [
        "Browse our online store and add iPhone to cart for purchase",
        "Ready to checkout? Add warranty protection to order now",
        "Shopping for electronics today, add laptop to wishlist",
        "Complete your purchase by adding shoes to basket",
        "Amazon marketplace: add books to cart before sale ends",
        "During Black Friday sale, add TV to cart immediately",
        "E-commerce website allows you to add products to cart",
        "Online retailer: add clothing items to shopping bag",
        "Digital store checkout: add accessories to order",
        "Marketplace shopping: add multiple items to cart"
    ]
    
    negative_examples = [
        "In the recipe instructions, add sugar to mixture",
        "Database management: add user to system database",
        "Add new employee to company payroll system",
        "Swimming pool maintenance: add chlorine to water",
        "Software development: add feature to application",
        "Meeting agenda: add discussion topic to list",
        "Library system: add book to catalog",
        "Add comment to document for review",
        "Garden care: add fertilizer to soil",
        "Add contact information to phone"
    ]
    
    trainer = ContextAwareTrainer(registry)
    
    config = TrainingConfig(
        optimization_metric='recall',  # E-commerce wants to catch opportunities
        chunking_strategies=['single', 'sliding', 'sentences']
    )
    
    trained_params = trainer.train(
        expression="add {product} to {destination}",
        expected_context="shopping cart purchase buy online store checkout retail",
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        config=config
    )
    
    trained_params.save('trained_ecommerce_model.json')
    print(f"\nModel saved to: trained_ecommerce_model.json")
    
    # Test the model
    print("\nTesting trained model:")
    trained_expr = TrainedContextAwareExpression(trained_params, registry)
    
    test_cases = [
        "Shopping website: add premium headphones to cart",
        "Recipe cooking: add salt to taste",
        "Digital marketplace: add software license to order"
    ]
    
    for test_text in test_cases:
        match = trained_expr.match_with_context(test_text)
        if match:
            print(f"✓ MATCH: '{test_text[:50]}...' -> {match.parameters} (confidence: {match.confidence:.2f})")
        else:
            print(f"✗ NO MATCH: '{test_text[:50]}...'")
    
    return trained_params


def train_legal_model():
    """Train a model for legal context matching"""
    print("\n" + "=" * 60)
    print("Training Legal Context Model")
    print("=" * 60)
    
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    positive_examples = [
        "Per the contract terms, the buyer shall pay damages within 30 days",
        "The court ruled that the defendant shall appear before the judge",
        "According to the agreement, the tenant shall vacate premises immediately",
        "The attorney advised that the client shall provide documents for review",
        "Legal settlement states the company shall compensate affected parties",
        "Jurisdiction requires that the corporation shall comply with regulations",
        "The legal department mandates the vendor shall deliver goods on time",
        "Court order specifies the debtor shall repay outstanding amounts",
        "Contractual obligation: the party shall perform duties as specified",
        "Legal precedent dictates the organization shall maintain standards"
    ]
    
    negative_examples = [
        "The soccer player shall score goals for the team",
        "The chef shall prepare dinner for restaurant guests",
        "The driver shall maintain speed limits on highway",
        "The artist shall complete painting by deadline",
        "The teacher shall educate students effectively",
        "The engineer shall design bridge specifications",
        "The musician shall perform concert next week",
        "The gardener shall water plants regularly",
        "The pilot shall fly airplane safely",
        "The writer shall finish novel manuscript"
    ]
    
    trainer = ContextAwareTrainer(registry)
    
    config = TrainingConfig(
        optimization_metric='precision',  # Legal needs very high precision
        threshold_range=(0.5, 0.9),
        window_sizes=[100, 200, 'sentence']
    )
    
    trained_params = trainer.train(
        expression="the {party} shall {obligation}",
        expected_context="legal contract agreement law court attorney jurisdiction liability",
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        config=config
    )
    
    trained_params.save('trained_legal_model.json')
    print(f"\nModel saved to: trained_legal_model.json")
    
    # Test the model
    print("\nTesting trained model:")
    trained_expr = TrainedContextAwareExpression(trained_params, registry)
    
    test_cases = [
        "The contractual agreement states the supplier shall deliver materials",
        "The basketball team captain shall lead practice sessions",
        "Legal documentation requires the trustee shall manage assets"
    ]
    
    for test_text in test_cases:
        match = trained_expr.match_with_context(test_text)
        if match:
            print(f"✓ MATCH: '{test_text[:50]}...' -> {match.parameters} (confidence: {match.confidence:.2f})")
        else:
            print(f"✗ NO MATCH: '{test_text[:50]}...'")
    
    return trained_params


def demonstrate_model_comparison():
    """Compare trained vs untrained models"""
    print("\n" + "=" * 60)
    print("Comparing Trained vs Untrained Models")
    print("=" * 60)
    
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Load trained models
    try:
        healthcare_params = TrainedParameters.load('trained_healthcare_model.json')
        financial_params = TrainedParameters.load('trained_financial_model.json')
        
        healthcare_trained = TrainedContextAwareExpression(healthcare_params, registry)
        financial_trained = TrainedContextAwareExpression(financial_params, registry)
        
        # Create untrained baseline
        from semantic_bud_expressions import ContextAwareExpression
        
        healthcare_untrained = ContextAwareExpression(
            expression="patient has {condition}",
            expected_context="medical healthcare hospital doctor clinical treatment",
            context_threshold=0.5,  # Default threshold
            registry=registry
        )
        
        financial_untrained = ContextAwareExpression(
            expression="transfer {amount} to {account}",
            expected_context="banking financial payment transaction money account funds",
            context_threshold=0.5,
            registry=registry
        )
        
        # Test cases
        test_cases = [
            ("Healthcare edge case", "The veterinary clinic reported the patient has infection"),
            ("Financial edge case", "Cryptocurrency exchange: transfer tokens to wallet"),
            ("Clear healthcare", "Hospital records show the patient has pneumonia"),
            ("Clear financial", "Bank app: transfer $100 to savings account")
        ]
        
        print("\nComparison Results:")
        print("-" * 80)
        
        for description, text in test_cases:
            print(f"\nTest: {description}")
            print(f"Text: '{text[:60]}...'")
            
            # Healthcare comparison
            if "healthcare" in description.lower():
                untrained_match = healthcare_untrained.match_with_context(text)
                trained_match = healthcare_trained.match_with_context(text)
                
                print(f"  Healthcare Untrained: {'✓ MATCH' if untrained_match else '✗ NO MATCH'}")
                print(f"  Healthcare Trained:   {'✓ MATCH' if trained_match else '✗ NO MATCH'}", end="")
                if trained_match:
                    print(f" (confidence: {trained_match.confidence:.2f})")
                else:
                    print()
            
            # Financial comparison
            if "financial" in description.lower():
                untrained_match = financial_untrained.match_with_context(text)
                trained_match = financial_trained.match_with_context(text)
                
                print(f"  Financial Untrained:  {'✓ MATCH' if untrained_match else '✗ NO MATCH'}")
                print(f"  Financial Trained:    {'✓ MATCH' if trained_match else '✗ NO MATCH'}", end="")
                if trained_match:
                    print(f" (confidence: {trained_match.confidence:.2f})")
                else:
                    print()
    
    except FileNotFoundError as e:
        print(f"Trained model files not found: {e}")
        print("Run the individual training functions first.")


def main():
    """Run all training examples"""
    print("Context-Aware Expression Training Examples")
    print("=" * 60)
    print("This will train models for different domains and save them to files.")
    print()
    
    try:
        # Train all domain models
        healthcare_params = train_healthcare_model()
        financial_params = train_financial_model()
        ecommerce_params = train_ecommerce_model()
        legal_params = train_legal_model()
        
        # Demonstrate comparison
        demonstrate_model_comparison()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print("\nTrained models saved:")
        print("- trained_healthcare_model.json")
        print("- trained_financial_model.json")
        print("- trained_ecommerce_model.json")
        print("- trained_legal_model.json")
        print("\nYou can now load these models using:")
        print("TrainedContextAwareExpression.from_file('model_file.json')")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()