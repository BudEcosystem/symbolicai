#!/usr/bin/env python3
"""
Comprehensive test of context-aware matching across different industries,
text lengths, and threshold optimization strategies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import (
    ContextAwareExpression,
    UnifiedParameterTypeRegistry,
    EnhancedUnifiedParameterTypeRegistry
)
import time
from typing import List, Tuple, Dict
import statistics


class IndustryContextTester:
    """Test context-aware matching across different industries"""
    
    def __init__(self):
        self.registry = EnhancedUnifiedParameterTypeRegistry()
        self.registry.initialize_model()
        self.results = []
        
    def test_industry(
        self, 
        industry_name: str,
        expression: str,
        expected_context: str,
        test_cases: List[Tuple[str, bool, str]],  # (text, should_match, description)
        thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
    ):
        """Test an industry with multiple thresholds"""
        print(f"\n{'='*80}")
        print(f"Industry: {industry_name}")
        print(f"Expression: {expression}")
        print(f"Expected Context: {expected_context}")
        print(f"{'='*80}")
        
        threshold_results = {}
        
        for threshold in thresholds:
            print(f"\nTesting with threshold: {threshold}")
            print("-" * 40)
            
            expr = ContextAwareExpression(
                expression=expression,
                expected_context=expected_context,
                context_threshold=threshold,
                context_window='auto',
                registry=self.registry
            )
            
            correct = 0
            total = len(test_cases)
            
            for text, should_match, description in test_cases:
                start_time = time.time()
                match = expr.match_with_context(text)
                elapsed = (time.time() - start_time) * 1000  # ms
                
                matched = match is not None
                is_correct = matched == should_match
                
                if is_correct:
                    correct += 1
                
                status = "✓" if is_correct else "✗"
                match_status = "MATCHED" if matched else "NO MATCH"
                expected_status = "SHOULD MATCH" if should_match else "SHOULD NOT MATCH"
                
                # Get similarity score even if no match
                if match:
                    similarity = match.context_similarity
                    context = match.context_text[:50] + "..." if len(match.context_text) > 50 else match.context_text
                else:
                    # Try to get context and similarity manually for analysis
                    test_expr = ContextAwareExpression(
                        expression=expression,
                        expected_context=expected_context,
                        context_threshold=0.0,  # Set to 0 to always get result
                        context_window='auto',
                        registry=self.registry
                    )
                    test_match = test_expr.match_with_context(text)
                    if test_match:
                        similarity = test_match.context_similarity
                        context = test_match.context_text[:50] + "..."
                    else:
                        similarity = 0.0
                        context = "No context extracted"
                
                print(f"{status} {description[:40]:<40} | {match_status:<10} ({expected_status:<15}) | Similarity: {similarity:.3f} | Time: {elapsed:.1f}ms")
                if not is_correct:
                    print(f"   Context: '{context}'")
            
            accuracy = correct / total * 100
            threshold_results[threshold] = accuracy
            print(f"\nAccuracy at threshold {threshold}: {accuracy:.1f}%")
        
        # Find optimal threshold
        optimal_threshold = max(threshold_results, key=threshold_results.get)
        print(f"\n{'*'*60}")
        print(f"Optimal threshold for {industry_name}: {optimal_threshold} (Accuracy: {threshold_results[optimal_threshold]:.1f}%)")
        print(f"{'*'*60}")
        
        self.results.append({
            'industry': industry_name,
            'optimal_threshold': optimal_threshold,
            'accuracy': threshold_results[optimal_threshold],
            'all_results': threshold_results
        })
        
        return optimal_threshold, threshold_results


def main():
    tester = IndustryContextTester()
    
    # 1. Healthcare Industry
    tester.test_industry(
        industry_name="Healthcare",
        expression="patient has {symptom} and needs {treatment}",
        expected_context="medical healthcare diagnosis treatment hospital",
        test_cases=[
            # Should match - medical context
            ("The doctor examined the patient. The patient has fever and needs antibiotics", True, "Clear medical context"),
            ("In the emergency room, the patient has chest pain and needs immediate care", True, "Hospital setting"),
            ("Medical records show the patient has diabetes and needs insulin therapy", True, "Medical records context"),
            ("During the consultation, the patient has anxiety and needs counseling", True, "Medical consultation"),
            
            # Should NOT match - non-medical context
            ("My computer has virus and needs antivirus software", False, "Tech context"),
            ("The car has engine problems and needs repair", False, "Automotive context"),
            ("The plant has yellow leaves and needs fertilizer", False, "Gardening context"),
            ("The building has cracks and needs renovation", False, "Construction context"),
            
            # Edge cases
            ("The veterinary patient has infection and needs treatment", True, "Veterinary medical"),
            ("In our clinic today, the patient has allergies and needs medication", True, "Clinic context"),
        ]
    )
    
    # 2. Finance Industry
    tester.test_industry(
        industry_name="Finance",
        expression="transfer {amount} to {account}",
        expected_context="banking financial transaction money payment account",
        test_cases=[
            # Should match - financial context
            ("Please login to your bank account and transfer $1000 to savings account", True, "Banking context"),
            ("The financial advisor suggested to transfer 5000 to investment account", True, "Financial advisor"),
            ("Complete the wire transfer. Transfer 10000 to checking account", True, "Wire transfer"),
            ("Using online banking, transfer 500 to credit account", True, "Online banking"),
            
            # Should NOT match - non-financial context
            ("Please transfer the files to backup folder", False, "File transfer"),
            ("Transfer the patient to ICU immediately", False, "Medical transfer"),
            ("Transfer 50GB to external drive", False, "Data transfer"),
            ("The player wants to transfer to another team", False, "Sports transfer"),
            
            # Edge cases
            ("PayPal transaction: transfer 100 to merchant account", True, "Digital payment"),
            ("Cryptocurrency wallet: transfer 0.5 BTC to cold wallet", True, "Crypto context"),
        ]
    )
    
    # 3. E-commerce Industry
    tester.test_industry(
        industry_name="E-commerce",
        expression="add {product} to {destination}",
        expected_context="shopping cart purchase buy online store checkout",
        test_cases=[
            # Should match - shopping context
            ("Browse our online store and add iPhone to cart", True, "Online store"),
            ("Ready to checkout? Add warranty to order", True, "Checkout process"),
            ("Shopping for electronics. Add laptop to wishlist", True, "Shopping context"),
            ("Complete your purchase: add shoes to basket", True, "Purchase context"),
            
            # Should NOT match - non-shopping context
            ("In the recipe, add sugar to mixture", False, "Cooking context"),
            ("Add user to database", False, "Technical context"),
            ("Add employee to payroll", False, "HR context"),
            ("Add chlorine to pool", False, "Maintenance context"),
            
            # Edge cases
            ("Amazon shopping: add books to cart", True, "Brand name shopping"),
            ("During Black Friday sale, add TV to cart", True, "Sale event context"),
        ]
    )
    
    # 4. Education Industry
    tester.test_industry(
        industry_name="Education",
        expression="student {action} {subject}",
        expected_context="school education learning classroom teacher academic",
        test_cases=[
            # Should match - educational context
            ("In the classroom today, the student completed mathematics", True, "Classroom setting"),
            ("The teacher noted that the student excelled science", True, "Teacher observation"),
            ("According to academic records, the student failed history", True, "Academic records"),
            ("During the school year, the student studied literature", True, "School year context"),
            
            # Should NOT match - non-educational context
            ("The driving student passed test", False, "Driving context"),
            ("The yoga student mastered pose", False, "Yoga context"),
            ("The apprentice student learned welding", False, "Trade context"),
            ("The cooking student prepared dish", False, "Culinary context"),
            
            # Edge cases
            ("At university, the student researched biology", True, "University context"),
            ("Online learning platform: student completed programming", True, "E-learning context"),
        ]
    )
    
    # 5. Legal Industry
    tester.test_industry(
        industry_name="Legal",
        expression="the {party} shall {obligation}",
        expected_context="legal contract agreement law court attorney jurisdiction",
        test_cases=[
            # Should match - legal context
            ("Per the contract terms, the buyer shall pay damages", True, "Contract context"),
            ("The court ruled that the defendant shall appear tomorrow", True, "Court ruling"),
            ("According to the agreement, the tenant shall vacate premises", True, "Legal agreement"),
            ("The attorney advised that the client shall provide documents", True, "Attorney advice"),
            
            # Should NOT match - non-legal context
            ("The player shall score goals", False, "Sports context"),
            ("The chef shall prepare dinner", False, "Restaurant context"),
            ("The driver shall maintain speed", False, "Driving context"),
            ("The artist shall complete painting", False, "Art context"),
            
            # Edge cases
            ("In this jurisdiction, the company shall comply regulations", True, "Jurisdiction mentioned"),
            ("The legal department states the vendor shall deliver goods", True, "Legal department"),
        ]
    )
    
    # 6. Test different text lengths
    print("\n\n" + "="*80)
    print("TESTING DIFFERENT TEXT LENGTHS")
    print("="*80)
    
    # Short text (1-2 sentences)
    tester.test_industry(
        industry_name="Short Text Medical",
        expression="patient needs {treatment}",
        expected_context="medical healthcare hospital",
        test_cases=[
            ("Patient needs surgery.", True, "Very short medical"),
            ("Car needs repair.", False, "Very short non-medical"),
            ("The patient needs medication urgently.", True, "Short medical"),
            ("The computer needs update.", False, "Short non-medical"),
        ],
        thresholds=[0.3, 0.4, 0.5, 0.6]
    )
    
    # Medium text (paragraph)
    medium_medical = """The patient was admitted to the emergency department yesterday evening. 
    After thorough examination and running several tests, the medical team determined 
    that the patient needs immediate intervention."""
    
    medium_non_medical = """The project was submitted to the review committee yesterday evening.
    After thorough analysis and running several checks, the technical team determined
    that the system needs immediate updates."""
    
    tester.test_industry(
        industry_name="Medium Text Medical",
        expression="patient needs {treatment}",
        expected_context="medical healthcare hospital doctor",
        test_cases=[
            (medium_medical, True, "Medium medical text"),
            (medium_non_medical, False, "Medium non-medical text"),
        ],
        thresholds=[0.3, 0.4, 0.5, 0.6]
    )
    
    # Long text (multiple paragraphs)
    long_medical = """The medical facility has been serving the community for over 50 years.
    Our team of healthcare professionals is dedicated to providing the best care possible.
    We have state-of-the-art equipment and follow the latest medical protocols.
    
    In today's case, after consulting with specialists and reviewing all test results,
    we have determined that the patient needs surgical intervention. This decision was
    made after careful consideration of all available treatment options."""
    
    long_non_medical = """The technology company has been serving clients for over 20 years.
    Our team of software engineers is dedicated to providing the best solutions possible.
    We have cutting-edge infrastructure and follow the latest development practices.
    
    In today's meeting, after consulting with architects and reviewing all requirements,
    we have determined that the system needs major refactoring. This decision was
    made after careful consideration of all available technical options."""
    
    tester.test_industry(
        industry_name="Long Text Medical",
        expression="patient needs {treatment}",
        expected_context="medical healthcare hospital treatment",
        test_cases=[
            (long_medical, True, "Long medical text"),
            (long_non_medical, False, "Long non-medical text"),
        ],
        thresholds=[0.3, 0.4, 0.5, 0.6]
    )
    
    # Print summary
    print("\n\n" + "="*80)
    print("SUMMARY: OPTIMAL THRESHOLDS BY INDUSTRY")
    print("="*80)
    
    for result in tester.results:
        print(f"\n{result['industry']}:")
        print(f"  Optimal Threshold: {result['optimal_threshold']}")
        print(f"  Best Accuracy: {result['accuracy']:.1f}%")
        print(f"  All Results: {result['all_results']}")
    
    # Calculate overall statistics
    all_thresholds = [r['optimal_threshold'] for r in tester.results]
    print(f"\n{'*'*60}")
    print(f"OVERALL RECOMMENDATIONS:")
    print(f"  Average Optimal Threshold: {statistics.mean(all_thresholds):.2f}")
    print(f"  Median Optimal Threshold: {statistics.median(all_thresholds):.2f}")
    print(f"  Range: {min(all_thresholds)} - {max(all_thresholds)}")
    print(f"{'*'*60}")
    
    print("\nBEST PRACTICES:")
    print("1. Start with threshold 0.5 as default")
    print("2. For strict matching (legal, medical): use 0.5-0.6")
    print("3. For flexible matching (e-commerce, general): use 0.4-0.5")
    print("4. For very short texts: use lower thresholds (0.3-0.4)")
    print("5. Test with your specific domain data to find optimal threshold")
    print("6. Consider using 'auto' context window for best results")


if __name__ == "__main__":
    main()