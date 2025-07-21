#!/usr/bin/env python3
"""
Test suite for context-aware matching using TDD approach
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from semantic_bud_expressions import (
    UnifiedParameterTypeRegistry,
    UnifiedBudExpression
)


class TestContextAwareMatching(unittest.TestCase):
    """Test context-aware expression matching"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.registry = UnifiedParameterTypeRegistry()
        cls.registry.initialize_model()
    
    def test_context_aware_expression_creation(self):
        """Test creating a context-aware expression"""
        try:
            from semantic_bud_expressions import ContextAwareExpression
            
            expr = ContextAwareExpression(
                expression="I {emotion} {vehicle}",
                expected_context="discussion about cars and transportation",
                context_threshold=0.7,
                registry=self.registry
            )
            
            self.assertIsNotNone(expr)
            self.assertEqual(expr.expected_context, "discussion about cars and transportation")
            self.assertEqual(expr.context_threshold, 0.7)
            
        except ImportError:
            self.skipTest("ContextAwareExpression not yet implemented")
    
    def test_basic_context_matching(self):
        """Test basic context matching with preceding text"""
        try:
            from semantic_bud_expressions import ContextAwareExpression
            
            expr = ContextAwareExpression(
                expression="I {emotion} {vehicle}",
                expected_context="cars and automotive technology",
                context_threshold=0.6,
                registry=self.registry
            )
            
            # Text with matching context
            text1 = "The automotive industry has revolutionized transportation. Cars are amazing. I love Tesla"
            match1 = expr.match_with_context(text1)
            self.assertIsNotNone(match1, "Should match when context is about cars")
            
            # Text with non-matching context
            text2 = "The weather is beautiful today. Birds are singing. I love Tesla"
            match2 = expr.match_with_context(text2)
            self.assertIsNone(match2, "Should not match when context is about weather")
            
        except ImportError:
            self.skipTest("ContextAwareExpression not yet implemented")
    
    def test_context_extraction_strategies(self):
        """Test different context extraction strategies"""
        try:
            from semantic_bud_expressions import ContextAwareExpression
            
            text = "Paragraph one about nature. Paragraph two about cars and vehicles. I adore Ferrari"
            
            # Test word-based extraction
            expr_words = ContextAwareExpression(
                expression="I {emotion} {vehicle}",
                expected_context="cars and vehicles",
                context_window=10,  # Extract 10 words before match
                registry=self.registry
            )
            match_words = expr_words.match_with_context(text)
            self.assertIsNotNone(match_words)
            
            # Test sentence-based extraction
            expr_sentence = ContextAwareExpression(
                expression="I {emotion} {vehicle}",
                expected_context="cars and vehicles",
                context_window='sentence',  # Extract previous sentence
                registry=self.registry
            )
            match_sentence = expr_sentence.match_with_context(text)
            self.assertIsNotNone(match_sentence)
            
            # Test paragraph-based extraction
            expr_paragraph = ContextAwareExpression(
                expression="I {emotion} {vehicle}",
                expected_context="nature",
                context_window='paragraph',  # Extract current paragraph
                registry=self.registry
            )
            match_paragraph = expr_paragraph.match_with_context(text)
            # Should not match because current paragraph is about cars, not nature
            self.assertIsNone(match_paragraph)
            
        except ImportError:
            self.skipTest("ContextAwareExpression not yet implemented")
    
    def test_multiple_matches_different_contexts(self):
        """Test handling multiple potential matches with different contexts"""
        try:
            from semantic_bud_expressions import ContextAwareExpression
            
            expr = ContextAwareExpression(
                expression="I {emotion} {item}",
                expected_context="technology and gadgets",
                context_threshold=0.7,
                registry=self.registry
            )
            
            text = """
            Discussion about food and cooking. I love pizza.
            Now let's talk about technology and smartphones. I love iPhone.
            Back to recipes and ingredients. I love chocolate.
            """
            
            matches = expr.find_all_with_context(text)
            
            # Should only match the iPhone mention (technology context)
            self.assertEqual(len(matches), 1)
            self.assertIn("iPhone", matches[0].matched_text)
            
        except (ImportError, AttributeError):
            self.skipTest("ContextAwareExpression not yet implemented")
    
    def test_context_similarity_threshold(self):
        """Test context similarity threshold behavior"""
        try:
            from semantic_bud_expressions import ContextAwareExpression
            
            text = "Automobiles and transportation systems. I enjoy Mercedes"
            
            # High threshold - strict matching
            expr_strict = ContextAwareExpression(
                expression="I {emotion} {vehicle}",
                expected_context="cars and automotive",
                context_threshold=0.9,  # Very strict
                registry=self.registry
            )
            match_strict = expr_strict.match_with_context(text)
            # Might not match due to high threshold
            
            # Low threshold - lenient matching
            expr_lenient = ContextAwareExpression(
                expression="I {emotion} {vehicle}",
                expected_context="cars and automotive",
                context_threshold=0.3,  # Very lenient
                registry=self.registry
            )
            match_lenient = expr_lenient.match_with_context(text)
            self.assertIsNotNone(match_lenient, "Should match with low threshold")
            
        except ImportError:
            self.skipTest("ContextAwareExpression not yet implemented")
    
    def test_long_context_handling(self):
        """Test handling of long contexts (paragraphs)"""
        try:
            from semantic_bud_expressions import ContextAwareExpression
            
            long_context = """
            The automotive industry has undergone significant transformation in recent decades.
            From the early days of combustion engines to modern electric vehicles, cars have
            evolved dramatically. Manufacturers like Tesla, BMW, and Mercedes have pioneered
            new technologies including autonomous driving, advanced safety systems, and 
            sustainable materials. The future of transportation looks increasingly electric
            and connected, with smart cities integrating vehicle networks.
            """
            
            expr = ContextAwareExpression(
                expression="I {emotion} {vehicle}",
                expected_context="automotive industry electric vehicles autonomous driving",
                context_threshold=0.6,
                context_window='auto',  # Automatically determine best window
                registry=self.registry
            )
            
            text = long_context + " After all this progress, I admire Tesla"
            match = expr.match_with_context(text)
            
            self.assertIsNotNone(match)
            # Should extract relevant context despite length
            self.assertIsNotNone(expr.get_extracted_context())
            
        except (ImportError, AttributeError):
            self.skipTest("ContextAwareExpression not yet implemented")
    
    def test_context_caching(self):
        """Test context embedding caching for performance"""
        try:
            from semantic_bud_expressions import ContextAwareExpression
            import time
            
            expr = ContextAwareExpression(
                expression="I {emotion} {item}",
                expected_context="technology discussion",
                registry=self.registry
            )
            
            text = "Talking about smartphones and gadgets. I love technology"
            
            # First match - compute embeddings
            start1 = time.time()
            match1 = expr.match_with_context(text)
            time1 = time.time() - start1
            
            # Second match - should use cached embeddings
            start2 = time.time()
            match2 = expr.match_with_context(text)
            time2 = time.time() - start2
            
            # Cached match should be faster
            self.assertLess(time2, time1 * 0.5, "Cached match should be at least 2x faster")
            
        except ImportError:
            self.skipTest("ContextAwareExpression not yet implemented")
    
    def test_context_extraction_methods(self):
        """Test different context extraction methods"""
        try:
            from semantic_bud_expressions import ContextAwareExpression
            
            text = "First sentence. Second sentence about cars. I love Ferrari."
            
            expr = ContextAwareExpression(
                expression="I {emotion} {vehicle}",
                expected_context="cars",
                registry=self.registry
            )
            
            # Test extracting context at different positions
            context_before = expr.extract_context(text, position='before', window=20)
            self.assertIn("cars", context_before)
            
            context_sentence = expr.extract_context(text, position='sentence')
            self.assertEqual(context_sentence, "Second sentence about cars.")
            
            context_auto = expr.extract_context(text, position='auto')
            # Should intelligently extract relevant context
            self.assertIsNotNone(context_auto)
            
        except (ImportError, AttributeError):
            self.skipTest("Context extraction methods not yet implemented")
    
    def test_context_comparison_strategies(self):
        """Test different context comparison strategies"""
        try:
            from semantic_bud_expressions import ContextAwareExpression
            
            expr = ContextAwareExpression(
                expression="I {emotion} {vehicle}",
                expected_context="automotive and cars",
                context_comparison='chunked_mean',  # Average sentence embeddings
                registry=self.registry
            )
            
            text = "Cars are great. Vehicles revolutionized transport. I adore BMW"
            match = expr.match_with_context(text)
            
            self.assertIsNotNone(match)
            
            # Test different comparison methods
            similarity_direct = expr.compare_context_direct("cars and vehicles", "automotive and cars")
            similarity_chunked = expr.compare_context_chunked("cars and vehicles", "automotive and cars")
            
            # Both should give reasonable similarity scores
            self.assertGreater(similarity_direct, 0.5)
            self.assertGreater(similarity_chunked, 0.5)
            
        except (ImportError, AttributeError):
            self.skipTest("Context comparison strategies not yet implemented")
    
    def test_edge_cases(self):
        """Test edge cases for context matching"""
        try:
            from semantic_bud_expressions import ContextAwareExpression
            
            expr = ContextAwareExpression(
                expression="I {emotion} {item}",
                expected_context="technology",
                registry=self.registry
            )
            
            # Empty text
            match_empty = expr.match_with_context("")
            self.assertIsNone(match_empty)
            
            # No context before match
            match_no_context = expr.match_with_context("I love gadgets")
            # Should handle gracefully, possibly with lower similarity
            
            # Very short context
            match_short = expr.match_with_context("Tech. I love phones")
            # Should still work with short context
            
            # Multiple spaces and newlines
            match_whitespace = expr.match_with_context("Technology\n\n\n   I love    computers")
            self.assertIsNotNone(match_whitespace)
            
        except ImportError:
            self.skipTest("ContextAwareExpression not yet implemented")


class TestContextAwareExamples(unittest.TestCase):
    """Test real-world examples of context-aware matching"""
    
    def setUp(self):
        self.registry = UnifiedParameterTypeRegistry()
        self.registry.initialize_model()
    
    def test_customer_support_context(self):
        """Test customer support scenario with context"""
        try:
            from semantic_bud_expressions import ContextAwareExpression
            
            # Only respond to refund requests in complaint context
            expr = ContextAwareExpression(
                expression="I want a {action}",
                expected_context="product broken defective not working complaint",
                context_threshold=0.7,
                registry=self.registry
            )
            
            # Complaint context - should match
            complaint = "My phone arrived broken and doesn't turn on. I want a refund"
            match1 = expr.match_with_context(complaint)
            self.assertIsNotNone(match1)
            
            # General inquiry - should not match
            inquiry = "How long is the warranty? I want a refund"
            match2 = expr.match_with_context(inquiry)
            self.assertIsNone(match2)
            
        except ImportError:
            self.skipTest("ContextAwareExpression not yet implemented")
    
    def test_content_moderation_context(self):
        """Test content moderation with context awareness"""
        try:
            from semantic_bud_expressions import ContextAwareExpression
            
            # Flag certain words only in aggressive context
            expr = ContextAwareExpression(
                expression="you {word}",
                expected_context="hate angry fight argument insult",
                context_threshold=0.6,
                registry=self.registry
            )
            
            # Aggressive context - should flag
            aggressive = "I hate people like you. You fool"
            match1 = expr.match_with_context(aggressive)
            self.assertIsNotNone(match1)
            
            # Friendly context - should not flag
            friendly = "Thanks for helping! You fool around too much though"
            match2 = expr.match_with_context(friendly)
            self.assertIsNone(match2)
            
        except ImportError:
            self.skipTest("ContextAwareExpression not yet implemented")
    
    def test_intent_classification_context(self):
        """Test intent classification with context"""
        try:
            from semantic_bud_expressions import ContextAwareExpression
            
            # Buy intent only in product context
            buy_expr = ContextAwareExpression(
                expression="I want to {action} it",
                expected_context="product features price specifications review",
                context_threshold=0.65,
                registry=self.registry
            )
            
            # Product context
            product_text = "This laptop has great specs and reasonable price. I want to buy it"
            match1 = buy_expr.match_with_context(product_text)
            self.assertIsNotNone(match1)
            
            # Non-product context
            other_text = "The movie was entertaining. I want to see it"
            match2 = buy_expr.match_with_context(other_text)
            self.assertIsNone(match2)
            
        except ImportError:
            self.skipTest("ContextAwareExpression not yet implemented")


if __name__ == '__main__':
    unittest.main()