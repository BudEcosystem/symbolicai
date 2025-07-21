#!/usr/bin/env python3
"""
Test file for dynamic semantic matching functionality
Tests the automatic fallback to dynamic semantic matching when no predefined category exists
"""

import unittest
from semantic_bud_expressions import (
    SemanticBudExpression,
    SemanticParameterTypeRegistry
)


class TestDynamicSemanticMatching(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.registry = SemanticParameterTypeRegistry()
        self.registry.initialize_model()
        
    def test_predefined_category_takes_precedence(self):
        """Test that predefined categories are used when available"""
        # {fruit} is predefined, should use the predefined category
        expr = SemanticBudExpression("I love {fruit}", self.registry)
        match = expr.match("I love grapes")
        
        self.assertIsNotNone(match)
        self.assertEqual(match[0].value, "grapes")
        
    def test_dynamic_matching_for_undefined_category(self):
        """Test dynamic matching when category is not predefined"""
        # {cars} is not predefined, should use dynamic matching
        expr = SemanticBudExpression("I love {cars}", self.registry)
        match = expr.match("I love Ferrari")
        
        self.assertIsNotNone(match)
        self.assertEqual(match[0].value, "Ferrari")
        
    def test_dynamic_matching_semantic_similarity(self):
        """Test that dynamic matching uses semantic similarity"""
        # {vehicle} should match "Rolls Royce" based on semantic similarity
        expr = SemanticBudExpression("I drive a {automobile}", self.registry)
        
        # Should match semantically similar terms
        match1 = expr.match("I drive a Mercedes")
        self.assertIsNotNone(match1)
        self.assertEqual(match1[0].value, "Mercedes")
        
        match2 = expr.match("I drive a Tesla")
        self.assertIsNotNone(match2)
        self.assertEqual(match2[0].value, "Tesla")
        
    def test_dynamic_matching_threshold(self):
        """Test that dynamic matching respects similarity threshold"""
        # Test with very low threshold - should match dissimilar terms
        self.registry.set_dynamic_threshold(0.05)
        expr1 = SemanticBudExpression("I see a {thing}", self.registry)
        
        # Should match anything with very low threshold
        match1 = expr1.match("I see a computer")
        self.assertIsNotNone(match1)
        self.assertEqual(match1[0].value, "computer")
        
        # Now test with higher threshold - should be more selective
        self.registry.set_dynamic_threshold(0.4)
        # Need to clear cache to use new threshold
        self.registry._dynamic_types_cache.clear()
        
        expr2 = SemanticBudExpression("I see a {vehicle}", self.registry)
        
        # Should match car-like things
        match2 = expr2.match("I see a car")
        self.assertIsNotNone(match2)
        
        # Should not match unrelated things
        try:
            match3 = expr2.match("I see a banana")
            if match3:
                # Accessing value should raise error due to low similarity
                _ = match3[0].value
                self.fail("Should not match unrelated terms")
        except ValueError as e:
            # Expected - similarity too low
            self.assertIn("does not semantically match", str(e))
        
    def test_multiple_dynamic_parameters(self):
        """Test expression with multiple dynamic parameters"""
        # Use lower threshold for this test
        self.registry.set_dynamic_threshold(0.15)
        
        expr = SemanticBudExpression(
            "The {animal} chased the {animal}", 
            self.registry
        )
        
        match = expr.match("The cat chased the mouse")
        self.assertIsNotNone(match)
        self.assertEqual(match[0].value, "cat")
        self.assertEqual(match[1].value, "mouse")
        
    def test_mixed_predefined_and_dynamic(self):
        """Test mixing predefined and dynamic parameters"""
        expr = SemanticBudExpression(
            "I ate {fruit} in my {vehicle}",
            self.registry
        )
        
        match = expr.match("I ate apple in my Toyota")
        self.assertIsNotNone(match)
        self.assertEqual(match[0].value, "apple")  # Predefined
        self.assertEqual(match[1].value, "Toyota")  # Dynamic
        
    def test_dynamic_matching_case_insensitive(self):
        """Test that dynamic matching is case insensitive"""
        expr = SemanticBudExpression("I love {Cars}", self.registry)
        match = expr.match("I love ferrari")
        
        self.assertIsNotNone(match)
        self.assertEqual(match[0].value, "ferrari")
        
    def test_disable_dynamic_matching(self):
        """Test ability to disable dynamic matching"""
        # Disable dynamic matching BEFORE creating expression
        self.registry.enable_dynamic_matching(False)
        
        # Should raise error since cars is not predefined and dynamic is disabled
        with self.assertRaises(Exception) as context:
            expr = SemanticBudExpression("I love {cars}", self.registry)
            
        self.assertIn("Undefined parameter type", str(context.exception))


if __name__ == "__main__":
    unittest.main()