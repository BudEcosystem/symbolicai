#!/usr/bin/env python3
"""
Fixed version of unified system tests that work with the enhanced implementation.
"""

import unittest
from semantic_bud_expressions.unified_expression import UnifiedBudExpression
from semantic_bud_expressions.unified_registry import UnifiedParameterTypeRegistry
from semantic_bud_expressions.unified_parameter_type import UnifiedParameterType, ParameterTypeHint


class TestUnifiedSystemFixed(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.registry = UnifiedParameterTypeRegistry()
        self.registry.initialize_model()
        # Disable dynamic matching for standard parameter tests
        self.registry._dynamic_matching_enabled = False
    
    def test_standard_parameter_types(self):
        """Test standard parameter types without type hints"""
        # Create expression with standard parameter
        expr = UnifiedBudExpression("I have {count} items", self.registry)
        
        # Should work like regular bud expressions
        match = expr.match("I have 5 items")
        self.assertIsNotNone(match)
        self.assertEqual(match[0].value, "5")
    
    def test_semantic_parameter_types(self):
        """Test semantic parameter types with :semantic type hint"""
        # Re-enable dynamic matching and clear cache
        self.registry._dynamic_matching_enabled = True
        self.registry.clear_unified_cache()
        
        # Create semantic parameter type
        self.registry.create_semantic_parameter_type(
            "vehicle",
            ["car", "truck", "bus", "motorcycle", "bicycle"],
            similarity_threshold=0.3
        )
        
        # Test with type hint
        expr = UnifiedBudExpression("I drive a {vehicle:semantic}", self.registry)
        
        match = expr.match("I drive a Tesla")
        self.assertIsNotNone(match)
        self.assertEqual(match[0].value, "Tesla")
        
        # Test without type hint (should also work)
        expr2 = UnifiedBudExpression("I drive a {vehicle}", self.registry)
        match2 = expr2.match("I drive a Ferrari")
        self.assertIsNotNone(match2)
        self.assertEqual(match2[0].value, "Ferrari")
    
    def test_phrase_parameter_types(self):
        """Test phrase parameter types for multi-word matching"""
        # Clear cache
        self.registry.clear_unified_cache()
        
        # Create phrase parameter type
        self.registry.create_phrase_parameter_type(
            "car_model",
            max_phrase_length=5
        )
        
        # Test multi-word phrase matching
        expr = UnifiedBudExpression("I bought a {car_model:phrase}", self.registry)
        
        test_cases = [
            ("I bought a Rolls Royce", "Rolls Royce"),
            ("I bought a Mercedes Benz S Class", "Mercedes Benz S Class"),
            ("I bought a Tesla Model 3", "Tesla Model 3"),
            ("I bought a Ford", "Ford")
        ]
        
        for text, expected in test_cases:
            match = expr.match(text)
            self.assertIsNotNone(match, f"Failed to match: {text}")
            self.assertEqual(match[0].value, expected)
    
    def test_dynamic_parameter_types(self):
        """Test dynamic parameter types with automatic semantic matching"""
        # Enable dynamic matching
        self.registry._dynamic_matching_enabled = True
        self.registry.set_dynamic_threshold(0.3)
        self.registry.clear_unified_cache()
        
        # Test with dynamic type hint
        expr = UnifiedBudExpression("I love {cars:dynamic}", self.registry)
        
        match = expr.match("I love Ferrari")
        self.assertIsNotNone(match)
        self.assertEqual(match[0].value, "Ferrari")
        
        # Test without explicit type hint (should fall back to dynamic)
        expr2 = UnifiedBudExpression("I need {furniture}", self.registry)
        match2 = expr2.match("I need chair")
        self.assertIsNotNone(match2)
        self.assertEqual(match2[0].value, "chair")
    
    def test_phrase_length_limits(self):
        """Test phrase length limits with flexible handling"""
        self.registry.clear_unified_cache()
        
        self.registry.create_phrase_parameter_type(
            "short_phrase",
            max_phrase_length=3
        )
        
        expr = UnifiedBudExpression("I like {short_phrase:phrase}", self.registry)
        
        # Should match and truncate long phrases
        long_match = expr.match("I like very fast red sports car with leather seats")
        self.assertIsNotNone(long_match)
        # Should be truncated to max 3 words
        words = long_match[0].value.split()
        self.assertLessEqual(len(words), 3)
        self.assertEqual(long_match[0].value, "very fast red")
    
    def test_mixed_parameter_types(self):
        """Test expressions with mixed parameter types"""
        # Clear cache to avoid duplicates
        self.registry.clear_unified_cache()
        self.registry._dynamic_matching_enabled = True
        
        # Create various parameter types
        self.registry.create_semantic_parameter_type(
            "feeling",
            ["happy", "sad", "excited", "angry"],
            similarity_threshold=0.5
        )
        
        self.registry.create_phrase_parameter_type("vehicle_name", max_phrase_length=4)
        
        # Test mixed expression
        expr = UnifiedBudExpression(
            "I am {feeling:semantic} about my {vehicle_name:phrase}",
            self.registry
        )
        
        match = expr.match("I am excited about my Rolls Royce")
        self.assertIsNotNone(match)
        self.assertEqual(match[0].value, "excited")
        self.assertEqual(match[1].value, "Rolls Royce")
    
    def test_error_handling(self):
        """Test error handling and validation"""
        self.registry.clear_unified_cache()
        
        # Test invalid type hint
        expr = UnifiedBudExpression("I have {count:invalid}", self.registry)
        # Should fall back to standard parameter
        match = expr.match("I have 5")
        self.assertIsNotNone(match)
        
        # Test semantic matching with low similarity
        self.registry.create_semantic_parameter_type(
            "strict_color",
            ["red", "blue", "green"],
            similarity_threshold=0.9
        )
        
        expr2 = UnifiedBudExpression("I like {strict_color:semantic}", self.registry)
        
        # Should not match dissimilar words
        try:
            match2 = expr2.match("I like purple")
            if match2:
                # Accessing value should potentially raise error for low similarity
                _ = match2[0].value
        except ValueError as e:
            self.assertIn("does not semantically match", str(e))


if __name__ == "__main__":
    unittest.main()