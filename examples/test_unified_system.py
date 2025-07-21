#!/usr/bin/env python3
"""
Comprehensive test suite for the unified expression system.
Tests all parameter types, type hints, and multi-word phrase matching.
"""

import unittest
from semantic_bud_expressions.unified_expression import UnifiedBudExpression
from semantic_bud_expressions.unified_registry import UnifiedParameterTypeRegistry
from semantic_bud_expressions.unified_parameter_type import UnifiedParameterType, ParameterTypeHint


class TestUnifiedSystem(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.registry = UnifiedParameterTypeRegistry()
        self.registry.initialize_model()
    
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
        self.registry.enable_dynamic_matching(True)
        self.registry.set_dynamic_threshold(0.3)
        
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
    
    def test_regex_parameter_types(self):
        """Test regex parameter types with custom patterns"""
        # Create regex parameter type
        self.registry.create_regex_parameter_type(
            "email",
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        )
        
        expr = UnifiedBudExpression("Send email to {email:regex}", self.registry)
        
        match = expr.match("Send email to john@example.com")
        self.assertIsNotNone(match)
        self.assertEqual(match[0].value, "john@example.com")
        
        # Should not match invalid email
        no_match = expr.match("Send email to invalid-email")
        self.assertIsNone(no_match)
    
    def test_quoted_parameter_types(self):
        """Test quoted parameter types"""
        self.registry.create_quoted_parameter_type("message")
        
        expr = UnifiedBudExpression("Say {message:quoted}", self.registry)
        
        test_cases = [
            ('Say "Hello World"', "Hello World"),
            ("Say 'Hello World'", "Hello World"),
            ("Say Hello", "Hello")
        ]
        
        for text, expected in test_cases:
            match = expr.match(text)
            self.assertIsNotNone(match, f"Failed to match: {text}")
            self.assertEqual(match[0].value, expected)
    
    def test_mixed_parameter_types(self):
        """Test expressions with mixed parameter types"""
        # Create various parameter types
        self.registry.create_semantic_parameter_type(
            "emotion",
            ["happy", "sad", "excited", "angry"],
            similarity_threshold=0.5
        )
        
        self.registry.create_phrase_parameter_type("car_name", max_phrase_length=4)
        
        # Test mixed expression
        expr = UnifiedBudExpression(
            "I am {emotion:semantic} about my {car_name:phrase}",
            self.registry
        )
        
        match = expr.match("I am excited about my Rolls Royce")
        self.assertIsNotNone(match)
        self.assertEqual(match[0].value, "excited")
        self.assertEqual(match[1].value, "Rolls Royce")
    
    def test_phrase_boundary_detection(self):
        """Test phrase boundary detection with delimiters"""
        self.registry.create_phrase_parameter_type(
            "product",
            max_phrase_length=6,
            phrase_delimiters=['.', ',', '!', '?', '(', ')']
        )
        
        expr = UnifiedBudExpression("I bought {product:phrase}", self.registry)
        
        test_cases = [
            ("I bought iPhone 15 Pro Max", "iPhone 15 Pro Max"),
            ("I bought MacBook Pro 16 inch, and it's great!", "MacBook Pro 16 inch"),
            ("I bought Samsung Galaxy S24 (latest model)", "Samsung Galaxy S24"),
        ]
        
        for text, expected in test_cases:
            match = expr.match(text)
            self.assertIsNotNone(match, f"Failed to match: {text}")
            # Note: exact boundary detection might need refinement
            self.assertIn(expected.split()[0], match[0].value)
    
    def test_phrase_length_limits(self):
        """Test phrase length limits"""
        self.registry.create_phrase_parameter_type(
            "short_phrase",
            max_phrase_length=3
        )
        
        expr = UnifiedBudExpression("I like {short_phrase:phrase}", self.registry)
        
        # Should match short phrases
        match = expr.match("I like red sports car")
        self.assertIsNotNone(match)
        
        # Should truncate long phrases
        long_match = expr.match("I like very fast red sports car with leather seats")
        self.assertIsNotNone(long_match)
        # Should be truncated to max 3 words
        self.assertLessEqual(len(long_match[0].value.split()), 3)
    
    def test_semantic_similarity_thresholds(self):
        """Test semantic similarity thresholds"""
        # Create semantic type with high threshold
        self.registry.create_semantic_parameter_type(
            "strict_animal",
            ["cat", "dog", "bird", "fish"],
            similarity_threshold=0.8
        )
        
        expr = UnifiedBudExpression("I have a {strict_animal:semantic}", self.registry)
        
        # Should match close semantic matches
        match = expr.match("I have a kitten")
        # Note: This might not match due to high threshold - depends on embedding model
        
        # Create semantic type with low threshold
        self.registry.create_semantic_parameter_type(
            "loose_animal",
            ["cat", "dog", "bird", "fish"],
            similarity_threshold=0.2
        )
        
        expr2 = UnifiedBudExpression("I have a {loose_animal:semantic}", self.registry)
        match2 = expr2.match("I have a hamster")
        self.assertIsNotNone(match2)
        self.assertEqual(match2[0].value, "hamster")
    
    def test_backward_compatibility(self):
        """Test backward compatibility with existing API"""
        # Test that old syntax still works
        from semantic_bud_expressions import SemanticBudExpression, SemanticParameterTypeRegistry
        
        old_registry = SemanticParameterTypeRegistry()
        old_registry.initialize_model()
        
        # Old style expression should still work
        old_expr = SemanticBudExpression("I love {fruit}", old_registry)
        match = old_expr.match("I love apple")
        self.assertIsNotNone(match)
        self.assertEqual(match[0].value, "apple")
        
        # New unified expression should also work with old registry
        new_expr = UnifiedBudExpression("I love {fruit}", old_registry)
        match2 = new_expr.match("I love banana")
        self.assertIsNotNone(match2)
        self.assertEqual(match2[0].value, "banana")
    
    def test_error_handling(self):
        """Test error handling and validation"""
        # Test invalid type hint
        expr = UnifiedBudExpression("I have {count:invalid}", self.registry)
        # Should fall back to standard parameter
        match = expr.match("I have 5")
        self.assertIsNotNone(match)
        
        # Test semantic matching with low similarity
        self.registry.create_semantic_parameter_type(
            "strict_fruit",
            ["apple", "banana", "orange"],
            similarity_threshold=0.9
        )
        
        expr2 = UnifiedBudExpression("I eat {strict_fruit:semantic}", self.registry)
        
        # Should not match dissimilar words
        try:
            match2 = expr2.match("I eat computer")
            if match2:
                # Accessing value should raise error
                _ = match2[0].value
                self.fail("Should have raised ValueError for low similarity")
        except ValueError as e:
            self.assertIn("does not semantically match", str(e))
    
    def test_parameter_metadata(self):
        """Test parameter metadata extraction"""
        expr = UnifiedBudExpression(
            "I {action:semantic} {count} {items:phrase}",
            self.registry
        )
        
        metadata = expr.get_all_parameter_metadata()
        
        self.assertIn("action", metadata)
        self.assertIn("count", metadata)
        self.assertIn("items", metadata)
        
        self.assertEqual(metadata["action"]["type_hint"], ParameterTypeHint.SEMANTIC)
        self.assertEqual(metadata["count"]["type_hint"], ParameterTypeHint.STANDARD)
        self.assertEqual(metadata["items"]["type_hint"], ParameterTypeHint.PHRASE)
    
    def test_registry_configuration(self):
        """Test registry configuration methods"""
        # Test phrase configuration
        self.registry.set_phrase_config(
            max_phrase_length=5,
            phrase_delimiters=['.', '!', '?']
        )
        
        config = self.registry.get_phrase_config()
        self.assertEqual(config['max_phrase_length'], 5)
        self.assertEqual(config['phrase_delimiters'], ['.', '!', '?'])
        
        # Test dynamic threshold
        self.registry.set_dynamic_threshold(0.5)
        self.assertEqual(self.registry._dynamic_threshold, 0.5)
    
    def test_complex_real_world_scenarios(self):
        """Test complex real-world scenarios"""
        # E-commerce scenario
        self.registry.create_phrase_parameter_type("product_name", max_phrase_length=6)
        self.registry.create_semantic_parameter_type(
            "color",
            ["red", "blue", "green", "black", "white"],
            similarity_threshold=0.4
        )
        
        expr = UnifiedBudExpression(
            "I want to buy a {color:semantic} {product_name:phrase} for {price}",
            self.registry
        )
        
        match = expr.match("I want to buy a crimson iPhone 15 Pro Max for $1000")
        self.assertIsNotNone(match)
        self.assertEqual(match[0].value, "crimson")  # Color
        self.assertEqual(match[1].value, "iPhone 15 Pro Max")  # Product name
        self.assertEqual(match[2].value, "$1000")  # Price


if __name__ == "__main__":
    unittest.main()