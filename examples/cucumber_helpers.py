#!/usr/bin/env python3
"""
Helper functions for creating true Cucumber-style exact match parameter types.
NO semantic matching - only exact matches from predefined lists.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from typing import List, Callable, Optional, Any
from semantic_bud_expressions import ParameterType, ParameterTypeRegistry


class CucumberParameterType:
    """Helper class for creating Cucumber-style exact match parameter types"""
    
    @staticmethod
    def create_exact_match(
        name: str,
        values: List[str],
        case_sensitive: bool = True,
        transformer: Optional[Callable] = None
    ) -> ParameterType:
        """
        Create an exact match parameter type (Cucumber-style).
        
        Args:
            name: Parameter name (e.g., 'fruit', 'vehicle')
            values: List of exact values to match
            case_sensitive: Whether matching is case-sensitive
            transformer: Optional transformer function
            
        Returns:
            ParameterType configured for exact matching
        """
        if not case_sensitive:
            # Add case variations
            all_values = []
            for v in values:
                all_values.extend([v.lower(), v.upper(), v.capitalize()])
                # Also add the original
                if v not in all_values:
                    all_values.append(v)
            values = list(set(all_values))  # Remove duplicates
        
        # Escape special regex characters in values
        escaped_values = [re.escape(v) for v in values]
        pattern = "|".join(escaped_values)
        
        # Default transformer normalizes case if case-insensitive
        if transformer is None and not case_sensitive:
            transformer = lambda x: x.lower()
        
        return ParameterType(
            name=name,
            regexp=pattern,
            type=str,
            transformer=transformer
        )
    
    @staticmethod
    def create_enum_match(
        name: str,
        enum_class: type,
        case_sensitive: bool = False
    ) -> ParameterType:
        """
        Create parameter type from Python enum.
        
        Args:
            name: Parameter name
            enum_class: Python Enum class
            case_sensitive: Whether matching is case-sensitive
            
        Returns:
            ParameterType that matches enum values
        """
        values = [e.value for e in enum_class]
        return CucumberParameterType.create_exact_match(
            name=name,
            values=values,
            case_sensitive=case_sensitive,
            transformer=lambda x: enum_class(x.lower() if not case_sensitive else x)
        )


def create_cucumber_registry(definitions: dict) -> ParameterTypeRegistry:
    """
    Create a registry with Cucumber-style exact match parameters.
    
    Args:
        definitions: Dict of parameter_name -> list of values
        
    Example:
        registry = create_cucumber_registry({
            'fruit': ['apple', 'banana', 'orange'],
            'color': ['red', 'blue', 'green']
        })
    """
    registry = ParameterTypeRegistry()
    
    for param_name, values in definitions.items():
        param_type = CucumberParameterType.create_exact_match(
            name=param_name,
            values=values,
            case_sensitive=False  # Default to case-insensitive
        )
        registry.define_parameter_type(param_type)
    
    return registry


# Example usage
if __name__ == "__main__":
    from semantic_bud_expressions import budExpression
    from enum import Enum
    
    print("Cucumber-Style Helpers Demo")
    print("=" * 60)
    print()
    
    # Method 1: Using the helper class
    print("1. Using CucumberParameterType helper:")
    registry = ParameterTypeRegistry()
    
    # Create exact match parameters
    fruit_type = CucumberParameterType.create_exact_match(
        name="fruit",
        values=["apple", "banana", "orange", "grape"],
        case_sensitive=False
    )
    registry.define_parameter_type(fruit_type)
    
    vehicle_type = CucumberParameterType.create_exact_match(
        name="vehicle",
        values=["car", "truck", "bus", "motorcycle"],
        case_sensitive=False
    )
    registry.define_parameter_type(vehicle_type)
    
    expr = budExpression("I love {fruit} and drive a {vehicle}", registry)
    test = "I love APPLE and drive a Car"
    match = expr.match(test)
    if match:
        print(f"  ✓ Matched: fruit={match[0].value}, vehicle={match[1].value}")
    print()
    
    # Method 2: Using the quick registry creator
    print("2. Using create_cucumber_registry:")
    registry2 = create_cucumber_registry({
        'fruit': ['apple', 'banana', 'orange'],
        'color': ['red', 'blue', 'green', 'yellow'],
        'size': ['small', 'medium', 'large']
    })
    
    expr2 = budExpression("The {size} {color} {fruit}", registry2)
    test2 = "The LARGE RED apple"
    match2 = expr2.match(test2)
    if match2:
        print(f"  ✓ Matched: size={match2[0].value}, color={match2[1].value}, fruit={match2[2].value}")
    print()
    
    # Method 3: Using enums
    print("3. Using Python Enums:")
    
    class Status(Enum):
        PENDING = "pending"
        APPROVED = "approved"
        REJECTED = "rejected"
    
    class Priority(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
    
    registry3 = ParameterTypeRegistry()
    
    status_type = CucumberParameterType.create_enum_match(
        name="status",
        enum_class=Status,
        case_sensitive=False
    )
    registry3.define_parameter_type(status_type)
    
    priority_type = CucumberParameterType.create_enum_match(
        name="priority",
        enum_class=Priority,
        case_sensitive=False
    )
    registry3.define_parameter_type(priority_type)
    
    expr3 = budExpression("Task is {status} with {priority} priority", registry3)
    test3 = "Task is APPROVED with High priority"
    match3 = expr3.match(test3)
    if match3:
        print(f"  ✓ Matched: status={match3[0].value}, priority={match3[1].value}")
        print(f"  ✓ Enum values: status={match3[0].value}, priority={match3[1].value}")
    print()
    
    # Method 4: Custom transformer
    print("4. With custom transformer:")
    
    def price_transformer(value: str) -> float:
        """Convert price string to float"""
        # Remove currency symbols and convert
        clean_value = value.replace('$', '').replace(',', '')
        return float(clean_value)
    
    price_type = CucumberParameterType.create_exact_match(
        name="price",
        values=["$10", "$20", "$50", "$100", "$1,000"],
        case_sensitive=True,
        transformer=price_transformer
    )
    registry.define_parameter_type(price_type)
    
    expr4 = budExpression("The item costs {price}", registry)
    test4 = "The item costs $1,000"
    match4 = expr4.match(test4)
    if match4:
        print(f"  ✓ Matched: price={match4[0].value} (type: {type(match4[0].value)})")
    
    print()
    print("=" * 60)
    print("Benefits:")
    print("✓ True Cucumber-style exact matching")
    print("✓ Easy case-insensitive support")
    print("✓ Enum integration")
    print("✓ Custom transformers")
    print("✓ NO semantic matching confusion")