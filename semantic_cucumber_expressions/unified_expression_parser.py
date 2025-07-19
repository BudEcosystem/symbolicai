# semantic_cucumber_expressions/unified_expression_parser.py
from __future__ import annotations
from typing import NamedTuple, Optional, List, Dict, Any
import re

from cucumber_expressions.ast import Token, TokenType, Node, NodeType
from cucumber_expressions.errors import InvalidParameterTypeNameInNode
from .expression_parser import CucumberExpressionParser, Parser, Result
from .unified_parameter_type import ParameterTypeHint


class UnifiedExpressionParser(CucumberExpressionParser):
    """
    Enhanced expression parser that supports type hints in parameter syntax.
    
    Supports syntax like:
    - {param}            # Standard parameter
    - {param:semantic}   # Semantic parameter
    - {param:phrase}     # Multi-word phrase parameter
    - {param:dynamic}    # Dynamic semantic parameter
    - {param:regex}      # Custom regex parameter
    - {param:math}       # Mathematical expression
    - {param:quoted}     # Quoted string parameter
    """
    
    TYPE_HINT_PATTERN = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*):([a-zA-Z_][a-zA-Z0-9_]*)$')
    
    def __init__(self):
        super().__init__()
        self.parameter_metadata = {}  # Store parameter metadata during parsing
    
    def parse_name_with_type_hint(self, parser: Parser) -> Result:
        """
        Enhanced name parser that supports type hints.
        
        Parses parameter names like:
        - param
        - param:semantic
        - param:phrase
        - etc.
        """
        token = parser.tokens[parser.current]
        
        if token.ast_type in [TokenType.WHITE_SPACE, TokenType.TEXT]:
            # Check if this is a type hint syntax
            text = token.text
            type_hint_match = self.TYPE_HINT_PATTERN.match(text)
            
            if type_hint_match:
                param_name = type_hint_match.group(1)
                type_hint_str = type_hint_match.group(2)
                
                # Validate type hint
                try:
                    type_hint = ParameterTypeHint(type_hint_str)
                except ValueError:
                    # Invalid type hint, treat as regular parameter name
                    type_hint = ParameterTypeHint.STANDARD
                    param_name = text
                
                # Store metadata for later use
                self.parameter_metadata[param_name] = {
                    'type_hint': type_hint,
                    'original_text': text,
                    'position': (token.start, token.end)
                }
                
                # Return node with just the parameter name (type hint is stored separately)
                return Result(
                    1, Node(NodeType.TEXT, None, param_name, token.start, token.end)
                )
            else:
                # Regular parameter name
                self.parameter_metadata[text] = {
                    'type_hint': ParameterTypeHint.STANDARD,
                    'original_text': text,
                    'position': (token.start, token.end)
                }
                
                return Result(
                    1, Node(NodeType.TEXT, None, text, token.start, token.end)
                )
        
        # Handle invalid characters
        if token.ast_type in [
            TokenType.BEGIN_PARAMETER,
            TokenType.END_PARAMETER,
            TokenType.BEGIN_OPTIONAL,
            TokenType.END_OPTIONAL,
            TokenType.ALTERNATION,
        ]:
            raise InvalidParameterTypeNameInNode(parser.expression, token)
        
        if token.ast_type in [TokenType.START_OF_LINE, TokenType.END_OF_LINE]:
            return Result(0, None)
        
        return Result(0, None)
    
    def parse(self, expression: str) -> Node:
        """
        Parse expression with type hint support.
        
        Returns an AST node with parameter metadata attached.
        """
        # Clear previous metadata
        self.parameter_metadata = {}
        
        # Create enhanced parameter parser
        def parse_parameter_with_hints(parser: Parser) -> Result:
            # parameter := '{' + name_with_type_hint* + '}'
            return self.parse_between(
                NodeType.PARAMETER,
                TokenType.BEGIN_PARAMETER,
                TokenType.END_PARAMETER,
                [self.parse_name_with_type_hint],
            )(parser)
        
        # Use enhanced parameter parser
        optional_sub_parsers = []
        parse_optional = self.parse_between(
            NodeType.OPTIONAL,
            TokenType.BEGIN_OPTIONAL,
            TokenType.END_OPTIONAL,
            optional_sub_parsers,
        )
        
        optional_sub_parsers.extend([parse_optional, parse_parameter_with_hints, self.parse_text])
        
        # alternation := alternative* + ( '/' + alternative* )+
        def parse_alternative_separator(parser: Parser) -> Result:
            tokens = parser.tokens
            if not self.looking_at(tokens, parser.current, TokenType.ALTERNATION):
                return Result(0, None)
            token = tokens[parser.current]
            return Result(
                1, Node(NodeType.ALTERNATIVE, None, token.text, token.start, token.end)
            )
        
        alternative_parsers = [
            parse_alternative_separator,
            parse_optional,
            parse_parameter_with_hints,
            self.parse_text,
        ]
        
        # Parse the expression
        result = self.parse_with_parsers(expression, alternative_parsers)
        
        # Attach metadata to the result
        if hasattr(result, 'parameter_metadata'):
            result.parameter_metadata.update(self.parameter_metadata)
        else:
            result.parameter_metadata = self.parameter_metadata
        
        return result
    
    def get_parameter_metadata(self, param_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific parameter"""
        return self.parameter_metadata.get(param_name)
    
    def get_all_parameter_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all parameters"""
        return self.parameter_metadata.copy()
    
    def parse_with_parsers(self, expression: str, parsers: List) -> Node:
        """
        Parse expression using the given parsers.
        This is a helper method that handles the main parsing logic.
        """
        from cucumber_expressions.expression_tokenizer import CucumberExpressionTokenizer
        
        tokenizer = CucumberExpressionTokenizer()
        tokens = tokenizer.tokenize(expression)
        
        parser = Parser(expression, tokens, 0)
        
        # alternation := (?<=left-boundary) + alternative* + ( '/' + alternative* )+ + (?=right-boundary)
        def parse_alternation(parser: Parser) -> Result:
            previous = parser.current - 1
            
            # Check left boundary
            if previous >= 0:
                previous_token = parser.tokens[previous]
                if previous_token.ast_type not in [
                    TokenType.WHITE_SPACE,
                    TokenType.END_PARAMETER,
                    TokenType.END_OPTIONAL,
                    TokenType.START_OF_LINE,
                ]:
                    return Result(0, None)
            
            result = self.parse_token_sequence(parser, parsers)
            
            if result.ast_node is None:
                return Result(0, None)
            
            # Check if we have alternation
            if not self.looking_at(parser.tokens, parser.current, TokenType.ALTERNATION):
                return Result(0, None)
            
            # Parse alternatives
            alternatives = [result.ast_node]
            current_consumed = result.consumed
            
            while self.looking_at(parser.tokens, parser.current, TokenType.ALTERNATION):
                # Skip alternation token
                current_consumed += 1
                parser = Parser(parser.expression, parser.tokens, parser.current + 1)
                
                # Parse next alternative
                alt_result = self.parse_token_sequence(parser, parsers)
                if alt_result.ast_node is None:
                    break
                
                alternatives.append(alt_result.ast_node)
                current_consumed += alt_result.consumed
                parser = Parser(parser.expression, parser.tokens, parser.current + alt_result.consumed)
            
            # Check right boundary
            if parser.current < len(parser.tokens):
                next_token = parser.tokens[parser.current]
                if next_token.ast_type not in [
                    TokenType.WHITE_SPACE,
                    TokenType.BEGIN_PARAMETER,
                    TokenType.BEGIN_OPTIONAL,
                    TokenType.END_OF_LINE,
                ]:
                    return Result(0, None)
            
            # Create alternation node
            alternation_node = Node(NodeType.ALTERNATION, alternatives, None, 0, 0)
            return Result(current_consumed, alternation_node)
        
        # Main expression parser
        def parse_expression(parser: Parser) -> Result:
            results = []
            current_parser = parser
            
            while current_parser.current < len(current_parser.tokens):
                # Try alternation first
                alt_result = parse_alternation(current_parser)
                if alt_result.ast_node is not None:
                    results.append(alt_result.ast_node)
                    current_parser = Parser(
                        current_parser.expression,
                        current_parser.tokens,
                        current_parser.current + alt_result.consumed
                    )
                    continue
                
                # Try other parsers
                found_match = False
                for parse_func in parsers:
                    result = parse_func(current_parser)
                    if result.ast_node is not None:
                        results.append(result.ast_node)
                        current_parser = Parser(
                            current_parser.expression,
                            current_parser.tokens,
                            current_parser.current + result.consumed
                        )
                        found_match = True
                        break
                
                if not found_match:
                    # Skip token if no parser matched
                    current_parser = Parser(
                        current_parser.expression,
                        current_parser.tokens,
                        current_parser.current + 1
                    )
            
            if not results:
                return Result(0, None)
            
            # Create expression node
            expression_node = Node(NodeType.EXPRESSION, results, None, 0, len(expression))
            return Result(len(parser.tokens), expression_node)
        
        result = parse_expression(parser)
        return result.ast_node or Node(NodeType.EXPRESSION, [], None, 0, len(expression))
    
    def parse_token_sequence(self, parser: Parser, parsers: List) -> Result:
        """Parse a sequence of tokens using the given parsers"""
        results = []
        current_parser = parser
        total_consumed = 0
        
        while current_parser.current < len(current_parser.tokens):
            found_match = False
            
            for parse_func in parsers:
                result = parse_func(current_parser)
                if result.ast_node is not None:
                    results.append(result.ast_node)
                    current_parser = Parser(
                        current_parser.expression,
                        current_parser.tokens,
                        current_parser.current + result.consumed
                    )
                    total_consumed += result.consumed
                    found_match = True
                    break
            
            if not found_match:
                break
        
        if not results:
            return Result(0, None)
        
        # Create a sequence node
        sequence_node = Node(NodeType.EXPRESSION, results, None, 0, 0)
        return Result(total_consumed, sequence_node)