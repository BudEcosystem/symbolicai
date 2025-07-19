# semantic_cucumber_expressions/dynamic_semantic_parameter_type.py
from typing import Any, List
from .semantic_parameter_type import SemanticParameterType
from .model_manager import Model2VecManager


class DynamicSemanticParameterType(SemanticParameterType):
    """
    A parameter type that dynamically matches based on semantic similarity
    between the parameter name and the matched value.
    
    For example, {cars} will match "Ferrari" by comparing the semantic
    similarity between "cars" and "Ferrari".
    """
    
    def __init__(self, name: str, similarity_threshold: float = 0.6):
        """
        Create a dynamic semantic parameter type.
        
        Args:
            name: The name of the parameter (e.g., 'cars', 'vehicle')
            similarity_threshold: Minimum similarity score (0-1)
        """
        # Use the parameter name itself as the prototype
        super().__init__(
            name=name,
            prototypes=[name],  # The name itself is the semantic reference
            similarity_threshold=similarity_threshold,
            use_for_snippets=True,
            prefer_for_regexp_match=False
        )
        self.is_dynamic = True
        
    def matches_semantically(self, text: str) -> tuple[bool, float, str]:
        """
        Check if text matches semantically with the parameter name.
        
        Returns:
            Tuple of (matches, similarity_score, parameter_name)
        """
        # Get embedding for input text
        text_embedding = self.model_manager.embed_sync([text])[0]
        
        # Get embedding for parameter name (already computed as prototype)
        self._ensure_embeddings()
        name_embedding = self._prototype_embeddings[0]
        
        # Calculate similarity
        similarity = self.model_manager.cosine_similarity(
            text_embedding, 
            name_embedding
        )
        
        matches = similarity >= self.similarity_threshold
        return matches, similarity, self.name
        
    def transform(self, group_values: List[str]) -> Any:
        """Transform with dynamic semantic validation"""
        if not group_values:
            return None
            
        value = group_values[0]
        
        # Check semantic match
        matches, similarity, _ = self.matches_semantically(value)
        
        if not matches:
            raise ValueError(
                f"'{value}' does not semantically match parameter type '{self.name}' "
                f"(similarity: {similarity:.2f}, threshold: {self.similarity_threshold})"
            )
        
        # Return the value if it matches
        return value