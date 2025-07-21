# semantic_bud_expressions/semantic_registry.py
from typing import Dict, List, Optional
from .parameter_type_registry import ParameterTypeRegistry
from .parameter_type import ParameterType
from .semantic_parameter_type import SemanticParameterType
from .math_parameter_type import MathParameterType
from .dynamic_semantic_parameter_type import DynamicSemanticParameterType
from .model_manager import Model2VecManager

class SemanticParameterTypeRegistry(ParameterTypeRegistry):
    """Enhanced parameter type registry with semantic capabilities"""
    
    def __init__(self):
        super().__init__()
        self.model_manager = Model2VecManager()
        self._dynamic_matching_enabled = True
        self._dynamic_threshold = 0.3  # Lower default for category-to-instance matching
        self._dynamic_types_cache = {}  # Cache for created dynamic types
        self._define_semantic_types()
        
    def _define_semantic_types(self):
        """Define common semantic parameter types"""
        
        # Greeting parameter type
        self.define_parameter_type(SemanticParameterType(
            name="greeting",
            prototypes=["hello", "hi", "hey", "greetings", "good morning", 
                       "good afternoon", "good evening", "howdy", "salutations"],
            similarity_threshold=0.75
        ))
        
        # Fruit parameter type
        self.define_parameter_type(SemanticParameterType(
            name="fruit",
            prototypes=["apple", "banana", "orange", "grape", "strawberry",
                       "mango", "pineapple", "watermelon", "peach", "pear"],
            similarity_threshold=0.7
        ))
        
        # Emotion parameter type
        self.define_parameter_type(SemanticParameterType(
            name="emotion",
            prototypes=["happy", "sad", "angry", "excited", "frustrated",
                       "joyful", "depressed", "anxious", "calm", "worried"],
            similarity_threshold=0.75
        ))
        
        # Action parameter type
        self.define_parameter_type(SemanticParameterType(
            name="action",
            prototypes=["run", "walk", "jump", "sit", "stand", "crawl",
                       "sprint", "jog", "leap", "hop", "skip"],
            similarity_threshold=0.7
        ))
        
        # Math expression parameter type
        self.define_parameter_type(MathParameterType(
            name="math",
            simplify_expressions=True
        ))
        
    def define_semantic_category(
        self, 
        name: str, 
        prototypes: List[str],
        similarity_threshold: float = 0.7,
        **kwargs
    ) -> SemanticParameterType:
        """Convenience method to define semantic categories"""
        param_type = SemanticParameterType(
            name=name,
            prototypes=prototypes,
            similarity_threshold=similarity_threshold,
            **kwargs
        )
        self.define_parameter_type(param_type)
        return param_type
    
    def initialize_model(self, model_name: str = 'minishlab/potion-base-8M'):
        """Initialize the model2vec model"""
        self.model_manager.initialize_sync(model_name)
        # Pre-compute embeddings for all semantic types
        self._precompute_all_embeddings()
        
    def enable_dynamic_matching(self, enabled: bool = True):
        """Enable or disable dynamic semantic matching"""
        self._dynamic_matching_enabled = enabled
        
    def set_dynamic_threshold(self, threshold: float):
        """Set the similarity threshold for dynamic matching"""
        self._dynamic_threshold = threshold
        
    def lookup_by_type_name(self, name: str) -> Optional[ParameterType]:
        """
        Override to support dynamic parameter type creation.
        
        If a parameter type is not found and dynamic matching is enabled,
        create a dynamic semantic parameter type on the fly.
        """
        # First try the standard lookup
        param_type = super().lookup_by_type_name(name)
        
        if param_type is not None:
            return param_type
            
        # If not found and dynamic matching is enabled
        if self._dynamic_matching_enabled:
            # Check cache first
            if name in self._dynamic_types_cache:
                return self._dynamic_types_cache[name]
                
            # Create a new dynamic parameter type
            dynamic_type = DynamicSemanticParameterType(
                name=name.lower(),  # Normalize to lowercase
                similarity_threshold=self._dynamic_threshold
            )
            
            # Cache it for future use
            self._dynamic_types_cache[name] = dynamic_type
            
            # Also register it properly (but don't add to predefined types)
            # This ensures it works with the expression machinery
            self.define_parameter_type(dynamic_type)
            
            return dynamic_type
            
        return None
    
    def _precompute_all_embeddings(self):
        """Pre-compute embeddings for all semantic parameter types"""
        all_prototypes = []
        param_type_indices = []
        
        # Collect all prototypes from semantic types
        for param_type in self.parameter_types:
            if isinstance(param_type, SemanticParameterType) and hasattr(param_type, 'prototypes'):
                start_idx = len(all_prototypes)
                all_prototypes.extend(param_type.prototypes)
                end_idx = len(all_prototypes)
                param_type_indices.append((param_type, start_idx, end_idx))
        
        if all_prototypes:
            # Batch compute all embeddings at once
            all_embeddings = self.model_manager.embed_sync(all_prototypes)
            
            # Assign embeddings back to each parameter type
            for param_type, start_idx, end_idx in param_type_indices:
                param_type._prototype_embeddings = all_embeddings[start_idx:end_idx]
