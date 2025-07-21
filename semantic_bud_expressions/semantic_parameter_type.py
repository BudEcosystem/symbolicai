# semantic_bud_expressions/semantic_parameter_type.py
from typing import List, Optional, Callable, Dict, Any
import re
from .parameter_type import ParameterType
from .model_manager import Model2VecManager
from .faiss_manager import FAISSManager
import numpy as np

class SemanticParameterType(ParameterType):
    """Parameter type that uses semantic similarity for matching"""
    
    def __init__(
        self,
        name: str,
        prototypes: List[str],
        type: type = str,
        transformer: Optional[Callable] = None,
        similarity_threshold: float = 0.7,
        use_for_snippets: bool = True,
        prefer_for_regexp_match: bool = False,
        fallback_regexp: Optional[str] = None,
        context_hints: Optional[Dict[str, Any]] = None,
        use_faiss: Optional[bool] = None
    ):
        """
        Create a semantic parameter type
        
        Args:
            name: Name of the parameter type (e.g., 'fruit', 'greeting')
            prototypes: List of prototype examples (e.g., ['apple', 'banana', 'orange'])
            type: The type to transform to
            transformer: Custom transformation function
            similarity_threshold: Minimum similarity score (0-1)
            use_for_snippets: Whether to use for snippet generation
            prefer_for_regexp_match: Whether to prefer for regexp matching
            fallback_regexp: Optional regex pattern as fallback
            context_hints: Additional context for semantic matching
            use_faiss: Whether to use FAISS for similarity search (None=auto-detect)
        """
        # Use a broad regex that captures words
        base_regexp = fallback_regexp or r'\w+'
        
        super().__init__(
            name=name,
            regexp=base_regexp,
            type=type,
            transformer=transformer,
            use_for_snippets=use_for_snippets,
            prefer_for_regexp_match=prefer_for_regexp_match
        )
        
        self.prototypes = prototypes
        self.similarity_threshold = similarity_threshold
        self.context_hints = context_hints or {}
        self.model_manager = Model2VecManager()
        self._prototype_embeddings = None
        self._precompute_embeddings = True  # Enable pre-computation by default
        
        # FAISS configuration
        if use_faiss is None:
            # Auto-detect based on prototype count
            self.use_faiss = len(prototypes) >= 100
        else:
            self.use_faiss = use_faiss
            
        self.faiss_manager = FAISSManager() if self.use_faiss else None
        self._faiss_index = None
        
    def _ensure_embeddings(self):
        """Ensure prototype embeddings are computed"""
        if self._prototype_embeddings is None:
            self._prototype_embeddings = self.model_manager.embed_sync(self.prototypes)
            
            # Create FAISS index if using FAISS
            if self.use_faiss and self.faiss_manager:
                self._create_faiss_index()
    
    def _create_faiss_index(self):
        """Create FAISS index for prototype embeddings"""
        if not self.faiss_manager or self._prototype_embeddings is None:
            return
            
        # Convert embeddings to numpy array
        embeddings_array = np.array(self._prototype_embeddings, dtype=np.float32)
        
        # Create appropriate index
        dimension = embeddings_array.shape[1]
        self._faiss_index = self.faiss_manager.create_auto_index(
            dimension=dimension,
            n_vectors=len(self.prototypes)
        )
        
        # Add embeddings to index
        if self._faiss_index is not None:
            self.faiss_manager.add_embeddings(self._faiss_index, embeddings_array)
    
    def matches_semantically(self, text: str) -> tuple[bool, float, str]:
        """
        Check if text matches semantically
        
        Returns:
            Tuple of (matches, similarity_score, closest_prototype)
        """
        self._ensure_embeddings()
        
        # Get embedding for input text
        text_embedding = self.model_manager.embed_sync([text])[0]
        
        if self.use_faiss and self._faiss_index is not None:
            # Use FAISS for similarity search
            return self._matches_semantically_faiss(text_embedding)
        else:
            # Use numpy-based search
            return self._matches_semantically_numpy(text_embedding)
    
    def _matches_semantically_faiss(self, text_embedding: np.ndarray) -> tuple[bool, float, str]:
        """Use FAISS for semantic matching"""
        # Reshape for FAISS
        query = text_embedding.reshape(1, -1).astype(np.float32)
        
        # Search for nearest prototype
        distances, indices = self.faiss_manager.search(
            self._faiss_index,
            query,
            k=1
        )
        
        # Get results
        similarity = float(distances[0, 0])
        closest_idx = int(indices[0, 0])
        closest_prototype = self.prototypes[closest_idx]
        
        matches = similarity >= self.similarity_threshold
        return matches, similarity, closest_prototype
    
    def _matches_semantically_numpy(self, text_embedding: np.ndarray) -> tuple[bool, float, str]:
        """Use numpy for semantic matching (original implementation)"""
        max_similarity = 0.0
        closest_prototype = None
        
        for i, prototype in enumerate(self.prototypes):
            similarity = self.model_manager.cosine_similarity(
                text_embedding, 
                self._prototype_embeddings[i]
            )
            
            if similarity > max_similarity:
                max_similarity = similarity
                closest_prototype = prototype
        
        matches = max_similarity >= self.similarity_threshold
        return matches, max_similarity, closest_prototype
    
    def transform(self, group_values: List[str]) -> Any:
        """Transform with semantic validation"""
        if not group_values:
            return None
            
        value = group_values[0]
        
        # Check semantic match
        matches, similarity, closest = self.matches_semantically(value)
        
        if not matches:
            raise ValueError(
                f"'{value}' does not semantically match parameter type '{self.name}' "
                f"(similarity: {similarity:.2f}, threshold: {self.similarity_threshold})"
            )
        
        # Apply transformer - use parent's transform method which handles the default case
        return super().transform(group_values)
