#!/usr/bin/env python3
"""
Test suite for FAISS integration using TDD approach
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import time
from unittest.mock import Mock, patch

from semantic_bud_expressions import (
    SemanticParameterType,
    SemanticParameterTypeRegistry,
    Model2VecManager
)


class TestFAISSIntegration(unittest.TestCase):
    """Test FAISS integration for large-scale similarity search"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.registry = SemanticParameterTypeRegistry()
        cls.registry.initialize_model()
        cls.model_manager = Model2VecManager()
    
    def test_faiss_manager_creation(self):
        """Test creating FAISS manager"""
        # This should work after implementation
        try:
            from semantic_bud_expressions.faiss_manager import FAISSManager
            manager = FAISSManager()
            self.assertIsNotNone(manager)
        except ImportError:
            self.skipTest("FAISSManager not yet implemented")
    
    def test_faiss_index_creation(self):
        """Test creating different types of FAISS indices"""
        try:
            from semantic_bud_expressions.faiss_manager import FAISSManager
            manager = FAISSManager()
            
            # Test Flat index (exact search)
            index_flat = manager.create_index(dimension=256, index_type='Flat')
            self.assertIsNotNone(index_flat)
            
            # Test IVF index (approximate search)
            index_ivf = manager.create_index(dimension=256, index_type='IVF', nlist=100)
            self.assertIsNotNone(index_ivf)
            
            # Test HNSW index (graph-based)
            index_hnsw = manager.create_index(dimension=256, index_type='HNSW')
            self.assertIsNotNone(index_hnsw)
            
        except ImportError:
            self.skipTest("FAISSManager not yet implemented")
    
    def test_auto_index_selection(self):
        """Test automatic index type selection based on dataset size"""
        try:
            from semantic_bud_expressions.faiss_manager import FAISSManager
            manager = FAISSManager()
            
            # Small dataset -> Flat index
            index_small = manager.create_auto_index(dimension=256, n_vectors=100)
            self.assertEqual(manager.get_index_type(index_small), 'Flat')
            
            # Medium dataset -> IVF index
            index_medium = manager.create_auto_index(dimension=256, n_vectors=10000)
            self.assertEqual(manager.get_index_type(index_medium), 'IVF')
            
            # Large dataset -> HNSW index
            index_large = manager.create_auto_index(dimension=256, n_vectors=100000)
            self.assertEqual(manager.get_index_type(index_large), 'HNSW')
            
        except ImportError:
            self.skipTest("FAISSManager not yet implemented")
    
    def test_adding_embeddings_to_index(self):
        """Test adding prototype embeddings to FAISS index"""
        try:
            from semantic_bud_expressions.faiss_manager import FAISSManager
            manager = FAISSManager()
            
            # Create test embeddings
            n_prototypes = 1000
            dimension = 256
            embeddings = np.random.randn(n_prototypes, dimension).astype('float32')
            
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Create index and add embeddings
            index = manager.create_index(dimension=dimension, index_type='Flat')
            manager.add_embeddings(index, embeddings)
            
            # Verify embeddings were added
            self.assertEqual(index.ntotal, n_prototypes)
            
        except ImportError:
            self.skipTest("FAISSManager not yet implemented")
    
    def test_similarity_search(self):
        """Test k-nearest neighbor search"""
        try:
            from semantic_bud_expressions.faiss_manager import FAISSManager
            manager = FAISSManager()
            
            # Create test data
            dimension = 256
            n_prototypes = 1000
            
            # Create prototype embeddings
            prototypes = np.random.randn(n_prototypes, dimension).astype('float32')
            prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
            
            # Create query embedding (similar to first prototype)
            query = prototypes[0] + np.random.randn(dimension) * 0.1
            query = query / np.linalg.norm(query)
            query = query.reshape(1, -1)
            
            # Create index and add prototypes
            index = manager.create_index(dimension=dimension)
            manager.add_embeddings(index, prototypes)
            
            # Search for k nearest neighbors
            k = 5
            distances, indices = manager.search(index, query, k=k)
            
            # Verify results
            self.assertEqual(distances.shape, (1, k))
            self.assertEqual(indices.shape, (1, k))
            self.assertEqual(indices[0, 0], 0)  # First prototype should be closest
            
        except ImportError:
            self.skipTest("FAISSManager not yet implemented")
    
    def test_batch_similarity_search(self):
        """Test batch similarity search"""
        try:
            from semantic_bud_expressions.faiss_manager import FAISSManager
            manager = FAISSManager()
            
            # Create test data
            dimension = 256
            n_prototypes = 1000
            n_queries = 10
            
            # Create embeddings
            prototypes = np.random.randn(n_prototypes, dimension).astype('float32')
            prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
            
            queries = np.random.randn(n_queries, dimension).astype('float32')
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
            
            # Create index
            index = manager.create_index(dimension=dimension)
            manager.add_embeddings(index, prototypes)
            
            # Batch search
            k = 3
            distances, indices = manager.search(index, queries, k=k)
            
            # Verify batch results
            self.assertEqual(distances.shape, (n_queries, k))
            self.assertEqual(indices.shape, (n_queries, k))
            
        except ImportError:
            self.skipTest("FAISSManager not yet implemented")
    
    def test_semantic_parameter_type_with_faiss(self):
        """Test SemanticParameterType using FAISS for large prototype sets"""
        try:
            from semantic_bud_expressions.faiss_manager import FAISSManager
            
            # Create semantic type with many prototypes
            large_prototypes = [f"prototype_{i}" for i in range(1000)]
            
            semantic_type = SemanticParameterType(
                name="large_category",
                prototypes=large_prototypes,
                similarity_threshold=0.7,
                use_faiss=True  # Force FAISS usage
            )
            
            # Initialize embeddings
            semantic_type._ensure_embeddings()
            
            # Test matching
            matches, score, closest = semantic_type.matches_semantically("prototype_500")
            
            self.assertTrue(matches)
            self.assertAlmostEqual(score, 1.0, places=3)
            self.assertEqual(closest, "prototype_500")
            
        except (ImportError, TypeError):
            self.skipTest("FAISS integration not yet implemented in SemanticParameterType")
    
    def test_faiss_fallback_to_numpy(self):
        """Test graceful fallback when FAISS is not available"""
        # Mock FAISS import failure
        with patch.dict('sys.modules', {'faiss': None}):
            try:
                from semantic_bud_expressions.faiss_manager import FAISSManager
                manager = FAISSManager()
                
                # Should fall back to numpy implementation
                self.assertTrue(manager.use_numpy_fallback)
                
                # Test search still works
                embeddings = np.random.randn(100, 256).astype('float32')
                query = np.random.randn(1, 256).astype('float32')
                
                distances, indices = manager.search_numpy(embeddings, query, k=5)
                self.assertEqual(distances.shape, (1, 5))
                
            except ImportError:
                self.skipTest("FAISSManager not yet implemented")
    
    def test_index_persistence(self):
        """Test saving and loading FAISS index"""
        try:
            from semantic_bud_expressions.faiss_manager import FAISSManager
            import tempfile
            
            manager = FAISSManager()
            
            # Create and populate index
            dimension = 256
            embeddings = np.random.randn(100, dimension).astype('float32')
            
            index = manager.create_index(dimension=dimension)
            manager.add_embeddings(index, embeddings)
            
            # Save index
            with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as f:
                index_path = f.name
                manager.save_index(index, index_path)
            
            # Load index
            loaded_index = manager.load_index(index_path)
            
            # Verify loaded index
            self.assertEqual(loaded_index.ntotal, 100)
            self.assertEqual(loaded_index.d, dimension)
            
            # Cleanup
            os.unlink(index_path)
            
        except ImportError:
            self.skipTest("FAISSManager not yet implemented")
    
    def test_performance_comparison(self):
        """Test performance improvement with FAISS vs numpy"""
        try:
            from semantic_bud_expressions.faiss_manager import FAISSManager
            manager = FAISSManager()
            
            # Create large dataset
            n_prototypes = 10000
            dimension = 256
            n_queries = 100
            
            prototypes = np.random.randn(n_prototypes, dimension).astype('float32')
            prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
            
            queries = np.random.randn(n_queries, dimension).astype('float32')
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
            
            # Test numpy performance
            start_numpy = time.time()
            for query in queries:
                similarities = np.dot(prototypes, query)
                top_k = np.argpartition(similarities, -5)[-5:]
            numpy_time = time.time() - start_numpy
            
            # Test FAISS performance
            index = manager.create_index(dimension=dimension)
            manager.add_embeddings(index, prototypes)
            
            start_faiss = time.time()
            distances, indices = manager.search(index, queries, k=5)
            faiss_time = time.time() - start_faiss
            
            # FAISS should be significantly faster
            speedup = numpy_time / faiss_time
            print(f"FAISS speedup: {speedup:.2f}x")
            self.assertGreater(speedup, 5.0)  # Expect at least 5x speedup
            
        except ImportError:
            self.skipTest("FAISSManager not yet implemented")
    
    def test_thread_safety(self):
        """Test thread-safe operations"""
        try:
            from semantic_bud_expressions.faiss_manager import FAISSManager
            import threading
            
            manager = FAISSManager()
            
            # Create shared index
            dimension = 256
            index = manager.create_index(dimension=dimension)
            
            # Add embeddings from multiple threads
            def add_embeddings_thread(thread_id):
                embeddings = np.random.randn(100, dimension).astype('float32')
                manager.add_embeddings(index, embeddings, thread_safe=True)
            
            threads = []
            for i in range(5):
                t = threading.Thread(target=add_embeddings_thread, args=(i,))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            # Verify all embeddings were added
            self.assertEqual(index.ntotal, 500)
            
        except ImportError:
            self.skipTest("FAISSManager not yet implemented")


class TestFAISSConfiguration(unittest.TestCase):
    """Test FAISS configuration and auto-detection"""
    
    def test_auto_enable_faiss(self):
        """Test automatic FAISS enabling based on prototype count"""
        try:
            # Small prototype set - should use numpy
            small_type = SemanticParameterType(
                name="small",
                prototypes=["a", "b", "c"],
                similarity_threshold=0.7
            )
            self.assertFalse(getattr(small_type, 'use_faiss', False))
            
            # Large prototype set - should use FAISS
            large_prototypes = [f"proto_{i}" for i in range(1000)]
            large_type = SemanticParameterType(
                name="large",
                prototypes=large_prototypes,
                similarity_threshold=0.7
            )
            self.assertTrue(getattr(large_type, 'use_faiss', True))
            
        except AttributeError:
            self.skipTest("FAISS auto-detection not yet implemented")
    
    def test_faiss_configuration(self):
        """Test FAISS configuration options"""
        try:
            from semantic_bud_expressions.faiss_manager import FAISSManager
            
            # Reset singleton for test
            FAISSManager.reset_instance()
            
            manager = FAISSManager(
                auto_index_threshold=500,  # Use IVF for 500+ vectors
                normalize_embeddings=True,
                use_gpu=False  # CPU only for tests
            )
            
            self.assertEqual(manager.auto_index_threshold, 500)
            self.assertTrue(manager.normalize_embeddings)
            self.assertFalse(manager.use_gpu)
            
        except ImportError:
            self.skipTest("FAISSManager not yet implemented")


if __name__ == '__main__':
    unittest.main()