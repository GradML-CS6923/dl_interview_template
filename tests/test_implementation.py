"""
Unit tests for TA interview implementation.

Run tests:
    pytest tests/ -v
    
Check coverage:
    pytest --cov=src tests/
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.implementation import *


class TestImplementation:
    """Test suite for your implementation."""
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input for testing."""
        return torch.randn(16, 32)  # Adjust based on your implementation
    
    def test_initialization(self):
        """Test that the implementation initializes correctly."""
        # TODO: Replace with actual test for your implementation
        model = YourImplementation(param1=10, param2=0.5)
        assert model.param1 == 10
        assert model.param2 == 0.5
    
    def test_forward_pass_shape(self, sample_input):
        """Test that forward pass produces correct output shape."""
        # TODO: Replace with actual test for your implementation
        model = YourImplementation(param1=10)
        output = model.forward(sample_input)
        
        # Check output shape matches expected
        assert output.shape == sample_input.shape  # Adjust based on your implementation
    
    def test_forward_pass_values(self):
        """Test that forward pass produces correct values."""
        # TODO: Add specific test cases for your implementation
        # Example: known input -> expected output
        pass
    
    def test_edge_case_empty_input(self):
        """Test handling of empty input."""
        # TODO: Test how your implementation handles edge cases
        model = YourImplementation(param1=10)
        
        # Test with empty tensor
        empty_input = torch.tensor([])
        with pytest.raises(ValueError):
            model.forward(empty_input)
    
    def test_edge_case_single_element(self):
        """Test handling of single element input."""
        # TODO: Adjust for your implementation
        pass
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # TODO: Test with very large/small values
        pass
    
    @pytest.mark.parametrize("batch_size,feature_size", [
        (1, 10),
        (32, 64),
        (100, 256),
    ])
    def test_different_sizes(self, batch_size, feature_size):
        """Test with different input sizes."""
        # TODO: Parametrized test for different configurations
        pass
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly (if applicable)."""
        # TODO: For implementations involving gradients
        pass
    
    def test_comparison_with_reference(self):
        """Compare with reference implementation (e.g., PyTorch)."""
        # TODO: If applicable, compare with library implementation
        pass


class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_invalid_input_type(self):
        """Test handling of invalid input types."""
        model = YourImplementation(param1=10)
        
        with pytest.raises(Exception):  # Adjust exception type
            model.forward("not a tensor")
    
    def test_device_compatibility(self):
        """Test CPU/GPU compatibility if relevant."""
        # TODO: Test device handling
        pass


class TestPerformance:
    """Performance and efficiency tests."""
    
    @pytest.mark.slow
    def test_large_scale_performance(self):
        """Test performance with large inputs."""
        # TODO: Add performance benchmarks
        pass
    
    def test_memory_efficiency(self):
        """Test memory usage is reasonable."""
        # TODO: Monitor memory usage
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
