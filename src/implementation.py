"""
Deep Learning TA Interview - Implementation Exercise

Student Name: [Your Name Here]
Topic Chosen: [Specify Your Choice]
Date Started: [Date]

This file contains the main implementation for your chosen topic.
Remove this template code and implement your chosen algorithm.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# YOUR IMPLEMENTATION GOES HERE
# Choose ONE topic and implement it fully
# Delete this example code and write your own
# ============================================================================

class YourImplementation:
    """
    Replace this with your actual implementation.
    
    This is just a template to show expected structure.
    Your implementation should be complete and well-documented.
    """
    
    def __init__(self, param1: int, param2: float = 0.0):
        """
        Initialize your implementation.
        
        Args:
            param1: Description of parameter 1
            param2: Description of parameter 2
        """
        self.param1 = param1
        self.param2 = param2
        
        # Initialize any other components needed
        self._setup()
        
    def _setup(self):
        """Private method for additional setup if needed."""
        pass
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of your implementation.
        
        Args:
            x: Input tensor with shape [...]
            
        Returns:
            output: Output tensor with shape [...]
            
        Raises:
            ValueError: If input shape is invalid
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected input with at least 2 dimensions, got {x.dim()}")
        
        # Your implementation here
        output = x  # Replace with actual computation
        
        return output
    
    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Optional: Implement backward pass if relevant to your topic.
        
        Args:
            grad_output: Gradient with respect to output
            
        Returns:
            grad_input: Gradient with respect to input
        """
        raise NotImplementedError("Backward pass not implemented")


# ============================================================================
# HELPER FUNCTIONS (if needed)
# ============================================================================

def visualize_results(data: np.ndarray, title: str = "Results"):
    """
    Example helper function for visualization.
    
    Replace with your own helper functions as needed.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# DEMO / EXAMPLE USAGE
# ============================================================================

def main():
    """
    Demonstrate your implementation with a simple example.
    
    This should show:
    1. How to use your implementation
    2. That it works correctly
    3. Any key features or capabilities
    """
    print("Deep Learning TA Interview - Implementation Demo")
    print("=" * 50)
    
    # Create instance of your implementation
    model = YourImplementation(param1=10, param2=0.5)
    
    # Create sample input
    x = torch.randn(32, 10)  # Batch size 32, feature size 10
    
    # Run forward pass
    output = model.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Add more demonstration as appropriate for your topic
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
