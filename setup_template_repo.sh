#!/bin/bash

# Complete setup script for GradML-CS6923 TA Interview Template
# Run this in your cloned ta-interview-template directory

echo "Setting up GradML TA Interview Template Repository..."

# Create directory structure
echo "Creating directory structure..."
mkdir -p src tests teaching/diagrams .github/classroom .github/workflows

# Create empty __init__.py files
touch src/__init__.py tests/__init__.py

# Create main README
cat > README.md << 'EOF'
# TA Interview Exercise - Deep Learning (GradML CS6923)

## 📋 Overview

This is your take-home exercise for the Deep Learning TA position. You have **72 hours** to complete this assignment from when you accept it.

**Expected time:** 3-4 hours of focused work

## 🎯 Assignment Structure

### Part 1: Implementation (60% of time)
Choose **ONE** of the following topics to implement from scratch:

1. **Scaled Dot-Product Attention with Extensions**
   - Implement the core attention mechanism
   - Extend to multi-head attention
   - Support causal and padding masks
   - Compare with at least one variant (additive/multiplicative)

2. **Custom Autograd System**
   - Build automatic differentiation engine
   - Support: add, multiply, matmul, ReLU, mean
   - Implement backward passes for each
   - Train a 2-layer network on a toy problem

3. **Batch Normalization from Scratch**
   - Forward pass with running statistics
   - Complete backward pass implementation
   - Training vs evaluation mode
   - Demonstrate impact on deep network training

4. **Advanced Optimization Technique**
   - Choose ONE: Adam from scratch, gradient accumulation, or mixed precision training (simulated)
   - Implement the core algorithm
   - Compare convergence with SGD
   - Visualize the optimization trajectory

5. **Knowledge Distillation Framework**
   - Teacher-student architecture
   - Temperature-scaled softmax
   - Combined loss implementation
   - Demonstrate compression on MNIST/CIFAR-10

6. **Modern Regularization Method**
   - Choose ONE: MixUp, CutMix, or DropBlock
   - Implement the augmentation strategy
   - Integration into training loop
   - Show effect on generalization

### Part 2: Teaching Materials (30% of time)
Create educational content for your implementation:
- Jupyter notebook tutorial
- Visual explanations
- Common student questions
- Debugging tips

### Part 3: Code Quality (10% of time)
- Comprehensive tests
- Clear documentation
- Performance analysis

## 📁 Repository Structure

```
your-github-username-ta-interview/
├── src/
│   ├── __init__.py
│   ├── implementation.py    # Your main implementation
│   └── utils.py            # Helper functions (optional)
├── tests/
│   ├── __init__.py
│   └── test_implementation.py
├── teaching/
│   ├── tutorial.ipynb      # Teaching notebook
│   ├── diagrams/           # Your visualizations
│   └── student_faq.md      # Common questions (optional)
├── requirements.txt
├── README.md               # This file
└── SUBMISSION.md          # Your submission notes
```

## 🚀 Getting Started

1. **Set up your environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Choose your topic** and update `SUBMISSION.md` with your choice

3. **Start implementing** in `src/implementation.py`

4. **Commit your work regularly** - we want to see your development process

## 📝 Submission Requirements

### Code Requirements
- [ ] Clean, well-commented code
- [ ] Type hints where appropriate
- [ ] Docstrings for all functions/classes
- [ ] No copying from existing implementations

### Testing Requirements
- [ ] Unit tests with >80% coverage
- [ ] Edge case handling
- [ ] Performance benchmarks
- [ ] Comparison with library implementations (if applicable)

### Teaching Requirements
- [ ] Clear, runnable Jupyter notebook
- [ ] Visual aids (hand-drawn is fine!)
- [ ] 3+ anticipated student questions
- [ ] Common pitfalls section

### Git Requirements
- [ ] Regular commits showing progress
- [ ] Meaningful commit messages
- [ ] All work in your repository

## 🔄 Submission Process

1. **Complete your work** with regular commits

2. **Run final tests:**
   ```bash
   pytest tests/ -v
   python -m pytest --cov=src tests/
   ```

3. **Update `SUBMISSION.md`** with all required information

4. **Push your final version**

5. **Submit** by filling out the submission form (link will be provided)

## 📊 Evaluation Criteria

### Technical Implementation (40%)
- Correctness: 15%
- Code quality: 10%
- Testing: 10%
- Efficiency: 5%

### Teaching Materials (40%)
- Tutorial clarity: 15%
- Visual aids: 10%
- Student engagement: 10%
- Organization: 5%

### Professional Skills (20%)
- Documentation: 10%
- Git usage: 5%
- Communication: 5%

## ⏰ Important Notes

- **Start Time**: When you accept the GitHub Classroom assignment
- **Duration**: 72 hours from acceptance
- **Late Policy**: No late submissions accepted
- **Questions**: Email ta-interviews@gradml.com

## 🤝 Academic Integrity

- You may use official documentation and learning resources
- You may NOT copy implementations from existing code
- You may NOT use AI to generate your core implementation
- You MAY use AI for help with tests, documentation, or debugging
- All work must be your own

## 💡 Tips for Success

1. **Start simple** - get a basic version working first
2. **Commit often** - show your development process
3. **Test as you go** - don't leave testing until the end
4. **Focus on clarity** - we're evaluating teaching ability
5. **Ask questions** - reach out if you're stuck

## 📧 Contact

- Technical questions: ta-interviews@gradml.com
- Urgent issues: Post in the course Slack/Discord

Good luck! We're excited to see your work and learn about your teaching approach.

---
*GradML CS6923 - Teaching Assistant Selection*
EOF

# Create SUBMISSION.md template
cat > SUBMISSION.md << 'EOF'
# Submission Notes

**Name:** [Your Name]  
**Email:** [Your Email]  
**GitHub Username:** [Your GitHub Username]  
**Submission Date:** [Date]

## Implementation Choice

**Topic Selected:** [Attention / Autograd / BatchNorm / Optimization / Distillation / Regularization]

**Why I chose this topic:**
[2-3 sentences explaining your choice]

## Time Breakdown

- **Setup & Research:** [X hours]
- **Core Implementation:** [X hours]
- **Testing & Debugging:** [X hours]
- **Teaching Materials:** [X hours]
- **Documentation & Polish:** [X hours]
- **Total Time:** [X hours]

## Implementation Details

### Key Design Decisions

1. **[Decision 1]**
   - What: [Brief description]
   - Why: [Reasoning]
   - Trade-offs: [What you considered]

2. **[Decision 2]**
   - What: [Brief description]
   - Why: [Reasoning]
   - Trade-offs: [What you considered]

### Assumptions Made

1. [Assumption 1 and justification]
2. [Assumption 2 and justification]
3. [Any other assumptions]

### Challenges Faced

1. **Challenge:** [Description]
   **Solution:** [How you solved it]
   **Learning:** [What you learned]

2. **Challenge:** [Description]
   **Solution:** [How you solved it]
   **Learning:** [What you learned]

## Testing Approach

**Test Coverage:** [X%]

**Testing Strategy:**
- Unit tests for: [List key components tested]
- Edge cases tested: [List edge cases]
- Performance benchmarks: [Brief results]

## Teaching Materials

**Tutorial Structure:**
1. [Section 1 - what it covers]
2. [Section 2 - what it covers]
3. [Section 3 - what it covers]

**Key Visualizations:**
- [Visualization 1 - what it shows]
- [Visualization 2 - what it shows]

**Anticipated Student Questions:**
1. Q: [Question]
   A: [Your answer approach]

2. Q: [Question]
   A: [Your answer approach]

3. Q: [Question]
   A: [Your answer approach]

## AI Tool Usage Declaration

**I used AI tools for:**
- [Specific use case, e.g., "Debugging syntax errors"]
- [Specific use case, e.g., "Improving docstring clarity"]

**I did NOT use AI tools for:**
- Core algorithm implementation
- Architecture design decisions
- Teaching content creation
- Test case design

## Reflection

**What went well:**
[2-3 things that went smoothly]

**What was challenging:**
[2-3 things that were difficult]

**What I would do differently:**
[2-3 things you'd change with hindsight]

**Key learnings:**
[2-3 things you learned from this exercise]

## Future Improvements

If I had more time, I would:
1. [Improvement 1]
2. [Improvement 2]
3. [Improvement 3]

## Submission Checklist

- [ ] Core algorithm implemented and working
- [ ] All tests passing with good coverage
- [ ] Teaching notebook is clear and runnable
- [ ] Code is well-documented with docstrings
- [ ] Visualizations are included and helpful
- [ ] Git history shows iterative development
- [ ] SUBMISSION.md is complete
- [ ] All files are pushed to GitHub

---

**Declaration:** I confirm that this submission is my own work, completed in accordance with the academic integrity guidelines.

**Signature:** [Your Name]  
**Date:** [Date]
EOF

# Create implementation starter file
cat > src/implementation.py << 'EOF'
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
EOF

# Create test file
cat > tests/test_implementation.py << 'EOF'
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
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core dependencies
numpy>=1.21.0,<2.0.0
torch>=2.0.0,<3.0.0
matplotlib>=3.5.0,<4.0.0
jupyter>=1.0.0
notebook>=6.5.0

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0

# Code quality
black>=22.0.0
flake8>=4.0.0
mypy>=0.950

# Visualization
seaborn>=0.11.0
ipywidgets>=7.7.0

# Additional useful packages
tqdm>=4.64.0
pandas>=1.4.0
scikit-learn>=1.0.0
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
.venv

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/
.coverage.*
*.cover
.hypothesis/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.project
.pydevproject

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
*.log
logs/
data/
*.csv
*.pkl
*.pth
*.h5
*.hdf5
*.npy
*.npz

# Documentation
docs/_build/
docs/.doctrees/

# Temporary files
*.tmp
*.bak
.~*
EOF

# Create teaching directory files
mkdir -p teaching/diagrams

# Create notebook template
cat > teaching/tutorial.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Tutorial: [Your Implementation Topic]\n",
    "\n",
    "**Author:** [Your Name]  \n",
    "**Date:** [Date]  \n",
    "**Topic:** [Your chosen implementation topic]\n",
    "\n",
    "## 📚 Overview\n",
    "\n",
    "In this tutorial, we'll explore [topic] through hands-on implementation and visualization. This notebook is designed for students who have basic knowledge of deep learning but want to understand [topic] in depth.\n",
    "\n",
    "### Learning Objectives\n",
    "\n",
    "By the end of this tutorial, you will:\n",
    "1. Understand the mathematical foundations of [topic]\n",
    "2. Implement [topic] from scratch\n",
    "3. Debug common issues that arise\n",
    "4. Apply [topic] to real problems\n",
    "5. Know when and why to use [topic]\n",
    "\n",
    "### Prerequisites\n",
    "\n",
    "- Basic Python programming\n",
    "- Understanding of neural networks\n",
    "- Familiarity with PyTorch basics\n",
    "- Linear algebra fundamentals"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔧 Setup\n",
    "\n",
    "Let's start by importing the necessary libraries and our implementation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Standard imports\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from IPython.display import display, HTML\n",
    "\n",
    "# Configure visualization\n",
    "plt.style.use('seaborn-v0_8-darkgrid')\n",
    "sns.set_palette(\"husl\")\n",
    "\n",
    "# Import our implementation\n",
    "import sys\n",
    "sys.path.append('..')\n",
    "from src.implementation import *\n",
    "\n",
    "# Set random seeds for reproducibility\n",
    "torch.manual_seed(42)\n",
    "np.random.seed(42)\n",
    "\n",
    "print(\"Setup complete! ✅\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📖 Part 1: Understanding the Concept\n",
    "\n",
    "### What is [Topic]?\n",
    "\n",
    "[Provide intuitive explanation]\n",
    "\n",
    "### Why do we need it?\n",
    "\n",
    "[Explain the motivation and problems it solves]\n",
    "\n",
    "### Mathematical Foundation\n",
    "\n",
    "[Include key equations with explanations]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualization of the concept\n",
    "# TODO: Add your visualization code here"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔨 Part 2: Implementation Deep Dive\n",
    "\n",
    "Now let's look at how [topic] is implemented step by step."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step-by-step implementation demonstration\n",
    "# TODO: Break down your implementation into understandable chunks"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🧪 Part 3: Experiments and Visualization\n",
    "\n",
    "Let's experiment with our implementation to better understand its behavior."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Experiment 1: Basic functionality\n",
    "# TODO: Add experiments that demonstrate key properties"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ⚠️ Part 4: Common Pitfalls and Debugging\n",
    "\n",
    "### Pitfall 1: [Name]\n",
    "[Explain the issue and how to avoid it]\n",
    "\n",
    "### Pitfall 2: [Name]\n",
    "[Explain the issue and how to avoid it]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Debugging example\n",
    "# TODO: Show common error and how to fix it"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎯 Part 5: Practical Applications\n",
    "\n",
    "Let's apply [topic] to a real problem to see its effectiveness."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Real-world application\n",
    "# TODO: Demonstrate practical use case"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📊 Part 6: Performance Analysis\n",
    "\n",
    "How does our implementation compare to existing libraries?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Performance comparison\n",
    "# TODO: Benchmark your implementation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 💡 Part 7: Exercises for Students\n",
    "\n",
    "Try these exercises to deepen your understanding:\n",
    "\n",
    "### Exercise 1: [Title]\n",
    "[Description of exercise]\n",
    "\n",
    "### Exercise 2: [Title]\n",
    "[Description of exercise]\n",
    "\n",
    "### Exercise 3: [Title]\n",
    "[Description of exercise]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Space for students to try exercises\n",
    "# TODO: Provide starter code for exercises"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎓 Summary and Key Takeaways\n",
    "\n",
    "In this tutorial, we've covered:\n",
    "\n",
    "1. ✅ The theory behind [topic]\n",
    "2. ✅ Step-by-step implementation\n",
    "3. ✅ Common pitfalls and how to avoid them\n",
    "4. ✅ Practical applications\n",
    "5. ✅ Performance considerations\n",
    "\n",
    "### Key Insights\n",
    "\n",
    "- [Insight 1]\n",
    "- [Insight 2]\n",
    "- [Insight 3]\n",
    "\n",
    "### When to Use [Topic]\n",
    "\n",
    "- ✅ [Scenario 1]\n",
    "- ✅ [Scenario 2]\n",
    "- ❌ [When NOT to use it]\n",
    "\n",
    "### Further Reading\n",
    "\n",
    "- [Paper/Resource 1]\n",
    "- [Paper/Resource 2]\n",
    "- [Paper/Resource 3]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🤔 Frequently Asked Questions\n",
    "\n",
    "**Q1: [Common question]**\n",
    "\n",
    "A: [Clear answer with example if needed]\n",
    "\n",
    "**Q2: [Common question]**\n",
    "\n",
    "A: [Clear answer with example if needed]\n",
    "\n",
    "**Q3: [Common question]**\n",
    "\n",
    "A: [Clear answer with example if needed]"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create GitHub Classroom autograding config
cat > .github/classroom/autograding.json << 'EOF'
{
  "tests": [
    {
      "name": "Repository Structure",
      "setup": "",
      "run": "test -f src/implementation.py && test -f tests/test_implementation.py && test -f SUBMISSION.md && test -d teaching",
      "input": "",
      "output": "",
      "comparison": "included",
      "timeout": 10,
      "points": 5
    },
    {
      "name": "Dependencies Installation",
      "setup": "pip install -r requirements.txt",
      "run": "python -c 'import torch; import numpy; import pytest; print(\"Dependencies OK\")'",
      "input": "",
      "output": "Dependencies OK",
      "comparison": "included",
      "timeout": 60,
      "points": 5
    },
    {
      "name": "Code Imports",
      "setup": "",
      "run": "python -c 'import sys; sys.path.insert(0, \".\"); from src.implementation import *; print(\"Import successful\")'",
      "input": "",
      "output": "Import successful",
      "comparison": "included",
      "timeout": 30,
      "points": 10
    },
    {
      "name": "Submission Info",
      "setup": "",
      "run": "grep -E '(Topic Selected:|Time Breakdown:|Key Design Decisions:)' SUBMISSION.md",
      "input": "",
      "output": "",
      "comparison": "included",
      "timeout": 10,
      "points": 5
    }
  ]
}
EOF

# Create GitHub Actions workflow
cat > .github/workflows/classroom.yml << 'EOF'
name: GitHub Classroom Workflow

on: 
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

permissions:
  checks: write
  actions: read
  contents: read

jobs:
  build:
    name: Autograding
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python 3.9
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          
      - name: Run autograding
        uses: education/autograding@v1
        
      - name: Post comment on failure
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '❌ Autograding failed. Please check the Actions tab for details.'
            })
EOF

# Create a helpful utils file template
cat > src/utils.py << 'EOF'
"""
Utility functions for the TA interview implementation.

Add any helper functions, visualization utilities, or
common operations here.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def plot_loss_curve(losses: List[float], title: str = "Training Loss"):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.show()


# Add more utility functions as needed for your implementation
EOF

# Create an FAQ template for teaching
cat > teaching/student_faq.md << 'EOF'
# Frequently Asked Questions - [Your Topic]

## General Understanding

**Q: What is the intuition behind [topic]?**

A: [Provide clear, intuitive explanation]

**Q: When should I use [topic] instead of [alternative]?**

A: [Explain use cases and trade-offs]

**Q: What are the main hyperparameters and how do I tune them?**

A: [List key hyperparameters and tuning strategies]

## Implementation Details

**Q: Why do we need [specific component]?**

A: [Explain the necessity]

**Q: What happens if I forget to [common mistake]?**

A: [Explain consequences and how to debug]

**Q: How can I make this more efficient?**

A: [Provide optimization tips]

## Common Errors

**Q: I'm getting error X, what does it mean?**

A: [Explain error and solution]

**Q: My results don't match the expected output.**

A: Check these common issues:
1. [Issue 1]
2. [Issue 2]
3. [Issue 3]

## Advanced Topics

**Q: How does this relate to [related concept]?**

A: [Explain relationship]

**Q: Can this be extended to [use case]?**

A: [Discuss extensions and limitations]

## Resources

**Q: Where can I learn more?**

A: Here are some recommended resources:
- [Resource 1]
- [Resource 2]
- [Resource 3]
EOF

# Initialize git repository
git init
git add .
git commit -m "Initial template structure for GradML TA interview"

echo ""
echo "✅ Template repository created successfully!"
echo ""
echo "Next steps:"
echo "1. Push to GitHub:"
echo "   git remote add origin https://github.com/GradML-CS6923/ta-interview-template.git"
echo "   git push -u origin main"
echo ""
echo "2. On GitHub (github.com/GradML-CS6923/ta-interview-template):"
echo "   - Go to Settings"
echo "   - Scroll down to 'Template repository'"
echo "   - Check ✓ Template repository"
echo ""
echo "3. Create GitHub Classroom assignment:"
echo "   - Go to classroom.github.com"
echo "   - Select your GradML classroom"
echo "   - Create new assignment"
echo "   - Choose 'ta-interview-template' as template"
echo ""
echo "Repository URL: https://github.com/GradML-CS6923/ta-interview-template"
EOF