# TA Interview Exercise - Deep Learning (CS-GY 6953/ECE-GY 7123)

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
   - Build an automatic differentiation engine
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
   - Integration into the training loop
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
- **Late Policy**: *No late submissions accepted*
- **Questions**: Use Slack

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

- Technical questions: Post in Slack 
- Urgent issues: Post in the course Slack

Good luck! We're excited to see your work and learn about your teaching approach.

---
*Grad Deep Learning - Teaching Assistant Selection*
