# Step-by-Step Implementation Guide: [Exercise Name]

**Module**: [Module Number and Name]
**Exercise**: [Exercise Number]
**Estimated Time**: [X] hours
**Difficulty**: [Beginner/Intermediate/Advanced]

---

## 📋 Overview

This guide walks you through implementing [brief description of what you'll build].

**What You'll Learn**:
- [Key learning objective 1]
- [Key learning objective 2]
- [Key learning objective 3]

**Prerequisites**:
- Completed: [List prerequisite modules/exercises]
- Installed: [Required software/tools]
- Knowledge of: [Required concepts]

---

## 🎯 Learning Objectives

By completing this exercise, you will be able to:
1. [Specific, measurable objective]
2. [Specific, measurable objective]
3. [Specific, measurable objective]

---

## 📁 Project Structure

```
exercise-name/
├── README.md
├── STEP_BY_STEP.md (this file)
├── src/
│   ├── main.py
│   ├── [other files]
├── tests/
│   └── test_main.py
├── docs/
│   ├── API.md
│   └── ARCHITECTURE.md
├── [config files]
└── requirements.txt
```

---

## 🚀 Step-by-Step Instructions

### Step 1: Project Setup

**Goal**: Initialize the project structure and dependencies

**Tasks**:
1. Create project directory
   ```bash
   mkdir [project-name]
   cd [project-name]
   ```

2. Set up virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Create directory structure
   ```bash
   mkdir -p src tests docs
   touch src/__init__.py tests/__init__.py
   ```

**✅ Checkpoint**: You should have a clean project structure with virtual environment active

---

### Step 2: Install Dependencies

**Goal**: Set up all required libraries

**Tasks**:
1. Create `requirements.txt`
   ```txt
   [List main dependencies with versions]
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation
   ```bash
   pip list | grep [key-package]
   ```

**✅ Checkpoint**: All dependencies installed without errors

---

### Step 3: [Descriptive Step Name]

**Goal**: [What this step accomplishes]

**Explanation**:
[Explain the concept or approach for this step]

**Implementation**:

Create `src/[filename].py`:
```python
# [Comments explaining the code]
[Code implementation]
```

**Why this approach?**
[Explain design decisions, trade-offs, best practices]

**Common Pitfalls**:
- ⚠️ [Common mistake]: [How to avoid it]
- ⚠️ [Another mistake]: [Solution]

**✅ Checkpoint**: [How to verify this step works]

---

### Step 4: [Next Step]

[Continue pattern for each major implementation step]

---

### Step N: Testing

**Goal**: Verify your implementation works correctly

**Tasks**:
1. Create test file `tests/test_[module].py`
   ```python
   import pytest
   from src.[module] import [function]

   def test_[functionality]():
       # Arrange
       [setup code]

       # Act
       result = [function call]

       # Assert
       assert result == [expected]
   ```

2. Run tests
   ```bash
   pytest tests/ -v
   ```

3. Check coverage
   ```bash
   pytest --cov=src tests/
   ```

**✅ Checkpoint**: All tests passing, >80% coverage

---

### Step N+1: Documentation

**Goal**: Create clear documentation for your code

**Tasks**:
1. Add docstrings to all functions
2. Create API documentation (if applicable)
3. Update README with usage examples

**✅ Checkpoint**: All code documented

---

### Step N+2: [Deployment/Integration]

**Goal**: [Deploy or integrate the solution]

[Deployment-specific steps]

**✅ Checkpoint**: [Verify deployment]

---

## 🧪 Validation & Testing

### Manual Testing
1. [Step to manually test functionality]
2. [Another manual test]

### Automated Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test
pytest tests/test_[module].py::test_[function]
```

### Expected Output
```
[Show what successful output looks like]
```

---

## 🐛 Troubleshooting

### Issue: [Common Problem 1]
**Symptoms**: [What you see]
**Cause**: [Why it happens]
**Solution**:
```bash
[Commands or fixes]
```

### Issue: [Common Problem 2]
[Same format]

### Issue: [Common Problem 3]
[Same format]

---

## 📊 Code Quality Checklist

Before considering this exercise complete:

- [ ] All code follows PEP 8 style guide
- [ ] All functions have docstrings
- [ ] All tests pass
- [ ] Code coverage > 80%
- [ ] No security vulnerabilities
- [ ] Error handling implemented
- [ ] Logging added where appropriate
- [ ] Configuration externalized
- [ ] Documentation complete
- [ ] Code reviewed (self or peer)

---

## 🎓 Key Takeaways

**What You Learned**:
1. [Key concept or skill acquired]
2. [Another important learning]
3. [Technical skill developed]

**Best Practices Applied**:
- [Best practice 1]
- [Best practice 2]
- [Best practice 3]

**Common Patterns**:
- [Reusable pattern 1]
- [Reusable pattern 2]

---

## 🔄 Variations & Extensions

### Beginner Extension
[Simpler variation or practice exercise]

### Intermediate Extension
[More challenging variation]

### Advanced Extension
[Production-grade or complex variation]

---

## 📚 Additional Resources

### Official Documentation
- [Link to relevant docs]
- [Another resource]

### Tutorials
- [Recommended tutorial 1]
- [Recommended tutorial 2]

### Related Exercises
- [Previous exercise that prepares for this]
- [Next exercise that builds on this]

---

## 💡 Interview Prep

Questions you should be able to answer after this exercise:

1. **Explain [key concept]**
   - [Expected answer points]

2. **What are the trade-offs of [approach used]?**
   - [Expected answer points]

3. **How would you scale this solution?**
   - [Expected answer points]

---

## ✅ Exercise Complete!

Congratulations! You've successfully completed this exercise.

**Next Steps**:
1. Review your code and refactor if needed
2. Add this project to your portfolio
3. Write a blog post explaining your implementation
4. Proceed to [next exercise]

---

**Estimated Completion Time**: [X] hours
**Actual Time Taken**: _____ hours
**Difficulty Rating** (1-5): _____
**Notes for Improvement**:
_______________________________________________________________

---

*This template is part of the AI Infrastructure Junior Engineer Solutions Repository*
