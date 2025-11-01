# Contributing Guide

Thank you for contributing to the ML Inference API project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Add upstream: `git remote add upstream <original-repo-url>`
4. Create a branch: `git switch -c feature/your-feature`

## Making Changes

### Branch Naming
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `refactor/` - Code refactoring
- `test/` - Test additions
- `chore/` - Maintenance tasks

### Commit Messages
Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Examples:
- `feat(api): add batch prediction endpoint`
- `fix(preprocessing): handle grayscale images`
- `docs(readme): update installation instructions`

### Code Style
- Follow PEP 8 for Python code
- Use type hints
- Add docstrings to all public functions
- Keep functions focused and single-purpose

### Testing
- Write tests for all new functionality
- Ensure all existing tests pass
- Aim for >80% code coverage
- Include edge cases in tests

## Pull Request Process

1. **Update your branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests locally**
   ```bash
   pytest tests/ -v
   ```

3. **Push to your fork**
   ```bash
   git push origin feature/your-feature
   ```

4. **Create Pull Request**
   - Use the PR template
   - Link related issues
   - Provide clear description
   - Add screenshots if applicable

5. **Address Review Feedback**
   - Respond to all comments
   - Make requested changes
   - Push updates
   - Request re-review

## Code Review

### For Authors
- Be open to feedback
- Ask questions if unclear
- Don't take criticism personally
- Thank reviewers for their time

### For Reviewers
- Be constructive and respectful
- Explain the "why" not just "what"
- Distinguish between blocking and non-blocking comments
- Praise good code

## Community Guidelines

- Be respectful and inclusive
- Assume good intentions
- Help newcomers
- Share knowledge generously

## Questions?

Feel free to open an issue or reach out to maintainers!
