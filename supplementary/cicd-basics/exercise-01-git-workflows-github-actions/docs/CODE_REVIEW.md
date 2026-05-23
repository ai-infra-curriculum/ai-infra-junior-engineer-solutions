# Code Review Guidelines

## For Pull Request Authors

### Before Creating PR

- [ ] Code compiles and runs
- [ ] All tests pass
- [ ] Added tests for new features
- [ ] Updated documentation
- [ ] Ran local quality checks
- [ ] No debug code or print statements
- [ ] No secrets or sensitive data
- [ ] Followed coding standards

### PR Description

Include:
1. **Summary** - What and why
2. **Changes** - Bulleted list of changes
3. **Testing** - How to test
4. **Screenshots** - For UI changes
5. **Related Issues** - Link issues (Fixes #123)
6. **Breaking Changes** - If any

### Responding to Reviews

- Address all comments
- Mark resolved when fixed
- Ask questions if unclear
- Don't take feedback personally
- Push fixes as new commits (for review tracking)

## For Reviewers

### What to Check

1. **Functionality**
   - Does it solve the problem?
   - Are edge cases handled?
   - Is error handling appropriate?

2. **Code Quality**
   - Is it readable and maintainable?
   - Are variables/functions well-named?
   - Is it DRY (Don't Repeat Yourself)?
   - Appropriate comments?

3. **Tests**
   - Are tests comprehensive?
   - Do they test the right things?
   - Are edge cases covered?

4. **Performance**
   - Any obvious performance issues?
   - Appropriate data structures?
   - Unnecessary computations?

5. **Security**
   - No hardcoded secrets?
   - Input validation?
   - SQL injection risks?
   - XSS vulnerabilities?

6. **Documentation**
   - README updated?
   - API docs current?
   - Comments where needed?

### Review Etiquette

**Do:**
- Be respectful and constructive
- Explain the "why" behind suggestions
- Provide examples or resources
- Praise good code
- Review promptly (within 24 hours)

**Don't:**
- Be dismissive or rude
- Nitpick minor style issues (use automated tools)
- Block on personal preferences
- Make demands without explanation

### Review Comments Examples

**Good:**
```
This function could be clearer with early returns to reduce nesting:

```python
if not user:
    return None
if not user.is_active:
    return None
return user
```

This makes the happy path more obvious.
```

**Bad:**
```
This is wrong. Fix it.
```

## Review Checklist

### Security
- [ ] No hardcoded credentials
- [ ] Input validation
- [ ] Error messages don't leak info
- [ ] Dependencies scanned for vulnerabilities

### Performance
- [ ] No N+1 queries
- [ ] Appropriate caching
- [ ] No unnecessary loops
- [ ] Efficient algorithms

### Maintainability
- [ ] Code is self-documenting
- [ ] Functions are single-purpose
- [ ] Appropriate abstractions
- [ ] Error handling is clear

### Testing
- [ ] Unit tests for new code
- [ ] Integration tests where needed
- [ ] Tests are independent
- [ ] Tests are deterministic

## Resources

- [Google's Code Review Guide](https://google.github.io/eng-practices/review/)
- [Pull Request Best Practices](https://github.com/blog/1943-how-to-write-the-perfect-pull-request)
