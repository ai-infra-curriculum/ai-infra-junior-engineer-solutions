# Exercise 07: Testing Python ML Code — Solution

## What the exercise asked for

Write unit, integration, and property-based tests for an ML
inference pipeline. Use pytest, fixtures, mocking, and CI
integration.

## Test pyramid for ML code

```
        /\
       /  \    E2E (slow, expensive, few)
      /----\
     /      \    Integration (medium count)
    /--------\
   /          \    Unit (fast, many)
  /------------\
```

For ML code specifically:

- **Unit tests**: pure functions (preprocessing, feature
  engineering, parsing).
- **Integration tests**: pipeline stages talking to each
  other (preprocessor → model → postprocessor).
- **E2E tests**: full prediction path including model load.
- **Smoke tests**: "the model loads without error" — useful
  as a CI gate.

## Sample structure

See [`test_examples.py`](./test_examples.py) for working
examples of each pattern.

## Mocking the model

Loading a real model in every test is slow. Use a mock:

```python
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def fake_model():
    model = MagicMock()
    model.predict.return_value = [0.7, 0.2, 0.1]
    return model

def test_classify_returns_top_class(fake_model):
    from src.api import classify
    result = classify(fake_model, "some image")
    assert result == 0
    fake_model.predict.assert_called_once()
```

## Property-based tests

For functions whose output should hold an invariant
regardless of input shape, use `hypothesis`:

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=0, max_value=1), min_size=2))
def test_softmax_sums_to_one(logits):
    from src.utils import softmax
    result = softmax(logits)
    assert abs(sum(result) - 1.0) < 1e-6
    assert all(0 <= p <= 1 for p in result)
```

`hypothesis` generates many input shapes and finds the edge
case that breaks your function.

## Fixtures + parametrization

```python
@pytest.fixture(scope="module")
def loaded_model():
    """Loaded once per test module — expensive operations
    don't repeat per test."""
    from src.model import load_model
    return load_model("models/test_model.bin")

@pytest.mark.parametrize("input_val,expected", [
    ("cat", 0),
    ("dog", 1),
    ("fish", 2),
])
def test_classify_known_inputs(loaded_model, input_val, expected):
    assert classify(loaded_model, input_val) == expected
```

## CI integration

```yaml
# .github/workflows/test.yml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov hypothesis
      - run: pytest --cov=src --cov-report=term-missing
```

A passing CI gate is the difference between "we tested" and
"we keep testing as the code evolves."

## Common mistakes

- Tests that exercise the framework, not your code (e.g.,
  testing pytest itself or testing that `numpy.array([1])`
  has length 1).
- Tests that depend on external services without mocking
  (flaky).
- One huge test that asserts 12 things — when it fails, you
  don't know which.
- Tests that mutate shared state (other tests fail
  unpredictably).

## Coverage targets for ML code

- ≥80% line coverage on `src/`.
- 100% on critical paths (preprocessing, model loading, the
  prediction endpoint).
- Property tests for any pure transformation.

## Cross-references

- Exercise prompt:
  `ai-infra-junior-engineer-learning/lessons/mod-001-python-fundamentals/exercises/exercise-07-testing.md`
- Engineer-track has tests at depth:
  `ai-infra-engineer-solutions/modules/mod-101-foundations/exercise-08-production-model-serving/tests/`.
