"""Reference test patterns for ML code.

Demonstrates:
  - unit tests (pure functions)
  - fixtures for shared setup
  - mocking expensive resources
  - parameterized tests
  - property-based tests with hypothesis

Run:
    pip install pytest hypothesis
    pytest test_examples.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from hypothesis import given, strategies as st


# ----- The code under test (a tiny example) -----


def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax."""
    if not logits:
        raise ValueError("logits must not be empty")
    m = max(logits)
    exps = [pow(2.71828182845905, x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def top_class(probs: list[float]) -> int:
    """Index of the highest-probability class."""
    if not probs:
        raise ValueError("probs must not be empty")
    return max(range(len(probs)), key=probs.__getitem__)


def classify(model, item) -> int:
    """The classification path: model.predict returns probs;
    we return the argmax class."""
    probs = model.predict(item)
    return top_class(probs)


# ----- Unit tests on pure functions -----


def test_softmax_sums_to_one():
    result = softmax([1.0, 2.0, 3.0])
    assert abs(sum(result) - 1.0) < 1e-6


def test_softmax_empty_raises():
    with pytest.raises(ValueError):
        softmax([])


def test_top_class_picks_argmax():
    assert top_class([0.1, 0.9, 0.0]) == 1


# ----- Parameterized tests -----


@pytest.mark.parametrize(
    "probs,expected",
    [
        ([1.0, 0.0, 0.0], 0),
        ([0.0, 1.0, 0.0], 1),
        ([0.0, 0.0, 1.0], 2),
        ([0.3, 0.3, 0.4], 2),
    ],
)
def test_top_class_param(probs, expected):
    assert top_class(probs) == expected


# ----- Property-based test -----


@given(st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=2, max_size=20))
def test_softmax_properties(logits):
    """For any non-empty list of finite floats, softmax must:
    - sum to 1 (within tolerance)
    - have all values in [0, 1]
    """
    result = softmax(logits)
    assert abs(sum(result) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in result)


# ----- Mocking expensive dependencies -----


@pytest.fixture
def fake_model():
    """A fake model that doesn't require loading real weights."""
    model = MagicMock()
    model.predict.return_value = [0.1, 0.7, 0.2]
    return model


def test_classify_uses_model_predict(fake_model):
    result = classify(fake_model, "some image")
    assert result == 1
    fake_model.predict.assert_called_once_with("some image")


def test_classify_returns_top_class(fake_model):
    fake_model.predict.return_value = [0.05, 0.05, 0.9]
    assert classify(fake_model, "any") == 2


# ----- A fixture that's module-scoped (load once, reuse) -----


@pytest.fixture(scope="module")
def example_corpus():
    """In real code this would load a dataset; here it's a tiny
    fixture to demonstrate the pattern."""
    return [
        ("cat-image", 0),
        ("dog-image", 1),
        ("fish-image", 2),
    ]


def test_corpus_classifies_to_known_labels(fake_model, example_corpus):
    expected_argmaxes = [0, 1, 2]
    for (item, label), expected in zip(example_corpus, expected_argmaxes):
        fake_model.predict.return_value = [
            1.0 if i == expected else 0.0 for i in range(3)
        ]
        assert classify(fake_model, item) == expected
