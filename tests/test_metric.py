import pytest

from zhtools.metric.similarity import (
    cosine,
    dice,
    jaccard,
    lcs,
)


@pytest.mark.parametrize(
    'first, second, partial, sim',
    [
        ([], [], False, 1.0),
        ([], ['a', 'b', 'c', 'd'], False, 0.0),
        (['a', 'b', 'c', 'd'], ['b', 'c', 'd', 'e'], False, 0.75),
        (['a', 'b', 'c', 'd'], ['c', 'd', 'e'], True, 0.5),
    ]
)
def test_cosine(first, second, partial, sim):
    assert cosine(first, second, partial) == sim


@pytest.mark.parametrize(
    'first, second, partial, sim',
    [
        ('', '', False, 1.0),
        ('', 'abc', False, 0.0),
        ('abcd', 'bcde', False, 0.75),
        ('abcde', 'bcd', False, 0.75),
        ('abcde', 'bcd', True, 0.6),
    ]
)
def test_dice(first, second, partial, sim):
    assert dice(first, second, partial) == sim


@pytest.mark.parametrize(
    'first, second, partial, sim',
    [
        ('', '', False, 1.0),
        ('', 'abc', False, 0.0),
        ('abcd', 'bcde', False, 0.6),
        ('abcde', 'bcd', False, 0.6),
        ('abcde', 'bcd', False, 0.6),
        ('abcd', 'bcd', True, 0.75),
    ]
)
def test_jaccard(first, second, partial, sim):
    assert jaccard(first, second, partial) == sim


@pytest.mark.parametrize(
    'first, second, partial, sim',
    [
        ('', '', False, 1.0),
        ('', 'abc', False, 0.0),
        ('abcd', 'bcde', False, 0.75),
        ('abcd', 'becd', False, 0.75),
        ('abcd', 'bcd', False, 6.0 / 7),
        ('abcd', 'bcd', True, 0.75),
    ]
)
def test_lcs(first, second, partial, sim):
    assert lcs(first, second, partial) == sim
