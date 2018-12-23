import pytest

from zhtools.tokenize import get_tokenizer
from zhtools.similarity import compute_similarity


@pytest.mark.parametrize(
    'first, second, tokenizer, ngram_range, ngram_weights, sim',
    [
        ('abcde', 'bcd', None, None, None, 0.6),
        ('abcde', 'bcd', None, [1, 2], None, 0.55),
        ('abcde', 'bcd', None, [1, 2], [0.1, 0.4], 0.52),
        ('abcde', 'bcd', get_tokenizer('ngram', level=2), None, None, 0.5),
    ]
)
def test_similarity(first, second, tokenizer, ngram_range, ngram_weights, sim):
    # use jaccard default
    similarity = compute_similarity(first, second,
                                    tokenizer=tokenizer,
                                    ngram_range=ngram_range,
                                    ngram_weights=ngram_weights)
    assert similarity == sim


@pytest.mark.parametrize(
    'first, second, method',
    [('abcde', 'bcd', 'some_unknown_method')]
)
def test_similarity_error(first, second, method):
    with pytest.raises(ValueError):
        compute_similarity(first, second, method=method)
