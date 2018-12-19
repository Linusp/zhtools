from more_itertools import windowed

from zhtools.metric.similarity import cosine, dice, jaccard, lcs
from zhtools.tokenize import get_tokenizer


__all__ = ["compute_similarity"]


def compute_similarity(first, second, method='jaccard', tokenizer=None,
                       partial=False, ngram_range=None, ngram_weights=None):
    assert isinstance(first, str) and isinstance(second, str)
    if not ngram_range:
        ngram_range = [1]
        ngram_weights = [1.0]

    metric_func = {
        'lcs': lcs,
        'jaccard': jaccard,
        'cosine': cosine,
        'dice': dice,
    }.get(method)
    if not metric_func:
        raise ValueError("unsupported method `{}`".format(method))

    tokenizer = tokenizer or get_tokenizer("ngram", level=1)

    first_terms = tokenizer.lcut(first)
    second_terms = tokenizer.lcut(second)

    similarity = 0.0
    ngram_levels = list(range(ngram_range[0], ngram_range[-1] + 1))
    if not ngram_weights:
        ngram_weights = [1 for _ in range(len(ngram_levels))]
    for ngram_level, weight in zip(ngram_levels, ngram_weights):
        first_ngrams = list(windowed(first_terms, ngram_level))
        second_ngrams = list(windowed(second_terms, ngram_level))
        similarity += weight * metric_func(first_ngrams, second_ngrams, partial=partial)

    return similarity / sum(ngram_weights)
