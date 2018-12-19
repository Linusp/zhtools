from collections import defaultdict
from difflib import SequenceMatcher
from math import sqrt


def cosine(first, second, partial=False):
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0

    first_term_freq = defaultdict(int)
    for term in first:
        first_term_freq[term] += 1

    second_term_freq = defaultdict(int)
    for term in second:
        second_term_freq[term] += 1

    first_norm, second_norm, inner_product = 0, 0, 0
    for term, freq in first_term_freq.items():
        first_norm += freq ** 2
        if partial:
            inner_product += freq * min(freq, second_term_freq[term])
        else:
            inner_product += freq * second_term_freq[term]

    for term, freq in second_term_freq.items():
        second_norm += freq ** 2

    if partial:
        return inner_product / first_norm

    return inner_product / sqrt(first_norm * second_norm)


def dice(first, second, partial=False):
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0

    first_set = set(first)
    second_set = set(second)

    common_set = first_set & second_set
    if partial:
        return len(common_set) / len(first_set)

    return 2 * len(common_set) / (len(first_set) + len(second_set))


def jaccard(first, second, partial=False):
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0

    first_set = set(first)
    second_set = set(second)

    common_set = first_set & second_set
    if partial:
        return len(common_set) / len(first_set)

    return len(common_set) / len(first_set | second_set)


def lcs(first, second, partial=False):
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0

    alignments = SequenceMatcher(a=first, b=second, autojunk=False)
    lcs_length = sum([size for _, _, size in alignments.get_matching_blocks()])
    if partial:
        return lcs_length / len(first)

    return 2 * lcs_length / (len(first) + len(second))
