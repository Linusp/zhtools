import re

import pytest
import regex

from zhtools.tokenize import (
    Tokenizer,
    get_tokenizer,
    register_tokenizer,
)
from zhtools.tokenize.base import NgramTokenizer


def test_get_tokenizer():
    first = get_tokenizer('ngram', level=3)
    second = get_tokenizer('ngram', level=3)

    assert first is second


def test_get_tokenizer_error():
    with pytest.raises(ValueError):
        get_tokenizer('some_unknown_tokenizer')


def test_register_tokenizer_error():
    with pytest.raises(ValueError):
        @register_tokenizer('ngram')
        class MyNgramTokenizer(Tokenizer):
            def tokenize(self, text):
                pass


def test_ngram_tokenizer_init():
    assert NgramTokenizer(level=3,
                          step=1,
                          lowercase=True,
                          filter_pattern=r'.*\p{P}')
    assert NgramTokenizer(level=3,
                          step=1,
                          lowercase=True,
                          filter_pattern=re.compile(r'.*\s'))
    assert NgramTokenizer(level=3,
                          step=1,
                          lowercase=True,
                          filter_pattern=regex.compile(r'.*\p{P}'))


def test_ngram_tokenizer_init_erro():
    with pytest.raises(ValueError):
        assert NgramTokenizer(level=3,
                              step=1,
                              lowercase=True,
                              filter_pattern=('not', 'a', 'pattern'))


def test_ngram_tokenization():
    tokenizer = get_tokenizer('ngram', level=3)
    text = '你好啊！吃过了没有？'
    tokens = ['你好啊', '吃过了', '过了没', '了没有']
    assert tokenizer.tokenize(text) == tokens
    assert tokenizer.cut(text) == tokens
    assert tokenizer.lcut(text) == tokens
    assert tokenizer(text) == tokens
