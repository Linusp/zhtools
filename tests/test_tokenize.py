import re

import pytest
import regex

from zhtools.tokenize import (
    Tokenizer,
    get_tokenizer,
    register_tokenizer,
    NgramTokenizer,
)
from zhtools.tokenize.token import Token


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
    tokens = [
        Token('你好啊', 0, 3),
        Token('吃过了', 4, 7),
        Token('过了没', 5, 8),
        Token('了没有', 6, 9)
    ]
    words = ['你好啊', '吃过了', '过了没', '了没有']
    assert list(tokenizer.tokenize(text)) == tokens
    assert list(tokenizer.cut(text)) == words
    assert tokenizer.lcut(text) == words
    assert list(tokenizer(text)) == tokens


def test_space_tokenization():
    tokenizer = get_tokenizer('space')
    text = 'hello world'
    tokens = [Token('hello', 0, 5), Token('world', 6, 11)]
    words = ['hello', 'world']
    assert list(tokenizer.tokenize(text)) == tokens
    assert list(tokenizer.cut(text)) == words
    assert tokenizer.lcut(text) == words
    assert list(tokenizer(text)) == tokens


def test_jieba_tokenizer():
    tokenizer = get_tokenizer('jieba', lazy_load=False)
    text = '今天的天气真好'
    tokens = [
        Token('今天', 0, 2),
        Token('的', 2, 3),
        Token('天气', 3, 5),
        Token('真', 5, 6),
        Token('好', 6, 7)
    ]
    words = ['今天', '的', '天气', '真', '好']
    assert list(tokenizer.tokenize(text)) == tokens
    assert list(tokenizer.cut(text)) == words
    assert tokenizer.lcut(text) == words
    assert list(tokenizer(text)) == tokens


def test_pku_tokenizer():
    tokenizer = get_tokenizer('pku')
    text = '今天的天气真好'
    tokens = [
        Token('今天', 0, 2),
        Token('的', 2, 3),
        Token('天气', 3, 5),
        Token('真', 5, 6),
        Token('好', 6, 7)
    ]
    words = ['今天', '的', '天气', '真', '好']
    assert list(tokenizer.tokenize(text)) == tokens
    assert list(tokenizer.cut(text)) == words
    assert tokenizer.lcut(text) == words
    assert list(tokenizer(text)) == tokens


@pytest.mark.usefixtures('mock_corenlp_post')
def test_corenlp_tokenizer():
    tokenizer = get_tokenizer('corenlp', url="http://127.0.0.1:8000")
    text = '今天的天气真好'
    tokens = [
        Token('今天', 0, 2),
        Token('的', 2, 3),
        Token('天气', 3, 5),
        Token('真', 5, 6),
        Token('好', 6, 7)
    ]
    words = ['今天', '的', '天气', '真', '好']
    assert list(tokenizer.tokenize(text)) == tokens
    assert list(tokenizer.cut(text)) == words
    assert tokenizer.lcut(text) == words
    assert list(tokenizer(text)) == tokens
