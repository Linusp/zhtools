from abc import ABC, abstractmethod
import json
import logging
from typing.re import Pattern

import regex
from more_itertools import windowed


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
_TOKENIZER_CLS_MAP = {}


def register_tokenizer(name):
    def wrap(cls):
        global _TOKENIZER_CLS_MAP
        if name not in _TOKENIZER_CLS_MAP:
            _TOKENIZER_CLS_MAP[name] = cls
        else:
            raise ValueError("name `{}` is already registered!".format(name))

        return cls

    return wrap


def get_tokenizer(tokenizer_type, **kwargs):
    return Tokenizer.from_config({"type": tokenizer_type, "parameters": kwargs})


class Tokenizer(ABC):

    _tokenizers = {}

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        signature = json.dumps(config, sort_keys=True)
        if signature in cls._tokenizers:
            return cls._tokenizers[signature]

        name = config['type']
        sub_cls = _TOKENIZER_CLS_MAP.get(name)
        if not sub_cls:
            raise ValueError("no tokenizer was registered with name `{}`".format(name))

        parameters = config['parameters']
        tokenizer = sub_cls(**parameters)
        cls._tokenizers[signature] = tokenizer
        return tokenizer

    @abstractmethod
    def tokenize(self, text):
        raise NotImplementedError

    def cut(self, text):
        return self.tokenize(text)

    def lcut(self, text):
        return list(self.tokenize(text))

    def __call__(self, text):
        return self.tokenize(text)


@register_tokenizer('ngram')
class NgramTokenizer(Tokenizer):

    def __init__(self, level=3, step=1, lowercase=True, filter_pattern=r'^.*?[\p{P}\s].*?$'):
        self.level = level
        self.step = step
        self.lowercase = lowercase
        self.filter_pattern = None
        if filter_pattern:
            if isinstance(filter_pattern, str):
                self.filter_pattern = regex.compile(filter_pattern)
            elif isinstance(filter_pattern, (Pattern, regex.regex.Pattern)):
                self.filter_pattern = filter_pattern
            else:
                raise ValueError("invalid filter pattern: {}".format(filter_pattern))

    def tokenize(self, text):
        tokens = []
        text = text if not self.lowercase else text.lower()
        for chars in windowed(text, self.level, step=self.step, fillvalue=''):
            term = ''.join(chars)
            if self.filter_pattern and self.filter_pattern.match(term):
                LOGGER.debug("term is dropped by filter pattern: %s", term)
                continue

            tokens.append(term)

        return tokens
