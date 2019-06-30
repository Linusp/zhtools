from os import path

import requests

from .token import Token
from .utils import align_tokens
from .base import register_tokenizer, Tokenizer


@register_tokenizer('jieba')
class JiebaTokenizer(Tokenizer):

    def __init__(self, dict_file=None, tmp_dir=None, lazy_load=True):
        from jieba import Tokenizer as T
        self._tokenizer = T(dictionary=dict_file)
        self._tokenizer.tmp_dir = tmp_dir or path.abspath('./')

        if not lazy_load:
            self._tokenizer.initialize()

    def tokenize(self, text):
        for token in self._tokenizer.tokenize(text):
            word, start, end = token
            if word.strip():
                yield Token(word, start, end)


@register_tokenizer('pku')
class PKUTokenizer(Tokenizer):

    def __init__(self, model_name='default', user_dict='default'):
        from pkuseg import pkuseg as T
        self._tokenizer = T(model_name=model_name, user_dict=user_dict)

    def tokenize(self, text):
        words = self._tokenizer.cut(text)
        for word, span in zip(words, align_tokens(words, text)):
            yield Token(word, span[0], span[1])


@register_tokenizer('corenlp')
class CoreNLPTokenizer(Tokenizer):

    def __init__(self, url, annotators='ssplit,tokenize', lang='zh'):
        self.url = url
        self.annotators = annotators
        self.lang = lang

    def tokenize(self, text):
        properties = {
            'annotators': self.annotators,
            'pipelineLanguage': self.lang,
            'outputFormat': 'json'
        }
        params = {'properties': str(properties)}
        data = text.encode('utf-8')
        headers = {'Connection': 'close'}
        response = requests.post(self.url, params=params, data=data, headers=headers)
        for sentence in response.json().get('sentences', []):
            for token in sentence['tokens']:
                yield Token(
                    token['originalText'],
                    token['characterOffsetBegin'],
                    token['characterOffsetEnd'],
                )
