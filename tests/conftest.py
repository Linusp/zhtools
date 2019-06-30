import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa

from zhtools.tokenize.utils import align_tokens


class MockResponse():

    def __init__(self, data):
        self._json = data

    def json(self):
        return self._json


@pytest.fixture
def mock_corenlp_post(monkeypatch):

    text_to_words = {
        '今天的天气真好': ['今天', '的', '天气', '真', '好']
    }

    def fake_post(self, *args, data=None, **kwargs):
        text = data.decode('utf-8')
        words = text_to_words.get(text, [])

        tokens = []
        for word, span in zip(words, align_tokens(words, text)):
            tokens.append({
                'originalText': word,
                'characterOffsetBegin': span[0],
                'characterOffsetEnd': span[1]
            })

        data = {}
        if tokens:
            data = {'sentences': [{'tokens': tokens}]}

        return MockResponse(data)

    monkeypatch.setattr(requests, 'post', fake_post)
