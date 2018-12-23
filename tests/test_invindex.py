from contextlib import contextmanager
from os.path import join
import shutil
import tempfile

import pytest

from zhtools.utils.storage import MemoryDocumentStorage
from zhtools.utils.inverted_index import IndexSchema, InvertedIndex, FieldNotExistsError
from zhtools.similarity import compute_similarity
from zhtools.tokenize import get_tokenizer


TOKENIZER = get_tokenizer('ngram', level=2)


@contextmanager
def tempdir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
        except IOError:
            pass


class TestIndexSchema():

    def setup(self):
        self.schema = IndexSchema({"id": {"type": "str", "uuid": True}, "text": {"type": "str"}})

    @pytest.mark.parametrize(
        "schema_data",
        [
            {"id": {"type": "str"}, "text": {"type": "str"}},
            {"id": {"type": "unknown_type"}},
            {
                "id": {"type": "str", "uuid": True, "index": False},
                "text": {"type": "str", "index": False},
            },
        ]
    )
    def test_init_error(self, schema_data):
        with pytest.raises(ValueError):
            IndexSchema(schema_data)

    def test_get_field_type(self):
        assert self.schema.get_field_type('id') is str
        assert self.schema.get_field_type('unknown') is None


class TestInvertedIndex():

    SCHEMA = {
        'id': {'type': 'str', 'index': True, 'uuid': True},
        'text': {'type': 'str'},
        'cnt': {'type': 'int'},
        'meta': {'type': 'str', 'index': False},
    }

    def setup(self):
        self.index = InvertedIndex(self.SCHEMA)
        self.storage = MemoryDocumentStorage('id')
        documents = [
            {'id': '1', 'text': 'first doc', 'desc': 'first desc', 'cnt': 4},
            {'id': '2', 'text': 'second doc', 'desc': 'second desc', 'cnt': 3},
            {'id': '3', 'text': 'third doc', 'desc': 'third desc', 'cnt': 2},
            {'id': '4', 'text': 'fourth doc', 'desc': 'fourth desc', 'cnt': 1},
            {'id': '5'},
        ]
        for doc in documents:
            self.index.add_document(doc)
            self.storage.add_document(doc)

    def test_add_duplicate_document(self):
        assert self.storage.add_document({"id": "5"}) is False

    def test_dump_load(self):
        with tempdir() as base_dir:
            self.index.dump(join(base_dir, 'test.index'))
            self.index = InvertedIndex.load(join(base_dir, 'test.index'))

    @pytest.mark.parametrize(
        ("bad_doc", "exception"),
        [
            (
                {'id': 5, 'text': 'some bad doc', 'desc': 'some desc', 'cnt': 1},
                ValueError
            ),
            (
                {'id': '6', 'text': 666666, 'cnt': 1},
                ValueError
            ),
            (
                {},
                FieldNotExistsError
            )
        ]
    )
    def test_add_doc_error(self, bad_doc, exception):
        with pytest.raises(exception):
            self.index.add_document(bad_doc)

    @pytest.mark.parametrize(
        ("query", "field", "limit", "threshold", "results"),
        [
            (
                "first", "text", 1, 0,
                [{
                    'document': {'id': '1', 'text': 'first doc', 'desc': 'first desc', 'cnt': 4},
                    'score': compute_similarity('first', 'first_doc', tokenizer=TOKENIZER),
                }]
            ),
            ("first", "text", 1, 0.9, []),
            (
                "first book", "text", 1, 0,
                [{
                    'document': {'id': '1', 'text': 'first doc', 'desc': 'first desc', 'cnt': 4},
                    'score': compute_similarity('first book', 'first_doc', tokenizer=TOKENIZER),
                }]
            ),
            ("metameta", "meta", 1, 0, []),
            (
                4, 'cnt', 1, 0,
                [{
                    'document': {'id': '1', 'text': 'first doc', 'desc': 'first desc', 'cnt': 4},
                    'score': 1.0
                }]
            )
        ]
    )
    def test_retrieve(self, query, field, limit, threshold, results):
        ret = self.index.retrieve(self.storage, query, field, limit=limit, threshold=threshold)
        assert ret == results

    @pytest.mark.parametrize(
        ("query", "field", "limit", "metric_base", "results"),
        [
            (
                "first", "text", 1, "query",
                [{
                    'document': {'id': '1', 'text': 'first doc', 'desc': 'first desc', 'cnt': 4},
                    'score': compute_similarity(
                        'first', 'first doc', tokenizer=TOKENIZER, partial=True
                    ),
                }]
            ),
            (
                "first", "text", 1, "document",
                [{
                    'document': {'id': '1', 'text': 'first doc', 'desc': 'first desc', 'cnt': 4},
                    'score': compute_similarity(
                        'first doc', 'first', tokenizer=TOKENIZER, partial=True
                    ),
                }]
            ),
        ]
    )
    def test_retrieve_partial(self, query, field, limit, metric_base, results):
        ret = self.index.retrieve(self.storage, query, field, limit=limit, metric_base=metric_base)
        assert ret == results

    def test_retrieve_error(self):
        with pytest.raises(FieldNotExistsError):
            self.index.retrieve(self.storage, 'first', 'some field', limit=1)

    def test_match_on_field(self):
        results = self.index.match_on_field(self.storage, 'id', '1')
        assert len(results) == 1 and results[0]['id'] == '1'

        results = self.index.match_on_field(self.storage, 'text', 'first doc')
        assert len(results) == 1 and results[0]['id'] == '1'

        results = self.index.match_on_field(self.storage, 'cnt', 4)
        assert len(results) == 1 and results[0]['id'] == '1'

        results = self.index.match_on_field(self.storage, 'id', 1)
        assert not results

        results = self.index.match_on_field(self.storage, 'text', 'first game')
        assert not results

        results = self.index.match_on_field(self.storage, 'cnt', 5)
        assert not results

    def test_match_on_field_error(self):
        with pytest.raises(FieldNotExistsError):
            self.index.match_on_field(self.storage, 'some field', 'some value')
