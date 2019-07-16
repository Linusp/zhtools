from operator import itemgetter
from collections import defaultdict, namedtuple
import logging
import pickle
from enum import IntEnum
from copy import deepcopy

from zhtools.preprocess import to_halfwidth
from zhtools.tokenize import get_tokenizer
from zhtools.similarity import compute_similarity


LOGGER = logging.getLogger(__name__)


class FieldNotExistsError(KeyError):

    def __init__(self, field):
        self.field = field
        self.message = "Field is missing"


FieldInfo = namedtuple('FieldInfo', 'type, index, uuid')


class FieldType(IntEnum):

    STRING = 0
    INT = 1
    FLOAT = 2

    @staticmethod
    def get_type(type_name):
        type_map = {
            'str': FieldType.STRING,
            'int': FieldType.INT,
            'float': FieldType.FLOAT,
        }
        if type_name in type_map:
            return type_map[type_name]

        raise ValueError(f'Invalid type name: {type_name}')


class IndexSchema():

    """
    Parameters
    ----------
    schema_data: dict
        InvertedIndex 的 Schema 定义，key 为 字段名，value 指定该字段的类型，如:
        {
            'id': {'type': 'str', 'index': False, 'uuid': True},
            'text': {'type': 'str', 'index': True},
            'text_list': {'type': 'str', 'index': True}
        }
        其中
        - type: 指定该字段的值类型，用于校验实际数据，目前仅支持: str/int/float
        - index: 指定该字段是否要用于索引，默认为 True，至少有一个字段为 True
        - uuid: 指定该字段为唯一标识符，默认为 False，有且仅有一个字段可为 True
    """

    FIELD_TYPE_MAPS = {
        FieldType.STRING: str,
        FieldType.INT: int,
        FieldType.FLOAT: float,
    }

    def __init__(self, schema_data):
        self._fields = {}
        self._index_fields = set()

        uuid_fields = set()
        for field, info in schema_data.items():
            self._fields[field] = FieldInfo(
                FieldType.get_type(info['type']),
                bool(info.get('index', True)),
                bool(info.get('uuid', False))
            )
            if self._fields[field].uuid:
                uuid_fields.add(field)

            if self._fields[field].index:
                self._index_fields.add(field)

        if len(uuid_fields) != 1:
            raise ValueError(
                f"Expect 1 uuid field but got {len(uuid_fields)}: {uuid_fields}"
            )

        if not self.index_fields:
            raise ValueError("No fields could be indexed")

        self._uuid_field = list(uuid_fields)[0]

    def validate(self, document):
        if self.uuid_field not in document:
            raise FieldNotExistsError(self.uuid_field)

        for field, field_info in self.fields.items():
            if field not in document:
                continue

            value = document[field]
            field_type = self.FIELD_TYPE_MAPS[field_info.type]
            if not isinstance(value, field_type):
                raise ValueError(
                    "Field `{field}` is defined as "
                    "`(type:{field_type})` but got "
                    "`{type(value)}`"
                )

    def get_field_type(self, field):
        if field not in self.fields:
            return None

        field_info = self.fields[field]
        return self.FIELD_TYPE_MAPS[field_info.type]

    @property
    def fields(self):
        return deepcopy(self._fields)

    @property
    def index_fields(self):
        return deepcopy(self._index_fields)

    @property
    def uuid_field(self):
        return self._uuid_field


class InvertedIndex():

    """
    Parameters
    ----------
    schema: dict
        文档的结构定义，指定文档每个字段的值类型、是否要索引、是否是唯一字段

    Instance Methods
    -------
    add_document(document)
        将一个文档添加到索引中

    retrieve(query, fields=None, limit=None, rank_metric='jaccard',
             metric_base='both', threshold=None)
        检索与 query 相关的文档

    retrieve_on_field(query, fields=None, limit=None, rank_metric='jaccard',
                      metric_base='both', threshold=None)
        根据指定 field 检索与 query 相关的文档

    match_on_field(field, value)
        获取指定字段值与 value 相等的文档

    dump(filename)
        将索引及 storage 保存到文件，storage 的保存行为由对应的类决定，如 MemoryDocumentStorage
        会将数据本身也一起保存到文件


    Class Methods
    -------------
    preprocess(text)
        在建立索引或检索文档时，对文本进行预处理以去除一些对检索可能无用的信息

    load(filename)
        从 dump 方法导出的索引文件中恢复 InvertedIndex 对象


    Examples
    --------
    In [1]: schema = {"id": str, "content": str}
    In [2]: inv_index = InvertedIndex(schema)
    In [3]: inv_index.add_document({"id": "1", "content": "hello world"})
    In [4]: inv_index.add_document({"id": "2", "content": "world wide web"})
    In [5]: inv_index.retrieve("world")
    Out[5]: [{"score": 0.5, "document": {"id": "1", "content": "hello world"}},
             {"score": 0.33, "document": {"id": "2", "content": "world wide web"}}]
    In [6]: inv_index.match("content", "hello world")
    Out[6]: [{"id": "2", "content": "world wide web"}]
    """

    FIELD_ID = 'id'
    METRICS = set(['lcs', 'jaccard', 'dice', 'cosine'])
    PREPROCESSORS = [to_halfwidth]

    __slots__ = ('schema', 'fields', 'term_dict', 'index', 'tokenizer')

    def __init__(self, schema):
        self.schema = IndexSchema(schema)
        self.fields = sorted(self.schema.index_fields)
        self.term_dict = dict()
        self.index = [defaultdict(set) for _ in self.fields]
        self.tokenizer = get_tokenizer("ngram", level=2)

    @classmethod
    def preprocess(cls, text):
        for func in cls.PREPROCESSORS:
            text = func(text)

        return text

    def add_document(self, document):
        """将一个文档添加到索引中"""
        self.schema.validate(document)

        uuid = document[self.schema.uuid_field]
        for field, value in document.items():
            if field not in self.schema.index_fields:
                continue

            terms = []
            field_info = self.schema.fields[field]
            # 仅对 str 类型的 value 进行 tokenization
            if field_info.type == FieldType.STRING:
                value = self.preprocess(value)
                terms = self.tokenizer.lcut(value)
            else:
                terms = [value]

            field_id = self.fields.index(field)
            for term in terms:
                if term in self.term_dict:
                    term_id = self.term_dict[term]
                else:
                    term_id = len(self.term_dict)
                    self.term_dict[term] = term_id
                self.index[field_id][term_id].add(uuid)

    def retrieve(self, storage, query, field, limit=None,
                 rank_metric='jaccard', metric_base='both', threshold=None):
        """检索与 query 相关的文档

        Parameters
        ----------
        storage: Storage
            存储后端，用于获取实际的文档内容
        query: any
            用于检索文档的值
        field: str
            要匹配的文档的字段，若不存在触发 FieldNotExistsError 异常
        limit: int(optional)
            返回结果的最大数量限制，若不设置则返回全部
        rank_metric: str(optional), default 'jaccard'
            计算检索结果与 query 相似度的方法，最终结果将按此进行排序
        metric_base: str(optional), default 'both'
            计算文档与 query 相似度时，以哪一方为准，有三个选项
            1. both: 计算对称的相似度，即 S(query, document)=S(document, query)
            2. query: 计算 document 与 query 的有偏的相似度
            3. document: 计算 query 与 document 的有偏的相似度
            例: query='abc', document='abdc', 使用 jaccard
            1. 选项为 both 时，得到的结果为 0.25
            2. 选项为 query 时，得到的结果为 0.5
            3. 选项为 document 时，得到的结果为 1/3
        threshold: float(optional)
            query 与文档的相似度阈值，若相似度低于阈值则不会被返回

        Return
        ------
        matches: list, 如: [{"document": <Document>, "score": 1.0}, ...]
        """
        assert rank_metric in set(['lcs', 'jaccard', 'dice', 'lcs', 'cosine'])
        assert metric_base in set(['query', 'document', 'both'])

        # field 不存在则抛异常
        if field not in self.schema.fields:
            raise FieldNotExistsError(field)

        # 1. 若 field 未被索引，则认为无匹配结果
        # 2. 若 value 与 schema 中 field value 的类型不一致，则认为无匹配结果
        field_info = self.schema.fields[field]
        if not field_info.index or \
           not isinstance(query, self.schema.get_field_type(field)):
            return []

        # 若指定 field 不是 str 类型，那么进行严格匹配
        if field_info.type != FieldType.STRING:
            documents = self.match_on_field(storage, field, query)
            results = [dict(document=doc, score=1.0) for doc in documents][:limit]
            return results if not limit else results[:limit]

        related, field_id = set(), self.fields.index(field)

        # 切分 terms 后寻找相关文档
        query = self.preprocess(query)
        terms = self.tokenizer.lcut(query)
        for term in terms:
            term_id = self.term_dict.get(term)
            if term_id is None:
                continue

            for docid in self.index[field_id][term_id]:
                related.add(docid)

        # 准备 compute_similarity 的参数
        parameters = {"method": rank_metric}
        if metric_base in ('query', 'document'):
            parameters["partial"] = True

        parameters["tokenizer"] = self.tokenizer

        # 对结果进行排序
        # TODO: 使用小顶堆优化内存占用和速度
        results = []
        for docid in related:
            document = storage.get_by_id(docid)
            text = self.preprocess(document[field])

            if metric_base == 'document':
                score = compute_similarity(text, query, **parameters)
            else:
                score = compute_similarity(query, text, **parameters)

            if threshold and score < threshold:
                continue

            results.append(dict(document=document, score=score))

        results.sort(key=itemgetter('score'), reverse=True)
        return results if not limit else results[:limit]

    def match_on_field(self, storage, field, value):
        """查找对应字段值与 value 完全相等的文档

        Parameters
        ----------
        storage: Storage
            存储后端，用于获取实际的文档内容
        field: str
            要匹配的文档的字段，若不存在触发 FieldNotExistsError 异常
        value: any
            需匹配的字段的值

        Return
        ------
        documents: list
            匹配到的文档列表
        """
        # field 不存在则抛异常
        if field not in self.schema.fields:
            raise FieldNotExistsError(field)

        # 1. 若 field 未被索引，则认为无匹配结果
        # 2. 若 value 与 schema 中 field value 的类型不一致，则认为无匹配结果
        field_info = self.schema.fields[field]
        if not field_info.index or \
           not isinstance(value, self.schema.get_field_type(field)):
            return []

        # 当 field 为 id 时，直接使用 storage 的方法来获取
        if field == self.schema.uuid_field:
            document = storage.get_by_id(value)
            return [document] if document else []

        documents = []
        field_id = self.fields.index(field)

        # 当 value 为非字符串内容时，先从 index 中获得文档的 id，再取得文档
        if not isinstance(value, str):
            term_id = self.term_dict.get(value)
            if term_id is None:
                return []
            for docid in self.index[field_id].get(term_id, set()):
                documents.append(storage.get_by_id(docid))

            return documents

        # 当 value 为字符串内容时，将字符串切分为 term，检索出相关 docid
        text = self.preprocess(value)
        terms = self.tokenizer.lcut(text)
        related = defaultdict(int)
        for term in terms:
            # 文本中存在未索引的 term，认为不会有匹配的结果
            term_id = self.term_dict.get(term)
            if term_id is None:
                return []

            for docid in self.index[field_id][term_id]:
                related[docid] += 1

        for docid, score in related.items():
            # 若 docid 被匹配到的次数比 terms 少，说明 terms 中某个 term 在
            # 对应的 field value 中不存在，可以认为两者无法匹配
            if score < len(terms):
                continue

            document = storage.get_by_id(docid)
            if document[field] == value:
                documents.append(document)

        return documents

    def dump(self, filename):
        with open(filename, 'wb') as fout:
            pickle.dump(self, fout)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as fin:
            return pickle.load(fin)
