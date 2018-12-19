from copy import deepcopy
from abc import ABC, abstractmethod


class Storage(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_by_id(self, uuid):
        pass

    @abstractmethod
    def add_document(self, document):
        pass


class MemoryDocumentStorage(Storage):

    __slots__ = ('data')

    def __init__(self, uuid_field, data=None):
        self.uuid_field = uuid_field
        self.data = {}

    def get_by_id(self, uuid):
        return self.data.get(uuid)

    def add_document(self, document):
        uuid = document[self.uuid_field]

        if uuid not in self.data:
            self.data[uuid] = deepcopy(document)
            return True

        return False
